'''
Author: xyx && yx282947664@163.com
Date: 2023-07-04 13:29:59
LastEditors: xyx && yx282947664@163.com
LastEditTime: 2023-07-04 13:47:14
Copyright (c) 2023 by xyx && yx282947664@163.com, All Rights Reserved. 
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from swin_transformer import swin_t
from getCWTData import CWT
import time
import os
from utils import pad4

import warnings
warnings.filterwarnings("ignore")

myseed=6
np.random.seed(myseed)
# random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#得到数据
path=r"data"
train=CWT(path,224,"train")
val=CWT(path,224,"val")
test=CWT(path,224,"test")

print(len(train))
print(len(val))
print(len(test))

from torch import optim
from torch.optim import lr_scheduler

# 定义超参数
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=swin_t().to(device)
batch_size = 16
lr = 1e-3
EPOCHS = 100
loss_F = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr,weight_decay=1e-5)

# 计算模型参数量
total_parameters = 0
for i in model.parameters():
    total_parameters += i.numel()
print("模型总参数量:{}".format(total_parameters))

from torch.utils.data import DataLoader
train_loader=DataLoader(train,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test,batch_size=batch_size,shuffle=True)

import wandb
# 定义保存权重名称,并使用wandb记录训练过程
_weight_name = "xie_cwt_swinT"+str(myseed)+"_b"+str(batch_size)+"_lr"+str(lr)+"_e"+str(EPOCHS)
wandb.init(project='xie_cwt_swinT', name=time.strftime('%m%d%H%M%S-'+_weight_name))

from tqdm.auto import tqdm
best_acc = 0

# 定义训练和测试准确率和损失函数值,保存为csv文件
df = pd.DataFrame()
df_train_acc=[]
df_train_loss=[]
df_valid_acc=[]
df_valid_loss=[]

for epoch in range(EPOCHS):
    # ---------- Train ----------
    model.train()
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs,labels=batch
        logits = model(imgs.to(device))
        # print("train_logits:{}".format(logits.shape))
        loss = loss_F(logits, labels.long().to(device))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)
    
    # 一个epoch的平均loss和acc
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # wandb记录每一轮的训练损失函数和训练准确率
    wandb.log({"train_loss": train_loss,"epoch":epoch+1})
    wandb.log({"train_acc": train_acc,"epoch":epoch+1})

    # 保存每一轮的训练损失函数和训练准确率以供保存
    df_train_loss.append(train_loss)
    df_train_acc.append(train_acc.cpu().numpy())

    print(f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
    # lr_scheduler.step()
    
    # ---------- Validation ----------
    model.eval()
    valid_loss = []
    valid_accs = []
    for batch in tqdm(val_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        loss = loss_F(logits, labels.long().to(device))

        acc = (logits.argmax(dim=1) == labels.to(device)).float().mean()
    
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    wandb.log({"valid_loss": valid_loss,"epoch":epoch+1})
    wandb.log({"valid_acc": valid_acc,"epoch":epoch+1})

    df_valid_loss.append(valid_loss)
    df_valid_acc.append(valid_acc.cpu().numpy())

    print(f"[ Valid | {epoch + 1:03d}/{EPOCHS:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}") 
    
    # 判断当前轮valid_acc是否是最好,如是打印，保存权值
    if valid_acc > best_acc:
        print(f"[ Valid | {epoch + 1:03d}/{EPOCHS:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} "+"\033[1;31;40m -> best \033[0m ")
        print(f"Best model found at epoch {epoch+1}, saving model")
        torch.save(model.state_dict(), f"{_weight_name}_best.ckpt") 
        best_acc = valid_acc
    print("--"*30)
    
#记录训练和验证集结果
# print(df_train_acc)
df["epoch"] = [pad4(i) for i in range(1,EPOCHS+1)]
df["tarin_acc"] = df_train_acc
df["train_loss"] = df_train_loss
df["valid_acc"] = df_valid_acc
df["valid_loss"] = df_valid_loss
df.to_csv(_weight_name+"训练集和验证集结果.csv",index = False)

# 测试
# 导入权重推理测试
model.load_state_dict(torch.load(f"{_weight_name}_best.ckpt"))

prediction = []
true = []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        test_pred = model(x.to(device))
        # 预测的结果
        test_label = np.argmax(test_pred.cpu().numpy(), axis=1)
        prediction += test_label.tolist()
        true += y.cpu().numpy().tolist()

# 保存测试集结果
df2 = pd.DataFrame()
df2["Id"] = [pad4(i) for i in range(1, len(test) + 1)]
df2["Prediction"] = prediction
df2["True"] = true
df2.to_csv(_weight_name + "测试集结果.csv", index=False)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

accuracy = accuracy_score(df2["True"], df2["Prediction"])
F1 = f1_score(
    df2["True"],
    df2["Prediction"],
    average="macro",
    labels=np.unique(df2["Prediction"]),
)
recall = recall_score(
    df2["True"],
    df2["Prediction"],
    average="macro",
    labels=np.unique(df2["Prediction"]),
)
precision = precision_score(
    df2["True"],
    df2["Prediction"],
    average="macro",
    labels=np.unique(df2["Prediction"]),
)

print("accuracy:", accuracy)
print("F1:", F1)
print("recall:", recall)
print("precision:", precision)