'''
Author: xyx && yx282947664@163.com
Date: 2023-07-04 13:22:37
LastEditors: xyx && yx282947664@163.com
LastEditTime: 2023-07-04 13:36:55
Copyright (c) 2023 by xyx && yx282947664@163.com, All Rights Reserved. 
Description: 
'''
import  torch
import  os, glob
import  random, csv
import  visdom
import  time
import  torchvision
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms
from    PIL import Image
import matplotlib.pyplot as plt


class CWT(Dataset):

    def __init__(self, root, resize, mode):
        super().__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} 
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('cwt.csv')
        rate=[0.7,0.85]
        if mode=='train': 
            self.images = self.images[:int(rate[0]*len(self.images))]
            self.labels = self.labels[:int(rate[0]*len(self.labels))]
        elif mode=='val': 
            self.images = self.images[int(rate[0]*len(self.images)):int(rate[1]*len(self.images))]
            self.labels = self.labels[int(rate[0]*len(self.labels)):int(rate[1]*len(self.labels))]
        elif mode=="test": 
            self.images = self.images[int(rate[1]*len(self.images)):]
            self.labels = self.labels[int(rate[1]*len(self.labels)):]
        else:
            print("数据集模式选择错误！！！")


    def load_csv(self, filename):
        #如果以前生成过，直接读取  root+filename
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))

            print(len(images))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: 
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
            print('保存为csv文件成功,文件名为:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label
