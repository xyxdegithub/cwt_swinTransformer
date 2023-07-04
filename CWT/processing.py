import random
import matplotlib
matplotlib.use('Agg')
from scipy.io import loadmat
import numpy as np

def split(DATA):
    step = 400;
    size = 1024;
    data = []
    for i in range(1, len(DATA) - size, step):
        data1 = DATA[i:i + size]
        data.append(data1)
        random.shuffle(data)

    npy = data[:300]
    npy = np.asarray(npy)

    return npy

# 读取CWRU数据集

B007 = loadmat('12KHZ_Data/2HP/12k_Drive_End_B007_2_120.mat')['X120_DE_time'].ravel()
B007_npy = split(B007)
np.save('npy/2HP/B007_data.npy', B007_npy)


B014 = loadmat('12KHZ_Data/2HP/12k_Drive_End_B014_2_187.mat')['X187_DE_time'].ravel()
B014_npy = split(B014)
np.save('npy/2HP/B014_data.npy', B014_npy)

B021 = loadmat('12KHZ_Data/2HP/12k_Drive_End_B021_2_224.mat')['X224_DE_time'].ravel()
B021_npy = split(B021)
np.save('npy/2HP/B021_data.npy', B021_npy)

IR007 = loadmat("12KHZ_Data/2HP/12k_Drive_End_IR007_2_107.mat")["X107_DE_time"].ravel()
IR007_npy = split(IR007)
np.save('npy/2HP/IR007_data.npy', IR007_npy)

IR014 = loadmat("12KHZ_Data/2HP/12k_Drive_End_IR014_2_171.mat")["X171_DE_time"].ravel()
IR014_npy = split(IR014)
np.save('npy/2HP/IR014_data.npy', IR014_npy)


IR021 = loadmat("12KHZ_Data/2HP/12k_Drive_End_IR021_2_211.mat")["X211_DE_time"].ravel()
IR021_npy  = split(IR021)
np.save('npy/2HP/IR021_data.npy', IR021_npy)


OR007 = loadmat("12KHZ_Data/2HP/12k_Drive_End_OR007@6_2_132.mat")["X132_DE_time"].ravel()
OR007_npy = split(OR007)
np.save('npy/2HP/OR007_data.npy', OR007_npy)


OR014 = loadmat("12KHZ_Data/2HP/12k_Drive_End_OR014@6_2_199.mat")["X199_DE_time"].ravel()
OR014_npy = split(OR014)
np.save('npy/2HP/OR014_data.npy', OR014_npy)


OR021 = loadmat("12KHZ_Data/2HP/12k_Drive_End_OR021@6_2_236.mat")["X236_DE_time"].ravel()
OR021_npy = split(OR021)
np.save('npy/2HP/OR021_data.npy',OR021_npy)

# # normal

Normal = loadmat("12KHZ_Data/2HP/normal_2_99.mat")["X099_DE_time"].ravel()
Normal_npy = split(Normal)
np.save('npy/2HP/Normal_data.npy', Normal_npy)
