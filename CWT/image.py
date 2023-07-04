"""
连续小波变换 CWT
参考论文：https://www.mdpi.com/2076-3417/8/7/1102/html
morlet 小波在轴承故障诊断中比较常用
"""
import numpy as np
import pywt
import matplotlib.pyplot as plt

def CWT(data, fs=12000):
    t = np.arange(0, len(data)) / fs
    #wavename = "cgau8"   # Frequence_cgau8 小波
    wavename = "morl"  # morlet 小波
    # wavename = "cmor3-3"  # cmor 小波
    totalscale = 256
    fc = pywt.central_frequency(wavename)  # 中心频率
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换
    return t,cwtmatr,frequencies

def Toimage(data,lable):
    n = 0
    for i in range(len(data)):
        data1 = data[i]
        # print(data1.shape)
        t,cwtmatr,frequencies = CWT(data1,fs=12000)
        n = n+1
        plt.figure(figsize=(1, 1), dpi=224)
        plt.contourf(t, frequencies, abs(cwtmatr))  # 画等高线图
        plt.ylabel("Frequency(Hz)")
        plt.xlabel('Time [sec]')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./image/2HP' + '/' + str(lable) + '/' + '{}'.format(n) + '.jpg',
                    pad_inches=0, bbox_inches='tight', format='jpg')
        plt.close()

B007 = np.load('npy/2HP/B007_data.npy')
B014 = np.load('npy/2HP/B014_data.npy')
B021 = np.load('npy/2HP/B021_data.npy')
IR007 = np.load('npy/2HP/IR007_data.npy')
IR014 = np.load('npy/2HP/IR014_data.npy')
IR021 = np.load('npy/2HP/IR021_data.npy')
OR007 = np.load('npy/2HP/OR007_data.npy')
OR014 = np.load('npy/2HP/OR014_data.npy')
OR021 = np.load('npy/2HP/OR021_data.npy')
Normal = np.load('npy/2HP/Normal_data.npy')

Toimage(Normal,'Normal')
Toimage(B007,'B007')
Toimage(B014,'B014')
Toimage(B021,'B021')
Toimage(IR007,'IR007')
Toimage(IR014,'IR014')
Toimage(IR021,'IR021')
Toimage(OR007,'OR007')
Toimage(OR014,'OR014')
Toimage(OR021,'OR021')