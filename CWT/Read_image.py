import cv2
# 读取图像
img = cv2.imread('image/2HP/test/B007/1.jpg')
# 获取图像尺寸
height, width, channels = img.shape
print('图像宽度为：', width)
print('图像高度为：', height)
print('图像通道数为：', channels)
