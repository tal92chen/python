import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def downsampleY(y):
    reshapeY = y.reshape((y.shape[0] // 2, 2, y.shape[1] // 2, 2))
    resizeY = reshapeY.mean(axis = (1,3)).round()
    return resizeY.astype('uint8')

def readYUV(yuv_path, width=3840, height=2160, pixel_stride=2):
    with open(yuv_path,"rb") as file:
        yuv422_data = file.read()
    y_size = width * height
    u_size = width * height // 2 - 1
    u_size = width * height // 2 - 1
    y = np.frombuffer(yuv422_data[:y_size],dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(yuv422_data[y_size:y_size+u_size:2],dtype=np.uint8).reshape((height//2, width//2))
    v = np.frombuffer(yuv422_data[y_size+u_size::2],dtype=np.uint8).reshape((height//2, width//2))
    return y, u, v


file_path = r'C:\Users\user\Downloads\capture1016-12815\capture1016-12815.bin'
y, u, v = readYUV(file_path)
y_d = downsampleY(y)
yuv = np.stack((y_d,u,v),-1)
rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

plt.figure()
plt.imshow(rgb)
plt.show()

