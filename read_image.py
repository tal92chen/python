import numpy as np
import cv2
import matplotlib.pyplot as plt


def yuv422_to_rgb(yuv422_data, width, height):
    frame_size = width * height * 2
    if len(yuv422_data) != frame_size:
        raise ValueError("The size of the YUV422 data does not match the expected frame size")
    yuv = np.frombuffer(yuv422_data,dtype=np.uint8).reshape((height,width*2))
    rgb = np.zeros((height, width, 3),dtype= np.uint8)
    y = yuv[:, 0::2]
    u = yuv[:, 1::4].repeat(2, axis=1)
    v = yuv[:, 1::4].repeat(2, axis=1)

    yuv = np.stack((y, u, v), axis= -1)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    return rgb ,yuv


width = 3840
height = 2160
file_path = r'C:\Users\user\Downloads\capture1016-12815\capture1016-12815.bin'
with open(file_path,"rb") as file:
    yuv422_data = file.read()
yuv422_data = yuv422_data + b'00'
(rgb_image, yuv_image) = yuv422_to_rgb(yuv422_data, width, height)
plt.imshow(rgb_image)
plt.axis('off')
plt.show()
plt.imshow(yuv_image)
plt.axis('off')
plt.show()