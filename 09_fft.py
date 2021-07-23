# 本节是学习傅里叶变换操作.

# 在看本节的时候除去看教程以外,更多的是去思考了傅里叶背后的数学意义.
# 傅里叶公式有去看,但是目前来看太复杂了,需要慢慢吃透.
# 傅里叶中详细指出任何周期函数都可以认为是不同振幅,不同相位正弦波的叠加方法.
# 傅里叶有很多种形式,目前还在学习.

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = "lena.jpg"
image = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w, c = image.shape

# 1.制作正弦波灰度图像,通过快速离散傅里叶变换并观察spectrum数值.
N = 256
x = np.linspace(-np.pi, np.pi, N)
sine1D = 128.0 + (127.0 * np.sin(x * 2.0))
sine1D = np.uint8(sine1D)
sine2D = np.tile(sine1D, (N, 1))
plt.imshow(sine2D, cmap="gray")
plt.savefig("a.jpg")
gray = cv2.imread("a.jpg", 0)
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
print("======================================================")
print("================raw spectrum number===================")
print("======================================================")
print(fshift)
magnitude_spectrum = 10 * np.log(np.abs(fshift))
print("======================================================")
print("==============process spectrum number=================")
print("======================================================")
print(magnitude_spectrum)
plt.figure()
plt.title("1")
plt.imshow(magnitude_spectrum, cmap="gray")

# 2.制作正弦波灰度图像,按照0.5权重合成一幅新图像,通过快速离散傅里叶变换并观察spectrum数值.
N = 256
x = np.linspace(-np.pi, np.pi, N)

sine1D = 128.0 + (127.0 * np.sin(x * 2.0))
sine1D = np.uint8(sine1D)
sine2D = np.tile(sine1D, (N, 1))
plt.imshow(sine2D, cmap="gray")
plt.savefig("a.jpg")

sine1D = 128.0 + (127.0 * np.sin(x * 4.0))
sine1D = np.uint8(sine1D)
sine2D = np.tile(sine1D, (N, 1))
plt.imshow(sine2D, cmap="gray")
plt.savefig("b.jpg")

# 创建一幅224 * 224大小的全黑图像.
c = np.zeros((N, N, 3), np.int8)
a = cv2.imread("a.jpg", cv2.IMREAD_COLOR)
b = cv2.imread("b.jpg", cv2.IMREAD_COLOR)
# a 设置权重为0.5, b 设置权重为0.5, 不需要修正像素值,设置gamma为0.
c = cv2.addWeighted(a, 0.5, b, 0.5, 0)
cv2.imshow("add image(0.5+0.5)", c)
cv2.imwrite("c.jpg", c)

gray = cv2.imread("c.jpg", 0)
plt.figure()
plt.title("gray")
plt.imshow(gray, cmap="gray")
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
print("======================================================")
print("================raw spectrum number===================")
print("======================================================")
print(fshift)
magnitude_spectrum = 10 * np.log(np.abs(fshift))
print("======================================================")
print("==============process spectrum number=================")
print("======================================================")
print(magnitude_spectrum)
plt.figure()
plt.title("1")
plt.imshow(magnitude_spectrum, cmap="gray")

# 3.对Lena图像进行傅里叶变换,并且将频域数据乘以峰值为1的高斯分布,再进行反快速离散傅里叶变换.观察处理前与处理后的图像区别.
plt.figure()
plt.title("gray")
plt.imshow(gray, cmap="gray")

f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
print("======================================================")
print("================raw spectrum number===================")
print("======================================================")
print(fshift)
magnitude_spectrum = 10 * np.log(np.abs(fshift))
print("======================================================")
print("==============process spectrum number=================")
print("======================================================")
print(magnitude_spectrum)
plt.figure()
plt.title("no gaussian")
plt.imshow(magnitude_spectrum, cmap="gray")
edges = cv2.Canny(gray, 70, 150)
cv2.imshow("edges(no fft)", edges)

dst = cv2.GaussianBlur(gray, (5, 5), 0)
f = np.fft.fft2(dst)
fshift = np.fft.fftshift(f)
print("======================================================")
print("================raw spectrum number===================")
print("======================================================")
print(fshift)
magnitude_spectrum = 10 * np.log(np.abs(fshift))
print("======================================================")
print("==============process spectrum number=================")
print("======================================================")
print(magnitude_spectrum)
plt.figure()
plt.title("gaussian")
plt.imshow(magnitude_spectrum, cmap="gray")

ishift = np.fft.ifftshift(fshift)
f = np.fft.ifft2(ishift)
dst = np.abs(f)
plt.figure()
plt.title("rever gaussian")
plt.imshow(dst, cmap="gray")
plt.savefig("d.jpg")
gray = cv2.imread("d.jpg", 0)
edges = cv2.Canny(gray, 70, 150)
cv2.imshow("edges(ifft)", edges)

cv2.waitKey(0)
plt.show()
