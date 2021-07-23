# 本节学习图像直方图的操作.

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = "lena.jpg"

################################################################
# 直方图是对图像内像素值的统计分布，是蕴含了图像灰度/色彩特征分布信息的统计图.
# 通过对图像直方图的分析，可以了解一幅图像中都包含了哪些灰度/色彩信息，
# 进而对所要处理的图像信息进行有效提取与特征强化。
################################################################
print("Skip...")

################################################################
# 求一幅图像或者一个任意类型的数组的直方图，所用的函数是imageProcess::calcHistogram函数，
# 在imageProcess.h中有定义，可以详细参照源码的编程实现。
# 本函数中使用OpenCV的函数进行直方图的计算和获取。
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.imshow("bgr", image)
cv2.imshow("gray", gray)
# 调用OpenCV函数求灰度图像直方图.
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# 输出灰度图像直方图
plt.figure()
plt.title("gray histogram")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.xlim([0, 256])
plt.plot(hist)

# 调用OpenCV函数求彩色图像直方图.
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
channels = cv2.split(image)  # 分离BGR色彩.
colors = ("b", "g", "r")  # 可视化BGR直方图.

# 输出BGR图像直方图
plt.figure()
plt.title("bgr histogram")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.xlim([0, 256])
for (channel, color) in zip(channels, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

################################################################
# 对一幅图像的直方图进行拉伸，可以调整图像的明暗程度，能够对图像中无用信息进行有效截取，增强图像有用信息。
# 直方图的拉伸处理函数是imageProcess::stretchHisogram，在imageProcess.h中有定义。
# 直方图拉伸的重点在于如何确定拉伸的阈值（直方图左边阈值和右边阈值），
# 确定拉伸阈值的方法有两种：
# 1-固定阈值，设定左边和右边阈值为固定数值进行拉伸操作；
# 2-动态阈值，通过计算直方图左右两边像素数量相对于图像尺寸的比例，动态地计算直方图拉伸的阈值。
################################################################
gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.imshow("gray(hist full)", gray)
# 调用OpenCV函数求灰度图像直方图.
hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
# 输出灰度图像直方图
plt.figure()
plt.title("gray histogram(hist full)")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.xlim([0, 256])
plt.plot(hist_full)


def stretch_histogram1(image, pixel_max, pixel_min) -> np.ndarray:
    """实现用灰度变换公式拉伸直方图(固定阈值方式)

    Args:
        参数说明不想写.

    Notes:
        需要注意的是输出0~255范围的数值, 所以是unsigned int8类型. 要转换哈子.

    Returns:
        ...
    """
    normalize_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    stretch_image = normalize_image * (pixel_max - pixel_min) + pixel_min
    return stretch_image.astype("uint8")


def stretch_histogram2(image) -> np.ndarray:
    """实现用灰度变换公式拉伸直方图(动态阈值方式)

    Args:
        参数说明不想写.

    Returns:
        ...
    """
    # 获取图像宽高.
    image_height, image_width = image.shape
    # 创建直方图
    n = np.zeros(256, dtype=np.float64)
    p = np.zeros(256, dtype=np.float64)
    c = np.zeros(256, dtype=np.float64)
    # 遍历图像的每个像素,得到统计分布直方图
    for height in range(0, image_height):
        for width in range(0, image_width):
            n[image[height][width]] += 1
    # 归一化
    for i in range(0, 256):
        p[i] = n[i] / float(image.size)
    # 计算累积直方图
    c[0] = p[0]
    for i in range(1, 256):
        c[i] = c[i - 1] + p[i]
    # 计算新像素的值.
    stretch_image = np.zeros((image_width, image_height), dtype=np.uint8)
    for x in range(0, image_width):
        for y in range(0, image_height):
            stretch_image[x][y] = 255 * c[image[x][y]]

    return stretch_image


# 这行是借助公式灰度变换公式拉伸直方图(固定阈值方式).
gray_stretch1 = stretch_histogram1(gray, 255.0, 0.0)
cv2.imshow("gray(hist stretch1)", gray_stretch1)
# 调用OpenCV函数求拉伸后的灰度图像直方图.
hist_stretch1 = cv2.calcHist([gray_stretch1], [0], None, [256], [0, 256])
# 输出拉伸后的灰度图像直方图
plt.figure()
plt.title("gray histogram(hist stretch1)")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.xlim([0, 256])
plt.plot(hist_stretch1)

# 这行是借助公式灰度变换公式拉伸直方图(动态阈值方式).
gray_stretch2 = stretch_histogram2(gray)
cv2.imshow("gray(hist stretch2)", gray_stretch2)
# 调用OpenCV函数求拉伸后的灰度图像直方图.
hist_stretch2 = cv2.calcHist([gray_stretch2], [0], None, [256], [0, 256])
# 输出拉伸后的灰度图像直方图
plt.figure()
plt.title("gray histogram(hist stretch2)")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.xlim([0, 256])
plt.plot(hist_stretch2)

################################################################
# 对一幅图像进行Gamma变换，能够在尽量对亮/暗区域不饱和的情况下对暗/亮的区域进行图像增强处理，
# 相对于直方图拉伸，Gamma变换能够使亮/暗的部分进行增强的同时保证暗/亮的部分不发生饱和现象，
# 有点类似与HDR但效果比HDR有限，可以对单幅图像进行亮度简单调整时使用。
# Gamma变换函数是imageProcess::gammaTranform，
# 在imageProcess.h中有定义，可以具体参考源码实现。
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)
# 通过除以像素最大值先将图像像素值调整到0-1之间，然后进行不同γ值的gamma矫正.
gray_gamma0 = np.power(image / float(np.max(image)), 1)
gray_gamma1 = np.power(image / float(np.max(image)), 0.5)
gray_gamma2 = np.power(image / float(np.max(image)), 1.5)
cv2.imshow("src", gray_gamma0)
cv2.imshow("gamma gray(0.5)", gray_gamma1)
cv2.imshow("gamma gray(1.5)", gray_gamma2)

################################################################
# 明亮度和对比度的调整，能够调整图像整体的明/暗的程度以及明暗对比的程度，
# 具体函数是imageProcess::imgBrightnessContrast，
# 在imageProcess.h中有具体的实现，可详细参考。
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)


def contrast_image(image, alpha):
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=np.uint8)
    blank = np.zeros(image.shape, image.dtype)
    dst = cv2.addWeighted(image, alpha, blank, 1 - alpha, 0)

    return dst


# 小于1则返回较暗的图像. 大于1则增大亮度.
dst_image1 = contrast_image(image, 0.5)
dst_image2 = contrast_image(image, 1.5)
cv2.imshow("src", image)
cv2.imshow("contrast(0.5)", dst_image1)
cv2.imshow("contrast(1.5)", dst_image2)

################################################################
# 练习: 图像直方图输出练习：通过calcHistogram函数的调用，
# 将Lena灰度以及彩色的图像直方图进行输出，并保存为csv文件，利用excel的图表功能将直方图显示出来.
################################################################
# 已完成

################################################################
# 练习: 直方图拉伸练习：通过stretchHisogram函数的调用，反复调整直方图拉伸阈值，
# 观察Lena灰度图像的变化，并设定一个拉伸阈值，输出直方图，观察直方图拉伸前后，图像直方图的变化。
# 实际体会直方图拉伸的效果以及作用，理解直方图拉伸处理的应用场合。
################################################################
# 直方图拉伸对于图像中前景或者背景都亮或者暗都有所帮助. 在X光图像中对于骨骼结构过曝的情况也有所帮助.

################################################################
# 练习: Gamma变换练习：通过gammaTransform函数的调用，反复调整gamma数值，
# 观察gamma值在0-1区间对图像亮度的影响；观察gamma值在1-2区间，对图像亮度的影响。
# 确定一个0-1的Gamma值，确定一个1-2的Gamma，分别输出Gamma变换前后图像的直方图，
# 通过excel的相关图，观察Gamma变换对直方图的影响。
# 实际体会Gamma变换的效果和作用，理解Gamma变换的应用场合
################################################################

################################################################
# 练习: 图像亮度与对比图调整练习：通过imgBrightnessContrast函数的调用，反复调整b和c的参数值，
# 观察图像亮度与对比度对图像的影响。
################################################################
plt.show()
cv2.waitKey(0)
