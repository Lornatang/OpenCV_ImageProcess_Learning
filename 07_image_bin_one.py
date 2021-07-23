# 本节是图像二值化第一部分 学习二值化阈值的确定操作.

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = "lena.jpg"
note = "07.jpg"


# 求解图像二值化的阈值，主要的方法有三种：
# 1.	p-Tile法
# 2.	Mode法(x)
# 3.	判别分析法
# 经常使用的是p-tile法以及判别分析法，Mode方法不经常使用且不推荐使用。
# 本节中使用的方法可以从第三节直方图操作中扩展而来.


def show_histogram(gray, winname):
    # 调用OpenCV函数求灰度图像直方图.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # 输出灰度图像直方图
    plt.figure()
    plt.title(winname)
    plt.xlabel("Bins")
    plt.ylabel("num of Pixels")
    plt.xlim([0, 256])
    plt.plot(hist)


# 这里不局限于教程中的三种方法,从现有资料可以知道大约有13种方法.
lena_image = cv2.imread(filename, cv2.IMREAD_COLOR)
note_image = cv2.imread(note, cv2.IMREAD_COLOR)
lena_gray = cv2.cvtColor(lena_image, cv2.COLOR_BGR2GRAY)
note_gray = cv2.cvtColor(note_image, cv2.COLOR_BGR2GRAY)

lena_h, lena_w, _ = lena_image.shape
note_h, note_w, _ = lena_image.shape
cv2.imshow("BGR Lena", lena_image)
cv2.imshow("BGR note", note_image)
cv2.imshow("Gray lena", lena_gray)
cv2.imshow("Gray note", note_gray)
show_histogram(lena_gray, "Gray lena")
show_histogram(note_gray, "Gray note")


# 灰度平局值值法.
def function1(image) -> np.ndarray:
    """传入BGR图像"""
    dst = np.zeros((note_h, note_w), dtype=image.dtype)
    for i in range(note_h):
        for j in range(note_w):
            b = int(image[i, j, 0])
            g = int(image[i, j, 1])
            r = int(image[i, j, 2])
            dst[i, j] = (b + g + r) / 3

    return dst


# 百分比阈值（P-Tile法）.
# 根据先验概率来设定阈值，使得二值化后的目标或背景像素比例等于先验概率.
# 缺点:对于难以估计先验概率的图像很难确定其阈值.
def function2(gray, precent: float = 0.2) -> np.ndarray:
    """传入Gray图像"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    amount, count = 0, 0
    for i in range(256):
        amount += hist[i]
    for i in range(256):
        count = count + hist[i]
        if count >= amount * precent:
            return i


# 基于谷底最小值的阈值
# 判断直方图是否为双峰函数.
def is_two_peaks(hist) -> bool:
    count = 0
    for i in range(1, 255):
        if hist[i - 1] < hist[i] or hist[i + 1] < hist[i]:
            count += 1
            if count > 2:
                return False
    if count == 2:
        return True
    else:
        return False


def function3(gray) -> np.ndarray:
    iters = 0
    # 阈值极为两峰之间的最小值.
    peak = False
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist1 = np.ones([256], dtype=np.float64)
    hist2 = np.ones([256], dtype=np.float64)

    for i in range(256):
        hist1[i] = hist[i]
        hist2[i] = hist[i]

    if not is_two_peaks(hist2):
        hist2[0] = (hist[0] + hist[0] + hist[1]) / 3
        for i in range(1, 255):
            # 中间的点
            hist2[i] = (hist1[i - 1] + hist1[i] + hist1[i + 1]) / 3
        # 最后一点
        hist2[255] = (hist1[254] + hist1[255] + hist1[255]) / 3

        iters += 1
        if iters >= 1000:
            return False

    for i in range(1, 255):
        if hist2[i - 1] < hist2[i] or hist2[i + 1] < hist2[i]:
            peak = True
        if peak or hist2[i - 1] >= hist2[i] or hist2[i + 1] >= hist[i]:
            return i - 1

    return True


# OSTU大律法
def function4(gray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


# 二值化方法
def function5(gray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    return thresh


# P-tile方法
def function6(image) -> np.ndarray:
    min_threshold = function2(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, min_threshold, 255, cv2.THRESH_BINARY)

    return thresh


dst = function1(note_image)
cv2.imshow("function1", dst)
show_histogram(dst, "function1")

print(f"Function2() threshold: {function2(note_image)}")
dst = function6(note_image)
cv2.imshow("function6", dst)
show_histogram(dst, "function6")

print(f"Function3() threshold: {function3(note_image)}")

dst = function4(note_gray)
cv2.imshow("function4", dst)
show_histogram(dst, "function4")

dst = function5(note_gray)
cv2.imshow("function5", dst)
show_histogram(dst, "function5")

plt.show()
cv2.waitKey(0)
