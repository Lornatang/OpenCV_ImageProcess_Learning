# 本节是图像二值化第二部分 学习二值化阈值的确定操作.

# 从该节中认识到, 认知到学习编程最重要的是思维.
# 在本教程中学习到了很多图像处理的方法. 总而言之:多学多用多看.
# 目前更多的是知道遇到这类图像应该如何处理,欠缺的是其背后的数学理论知识.

# 二值化图像处理更多应用在OCR,医学图像分析着一些.

# 霍夫曼圆心也是一种办法.

# 二值图像处理思维方式:
# 1. 灰度化.(必须,二值化图像只支持灰度图)
# 2. 平滑去噪.
# 3. 运用第三讲的直方图操作.
# 4. 运用第七讲的确定阈值操作.
# 5. 获取二值化图像.
# 6. 处理二值化图像.(该步骤有许多需要注意的细节)

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = "08.jpg"


def p_tile(histogram, precent: float) -> int:
    amount, count = 0, 0

    for i in range(256):
        amount += histogram[i]

    for i in range(256):
        count = count + histogram[i]
        if count >= amount * precent:
            return i


image = cv2.imread(filename, cv2.IMREAD_COLOR)
cv2.imshow("image", image)

# 1.灰度化.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 2.平滑去噪(中值方法).
median_blur = cv2.medianBlur(gray, 5)
cv2.imshow("median_blur", median_blur)

# 3.计算直方图
histogram = cv2.calcHist([median_blur], [0], None, [256], [0, 256])
plt.figure()
plt.title("histogram")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.xlim([0, 256])
plt.plot(histogram)

# 4.分析直方图
# 使用P_tile法.
p_tile_thresh = p_tile(histogram, 0.331)
print(f"P_tile function thresh: {p_tile_thresh}.")
# 使用判别分析法.
_, otsu_thresh = cv2.threshold(median_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("otsu_thresh", otsu_thresh)

# 5.获取二值化图像.
_, thresh = cv2.threshold(median_blur, p_tile_thresh, 255, cv2.THRESH_BINARY)
cv2.imshow("p_tile_thresh", thresh)

# 6.使用形态学,先膨胀再收缩操作,去除图像上的白色小点.
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(thresh, kernel, iterations=3)
cv2.imshow("erosion", erosion)
dilate = cv2.dilate(erosion, kernel, iterations=3)
cv2.imshow("dilate", dilate)

# 7.颜色反转.
color_inversion = 255 - dilate
cv2.imshow("color_inversion", color_inversion)

# 8.使用moment方法求重心.
mu = cv2.moments(color_inversion, True)
x = mu["m10"] / mu["m00"]
y = mu["m01"] / mu["m00"]

# 9.输出重心坐标.
print(f"image center x:{x:.3f}, y:{y:.3f}")

cv2.circle(color_inversion, (int(x), int(y)), 4, 100, 2, 4)
plt.imshow(color_inversion)
plt.colorbar()

# 展示图像.
plt.show()
cv2.waitKey(0)
