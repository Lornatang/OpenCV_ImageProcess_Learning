# 本节学习图像边缘检测滤波器的操作.

import cv2
import numpy as np

filename = "lena.jpg"
kernel_size = (5, 5)

################################################################
# 1.	边缘检测滤波器常用于对一幅图像中物体轮廓的增强和提取。
# 2.	边缘检测滤波器的核心是对图像进行微分。
# 3.	通常使用的边缘检测滤波器有：Sobel滤波器，Laplace滤波器，Canny边缘检测滤波器等。
################################################################
gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.imshow("gray", gray)
# Sobel().
# 第二个参数比较重要,这里参考了原文档.
# Sobel函数求完导数后会有负值，还有会大于255的值。
# 而原图像是uint8，即8位无符号数，所以Sobel()建立的图像位数不够，
# 会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S.
# 这点在培训文档中也有体现,最好是直接用现有的库先转换成16位再转为无符号8位.
edges_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
edges_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
# 从16位转为8位.
edges_x = cv2.convertScaleAbs(edges_x)
edges_y = cv2.convertScaleAbs(edges_y)
edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)
cv2.imshow("edges(Sobel)", edges)

# Canny().
# 第二个参数是最小像素.
# 第三个像素是最大像素.类似于阈值的性质.
# 操作很简单,同时可视化效果也比Sobel要好,也不会不
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("edges(100, 200)", edges)
edges = cv2.Canny(gray, 70, 150)
cv2.imshow("edges(70, 150)", edges)
edges = cv2.Canny(gray, 50, 100)
cv2.imshow("edges(50, 100)", edges)

# Laplace().
# ksize是滤波器尺寸大小. 是奇数.
# 同样要注意边界. 转换吧.
laplace = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
laplace = cv2.convertScaleAbs(laplace)
cv2.imshow("laplace(ksize=3)", laplace)
laplace = cv2.Laplacian(gray, cv2.CV_16S, ksize=5)
laplace = cv2.convertScaleAbs(laplace)
cv2.imshow("laplace(ksize=5)", laplace)
laplace = cv2.Laplacian(gray, cv2.CV_16S, ksize=7)
laplace = cv2.convertScaleAbs(laplace)
cv2.imshow("laplace(ksize=7)", laplace)

# 腐蚀.减小前景物体的边界大小.
# iterations参数表示只处理一次.
kernel = np.ones(tuple(kernel_size), np.uint8)
erosion = cv2.erode(gray, kernel, iterations=1)
cv2.imshow("erosion", erosion)
# 膨胀.增大前景物体的边界大小.
kernel = np.ones(tuple(kernel_size), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=1)
cv2.imshow("dilation", dilation)
# 开运算. 消除噪声.
# 先腐蚀后膨胀.
kernel = np.ones(tuple(kernel_size), np.uint8)
MORPH_OPEN = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imshow("MORPH_OPEN", MORPH_OPEN)
# 闭运算. 过滤前景对象中的黑点.
# 先膨胀后腐蚀.
kernel = np.ones(tuple(kernel_size), np.uint8)
MORPH_CLOSE = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imshow("MORPH_CLOSE", MORPH_CLOSE)
# 形态学梯度. 显示膨胀和腐蚀之间的差异.
kernel = np.ones(tuple(kernel_size), np.uint8)
MORPH_GRADIENT = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("MORPH_GRADIENT", MORPH_GRADIENT)
# 礼帽. 显示图像和开运算之间的差异.
kernel = np.ones(tuple(kernel_size), np.uint8)
MORPH_TOPHAT = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("MORPH_TOPHAT", MORPH_TOPHAT)
# 黑帽. 显示图像和闭运算之间的差异.
kernel = np.ones(tuple(kernel_size), np.uint8)
MORPH_BLACKHAT = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("MORPH_BLACKHAT", MORPH_BLACKHAT)

# 总结:
# 1. Sobel对噪声多的图像处理好,但是通常精度不高.
# 2. Canny很容易检测强边缘和弱边缘,调节阈值就好.
# 3. Laplace对噪声敏感,参考资料解释道:该算子是一种二阶导数算子,再边缘处容易产生一个陡峭的交叉. 算是一种锐化吧.
# 4. 后续一些形态变换多使用就掌握其中了原理.

cv2.waitKey(0)
