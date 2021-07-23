# 本节学习图像平滑滤波器的操作.

import cv2
import numpy as np

filename = "lena.jpg"
noise_number = 256
kernel_size = (5, 5)

################################################################
# 读一幅图象的滤波处理,就是在图像上用滤波算子进行图像卷积操作.
################################################################

################################################################
# 在本质上,图像的滤波处理是：
# a.将图像进行傅立叶变换,将其变为频域；
# b.对图像频域信息进行低通、高通、带通等滤波器处理,这个处理过程是图像频域信息乘以滤波器的计算；
# c.将滤波器处理后的图像频域信息进行反傅立叶变换；
# d.获得时域信息,就是滤波之后的图像.
################################################################

################################################################
# 在频域中的相乘计算,根据傅立叶变换的特性,就是在时域做卷积运算,因此,对图像进行滤波处理,
# 其实就是图像的卷积操作.卷积所使用的算子,就是滤波器算子.
################################################################

################################################################
# 实际中通常使用的滤波器主要有：
# 高斯滤波器：它的滤波器算子是高斯分布；
# 中值滤波器：它的滤波器算子是取中值；
# 均值滤波器：它的滤波器算子是计算平均值；
# 双边滤波器：它的滤波器算子是双边算法.
################################################################

################################################################
# 这些常用滤波器,在XLibrary库中imageProcess内,具体函数如下：
################################################################


################################################################
# 重要的是对于滤波器尺寸的设定,也就是各函数中输入参数filterSize值的设定.
# 根据不同的应用场景、不同的滤波程度,filterSize的值也不相同.
# 例如需要对一幅原始图像中的白噪声进行滤波处理,由于白噪声是高频噪声,
# 因此绝大多数的白噪声滤波处理都可以使用高斯滤波器.
# 例如对于图像中存在的稀疏突变噪声进行滤波处理,可以使用中值滤波器.
# 例如既想保留边缘信息,又想把平缓的像素部分磨平,可以使用双边滤波器.
# 例如噪声密度很大,可以使用均值滤波器.
# 例如希望减少二值化之后图像中白色或黑色噪点,获得完整的二值图像区域,这时高斯滤波器是不错的选择.
################################################################

################################################################
# 因此滤波器的作用,不仅仅是去除图像中的噪声,而且在一些特殊的处理场合中能够实现较好的使用效果.
# 不同滤波器的使用条件、作用以及所实现的效果,需要通过实验进行体会
################################################################

################################################################
# 练习
# 	对Lena的灰度图像,使用不同类型的滤波器,相同的滤波器尺寸,进行滤波处理,体会处理前后图像所发生的变化.
# 	对Lena的灰度图像,使用相同的滤波器,不同的滤波器尺寸,进行滤波处理,体会处理前后图像所发生的变化.
# 	对Lena的灰度图像,通过图像像素操作,人为地加入一些噪声（整型随机数）,使用不通类型的滤波器以及调整滤波器尺寸,体会处理前后图像所发生的变化.
# 	对Lena的灰度图像,使用imageProcess中imgBinarize函数,对其进行二值化,观察二值化后的黑白图像；然后先对Lena的灰度图像进行滤波处理,再对其进行二值化,观察二值化后的黑白图像.比较两幅黑白图像,体会滤波器对二值化结果的影响以及作用.
################################################################
gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
h, w = gray.shape


# 椒盐噪声.
def create_salt_noise(image) -> np.ndarray:
    """模拟生成椒盐噪声图片, 传入的是灰度图"""
    salt_image = image.copy()

    # 椒盐随机分布点
    height_noise = np.random.randint(0, image.shape[0], noise_number)
    width_noise = np.random.randint(0, image.shape[1], noise_number)
    for index in range(noise_number):
        h_index = height_noise[index]
        w_index = width_noise[index]
        # 椒盐白色, 这个[]是w,h不是OpenCV中所说的HWC,坑.
        salt_image[w_index, h_index] = 255.0

    return salt_image


# 生成一张椒盐图像.
salt_image = create_salt_noise(gray)
cv2.imshow("gray", salt_image)

# 2D卷积(图像过滤).
kernel = np.ones(tuple(kernel_size), np.float32) / 25
dst = cv2.filter2D(salt_image, -1, kernel)
cv2.imshow("2d conv", dst)

# 均值滤波.
# 能够去除均匀分布和高斯分布的噪声,
# 但是在过滤掉噪声的同时,会对图像造成一定的模糊,使用的窗口越大,造成的模糊也就越明显.
dst = cv2.blur(salt_image, tuple(kernel_size))
cv2.imshow("blur", dst)

# 高斯滤波.
# 指定卷积核的宽度和高度,它应该是正数并且是奇数.
# 高斯模糊在从图像中去除高斯噪声方面非常有效.
dst = cv2.GaussianBlur(salt_image, tuple(kernel_size), 0)
cv2.imshow("GaussianBlur", dst)

# 中值滤波.
# 这对去除图像中的椒盐噪声非常有效.添加了50％的噪点并应用了中值模糊.
dst = cv2.medianBlur(salt_image, int(kernel_size[0]))
cv2.imshow("medianBlur", dst)

# 双边过滤.
# 在降低噪音方面非常有效,同时保持边缘清晰.
# 倒数第二个参数是颜色空间过滤器的Sigma值,表明该像素邻域内一定范围的颜色会混合在一起.
# 倒数第一个参数的空间坐标过滤器的Sigma值,表明两个不同坐标的颜色将会相互影响.
dst = cv2.bilateralFilter(salt_image, int(kernel_size[0]), 10, 10)
cv2.imshow("bilateralFilter", dst)

# 二值化图像.
# 函数第一个参数是源图像,它应该是灰度图像.
# 第二个参数是用于对像素值进行分类的阈值.
# 第三个参数是maxVal,它表示如果像素值大于（有时小于）阈值则要给出的值.
_, thresh = cv2.threshold(salt_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("THRESH_BINARY", thresh)
_, thresh = cv2.threshold(salt_image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("THRESH_BINARY_INV", thresh)
_, thresh = cv2.threshold(salt_image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow("THRESH_TRUNC", thresh)
_, thresh = cv2.threshold(salt_image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow("THRESH_TOZERO", thresh)
_, thresh = cv2.threshold(salt_image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow("THRESH_TOZERO_INV", thresh)

# 自适应阈值.
# 函数第一个参数是源图像,它应该是灰度图像.
# 函数第二个参数是maxVal,它表示如果像素值大于（有时小于）阈值则要给出的值.
# 函数第三个参数是自适应方法,决定如何计算阈值.
# 函数第四个参数是阈值类型.
# 函数第五个参数是邻域大小,它决定了阈值区域的大小.
# 函数第六个参数是从计算的平均值或加权平均值中减去的常数
thresh = cv2.adaptiveThreshold(salt_image,
                               maxValue=255,
                               adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                               thresholdType=cv2.THRESH_BINARY,
                               blockSize=3,
                               C=5)
cv2.imshow("ADAPTIVE_THRESH_MEAN_C", thresh)
thresh = cv2.adaptiveThreshold(salt_image,
                               maxValue=255,
                               adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               thresholdType=cv2.THRESH_BINARY,
                               blockSize=3,
                               C=5)
cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", thresh)

# Otsu's 二值化.
# Otsu's thresholding
_, thresh = cv2.threshold(salt_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("cv2.THRESH_BINARY + cv2.THRESH_OTSU", thresh)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(salt_image, tuple(kernel_size), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("cv2.THRESH_BINARY + cv2.THRESH_OTSU", thresh)

cv2.waitKey(0)
