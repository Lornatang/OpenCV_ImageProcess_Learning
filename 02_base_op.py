# 本节学习彩色图像与灰度图像的操作.
import warnings

import cv2
import numpy as np

filename = "lena.jpg"
van_gogh = "van_gogh.jpg"

################################################################
# 学习了如何读取一幅彩色图像, 以及如何以灰度图的方式读取一幅彩色图像.
# 这里可以尝试一下如何以彩色图的方式读取一幅灰度图像.
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)
cv2.imshow("bgr", image)

gray_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.imshow("bgr2gray", gray_image)

################################################################
# 通过getWidth()以及getHeight()的类成员函数, 对图像的宽和长进行获取, 并使用cout进行终端显示.
# 这里可以进入imageType源文件, 看看除了获取图像大小, 还有什么实用的图像信息获取函数.
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)
# image.shape函数返回HWC数据.
width, height, channels = image.shape

print(f"Use `*.shape` function.")
print(f"Image width:  {width}")
print(f"Image height: {height}")

print(f"Use `*.shape[0]/[1]` function.")
print(f"Image width:  {image.shape[0]}")
print(f"Image height: {image.shape[1]}")

################################################################
# 学习了如何创建空(黑色)的灰度图像以及彩色图像, 这里需要特别注意的是, 图像的宽对应矩阵的列数, 
# 图像的高对应矩阵的行数, 实际使用时, 注意不要弄混.
################################################################
# 创建224 * 224大小的黑色图像.
black_image = np.zeros((224, 224, 3), np.uint8)
cv2.imshow("black image", black_image)
# 创建224 * 224大小的白色图像.
white_image = np.zeros((224, 224, 3), np.uint8) + 255.0
cv2.imshow("white image", white_image)
# 创建224 * 224大小的灰色图像.
gray_image = np.zeros((224, 224, 1), np.uint8)
gray_image[:, :, 0] = np.zeros([224, 224]) + 127.5
cv2.imshow("gray image", gray_image)
# 创建224 * 224大小的蓝色图像. 注意OpenCV是BGR格式.
blue_image = np.zeros((224, 224, 3), np.uint8)
blue_image[:, :, 0] = np.zeros([224, 224]) + 255.0
cv2.imshow("blue image", blue_image)
# 创建224 * 224大小的绿色图像. 注意OpenCV是BGR格式.
green_image = np.zeros((224, 224, 3), np.uint8)
green_image[:, :, 1] = np.zeros([224, 224]) + 255.0
cv2.imshow("green image", green_image)
# 创建224 * 224大小的红色图像. 注意OpenCV是BGR格式.
red_image = np.zeros((224, 224, 3), np.uint8)
red_image[:, :, 2] = np.zeros([224, 224]) + 255.0
cv2.imshow("red image", red_image)

################################################################
# 尝试对图像的像素进行了操作，对于灰度图像，通过“[]”重载的运算符进行操作，对于彩色图像，
# 则通过调用setPixel以及getPixel成员函数进行操作。
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)
# 提取第十行,第十列的像素值,返回值是BGR格式.
pixel = image[10, 10]
print(f"B: {pixel[0]}\n"
      f"G: {pixel[1]}\n"
      f"R: {pixel[2]}\n")
# 修改第十行十列的像素值. 修改成白色.
raw_pixel = image[10, 10]  # [104, 128, 228]
new_pixel = [255, 255, 255]
cv2.imshow("raw pixel image", image)
image[10, 10] = new_pixel
cv2.imshow("set pixel image", image)

################################################################
# 对图像保存进行了实验，通过使用“save”成员函数进行图像保存的操作.
# 这里可以尝试对图像保存函数输入参数进行修改，看看保存后的图像相对于原始图像发生了哪些变化.
################################################################
image = cv2.imread(filename, cv2.IMREAD_COLOR)
image[10, 10] = [255, 255, 255]
cv2.imshow("02_set_pixel_lena", image)
cv2.imwrite("02_set_pixel_lena.jpg", image)
# 调整保存时候的参数. cv2.IMWRITE参数可以进行调整.
# IMWRITE_JPEG_QUALITY: 保存成JPEG格式的文件的图像质量，分成0-100等级，默认95.
# IMWRITE_JPEG_PROGRESSIVE: 增强JPEG格式，启用为1，默认值为0（False）.
# IMWRITE_JPEG_OPTIMIZE: 对JPEG格式进行优化，启用为1，默认参数为0（False）.
# IMWRITE_JPEG_LUMA_QUALITY: JPEG格式文件单独的亮度质量等级，分成0-100，默认为0.
# IMWRITE_JPEG_CHROMA_QUALITY: JPEG格式文件单独的色度质量等级，分成0-100，默认为0.
# IMWRITE_PNG_COMPRESSION: 保存成PNG格式文件压缩级别，从0-9，只越高意味着更小尺寸和更长的压缩时间，默认值为1（最佳速度设置）
# IMWRITE_TIFF_COMPRESSION: 保存成TIFF格式文件压缩方案
# ...
cv2.imwrite("02_set_pixel_lena2.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

################################################################
# 练习：将任意的两幅图像的透明度进行调整，并合成一幅图像进行保存，看看得到什么样的效果.
################################################################
lena_image = cv2.imread(filename, cv2.IMREAD_COLOR)
van_image = cv2.imread(van_gogh, cv2.IMREAD_COLOR)
cv2.imshow("lena", lena_image)
cv2.imshow("van", van_image)

# 两幅图像相加前提保证两张图像大小相等. 不一致则调整.
# 默认认为两张图像高宽是相等的.
height_equal, width_equal = True, True

# 判断图像高度是否一致.
if lena_image.shape[0] != van_image.shape[0]:
    warnings.warn("Two image width is not equal!")
    height_equal = False
# 判断图像宽度是否一致.
if lena_image.shape[1] != van_image.shape[1]:
    warnings.warn("Two image height is not equal!")
    width_equal = False

# 只要两幅图像宽或高有任意一项不一致就调整.
if not width_equal or not height_equal:
    print(f"Adjust two images to make them equal.")
    new_h, new_w, _ = lena_image.shape
    van_image = cv2.resize(van_image, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("resized van", van_image)
else:
    print(f"Two image size is equal.")

# 创建一幅224 * 224大小的全黑图像.
add_image = np.zeros((224, 224, 3), np.int8)
# lena 设置权重为0.5, van 设置权重为0.5, 不需要修正像素值,设置gamma为0.
add_image = cv2.addWeighted(lena_image, 0.5, van_image, 0.5, 0, add_image)
cv2.imshow("add image(0.5+0.5)", add_image)
# 创建一幅224 * 224大小的全黑图像.
add_image = np.zeros((224, 224, 3), np.int8)
# lena 设置权重为0.3, van 设置权重为0.7, 不需要修正像素值,设置gamma为0.
add_image = cv2.addWeighted(lena_image, 0.3, van_image, 0.7, 0, add_image)
cv2.imshow("add image(0.3+0.7)", add_image)

cv2.waitKey(0)
