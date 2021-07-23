# 本节是学习仿射变换操作.

import matplotlib.pyplot as plt
import numpy as np
import cv2

filename = "lena.jpg"
affine = "sudokusmall.jpg"

# cv2.warpAffine()
img = cv2.imread(filename, cv2.IMREAD_COLOR)
rows, cols, _ = img.shape

M = np.float32([[1, 0, 100], [0, 1, 50]])  #右移100-下移50
img_ret1 = cv2.warpAffine(img, M, (cols, rows))
M = np.float32([[1, 0, -100], [0, 1, -50]])  #左移100-上移50
img_ret2 = cv2.warpAffine(img, M, (cols, rows))
M = np.float32([[1, 0, -100], [0, 1, 50]])  #左移100-下移50
img_ret3 = cv2.warpAffine(img, M, (cols, rows))

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title("src")
ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  #matplotlib显示图像为rgb格式
ax[0, 1].set_title("right100-down50")
ax[0, 1].imshow(cv2.cvtColor(img_ret1, cv2.COLOR_BGR2RGB))
ax[1, 0].set_title("left100-up50")
ax[1, 0].imshow(cv2.cvtColor(img_ret2, cv2.COLOR_BGR2RGB))
ax[1, 1].set_title("left100-down50")
ax[1, 1].imshow(cv2.cvtColor(img_ret3, cv2.COLOR_BGR2RGB))

# cv2.warpAffine()
img = cv2.imread(affine, cv2.IMREAD_COLOR)
h, w, c = img.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (300, 300))

plt.subplot(121), plt.imshow(img), plt.title("Input")
plt.subplot(122), plt.imshow(dst), plt.title("Output")

# cv2.resize()
img = cv2.imread(filename, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

plt.subplot(121), plt.imshow(img), plt.title("Input")
plt.subplot(122), plt.imshow(dst), plt.title("Output")
plt.show()
