# 本节学习图像色彩空间的变换的操作.

import cv2
import numpy as np

filename = "lena.jpg"


# 可见光波长在400~700nm之间
# 波长<400nm: 紫外线
# 波长=400nm: 蓝光
# 波长>700nm: 红外线
# 波长=700nm: 红光

# 色彩空间变换.
def rgb2xyz(image) -> np.ndarray:
    h, w, _ = image.shape

    for i in range(h):
        for j in range(w):
            (r, g, b) = image[i, j]
            x = 100 * (0.1903 * (pow(b / 255, 2.2)) + 0.3651 * (pow(g / 255, 2.2) + 0.3933 * (pow(r / 255, 2.2))))
            y = 100 * (0.0859 * (pow(b / 255, 2.2)) + 0.7071 * (pow(g / 255, 2.2) + 0.2123 * (pow(r / 255, 2.2))))
            z = 100 * (0.9570 * (pow(b / 255, 2.2)) + 0.1117 * (pow(g / 255, 2.2) + 0.0182 * (pow(r / 255, 2.2))))
            image[i, j] = [x, y, z]

    return image


def xyz2rgb(image) -> np.ndarray:
    pass


image = cv2.imread(filename, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("rgb", image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
cv2.imshow("COLOR_RGB2XYZ(OpenCV)", image)
image = cv2.cvtColor(image, cv2.COLOR_XYZ2RGB)
cv2.imshow("COLOR_XYZ2RGB(OpenCV)", image)
image = rgb2xyz(image)
cv2.imshow("COLOR_BGR2XYZ(Ours)", image)
cv2.waitKey(0)

