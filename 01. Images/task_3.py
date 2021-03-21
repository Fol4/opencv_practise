import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

"""
Оба алгоритм основаны на нахождения нового центра и пернос картинки в это место путем изменения матрицы преоброзования.
После переноса убераем все черные границы
"""

def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    image_height, image_width, _ = image.shape
    point = (image_width // 2, image_height // 2)

    M = cv2.getRotationMatrix2D(point, angle, 1)

    new_image_size = int(math.ceil(cv2.norm((image_height, image_width), normType=cv2.NORM_L2)))
    M[0, 2] += (new_image_size - image_width) // 2
    M[1, 2] += (new_image_size - image_height) // 2

    image = cv2.warpAffine(image, M, (new_image_size, new_image_size))

    bordersY = []
    for i in range(len(image) - 1):
        if (np.mean(image[i]) == np.mean(image[0]) and np.mean(image[i + 1]) != np.mean(image[0])
                or np.mean(image[i + 1]) == np.mean(image[0]) and np.mean(image[i]) != np.mean(image[0])):
            bordersY.append(i + 1)

    bordersX = []
    for i in range(len(image[0]) - 1):
        if (np.mean(image[:, i]) == np.mean(image[:, 0]) and np.mean(image[:, i + 1]) != np.mean(image[:, 0])
                or np.mean(image[:, i + 1]) == np.mean(image[:, 0]) and np.mean(image[:, i]) != np.mean(image[:, 0])):
            bordersX.append(i + 1)

    image = image[bordersY[0]: bordersY[1], bordersX[0]: bordersX[1]]
    return image


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    image_height, image_width, _ = image.shape

    M = cv2.getAffineTransform(points1, points2)

    new_image_size = int(math.ceil(cv2.norm((image_height, image_width), normType=cv2.NORM_L2)))
    M[0, 2] += (new_image_size - image_width) // 2
    M[1, 2] += (new_image_size - image_height) // 2

    image = cv2.warpAffine(image, M, (new_image_size, new_image_size))

    bordersY = []
    for i in range(len(image) - 1):
        if (np.mean(image[i]) == np.mean(image[0]) and np.mean(image[i + 1]) != np.mean(image[0])
                or np.mean(image[i + 1]) == np.mean(image[0]) and np.mean(image[i]) != np.mean(image[0])):
            bordersY.append(i + 1)

    bordersX = []
    for i in range(len(image[0]) - 1):
        if (np.mean(image[:, i]) == np.mean(image[:, 0]) and np.mean(image[:, i + 1]) != np.mean(image[:, 0])
                or np.mean(image[:, i + 1]) == np.mean(image[:, 0]) and np.mean(image[:, i]) != np.mean(image[:, 0])):
            bordersX.append(i + 1)

    image = image[bordersY[0]: bordersY[1], bordersX[0]: bordersX[1]]
    return image

test_image = cv2.imread('task_3/lk.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

test_point = (0, 0)
test_angle = 90

transformed_image = rotate(test_image, test_point, test_angle)

test_point_1 = np.float32([[50, 50], [400, 50], [50, 200]])
test_point_2 = np.float32([[100, 100], [200, 20], [100, 250]])

transformed1_image = apply_warpAffine(test_image, test_point_1, test_point_2)

plt.imshow(transformed_image)
plt.axis('off')
plt.show()
plt.imshow(transformed1_image)
plt.axis('off')
plt.show()
