import cv2
import numpy as np
import sys

import matplotlib.pyplot as plt

test_image = cv2.imread('task_2/image_02.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
plt.imshow(test_image)
plt.axis('off')
plt.show()

def decided(image):
    borders = []
    for pixel in range(1, len(image[0])):
        if(100 <= image[0][pixel][0] < 220 and (image[0][pixel - 1][0] <= 20 or 220 <= image[0][pixel - 1][0]) or 213 <= image[0][pixel - 1][0] < 220 and (image[0][pixel][0] <= 20 or 220 <= image[0][pixel][0])):
            borders.append(pixel)

    return borders

def object_search(image):
    borders = decided(image)
    car = -1
    emptyRoad = -1
    for road in range(len(borders) // 2):
        object = False
        for line in range(len(image)):
            for pixel in range(borders[road * 2], borders[road * 2 + 1]):
                if(not 200 <= image[line][pixel][0] < 220):
                    if(image[line][pixel][0] > 100):
                        object = True
                    else:
                        car = road
                        break;

        if (not object):
            emptyRoad = road

        if (car != -1 and emptyRoad != -1):
            return car, emptyRoad

car, road = object_search(test_image)

if(car == road):
    print('Не нужно перестраиваться')
else:
    print(f'Нужно перестроиться на дорогу номер {road + 1}')
