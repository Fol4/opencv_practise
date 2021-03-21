import cv2
import numpy as np
import sys

import matplotlib.pyplot as plt

image = cv2.imread("20 by 20 orthogonal maze.png")

def transpose(m):
    return [[row[i] for row in m] for i in range(len(m[0]))]

def start(map): # Пиксель входа
    for pixel in range(len(map[0])):
        if(map[0][pixel] == 0):
            return pixel

def end(map): # Пиксель выхода
    for pixel in range(len(map[-1])):
        if(map[-1][pixel] == 0):
            return pixel

def compession_black_pixels(image): # сжатие всех черных линий
    borderSize = 0
    for pixel in range(len(image[0])):
        border = True
        for line in range(len(image)):
            if ((image[line][pixel] == [255, 255, 255]).all()):
                border = False
                break

        if (border):
            borderSize += 1
        else:
            break

    return image[::borderSize, ::borderSize]

def new_list(image): # Замена картинки на бинарное поле (для удобства)
    comp = np.zeros((len(image), len(image[0])))

    for line in range(len(image)):
        for pixel in range(len(image[0])):
            if((image[line][pixel] == [0, 0, 0]).all()):
                comp[line][pixel] = -1
            else:
                comp[line][pixel] = 0

    return comp

def compression(map):
    '''
    В общем случае алгоритм звучит довольно легко, надо просто сжать все белые квадратики
    определенной размерность в 1 пиксель, но существует один фактор, который мешает это сделать.

    -1 -1 -1 -1 -1 ......
    -1  0  0  0 -1 ......
    -1  0  0  0  0 ......
    -1  0  0  0  0 ......
    -1  0  0  0  0 ......

    На примере мы видим, что мы не можем просто так сжать по определенному размеру квадрата так, как
    в 5 столбце образуется не квадрат а белая линия.
    Чтобы избежать этого было принято помечать эти значения и обрабатовать их, как черный пиксель
    '''
    width = set()
    height = set()
    blockSize = 0

    for pixel in map[0]:
        if(pixel == 0):
            blockSize += 1

    for line in range(1, len(map) - 1): # нахождение тех самых особых линий
        for pixel in range(1, len(map[0]) - 1):
            if(map[line][pixel] == -1 and map[line][pixel + 1] == map[line][pixel - 1] == 0 ):
                height.add(pixel)
            elif(map[line][pixel] == -1 and map[line - 1][pixel] == map[line + 1][pixel] == 0 ):
                width.add(line)

    for pixel in height: # разметка этих линий по вериткали
        for line in range(1, len(map) - 1):
            if(map[line][pixel] == 0):
                map[line][pixel] = -2

    for line in width: # разметка этих линий по горизонтали
        for pixel in range(1, len(map[0]) - 1):
            if(map[line][pixel] == 0):
                map[line][pixel] = -2

    comp1 = [[] for i in range(len(map))]
    pixel = 0

    while (pixel < len(map[0])):

        '''
        Основная из идей зжатия была в том, что если какой то кусок линии сжался в один размер,
        то все куски линии ниже и выше сжимаются также, как и та линия  
        '''

        for line in range(len(map)):
            comp1[line].append(map[line][pixel])

        if (map[0][pixel] == -1): # Определяем новую линия в сжатом массиве основоваясь на првилах описанных выше
            if (map[1][pixel] == 0):
                pixel += blockSize - 1
            pixel += 1
        elif (map[0][pixel] == 0):
            pixel += blockSize

    map = transpose(comp1)
    comp1 = [[] for i in range(len(map))]
    pixel = 0

    while (pixel < len(map[0])):
        for line in range(len(map)):
            comp1[line].append(map[line][pixel])
        if (map[0][pixel] == -1):
            if (map[1][pixel] == 0):
                pixel += blockSize - 1
            pixel += 1
        elif (map[0][pixel] == 0):
            pixel += blockSize

    return transpose(transpose(transpose(comp1))), height, width

def wave(line, pixel, deep, map, endX, endY): # Обычный волновой алгоритм
    if(line == endY and pixel == endX):
        map[line][pixel] = deep
        return map

    if(map[line][pixel] == -1 or (map[line][pixel] <= deep and map[line][pixel] > 0)):
        return map

    map[line][pixel] = deep

    if(pixel > 0):
        wave(line, pixel - 1, deep + 1, map, endX, endY)
    if(line > 0):
        wave(line - 1, pixel, deep + 1, map, endX, endY)
    if(line < len(map) - 1):
        wave(line + 1, pixel, deep + 1, map, endX, endY)
    if(pixel < len(map[0]) - 1):
        wave(line, pixel + 1, deep + 1, map, endX, endY)

    return map

def search_path(map, line, pixel, endX, endY, path): # Функция восстановления пути, по которому прошел волновой алгоритм
    path.append([line, pixel])

    if(line == endY and pixel == endX):
        return path
    elif (line > 0 and map[line - 1][pixel] == map[line][pixel] - 1):
        search_path(map, line - 1, pixel, endX, endY, path)
    elif (pixel > 0 and map[line][pixel - 1] == map[line][pixel] - 1):
        search_path(map, line, pixel - 1, endX, endY, path)
    elif (pixel < len(map[0]) - 1 and map[line][pixel + 1] == map[line][pixel] - 1):
        search_path(map, line , pixel + 1, endX, endY, path)
    elif (line < len(map) - 1 and map[line + 1][pixel] == map[line][pixel] - 1):
        search_path(map, line + 1, pixel, endX, endY, path)

    return path

def get_image_coord(pathX, pathY, height, width): # Получения координат точки из зжатой картинки в исходной
    now = 0
    while(pathX >= height[now] - (now + 1) * 6):
        now += 1

    imageX = (pathX - now) * 14 + now * 2 - 6

    now = 0
    while (now < len(width) and pathY >= width[now] - (now + 1) * 6):
        now += 1

    if(pathY > 0):
        imageY = (pathY - now) * 14 + now * 2 - 6
    else:
        imageY = 0

    return [imageX, imageY]

def compile_path(image, path, height, width): #Постороение пути (соединяем точки)
    for coord in range(len(path)):
        path[coord] = get_image_coord(path[coord][1], path[coord][0], height, width)

    coord = 0;
    while(coord < len(path) - 1):
        while (coord < len(path) - 1 and path[coord][0] == path[coord + 1][0] ):
            if (path[coord][1] < path[coord + 1][1] ):
                image[path[coord][1] : path[coord + 1][1] + 1, path[coord][0]: path[coord][0] + 2] = [255, 0, 0]
                coord += 1
            else:
                image[path[coord + 1][1]: path[coord][1] + 1, path[coord][0]: path[coord][0] + 2] = [255, 0, 0]
                coord += 1
        while (coord < len(path) - 1 and path[coord][1] == path[coord + 1][1]):
            if(path[coord][0] < path[coord + 1][0]):
                image[path[coord][1] : path[coord][1] + 2, path[coord][0]: path[coord + 1][0] + 1] = [255, 0, 0]
                coord += 1
            else:
                image[path[coord][1]: path[coord][1] + 2, path[coord + 1][0]: path[coord][0] + 1] = [255, 0, 0]
                coord += 1

    image[path[-1][1] : path[-2][1] + 1, path[-1][0]: path[-2][0] + 2] = [255, 0, 0]

    return image

map = new_list(compession_black_pixels(image))
map, height, width = compression(map)
width = list(width)
width.sort()
height = list(height)
height.sort()
startX, startY = start(map), 0
endX, endY = end(map), len(map) - 1
image = compile_path(image, search_path(wave(startY, startX, 1, map, endX, endY), endY, endX, startX, startY, []), height, width)

plt.imshow(image)
plt.axis('off')
plt.show()