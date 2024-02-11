import cv2 as cv
import numpy as np
import os
import shutil


def canny_demo(image):
    t = 50
    canny_output = cv.Canny(image, t, t * 2)

    return canny_output


def GetYellow(img):
    """
    提取图中的红色部分
    """
    # 转化为hsv空间
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # print(hsv.shape)
    # 颜色在HSV空间下的上下限
    low_hsv = np.array([11, 43, 46])
    high_hsv = np.array([25, 255, 255])

    # 使用opencv的inRange函数提取颜色
    mask = cv.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    Red = cv.bitwise_and(img, img, mask=mask)
    return Red


def get_roi(img):
    src = GetYellow(img)
    binary = canny_demo(src)
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 判断是否具有边框
    # if len(contours) == 0:
    #     print("detecting error")
    #     return 0, 0, 0, 0
    # if len(contours) != 2:
    #     print("no detection")
    #     return 0, 0, 0, 0
    # print(len(contours))
    list = []
    for c in range(len(contours)):
        # print(c)
        area = cv.contourArea(contours[c])
        arclen = cv.arcLength(contours[c], True)
        if area < 40 or arclen < 40:
            continue
        rect = cv.minAreaRect(contours[c])
        cx, cy = rect[0]

        box = cv.boxPoints(rect)
        box = np.int0(box)
        listX = [box[0][0], box[1][0], box[2][0], box[3][0]]
        listY = [box[0][1], box[1][1], box[2][1], box[3][1]]
        x1 = min(listX)
        y1 = min(listY)
        x2 = max(listX)
        y2 = max(listY)
        # print(x1, y1, x2, y2)
        width = np.int32(x2 - x1)
        height = np.int32(y2 - y1)
        if width < 20 or height < 20:
            continue
        roi = img[y1: y2, x1: x2]
        # print(width, height)
        # print('contours:')
        # print(x1, y1, x2, y2)
        dict = {'x1': str(x1), 'y1': str(y1), 'x2': str(x2), 'y2': str(y2)}
        list.append(dict)
    cv.destroyAllWindows()
    # print(list)
    return list
