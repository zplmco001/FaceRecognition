from math import sqrt

import numpy
import cv2

imgPath1 = '2.jpg'
imgPath2 = '1.jpg'


def get_gray_scale(img_path):
    rgb_image = cv2.imread(img_path)
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)


def detect_faces(img_path):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_scale = get_gray_scale(img_path)
    faces = face_cascade.detectMultiScale(gray_scale, 1.2, 4)
    maxArea = 0
    bound = (0, 0, 0, 0)
    for (x, y, w, h) in faces:
        area = w * h
        if area > maxArea:
            bound = (x, y, w, h)
            maxArea = area

    cv2.rectangle(gray_scale, (bound[0], bound[1]), (bound[0] + bound[2], bound[1] + bound[3]), (255, 0, 0), 1)
    return gray_scale[y:bound[1] + bound[3], x:bound[0] + bound[2]]


def get_lbp(crop):
    cropH = crop.shape[0]
    cropW = crop.shape[1]
    lbpImg = numpy.zeros(shape=(cropH, cropW))
    for i in range(crop.shape[0]):  # traverses through height of the image
        for j in range(crop.shape[1]):  # traverses through width of the image
            if i != 0 and i != cropH - 1 and j != 0 and j != cropW - 1:
                threshold = crop[i][j]
                binary: str = ''
                for k in range(3):
                    for m in range(3):
                        if k != 1 or 1 != m:
                            if crop[i + k - 1][j + m - 1] >= threshold:
                                binary = binary + '1'
                            else:
                                binary = binary + '0'
                lbpImg[i][j] = int(binary, 2) / 255
                # print(lbpImg[i][j] * 255)
            else:
                lbpImg[i][j] = crop[i][j] / 255
    return lbpImg


def cal_histogram(lbp_img, crop):
    imgArea = crop.shape[0] * crop.shape[1]
    regionH = int(crop.shape[0]/8)
    regionW = int(crop.shape[1]/8)
    histogram = numpy.zeros(shape=256*8*8)
    for i in range(64):
        for a in range(regionH):
            for b in range(regionW):
                y = regionH * int(i / 8) + a
                x = regionW * int(i % 8) + b
                index = int(lbp_img[y][x] * 255) + (i * 256)
                histogram[index] = histogram[index] + 1
    for d in range(256 * 8 * 8):
        histogram[d] = histogram[d] / imgArea
    '''for a in range(lbp_img.shape[0]):
        for b in range(lbp_img.shape[1]):
            index = int(lbp_img[a][b] * 255)
            histogram[index] = histogram[index] + 1
    for d in range(256*8*8):
        histogram[d] = histogram[d] / imgArea'''
    return histogram


def compare_histograms(his1, his2):
    if len(his1) != len(his2):
        return -1
    distance = 0
    for ite in range(len(his2)):
        distance = distance + sqrt((his1[ite]-his2[ite])*(his1[ite]-his2[ite]))
    return distance


crop1 = detect_faces(imgPath1)
lbp1 = get_lbp(crop1)
hist = cal_histogram(lbp1, crop1)

crop2 = detect_faces(imgPath2)
lbp2 = get_lbp(crop2)
hist2 = cal_histogram(lbp2, crop2)

print(compare_histograms(hist, hist2))

'''for c in range(len(hist)):
    print('hist' + str(c) + ':' + str(hist[c]))'''

cv2.imshow('img', lbp1)
cv2.imshow('img2', lbp2)