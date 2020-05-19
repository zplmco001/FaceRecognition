import numpy
import cv2

imgPath = '3.jpg'


def get_gray_scale(img_path):
    rgb_image = cv2.imread(img_path)
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)


def detect_faces(img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_scale = get_gray_scale(img_path)
    return face_cascade.detectMultiScale(gray_scale, 1.2, 4)


def crop_face(img_path, x, y, w, h):
    grayScale = get_gray_scale(img_path)
    cv2.rectangle(grayScale, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return grayScale[y:y + h, x:x + w]


faces = detect_faces(imgPath)
maxArea = 0
bound = (0, 0, 0, 0)

for (x, y, w, h) in faces:
    area = w*h
    if area > maxArea:
        bound = (x, y, w, h)

crop = crop_face(imgPath, bound[0], bound[1], bound[2], bound[3])
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
            print(lbpImg[i][j] * 255)
        else:
            lbpImg[i][j] = crop[i][j] / 255


def cal_histogram(lbp_img):
    histogram = numpy.zeros(shape=256)
    for a in range(lbp_img.shape[0]):
        for b in range(lbp_img.shape[1]):
            index = int(lbp_img[a][b] * 255)
            histogram[index] = histogram[index] + 1
    return histogram


hist = cal_histogram(lbpImg)

for c in range(len(hist)):
    print('hist' + str(c) + ':' + str(hist[c]))

cv2.imshow('img', lbpImg)
cv2.waitKey(0)
