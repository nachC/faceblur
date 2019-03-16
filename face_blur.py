import sys
import cv2
import numpy as np

def processFaces(faces, img):
    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi_color, (31, 31), 10)
        img[y:y+h, x:x+w] = blur
    return

img_name = sys.argv[1]
scaleFactor = float(sys.argv[2])
minNeighbors = int(sys.argv[3])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_static = cv2.imread('%s.jpg' % (img_name))

# h, w, c = img_static.shape
# img_static = cv2.resize(img_static, (w, h))

gray_static = cv2.cvtColor(img_static, cv2.COLOR_BGR2GRAY)
faces_static = face_cascade.detectMultiScale(gray_static, scaleFactor, minNeighbors)
processFaces(faces_static, img_static)

cv2.imshow('img', img_static)

cv2.waitKey(0)
cv2.destroyAllWindows()
    