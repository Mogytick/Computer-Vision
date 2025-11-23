import cv2
import numpy as np
img1 = cv2.imread('images/3.jpg')
img1 = cv2.resize(img1,(1500,400))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.Canny(img1, 80, 80)


cv2.imshow('Image', img1)
cv2.waitKey(0)
