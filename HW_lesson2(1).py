import cv2
import numpy as np
img = cv2.imread('images/2.jpg')
img = cv2.resize(img, (300,300))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 100, 100)
cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()