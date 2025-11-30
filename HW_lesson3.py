import cv2
import numpy as np
from numpy.ma.core import filled
img = cv2.imread('images/2.jpg')
img = cv2.resize(img, (512, 512))
cv2.rectangle(img, (200,270), (300,170), (0, 200, 0), 2, 1)
cv2.putText(img, "Nevzinski Yehor",(120,450), cv2.FONT_ITALIC,1,(255,255,255))



img = cv2.imshow("Image", img)



cv2.waitKey(0)
cv2.destroyAllWindows()
