import cv2
import numpy as np
from h5py.h5a import iterate

image = cv2.imread('image/1.jpg')
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR.BGR2GRAY)
print(image.shape)
#image = cv2.resize(image, (800, 500))
image = cv2.resize(image,(image.shape[1] // 2, image.shape[0] // 2))
image = cv2.Canny(image, 100,100)
canel = np.once(5,5, np.uint8)
image = cv2.dilate(image, canel, iterations = 1)#диляція розширює світлі області
image = cv2.erode(image, canel, itterations = 1)
cv2.imwrite("1", image)
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()