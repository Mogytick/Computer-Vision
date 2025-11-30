import numpy as np
import cv2
from numpy.ma.core import filled

img = np.zeros((512, 512, 3), np.uint8)

# img[100:150,200:280] = 109, 230, 130
# img[:] = 180, 200, 90

cv2.rectangle(img, (100,100), (200,200), (255, 60, 90), -2, 1)
cv2.line(img, (200,200), (242,100), (255, 60, 90), 2)
print(img.shape)
# cv2.line(img, (0,img.shape[0]//2), (img.shape[1], img.shape[0] // 2) (255, 256, 0),2)
cv2.circle(img, (200,200), 20, (255, 255, 0), -1)
cv2.putText(img, "NaVi<B8",(100,450), cv2.FONT_ITALIC,1,(255,255,255))



cv2.imshow("примітиви", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
