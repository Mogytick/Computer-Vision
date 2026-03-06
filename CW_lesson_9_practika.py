import cv2
import numpy as np
import os



eye_cascade = cv2.CascadeClassifier('data/Haarcascades/haarcascade_eye.xml')
face_net = cv2.dnn.readNetFromCaffe('data/DNN/deploy.prototxt','data/DNN/res10_300x300_ssd_iter_140000.caffemodel')

img = cv2.imread('Photo/img.jpg')
img = cv2.resize(img, (300, 300))




(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0 , (300,300), (104.0, 177.0, 123.0))# format foto

face_net.setInput(blob)
detections = face_net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]


    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x2, y2) = box.astype("int")
        x, y = max(0, x), max(0, y)
        x2, y2 = min( w - 1, x2 ), min(h - 1, y2)

        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)


# -----------------------Cascade-----------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray, 1.1, 10, minSize = (10, 10))
for (x, y, w, h) in eyes:
   cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#_________________________________________________________________
input_folder = "Photo"
output_folder = "image"
formats = (".jpg",".jpeg",".png",".webp", ".tiff")
os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(input_folder))

for file in files:
  if not file.lower().endswith(formats):
      continue

  path = os.path.join(input_folder, file)
  img1 = cv2.imread(path)
  if img1 is None:
      continue
  output_path = os.path.join(output_folder, file)
  cv2.imwrite(output_path, img)















img = cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()