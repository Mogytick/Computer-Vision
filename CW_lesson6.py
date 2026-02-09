import cv2
csp = cv2.VideoCapture(0)
while True:
    ret, frame = csp.read()

gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
gray1 = cv2.convertScaleAbs(gray1, alpha=(1.2), beta=(50))#korekcia kontrasty

while True:
    ret, frame2 = cap.read()
    if not ret:
        break


    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    gray2 = cv2.convertScaleAbs(gray2, alpha=(1.2), beta=(50))
    dif = cv2.subtract(gray1, gray2)


    thresh = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)


    countours, _ = cv2.findContours(thresh, cv2.RETR-EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in countours:
        if cv2.contourArea(cnt) > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)

            gray1 = gray2[y:y + h, x:x + w]
            cv2.imshow('gray1', frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()








