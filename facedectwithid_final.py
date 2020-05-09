import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = r"C:/SecuroServ/meh/Cascadehaar/haarcascade_frontalface_default.xml"
eyePath = r"C:/SecuroServ/meh/Cascadehaar/haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
eyeCascade = cv2.CascadeClassifier(eyePath);
names = []

with open('faces.faces') as f:
    names = f.read().splitlines()
font = cv2.FONT_HERSHEY_SIMPLEX
p = 1
# number of faces
id = 0
# Example if face id : ==> Moose id=1,  and another
# Start capturing, where 0 is number of your cam in PC
cam = cv2.VideoCapture(0)
# cam.set(3, 640) # set video widht
# cam.set(4, 480) # set video height
print (names)
# Min sizes of image
minW = 30  # 0.1*cam.get(3)
minH = 30  # 0.1*cam.get(4)
while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(int(minW), int(minH)),
    )
    # eye detection
    eyz = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(int(minW), int(minH)),
    )
    # if cv2.waitKey(1) & 0xFF == ord('s'):
    #    p = p + 1
    # if ((p % 2) == 0):
    for (x, y, w, h) in eyz:
        cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h - 5), (255, 0, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Checking the confidence (Change values)
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('pic.jpg', img)
        print('Фото')
    k = cv2.waitKey(10) & 0xff  # ESC to exit
    if k == 27:
        break

print("\n [INFO] Exiting")
cam.release()
cv2.destroyAllWindows()
f.close()
