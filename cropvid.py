import cv2
import os
import sys
import time

f = open('faces.faces', 'a')
g = open('facenum.num', 'r')
kolvofaces = g.read()
g = open('facenum.num', 'w')
facenum = int(kolvofaces)
a = 1
p = 1
# b = 'a.jpg'
h = 30
x = 30
y = 30
w = 30
# ln = 0
person = kolvofaces
path = 'Faceid/'
cascPath = os.getcwd() + "/Cascadehaar/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
print('Input name of the user')
name = input()
f.write("\n{}".format(name))
# ret, frame = video_capture.read()
# time.sleep(0.1)
# video_capture.release()
while True:

    # Press C to start making samples
    # Press C again to stop
    # Press ESC to exit
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Crop the face
    for (x, y, w, h) in faces:
        print(a)
    crop_img = frame[y:y + h, x:x + w]
    if cv2.waitKey(1) & 0xFF == ord('c'):
        p = p + 1
    if ((p % 2) == 0):
        print("User.{}.{}.jpg".format(person, a))
        b = "User.{}.{}.jpg".format(person, a)
        cv2.imwrite(os.path.join(path, b), crop_img)
        a = a + 1
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 0)

    cv2.imshow('Video', crop_img)

    #
    k = cv2.waitKey(10) & 0xff  # ESC to exit
    if k == 27:
        break
    #

# Stop capture
video_capture.release()
cv2.destroyAllWindows()
facenum = facenum + 1
g.write(str(facenum))
g.close()
f.close()
import trainer_final
