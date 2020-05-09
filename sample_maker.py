from tkinter import *
from tkinter import messagebox
import cv2
import os
import time
import sys

path = 'Faceid/'
f = open('faces.faces', 'a')
g = open('facenum.num', 'r')
kolvofaces = g.read()
facenum = int(kolvofaces)
person = str(facenum)
g = open('facenum.num', 'w')
lis = set()
nums = int(0)
video_capture = cv2.VideoCapture(0)  # Video Source
cascPath = os.getcwd() + "/Cascadehaar/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)  # Path to haarcascade
lis = ['Остановить', 'Начать']
lis = lis * 1000
text = ''
bl = bool(0)
ret, frame = video_capture.read()

global num
num = 0
a = 1


def clicked():
    global text
    if txt.get() != '' and txt.get() != ' ':
        text = txt.get()
        window.destroy()
    else:
        messagebox.showwarning('ERROR', 'Input your name')


def cloaked():
    global a
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        print(a)
    crop_img = frame[y:y + h, x:x + w]
    cv2.imshow('Video', crop_img)
    global num
    num += 1
    # btn.configure(text=lis[num])
    b = "User.{}.{}.jpg".format(person, a)
    a += 1
    cv2.imwrite(os.path.join(path, b), crop_img)
    res = (num, 'samples')
    lbl.configure(text=res)


def cloakid():
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        print(a)
    crop_img = frame[y:y + h, x:x + w]
    cv2.imshow('Video', crop_img) # show cropped


def cleamed():
    window.destroy()


window = Tk()
window.geometry('200x90')
window.title("Create samples")
lbl = Label(window, text="Input your name", font=("Arial Bold", 20))
print(num)
lbl.grid(column=3, row=0)
txt = Entry(window, width=30)
txt.grid(column=3, row=1)
btn = Button(window, text="Done", command=clicked)
btn.grid(column=3, row=2)
window.mainloop()

f.write("\n{}".format(text))

window = Tk()
lbl = Label(window, text=(num, 'samples'), font=("Arial Bold", 20))
lbl.grid(column=1, row=0)
selected = IntVar()
window.geometry('330x70')
window.title("Face detection")
# rad1 = Radiobutton(window, text='Работа программы', value=1, variable=selected)
# rad1.grid(column=1, row=5)
btn = Button(window, text="Снимок", command=cloaked)
btn.grid(column=0, row=2)
bt = Button(window, text="Выход", command=cleamed)
bt.grid(column=1, row=2)
btnt = Button(window, text="Просмотр", command=cloakid)
btnt.grid(column=2, row=2)
window.mainloop()

facenum = facenum + 1  # update number of faces
g.write(str(facenum))
g.close()
f.close()
video_capture.release()  # stop capturing
cv2.destroyAllWindows()
import trainer_final
