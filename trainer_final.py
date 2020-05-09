import cv2
import numpy as np
from PIL import Image
import os

# Path to database with faces
path = "Faceid/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascPath = os.getcwd() + "/Cascadehaar/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascPath);


# Take faces and id's from the path to database
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids


print ("\n [INFO] Creating trainer")
faces, ids = getImagesAndLabels(path)
# Creates trainer
recognizer.train(faces, np.array(ids))

# Saving trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Saving
print("\n [INFO] {0} faces scanneed. Exiting...".format(len(np.unique(ids))))
import facedectwithid_final
