import numpy as np
import cv2
import random
import os
import detectFaces
import pickle

faceDetector = detectFaces.FaceDetector()
faceDetector.getFacesFromFolder('obama', 'obama_faces')
faceDetector.getFacesFromFolder('bel-gets', 'otherFaces')
faceDetector.getFacesFromFolder('trump', 'otherFaces')
faceDetector.getFacesFromFolder('other', 'otherFaces')
faceDetector.getFacesFromFolder('omar', 'omarFaces')



training_data = []

names = os.listdir('obama_faces')
for name in names:
    path = os.path.join('obama_faces', name)
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        training_data.append([img, 0])
    except Exception as e:
        pass

names = os.listdir('omarFaces')
for name in names:
    path = os.path.join('omarFaces', name)
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        training_data.append([img, 1])
    except Exception as e:
        pass

print('omar_faces')

names = os.listdir('otherFaces')
for name in names:
    path = os.path.join('otherFaces', name)
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        training_data.append([img, 2])
    except Exception as e:
        pass

print(len(training_data))

random.shuffle(training_data)

out = open('dataset.pickle', 'wb')
pickle.dump(training_data, out)
out.close()

_in = open('dataset.pickle', 'rb')
training_data = pickle.load(_in)

print(training_data[0:10])





