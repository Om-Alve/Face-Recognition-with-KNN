import cv2
import numpy as np
import os
face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

name = input()
results = []
frames = []
cam = cv2.VideoCapture(0)
while True:
    ret,frame = cam.read()
    if ret == False:
        break
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0))
        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(100,100))
        cv2.imshow('face',gray)
    cv2.imshow('Webcam',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if(key == ord('c')):
        frames.append(gray.flatten())
        results.append([name])

data = np.hstack([results,frames])

filename = "faces.npy"

if os.path.exists(filename):
    loaded = np.load(filename)
    data = np.vstack([loaded,data])

np.save(filename,data)

cv2.destroyAllWindows()
