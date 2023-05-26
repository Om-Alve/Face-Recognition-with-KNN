import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
data = np.load('faces.npy')

X = data[:,1:].astype(np.uint8)
y = data[:,0]

knn.fit(X,y)

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    if ret == False:
        break
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(100,100))
        pred =  knn.predict([gray.flatten()])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0))
        cv2.putText(frame,str(pred[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(255, 0, 0),2)
    cv2.imshow('Webcam',frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
