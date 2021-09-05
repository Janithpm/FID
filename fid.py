import numpy as np
import cv2 
import pickle

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)
face_cascade = cv2.CascadeClassifier('cascades\\data\\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades\\data\\haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner\\trainner.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        endX = x + w
        endY = y + h
        cv2.rectangle(frame, (x, y), (endX, endY), (255, 0, 0), 2)

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
          print(id_)
          print(labels[id_])
          font = cv2.FONT_HERSHEY_SIMPLEX
          name = labels[id_].capitalize()
          posY = y - 10
          posX = x - 5
          cv2.putText(frame, name, (posX , posY), font, 0.5, (255,255,255), 1, cv2.LINE_AA) 

        eyes = eye_cascade.detectMultiScale(gray)
        for(ex, ey, ew, eh) in eyes:
            e_endX = ex + ew
            e_endY = ey + eh
            cv2.rectangle(frame, (ex, ey), (e_endX, e_endY), (0, 255, 0), 1)

    cv2.imshow('FID', frame)
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()