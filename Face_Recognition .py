#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import os
from sklearn.svm import SVC

SVM = SVC(kernel="poly", C = 3.0)

all_names = os.listdir("./images")


all_faces = []
all_labels = []


personnames = {}
c = 0


# Data Preprocessing  - Making structured Data (X, Y) pair of data points
for name in all_names:    
    print(name +" loaded")
    
    x= np.load("images/"+name,allow_pickle=True)
    all_faces.append(x)
    
    l = c*np.ones(x.shape[0])
    all_labels.append(l)

    if personnames.get(c) is None:
        personnames[c] = name[0:-4]
        c +=1


X = np.concatenate(all_faces, axis = 0)
Y = np.concatenate(all_labels, axis = 0).reshape(-1, 1)
Y=Y.reshape((Y.shape[0],))


def train(X, Y, x_query):
    m = X.shape[0]
    SVM.fit(X,Y)
    pred = SVM.predict(x_query)
    return pred

# For capturing image from webcame
cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")


# In[2]:


while(True):
    ret,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    if len(faces)>0:
        for (a,b,c,d) in faces:
            print(a,b,c,d)
            roi_gray = gray[b:b+d,a:a+c]
            roi_color = frame[b:b+d,a:a+c]
            img_name = "my_image.png"
            cv2.imwrite(img_name,roi_gray)
            color = (255,0,0)
            stroke = 2
            wa=a+c
            hb=b+d
            cv2.rectangle(frame,(a,b),(wa,hb),color,stroke)
            offset = 5
            face_section = frame[b-offset: b+d+offset , a- offset : a+c+ offset]

            face_section = cv2.resize(face_section, (100,100))
    
            face_section = face_section.reshape(1, 30000)
            
            pred = train(X, Y, face_section)
            
            name = personnames[int(pred)]
            
            # Adding name of the person on the frame
            cv2.putText(frame, name, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2 , cv2.LINE_AA)
        cv2.imshow('Face Recognizer',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
            
cam.release()
cv2.destroyAllWindows()


# In[ ]:




