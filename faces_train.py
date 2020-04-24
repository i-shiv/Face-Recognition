import os
import numpy as np
from PIL import Image
import cv2
import pickle

ba = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(ba,"images")

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id=0
labels_ids={}
x_train=[]
y_labels=[]

for root,dirs,files in os.walk(img_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
            path= os.path.join(root, file)
            label= os.path.basename(os.path.dirname(path)).replace(" ","_")  #label= os.path.basename(root).replace(" ","_")
            #print(label,path)
            if label in labels_ids:
                pass
            else:
                labels_ids[label]= current_id
                current_id += 1

            id_ = labels_ids[label]
            #print(labels_ids)
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("labels.pickle","wb") as f:
    pickle.dump(labels_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")