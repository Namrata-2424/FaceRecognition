import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load ID-Name Mapping
df = pd.read_csv('id-names.csv')
id_names = dict(zip(df['id'], df['name']))

# Load Haar Classifier and LBPH Model
face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')
lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
lbph.read('Classifiers/TrainedLBPH.yml')

# Load CNN Model for Secondary Verification
cnn_model = load_model('Classifiers/FaceCNN.h5')

# Function for Illumination Correction
def adjust_gamma(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype('uint8')
    return cv.LUT(image, table)

# Function to Extract CNN Features
def extract_features(face):
    face_resized = cv.resize(face, (64, 64)) / 255.0
    return np.expand_dims(face_resized, axis=[0, -1])

# Open Camera
camera = cv.VideoCapture(0)

while cv.waitKey(1) & 0xFF != ord('q'):
    ret, img = camera.read()
    if not ret:
        break

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grey = adjust_gamma(grey)
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for x, y, w, h in faces:
        face_region = grey[y:y+h, x:x+w]
        face_region_resized = cv.resize(face_region, (220, 220))

        # LBPH Recognition
        label, confidence = lbph.predict(face_region_resized)
        name = "Unknown"

        if label in id_names and confidence < 100:
            name = id_names[label]
        else:
            # Use CNN Model for Secondary Verification
            features = extract_features(face_region_resized)
            cnn_prediction = cnn_model.predict(features)
            label_cnn = np.argmax(cnn_prediction)
            if label_cnn in id_names and cnn_prediction[0][label_cnn] > 0.7:
                name = id_names[label_cnn]

        # Draw Results
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(img, name, (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    
    cv.imshow('Face Recognition', img)

camera.release()
cv.destroyAllWindows()
