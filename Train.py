import os
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


id_names = pd.read_csv('id-names.csv')
id_names = id_names[['id', 'name']]

lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)


def create_train():
    faces = []
    labels = []
    for id in os.listdir('faces'):
        path = os.path.join('faces', id)
        try:
            os.listdir(path)
        except:
            continue
        for img in os.listdir(path):
            try:
                face = cv.imread(os.path.join(path, img))
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

                faces.append(face)
                labels.append(int(id))
            except:
                pass
    return np.array(faces), np.array(labels)


# Train LBPH Model
faces, labels = create_train()
print('Training LBPH Model...')
lbph.train(faces, labels)
lbph.save('Classifiers/TrainedLBPH.yml')
print('LBPH Training Complete!')

# Define Dataset Paths
train_dir = 'faces/'
val_dir = 'faces/'

# Image Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(64, 64), color_mode='grayscale', batch_size=32, class_mode='sparse')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(64, 64), color_mode='grayscale', batch_size=32, class_mode='sparse')

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN Model
print('Training CNN Model...')
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the Model
if not os.path.exists('Classifiers'):
    os.makedirs('Classifiers')

model.save('Classifiers/FaceCNN.h5')
print('CNN Model Training Complete! Saved as Classifiers/FaceCNN.h5')
