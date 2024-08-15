import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data(data_dir, target_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            label = int(filename.split('.')[1])
            img_path = os.path.join(data_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                image = cv2.resize(image, target_size)
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

data_dir = 'dataset'
images, labels = load_data(data_dir)
images = images / 255.0
images = np.expand_dims(images, axis=-1)
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('face_recognition_model.h5')
