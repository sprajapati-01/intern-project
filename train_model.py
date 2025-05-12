import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('fer2013.csv')


# Preprocesspip install scikit-learn
X = []
y = []

for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32')
    pixels = pixels.reshape(48, 48, 1)
    X.append(pixels)
    y.append(row['emotion'])

X = np.array(X) / 255.0
y = to_categorical(y, num_classes=7)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('emotion_model.h5')
