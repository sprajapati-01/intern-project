import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# loading model
model = load_model('emotion_model.h5')

# emotion labels
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
   4: "Sad", 5: "Surprise", 6: "Neutral"}

# start webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi)
        emotion = emotion_dict[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(10) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
