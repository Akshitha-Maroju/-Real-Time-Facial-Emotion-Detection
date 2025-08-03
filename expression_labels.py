import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load model
model = load_model("fer2013_mini_XCEPTION.110-0.65.hdf5", compile=False)

# Emotion labels and emojis (optional: add path to emoji images)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0),
    'Neutral': (128, 128, 128)
}

# Load Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        emotion_probability = np.max(preds)
        label = emotion_labels[preds.argmax()]

        # Draw face rectangle
        color = emotion_colors[label]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw emotion label
        cv2.putText(frame, f'{label}: {emotion_probability*100:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        # Draw probability bars for all emotions
        for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
            bar_width = int(prob * 100)
            cv2.rectangle(frame, (x + w + 10, y + i * 20), (x + w + 10 + bar_width, y + (i + 1) * 20 - 5),
                          emotion_colors[emotion], -1)
            cv2.putText(frame, f'{emotion} {int(prob*100)}%', (x + w + 120, y + i * 20 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Real-time Facial Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

