from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load model
model = load_model('fer2013_mini_XCEPTION.110-0.65.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (64,64))
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                label = emotion_labels[preds.argmax()]
                emotion_probability = np.max(preds)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {emotion_probability*100:.2f}%", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Draw probability bars
                bar_x = x + w + 20
                bar_y = y
                for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
                    bar_length = int(prob * 100)
                    bar_height = 18
                    bar_color = (255, 0, 0)

                    # Filled bar
                    cv2.rectangle(frame, (bar_x, bar_y + i*25), (bar_x + bar_length, bar_y + i*25 + bar_height), bar_color, -1)
                    # Border
                    cv2.rectangle(frame, (bar_x, bar_y + i*25), (bar_x + 100, bar_y + i*25 + bar_height), (255,255,255), 1)

                    # Text
                    cv2.putText(frame, f"{emotion} ({int(prob*100)}%)", (bar_x + 110, bar_y + i*25 + 14), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

