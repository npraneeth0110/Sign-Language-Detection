from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import base64

app = Flask(__name__)

cap = None
is_running = False  # Set detection status

detector = HandDetector(maxHands=1)
classifier = Classifier(r"D:\Sign Language Detection\Model\keras_model.h5", r"D:\Sign Language Detection\Model\labels.txt")
labels = ["Hello", "Thank you", "Yes", "I Love You"]

offset = 20
imgSize = 300

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(img):
    """ Process the image to detect a hand sign and classify it. """
    hands, _ = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if w > 10 and h > 10:  # Ensure valid hand detection
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]), 
                          max(0, x - offset):min(x + w + offset, img.shape[1])]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = min(math.floor(k * w), imgSize)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + wCal] = imgResize[:, :min(wCal, imgSize)]
                else:
                    k = imgSize / w
                    hCal = min(math.floor(k * h), imgSize)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap:hGap + hCal, :] = imgResize[:min(hCal, imgSize), :]
                
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                return labels[index]
    return "No hand detected"

@app.route('/')
def index():
    return render_template('index.html', is_running=is_running, detected_sign=None, uploaded_image=None)

@app.route('/video')
def video():
    if not is_running:
        return "", 204
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global cap, is_running
    cap = cv2.VideoCapture(0)

    while is_running:
        success, img = cap.read()
        if not success:
            break

        detected_sign = process_image(img)
        cv2.putText(img, detected_sign, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Trebuchet MS substitution
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/start')
def start():
    global is_running
    is_running = True
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global is_running, cap
    is_running = False
    if cap:
        cap.release()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    img = cv2.imdecode(np.fromfile(filepath, np.uint8), cv2.IMREAD_COLOR)
    detected_sign = process_image(img)
    
    _, img_encoded = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(img_encoded).decode('utf-8')
    
    return render_template('index.html', is_running=is_running, detected_sign=detected_sign, uploaded_image=f"data:image/jpeg;base64,{img_b64}")

if __name__ == "__main__":
    app.run(debug=True)
