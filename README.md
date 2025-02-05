# ✋ Sign Language Detection 🤟

## 🌟 Overview
This project implements a sign language detection system using computer vision and deep learning. It captures hand gestures via a webcam, processes the images, and classifies them into predefined sign language gestures. The system uses OpenCV for image processing, MediaPipe for hand tracking, and TensorFlow/Keras for classification.

![Screenshot 2025-02-05 133433](https://github.com/user-attachments/assets/f8b1e338-7127-41ca-86fb-573973ed7c9d)

## 🚀 Features
- 🎥 **Real-time Sign Language Detection**: Uses a webcam to detect and classify hand signs.
- ✋ **Hand Tracking**: Utilizes MediaPipe-based hand detection.
- 🧠 **Sign Classification**: Employs a trained deep learning model to classify hand gestures.
- 🌐 **Web Interface**: Provides a Flask-based web application for ease of use.
- 🖼 **Image Processing**: Crops and resizes hand regions for better model accuracy.

## 📂 Project Structure
- 🗂 `datacollection.py`: Captures hand gesture images and stores them for training.
- 🖥 `test.py`: Runs the Flask web application for real-time detection.
- 📜 `versions_needed.txt`: Lists required dependencies and their versions.

## 🛠 Installation
### Prerequisites
Ensure you have Python installed (required: Python 3.10).
Ensure you have Python installed (required: Python 3.10). Then, install the necessary dependencies:
```sh
pip install -r versions_needed.txt
```

## ▶️ Running the Project
### 📸 Collect Training Data
Run the following command to start capturing hand images for training:
```sh
python datacollection.py
```
Press `s` to save images, and `Esc` to exit.

### 🌍 Start the Web Application
Run the Flask app for real-time sign detection:
```sh
python test.py
```
Access the web interface at `http://127.0.0.1:5000/`.

## 🎯 Usage
- 📹 **Live Video Mode**: Click "Start" on the web interface to begin sign detection.
- 📤 **Upload Mode**: Upload an image to classify a sign.
- ⏹ **Stop Detection**: Click "Stop" to end the detection session.

## 📦 Dependencies
The following libraries are required:
- 🔢 TensorFlow 2.12.0
- 🔍 Keras 2.12.0
- 📷 OpenCV 4.8.0.76
- 🖐 cvzone 1.5.6
- 🖐 MediaPipe 0.10.0
- 🔢 NumPy, SciPy, Pillow, Matplotlib

## 🔮 Future Improvements
- ➕ Support for more sign language gestures.
- 🎯 Enhance model accuracy with additional training data.
- 📱 Implement a mobile-friendly UI.
- 🌎 Expand the dataset for multilingual sign recognition.

## ❤️ Acknowledgments
This project is built using open-source technologies like OpenCV, MediaPipe, and TensorFlow. Special thanks to the open-source community for their contributions. 🙌

