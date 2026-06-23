# asl-number-detector

## What It does
This project uses a random forest classifier to predict what number (1-9) someone is signing through a camera. Uses Mediapipe for hand tracking and OpenCV for displaying and capturing video from a camera.

## Demo
Coming soon

## How it works
I am using OpenCV to access my camera as a live video, and then using the provided frames with Mediapipe. Mediapipe then makes 21 Landmarks, which I turn into 63 features. The data from them is normalized. It then predicts what number is being signed based on the data, and from my ML model.

## Setup
1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Download the MediaPipe hand landmark model and place `hand_landmarker.task` in the project folder
4. Run:
   python predict.py

- Note: To change the camera, you must see the line `cap = cv2.VideoCapture(0)`. Change `0` to any number, and it will choose the camera from your device list accordingly. `0` means default camera.

## Requirements
- opencv-python
- mediapipe
- numpy
- scikit-learn
- joblib

| Built using Python 3.14.6
