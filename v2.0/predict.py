import cv2
import mediapipe as mp
import pyttsx3
import numpy as np

import joblib
model = joblib.load("asl_model.pkl")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(cap.isOpened())

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'), 
    running_mode=RunningMode.VIDEO,  
    num_hands=1
)


def normalize(landmarks, hand_label):
    wrist = landmarks[0]
    coords = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in landmarks]
    
    if hand_label == "Left":
        coords = [(-x, y, z) for x, y, z in coords]
    scale = max(abs(v)for loc in coords  for v in loc)
    if scale == 0:
        scale = 1
    return [v / scale for loc in coords for v in loc]

lastNum = None
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("The camera did not capture frame")
            break

        output_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image)
        result = landmarker.detect_for_video(mp_image, timestamp)

        num = None
            
        if result.hand_landmarks and result.handedness:
            landmarks = result.hand_landmarks[0]
            hand_label = result.handedness[0][0].category_name
            features = normalize(landmarks, hand_label)
            X_input = np.array(features).reshape(1, -1)
            num = model.predict(X_input)

            cv2.putText(frame, str(num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        
        cv2.imshow("Webcam", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
