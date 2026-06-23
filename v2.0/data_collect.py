import cv2
import mediapipe as mp
import csv

cap = cv2.VideoCapture(0)



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

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        key = cv2.waitKey(1) & 0xFF
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks and result.handedness:
            landmarks = result.hand_landmarks[0]
            hand_label = result.handedness[0][0].category_name
            features = normalize(landmarks, hand_label)
            
            #print(features)

            count = 0
            with open("hand_data.csv", "r") as f:
                for _ in csv.reader(f):
                    count += 1

            if ord("1") <= key <= ord("9"):
                with open("hand_data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([chr(key)] + features)
                    print("Captured ASL for ", [chr(key)], "Total lines are now:", count)

        cv2.imshow("Collect", frame)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()