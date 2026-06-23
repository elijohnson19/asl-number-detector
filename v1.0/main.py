import cv2
import mediapipe as mp
import pyttsx3

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
        
        if result.hand_landmarks:
            print("Hand detected!")
            for i, landmark in enumerate(result.hand_landmarks[0]):
                print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}")
        
            Landmarks = result.hand_landmarks[0]
            if Landmarks[8].y < Landmarks[5].y:
                if Landmarks[8].y < Landmarks[6].y:
                    index = True
            else:
                index = False

            if Landmarks[12].y < Landmarks[9].y:
                if Landmarks[12].y < Landmarks[10].y:
                    middle = True
            else:
                middle = False
            
            if Landmarks[16].y < Landmarks[13].y:
                if Landmarks[16].y < Landmarks[14].y:
                    ring = True
            else:
                ring = False

            if Landmarks[20].y < Landmarks[17].y:
                if Landmarks[20].y < Landmarks[18].y:
                    pinky = True
            else:
                pinky = False

            if Landmarks[4].y < Landmarks[2].y:
                if Landmarks[4].x > Landmarks[5].x:
                    
                    thumb = True
                else:
                    thumb = False
            else:
                thumb = False

            if Landmarks[5].y < Landmarks[0].y:
                orientation = True
            else:
                orientation = False

            if orientation:
                if index:
                    if not middle:
                        if not ring:
                            if not pinky:
                                
                                num = 1
                    if middle:
                        if not ring:
                            if not pinky:
                                if not thumb:
                                    num = 2
                                if thumb:
                                    num = 3
                        if ring:
                            if pinky:
                                
                                if Landmarks[4].x < Landmarks[8].x:
                                    if Landmarks[4].y - Landmarks[16].y < 50:
                                        if Landmarks[4].x - Landmarks[8].x < 50:
                                            num = 9
                                        num = 4
                                if thumb:
                                    if Landmarks[4].x > Landmarks[8].x:
                                        num = 5
                            if Landmarks[19].y < Landmarks[20].y:
                                if not thumb:
                                    num = 6
                        if not ring:
                            if pinky:
                                if not thumb:
                                    num = 7
                    if not middle:
                        if ring:
                            if pinky:
                                if not thumb:
                                    num = 8
                if not index:
                    if middle:
                        if ring:
                            if pinky:
                                if thumb:
                                    
                                    num = 9
            if num is not None:
                print(num)

                cv2.putText(frame, str(num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        
        cv2.imshow("Webcam", frame)

        #if lastNum != num:
        #    pyttsx3.speak(num)

        #lastNum = num
    
            

            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
