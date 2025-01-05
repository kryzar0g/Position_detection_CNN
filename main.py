import cv2
import mediapipe as mp
import numpy as np
import requests
import imutils

url= "https://172.16.5.149:8080/video"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# POSE ESTIMATION
cap = cv2.VideoCapture(url)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,enable_segmentation=True,refine_face_landmarks=True) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Recolor image to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make detection
        results = holistic.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks on the face, pose, and hands
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      landmark_drawing_spec = None,  
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        

        
                    

       
        

        # Display the image
        cv2.imshow('Mediapipe Holistic Feed', cv2.flip(image, 1))

        # Break loop when 'ESC' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
