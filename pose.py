import cv2
import mediapipe as mp
import time
import csv

# Initialize Mediapipe and camera
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# CSV file setup
csv_filename = "pose_rehor.csv"
batch_data = []

# Video capture setup
cap = cv2.VideoCapture(0)  # Replace 0 with your video URL if using an IP camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables for FPS calculation
frame_count = 0
prev_time = 0

# Open CSV file and keep it open during the entire loop
with open(csv_filename, mode="w", newline="") as file:
    csv_writer = csv.writer(file)
    # Write header row with joint names
    header = ["frame"]
    joint_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
        "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
        "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
        "right_heel", "left_foot_index", "right_foot_index"
    ]
    for name in joint_names:
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]
    csv_writer.writerow(header)

    with mp_pose.Pose(
        min_detection_confidence=0.5,  # Lower confidence for faster processing
        min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Process the image for pose detection
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Extract joint data
                landmarks = results.pose_landmarks.landmark
                row = [frame_count]
                for landmark in landmarks:
                    row += [landmark.x, landmark.y, landmark.z, landmark.visibility]
                batch_data.append(row)

            # Write batch to file every 10 frames
            if frame_count % 10 == 0 and batch_data:
                csv_writer.writerows(batch_data)
                batch_data = []

            # Overlay FPS on the video
            cv2.putText(
                image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

            # Flip the image horizontally for a selfie-view display
            cv2.imshow('MediaPipe Pose', image)

            # Break the loop on 'ESC' key press
            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Pose landmarks saved to {csv_filename}")
