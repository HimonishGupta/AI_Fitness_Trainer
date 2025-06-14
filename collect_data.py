import cv2
import mediapipe as mp
import pandas as pd

POSE_LABEL = input("Enter exercise label (e.g., squat, pushup): ").strip()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

landmarks_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_landmarks = []
        for lm in results.pose_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])
        frame_landmarks.append(POSE_LABEL)
        landmarks_list.append(frame_landmarks)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

columns = [f'{axis}{i}' for i in range(33) for axis in ['x','y','z']] + ['label']
df = pd.DataFrame(landmarks_list, columns=columns)
df.to_csv(f"pose_data_{POSE_LABEL}.csv", index=False)

print(f"Data saved to pose_data_{POSE_LABEL}.csv")
