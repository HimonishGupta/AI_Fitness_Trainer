# real_time_feedback.py
import cv2
import mediapipe as mp
import joblib
import numpy as np
from rep_counter import RepCounter

model = joblib.load("pose_classifier.pkl")
counter_dict = {}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
        label = model.predict(flat)[0]

        if label not in counter_dict:
            counter_dict[label] = RepCounter(label)
        count = counter_dict[label].update(landmarks)

        cv2.putText(frame, f"{label.upper()} | Reps: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Trainer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
