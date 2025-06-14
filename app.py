import streamlit as st
import cv2
import numpy as np
import joblib
from rep_counter import RepCounter
import mediapipe as mp

st.title(" AI Fitness Trainer")
run = st.checkbox("Start Camera")

model = joblib.load("pose_classifier.pkl")
counter_dict = {}

if run:
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    FRAME_WINDOW = st.image([])

    while True:
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
