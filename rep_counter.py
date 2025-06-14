# rep_counter.py
import numpy as np

def calculate_angle(a, b, c):
    # Convert Mediapipe landmarks to numpy arrays of (x, y) coordinates
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

class RepCounter:
    def __init__(self, exercise):
        self.exercise = exercise
        self.count = 0
        self.stage = None

    def update(self, landmarks):
        if self.exercise == "squat":
            hip = landmarks[24]  # Right hip
            knee = landmarks[26]
            ankle = landmarks[28]
            angle = calculate_angle(hip, knee, ankle)

            if angle < 90:
                self.stage = "down"
            if self.stage == "down" and angle > 160:
                self.stage = "up"
                self.count += 1

        # Add more exercises below
        elif self.exercise == "bicep_curl":
            shoulder = landmarks[12]  # Right shoulder
            elbow = landmarks[14]
            wrist = landmarks[16]
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 160:
                self.stage = "down"
            if self.stage == "down" and angle < 45:
                self.stage = "up"
                self.count += 1

        elif self.exercise == "pushup":
            shoulder = landmarks[12]
            elbow = landmarks[14]
            wrist = landmarks[16]
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 160:
                self.stage = "up"
            if self.stage == "up" and angle < 90:
                self.stage = "down"
                self.count += 1

        elif self.exercise == "tricep_extension":
            shoulder = landmarks[12]
            elbow = landmarks[14]
            wrist = landmarks[16]
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle < 60:
                self.stage = "bent"
            if self.stage == "bent" and angle > 160:
                self.stage = "extended"
                self.count += 1

        elif self.exercise == "shoulder_press":
            shoulder = landmarks[12]
            elbow = landmarks[14]
            wrist = landmarks[16]
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 160:
                self.stage = "down"
            if self.stage == "down" and angle < 90:
                self.stage = "up"
                self.count += 1

        return self.count
