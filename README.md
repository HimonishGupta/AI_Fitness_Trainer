#  AI Fitness Trainer with Rep Counter

This project is an AI-powered fitness trainer built using **Python**, **OpenCV**, and **MediaPipe**. It performs **real-time pose detection**, provides **rep counting**, and features a **Streamlit web interface** to guide and monitor users while exercising.

---

##  Features

- Real-time pose detection using MediaPipe
- Automatic rep counting based on joint angles
- Supports multiple exercises:
  -  Squats
  -  Push-ups
  -  Bicep Curls
  -  Tricep Extensions
  -  Shoulder Press
- Streamlit UI for ease of use and display
- Accurate tracking using angle-based logic for each exercise

---

##  How It Works

- MediaPipe captures 3D body landmarks via webcam.
- The `rep_counter.py` module calculates angles between joints (e.g., hip, knee, shoulder).
- Each exercise has angle thresholds and positional stages (e.g., "down", "up").
- When a valid movement pattern is completed (e.g., down → up), the rep counter increments.

---

##  Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- scikit-learn
- pandas



##  Sample Output / Screenshot

When the app is running and the user performs an exercise, the interface will display:

-  Real-time webcam feed with pose landmarks drawn
-  Detected exercise type using the trained ML model
-  Live rep counter for that exercise

![image](https://github.com/user-attachments/assets/11087e68-30ab-42e6-bd05-5915fb35a49c)

