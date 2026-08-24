# Gait-Based Authentication MVP (Physical Access Control)

A complete Python implementation of a gait recognition biometric access control system using standard video feeds (webcam) and pose estimation.

## Features

1. **Pose Extraction**: Utilizes **MediaPipe Pose** to extract 3D skeletal keypoints per frame in real-time.
2. **Kinematic Feature Extraction**: Derives robust biomechanical gait signatures from walking sequences:
   - Stride length proxy (ankle distance variations)
   - Vertical head bob (head relative to hips)
   - Knee joint angle flexion (left and right knees)
   - Arm swing amplitude (wrist-shoulder motion)
3. **Enrollment Workflow**: Captures a 10-second walking pass, computes the feature embedding, and stores it in the local vector database (`enrolled_profiles/`).
4. **Continuous Recognition & Decision Layer**: Compares live walking clips against enrolled profiles using cosine similarity with a live confidence threshold slider, issuing `ACCESS GRANTED` or `ACCESS DENIED` decisions.
5. **Modern Desktop GUI**: Built with `customtkinter` featuring live skeleton overlay, status cards, and intuitive controls.

---

## Installation & Setup

1. Ensure Python 3.11+ is installed.
2. Install required dependencies:
   ```bash
   pip install mediapipe opencv-python numpy scikit-learn customtkinter
   ```

---

## Running the MVP

Launch the graphical user interface:
```bash
python gait_app.py
```

### How to Use:
1. **Enroll a Subject**:
   - Click **"Enroll New Person (10s)"**.
   - Enter your name or ID when prompted.
   - Walk across the webcam's field of view for 10 seconds.
2. **Test Recognition**:
   - Click **"Start Continuous Recognition"**.
   - Walk in front of the camera. The system will compute similarity against enrolled profiles in real-time.
   - Adjust the **Confidence Threshold** slider (0.50 – 0.95) to observe FAR / FRR trade-offs and access control decisions.

---

## Running Unit Tests

Verify core kinematic and similarity functions:
```bash
python test_gait.py
```
