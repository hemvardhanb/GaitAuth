import os
import json
import time
import numpy as np

ENROLLMENTS_DIR = "enrolled_profiles"

def save_signature(name, signature_vector):
    os.makedirs(ENROLLMENTS_DIR, exist_ok=True)
    filepath = os.path.join(ENROLLMENTS_DIR, f"{name}.json")
    
    # If file exists, load existing and append or average
    data = {"name": name, "walks": []}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except:
            pass
            
    data["walks"].append(signature_vector.tolist())
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_all_enrolled():
    profiles = []
    if not os.path.exists(ENROLLMENTS_DIR):
        return profiles
        
    for filename in os.listdir(ENROLLMENTS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(ENROLLMENTS_DIR, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    name = data["name"]
                    walks = data["walks"]
                    # Calculate mean signature vector for this user
                    mean_vector = np.mean(walks, axis=0)
                    profiles.append({"name": name, "vector": mean_vector})
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return profiles

def extract_gait_features(keypoints_sequence):
    """
    keypoints_sequence: list of dicts or numpy arrays per frame.
    Each frame contains 33 pose landmarks (x, y, z, visibility).
    We extract robust kinematic features:
    1. Cadence / Step frequency (from ankle/hip cyclical motion)
    2. Stride length proxy (max distance between ankles / wrists)
    3. Joint angles (knee and hip flexion angles)
    4. Vertical head bob (y-coordinate of nose/head variance)
    5. Arm swing amplitude (shoulder-wrist distance variance)
    """
    if len(keypoints_sequence) < 10:
        return None
        
    seq = np.array(keypoints_sequence) # shape: (frames, 33, 4) [x, y, z, vis]
    
    # Landmark Indices (MediaPipe Pose):
    # 0: nose, 11: left_shoulder, 12: right_shoulder
    # 23: left_hip, 24: right_hip
    # 25: left_knee, 26: right_knee
    # 27: left_ankle, 28: right_ankle
    # 15: left_wrist, 16: right_wrist
    
    n_frames = seq.shape[0]
    
    # Extract time series signals
    # 1. Ankle distance (stride proxy)
    left_ankles = seq[:, 27, :2]
    right_ankles = seq[:, 28, :2]
    ankle_distances = np.linalg.norm(left_ankles - right_ankles, axis=1)
    
    # 2. Vertical head bob (nose y-coordinate relative to hip y-coordinate)
    noses = seq[:, 0, 1]
    hips_y = (seq[:, 23, 1] + seq[:, 24, 1]) / 2.0
    head_bob = noses - hips_y
    
    # 3. Knee angles (Left and Right knee flexion over time)
    def compute_angle(a, b, c):
        # Angle at point b given a, b, c
        ba = a - b
        bc = c - b
        cosine_angle = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return angle

    l_hip = seq[:, 23, :2]
    l_knee = seq[:, 25, :2]
    l_ankle = seq[:, 27, :2]
    left_knee_angles = compute_angle(l_hip, l_knee, l_ankle)
    
    r_hip = seq[:, 24, :2]
    r_knee = seq[:, 26, :2]
    r_ankle = seq[:, 28, :2]
    right_knee_angles = compute_angle(r_hip, r_knee, r_ankle)
    
    # 4. Arm swing (wrist-shoulder distance or horizontal offset)
    l_shoulder = seq[:, 11, :2]
    l_wrist = seq[:, 15, :2]
    left_arm_swing = np.linalg.norm(l_wrist - l_shoulder, axis=1)
    
    r_shoulder = seq[:, 12, :2]
    r_wrist = seq[:, 16, :2]
    right_arm_swing = np.linalg.norm(r_wrist - r_shoulder, axis=1)
    
    # Compute summary statistics (mean, std, min, max) for each feature stream
    features = []
    for sig in [ankle_distances, head_bob, left_knee_angles, right_knee_angles, left_arm_swing, right_arm_swing]:
        features.extend([
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig)
        ])
        
    return np.array(features, dtype=np.float32)

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
