import cv2
import mediapipe as mp
import numpy as np
import time
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, messagebox

from gait_engine import extract_gait_features, save_signature, load_all_enrolled, cosine_similarity

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class GaitAuthApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Gait-Based Authentication MVP (Physical Access Control)")
        self.geometry("1100=700" if False else "1100x750")
        self.minsize(950, 650)

        # Video capture setup
        self.cap = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # State variables
        self.mode = "IDLE"  # IDLE, ENROLLING, RECOGNIZING
        self.enroll_name = ""
        self.enroll_start_time = 0
        self.enroll_duration = 10.0 # 10 seconds
        self.current_keypoints_buffer = []
        
        self.last_recognized_name = "Unknown"
        self.last_confidence = 0.0
        self.access_granted = False
        
        # Configurable threshold (slider)
        self.threshold = 0.75

        self.setup_ui()

        # Start video loop
        self.update_video_feed()

    def setup_ui(self):
        # Configure grid weight
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: Video Feed & Skeleton Overlay ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=10)
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.left_frame, text="Initializing Camera...", fg_color="black")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- RIGHT PANEL: Controls, Status & Decision Layer ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=10)
        self.right_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Title / Header
        title_label = ctk.CTkLabel(self.right_frame, text="Gait Access Control MVP", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Status Display Card
        self.status_card = ctk.CTkFrame(self.right_frame, fg_color=("gray85", "gray20"))
        self.status_card.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.status_card.grid_columnconfigure(0, weight=1)

        self.status_title = ctk.CTkLabel(self.status_card, text="SYSTEM STATUS: IDLE", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_title.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        self.identity_label = ctk.CTkLabel(self.status_card, text="Subject: —", font=ctk.CTkFont(size=13))
        self.identity_label.grid(row=1, column=0, padx=15, pady=2, sticky="w")

        self.confidence_label = ctk.CTkLabel(self.status_card, text="Confidence Score: 0.00", font=ctk.CTkFont(size=13))
        self.confidence_label.grid(row=2, column=0, padx=15, pady=2, sticky="w")

        self.access_decision_label = ctk.CTkLabel(self.status_card, text="Access: WAITING", font=ctk.CTkFont(size=16, weight="bold"), text_color="orange")
        self.access_decision_label.grid(row=3, column=0, padx=15, pady=(5, 15), sticky="w")

        # Action Buttons Frame
        btn_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=20, pady=15, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)

        self.enroll_btn = ctk.CTkButton(btn_frame, text="Enroll New Person (10s)", command=self.start_enrollment, fg_color="#2b8a3e", hover_color="#237032", height=40)
        self.enroll_btn.grid(row=0, column=0, pady=8, sticky="ew")

        self.recognize_btn = ctk.CTkButton(btn_frame, text="Start Continuous Recognition", command=self.toggle_recognition, fg_color="#1864ab", hover_color="#104d82", height=40)
        self.recognize_btn.grid(row=1, column=0, pady=8, sticky="ew")

        # Threshold Slider Frame
        slider_frame = ctk.CTkFrame(self.right_frame, fg_color=("gray90", "gray16"))
        slider_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        slider_frame.grid_columnconfigure(0, weight=1)

        self.slider_label = ctk.CTkLabel(slider_frame, text=f"Confidence Threshold: {self.threshold:.2f}", font=ctk.CTkFont(size=12))
        self.slider_label.grid(row=0, column=0, padx=15, pady=(10, 0), sticky="w")

        self.threshold_slider = ctk.CTkSlider(slider_frame, from_=0.5, to=0.95, number_of_steps=45, command=self.update_threshold)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="ew")

        # Enrolled Profiles List
        profiles_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        profiles_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        profiles_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(profiles_frame, text="Enrolled Database Profiles:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", pady=2)
        
        self.profiles_list_label = ctk.CTkLabel(profiles_frame, text=self.get_profiles_summary(), font=ctk.CTkFont(size=11), text_color="gray")
        self.profiles_list_label.grid(row=1, column=0, sticky="w", pady=2)

    def get_profiles_summary(self):
        profiles = load_all_enrolled()
        if not profiles:
            return "No profiles enrolled yet."
        names = [p["name"] for p in profiles]
        return f"Enrolled ({len(profiles)}): " + ", ".join(names)

    def update_threshold(self, val):
        self.threshold = float(val)
        self.slider_label.configure(text=f"Confidence Threshold: {self.threshold:.2f}")

    def start_enrollment(self):
        name = simpledialog.askstring("Enrollment", "Enter person's name / ID for enrollment:")
        if not name:
            return
        
        self.enroll_name = name.strip()
        self.mode = "ENROLLING"
        self.enroll_start_time = time.time()
        self.current_keypoints_buffer = []
        
        self.status_title.configure(text=f"SYSTEM STATUS: ENROLLING ({self.enroll_name})", text_color="#2b8a3e")
        self.identity_label.configure(text=f"Subject: {self.enroll_name}")
        self.confidence_label.configure(text="Please walk across camera view (10s)...")
        self.access_decision_label.configure(text="Action: RECORDING GAIT", text_color="#2b8a3e")

    def toggle_recognition(self):
        if self.mode == "RECOGNIZING":
            self.mode = "IDLE"
            self.status_title.configure(text="SYSTEM STATUS: IDLE", text_color="orange")
            self.identity_label.configure(text="Subject: —")
            self.confidence_label.configure(text="Confidence Score: 0.00")
            self.access_decision_label.configure(text="Access: WAITING", text_color="orange")
            self.recognize_btn.configure(text="Start Continuous Recognition", fg_color="#1864ab")
        else:
            profiles = load_all_enrolled()
            if not profiles:
                messagebox.showerror("Error", "No enrolled profiles found! Please enroll at least one person first.")
                return
            self.mode = "RECOGNIZING"
            self.current_keypoints_buffer = []
            self.status_title.configure(text="SYSTEM STATUS: RECOGNIZING", text_color="#1864ab")
            self.recognize_btn.configure(text="Stop Recognition", fg_color="#c92a2a", hover_color="#a61e1e")

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process via MediaPipe Pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # Extract keypoints array (33 landmarks x [x, y, z, visibility])
                landmarks = results.pose_landmarks.landmark
                kp_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
                
                if self.mode == "ENROLLING":
                    elapsed = time.time() - self.enroll_start_time
                    remaining = max(0.0, self.enroll_duration - elapsed)
                    self.current_keypoints_buffer.append(kp_array)
                    
                    # Update countdown UI
                    self.confidence_label.configure(text=f"Time remaining: {remaining:.1f}s | Frames: {len(self.current_keypoints_buffer)}")
                    
                    if elapsed >= self.enroll_duration:
                        # Finish enrollment
                        signature = extract_gait_features(self.current_keypoints_buffer)
                        if signature is not None:
                            save_signature(self.enroll_name, signature)
                            messagebox.showinfo("Success", f"Successfully enrolled gait signature for '{self.enroll_name}'!")
                            self.profiles_list_label.configure(text=self.get_profiles_summary())
                        else:
                            messagebox.showerror("Error", "Not enough movement detected during enrollment. Try again.")
                            
                        self.mode = "IDLE"
                        self.status_title.configure(text="SYSTEM STATUS: IDLE", text_color="orange")
                        self.access_decision_label.configure(text="Access: WAITING", text_color="orange")

                elif self.mode == "RECOGNIZING":
                    self.current_keypoints_buffer.append(kp_array)
                    # Keep sliding window of ~60-90 frames (~2-3 seconds)
                    if len(self.current_keypoints_buffer) > 90:
                        self.current_keypoints_buffer.pop(0)
                        
                    # Every 30 frames, perform recognition check
                    if len(self.current_keypoints_buffer) >= 60 and len(self.current_keypoints_buffer) % 15 == 0:
                        query_sig = extract_gait_features(self.current_keypoints_buffer)
                        profiles = load_all_enrolled()
                        
                        best_match = "Unknown"
                        best_score = -1.0
                        
                        for p in profiles:
                            score = cosine_similarity(query_sig, p["vector"])
                            if score > best_score:
                                best_score = score
                                best_match = p["name"]
                                
                        self.last_recognized_name = best_match
                        self.last_confidence = best_score
                        
                        self.identity_label.configure(text=f"Subject: {best_match}")
                        self.confidence_label.configure(text=f"Confidence Score: {best_score:.2f} (Threshold: {self.threshold:.2f})")
                        
                        if best_score >= self.threshold:
                            self.access_granted = True
                            self.access_decision_label.configure(text="ACCESS GRANTED (DOOR UNLOCKED)", text_color="#2b8a3e")
                        else:
                            self.access_granted = False
                            self.access_decision_label.configure(text="ACCESS DENIED", text_color="#c92a2a")

            # Convert to CTk compatible image
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            
            # Resize to fit left frame maintaining aspect ratio
            w_box = self.left_frame.winfo_width() - 20
            h_box = self.left_frame.winfo_height() - 20
            if w_box > 100 and h_box > 100:
                pil_image = pil_image.resize((w_box, h_box), Image.Resampling.BILINEAR)
                
            self.photo = ImageTk.PhotoImage(image=pil_image)
            self.video_label.configure(image=self.photo, text="")

        # Schedule next frame update (~30 fps)
        self.after(30, self.update_video_feed)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    app = GaitAuthApp()
    app.mainloop()
