"""
Smart CCTV Surveillance System - Stage 4: Unified 7-Threat Real-Time Detector
=============================================================================
Runs complete real-time inference on CCTV video streams or live camera feeds,
detecting ALL 7 SECURITY THREATS simultaneously:
  1. Tactical Weapons (Knife, Pistol, Rifle)
  2. Fire (Active Flames)
  3. Smoke Plumes
  4. Person Fallen (Collapse / Slip-and-Fall)
  5. Violence / Fighting (Brawls, Physical Assault)
  6. Theft & Shoplifting (Stealing, Unauthorized Concealment)
  7. Intrusion & Burglary (Restricted Security Zone Breach)

Features:
- Real-time bounding box annotations with threat-specific color coding.
- Temporal debouncing to eliminate false alarms.
- Virtual Tripwire & Polygon Intrusion zone monitoring.
- Top surveillance status HUD overlay (Green for secure, Red for alert).
- Saves full annotated evidence video to predict/outputs/.
"""

import os
import sys
import math
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add parent directory to path for local imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict.zone_monitor import ZoneMonitor
from train.train_action_classifier import TemporalActionModel

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "predict" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Color palettes for bounding boxes (BGR format)
THREAT_COLORS = {
    "weapon": (0, 0, 255),       # Bright Red
    "fire": (0, 140, 255),       # Orange
    "smoke": (180, 105, 255),    # Purple
    "fall": (255, 0, 255),       # Magenta
    "intrusion": (0, 0, 255),    # Red
    "violence": (0, 0, 220),     # Dark Red
    "theft": (0, 215, 255),      # Gold / Yellow
}


class UnifiedSurveillanceDetector:
    """Master AI Surveillance Pipeline integrating all 7 threat detectors."""
    def __init__(self, spatial_weights="models/best.pt", action_weights="models/action_classifier.pt", conf_thresh=0.35):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh

        # 1. Load Spatial YOLO Model (Weapons, Fire, Smoke, Fall)
        spatial_path = BASE_DIR / spatial_weights
        if spatial_path.exists():
            print(f"[PIPELINE] Loading fine-tuned spatial model: {spatial_path.name}")
            self.spatial_model = YOLO(str(spatial_path))
        else:
            print("[PIPELINE] 'models/best.pt' not found. Using pre-trained 'yolov8n.pt'.")
            self.spatial_model = YOLO("yolov8n.pt")

        # Dedicated person detection for intrusion monitoring (COCO class 0: person)
        self.person_model = YOLO("yolov8n.pt")

        # 2. Load Temporal Action Classifier (Violence & Theft)
        action_path = BASE_DIR / action_weights
        self.action_model = None
        self.action_classes = ["normal", "violence", "theft_or_intrusion"]
        if action_path.exists():
            print(f"[PIPELINE] Loading temporal action model: {action_path.name}")
            ckpt = torch.load(str(action_path), map_location=self.device, weights_only=False)
            self.action_model = TemporalActionModel(num_classes=3, pretrained=False).to(self.device)
            self.action_model.load_state_dict(ckpt["model_state"])
            self.action_model.eval()
            self.action_classes = ckpt.get("classes", ["normal", "violence", "theft_or_intrusion"])
        else:
            print("[PIPELINE] Action classifier not found. Running spatial + intrusion detection.")

        # 3. Initialize Intrusion Zone Monitor
        self.zone_monitor = ZoneMonitor()

        # Sliding window buffer for action recognition (16 frames)
        self.frame_buffer = deque(maxlen=16)

        # Temporal debouncing counters (threat_name -> consecutive_detections)
        self.debounce_tracker = {t: 0 for t in ["weapon", "fire", "smoke", "fall", "intrusion", "violence", "theft"]}
        self.CONFIRM_THRESHOLD = 2  # Requires 2 consecutive frames to confirm alarm

    def preprocess_action_frame(self, frame):
        """Preprocesses frame for MobileNetV2 temporal sequence."""
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        return np.transpose(norm, (2, 0, 1))

    def process_frame(self, frame):
        """Processes a single surveillance frame across all 7 threat checks."""
        h, w = frame.shape[:2]
        raw_frame = frame.copy()  # Unannotated frame for ML models
        current_frame_threats = set()
        active_confirmed_threats = set()

        # --- A. Spatial Detection (YOLO: Weapon, Fire, Smoke, Fall) ---
        spatial_res = self.spatial_model.predict(source=frame, conf=self.conf_thresh, verbose=False)[0]
        person_boxes = []

        # Detect people with person model (COCO class 0)
        person_res = self.person_model.predict(source=raw_frame, conf=0.40, classes=[0], verbose=False)[0]
        for pbox in person_res.boxes:
            person_boxes.append(list(map(int, pbox.xyxy[0].tolist())))

        for box in spatial_res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self.spatial_model.names.get(cls_id, f"class_{cls_id}").lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Fall is also considered a person state
            if name == "fall":
                person_boxes.append([x1, y1, x2, y2])

            if name in ["weapon", "fire", "smoke", "fall"]:
                current_frame_threats.add(name)
                color = THREAT_COLORS.get(name, (0, 0, 255))

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"{name.upper()} {conf*100:.0f}%"
                label_y = y1 - 6 if y1 >= 25 else y2 + 18
                bg_top = y1 - 22 if y1 >= 25 else y2
                bg_bottom = y1 if y1 >= 25 else y2 + 22
                cv2.rectangle(frame, (x1, bg_top), (x1 + len(tag)*10, bg_bottom), color, -1)
                cv2.putText(frame, tag, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --- B. Intrusion & Perimeter Monitor ---
        is_intrusion, breached_pts = self.zone_monitor.check_intrusion(person_boxes, frame.shape)
        if is_intrusion:
            current_frame_threats.add("intrusion")
            for box, pt in breached_pts:
                cv2.circle(frame, pt, 6, (0, 0, 255), -1)
                cv2.putText(frame, "INTRUDER", (pt[0] - 25, pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = self.zone_monitor.draw_zone(frame)

        # --- C. Temporal Action Recognition (Violence & Theft) ---
        self.frame_buffer.append(self.preprocess_action_frame(raw_frame))

        if self.action_model is not None and len(self.frame_buffer) == 16:
            seq_tensor = torch.tensor(np.array([list(self.frame_buffer)]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.action_model(seq_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                action_name = self.action_classes[pred_idx]
                action_conf = probs[pred_idx].item()

                if action_name == "violence" and action_conf > 0.55:
                    current_frame_threats.add("violence")
                elif action_name == "theft_or_intrusion" and action_conf > 0.60:
                    current_frame_threats.add("theft")

        # --- D. Temporal Debouncing (Anti-False-Alarm Filter with ceiling) ---
        MAX_DEBOUNCE = 10
        for threat in self.debounce_tracker:
            if threat in current_frame_threats:
                self.debounce_tracker[threat] = min(MAX_DEBOUNCE, self.debounce_tracker[threat] + 1)
            else:
                self.debounce_tracker[threat] = max(0, self.debounce_tracker[threat] - 1)
            if self.debounce_tracker[threat] >= self.CONFIRM_THRESHOLD:
                active_confirmed_threats.add(threat)

        # --- E. Draw Top Surveillance Status HUD Header ---
        header_color = (0, 0, 180) if active_confirmed_threats else (0, 100, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), header_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if active_confirmed_threats:
            threat_str = " | ".join(t.upper() for t in sorted(active_confirmed_threats))
            hud_text = f"[ALERT] THREAT DETECTED -> [{threat_str}]"
            text_color = (255, 255, 255)
        else:
            hud_text = "SURVEILLANCE ACTIVE | ALL 7 THREAT CHANNELS SECURE"
            text_color = (180, 255, 180)

        cv2.putText(frame, hud_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)

        return frame, active_confirmed_threats


def run_surveillance(source_path, max_frames: int = 300, conf: float = 0.35):
    """Processes any video or webcam feed with full 7-threat surveillance."""
    detector = UnifiedSurveillanceDetector(conf_thresh=conf)
    is_cam = isinstance(source_path, int) or (isinstance(source_path, str) and str(source_path).isdigit())
    cap = cv2.VideoCapture(int(source_path) if is_cam else str(source_path))

    if not cap.isOpened():
        print(f"[ERROR] Unable to open surveillance feed: {source_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_raw) if fps_raw and not math.isnan(fps_raw) and fps_raw > 0 else 25.0

    if width <= 0 or height <= 0:
        ret, test_frame = cap.read()
        if ret:
            height, width = test_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            width, height = 640, 480

    src_name = "live_cctv" if is_cam else Path(str(source_path)).stem
    out_path = OUTPUTS_DIR / f"surveillance_{src_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[WARNING] VideoWriter could not be opened for {out_path}")

    print(f"\n[SURVEILLANCE] Streaming '{src_name}' ({width}x{height} @ {fps:.1f} FPS)...")
    frame_idx = 0
    total_threats_logged = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        processed_frame, active_threats = detector.process_frame(frame)
        if out.isOpened():
            out.write(processed_frame)
        frame_idx += 1

        if active_threats:
            for t in active_threats:
                if t not in total_threats_logged:
                    print(f"  [ALERT] [FRAME {frame_idx}] CONFIRMED THREAT: {t.upper()}")
                total_threats_logged.add(t)

        if frame_idx % 50 == 0:
            print(f"  ... monitored {frame_idx} frames")

    cap.release()
    if out.isOpened():
        out.release()

    print("\n" + "=" * 65)
    print(" SURVEILLANCE RUN FINISHED!")
    print(f" Frames Analyzed: {frame_idx}")
    if total_threats_logged:
        print(f" Confirmed Threats Detected: {', '.join(t.upper() for t in total_threats_logged)}")
    else:
        print(" Security Status: 100% Secure (Zero Threats Detected)")
    print(f" Output Video Evidence Saved: {out_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified 7-Threat CCTV Surveillance Inference")
    parser.add_argument("--source", type=str, required=True, help="Video path (.mp4) or '0' for live webcam")
    parser.add_argument("--max-frames", type=int, default=250, help="Max frames to process (default: 250)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (default: 0.35)")
    args = parser.parse_args()

    run_surveillance(args.source, max_frames=args.max_frames, conf=args.conf)
