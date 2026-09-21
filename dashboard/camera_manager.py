"""
Smart CCTV Surveillance System - Multi-Camera & Feed Manager
============================================================
Supports multiple simultaneous camera streams (Webcams, IP cameras, uploaded videos).

Operating Modes:
1. LIVE MODE (Webcam / IP Camera):
   - Full real-time audio/visual alerts + flashing HUD banners.
   - Timestamped incident recording.
2. RECORDED MODE (Uploaded Video File):
   - Silent mode (no noisy siren).
   - Generates complete timeline event report with video timestamps.
"""

import time
import threading
from pathlib import Path
from collections import deque
import cv2
import numpy as np

# Import unified detection pipeline
from predict.predict_surveillance import UnifiedSurveillanceDetector

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "dashboard" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class CameraFeed:
    """Manages an individual camera feed with asynchronous processing."""
    def __init__(self, camera_id: str, name: str, source, mode: str = "live"):
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.mode = mode  # "live" or "recorded"

        self.cap = None
        self.is_running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

        self.detector = UnifiedSurveillanceDetector()
        self.incidents = []
        self.active_threats = set()
        self.fps = 25.0
        self.frame_count = 0
        self.start_time = None

    def start(self):
        if self.is_running:
            return
        # Open video capture
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[CAM-{self.camera_id}] Error opening source: {self.source}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[CAM-{self.camera_id}] Started feed '{self.name}' in {self.mode.upper()} mode.")
        return True

    def _capture_loop(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "recorded":
                    # Loop video if recorded for continuous demonstration
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self.frame_count += 1
            # Run 7-threat detection pipeline
            processed_frame, active_threats = self.detector.process_frame(frame)

            current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            video_timestamp = f"{int(self.frame_count / self.fps // 60):02d}:{int(self.frame_count / self.fps % 60):02d}"

            # Log incident if threats detected
            if active_threats:
                self.active_threats = active_threats
                for threat in active_threats:
                    # Log unique incident per 3-second window
                    if not self.incidents or (time.time() - self.incidents[-1]["unix_time"] > 2.5):
                        incident_entry = {
                            "camera_id": self.camera_id,
                            "camera_name": self.name,
                            "mode": self.mode,
                            "threat": threat.upper(),
                            "real_time": current_timestamp,
                            "video_time": video_timestamp,
                            "frame": self.frame_count,
                            "unix_time": time.time()
                        }
                        self.incidents.append(incident_entry)
                        # Keep last 100 incidents in memory
                        if len(self.incidents) > 100:
                            self.incidents.pop(0)
            else:
                self.active_threats = set()

            # Watermark mode on frame (LIVE vs RECORDED)
            mode_color = (0, 0, 255) if self.mode == "live" else (255, 165, 0)
            mode_text = f"[{self.mode.upper()} - {self.name}]"
            cv2.putText(processed_frame, mode_text, (processed_frame.shape[1] - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

            with self.lock:
                self.latest_frame = processed_frame

            # Slight sleep to match FPS playback on recorded video
            time.sleep(1.0 / (self.fps * 1.2))

        self.stop()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                return jpeg.tobytes() if ret else None
        return None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()


class MultiCameraManager:
    """Manages a registry of multiple camera feeds."""
    def __init__(self):
        self.cameras = {}

    def add_camera(self, camera_id: str, name: str, source, mode: str = "live"):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
        feed = CameraFeed(camera_id, name, source, mode)
        success = feed.start()
        if success:
            self.cameras[camera_id] = feed
            return feed
        return None

    def remove_camera(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]

    def get_all_incidents(self):
        all_incidents = []
        for cam in self.cameras.values():
            all_incidents.extend(cam.incidents)
        # Sort descending by timestamp
        all_incidents.sort(key=lambda x: x["unix_time"], reverse=True)
        return all_incidents

    def get_active_live_threats(self):
        """Returns active threats specifically from LIVE feeds (for siren/audio alerts)."""
        live_threats = set()
        for cam in self.cameras.values():
            if cam.mode == "live" and cam.active_threats:
                live_threats.update(cam.active_threats)
        return list(live_threats)
