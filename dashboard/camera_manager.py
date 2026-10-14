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
        self.last_threat_times = {}
        self.fps = 25.0
        self.frame_count = 0
        self.start_time = None

    def start(self):
        if self.is_running:
            return True
        # Open video capture
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[CAM-{self.camera_id}] Error opening source: {self.source}")
            return False

        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        if fps_val is None or np.isnan(fps_val) or fps_val <= 0:
            self.fps = 25.0
        else:
            self.fps = float(fps_val)

        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[CAM-{self.camera_id}] Started feed '{self.name}' in {self.mode.upper()} mode.")
        return True

    def _capture_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "recorded":
                    # Loop video if recorded for continuous demonstration
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break

            self.frame_count += 1
            # Run 7-threat detection pipeline
            processed_frame, active_threats = self.detector.process_frame(frame)

            current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            video_sec = self.frame_count / self.fps if self.fps > 0 else 0
            video_timestamp = f"{int(video_sec // 60):02d}:{int(video_sec % 60):02d}"

            # Log incident per-threat with 2.5-second debounce window
            if active_threats:
                with self.lock:
                    self.active_threats = set(active_threats)
                for threat in active_threats:
                    last_time = self.last_threat_times.get(threat, 0)
                    if time.time() - last_time > 2.5:
                        self.last_threat_times[threat] = time.time()
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
                        with self.lock:
                            self.incidents.append(incident_entry)
                            if len(self.incidents) > 100:
                                self.incidents.pop(0)
            else:
                with self.lock:
                    self.active_threats = set()

            # Watermark mode on frame (LIVE vs RECORDED)
            mode_color = (0, 0, 255) if self.mode == "live" else (255, 165, 0)
            mode_text = f"[{self.mode.upper()} - {self.name}]"
            cv2.putText(processed_frame, mode_text, (max(10, processed_frame.shape[1] - 260), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

            with self.lock:
                self.latest_frame = processed_frame

            # Pacing sleep only for recorded video to prevent buffering lag on live cameras
            if self.mode == "recorded":
                time.sleep(1.0 / (self.fps * 1.15))

        self.stop()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpeg.tobytes() if ret else None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive() and threading.current_thread() != self.thread:
            self.thread.join(timeout=2.0)
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            self.latest_frame = None


class MultiCameraManager:
    """Manages a thread-safe registry of multiple camera feeds."""
    def __init__(self):
        self.cameras = {}
        self.lock = threading.Lock()

    def add_camera(self, camera_id: str, name: str, source, mode: str = "live"):
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].stop()
        feed = CameraFeed(camera_id, name, source, mode)
        success = feed.start()
        if success:
            with self.lock:
                self.cameras[camera_id] = feed
            return feed
        return None

    def remove_camera(self, camera_id: str):
        with self.lock:
            feed = self.cameras.pop(camera_id, None)
        if feed:
            feed.stop()

    def get_all_incidents(self):
        with self.lock:
            cams = list(self.cameras.values())
        all_incidents = []
        for cam in cams:
            with cam.lock:
                all_incidents.extend(list(cam.incidents))
        all_incidents.sort(key=lambda x: x["unix_time"], reverse=True)
        return all_incidents

    def get_active_live_threats(self):
        """Returns active threats specifically from LIVE feeds (for siren/audio alerts)."""
        live_threats = set()
        with self.lock:
            cams = list(self.cameras.values())
        for cam in cams:
            if cam.mode == "live":
                with cam.lock:
                    if cam.active_threats:
                        live_threats.update(cam.active_threats)
        return list(live_threats)
