"""
Smart CCTV Surveillance System - Stage 4: Prediction & Threat Detection
=======================================================================
Runs inference on test images, surveillance videos, or live camera streams
using our fine-tuned CCTV threat detection model.

Detects 4 Spatial Security Threats:
  0: weapon (Red Bounding Box)
  1: fire   (Orange Bounding Box)
  2: smoke  (Purple Bounding Box)
  3: fall   (Magenta Bounding Box)

Saves annotated output to predict/outputs/ and prints real-time alerts.
"""

import math
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "predict" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

VALID_THREATS = {"weapon", "fire", "smoke", "fall"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def load_model():
    """Loads fine-tuned best.pt or falls back to pre-trained model."""
    best_model = MODELS_DIR / "best.pt"
    if best_model.exists():
        print(f"[MODEL] Loading fine-tuned surveillance model: {best_model}")
        return YOLO(str(best_model))
    else:
        print("[MODEL] 'models/best.pt' not found yet. Using base 'yolov8n.pt'.")
        return YOLO("yolov8n.pt")


def predict_image(image_path: str, conf_thresh: float = 0.35):
    """Runs threat detection on a single image file."""
    model = load_model()
    src = Path(image_path)
    if not src.exists():
        print(f"[ERROR] Image not found: {src}")
        return

    print(f"\n[INFERENCE] Analyzing image: {src.name}...")
    results = model.predict(source=str(src), conf=conf_thresh, verbose=False)
    r = results[0]

    detected = []
    if len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = model.names.get(cls_id, f"Class_{cls_id}").lower()
            if name in VALID_THREATS:
                detected.append((name, conf))
                print(f"  [ALERT] [THREAT DETECTED] {name.upper()} - Confidence: {conf * 100:.1f}%")

    if not detected:
        print("  [OK] No security threats detected (Scene Clear).")

    # Save annotated result
    out_path = OUTPUTS_DIR / f"pred_{src.name}"
    annotated_frame = r.plot()
    cv2.imwrite(str(out_path), annotated_frame)
    print(f"[OUTPUT] Saved annotated image to: {out_path}\n")


def predict_video(video_path: str, conf_thresh: float = 0.35, max_frames: int = 300):
    """Runs threat detection on a surveillance video file or live stream."""
    model = load_model()
    is_cam = isinstance(video_path, int) or (isinstance(video_path, str) and str(video_path).isdigit())
    cap = cv2.VideoCapture(int(video_path) if is_cam else str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
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

    src_name = "live_camera" if is_cam else Path(str(video_path)).stem
    out_path = OUTPUTS_DIR / f"pred_{src_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"\n[INFERENCE] Processing video: {src_name} ({width}x{height} @ {fps:.1f} FPS)...")
    frame_idx = 0
    threat_summary = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        r = results[0]

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            name = model.names.get(cls_id, f"Class_{cls_id}").lower()
            if name in VALID_THREATS:
                threat_summary.add(name)

        annotated_frame = r.plot()
        if out.isOpened():
            out.write(annotated_frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  ... processed {frame_idx} frames")

    cap.release()
    if out.isOpened():
        out.release()

    print(f"\n[COMPLETE] Processed {frame_idx} frames.")
    if threat_summary:
        print(f"[ALERT] Threats Detected During Video: {', '.join(t.upper() for t in threat_summary)}")
    else:
        print("[OK] Video Clear: No threats detected.")
    print(f"[OUTPUT] Saved annotated video to: {out_path}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CCTV Threat Detection Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames for video analysis (default: 300)")
    args = parser.parse_args()

    src_path = Path(args.source)
    if src_path.suffix.lower() in IMAGE_EXTENSIONS:
        predict_image(args.source, conf_thresh=args.conf)
    else:
        predict_video(args.source, conf_thresh=args.conf, max_frames=args.max_frames)
