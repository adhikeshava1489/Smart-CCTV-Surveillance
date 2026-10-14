"""
Smart CCTV Surveillance System - Stage 3: Model Training (Fine-Tuning)
======================================================================
Fine-tunes an existing pre-trained foundation model (YOLOv8n) on our
unified 4-threat CCTV surveillance dataset (Weapon, Fire, Smoke, Fall).

Universal Hardware Execution:
Automatically and natively runs on any computer (NVIDIA GPU, Apple Silicon,
or CPU) without any manual hardware configuration.
"""

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "datasets" / "processed" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_BEST_MODEL = MODELS_DIR / "best.pt"


def train(epochs: int = 15, imgsz: int = 640):
    """Fine-tunes the pre-trained YOLOv8 model on our surveillance dataset."""
    if not DATA_YAML.exists():
        print(f"[ERROR] Dataset configuration not found at {DATA_YAML}")
        print("Please run preprocessing/prepare_dataset.py first.")
        sys.exit(1)

    # 1. Load Pre-Trained Foundation Model
    print("\nLoading Pre-Trained Foundation Model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    # 2. Universal Training (Automatically adapts to any GPU, MPS, or CPU)
    print("Starting Fine-Tuning on Surveillance Threat Dataset...")
    print(f"Data: {DATA_YAML} | Epochs: {epochs} | Input Size: {imgsz}x{imgsz}\n")

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        project=str(BASE_DIR / "runs" / "train"),
        name="cctv_threat_detector",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        patience=5,
        save=True
    )

    # 3. Copy best weights to models/best.pt
    best_path = getattr(getattr(model, "trainer", None), "best", None)
    run_best = Path(best_path) if best_path and Path(best_path).exists() else BASE_DIR / "runs" / "train" / "cctv_threat_detector" / "weights" / "best.pt"
    if run_best.exists():
        shutil.copy2(run_best, FINAL_BEST_MODEL)
        print("\n" + "=" * 65)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print(f" Final Model Saved: {FINAL_BEST_MODEL}")
        print("=" * 65 + "\n")
    else:
        print("\n[WARNING] Training finished but best.pt was not found at expected location.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on CCTV Surveillance Dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    train(epochs=args.epochs, imgsz=args.imgsz)
