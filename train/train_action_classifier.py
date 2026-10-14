"""
Smart CCTV Surveillance System - Stage 3: Video Action Classifier Training
==========================================================================
Trains a temporal action recognition model for:
  Class 0: normal             (everyday surveillance, walking, queuing)
  Class 1: violence           (fights, street brawls, physical assault)
  Class 2: theft_or_intrusion (stealing, shoplifting, burglary, robbery)

Architecture:
  Pre-trained MobileNetV2 (Spatial Feature Extractor) + Temporal GRU + Classifier.
  Natively and universally runs on any GPU or CPU.
"""

import os
import sys
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "datasets" / "processed" / "video_manifest.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODELS_DIR / "action_classifier.pt"

LABEL_MAP = {
    "normal": 0,
    "violence": 1,
    "theft_or_intrusion": 2
}
CLASS_NAMES = ["normal", "violence", "theft_or_intrusion"]


class VideoDataset(Dataset):
    """Loads video clips, sampling a fixed sequence of frames."""
    def __init__(self, manifest_csv: Path, split: str = "train", num_frames: int = 16):
        self.num_frames = num_frames
        self.samples = []

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    full_path = BASE_DIR / row["relative_path"]
                    if full_path.exists():
                        self.samples.append((str(full_path), LABEL_MAP[row["label"]]))

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((self.num_frames, 3, 224, 224), dtype=np.float32)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = self.num_frames

            # Uniform indices
            indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)
            frames = []
            last_valid = np.zeros((3, 224, 224), dtype=np.float32)

            for target_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    frames.append(last_valid)
                    continue

                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Normalize to [0, 1] then ImageNet standard
                frame = frame.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                frame = (frame - mean) / std
                # (H, W, C) -> (C, H, W)
                frame = np.transpose(frame, (2, 0, 1))
                last_valid = frame
                frames.append(frame)

            return np.array(frames, dtype=np.float32)
        finally:
            cap.release()

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            frames = self.extract_frames(path)
        except Exception:
            frames = np.zeros((self.num_frames, 3, 224, 224), dtype=np.float32)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class TemporalActionModel(nn.Module):
    """Pre-trained MobileNetV2 + Temporal GRU Action Recognition Model."""
    def __init__(self, num_classes: int = 3, hidden_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze early backbone layers to speed up training
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        self.gru = nn.GRU(input_size=1280, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)  # (B*T, 1280)
        feat = feat.reshape(B, T, 1280)

        out, _ = self.gru(feat)            # (B, T, 2*hidden_dim)
        last_out = out.mean(dim=1)         # Temporal average pooling (correct for BiGRU)
        logits = self.classifier(last_out) # (B, num_classes)
        return logits


def train_action_model(epochs: int = 5, batch_size: int = 4, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(" STAGE 3: VIDEO ACTION CLASSIFIER TRAINING (VIOLENCE & THEFT)")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Manifest not found at {MANIFEST_PATH}. Run 'python preprocessing/prepare_dataset.py' first.")

    train_ds = VideoDataset(MANIFEST_PATH, split="train", num_frames=16)
    val_ds = VideoDataset(MANIFEST_PATH, split="val", num_frames=16)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"[ERROR] Video dataset is empty (train: {len(train_ds)}, val: {len(val_ds)}). Please verify raw video files exist.")

    print(f" Loaded {len(train_ds)} training videos and {len(val_ds)} validation videos.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TemporalActionModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep frozen BatchNorm layers in eval mode to prevent stats drift
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        total_loss, correct, total = 0.0, 0, 0

        for step, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if step % 10 == 0 or step == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)

        train_acc = (correct / total) * 100 if total > 0 else 0

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
        print(f"  -> Epoch {epoch} Results | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%\n", flush=True)

        if val_acc > best_val_acc and val_acc > 0:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "classes": CLASS_NAMES
            }, MODEL_SAVE_PATH)

    print("=" * 70)
    print(f" TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc:.1f}%")
    if MODEL_SAVE_PATH.exists():
        print(f" Saved Model: {MODEL_SAVE_PATH} ({MODEL_SAVE_PATH.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f" [WARNING] No model saved (validation accuracy was 0%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Video Action Classifier for Violence & Theft")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    args = parser.parse_args()

    train_action_model(epochs=args.epochs, batch_size=args.batch_size)
