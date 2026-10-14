"""
Smart CCTV Surveillance System - Step 2: Unified Dataset Preparation
====================================================================
Merges raw datasets into a standardized, ready-to-train YOLOv8 multi-threat dataset
under datasets/processed/:

Class Mapping:
  0: weapon       (from weapons dataset)
  1: fire         (from fire_smoke class 0)
  2: smoke        (from fire_smoke class 1)
  3: fall         (from falls dataset)

Also creates video_manifest.csv for video action recognition (80% train / 20% val).
"""

import os
import sys
import shutil
import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
PROC_DIR = BASE_DIR / "datasets" / "processed"

TRAIN_IMG = PROC_DIR / "images" / "train"
VAL_IMG = PROC_DIR / "images" / "val"
TRAIN_LBL = PROC_DIR / "labels" / "train"
VAL_LBL = PROC_DIR / "labels" / "val"

for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)


def process_weapons():
    """Processes Weapons dataset -> Unified Class 0 (weapon)."""
    print("[1/4] Processing Weapons Dataset -> Class 0: 'weapon'...")
    weapons_dir = RAW_DIR / "weapons"
    counts = {"train": 0, "val": 0}

    splits = [
        ("train", weapons_dir / "train" / "images", weapons_dir / "train" / "labels", TRAIN_IMG, TRAIN_LBL),
        ("val", weapons_dir / "valid" / "images", weapons_dir / "valid" / "labels", VAL_IMG, VAL_LBL),
    ]

    for split_name, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir in splits:
        if not src_img_dir.exists():
            continue
        for lbl_path in src_lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            # Find matching image
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = src_img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if not img_path:
                continue

            # Read lines and map class ID to 0
            new_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = "0"  # weapon
                        new_lines.append(" ".join(parts))

            if new_lines:
                # Target names with prefix to avoid collision
                tgt_stem = f"weapon_{stem}"
                # Copy image
                shutil.copy2(img_path, dst_img_dir / f"{tgt_stem}{img_path.suffix}")
                # Write updated label
                with open(dst_lbl_dir / f"{tgt_stem}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                counts[split_name] += 1

    print(f"      Weapons completed: {counts['train']} train, {counts['val']} val samples.")
    return counts


def process_fire_smoke():
    """Processes Fire & Smoke -> Class 1 (fire), Class 2 (smoke)."""
    print("[2/4] Processing Fire & Smoke Dataset -> Class 1: 'fire', Class 2: 'smoke'...")
    fire_dir = RAW_DIR / "fire_smoke"
    counts = {"train": 0, "val": 0}

    splits = [
        ("train", fire_dir / "train" / "images", fire_dir / "train" / "labels", TRAIN_IMG, TRAIN_LBL),
        ("val", fire_dir / "val" / "images", fire_dir / "val" / "labels", VAL_IMG, VAL_LBL),
    ]

    for split_name, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir in splits:
        if not src_img_dir.exists():
            continue
        for lbl_path in src_lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = src_img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if not img_path:
                continue

            new_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        old_cls = parts[0]
                        # In raw: 0 = Fire -> map to 1; 1 = Smoke -> map to 2
                        if old_cls == "0":
                            new_cls = "1"
                        elif old_cls == "1":
                            new_cls = "2"
                        else:
                            continue  # Skip unknown class IDs
                        parts[0] = new_cls
                        new_lines.append(" ".join(parts))

            if new_lines:
                tgt_stem = f"firesmoke_{stem}"
                shutil.copy2(img_path, dst_img_dir / f"{tgt_stem}{img_path.suffix}")
                with open(dst_lbl_dir / f"{tgt_stem}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                counts[split_name] += 1

    print(f"      Fire & Smoke completed: {counts['train']} train, {counts['val']} val samples.")
    return counts


def process_falls():
    """Processes Person Fallen Dataset -> Class 3: 'fall'."""
    print("[3/4] Processing Person Fallen Dataset -> Class 3: 'fall'...")
    falls_dir = RAW_DIR / "falls" / "fall_dataset"
    counts = {"train": 0, "val": 0}

    splits = [
        ("train", falls_dir / "images" / "train", falls_dir / "labels" / "train", TRAIN_IMG, TRAIN_LBL),
        ("val", falls_dir / "images" / "val", falls_dir / "labels" / "val", VAL_IMG, VAL_LBL),
    ]

    for split_name, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir in splits:
        if not src_lbl_dir.exists():
            continue
        for lbl_path in src_lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = src_img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if not img_path:
                continue

            new_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = "3"  # fall
                        new_lines.append(" ".join(parts))

            if new_lines:
                tgt_stem = f"fall_{stem}"
                shutil.copy2(img_path, dst_img_dir / f"{tgt_stem}{img_path.suffix}")
                with open(dst_lbl_dir / f"{tgt_stem}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                counts[split_name] += 1

    print(f"      Falls completed: {counts['train']} train, {counts['val']} val samples.")
    return counts


def create_data_yaml():
    """Creates standard data.yaml descriptor for YOLOv8 model training."""
    yaml_content = f"""# Smart CCTV Surveillance System - Multi-Threat Dataset Config
# Auto-generated by Step 2 Data Preparation

path: .
train: images/train
val: images/val

# 4 Unified Spatial Classes
nc: 4
names:
  0: weapon
  1: fire
  2: smoke
  3: fall
"""
    yaml_path = PROC_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[YOLO CONFIG] Created: {yaml_path}")
    return yaml_path


def process_video_manifest():
    """Creates a train/val split manifest for video action threats (theft, intrusion, violence)."""
    print("[4/4] Generating Video Action Manifest for Theft/Intrusion & Violence (80% Train / 20% Val)...")
    manifest_path = PROC_DIR / "video_manifest.csv"
    random.seed(42)

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
    rows = []
    # Collect theft/intrusion
    theft_dir = RAW_DIR / "theft_intrusion"
    if theft_dir.exists():
        for sub in sorted(theft_dir.iterdir()):
            if sub.is_dir():
                vids = sorted([f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS])
                random.shuffle(vids)
                n_train = int(len(vids) * 0.8)
                for i, v in enumerate(vids):
                    split = "train" if i < n_train else "val"
                    rows.append({
                        "filename": v.name,
                        "relative_path": v.relative_to(BASE_DIR).as_posix(),
                        "category": sub.name,
                        "domain": "theft_intrusion",
                        "label": "theft_or_intrusion",
                        "split": split
                    })

    # Collect violence
    violence_dir = RAW_DIR / "violence"
    if violence_dir.exists():
        for sub in sorted(violence_dir.iterdir()):
            if sub.is_dir():
                vids = sorted([f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS])
                random.shuffle(vids)
                n_train = int(len(vids) * 0.8)
                for i, v in enumerate(vids):
                    split = "train" if i < n_train else "val"
                    label = "normal" if "normal" in sub.name else "violence"
                    rows.append({
                        "filename": v.name,
                        "relative_path": v.relative_to(BASE_DIR).as_posix(),
                        "category": sub.name,
                        "domain": "violence",
                        "label": label,
                        "split": split
                    })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "relative_path", "category", "domain", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    train_vids = len([r for r in rows if r["split"] == "train"])
    val_vids = len([r for r in rows if r["split"] == "val"])
    print(f"      Video manifest created: {len(rows)} total videos ({train_vids} train, {val_vids} val).")
    print(f"      Saved to: {manifest_path}")


def main():
    print("=" * 70)
    print(" STEP 2: DATA PREPARATION & SPLITTING (MULTI-THREAT CCTV SURVEILLANCE)")
    print("=" * 70)

    w_counts = process_weapons()
    fs_counts = process_fire_smoke()
    f_counts = process_falls()

    total_train = w_counts["train"] + fs_counts["train"] + f_counts["train"]
    total_val = w_counts["val"] + fs_counts["val"] + f_counts["val"]

    yaml_path = create_data_yaml()
    process_video_manifest()

    print("\n" + "=" * 70)
    print(" STEP 2 PREPARATION COMPLETE! SUMMARY OF PREPARED DATASET:")
    print("=" * 70)
    print(f" Training Images & Labels:   {total_train:,} samples in {TRAIN_IMG}")
    print(f" Validation Images & Labels: {total_val:,} samples in {VAL_IMG}")
    print(f" Total Unified Samples:      {total_train + total_val:,} samples")
    print(f" YOLO Configuration:         {yaml_path}")
    print(f" Video Action Manifest:      {PROC_DIR / 'video_manifest.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
