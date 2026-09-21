"""
Smart CCTV Surveillance System - Master Dataset Installer & Verifier
====================================================================
100% Public, Open-Access Dataset Pipeline (Zero Login / Zero API Keys)

Downloads, extracts, and validates real-world CCTV surveillance data across
all 5 threat domains (7 Security Threats):
  1. Tactical Weapons (Knife, Pistol, Rifle, Grenade)
  2. Fire & Smoke
  3. Person Fallen (Collapse / Fall Detection)
  4. Theft & Intrusion (Burglary, Robbery, Shoplifting, Stealing)
  5. Violence (Assault, Fights, GodsEye Altercations, Normal Non-Fight)
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "raw"

# Category directories
DIR_WEAPONS = DATASETS_DIR / "weapons"
DIR_FIRE_SMOKE = DATASETS_DIR / "fire_smoke"
DIR_FALLS = DATASETS_DIR / "falls"
DIR_THEFT = DATASETS_DIR / "theft_intrusion"
DIR_VIOLENCE = DATASETS_DIR / "violence"


def check_and_report_existing():
    """Scans and reports existing datasets."""
    print("=" * 75)
    print(" Smart CCTV Surveillance - Dataset Status Verification")
    print("=" * 75)

    categories = [
        ("Weapons", DIR_WEAPONS, 4000),
        ("Fire & Smoke", DIR_FIRE_SMOKE, 1000),
        ("Person Fallen", DIR_FALLS, 400),
        ("Theft & Intrusion", DIR_THEFT, 50),
        ("Violence & Baseline", DIR_VIOLENCE, 50),
    ]

    all_ready = True
    for name, path, min_expected in categories:
        if path.exists():
            files = [f for f in path.rglob("*.*") if f.is_file()]
            size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            status = "READY" if len(files) >= min_expected else "PARTIAL"
            if len(files) < min_expected:
                all_ready = False
            print(f"  [{status}] {name:<22}: {len(files):>5} files ({size_mb:>7.2f} MB)")
        else:
            print(f"  [MISSING] {name:<22}: 0 files (Directory not found)")
            all_ready = False

    print("=" * 75)
    return all_ready


def download_stream(url: str, dest_path: Path, min_size: int = 10000) -> bool:
    """Streams a file from URL to disk safely with .part temporary file."""
    if dest_path.exists() and dest_path.stat().st_size >= min_size:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.parent / f"{dest_path.name}.part"
    try:
        with requests.get(url, stream=True, timeout=45) as r:
            if r.status_code == 200:
                with open(part_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if part_path.exists() and part_path.stat().st_size >= min_size:
                    part_path.replace(dest_path)
                    return True
            if part_path.exists():
                part_path.unlink()
            return False
    except Exception as e:
        if part_path.exists():
            part_path.unlink()
        return False


def install_weapons():
    """Installs weapons dataset from Hugging Face public zip archives."""
    print("\n[STEP 1/5] Checking Weapons Dataset (fcakyon/gun-object-detection)...")
    DIR_WEAPONS.mkdir(parents=True, exist_ok=True)
    existing = len(list(DIR_WEAPONS.rglob("*.*")))
    if existing >= 4000:
        print(f"  -> Weapons dataset already installed ({existing} files). Skipping.")
        return

    base_url = "https://huggingface.co/datasets/fcakyon/gun-object-detection/resolve/main/data"
    for split in ["train", "valid"]:
        zip_url = f"{base_url}/{split}.zip"
        dest_zip = DIR_WEAPONS / f"{split}.zip"
        print(f"  -> Downloading {split}.zip...")
        if download_stream(zip_url, dest_zip, min_size=5000000):
            print(f"  -> Extracting {split}.zip...")
            with zipfile.ZipFile(dest_zip, 'r') as zf:
                zf.extractall(DIR_WEAPONS / split)
            dest_zip.unlink()
    print("  -> Weapons installation complete.")


def install_fire_smoke(num_samples: int = 600):
    """Installs fire & smoke dataset from betasecond/jimei-fire-smoke-yolo-dataset."""
    print("\n[STEP 2/5] Checking Fire & Smoke Dataset (betasecond/jimei-fire-smoke)...")
    DIR_FIRE_SMOKE.mkdir(parents=True, exist_ok=True)
    existing = len(list(DIR_FIRE_SMOKE.rglob("*.*")))
    if existing >= 1000:
        print(f"  -> Fire & smoke dataset already installed ({existing} files). Skipping.")
        return

    print("  -> Querying manifest from Hugging Face...")
    api_url = "https://huggingface.co/api/datasets/betasecond/jimei-fire-smoke-yolo-dataset"
    base_url = "https://huggingface.co/datasets/betasecond/jimei-fire-smoke-yolo-dataset/resolve/main"

    try:
        r = requests.get(api_url, timeout=15).json()
        all_files = [s["rfilename"] for s in r.get("siblings", [])]
        train_imgs = [f for f in all_files if f.startswith("train/images/") and f.endswith((".jpg", ".png"))][:500]
        val_imgs = [f for f in all_files if f.startswith("valid/images/") and f.endswith((".jpg", ".png"))][:100]

        tasks = []
        for rel in train_imgs:
            stem = Path(rel).stem
            dest_img = DIR_FIRE_SMOKE / "train" / "images" / Path(rel).name
            dest_lbl = DIR_FIRE_SMOKE / "train" / "labels" / f"{stem}.txt"
            tasks.append((f"{base_url}/{rel}", dest_img))
            tasks.append((f"{base_url}/train/labels/{stem}.txt", dest_lbl))

        for rel in val_imgs:
            stem = Path(rel).stem
            dest_img = DIR_FIRE_SMOKE / "val" / "images" / Path(rel).name
            dest_lbl = DIR_FIRE_SMOKE / "val" / "labels" / f"{stem}.txt"
            tasks.append((f"{base_url}/{rel}", dest_img))
            tasks.append((f"{base_url}/valid/labels/{stem}.txt", dest_lbl))

        print(f"  -> Downloading {len(tasks)//2} image-label pairs with parallel workers...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(download_stream, url, dest, 10) for url, dest in tasks]
            for _ in as_completed(futures):
                pass
        print("  -> Fire & Smoke installation complete.")
    except Exception as e:
        print(f"  -> Error installing Fire & Smoke: {e}")


def install_falls():
    """Installs human fall detection dataset from DeZan/fall-detection."""
    print("\n[STEP 3/5] Checking Person Fallen Dataset (DeZan/fall-detection)...")
    DIR_FALLS.mkdir(parents=True, exist_ok=True)
    existing = len(list(DIR_FALLS.rglob("*.*")))
    if existing >= 800:
        print(f"  -> Falls dataset already installed ({existing} files). Skipping.")
        return

    print("  -> Querying manifest from Hugging Face...")
    api_url = "https://huggingface.co/api/datasets/DeZan/fall-detection"
    base_url = "https://huggingface.co/datasets/DeZan/fall-detection/resolve/main"

    try:
        r = requests.get(api_url, timeout=15).json()
        all_files = [s["rfilename"] for s in r.get("siblings", []) if s["rfilename"].startswith("fall_dataset/")]
        tasks = []
        for rel in all_files:
            dest = DIR_FALLS / rel
            tasks.append((f"{base_url}/{rel}", dest))

        print(f"  -> Downloading {len(tasks)} fall image/label files...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(download_stream, url, dest, 10) for url, dest in tasks]
            for _ in as_completed(futures):
                pass
        print("  -> Person Fallen installation complete.")
    except Exception as e:
        print(f"  -> Error installing Falls: {e}")


def install_theft_and_violence():
    """Delegates to download_expanded_surveillance.py for Theft & Violence videos."""
    print("\n[STEP 4 & 5] Checking Theft/Intrusion and Violence Datasets...")
    theft_count = len(list(DIR_THEFT.rglob("*.mp4")))
    violence_count = len(list(DIR_VIOLENCE.rglob("*.*")))

    if theft_count >= 80 and violence_count >= 100:
        print(f"  -> Theft & Intrusion already has {theft_count} videos. Skipping.")
        print(f"  -> Violence & Baseline already has {violence_count} videos/clips. Skipping.")
        return

    print("  -> Executing preprocessing/download_expanded_surveillance.py...")
    import subprocess
    subprocess.run([sys.executable, str(BASE_DIR / "preprocessing" / "download_expanded_surveillance.py")], check=True)


def main():
    print("\n===========================================================================")
    print(" SMART CCTV SURVEILLANCE - AUTOMATED DATASET INSTALLER")
    print("===========================================================================\n")

    install_weapons()
    install_fire_smoke()
    install_falls()
    install_theft_and_violence()

    print("\n")
    check_and_report_existing()
    print("\nDataset ready! You can verify integrity anytime by running:")
    print("  python scripts/verify_dataset.py\n")


if __name__ == "__main__":
    main()
