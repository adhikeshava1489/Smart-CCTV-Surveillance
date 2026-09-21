"""
Smart CCTV Surveillance System - Extended Public Surveillance Dataset Downloader
Expands Theft/Intrusion and Violence datasets using 100% public, open-access
Hugging Face repositories with zero login or API token requirements.
"""

import os
import sys
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"

# Target directories
DIR_BURGLARY = RAW_DIR / "theft_intrusion" / "burglary"
DIR_SHOPLIFTING = RAW_DIR / "theft_intrusion" / "shoplifting"
DIR_STEALING = RAW_DIR / "theft_intrusion" / "stealing"
DIR_ROBBERY = RAW_DIR / "theft_intrusion" / "robbery"

DIR_FIGHTS = RAW_DIR / "violence" / "fights"
DIR_ASSAULT = RAW_DIR / "violence" / "assault"
DIR_GODSEYE_FIGHTS = RAW_DIR / "violence" / "godseye_fights"
DIR_NORMAL = RAW_DIR / "violence" / "normal_nonfight"

for d in [DIR_BURGLARY, DIR_SHOPLIFTING, DIR_STEALING, DIR_ROBBERY, DIR_FIGHTS, DIR_ASSAULT, DIR_GODSEYE_FIGHTS, DIR_NORMAL]:
    d.mkdir(parents=True, exist_ok=True)


def cleanup_root_duplicates():
    """Removes duplicate files that accidentally ended up in the category root directories."""
    cleaned = 0
    for root_dir in [RAW_DIR / "theft_intrusion", RAW_DIR / "violence"]:
        for item in root_dir.glob("*.mp4"):
            if item.is_file():
                # Check if it exists in any subfolder
                for sub in root_dir.iterdir():
                    if sub.is_dir() and (sub / item.name).exists():
                        item.unlink()
                        cleaned += 1
                        break
        for item in root_dir.glob("*.avi"):
            if item.is_file():
                for sub in root_dir.iterdir():
                    if sub.is_dir() and (sub / item.name).exists():
                        item.unlink()
                        cleaned += 1
                        break
    if cleaned:
        print(f"[INFO] Cleaned up {cleaned} duplicate root files to keep directories organized.")


def build_download_tasks():
    tasks = []

    # 1. Fetch UCF-Crime manifest
    print("[INFO] Fetching UCF-Crime anomaly dataset manifest from Hugging Face...")
    url_ucf = "https://huggingface.co/api/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos"
    base_ucf = "https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos/resolve/main"

    try:
        r = requests.get(url_ucf, timeout=20).json()
        ucf_files = [s["rfilename"] for s in r.get("siblings", []) if s["rfilename"].endswith(".mp4")]

        targets = {
            "Burglary": (DIR_BURGLARY, 20),
            "Shoplifting": (DIR_SHOPLIFTING, 20),
            "Stealing": (DIR_STEALING, 20),
            "Robbery": (DIR_ROBBERY, 20),
            "Fighting": (DIR_FIGHTS, 20),
            "Assault": (DIR_ASSAULT, 20)
        }

        for cat_name, (dest_dir, target_count) in targets.items():
            existing_files = {f.name for f in dest_dir.glob("*.*") if f.is_file()}
            needed = target_count - len(existing_files)
            if needed <= 0:
                print(f"  [OK] {cat_name}: Already has {len(existing_files)} files (target {target_count})")
                continue

            candidates = [f for f in ucf_files if f"/{cat_name}/" in f]
            added = 0
            for rel in candidates:
                fname = Path(rel).name
                if fname not in existing_files:
                    tasks.append((fname, f"{base_ucf}/{rel}", dest_dir, cat_name))
                    added += 1
                    if added >= needed:
                        break
            print(f"  [QUEUE] {cat_name}: Queued {added} new videos (target {target_count})")
    except Exception as e:
        print(f"[ERROR] Failed to query UCF-Crime API: {e}")

    # 2. Fetch GodsEye Violence manifest
    print("[INFO] Fetching GodsEye violence dataset manifest from Hugging Face...")
    url_gods = "https://huggingface.co/api/datasets/valiantlynxz/godseye-violence-detection-dataset"
    base_gods = "https://huggingface.co/datasets/valiantlynxz/godseye-violence-detection-dataset/resolve/main"

    try:
        r2 = requests.get(url_gods, timeout=20).json()
        gods_files = [s["rfilename"] for s in r2.get("siblings", []) if s["rfilename"].endswith(".avi") and not s["rfilename"].endswith("_processed.avi")]

        # GodsEye Fights
        existing_gf = {f.name for f in DIR_GODSEYE_FIGHTS.glob("*.*") if f.is_file()}
        needed_gf = 30 - len(existing_gf)
        if needed_gf > 0:
            fight_cands = [f for f in gods_files if "Fight/" in f]
            added = 0
            for rel in fight_cands:
                fname = Path(rel).name
                if fname not in existing_gf:
                    tasks.append((fname, f"{base_gods}/{rel}", DIR_GODSEYE_FIGHTS, "GodsEye-Fight"))
                    added += 1
                    if added >= needed_gf:
                        break
            print(f"  [QUEUE] GodsEye Fights: Queued {added} new clips (target 30)")

        # GodsEye Non-Fight (Normal surveillance)
        existing_nf = {f.name for f in DIR_NORMAL.glob("*.*") if f.is_file()}
        needed_nf = 30 - len(existing_nf)
        if needed_nf > 0:
            nonfight_cands = [f for f in gods_files if "NonFight/" in f]
            added = 0
            for rel in nonfight_cands:
                fname = Path(rel).name
                if fname not in existing_nf:
                    tasks.append((fname, f"{base_gods}/{rel}", DIR_NORMAL, "Normal-CCTV"))
                    added += 1
                    if added >= needed_nf:
                        break
            print(f"  [QUEUE] Normal Non-Fight: Queued {added} new clips (target 30)")

    except Exception as e:
        print(f"[ERROR] Failed to query GodsEye API: {e}")

    return tasks


def download_single_video(task):
    filename, url, dest_dir, cat_name = task
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 100000:
        return True, filename, cat_name, dest.stat().st_size / (1024 * 1024), "Already Exists"

    tmp_dest = dest_dir / f"{filename}.part"
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code == 200:
                with open(tmp_dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if tmp_dest.exists() and tmp_dest.stat().st_size > 50000:
                    tmp_dest.replace(dest)
                    size_mb = dest.stat().st_size / (1024 * 1024)
                    return True, filename, cat_name, size_mb, "Downloaded"
                else:
                    if tmp_dest.exists():
                        tmp_dest.unlink()
                    return False, filename, cat_name, 0, "File too small / incomplete"
            else:
                return False, filename, cat_name, 0, f"HTTP {resp.status_code}"
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False, filename, cat_name, 0, str(e)


def main():
    print("=" * 75)
    print(" Smart CCTV Surveillance - Dataset Expansion (Theft/Intrusion & Violence)")
    print(" 100% Public Data, Zero Authentication Required")
    print("=" * 75)

    cleanup_root_duplicates()
    tasks = build_download_tasks()
    print(f"\n[INFO] Total download tasks scheduled: {len(tasks)}")

    if not tasks:
        print("[INFO] All target download counts already fulfilled!")
    else:
        completed = 0
        failed = 0
        total = len(tasks)
        print(f"[INFO] Launching multi-threaded downloader with 6 worker threads...\n")

        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_task = {executor.submit(download_single_video, t): t for t in tasks}
            for future in as_completed(future_to_task):
                success, fname, cat, size_mb, status = future.result()
                if success:
                    completed += 1
                    print(f"  [{completed + failed}/{total}] [OK] [{cat}] {fname:30} ({size_mb:.2f} MB) -> {status}", flush=True)
                else:
                    failed += 1
                    print(f"  [{completed + failed}/{total}] [FAIL] [{cat}] {fname:30} -> {status}", flush=True)

    # Print overall raw dataset inventory
    print("\n" + "=" * 75)
    print(" DATASET INVENTORY - COMPLETE VERIFICATION")
    print("=" * 75)

    total_files = 0
    total_bytes = 0

    for cat_dir in sorted(RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue

        all_files = [f for f in cat_dir.rglob("*.*") if f.is_file()]
        cat_size = sum(f.stat().st_size for f in all_files)
        total_files += len(all_files)
        total_bytes += cat_size

        subdirs = [d for d in cat_dir.iterdir() if d.is_dir()]
        print(f"\n[CATEGORY] datasets/raw/{cat_dir.name}/")
        print(f"   * Total Files in Category: {len(all_files):,} files")
        print(f"   * Total Size: {cat_size / (1024*1024):.2f} MB ({cat_size / (1024*1024*1024):.2f} GB)")

        if subdirs:
            for s in sorted(subdirs):
                s_files = [f for f in s.rglob("*.*") if f.is_file()]
                s_mb = sum(f.stat().st_size for f in s_files) / (1024 * 1024)
                print(f"     - Subfolder '{s.name}': {len(s_files)} files ({s_mb:.2f} MB)")

    print("\n" + "=" * 75)
    print(f"GLOBAL SUMMARY: {total_files:,} files | {total_bytes / (1024*1024*1024):.2f} GB across datasets/raw/")
    print("=" * 75)


if __name__ == "__main__":
    main()
