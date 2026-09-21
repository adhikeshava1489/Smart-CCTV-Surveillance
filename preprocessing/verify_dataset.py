import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"

print("=" * 80)
print(f"{'Category':<20} | {'Files':<10} | {'Size':<14} | Sub-breakdown")
print("=" * 80)

total_files = 0
total_size = 0

for cat in sorted(RAW_DIR.iterdir()):
    if cat.is_dir():
        files = [f for f in cat.rglob('*.*') if f.is_file()]
        cnt = len(files)
        sz = sum(f.stat().st_size for f in files) / (1024 * 1024)
        total_files += cnt
        total_size += sz

        subs = [d for d in cat.iterdir() if d.is_dir()]
        sub_str = ', '.join([f"{d.name}: {len([f for f in d.rglob('*.*') if f.is_file()])}" for d in sorted(subs)])
        print(f"{cat.name:<20} | {cnt:<10} | {sz:<10.2f} MB | {sub_str}")

print("=" * 80)
print(f"{'TOTAL':<20} | {total_files:<10} | {total_size / 1024:<10.2f} GB | Across 5 Security Threat Domains")
print("=" * 80)
