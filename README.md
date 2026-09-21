# Smart CCTV Surveillance System

A Deep Learning-powered Intelligent Surveillance System for real-time detection of 7 critical security threats in CCTV camera streams:
1. **Tactical Weapons** (Knife, Pistol, Rifle, Grenade)
2. **Fire Detection** (Active flames & fires)
3. **Smoke Detection** (Indoor & outdoor smoke plumes)
4. **Person Fallen** (Human collapse & slip-and-fall detection)
5. **Theft** (Stealing & retail shoplifting)
6. **Intrusion** (Property breach & forced entry)
7. **Violence** (Brawls, physical assault & altercations)

---

## Complete Dataset Documentation

All datasets used in this project are **100% public, open-access**, requiring zero logins, zero accounts, and zero API keys.

For detailed dataset architecture, category breakdowns, source links, and installation instructions, please refer to:
👉 **[DATASET.md](DATASET.md)**

---

## Dataset Access Options

### Option 1: Google Drive Mirror (1-Click Download)
Download the entire pre-packaged dataset archive or individual category zips directly:
- 🔗 **Google Drive Folder**: `[Google Drive Link: https://drive.google.com/drive/folders/YOUR_DATASET_FOLDER_ID?usp=sharing]`
- Command line download:
  ```bash
  pip install gdown
  gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O datasets/datasets_raw.zip
  ```

### Option 2: 1-Click Automated Python Installer (From Public Mirrors)
To download/sync all datasets directly from Hugging Face public mirrors:
```bash
python scripts/install_datasets.py
```

### Option 3: Verify Existing Data
To verify the integrity and file counts of the 11,684 surveillance files (4.04 GB):
```bash
python scripts/verify_dataset.py
```

