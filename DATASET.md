# Smart CCTV Surveillance System - Dataset Guide & Installation Manual

Comprehensive documentation for the **Smart CCTV Surveillance Dataset**, covering dataset architecture, public source provenance, category specifications, and step-by-step instructions for automated or manual installation.

---

## 1. Overview & Security Threat Coverage

The Smart CCTV Surveillance System is engineered to detect **7 critical real-time security threats** in live surveillance video streams:

| # | Threat Type | Threat Domain | Data Format | Detection Objective |
|---|---|---|---|---|
| 1 | **Tactical Weapons** | `weapons/` | Images + YOLO BBoxes | Detect drawn handguns, pistols, rifles, knives, and grenades |
| 2 | **Fire** | `fire_smoke/` | Images + YOLO BBoxes | Detect active flames and localized fire hazards |
| 3 | **Smoke** | `fire_smoke/` | Images + YOLO BBoxes | Detect indoor/outdoor smoke plumes before flame visibility |
| 4 | **Person Fallen** | `falls/` | Images + YOLO BBoxes | Detect human collapse, slip-and-fall, and medical emergencies |
| 5 | **Theft** | `theft_intrusion/` | CCTV Video (`.mp4`) | Detect stealing, pickpocketing, and retail shoplifting |
| 6 | **Intrusion** | `theft_intrusion/` | CCTV Video (`.mp4`) | Detect unauthorized perimeter breach, burglary, and forced entry |
| 7 | **Violence** | `violence/` | CCTV Video (`.mp4`, `.avi`) | Detect brawls, physical assault, fighting, and altercations |

---

## 2. Complete Dataset Inventory & Metrics

All raw datasets reside inside the `datasets/raw/` directory:

| Threat Domain | Subfolder Breakdown | File Count | Total Size | Primary Source |
|---|---|---|---|---|
| **Tactical Weapons** | • `train/` (7,523 files)<br>• `valid/` (1,811 files) | **9,334 files** | **91.25 MB** | Hugging Face (`fcakyon/gun-object-detection`) |
| **Fire & Smoke** | • `train/` (1,000 files)<br>• `val/` (200 files) | **1,200 files** | **31.58 MB** | Hugging Face (`betasecond/jimei-fire-smoke`) |
| **Person Fallen** | • `fall_dataset/` (970 files) | **970 files** | **50.14 MB** | Hugging Face (`DeZan/fall-detection`) |
| **Theft & Intrusion** | • `burglary/` (20 videos)<br>• `robbery/` (20 videos)<br>• `shoplifting/` (20 videos)<br>• `stealing/` (20 videos) | **80 videos** | **2,437.98 MB** (~2.38 GB) | Hugging Face (`ertiaM/Anomaly_Detection`) |
| **Violence & Baseline** | • `assault/` (20 videos)<br>• `fights/` (20 videos)<br>• `godseye_fights/` (30 clips)<br>• `normal_nonfight/` (30 clips) | **100 videos/clips** | **1,526.44 MB** (~1.49 GB) | Hugging Face (`ertiaM` & `godseye`) |
| **TOTAL DATASET** | **Across 5 Threat Domains** | **11,684 files** | **4.04 GB** | **100% Public, Zero Login Required** |

---

## 3. Directory Structure

```
datasets/raw/
│
├── weapons/                         <- 4,666 tactical weapon images & 4,668 YOLO annotations (179.30 MB)
│   ├── train/
│   │   ├── images/                  <- Training images (.jpg)
│   │   ├── labels/                  <- YOLO bounding boxes (.txt)
│   │   └── _annotations.coco.json   <- COCO format ground truth
│   └── valid/
│       ├── images/                  <- Validation images (.jpg)
│       ├── labels/                  <- YOLO bounding boxes (.txt)
│       └── _annotations.coco.json
│
├── fire_smoke/                      <- 600 flame & smoke images & 600 YOLO annotations (31.58 MB)
│   ├── train/
│   │   ├── images/                  <- 500 training images (.jpg)
│   │   └── labels/                  <- 500 YOLO label files (.txt)
│   └── val/
│       ├── images/                  <- 100 validation images (.jpg)
│       └── labels/                  <- 100 YOLO label files (.txt)
│
├── falls/                           <- 485 real fall images & 485 YOLO annotations (99.86 MB)
│   └── fall_dataset/
│       ├── images/                  <- 485 fall images (.jpg)
│       └── labels/                  <- 485 YOLO label files (.txt)
│
├── theft_intrusion/                 <- 80 real CCTV surveillance video recordings (2.38 GB)
│   ├── burglary/                    <- 20 property breach & forced entry videos (.mp4)
│   ├── robbery/                     <- 20 armed theft & hold-up videos (.mp4)
│   ├── shoplifting/                 <- 20 retail counter shoplifting videos (.mp4)
│   └── stealing/                    <- 20 stealth stealing & pickpocketing videos (.mp4)
│
└── violence/                        <- 100 real CCTV altercation & baseline clips (1.49 GB)
    ├── assault/                     <- 20 physical assault videos (.mp4)
    ├── fights/                      <- 20 street & indoor brawl videos (.mp4)
    ├── godseye_fights/              <- 30 real CCTV fight altercation clips (.avi)
    └── normal_nonfight/             <- 30 everyday normal surveillance clips (walking, queuing) (.avi)
```

---

## 4. Public Provenance & Open-Access Guarantee

Every single byte of data in this project is sourced from **100% public, open-access repositories**.

### Why No Kaggle or Roboflow Sign-ins are Required
1. **Zero Login Walls**: Standard Kaggle datasets require a private `kaggle.json` authentication token and block anonymous HTTP downloads. We do not use restricted Kaggle endpoints.
2. **Zero Paywalls / Tokens**: Roboflow private exports require workspace credentials and temporary download keys.
3. **High Availability**: Old academic university lab URLs frequently go offline. All data in this repository is hosted on **Hugging Face Hub open mirrors** with permanent HTTP 200 uptime.

### Source Repositories
1. **Weapons Dataset**:
   - **Repository**: [`fcakyon/gun-object-detection`](https://huggingface.co/datasets/fcakyon/gun-object-detection)
   - **Classes**: Knife, Pistol, Rifle, Grenade
   - **Direct Archive**: `https://huggingface.co/datasets/fcakyon/gun-object-detection/resolve/main/data/train.zip` and `valid.zip`
2. **Fire & Smoke Dataset**:
   - **Repository**: [`betasecond/jimei-fire-smoke-yolo-dataset`](https://huggingface.co/datasets/betasecond/jimei-fire-smoke-yolo-dataset)
   - **Classes**: `0: Fire`, `1: Smoke`
   - **Direct Resolve Base**: `https://huggingface.co/datasets/betasecond/jimei-fire-smoke-yolo-dataset/resolve/main/`
3. **Person Fallen Dataset**:
   - **Repository**: [`DeZan/fall-detection`](https://huggingface.co/datasets/DeZan/fall-detection)
   - **Classes**: `0: Fall Detected`
   - **Direct Resolve Base**: `https://huggingface.co/datasets/DeZan/fall-detection/resolve/main/`
4. **Theft & Intrusion Dataset**:
   - **Repository**: [`ertiaM/Anomaly_Detection_in_Surveillance_Videos`](https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos)
   - **Direct Resolve Base**: `https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos/resolve/main/`
5. **Violence Detection Dataset**:
   - **Repository 1**: [`ertiaM/Anomaly_Detection_in_Surveillance_Videos`](https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos) (`Fighting/` & `Assault/`)
   - **Repository 2**: [`valiantlynxz/godseye-violence-detection-dataset`](https://huggingface.co/datasets/valiantlynxz/godseye-violence-detection-dataset) (`Fight/` & `NonFight/`)
   - **Direct Resolve Base**: `https://huggingface.co/datasets/valiantlynxz/godseye-violence-detection-dataset/resolve/main/`

---

## 5. How to Install & Download the Dataset

We provide **three flexible installation methods** so anyone can access the full dataset:
1. **Google Drive Mirror**: 1-click cloud download of the pre-packaged zip archive.
2. **Automated Python Installer**: 1-click script that fetches directly from public Hugging Face mirrors.
3. **Manual Direct Download**: Step-by-step links for manual file-by-file retrieval.

---

### Method 1: Google Drive Mirror (1-Click Cloud Archive)

The entire verified 4.17 GB dataset has been packaged into ready-to-use archive files hosted on Google Drive with **Public Access ("Anyone with the link can view/download")**.

#### Download Links:
- 📦 **Full CCTV Surveillance Dataset Bundle (All 5 Domains)**:
  `[Google Drive Link: https://drive.google.com/drive/folders/YOUR_DATASET_FOLDER_ID?usp=sharing]`
- 📁 **Individual Domain Archives** (if you only need specific categories):
  - `weapons_yolo.zip` (~179 MB): `[Drive Link]`
  - `fire_smoke_yolo.zip` (~32 MB): `[Drive Link]`
  - `falls_yolo.zip` (~100 MB): `[Drive Link]`
  - `theft_intrusion_videos.zip` (~2.38 GB): `[Drive Link]`
  - `violence_videos.zip` (~1.49 GB): `[Drive Link]`

#### Option A: Download via Web Browser
1. Click the Google Drive link above.
2. Click **Download** (or **Download All**).
3. Unzip the contents into `datasets/raw/`.

#### Option B: Download via Command Line (`gdown`)
You can download directly into your workspace from the terminal without using a browser:
```bash
pip install gdown
# Download the full raw dataset zip from Google Drive
gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O datasets/datasets_raw.zip
# Unzip into datasets/raw/
tar -xf datasets/datasets_raw.zip -C datasets/
```

---

### Method 2: Automated 1-Click Python Installer (Reproducible Build)

#### 1. Prerequisites
Ensure Python 3.8+ is installed with the `requests` library:
```bash
pip install requests
```

#### 2. Run the Master Installer
From the project root directory, run:
```bash
python scripts/install_datasets.py
```

#### What the Installer Does:
1. **Status Audit**: Scans `datasets/raw/` to detect any files that are already downloaded.
2. **Differential Download**: Only downloads categories or files that are missing or incomplete.
3. **Multi-Threaded Acceleration**: Uses a thread pool (`ThreadPoolExecutor`) with chunked streams to maximize download throughput.
4. **Integrity Validation**: Verifies file size thresholds and unpacks compressed zip archives automatically.

---

### Method 3: Manual Direct Download & Setup

If you prefer to download files manually via your web browser:

#### 1. Tactical Weapons
1. Open your browser to:
   - `https://huggingface.co/datasets/fcakyon/gun-object-detection/resolve/main/data/train.zip`
   - `https://huggingface.co/datasets/fcakyon/gun-object-detection/resolve/main/data/valid.zip`
2. Extract `train.zip` into `datasets/raw/weapons/train/`
3. Extract `valid.zip` into `datasets/raw/weapons/valid/`

#### 2. Fire & Smoke
1. Browse to `https://huggingface.co/datasets/betasecond/jimei-fire-smoke-yolo-dataset/tree/main`
2. Download image files (`.jpg`) into `datasets/raw/fire_smoke/train/images/` and corresponding `.txt` labels into `datasets/raw/fire_smoke/train/labels/`
3. Download validation pairs into `datasets/raw/fire_smoke/val/images/` and `labels/`

#### 3. Person Fallen
1. Browse to `https://huggingface.co/datasets/DeZan/fall-detection/tree/main/fall_dataset`
2. Download `images/` into `datasets/raw/falls/fall_dataset/images/`
3. Download `labels/` into `datasets/raw/falls/fall_dataset/labels/`

#### 4. Theft, Intrusion & Violence Videos
1. Open UCF-Crime: `https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos/tree/main`
   - Save `Burglary*.mp4` to `datasets/raw/theft_intrusion/burglary/`
   - Save `Robbery*.mp4` to `datasets/raw/theft_intrusion/robbery/`
   - Save `Shoplifting*.mp4` to `datasets/raw/theft_intrusion/shoplifting/`
   - Save `Stealing*.mp4` to `datasets/raw/theft_intrusion/stealing/`
   - Save `Assault*.mp4` to `datasets/raw/violence/assault/`
   - Save `Fighting*.mp4` to `datasets/raw/violence/fights/`
2. Open GodsEye: `https://huggingface.co/datasets/valiantlynxz/godseye-violence-detection-dataset/tree/main/train`
   - Save `Fight/*.avi` clips to `datasets/raw/violence/godseye_fights/`
   - Save `NonFight/*.avi` clips to `datasets/raw/violence/normal_nonfight/`

---

## 6. How to Verify Dataset Integrity

To verify that all files are correctly structured, not corrupt, and account for the full 4.17 GB:

```bash
python scripts/verify_dataset.py
```

### Expected Output:
```text
================================================================================
Category             | Files      | Size           | Sub-breakdown
================================================================================
falls                | 971        | 99.86      MB | fall_dataset: 970
fire_smoke           | 1200       | 31.58      MB | train: 1000, val: 200
theft_intrusion      | 80         | 2437.98    MB | burglary: 20, robbery: 20, shoplifting: 20, stealing: 20
violence             | 100        | 1526.44    MB | assault: 20, fights: 20, godseye_fights: 30, normal_nonfight: 30
weapons              | 9336       | 179.30     MB | train: 7523, valid: 1811
================================================================================
TOTAL                | 11687      | 4.17       GB | Across 5 Security Threat Domains
================================================================================
```

---

## 7. Data Formats & Annotations

### Object Detection (YOLO Format)
All image categories (`weapons`, `fire_smoke`, `falls`) use standard YOLO normalized bounding box format:
```text
<class_index> <x_center> <y_center> <width> <height>
```
- Coordinates are normalized between `0.0` and `1.0` relative to image dimensions.
- Each image has a corresponding `.txt` annotation file with the exact same stem name.

### Action Recognition & Temporal Surveillance
Video categories (`theft_intrusion`, `violence`) are encoded as:
- **Container**: MP4 / AVI
- **Video Codec**: H.264 / AVC
- **Frame Rates**: 25 - 30 FPS
- **Resolution**: 240p to 720p real CCTV fixed-angle surveillance footage
- **Temporal Structure**: Contains both active incident clips and negative non-incident normal activity for false-positive suppression.
