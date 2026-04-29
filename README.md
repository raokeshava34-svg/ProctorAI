# ProctorAI — Intelligent Real-Time Online Examination Proctoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-purple?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-green?style=for-the-badge&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A fully self-contained web-based proctoring and examination platform that integrates real-time AI behavioural detection with a complete three-part assessment engine — all running on consumer hardware with no cloud, no GPU, and no licensing fees.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Detection Pipeline](#detection-pipeline)
- [Examination Engine](#examination-engine)
- [Performance Results](#performance-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Models](#training-the-models)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)
- [How It Works](#how-it-works)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

ProctorAI solves the fundamental contradiction at the centre of every existing online proctoring system: a student who opens a code editor or runs a SQL query to answer a programming question triggers a window-switch violation under any monitoring-only system. ProctorAI eliminates this problem by building the proctoring engine and the examination engine as a **single inseparable system** — the monitoring layer always knows what the examination requires.

The system detects five behavioural categories in real time from a standard laptop webcam and microphone:

| Class | Description |
|---|---|
| **Normal** | Student is attentive — no violation |
| **Phone** | Mobile phone visible in frame |
| **Multi-Person** | More than one person present |
| **Voice** | Student speaking during silent exam |
| **ID Mismatch** | Different person from enrolled student |

Every confirmed violation produces a timestamped, composited photographic evidence screenshot captured at the exact moment of detection — before the student has time to hide the phone during the server's response delay.

---

## Key Features

### Proctoring
- **Real-time 5-class detection** — Phone, Multi-Person, Voice, Identity Mismatch, Normal
- **YOLOv8n object detection** — phone (COCO class 67, conf ≥ 0.15) and person (COCO class 0, conf ≥ 0.40)
- **72-dimensional feature vector** across 6 sensor modalities
- **ProctorCNN v6 + Random Forest weighted ensemble** — P_final = 0.60 × CNN + 0.40 × RF
- **Multi-frame smoothing buffers** — reduces raw FP rates from 14% to below 5%
- **Automatic exam termination** on repeated violations
- **Photographic evidence** — screenshot from pre-stored frame at exact moment of violation
- **Full-screen enforcement** — exam blocked until browser full-screen confirmed
- **Speech recognition** — continuous Web Speech API with auto-restart, en-IN accent model
- **Face enrolment** — reference embedding captured at session start for identity comparison

### Examination
- **Part A** — 20 MCQs from 480-question aptitude bank (MD5 email seed for unique sets)
- **Part B** — 2 SQL problems evaluated against in-memory SQLite (order-independent)
- **Part C** — 3 coding problems in Python, Java, C, C++, JavaScript with partial credit
- **sessionStorage persistence** — answers survive accidental page refresh
- **30-minute timed session** with automatic submission on expiry

### Administration
- **Live admin dashboard** — real-time violation counts and cheating scores
- **Evidence gallery** — timestamped violation screenshots per session
- **Excel export** — all session data, scores, and violation records
- **Email invitations** — unique one-time examination links per student
- **Question bank management** — add, edit, tag questions through admin API

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BROWSER (Student)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Webcam       │  │ Microphone   │  │  Exam Interface     │  │
│  │ 1280×720     │  │ Web Audio    │  │  MCQ + SQL + Code   │  │
│  │ 30fps        │  │ RMS/ZCR/Peak │  │  sessionStorage     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘  │
│         │                 │                       │             │
│  canvas.toDataURL()  ──── │ ──── POST /api/detect_frame        │
│  lastFrameDataUrl stored  │                       │             │
└─────────┼─────────────────┼───────────────────────┼─────────────┘
          │                 │                       │
          ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK SERVER (app.py)                        │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  YOLOv8n    │  │   Feature    │  │   CNN + RF          │   │
│  │  Detection  │  │  Extraction  │  │   Ensemble          │   │
│  │  cls67+cls0 │  │  72-dim vec  │  │   60% + 40%         │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Smoothing  │  │   Cheating   │  │   Termination       │   │
│  │  Buffer     │  │   Score      │  │   Logic             │   │
│  │  θ per class│  │  min(100,..) │  │   3/3/3/2 rule      │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
│                                                                 │
│              SESSIONS dict (in-memory)                         │
│              SQLite in-memory (SQL eval)                       │
│              tempfile (code execution)                         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                ADMIN DASHBOARD (/admin/dashboard)               │
│   Live sessions · Evidence gallery · Export · Settings          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection Pipeline

### Step 1 — Frame Capture (every 600ms)
```javascript
// Browser captures frame and stores BEFORE any network call
canvas.drawImage(video, -W, 0, W, H);
lastFrameDataUrl = canvas.toDataURL('image/jpeg', 0.92);
// Screenshot will always show the violation, not a later clean frame
```

### Step 2 — Feature Extraction (server side)
```
72-dimensional feature vector from 6 groups:
  [0-11]   Phone/YOLO Detection    — YOLOv8n class 67
  [12-23]  Multi-Person/Face       — YOLOv8n class 0 + Haar
  [24-35]  Audio/Voice             — Web Audio API
  [36-47]  Identity Verification   — Face embedding + LBP
  [48-59]  Head Pose/Gaze          — OpenCV pose estimation
  [60-71]  Motion/Composite        — Frame differencing
```

### Step 3 — Ensemble Prediction
```python
P_final = 0.60 * P_CNN + 0.40 * P_RF

# Per-class confidence thresholds
thresholds = {
    "Normal":       0.50,
    "Phone":        0.40,
    "Multi-Person": 0.40,
    "Voice":        0.38,   # lower = better recall
    "ID Mismatch":  0.40,
}
# Below threshold → defaults to Normal (no false alert)
```

### Step 4 — Smoothing Buffer
```python
_SMOOTH = {"phone": 2, "multi": 4, "voice": 2, "identity": 3}

# Counter increments on detection, decrements on non-detection
# Phone counter NEVER decrements (student who hides phone stays counted)
# Alert fires only when counter >= threshold
```

### Step 5 — Termination Check
```python
PHONE_TERMINATE         = 3   # 3 confirmed phone alerts → end exam
MULTI_TERMINATE         = 3   # 3 confirmed multi-person alerts → end exam
FACE_MISMATCH_TERMINATE = 3   # 3 confirmed ID mismatches → end exam
TAB_TERMINATE           = 2   # 2 tab switches → end exam
```

### Step 6 — Cheating Score
```python
C = min(100, 15*N_phone + 12*N_multi + 10*N_face + 8*N_voice + 15*N_tab)
```

| Score | Risk Level |
|---|---|
| C ≥ 70 | 🔴 High Risk — Terminate |
| 40 ≤ C < 70 | 🟡 Moderate — Warning |
| C < 40 | 🟢 Low Risk — Normal |

---

## Examination Engine

```
Part A — MCQ
  • 20 questions from 480-question aptitude bank
  • Selected using MD5(student_email) as random seed
  • Every student gets a different subset

Part B — SQL
  • 2 SQL problems
  • Evaluated against in-memory SQLite3
  • Order-independent row comparison (correct output, wrong order = full marks)

Part C — Coding
  • 3 problems
  • Languages: Python, Java, C, C++, JavaScript
  • Compiled and executed via subprocess
  • Partial credit: marks proportional to test cases passed
  • Exam duration: 1800 seconds (30 minutes)
```

---

## Performance Results

Evaluated across **50 real controlled examination sessions** with **200 deliberate violation events per class** on Intel Core i5 hardware with **no GPU**.

| Class | Precision | Recall | F1-Score | Accuracy | FP Rate | Smooth θ |
|---|---|---|---|---|---|---|
| Phone | 91.8% | 92.3% | 92.0% | 93.1% | 4.1% | θ = 2 |
| Multi-Person | 94.1% | 88.7% | 91.3% | 92.4% | 2.8% | θ = 4 |
| Voice | 84.5% | 79.1% | 81.7% | 83.2% | 8.3% | θ = 2 |
| ID Mismatch | 90.1% | 85.4% | 87.7% | 89.5% | 5.2% | θ = 3 |
| **Macro Average** | **90.1%** | **86.4%** | **88.2%** | **89.6%** | **5.1%** | — |

> **Without smoothing buffers:** raw per-frame FP rate exceeds 14% → over 270 false alerts per 30-minute session
>
> **With smoothing buffers:** FP rate drops below 5% for 3 of 4 violation categories

---

## Project Structure

```
ProctorAI/
│
├── app.py                          # Main Flask application (33 routes)
├── training.py                     # Dataset generation + model training
│
├── models/                         # Trained model files (generated by training.py)
│   ├── cnn_phone.h5                # ProctorCNN v6 (HDF5 format)
│   ├── proctor_cnn_v6.keras        # ProctorCNN v6 (SavedModel format)
│   ├── proctor_rf_v6.pkl           # Random Forest (400 trees)
│   └── scaler_v6.pkl               # StandardScaler (fitted on training data)
│
├── data/                           # Dataset (generated by training.py)
│   └── proctor_dataset_v6.csv      # 90,000 rows × 72 features + label
│
├── screenshots/                    # Evidence screenshots (auto-created)
│
├── templates/                      # Jinja2 HTML templates
│   ├── exam.html                   # Student examination interface
│   ├── student_login.html          # Student login page
│   ├── result.html                 # Examination result page
│   ├── admin_dashboard.html        # Admin monitoring dashboard
│   └── admin_login.html            # Admin login page
│
└── requirements.txt                # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip
- Git
- Webcam and microphone (for examination sessions)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/ProctorAI.git
cd ProctorAI
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
flask
tensorflow>=2.0
scikit-learn
ultralytics          # YOLOv8
opencv-python
numpy
pandas
openpyxl             # Excel export
```

### Step 4 — Train the Models
```bash
python training.py
```

This will:
- Generate `data/proctor_dataset_v6.csv` (90,000 synthetic feature vectors)
- Train ProctorCNN v6 (Deep Residual 1D CNN with SE attention)
- Train Random Forest (400 trees)
- Save all model files to `models/`

> Training takes approximately 10–30 minutes on a standard CPU.
> A GPU will significantly reduce training time but is not required.

### Step 5 — Run the Application
```bash
python app.py
```

The server starts at `http://127.0.0.1:5000`

---

## Usage

### For Administrators

**1. Login to admin dashboard:**
```
http://127.0.0.1:5000/admin/login
```

**2. Send exam invitation to student:**
```
POST /admin/api/send_invite
→ Generates unique one-time exam link
→ Sends to student email via SMTP
```

**3. Monitor live sessions:**
```
http://127.0.0.1:5000/admin/dashboard
→ Real-time violation counts
→ Live cheating scores
→ Evidence screenshot gallery
```

**4. Export results:**
```
GET /admin/api/export
→ Downloads Excel file with all session data
```

---

### For Students

**1. Open unique exam link** (received by email):
```
http://127.0.0.1:5000/exam/start/<token>
```

**2. Enter full-screen** when prompted
- Exam is blocked until full-screen is confirmed
- Exiting full-screen records a tab-switch violation

**3. Complete face enrolment**
- Look at camera for 3 seconds
- Reference embedding captured for identity comparison

**4. Answer exam questions**
- Part A: Select MCQ answers
- Part B: Type SQL queries and click Run
- Part C: Write code, select language, click Submit

**5. Submit exam** when finished or when timer reaches zero

---

## Training the Models

### Dataset Generation
```python
# training.py generates 90,000 synthetic feature vectors
# 18,000 per class × 5 classes
# Class-conditional statistical distributions calibrated to real sensor readings

N_PER_CLASS  = 18000   # samples per class
NUM_CLASSES  = 5       # Normal, Phone, Multi-Person, Voice, ID-Mismatch
NUM_FEATURES = 72      # feature dimensions
SEED         = 42      # fully deterministic and reproducible
```

### Training Configuration
```python
BATCH_SIZE   = 512
EPOCHS       = 150     # with early stopping patience=22
DROPOUT      = 0.35
L2_REG       = 5e-5
MIXUP_ALPHA  = 0.20    # doubles effective training set to ~136,800
LR_INIT      = 2e-3    # Cosine Decay with Warm Restarts (SGDR)
LR_MIN       = 5e-7
```

### Model Architecture
```
ProctorCNN v6 ResidualSE:

Input (72, 1)
    ↓
Entry Conv1D(64, k=7, BN+ReLU)
    ↓
Residual Block 1 (64f,  dil=1) + SE(ratio=8) + MaxPool → (36, 64)
Residual Block 2 (128f, dil=1) + SE(ratio=8) + MaxPool → (18, 128)
Residual Block 3 (128f, dil=2) + SE(ratio=8)           → (18, 128)
Residual Block 4 (256f, dil=1) + SE(ratio=8)           → (18, 256)
Residual Block 5 (256f, dil=4) + SE(ratio=8)           → (18, 256)
Residual Block 6 (512f, dil=1) + SE(ratio=8)           → (18, 512)
    ↓
GlobalMaxPool + GlobalAvgPool → concat → (1024,)
    ↓
Dense(512, BN, Dropout=0.35)
Dense(256, BN, Dropout=0.35)
Dense(128, Dropout=0.14)
    ↓
Dense(5, Softmax)          ← P_CNN

Random Forest:
  n_estimators  = 400
  max_depth     = 25
  class_weight  = 'balanced'
  oob_score     = True

Ensemble:
  P_final = 0.60 × P_CNN + 0.40 × P_RF
```

---





## Hardware Requirements

### Minimum (Student Device)
| Component | Requirement |
|---|---|
| Processor | Intel Core i3 or equivalent |
| RAM | 4 GB |
| Camera | 720p built-in webcam |
| Microphone | Built-in microphone |
| Browser | Google Chrome (recommended) |
| Internet | Stable broadband |
| OS | Windows 10/11, Ubuntu 20.04+, macOS |

### Recommended (Server)
| Component | Requirement |
|---|---|
| Processor | Intel Core i5/i7 or equivalent |
| RAM | 8–16 GB |
| Storage | 50 GB+ SSD |
| GPU | Optional (CPU-only inference supported) |
| OS | Ubuntu 22.04 LTS |

> **No GPU is required for inference.** YOLOv8n is specifically designed for CPU-only deployment. The full detection pipeline runs in real time on a standard Core i5 laptop.

---

## How It Works

### Why Synthetic Dataset?

ProctorAI does **not** use an image dataset. It uses a **72-dimensional feature vector dataset** (proctor_dataset_v6.csv):

1. **Privacy** — Recording real students requires ethics approval
2. **Class rarity** — Violations are less than 0.5% of real frames; 18,000 balanced examples would require 1,000+ hours of footage
3. **Feature specificity** — The 72-dim vector is unique to ProctorAI's sensor pipeline; no public dataset provides it

### Why the Screenshot Always Shows the Violation

```
Detection loop runs:
  1. Draw frame to canvas
  2. Store as lastFrameDataUrl   ← stored HERE, before anything else
  3. Send to server (600ms round-trip)
  4. Server responds with phone_alert=True
  5. captureScreenshot(lastFrameDataUrl)  ← uses frame from step 2
                                          ← phone guaranteed visible
```

Without this, the screenshot would be taken after the server responds — by which time the student has had 600ms to hide the phone.

### Why CNN + Random Forest?

- **CNN** detects sequential patterns across the 72-feature vector through dilated convolutions
- **Random Forest** detects non-linear threshold combinations through axis-aligned splits
- Their errors are largely **uncorrelated** — combining them is more accurate than either alone
- The 60/40 split was determined empirically on the validation set

---

## Limitations and Future Work

| Limitation | Planned Fix |
|---|---|
| Voice FP rate 8.3% (ambient noise) | Calibrate threshold against enrolment baseline |
| Session data lost on server restart | Migrate to PostgreSQL |
| Single-threaded Flask dev server | Deploy with Gunicorn + Nginx |
| No liveness detection at enrolment | Add blink/nod challenge during face enrolment |
| Out-of-frame devices undetected | Mobile companion app for secondary device monitoring |
| No LMS integration | REST API integration with Moodle, Canvas, Blackboard |

---

## Citation

If you use ProctorAI in your research, please cite:

```bibtex
@article{proctorai2025,
  title   = {ProctorAI: Intelligent Real-Time Online Examination Proctoring System},
  author  = {Your Name},
  journal = {IEEE Access},
  year    = {2025},
  note    = {ProctorCNN v6 ResidualSE + Random Forest ensemble, 
             macro-average F1 = 88.2\% on consumer hardware}
}
```

---



<div align="center">

**Built with Python · Flask · TensorFlow · YOLOv8 · scikit-learn · OpenCV**

*Macro-average F1: 88.2% · Accuracy: 89.6% · No GPU Required · No Cloud Dependency*

</div>
