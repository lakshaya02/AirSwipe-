# Aero-Cursor
A gesture-controlled laptop interface that enables touchless interaction using computer vision for accessibility, productivity, and real-time control.

---

## 🔗 Important Links

| Resource | Link |
|---|---|
| 📊 Presentation Slides | [View Deck](https://docs.google.com/presentation/d/1N-_Js21nMpM5VEdOXUCWVjTUj8EUIO5I/edit?slide=id.p1#slide=id.p1) |
| 🎥 Video Pitch | [Watch Demo](https://drive.google.com/file/d/1KUo2Lrhvw0FbeS_Oxt9YZtJZVJJiIH4R/view?pli=1) |
| 🌐 Live Deployment | _Desktop app — no hosted URL (see setup below)_ |

---

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [How It Works — The Full Pipeline](#-how-it-works--the-full-pipeline)
- [Gesture Reference Table](#-gesture-reference-table)
- [Project Architecture](#-project-architecture)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [APIs & Models Used](#-apis--models-used)
- [Setup & Installation](#-setup--installation)
- [Configuration & Tuning](#-configuration--tuning)
- [Repository Structure](#-repository-structure)
- [Team & Module Ownership](#-team--module-ownership)
- [AI Usage Disclosure](#-ai-usage-disclosure)
- [Open Source Compliance & Attributions](#-open-source-compliance--attributions)
- [License](#-license)

---

## ✨ What It Does

GestureOS turns your webcam into a touchless human-computer interface. Using only hand shapes visible to a standard camera, you can:

- 🖱️ **Move the mouse cursor** — point with your index finger
- 👆 **Left / Right / Double click** — distinct finger patterns per action
- 📜 **Scroll up and down** — open palm or four-finger gesture
- ⌨️ **Trigger hotkeys** — browser back, Escape key
- ⏸️ **Pause & resume** — closed fist freezes all control instantly

There is no special hardware required. No gloves, no controllers, no sensors beyond the camera already built into your laptop.

---

## ⚙️ How It Works — The Full Pipeline

Every video frame passes through a strict, ordered pipeline before any action fires:

```
Webcam Frame
    │
    ▼
① CAPTURE & MIRROR
   cv2.VideoCapture → flip horizontally (natural mirror feel)
    │
    ▼
② HAND DETECTION  [MediaPipe Hand Landmarker]
   BGR → RGB → MediaPipe VIDEO mode
   Outputs 21 3D landmarks (x, y, z) normalized 0.0–1.0
    │
    ▼
③ LANDMARK CONVERSION  [gesture_classifier.py → to_px()]
   Normalized coords → pixel positions on the camera frame
    │
    ▼
④ FINGER STATE DETECTION  [gesture_classifier.py → finger_states()]
   Each finger: tip.y < pip.y? → extended (1) or curled (0)
   Thumb uses x-axis (mirrors correctly for both hands)
    │
    ▼
⑤ RAW GESTURE CLASSIFICATION  [gesture_classifier.py → classify()]
   Priority-ordered if/else table maps (t,i,m,r,pk) → gesture string
    │
    ▼
⑥ MAJORITY-VOTE SMOOTHING  [main.py]
   Rolling deque of last N frames → most common gesture wins
   Eliminates single-frame misdetections and flicker
    │
    ▼
⑦ ACTION EXECUTION  [gesture_action.py → execute_action()]
   Edge-triggered (clicks/hotkeys): fires ONCE on gesture change
   Continuous (move/scroll): fires EVERY frame with cooldown timer
   Exponential smoothing on mouse position → no jitter
    │
    ▼
⑧ HUD RENDER  [display.py]
   Hand skeleton overlay + gesture label + FPS + click flash ring
    │
    ▼
   cv2.imshow() → display window
```

### Key Design Decisions

**Why majority-vote smoothing?**
MediaPipe occasionally drops or misclassifies a single frame. A rolling window of 3 frames and a majority vote means one bad frame never fires the wrong action.

**Why exponential smoothing on the mouse?**
Raw landmark coordinates jitter slightly even when the hand is still. Blending the previous position toward the new target (`pos = pos + (target - pos) * factor`) produces a smooth, natural cursor glide without added latency.

**Why edge-triggering for clicks?**
A click gesture held for 5 frames should fire exactly once, not 5 times. The `is_new` flag (current gesture ≠ previous gesture) guarantees single-fire behaviour per gesture onset.

---

## 🤌 Gesture Reference Table

| Gesture | Hand Shape | Action | Trigger Type |
|---|---|---|---|
| ✊ Closed fist | All fingers down | **Pause / Resume** | Edge (toggle) |
| 🤙 Pinky only | Pinky up, thumb down | **Browser Back** (Alt+←) | Edge |
| 🖐️ Open palm | All 5 fingers up | **Scroll Up** | Continuous |
| 🖖 Four fingers | Index–Pinky, no thumb | **Scroll Down** | Continuous |
| ☝️ Index only | Index up only | **Mouse Move** | Continuous |
| 🤙 Shaka | Thumb + Pinky up | **Left Click** | Edge |
| ✌️ Peace sign | Index + Middle | **Right Click** | Edge |
| 🤟 Three fingers | Index + Middle + Ring | **Double Click** | Edge |

> **Tip:** Pause with a closed fist before resting your hand — prevents accidental actions.

---

## 🏗️ Project Architecture

The codebase is split into **four single-responsibility modules** — each owned by one team member — orchestrated by a thin `main.py` entry point.

```
┌─────────────────────────────────────────────────────┐
│                      main.py                        │
│          Frame loop · Module orchestration          │
└───┬───────────┬──────────────┬──────────────┬───────┘
    │           │              │              │
    ▼           ▼              ▼              ▼
config.py  gesture_       gesture_        display.py
           classifier.py  action.py

Constants   Landmark      System          HUD overlay
Settings    → Finger       actions:        Hand skeleton
Thresholds  states         click/scroll    FPS counter
Model cfg   → Gesture      hotkeys         Click flash
HUD meta    name           cooldowns       Pause text
```

**Data flows in one direction** — `config.py` is read-only by all modules; no module imports another except through `main.py`.

---

## 🛠️ Tech Stack & Dependencies

### Language
- **Python 3.11**

### Core Libraries

| Library | Version | Purpose |
|---|---|---|
| `mediapipe` | ≥ 0.10 | Hand landmark detection (21-point 3D skeleton) |
| `opencv-python` | ≥ 4.9 | Webcam capture, frame rendering, HUD drawing |
| `pyautogui` | ≥ 0.9.54 | Mouse move/click, keyboard hotkeys, screen size detection |
| `numpy` | ≥ 1.26 | Landmark array math, coordinate transforms |

### Installation

```bash
pip install mediapipe opencv-python pyautogui numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
mediapipe>=0.10
opencv-python>=4.9
pyautogui>=0.9.54
numpy>=1.26
```

> **Python version note:** Tested on Python 3.11. MediaPipe may not support Python 3.12+ — use 3.11 for guaranteed compatibility.

---

## 🔌 APIs & Models Used

### MediaPipe Hand Landmarker (Google)

| Property | Detail |
|---|---|
| **Model** | `hand_landmarker.task` (Float16, ~6 MB) |
| **Source** | Google MediaPipe Model Hub |
| **Download URL** | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` |
| **Auto-download** | Yes — `config.py` fetches it on first run if not present |
| **Running mode** | `VIDEO` — requires monotonically increasing timestamps per frame |
| **Landmarks output** | 21 3D keypoints per hand (normalized 0.0–1.0) |

**Configuration used in this project:**
```python
num_hands                     = 1      # Single hand tracked at a time
min_hand_detection_confidence = 0.8    # Threshold to first detect a hand
min_hand_presence_confidence  = 0.8    # Threshold to continue tracking
min_tracking_confidence       = 0.6    # Threshold for inter-frame tracking
```

### PyAutoGUI (System Control)

| Property | Detail |
|---|---|
| **Type** | Local OS automation library (no network calls) |
| **Usage** | `moveTo()`, `click()`, `rightClick()`, `doubleClick()`, `scroll()`, `hotkey()`, `press()` |
| **Safety** | `FAILSAFE = True` — move mouse to top-left corner to emergency-stop |

> No external APIs, cloud services, or paid subscriptions are required. The system runs entirely offline after the one-time model download.

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11
- A working webcam (built-in or USB)
- Windows / macOS / Linux

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/gesture-os.git
cd gesture-os
```

### Step 2 — Install dependencies

```bash
pip install mediapipe opencv-python pyautogui numpy
```

### Step 3 — Run

```bash
python3.11 main.py
```

On first launch, the hand landmark model (~6 MB) is downloaded automatically and saved as `hand_landmarker.task` in the project directory. All subsequent launches are instant.

### Step 4 — Use it

1. Position your hand clearly in the webcam frame
2. The HUD overlay shows the detected gesture in real time
3. Press **Q** in the camera window (or close it) to quit
4. Move mouse to **top-left screen corner** as emergency stop (PyAutoGUI failsafe)

### Platform Notes

| OS | Notes |
|---|---|
| **macOS** | Grant camera + accessibility permissions in System Settings |
| **Linux** | May need `sudo apt install python3-tk` for PyAutoGUI |
| **Windows** | No extra setup required |

---

## 🎛️ Configuration & Tuning

All tuneable parameters live exclusively in **`config.py`** — no other file needs editing for behaviour changes.

| Parameter | Default | Effect |
|---|---|---|
| `SCROLL_COOLDOWN` | `0.03` s | Lower = faster scroll |
| `SCROLL_SPEED` | `6` | Scroll units per tick |
| `CLICK_COOLDOWN` | `0.50` s | Minimum time between clicks |
| `HOTKEY_COOLDOWN` | `0.50` s | Minimum time between hotkeys |
| `MOUSE_SMOOTH` | `0.22` | `0` = instant snap · `1` = never moves |
| `MARGIN` | `0.12` | Edge dead-zone fraction — increase if cursor can't reach screen edges |
| `HISTORY_LEN` | `3` | Smoothing window — higher = more stable, slight extra lag |
| `CAM_W / CAM_H` | `640 × 480` | Webcam resolution |

---

## 📁 Repository Structure

```
gesture-control/
├── main.py                  # Entry point & main loop
├── config.py                # ALL settings, constants, model setup
├── gesture_classifier.py    # Landmark → finger states → gesture name
├── gesture_action.py        # Gesture name → system action (click/scroll/etc.)
├── display.py               # Camera window HUD overlay & hand skeleton
├── LICENSE                  # MIT License
└── README.md
```

---


## 🤖 AI Usage Disclosure

In full compliance with the hackathon's AI usage policy, we disclose the following:

| Tool | Purpose |
|---|---|
| **Claude (Anthropic)** | Code generation only — used to generate boilerplate code structure, docstrings, and utility scaffolding |

### Important Clarifications

- **Claude was used solely for code generation assistance** — it did not design the system architecture, choose the gesture mapping, or determine the core algorithms.
- All AI-generated code was **read, understood, tested, and validated** line-by-line by the responsible team member before being committed.
- The gesture classification logic, smoothing strategy, edge-trigger patterns, and coordinate-mapping math were **designed entirely by the team** and implemented with AI as a typing aid only.
- Every team member can fully explain the code they own during evaluation without referring to any AI output.
- Core innovation (gesture table design, smoothing pipeline, module architecture) is entirely human-authored.

---

## ✅ Open Source Compliance & Attributions

This project follows **FOSS (Free and Open Source Software)** principles in full:

- All source code is freely accessible, modifiable, and reusable under the MIT License.
- Core logic has **no dependency on closed or proprietary services** — the system runs 100% offline after model download.
- All third-party libraries are open source with proper attribution below.

### Attributions

| Library | License | Usage in this project |
|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache 2.0 | Hand landmark detection model and inference pipeline |
| [OpenCV](https://github.com/opencv/opencv) | Apache 2.0 | Webcam capture, image processing, HUD rendering |
| [PyAutoGUI](https://github.com/asweigart/pyautogui) | BSD 3-Clause | OS-level mouse, keyboard, and screen control |
| [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause | Array operations and coordinate math |

No plagiarized or uncredited code is present in this repository.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 AmanGupta-project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
```

---

<div align="center">

**Built with 🖐️ and a lot of hand-waving at BYTEVERSE · 2026**



</div>
