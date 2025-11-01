# 🛣️ Flow-Counter: Vehicle & People Counter using YOLOv8 + SORT

> Count vehicles and people in videos using deep learning and object tracking.

---

## ⚡ Quick Start

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
# or individually:
pip install ultralytics opencv-python cvzone filterpy
apt install ffmpeg -y   # for Colab re-encoding
```

### 2️⃣ Run baseline (stable)
```bash
python src/main.py
```

### 3️⃣ Run experimental (auto lane detection)
```bash
python src/main_two.py
```

**Output files:**
```
result_dual.mp4
result_single.mp4
```

---

## 🌐 Overview
Flow-Counter combines **YOLOv8** object detection and **SORT** tracking to automatically **count vehicles and pedestrians** in traffic videos.  
It detects whether a scene is **single-lane** or **dual-lane**, tracks moving objects, and produces an **annotated MP4 video** showing total and directional counts.

---

## 🧠 How It Works
```
VIDEO → YOLO Detection → SORT Tracking → Line Crossing Check → Count Update → Annotated Output
```
1. **YOLOv8** detects objects (cars, trucks, buses, motorbikes, people).  
2. **SORT** tracks each detected object across frames using Kalman filtering.  
3. When a tracked object crosses a “counting gate” (a red line), it’s counted once per direction.  
4. The system overlays all bounding boxes, lines, and counts on the output video.

---

## 🚀 Key Features
- Real-time object detection (YOLOv8)  
- Object tracking with SORT  
- Auto scene detection (single vs dual lane)  
- Directional flow counting  
- Annotated MP4 output  
- Manual gate tuning through fractional coordinates  
- Optional ROI masking for focus regions  

---

## 📁 Project Structure
```bash
flow-counter/
│
├── src/
│   ├── main.py          # ✅ Baseline (stable) YOLO+SORT counter
│   ├── main_two.py      # 🧪 Experimental version with auto lane detection
│   └── trackers/
│       └── sort.py      # SORT tracker (Kalman-based)
│
├── assets/
│   ├── traffic_cam.mp4  # Example input video
│   ├── mask.png         # Optional ROI mask
│
├── requirements.txt
├── README.md
└── result_dual.mp4      # Example annotated output
```

---

## ⚙️ Configuration
Edit parameters directly in `src/main.py` or `src/main_two.py`.

| Parameter | Description |
|------------|-------------|
| `MODEL_PATH` | YOLO weights file (`yolov8l.pt`, `yolov8n.pt`, etc.) |
| `VIDEO_PATH` | Input video |
| `AUTO_MODE` | Enable auto lane detection (only in `main_two.py`) |
| `FORCE_MODE` | Manually override mode (“single” or “dual”) |
| `DUAL_*_FRAC` | Gate line positions for dual-lane setup |
| `SINGLE_*_FRAC` | Mid-line position for single-lane setup |
| `GATE_NUDGE` | Fractional adjustments to fine-tune gate placement |
| `CONF_MIN`, `IOU_NMS` | YOLO detection thresholds |

---

## ▶️ Usage

### Baseline (Stable)
```bash
python src/main.py
```
This version uses fixed line coordinates. Reliable for dual-lane traffic videos.

### Experimental (Auto Mode)
```bash
python src/main_two.py
```
This version automatically determines whether the video is single or dual-lane, then adjusts gate positions.

---

## 📊 Output Visualization
| Element | Meaning |
|--------|---------|
| 🔴 Red lines | Counting gates |
| 🟩 Green flashes | Object successfully counted |
| 🟪 Magenta boxes | Active tracked objects |
| 🧾 On-screen text | Total / Up / Down counts |

---

## 🛠️ Prerequisites
You’ll need:
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- cvzone
- filterpy

Install all at once:
```bash
pip install ultralytics opencv-python cvzone filterpy
```

If using Google Colab:
```bash
apt install ffmpeg -y
```

---

## ⚠️ Notes & Limitations
- **Auto-mode detection:** Works best for clear overhead or angled scenes; may misclassify on short or diagonal clips.  
- **Counting sensitivity:** Objects may be tracked but not counted if they don’t fully cross a gate or lose ID due to blur/occlusion.  
- **Gate calibration:** Line positions are static but tunable.  
- **Frame rate sensitivity:** Low-FPS or motion blur can cause missed counts.  
- **Ethical limitation:** This project does not analyze demographics, race, or facial attributes — only movement and object type.

---

## 📈 Future Improvements
- Optical-flow–based adaptive gate placement (Farneback / RAFT)  
- Improved tracking for temporary occlusions  
- Streamlit web interface for uploads and analytics  
- Animated overlays (icons instead of boxes)  
- Statistical summaries: flow per minute, lane occupancy, etc.

---

## 🙌 Acknowledgments
- **Ultralytics YOLOv8** — object detection  
- **Alex Bewley’s SORT** — real-time tracking  
- **OpenCV & cvzone** — visualization tools  

---

## 🧩 Version Guide
| File | Role | Stability |
|------|------|-----------|
| `main.py` | Baseline version (manual gate setup, tested) | ✅ Stable |
| `main_two.py` | Auto scene detection & adaptive gates | 🧪 Experimental |

---

## 📸 Example Output (Placeholder)
Add your result screenshot here:
```bash
![Example Output](assets/example_frame.jpg)
```

---

**Developed by Toorese Lasebikan, 2025**

---

### requirements.txt
```bash
ultralytics>=8.0.0
opencv-python
cvzone
filterpy
```




