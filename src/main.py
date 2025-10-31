# Main runner for Flow Counter (YOLOv8 + SORT)

# ===============================================================
# Flow Counter — YOLOv8 + SORT (People & Vehicles)
# Detects vehicles + people, tracks with SORT, counts line crossings (robust side-change),
# overlays graphics, and saves result.mp4. Colab-safe and GPU/CPU-friendly.
# ===============================================================
# Flow Counter — YOLOv8 + SORT (People & Vehicles) | Lane-scoped gates
# ===============================================================
from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np
import math, os, subprocess

# --- Tracking ---------------------------------------------------
try:
    from trackers.sort import Sort
except Exception:
    from sort import Sort

# ===============================================================
# Config (tuned for better distant/blurred car recall)
# ===============================================================
MODEL_PATH = "yolov8s.pt"
VIDEO_PATH = "assets/traffic_cam.mp4"
MASK_PATH  = None

ALLOWED_CLASS_NAMES = {"person", "car", "truck", "bus", "motorcycle"}

CONF_MIN = 0.25
IOU_NMS  = 0.45

SORT_MAX_AGE  = 30
SORT_MIN_HITS = 1
SORT_IOU_THR  = 0.25

# Re-encode to standard MP4 at the end (fixes Colab/browser playback)
STANDARDIZE_MP4 = True

# ===============================================================
# Model & video setup
# ===============================================================
model = YOLO(MODEL_PATH)
NAMES = model.names
ALLOWED_CLASS_IDS = {cid for cid, name in NAMES.items() if name in ALLOWED_CLASS_NAMES}

vid = cv.VideoCapture(VIDEO_PATH)
if not vid.isOpened():
    raise RuntimeError(f"Could not open video at {VIDEO_PATH}")

mask = cv.imread(MASK_PATH) if MASK_PATH else None
width  = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps    = vid.get(cv.CAP_PROP_FPS) or 30

video_writer = cv.VideoWriter(
    "result.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THR)
count_up, count_down, total_count = [], [], []

# ===============================================================
# Lane-scoped horizontal lines (left = UP, right = DOWN)
# Tweaked fractions so:
#  - UP line is strictly within left lanes
#  - DOWN line starts further left (covers more of right lanes)
# If still off, nudge LEFT_X2_FRAC (smaller = shorter UP), RIGHT_X1_FRAC (smaller = longer DOWN).
# ===============================================================
LEFT_X1_FRAC  = 0.00
LEFT_X2_FRAC  = 0.29    # was 0.48 → shorten UP so it doesn't reach right lanes
RIGHT_X1_FRAC = 0.32    # was 0.52 → extend DOWN leftward into the divide/right lanes
RIGHT_X2_FRAC = 1.00

Y_UP_FRAC     = 0.38
Y_DOWN_FRAC   = 0.62

LEFT_X1   = int(width  * LEFT_X1_FRAC)
LEFT_X2   = int(width  * LEFT_X2_FRAC)
RIGHT_X1  = int(width  * RIGHT_X1_FRAC)
RIGHT_X2  = int(width  * RIGHT_X2_FRAC)
Y_UP      = int(height * Y_UP_FRAC)
Y_DOWN    = int(height * Y_DOWN_FRAC)

line_up   = [LEFT_X1,  Y_UP,   LEFT_X2,  Y_UP]
line_down = [RIGHT_X1, Y_DOWN, RIGHT_X2, Y_DOWN]

LINE_BAND = 12  # ± pixels around line Y used to count (tolerates jitter)

# ===============================================================
# Main loop
# ===============================================================
while True:
    ok, frame = vid.read()
    if not ok or frame is None:
        break

    frame_region = frame if mask is None else cv.bitwise_and(frame, mask)
    results = model(frame_region, conf=CONF_MIN, iou=IOU_NMS, imgsz=960, stream=True, verbose=False)

    detections = np.empty((0, 5), dtype=np.float32)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            if cls in ALLOWED_CLASS_IDS and conf >= CONF_MIN:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf], dtype=np.float32)))

    tracker_updates = tracker.update(detections)

    # Draw gates + small labels on the segments
    cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), 3)
    cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), 3)
    cv.putText(frame, "UP line",   (line_up[0] + 10,   line_up[1] - 12),   cv.FONT_HERSHEY_PLAIN, 2, (80,255,80), 3)
    cv.putText(frame, "DOWN line", (line_down[0] + 10, line_down[1] - 12), cv.FONT_HERSHEY_PLAIN, 2, (255,150,50), 3)

    for update in tracker_updates:
        x1, y1, x2, y2, tid = map(int, update)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + w//2), (y1 + h//2)

        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{tid}', (x1, y1), scale=1, thickness=2)

        # UP (left lanes)
        if (line_up[0] <= cx <= line_up[2]) and (line_up[1] - LINE_BAND <= cy <= line_up[3] + LINE_BAND):
            if tid not in total_count:
                total_count.append(tid)
                if tid not in count_up:
                    count_up.append(tid)
            cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 255, 0), 3)

        # DOWN (right lanes)
        if (line_down[0] <= cx <= line_down[2]) and (line_down[1] - LINE_BAND <= cy <= line_down[3] + LINE_BAND):
            if tid not in total_count:
                total_count.append(tid)
                if tid not in count_down:
                    count_down.append(tid)
            cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 255, 0), 3)

    # Counters + labels
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_up)),    (600,  85),  cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_down)),  (850,  85),  cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, "TOTAL", (150, 60), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 6)
    cv.putText(frame, "UP",    (560, 55),  cv.FONT_HERSHEY_PLAIN, 3, (80, 255, 80), 6)
    cv.putText(frame, "DOWN",  (820, 55),  cv.FONT_HERSHEY_PLAIN, 3, (255, 150, 50), 6)

    video_writer.write(frame)

# ===============================================================
# Cleanup + browser-friendly MP4
# ===============================================================
vid.release()
video_writer.release()
cv.destroyAllWindows()
print("✅ Done — saved annotated video to result.mp4")

if STANDARDIZE_MP4:
    src = "/content/flow-counter/result.mp4"
    dst = "/content/flow-counter/result_fixed.mp4"
    try:
        # Re-encode with H.264 / yuv420p for broad compatibility (Colab, browsers)
        cmd = ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"🎬 Re-encoded to {dst}")
    except Exception as e:
        print(f"ffmpeg re-encode skipped/failed: {e}")
