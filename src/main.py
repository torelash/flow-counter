# Main runner for Flow Counter (YOLOv8 + SORT)
# We will add logic step-by-step.
# --- Flow Counter (YOLOv8 + SORT) -------------------------------------------------
# Detects vehicles + people, tracks with SORT, counts line crossings (robust side-change),
# overlays graphics, and saves result.mp4. Colab-safe and GPU/CPU-friendly.
# -------------------------------------------------------------------------------

# STEP 1: Imports & environment setup
from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np
import math

# SORT import – prefer vendored path; fall back to plain sort.py if needed
try:
    from trackers.sort import Sort
except Exception:
    from sort import Sort

# === Config (easy to tweak) =======================================================
# Model/paths
MODEL_PATH = "yolo-weights/yolov8n.pt"     # start with 'n' for speed; switch to yolov8l.pt later
VIDEO_PATH = "assets/traffic_cam.mp4"
MASK_PATH  = "assets/mask.png"             # optional; set to None to disable

# Classes we want to count (vehicles + people)
ALLOWED_CLASS_NAMES = {"person", "car", "truck", "bus", "motorcycle"}

# Detection thresholds (speed/accuracy knobs)
CONF_MIN = 0.35     # drop low-confidence boxes
IOU_NMS  = 0.50     # YOLO NMS IoU

# SORT tracking parameters
SORT_MAX_AGE  = 22
SORT_MIN_HITS = 3
SORT_IOU_THR  = 0.30

# UI / Colab
SHOW_WINDOW = False  # Colab has no GUI; keep False. Set True if running locally.

# Counting lines (x1, y1, x2, y2) – adjust to your scene
line_up   = [180, 410,  640, 410]
line_down = [680, 400, 1280, 450]

# === Model & I/O init =============================================================
# Load YOLO model and names
model = YOLO(MODEL_PATH)
NAMES = model.names  # dict: {class_id: 'label'}

# Map allowed names -> numeric class IDs present in this model
ALLOWED_CLASS_IDS = {cid for cid, name in NAMES.items() if name in ALLOWED_CLASS_NAMES}

# Open video
vid = cv.VideoCapture(VIDEO_PATH)
if not vid.isOpened():
    raise RuntimeError(f"Could not open video at {VIDEO_PATH}")

# Optional mask
mask = cv.imread(MASK_PATH) if MASK_PATH else None

# Video writer
width  = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps    = vid.get(cv.CAP_PROP_FPS) or 30
video_writer = cv.VideoWriter(
    "result.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Mask sanity: must match frame size
if mask is not None:
    mh, mw = mask.shape[:2]
    if (mw, mh) != (width, height):
        print("[WARN] Mask size doesn't match video. Disabling mask.")
        mask = None

# STEP 5A: Preload overlay graphics (optional)
overlay_main  = cv.imread("assets/graphics.png",  cv.IMREAD_UNCHANGED)
overlay_count = cv.imread("assets/graphics1.png", cv.IMREAD_UNCHANGED)

# === Tracker & counting state =====================================================
tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THR)

# Using sets for efficiency (unique IDs)
count_up      = set()
count_down    = set()
total_count   = set()  # all IDs counted anywhere (prevents double counting)

# STEP 4A: Side-of-line helpers/state for robust counting
def point_side_of_line(x1, y1, x2, y2, px, py):
    """
    +1 if (px,py) is on one side of the directed line (x1,y1)->(x2,y2),
    -1 on the other side, 0 if exactly on the line.
    """
    val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if val > 0: return +1
    if val < 0: return -1
    return 0

last_side_up   = {}  # id -> {-1,0,+1}
last_side_down = {}  # id -> {-1,0,+1}

# === Main loop ====================================================================
while True:
    ok, frame = vid.read()
    if not ok or frame is None:
        print("🎬 Reached end of video or frame read error.")
        break

    # Apply ROI mask (if provided) to the image we feed into YOLO
    frame_region = cv.bitwise_and(frame, mask) if mask is not None else frame

    # YOLO inference (single call; apply thresholds here)
    results = model(frame_region, conf=CONF_MIN, iou=IOU_NMS, verbose=False)

    # Build detections array for SORT: [x1, y1, x2, y2, conf] as float32
    detections = np.empty((0, 5), dtype=np.float32)
    if results and len(results) > 0:
        r = results[0]
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            if cls in ALLOWED_CLASS_IDS and conf >= CONF_MIN:
                detections = np.vstack([detections, [x1, y1, x2, y2, conf]])

    # Update tracker
    tracker_updates = tracker.update(detections)  # ndarray: [x1,y1,x2,y2,track_id] per row

    # Draw counting lines (red baseline)
    cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), 3)
    cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), 3)

    # Overlays (if provided)
    if overlay_main is not None:
        frame = cvzone.overlayPNG(frame, overlay_main, (0, 0))
    if overlay_count is not None:
        frame = cvzone.overlayPNG(frame, overlay_count, (420, 0))

    # Draw tracks and do robust side-change counting
    for upd in tracker_updates:
        x1, y1, x2, y2, tid = map(int, upd)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # centroid marker
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Draw nice box + ID
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{tid}', (x1, y1), scale=1, thickness=2)

        # --- Robust side-change counting ---
        # Current side wrt each line
        side_up_now = point_side_of_line(line_up[0], line_up[1], line_up[2], line_up[3], cx, cy)
        side_dn_now = point_side_of_line(line_down[0], line_down[1], line_down[2], line_down[3], cx, cy)

        # Previous sides (0 if unknown)
        side_up_prev = last_side_up.get(tid, 0)
        side_dn_prev = last_side_down.get(tid, 0)

        # UP-line crossing: sign flip (+/-) and not zero
        if side_up_prev != 0 and side_up_now != 0 and side_up_prev != side_up_now:
            if tid not in total_count:
                total_count.add(tid)
                count_up.add(tid)
                # flash line green for feedback
                cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 255, 0), 3)

        # DOWN-line crossing: sign flip (+/-) and not zero
        if side_dn_prev != 0 and side_dn_now != 0 and side_dn_prev != side_dn_now:
            if tid not in total_count:
                total_count.add(tid)
                count_down.add(tid)
                cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 255, 0), 3)

        # Update memory
        last_side_up[tid]   = side_up_now
        last_side_down[tid] = side_dn_now

    # Counter readouts (positions match your overlay style; adjust as you like)
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_up)),   (600, 85),  cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_down)), (850, 85),  cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)

    # Output
    if SHOW_WINDOW:
        cv.imshow("vid", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_writer.write(frame)

# --- Cleanup ----------------------------------------------------------------------
vid.release()
video_writer.release()
try:
    cv.destroyAllWindows()
except:
    pass
print("✅ Done — saved annotated video to result.mp4")

