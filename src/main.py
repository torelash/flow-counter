# Main runner for Flow Counter (YOLOv8 + SORT)

# ===============================================================
# Flow Counter — YOLOv8 + SORT (People & Vehicles)
# Detects vehicles + people, tracks with SORT, counts line crossings (robust side-change),
# overlays graphics, and saves result.mp4. Colab-safe and GPU/CPU-friendly.
# ===============================================================
from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np
import math
from pathlib import Path

# --- Tracking ---------------------------------------------------
try:
    from trackers.sort import Sort
except Exception:
    from sort import Sort


# ===============================================================
# Config
# ===============================================================
MODEL_PATH = "yolov8n.pt"                 # auto-download lightweight YOLOv8n
VIDEO_PATH = "assets/traffic_cam.mp4"     # input video
MASK_PATH  = None                         # no mask by default
ALLOWED_CLASS_NAMES = {"person", "car", "truck", "bus", "motorcycle"}

# detection thresholds
CONF_MIN = 0.35
IOU_NMS  = 0.50

# SORT tracking
SORT_MAX_AGE  = 22
SORT_MIN_HITS = 3
SORT_IOU_THR  = 0.30

# auto-calibration
AUTO_CALIBRATE   = True
WARMUP_FRAMES    = 200
QUANTILES        = (0.35, 0.65)
MIN_FLOW_PIX_PER_FRAME = 0.5

# manual fallback lines
line_up   = [180, 410, 640, 410]
line_down = [680, 400, 1280, 450]


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
# Auto-calibration helper
# ===============================================================
def _unit(v):
    n = np.linalg.norm(v)
    return (v / (n + 1e-9)).astype(np.float32), n

def _endpoints_from_center_and_dir(center_xy, dir_xy, span):
    a = (center_xy - dir_xy * span).astype(int)
    b = (center_xy + dir_xy * span).astype(int)
    return [int(a[0]), int(a[1]), int(b[0]), int(b[1])]

def autocalibrate_gates(model, vid, tracker, warmup_frames, conf_min, iou_nms, width, height):
    warm_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=max(1, SORT_MIN_HITS-1), iou_threshold=SORT_IOU_THR)
    last_xy, flow_vecs, centroids = {}, [], []

    try:
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
    except:
        pass

    frames_seen = 0
    while frames_seen < warmup_frames:
        ok, frame = vid.read()
        if not ok or frame is None:
            break
        frames_seen += 1

        results = model(frame, conf=conf_min, iou=iou_nms, verbose=False)
        dets = np.empty((0, 5), dtype=np.float32)
        if results and len(results) > 0:
            r = results[0]
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                if cls in ALLOWED_CLASS_IDS and conf >= conf_min:
                    dets = np.vstack([dets, [x1, y1, x2, y2, conf]])

        tracks = warm_tracker.update(dets)
        for tr in tracks:
            x1, y1, x2, y2, tid = map(int, tr)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            centroids.append([cx, cy])
            if tid in last_xy:
                dx, dy = cx - last_xy[tid][0], cy - last_xy[tid][1]
                if dx*dx + dy*dy > 1.0:
                    flow_vecs.append([dx, dy])
            last_xy[tid] = (cx, cy)

    # rewind
    try:
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
    except:
        path = VIDEO_PATH
        vid.release()
        new_vid = cv.VideoCapture(path)
        globals()['vid'] = new_vid

    if len(flow_vecs) < 10 or len(centroids) < 50:
        print("[AUTO] Not enough motion; keeping manual lines.")
        return None, None

    flow = np.median(np.array(flow_vecs, dtype=np.float32), axis=0)
    d, flow_mag = _unit(flow)
    if flow_mag < MIN_FLOW_PIX_PER_FRAME:
        print(f"[AUTO] Flow too small ({flow_mag:.2f}); keeping manual lines.")
        return None, None

    g = np.array([-d[1], d[0]], dtype=np.float32)
    pts = np.array(centroids, dtype=np.float32)
    proj = pts @ d
    q_low, q_high = np.quantile(proj, QUANTILES)

    c1 = d * q_low
    c2 = d * q_high
    span = max(width, height) * 1.5
    L1 = _endpoints_from_center_and_dir(c1, g, span)
    L2 = _endpoints_from_center_and_dir(c2, g, span)

    y1 = (L1[1] + L1[3]) / 2.0
    y2 = (L2[1] + L2[3]) / 2.0
    line_hi, line_lo = (L1, L2) if y1 < y2 else (L2, L1)

    print(f"[AUTO] Flow ~ {flow} (|v|={flow_mag:.2f}); placed gates.")
    return line_hi, line_lo


# ===============================================================
# Optional auto-calibration call
# ===============================================================
if AUTO_CALIBRATE:
    hi, lo = autocalibrate_gates(model, vid, tracker, WARMUP_FRAMES, CONF_MIN, IOU_NMS, width, height)
    if hi is not None and lo is not None:
        line_up, line_down = hi, lo
        print(f"[AUTO] line_up   = {line_up}")
        print(f"[AUTO] line_down = {line_down}")
    else:
        print("[AUTO] Using manual lines.")


# ===============================================================
# Main loop
# ===============================================================
while True:
    ok, frame = vid.read()
    if not ok or frame is None:
        break

    frame_region = frame if mask is None else cv.bitwise_and(frame, mask)
    results = model(frame_region, conf=CONF_MIN, iou=IOU_NMS, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.floor(box.conf[0]*100)/100
            cls = int(box.cls[0])
            if cls in ALLOWED_CLASS_IDS:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    tracker_updates = tracker.update(detections)

    # draw counting lines
    cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), 3)
    cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), 3)

    # iterate through tracked objects
    for update in tracker_updates:
        x1, y1, x2, y2, tid = map(int, update)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + w//2), (y1 + h//2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # up-line
        if line_up[0] < cx < line_up[2] and line_up[1] - 5 < cy < line_up[3] + 5:
            if tid not in total_count:
                total_count.append(tid)
                if tid not in count_up:
                    count_up.append(tid)
        # down-line
        if line_down[0] < cx < line_down[2] and line_down[1] - 5 < cy < line_down[3] + 5:
            if tid not in total_count:
                total_count.append(tid)
                if tid not in count_down:
                    count_down.append(tid)

        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{tid}', (x1, y1), scale=1, thickness=2)

    # --- draw counts and labels --------------------------------------------
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_up)),    (600,  85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_down)),  (850,  85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)

    # new clear text labels
    cv.putText(frame, "TOTAL", (150, 60), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 6)
    cv.putText(frame, "UP",    (560, 55),  cv.FONT_HERSHEY_PLAIN, 3, (80, 255, 80), 6)
    cv.putText(frame, "DOWN",  (820, 55),  cv.FONT_HERSHEY_PLAIN, 3, (255, 150, 50), 6)

    video_writer.write(frame)

# ===============================================================
# Cleanup
# ===============================================================
vid.release()
video_writer.release()
cv.destroyAllWindows()
print("✅ Done — saved annotated video to result.mp4")
