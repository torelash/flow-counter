# ===============================================================
# main_two.py  — Baseline counter + tiny auto single/dual picker
# Keeps your proven counting logic. Only adds a fast warm-up to
# decide scene MODE and choose pre-tuned lines accordingly.
# ===============================================================

from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np
import time, os, subprocess

# ---------------------------
# Paths (edit as you use)
# ---------------------------
MODEL_PATH = "yolov8l.pt"                 # works well on CPU/GPU
VIDEO_PATH = "assets/traffic_cam.mp4"     # your input clip
MASK_PATH  = None                         # e.g. "assets/mask.png" or None

# ---------------------------
# Classes to count
# ---------------------------
ALLOWED_CLASS_NAMES = {"person", "car", "truck", "bus", "motorcycle"}

# ---------------------------
# YOLO / SORT knobs
# ---------------------------
CONF_MIN = 0.25
IOU_NMS  = 0.45
IMGSZ    = 960

# SORT (main run)
SORT_MAX_AGE  = 30
SORT_MIN_HITS = 1
SORT_IOU_THR  = 0.25

# ---------------------------
# Mode control (the only “new” thing)
# ---------------------------
AUTO_MODE  = True               # let warm-up decide "single" vs "dual"
FORCE_MODE = None               # set to "single" or "dual" to override

# ---------------------------
# Pre-tuned line presets you like
# (Dual: left lane = UP, right lane = DOWN)
# ---------------------------
DUAL_LEFT_X1_FRAC  = 0.00
DUAL_LEFT_X2_FRAC  = 0.29
DUAL_RIGHT_X1_FRAC = 0.32
DUAL_RIGHT_X2_FRAC = 1.00
DUAL_Y_UP_FRAC     = 0.38
DUAL_Y_DOWN_FRAC   = 0.62

# Single-lane: one middle gate
SINGLE_X1_FRAC  = 0.10
SINGLE_X2_FRAC  = 0.90
SINGLE_Y_FRAC   = 0.55

# Small nudges you asked for (fractions of width/height)
GATE_NUDGE = {
    "UP_X1_PAD":   -0.17,  # extend UP more left
    "UP_X2_PAD":   -0.05,  # trim UP right end
    "DOWN_X1_PAD": -0.13,  # extend DOWN more left
    "DOWN_X2_PAD":  0.00,
    "Y_UP_PAD":     0.00,
    "Y_DOWN_PAD":   0.15,  # move DOWN line lower
}

# ---------------------------
# Counting robustness (kept light)
# ---------------------------
GATE_HALF_THICKNESS_PX = 22      # band around each line
CROSS_COOLDOWN_MS      = 900     # per-id cooldown
SEGMENT_END_MARGIN_FRAC = 0.10   # allow crossing near trimmed ends

# ---------------------------
# Tiny utils
# ---------------------------
def _nudge_line(line, W, H, x1_pad=0.0, x2_pad=0.0, y_pad=0.0):
    x1, y1, x2, y2 = line
    x1 = int(np.clip(x1 + x1_pad * W, 0, W-1))
    x2 = int(np.clip(x2 + x2_pad * W, 0, W-1))
    y  = int(np.clip(y1 + y_pad  * H, 0, H-1))
    return [x1, y, x2, y]

def signed_distance_to_line(pt, line):
    x, y = pt
    x1, y1, x2, y2 = line
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def _in_gate_band(px, py, line, band_px, W=None):
    """Inside thick band around finite segment (a bit forgiving at ends)."""
    x1, y1, x2, y2 = line
    vx, vy = x2 - x1, y2 - y1
    L2 = vx*vx + vy*vy
    if L2 <= 1e-9:
        return False
    t = ((px - x1)*vx + (py - y1)*vy) / L2
    margin_t = SEGMENT_END_MARGIN_FRAC
    ok_seg = (-margin_t <= t <= 1.0 + margin_t)
    d = abs((px - x1)*(y2 - y1) - (py - y1)*(x2 - x1)) / max(1e-6, np.sqrt(L2))
    return ok_seg and (d <= band_px)

def compute_lines(mode, W, H):
    """Return (line_up, line_down, line_mid) for the chosen mode."""
    if mode == "dual":
        up   = [int(W*DUAL_LEFT_X1_FRAC),  int(H*DUAL_Y_UP_FRAC),
                int(W*DUAL_LEFT_X2_FRAC),  int(H*DUAL_Y_UP_FRAC)]
        down = [int(W*DUAL_RIGHT_X1_FRAC), int(H*DUAL_Y_DOWN_FRAC),
                int(W*DUAL_RIGHT_X2_FRAC), int(H*DUAL_Y_DOWN_FRAC)]
        up   = _nudge_line(up,   W, H, GATE_NUDGE["UP_X1_PAD"],   GATE_NUDGE["UP_X2_PAD"],   GATE_NUDGE["Y_UP_PAD"])
        down = _nudge_line(down, W, H, GATE_NUDGE["DOWN_X1_PAD"], GATE_NUDGE["DOWN_X2_PAD"], GATE_NUDGE["Y_DOWN_PAD"])
        return up, down, None
    else:
        mid = [int(W*SINGLE_X1_FRAC), int(H*SINGLE_Y_FRAC),
               int(W*SINGLE_X2_FRAC), int(H*SINGLE_Y_FRAC)]
        return None, None, mid

def auto_detect_mode(video_path, model, allowed_class_ids,
                     conf=0.25, iou=0.45, imgsz=640,
                     sample_frames=120, step=2, jitter_px=0.7,
                     min_events=6, min_frac=0.20):
    """
    Tiny warm-up: track dx direction. If we see both left→right and right→left
    (each ≥ min_frac of motion events), call it 'dual', else 'single'.
    """
    from trackers.sort import Sort
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return "single"

    warm = Sort(max_age=15, min_hits=1, iou_threshold=0.25)
    last_cx = {}
    pos = neg = events = 0
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if frames % step != 0:
            continue

        r = model(frame, conf=conf, iou=iou, imgsz=imgsz, stream=False, verbose=False)[0]
        dets = []
        for b in r.boxes:
            cls = int(b.cls[0])
            if cls in allowed_class_ids:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dets.append([x1, y1, x2, y2, float(b.conf[0])])
        dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)

        tracks = warm.update(dets)
        for x1, y1, x2, y2, tid in tracks:
            x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
            cx = (x1 + x2) / 2.0
            if tid in last_cx:
                dx = cx - last_cx[tid]
                if abs(dx) > jitter_px:
                    events += 1
                    if dx > 0: pos += 1
                    else:      neg += 1
            last_cx[tid] = cx

        if frames >= sample_frames:
            break

    cap.release()
    if events < min_events:
        return "single"
    pos_frac = pos / max(1, events)
    neg_frac = neg / max(1, events)
    return "dual" if (pos_frac >= min_frac and neg_frac >= min_frac) else "single"

# ===============================================================
# Setup
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

# Decide mode once
if FORCE_MODE is not None:
    MODE = FORCE_MODE
elif AUTO_MODE:
    MODE = auto_detect_mode(VIDEO_PATH, model, ALLOWED_CLASS_IDS, conf=CONF_MIN, iou=IOU_NMS, imgsz=640)
else:
    MODE = "dual"

print(f"▶️ MODE selected: {MODE}")

# Get lines for that mode
line_up, line_down, line_mid = compute_lines(MODE, width, height)

# Tracker for the main pass
from trackers.sort import Sort
tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THR)

# Counting state
count_up, count_down, total_count = set(), set(), set()
last_side_up, last_side_down, last_side_mid = {}, {}, {}
last_cross_time_up, last_cross_time_down, last_cross_time_mid = {}, {}, {}
t0 = time.time()

# Writer (temp; we’ll re-encode to standard mp4 after)
temp_name = "result_temp.mp4"
writer = cv.VideoWriter(temp_name, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ===============================================================
# Main loop
# ===============================================================
while True:
    ok, frame = vid.read()
    if not ok: break
    now_ms = int((time.time() - t0) * 1000)

    roi = frame if mask is None else cv.bitwise_and(frame, mask)
    r = model(roi, conf=CONF_MIN, iou=IOU_NMS, imgsz=IMGSZ, stream=False, verbose=False)[0]

    # YOLO detections → tracker
    dets = []
    for b in r.boxes:
        cls = int(b.cls[0])
        if cls in ALLOWED_CLASS_IDS:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            dets.append([x1, y1, x2, y2, float(b.conf[0])])
    dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)
    tracks = tracker.update(dets)

    # Draw lines
    if MODE == "dual":
        cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), 3)
        cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), 3)
        cv.putText(frame, "UP line",   (line_up[0] + 10, line_up[1] - 12),   cv.FONT_HERSHEY_PLAIN, 2, (80,255,80), 3)
        cv.putText(frame, "DOWN line", (line_down[0] + 10, line_down[1] - 12), cv.FONT_HERSHEY_PLAIN, 2, (255,150,50), 3)
    else:
        cv.line(frame, (line_mid[0], line_mid[1]), (line_mid[2], line_mid[3]), (0, 0, 255), 3)
        cv.putText(frame, "GATE", (line_mid[0] + 10, line_mid[1] - 12), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)

    # For each track
    for x1, y1, x2, y2, tid in tracks.astype(int):
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{int(tid)}', (x1, y1), scale=1, thickness=2)

        if MODE == "dual":
            # ---- UP ----
            s_now  = signed_distance_to_line((cx, cy), line_up)
            s_prev = last_side_up.get(tid)
            flip   = (s_prev is not None) and (s_prev * s_now < 0)
            inband = _in_gate_band(cx, cy, line_up, GATE_HALF_THICKNESS_PX, W=width)
            if flip and inband and (now_ms - last_cross_time_up.get(tid, -10**9) >= CROSS_COOLDOWN_MS):
                total_count.add(tid); count_up.add(tid)
                last_cross_time_up[tid] = now_ms
                cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 255, 0), 3)
            last_side_up[tid] = s_now

            # ---- DOWN ----
            s_now  = signed_distance_to_line((cx, cy), line_down)
            s_prev = last_side_down.get(tid)
            flip   = (s_prev is not None) and (s_prev * s_now < 0)
            inband = _in_gate_band(cx, cy, line_down, GATE_HALF_THICKNESS_PX, W=width)
            if flip and inband and (now_ms - last_cross_time_down.get(tid, -10**9) >= CROSS_COOLDOWN_MS):
                total_count.add(tid); count_down.add(tid)
                last_cross_time_down[tid] = now_ms
                cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 255, 0), 3)
            last_side_down[tid] = s_now

        else:
            # ---- SINGLE (MID) ----
            s_now  = signed_distance_to_line((cx, cy), line_mid)
            s_prev = last_side_mid.get(tid)
            flip   = (s_prev is not None) and (s_prev * s_now < 0)
            inband = _in_gate_band(cx, cy, line_mid, GATE_HALF_THICKNESS_PX, W=width)
            if flip and inband and (now_ms - last_cross_time_mid.get(tid, -10**9) >= CROSS_COOLDOWN_MS):
                total_count.add(tid)
                # decide direction by sign change
                if s_prev < 0 and s_now > 0:
                    count_up.add(tid)
                else:
                    count_down.add(tid)
                last_cross_time_mid[tid] = now_ms
                cv.line(frame, (line_mid[0], line_mid[1]), (line_mid[2], line_mid[3]), (0, 255, 0), 3)
            last_side_mid[tid] = s_now

    # HUD
    cv.putText(frame, "TOTAL", (150, 60), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 6)
    cv.putText(frame, "UP",    (560, 55), cv.FONT_HERSHEY_PLAIN, 3, (80, 255, 80), 6)
    cv.putText(frame, "DOWN",  (820, 55), cv.FONT_HERSHEY_PLAIN, 3, (255,150,50), 6)
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_up)),    (600,  85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_down)),  (850,  85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)

    writer.write(frame)

# ===============================================================
# Finish & re-encode to standard mp4 name
# ===============================================================
vid.release(); writer.release(); cv.destroyAllWindows()

VIDEO_OUT = f"result_{MODE}.mp4"
try:
    cmd = ["ffmpeg", "-y", "-i", "result_temp.mp4", "-vcodec", "libx264", "-pix_fmt", "yuv420p", VIDEO_OUT]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove("result_temp.mp4")
    print(f"🎬 Saved: {VIDEO_OUT}")
except Exception as e:
    print(f"⚠️ ffmpeg failed ({e}); keeping raw '{temp_name}'")
