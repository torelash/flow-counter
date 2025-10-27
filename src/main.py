# Main runner for Flow Counter (YOLOv8 + SORT)
# We will add logic step-by-step.
# --- STEP 1: Environment & Files ---

# Imports
from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np   # you use np later
import math

# If you vendored SORT to src/trackers/sort.py, use this import:
try:
    from trackers.sort import Sort
except Exception:
    # fallback if the module is named differently in your environment
    from sort import Sort

# Paths (adjust to your repo)
MODEL_PATH = "yolo-weights/yolov8n.pt"      # start small; swap to yolov8l.pt after sanity-checks
VIDEO_PATH = "assets/traffic_cam.mp4"
MASK_PATH  = "assets/mask.png"               # optional

# Load model and use its own label map (avoids 'motorbike' mismatch)
model = YOLO(MODEL_PATH)
NAMES = model.names  # dict: {id: 'person', ...}

# Open video with a safety check
vid = cv.VideoCapture(VIDEO_PATH)
if not vid.isOpened():
    raise RuntimeError(f"Could not open video at {VIDEO_PATH}")

# Optional mask (ensure size matches, otherwise disable)
mask = cv.imread(MASK_PATH) if MASK_PATH else None

# Tracker & counting state
tracker = Sort(max_age=22, min_hits=3, iou_threshold=0.3)
line_up   = [180, 410,  640, 410]
line_down = [680, 400, 1280, 450]
count_up, count_down, total_count = [], [], []

# Video writer (with fps fallback)
width  = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps    = vid.get(cv.CAP_PROP_FPS) or 30
video_writer = cv.VideoWriter(
    "result.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# If mask exists but size mismatched, warn & disable
if mask is not None:
    mh, mw = mask.shape[:2]
    if (mw, mh) != (width, height):
        print("[WARN] Mask size doesn't match video. Disabling mask for now.")
        mask = None

