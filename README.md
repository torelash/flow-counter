# Flow Counter — YOLOv8 + SORT (Vehicles & People)

# Flow Counter — YOLOv8 + SORT (Vehicles & People)

Effortlessly count **cars and people** in videos using a YOLOv8-powered detector plus SORT tracking. Designed for traffic and pedestrian flow monitoring, with clean counting lines, optional ROI masks, and a saved, fully annotated result video.

---

## 🌐 Overview

This project detects objects per frame (YOLOv8), then **tracks** them over time (SORT) to assign **stable IDs**. It counts when tracked objects **cross user-defined lines** (robust side-change logic), so you can measure directional flow (e.g., “up” vs “down” lanes) without double-counts.

- **Detector:** Ultralytics YOLOv8 (auto-downloads weights like `yolov8n.pt`)
- **Tracker:** SORT (Kalman filter + IoU/Hungarian matching)
- **Counting:** Line-crossing via side-change (geometric sign flip)
- **Output:** Annotated `result.mp4` with boxes, IDs, lines, and counts

---

## 🚀 Features

- **People + Vehicles**: count `person`, `car`, `truck`, `bus`, `motorcycle` by default (configurable).
- **Stable IDs**: SORT keeps the same ID attached to each object across frames.
- **Robust counting**: side-change (not band-based), so lingering near a line won’t inflate counts.
- **Optional ROI mask**: ignore irrelevant regions (trees/sky/sidewalk) to speed up and reduce noise.
- **Overlays**: optional PNG overlays for quick UI/branding.
- **Colab-friendly**: no GUI requirements; preview result inline.

---

## 🧱 Project Structure

