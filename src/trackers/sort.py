# Placeholder for SORT; we'll replace with real implementation later.
# Minimal SORT tracker for integration (Colab-safe, no GUI deps)
# Source adapted from: https://github.com/abewley/sort
# Kept: KalmanBoxTracker, Sort, iou_batch, linear_assignment
# Removed: matplotlib, skimage, CLI, display code.

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment  # Hungarian

np.random.seed(0)

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """Compute IoU between two sets of boxes [x1,y1,x2,y2]."""
    bb_gt = np.expand_dims(bb_gt, 0)    # [1, M, 4]
    bb_test = np.expand_dims(bb_test, 1)  # [N, 1, 4]

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[..., 2]   - bb_gt[..., 0])   * (bb_gt[..., 3]   - bb_gt[..., 1])

    union = area_test + area_gt - inter + 1e-9
    return inter / union

def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [x,y,s,r] (center x,y, area, aspect)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-9)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """[x,y,s,r] -> [x1,y1,x2,y2] (optionally with score)."""
    w = np.sqrt(max(0.0, x[2] * x[3]))
    h = x[2] / (w + 1e-9)
    x1 = x[0] - w / 2.
    y1 = x[1] - h / 2.
    x2 = x[0] + w / 2.
    y2 = x[1] + h / 2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """Tracks a single object using a Kalman filter."""
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update with observed bbox [x1,y1,x2,y2,score?]."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """Advance state and return predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Match detections to trackers using IoU + Hungarian.
    Returns (matches, unmatched_dets, unmatched_trks)
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    if iou_matrix.size > 0:
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] >= iou_threshold:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: array [[x1,y1,x2,y2,score], ...]
        return: array [[x1,y1,x2,y2,id], ...]
        """
        self.frame_count += 1

        # Predict existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Update matched trackers with detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialize new trackers for unmatched dets
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # Build results and cull stale trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # IDs are 1-based
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

