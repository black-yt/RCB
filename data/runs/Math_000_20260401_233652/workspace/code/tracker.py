"""
Multi-Object Tracking implementations:
- ByteTrack: Two-stage association with high/low confidence detections
- SparseTrack: Pseudo-depth estimation + hierarchical association for crowded scenes

Both algorithms are evaluated on a simulated multi-object sequence.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import json


# ─────────────────────────── Kalman Filter ────────────────────────────────

def create_kalman_filter(bbox):
    """Constant-velocity KF; state [cx, cy, w, h, vx, vy, vw, vh]."""
    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8)
    kf.F[:4, 4:] = np.eye(4)
    kf.H = np.eye(4, 8)
    kf.R *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    kf.x[:4] = np.array([[cx], [cy], [w], [h]])
    return kf


def bbox_from_kf(kf):
    cx, cy, w, h = kf.x[:4].flatten()
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ──────────────────────────── IoU utilities ───────────────────────────────

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(track_boxes, det_boxes):
    M, N = len(track_boxes), len(det_boxes)
    mat = np.zeros((M, N))
    for i, tb in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            mat[i, j] = iou(tb, db)
    return mat


def hungarian_match(cost_mat, threshold):
    row_ind, col_ind = linear_sum_assignment(-cost_mat)
    matches, unmatched_r, unmatched_c = [], list(range(cost_mat.shape[0])), list(range(cost_mat.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_mat[r, c] >= threshold:
            matches.append((r, c))
            if r in unmatched_r: unmatched_r.remove(r)
            if c in unmatched_c: unmatched_c.remove(c)
    return matches, unmatched_r, unmatched_c


# ─────────────────────────── Track class ──────────────────────────────────

class Track:
    _id_counter = 0

    def __init__(self, det_bbox, det_gt_id=None, min_hits=1):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.kf = create_kalman_filter(det_bbox)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.gt_id = det_gt_id
        self.min_hits = min_hits
        self.confirmed = (self.hits >= self.min_hits)
        self.bbox_history = [list(det_bbox)]
        self.frame_history = []  # frame indices

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det_bbox, gt_id=None):
        x1, y1, x2, y2 = det_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.update(np.array([[cx], [cy], [w], [h]]))
        self.hits += 1
        self.time_since_update = 0
        if gt_id is not None:
            self.gt_id = gt_id
        self.bbox_history.append(list(det_bbox))
        self.confirmed = (self.hits >= self.min_hits)

    def get_state(self):
        return bbox_from_kf(self.kf)

    def get_depth(self, frame_h=600):
        """Pseudo-depth of this track based on its predicted bounding box."""
        b = self.get_state()
        h = max(1, b[3] - b[1])
        return float(np.clip(1.0 - (h / frame_h) ** 0.6, 0, 1))

    @classmethod
    def reset(cls):
        cls._id_counter = 0


# ═══════════════════════════════════════════════════════════════════════════
#  ByteTrack
# ═══════════════════════════════════════════════════════════════════════════

class ByteTrack:
    """Two-stage association: high-conf dets first, then low-conf dets."""

    def __init__(self, high_thresh=0.5, low_thresh=0.1, iou_thresh=0.3,
                 max_age=30, min_hits=1):
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh
        self.iou_thresh  = iou_thresh
        self.max_age     = max_age
        self.min_hits    = min_hits
        self.tracks      = []
        Track.reset()

    def update(self, detections):
        for t in self.tracks:
            t.predict()

        high_dets = [d for d in detections if d['score'] >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]

        active_tracks = [t for t in self.tracks if t.time_since_update <= 1]

        # Stage 1: high-confidence ↔ active tracks
        unmatched_t, unmatched_h = self._associate(active_tracks, high_dets)

        # Stage 2: low-confidence ↔ remaining tracks
        remaining = [active_tracks[i] for i in unmatched_t]
        self._associate(remaining, low_dets)

        # New tracks from unmatched high-conf detections
        for i in unmatched_h:
            d = high_dets[i]
            self.tracks.append(Track(d['bbox'], d.get('gt_id'), self.min_hits))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed]

    def _associate(self, tracks, dets):
        if not tracks or not dets:
            return list(range(len(tracks))), list(range(len(dets)))
        tb = [t.get_state() for t in tracks]
        db = [d['bbox'] for d in dets]
        iou_mat = iou_matrix(tb, db)
        matches, unmatched_t, unmatched_d = hungarian_match(iou_mat, self.iou_thresh)
        for ti, di in matches:
            tracks[ti].update(dets[di]['bbox'], dets[di].get('gt_id'))
        return unmatched_t, unmatched_d


# ═══════════════════════════════════════════════════════════════════════════
#  SparseTrack
# ═══════════════════════════════════════════════════════════════════════════

def pseudo_depth(bbox, frame_h=600):
    """Pseudo-depth from bounding box height: small boxes → large depth (farther)."""
    h = max(1, bbox[3] - bbox[1])
    return float(np.clip(1.0 - (h / frame_h) ** 0.6, 0, 1))


def assign_depth_layer(depth_val, n_layers=3):
    edges = np.linspace(0, 1 + 1e-9, n_layers + 1)
    for i in range(n_layers):
        if edges[i] <= depth_val < edges[i + 1]:
            return i
    return n_layers - 1


def decompose_by_depth(items, depth_fn, n_layers=3):
    """Return list-of-lists partitioning items into n_layers depth buckets."""
    layers = [[] for _ in range(n_layers)]
    for item in items:
        d = depth_fn(item)
        layers[assign_depth_layer(d, n_layers)].append(item)
    return layers


class SparseTrack:
    """
    Hierarchical association via pseudo-depth decomposition.

    Algorithm:
      1. Split high-conf detections into n_layers depth layers.
      2. Split active tracks into corresponding depth layers.
      3. Associate within each layer (nearest → farthest), tracks matched
         in earlier layers are removed from subsequent layers.
      4. Fallback: remaining unmatched tracks ↔ low-conf detections.
      5. Create new tracks from still-unmatched high-conf detections.

    This reduces the effective density per sub-problem and prevents
    cross-depth mismatches that cause ID switches in occluded scenes.
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1,
                 iou_thresh=0.3, max_age=30, min_hits=1, n_layers=3):
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh
        self.iou_thresh  = iou_thresh
        self.max_age     = max_age
        self.min_hits    = min_hits
        self.n_layers    = n_layers
        self.tracks      = []
        Track.reset()

    def update(self, detections):
        for t in self.tracks:
            t.predict()

        high_dets = [d for d in detections if d['score'] >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]

        active_tracks = [t for t in self.tracks if t.time_since_update <= 1]

        # Depth-decompose detections and tracks
        det_layers   = decompose_by_depth(high_dets,    lambda d: pseudo_depth(d['bbox']),  self.n_layers)
        track_layers = decompose_by_depth(active_tracks, lambda t: t.get_depth(),            self.n_layers)

        matched_track_ptr = set()
        unmatched_high_dets = []

        # Hierarchical association: layer 0 (nearest) → layer n-1 (farthest)
        for layer_idx in range(self.n_layers):
            layer_dets   = det_layers[layer_idx]
            # Tracks in this layer that haven't been matched yet
            layer_tracks = [t for t in track_layers[layer_idx] if id(t) not in matched_track_ptr]

            if not layer_dets:
                continue
            if not layer_tracks:
                unmatched_high_dets.extend(layer_dets)
                continue

            tb = [t.get_state() for t in layer_tracks]
            db = [d['bbox'] for d in layer_dets]
            iou_mat = iou_matrix(tb, db)
            matches, unmatched_t, unmatched_d = hungarian_match(iou_mat, self.iou_thresh)

            for ti, di in matches:
                layer_tracks[ti].update(layer_dets[di]['bbox'], layer_dets[di].get('gt_id'))
                matched_track_ptr.add(id(layer_tracks[ti]))

            unmatched_high_dets.extend([layer_dets[i] for i in unmatched_d])

        # Fallback: remaining tracks ↔ low-conf detections
        remaining_tracks = [t for t in active_tracks if id(t) not in matched_track_ptr]
        if remaining_tracks and low_dets:
            tb = [t.get_state() for t in remaining_tracks]
            db = [d['bbox'] for d in low_dets]
            iou_mat = iou_matrix(tb, db)
            matches2, _, _ = hungarian_match(iou_mat, self.iou_thresh)
            for ti, di in matches2:
                remaining_tracks[ti].update(low_dets[di]['bbox'], low_dets[di].get('gt_id'))
                matched_track_ptr.add(id(remaining_tracks[ti]))

        # New tracks from still-unmatched high-conf detections
        for d in unmatched_high_dets:
            self.tracks.append(Track(d['bbox'], d.get('gt_id'), self.min_hits))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed]


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(gt_data, track_results, iou_threshold=0.5):
    """Compute MOTA, MOTP, IDF1, ID-switches, FP, FN."""
    total_tp = total_fp = total_fn = total_id_sw = 0
    total_iou = 0.0
    total_gt = 0
    gt_id_tp  = {}
    gt_id_cnt = {}
    prev_gt_to_track = {}

    for gt_frame, tracks in zip(gt_data, track_results):
        gt_ids   = gt_frame['gt_ids']
        gt_boxes = gt_frame['gt_bboxes']
        n_gt = len(gt_ids)
        total_gt += n_gt
        for gid in gt_ids:
            gt_id_cnt[gid] = gt_id_cnt.get(gid, 0) + 1

        if not tracks:
            total_fn += n_gt
            continue

        track_boxes = [t.get_state() for t in tracks]
        track_ids   = [t.track_id for t in tracks]

        iou_mat = np.zeros((n_gt, len(tracks)))
        for gi, gb in enumerate(gt_boxes):
            for ti, tb in enumerate(track_boxes):
                iou_mat[gi, ti] = iou(gb, tb)

        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_gt, matched_tr = set(), set()
        current_gt_to_track = {}

        for gi, ti in zip(row_ind, col_ind):
            if iou_mat[gi, ti] >= iou_threshold:
                total_tp += 1
                total_iou += iou_mat[gi, ti]
                matched_gt.add(gi)
                matched_tr.add(ti)
                gid = gt_ids[gi]
                tid = track_ids[ti]
                current_gt_to_track[gid] = tid
                gt_id_tp[gid] = gt_id_tp.get(gid, 0) + 1
                if gid in prev_gt_to_track and prev_gt_to_track[gid] != tid:
                    total_id_sw += 1

        total_fn += n_gt - len(matched_gt)
        total_fp += len(tracks) - len(matched_tr)
        prev_gt_to_track.update(current_gt_to_track)

    mota = 1.0 - (total_fn + total_fp + total_id_sw) / max(1, total_gt)
    motp = total_iou / max(1, total_tp)

    total_hyp = sum(len(t) for t in track_results)
    sum_tp  = sum(gt_id_tp.values())
    sum_cnt = sum(gt_id_cnt.values())
    idp = sum_tp / max(1, total_hyp)
    idr = sum_tp / max(1, sum_cnt)
    idf1 = 2 * idp * idr / max(1e-9, idp + idr)

    return {
        'MOTA': round(mota, 4),
        'MOTP': round(float(motp), 4),
        'IDF1': round(idf1, 4),
        'ID_switches': total_id_sw,
        'FP': total_fp,
        'FN': total_fn,
        'TP': total_tp,
        'total_gt': total_gt,
    }


def compute_per_frame_metrics(gt_data, track_results, iou_threshold=0.5):
    """Compute per-frame TP, FP, FN, ID-switches."""
    per_frame = []
    prev_gt_to_track = {}
    for frame_idx, (gt_frame, tracks) in enumerate(zip(gt_data, track_results)):
        gt_ids   = gt_frame['gt_ids']
        gt_boxes = gt_frame['gt_bboxes']
        n_gt = len(gt_ids)

        if not tracks:
            per_frame.append({'frame': frame_idx, 'TP': 0, 'FP': 0, 'FN': n_gt, 'IDS': 0, 'n_tracks': 0})
            continue

        track_boxes = [t.get_state() for t in tracks]
        track_ids   = [t.track_id for t in tracks]

        iou_mat = np.zeros((n_gt, len(tracks)))
        for gi, gb in enumerate(gt_boxes):
            for ti, tb in enumerate(track_boxes):
                iou_mat[gi, ti] = iou(gb, tb)

        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_gt, matched_tr = set(), set()
        ids_frame = 0
        current_gt_to_track = {}

        for gi, ti in zip(row_ind, col_ind):
            if iou_mat[gi, ti] >= iou_threshold:
                matched_gt.add(gi)
                matched_tr.add(ti)
                gid = gt_ids[gi]
                tid = track_ids[ti]
                current_gt_to_track[gid] = tid
                if gid in prev_gt_to_track and prev_gt_to_track[gid] != tid:
                    ids_frame += 1

        per_frame.append({
            'frame': frame_idx,
            'TP': len(matched_gt),
            'FP': len(tracks) - len(matched_tr),
            'FN': n_gt - len(matched_gt),
            'IDS': ids_frame,
            'n_tracks': len(tracks),
        })
        prev_gt_to_track.update(current_gt_to_track)
    return per_frame


def run_tracker(TrackerClass, data, **kwargs):
    tracker = TrackerClass(**kwargs)
    results = []
    for frame in data:
        active = tracker.update(frame['detections'])
        results.append(list(active))
    return results


if __name__ == '__main__':
    with open('../data/simulated_sequence.json') as f:
        data = json.load(f)

    print("Running ByteTrack...")
    bt_results = run_tracker(ByteTrack, data, high_thresh=0.5, low_thresh=0.1,
                             iou_thresh=0.3, min_hits=1)
    bt_metrics = compute_metrics(data, bt_results)
    print("ByteTrack:", bt_metrics)

    print("Running SparseTrack...")
    st_results = run_tracker(SparseTrack, data, high_thresh=0.5, low_thresh=0.1,
                             iou_thresh=0.3, n_layers=3, min_hits=1)
    st_metrics = compute_metrics(data, st_results)
    print("SparseTrack:", st_metrics)
