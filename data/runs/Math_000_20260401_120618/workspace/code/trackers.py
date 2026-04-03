import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Detection:
    frame: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    gt_id: Optional[int] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Track:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    is_active: bool = True

    def last_detection(self) -> Detection:
        return self.detections[-1]

    def add_detection(self, det: Detection):
        self.detections.append(det)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def pseudo_depth(det: Detection) -> float:
    # Simple proxy: larger area -> closer (smaller depth)
    # We return "depth rank" such that larger area => smaller value.
    # The absolute scale is irrelevant; only ordering matters.
    return -det.area


class HierarchicalSparseTracker:
    """Hierarchical association with pseudo-depth-based sparse subsets.

    Algorithm sketch (per frame):
      1. Split detections into high/low score groups (ByteTrack-style).
      2. Within each group, further decompose into sparse layers by pseudo-depth
         so that 2D overlaps inside each layer stay below an IoU threshold.
      3. Associate active tracks with detections layer by layer using IoU.
         - Process near-depth layers first (large boxes, depth small).
         - Use Hungarian assignment on IoU cost.
      4. Unmatched high-score detections start new tracks.
      5. Tracks are kept alive for a small number of frames (max_age).
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        high_score_thresh: float = 0.4,
        low_score_thresh: float = 0.1,
        max_age: int = 3,
        max_iou_within_layer: float = 0.1,
    ):
        self.iou_thresh = iou_thresh
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.max_age = max_age
        self.max_iou_within_layer = max_iou_within_layer

        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 0
        self.last_seen: Dict[int, int] = {}

    def _build_sparse_layers(self, dets: List[Detection]) -> List[List[Detection]]:
        # Sort by pseudo-depth (near to far).
        dets_sorted = sorted(dets, key=pseudo_depth)
        layers: List[List[Detection]] = []
        for det in dets_sorted:
            placed = False
            for layer in layers:
                if all(iou(det.bbox, d.bbox) <= self.max_iou_within_layer for d in layer):
                    layer.append(det)
                    placed = True
                    break
            if not placed:
                layers.append([det])
        return layers

    def _associate_layer(self, frame: int, layer_dets: List[Detection]):
        if not layer_dets:
            return

        # Active track indices
        active_ids = [tid for tid, t in self.tracks.items() if t.is_active]
        if not active_ids:
            return

        tracks_last = [self.tracks[tid].last_detection() for tid in active_ids]
        cost = np.zeros((len(tracks_last), len(layer_dets)), dtype=float)
        for i, t_det in enumerate(tracks_last):
            for j, det in enumerate(layer_dets):
                cost[i, j] = 1.0 - iou(t_det.bbox, det.bbox)

        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_dets = set()
        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] < self.iou_thresh:
                continue
            tid = active_ids[r]
            det = layer_dets[c]
            self.tracks[tid].add_detection(det)
            self.last_seen[tid] = frame
            assigned_tracks.add(tid)
            assigned_dets.add(c)

        # No track creation here; that is done after processing all layers.
        # Unmatched tracks may later match low-score detections in deeper layers.

        return assigned_dets

    def step(self, frame: int, detections: List[Detection]):
        # Split by score
        high = [d for d in detections if d.score >= self.high_score_thresh]
        low = [d for d in detections if self.low_score_thresh <= d.score < self.high_score_thresh]

        # Build layers for high-score detections and associate first
        matched_high_indices: set = set()
        for layer in self._build_sparse_layers(high):
            matched = self._associate_layer(frame, layer)
            if matched is not None:
                matched_high_indices.update({id(d) for i, d in enumerate(layer) if i in matched})

        # For unmatched high-score detections, start new tracks
        for det in high:
            if id(det) in matched_high_indices:
                continue
            self.tracks[self.next_id] = Track(track_id=self.next_id, detections=[det])
            self.last_seen[self.next_id] = frame
            self.next_id += 1

        # Now allow low-score detections to recover missing associations
        for layer in self._build_sparse_layers(low):
            self._associate_layer(frame, layer)

        # Deactivate tracks that have been missing for too long
        for tid, last in list(self.last_seen.items()):
            if frame - last > self.max_age:
                self.tracks[tid].is_active = False

    def get_tracks(self) -> List[Track]:
        return list(self.tracks.values())


class ByteLikeTracker:
    """Simplified ByteTrack-like tracker (no depth decomposition).

    Used as a baseline to compare with the hierarchical sparse tracker.
    """

    def __init__(self, iou_thresh: float = 0.3, high_score_thresh: float = 0.4, low_score_thresh: float = 0.1, max_age: int = 3):
        self.iou_thresh = iou_thresh
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.max_age = max_age

        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 0
        self.last_seen: Dict[int, int] = {}

    def _associate(self, frame: int, dets: List[Detection]):
        if not dets:
            return set()
        active_ids = [tid for tid, t in self.tracks.items() if t.is_active]
        if not active_ids:
            return set()
        tracks_last = [self.tracks[tid].last_detection() for tid in active_ids]
        cost = np.zeros((len(tracks_last), len(dets)), dtype=float)
        for i, t_det in enumerate(tracks_last):
            for j, det in enumerate(dets):
                cost[i, j] = 1.0 - iou(t_det.bbox, det.bbox)

        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = set()
        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] < self.iou_thresh:
                continue
            tid = active_ids[r]
            det = dets[c]
            self.tracks[tid].add_detection(det)
            self.last_seen[tid] = frame
            matched.add(c)
        return matched

    def step(self, frame: int, detections: List[Detection]):
        high = [d for d in detections if d.score >= self.high_score_thresh]
        low = [d for d in detections if self.low_score_thresh <= d.score < self.high_score_thresh]

        # 1) Associate high-score detections
        matched_high = self._associate(frame, high)

        # 2) Start new tracks for unmatched high-score detections
        for idx, det in enumerate(high):
            if idx in matched_high:
                continue
            self.tracks[self.next_id] = Track(track_id=self.next_id, detections=[det])
            self.last_seen[self.next_id] = frame
            self.next_id += 1

        # 3) Use low-score detections to fill gaps
        self._associate(frame, low)

        # 4) Deactivate stale tracks
        for tid, last in list(self.last_seen.items()):
            if frame - last > self.max_age:
                self.tracks[tid].is_active = False

    def get_tracks(self) -> List[Track]:
        return list(self.tracks.values())
