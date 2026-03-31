import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
import seaborn as sns
from scipy.optimize import linear_sum_assignment

sns.set_theme(style="whitegrid")


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    last_frame: int
    hits: int = 1
    age: int = 0
    history: List[Dict] = field(default_factory=list)
    pseudo_depth: float = 0.0

    def update(self, bbox: np.ndarray, score: float, frame_idx: int, depth: float):
        self.bbox = bbox.astype(float)
        self.score = float(score)
        self.last_frame = frame_idx
        self.hits += 1
        self.age = 0
        self.pseudo_depth = float(depth)
        self.history.append({
            "frame": frame_idx,
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "pseudo_depth": self.pseudo_depth,
        })

    def mark_missed(self):
        self.age += 1


def xyxy_to_tlwh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pairwise_iou(boxes):
    n = len(boxes)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(iou(boxes[i], boxes[j]))
    return vals


def bbox_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def pseudo_depth_from_bbox(box, image_height=600.0):
    h = max(1.0, box[3] - box[1])
    y_center = 0.5 * (box[1] + box[3])
    return 0.7 * (h / image_height) + 0.3 * (y_center / image_height)


def assign_tracks(track_indices, det_indices, tracks, detections, iou_threshold=0.1, depth_gate=None, lambda_depth=0.4):
    if not track_indices or not det_indices:
        return [], list(track_indices), list(det_indices)
    cost = np.full((len(track_indices), len(det_indices)), 1e6, dtype=float)
    for r, tidx in enumerate(track_indices):
        tr = tracks[tidx]
        for c, didx in enumerate(det_indices):
            det = detections[didx]
            ov = iou(tr.bbox, np.array(det["bbox"], dtype=float))
            if ov < iou_threshold:
                continue
            depth_cost = 0.0
            if depth_gate is not None:
                diff = abs(tr.pseudo_depth - det["pseudo_depth"])
                if diff > depth_gate:
                    continue
                depth_cost = lambda_depth * diff
            cost[r, c] = 1.0 - ov + depth_cost
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    unmatched_tracks = set(track_indices)
    unmatched_dets = set(det_indices)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e5:
            continue
        tidx = track_indices[r]
        didx = det_indices[c]
        matches.append((tidx, didx))
        unmatched_tracks.discard(tidx)
        unmatched_dets.discard(didx)
    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


class ByteTrackerLike:
    def __init__(self, high_thresh=0.6, low_thresh=0.1, iou_high=0.3, iou_low=0.15, max_age=10):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_logs = []

    def step(self, detections, frame_idx):
        for det in detections:
            det["pseudo_depth"] = pseudo_depth_from_bbox(det["bbox"])
        high = [i for i, d in enumerate(detections) if d["score"] >= self.high_thresh]
        low = [i for i, d in enumerate(detections) if self.low_thresh <= d["score"] < self.high_thresh]
        active_indices = [i for i, tr in enumerate(self.tracks) if tr.age <= self.max_age]
        matches1, unmatched_tracks, unmatched_high = assign_tracks(active_indices, high, self.tracks, detections, self.iou_high)
        for tidx, didx in matches1:
            det = detections[didx]
            self.tracks[tidx].update(np.array(det["bbox"]), det["score"], frame_idx, det["pseudo_depth"])
        matches2, unmatched_tracks2, unmatched_low = assign_tracks(unmatched_tracks, low, self.tracks, detections, self.iou_low)
        for tidx, didx in matches2:
            det = detections[didx]
            self.tracks[tidx].update(np.array(det["bbox"]), det["score"], frame_idx, det["pseudo_depth"])
        matched_track_indices = {tidx for tidx, _ in matches1 + matches2}
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_track_indices:
                tr.mark_missed()
        for didx in unmatched_high:
            det = detections[didx]
            track = Track(self.next_id, np.array(det["bbox"], dtype=float), det["score"], frame_idx)
            track.update(np.array(det["bbox"], dtype=float), det["score"], frame_idx, det["pseudo_depth"])
            self.tracks.append(track)
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.age <= self.max_age]
        frame_output = []
        for tr in self.tracks:
            if tr.last_frame == frame_idx:
                frame_output.append({"track_id": tr.track_id, "bbox": tr.bbox.tolist(), "score": tr.score, "pseudo_depth": tr.pseudo_depth})
        self.frame_logs.append({
            "frame": frame_idx,
            "num_detections": len(detections),
            "num_tracks_output": len(frame_output),
            "matches_high": len(matches1),
            "matches_low": len(matches2),
            "new_tracks": len(unmatched_high),
        })
        return frame_output


class SparseHierarchicalTracker(ByteTrackerLike):
    def __init__(self, high_thresh=0.6, low_thresh=0.1, iou_high=0.25, iou_low=0.12, max_age=12, depth_bins=4, depth_gate=0.12):
        super().__init__(high_thresh, low_thresh, iou_high, iou_low, max_age)
        self.depth_bins = depth_bins
        self.depth_gate = depth_gate

    def step(self, detections, frame_idx):
        for det in detections:
            det["pseudo_depth"] = pseudo_depth_from_bbox(det["bbox"])
        high = [i for i, d in enumerate(detections) if d["score"] >= self.high_thresh]
        low = [i for i, d in enumerate(detections) if self.low_thresh <= d["score"] < self.high_thresh]
        active_indices = [i for i, tr in enumerate(self.tracks) if tr.age <= self.max_age]
        depth_edges = np.linspace(0.0, 1.01, self.depth_bins + 1)

        def bin_id(depth):
            return int(np.clip(np.digitize([depth], depth_edges)[0] - 1, 0, self.depth_bins - 1))

        det_bins = {}
        for idx in high:
            det_bins.setdefault(bin_id(detections[idx]["pseudo_depth"]), []).append(idx)
        track_bins = {}
        for idx in active_indices:
            track_bins.setdefault(bin_id(self.tracks[idx].pseudo_depth), []).append(idx)

        matches1 = []
        unmatched_tracks = set(active_indices)
        unmatched_high = set(high)
        for b in range(self.depth_bins):
            neighbor_tracks = []
            neighbor_dets = []
            for nb in [b - 1, b, b + 1]:
                if 0 <= nb < self.depth_bins:
                    neighbor_tracks.extend(track_bins.get(nb, []))
                    neighbor_dets.extend(det_bins.get(nb, []))
            m, ut, ud = assign_tracks(neighbor_tracks, neighbor_dets, self.tracks, detections, self.iou_high, depth_gate=self.depth_gate, lambda_depth=0.6)
            matches1.extend(m)
            unmatched_tracks.difference_update({x for x, _ in m})
            unmatched_high.difference_update({y for _, y in m})
        fallback_matches, remaining_tracks, remaining_high = assign_tracks(sorted(unmatched_tracks), sorted(unmatched_high), self.tracks, detections, self.iou_high)
        matches1.extend(fallback_matches)
        unmatched_tracks = remaining_tracks
        unmatched_high = remaining_high
        for tidx, didx in matches1:
            det = detections[didx]
            self.tracks[tidx].update(np.array(det["bbox"]), det["score"], frame_idx, det["pseudo_depth"])

        low_det_bins = {}
        for idx in low:
            low_det_bins.setdefault(bin_id(detections[idx]["pseudo_depth"]), []).append(idx)
        matches2 = []
        unmatched_low = set(low)
        active_unmatched = list(unmatched_tracks)
        track_bins2 = {}
        for idx in active_unmatched:
            track_bins2.setdefault(bin_id(self.tracks[idx].pseudo_depth), []).append(idx)
        for b in range(self.depth_bins):
            neighbor_tracks = []
            neighbor_dets = []
            for nb in [b - 1, b, b + 1]:
                if 0 <= nb < self.depth_bins:
                    neighbor_tracks.extend(track_bins2.get(nb, []))
                    neighbor_dets.extend(low_det_bins.get(nb, []))
            m, ut, ud = assign_tracks(neighbor_tracks, neighbor_dets, self.tracks, detections, self.iou_low, depth_gate=self.depth_gate * 1.2, lambda_depth=0.4)
            matches2.extend(m)
            unmatched_tracks = [t for t in unmatched_tracks if t not in {x for x, _ in m}]
            unmatched_low.difference_update({y for _, y in m})
        for tidx, didx in matches2:
            det = detections[didx]
            self.tracks[tidx].update(np.array(det["bbox"]), det["score"], frame_idx, det["pseudo_depth"])

        matched_track_indices = {tidx for tidx, _ in matches1 + matches2}
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_track_indices:
                tr.mark_missed()
        for didx in unmatched_high:
            det = detections[didx]
            track = Track(self.next_id, np.array(det["bbox"], dtype=float), det["score"], frame_idx)
            track.update(np.array(det["bbox"], dtype=float), det["score"], frame_idx, det["pseudo_depth"])
            self.tracks.append(track)
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.age <= self.max_age]
        frame_output = []
        for tr in self.tracks:
            if tr.last_frame == frame_idx:
                frame_output.append({"track_id": tr.track_id, "bbox": tr.bbox.tolist(), "score": tr.score, "pseudo_depth": tr.pseudo_depth})
        self.frame_logs.append({
            "frame": frame_idx,
            "num_detections": len(detections),
            "num_tracks_output": len(frame_output),
            "matches_high": len(matches1),
            "matches_low": len(matches2),
            "new_tracks": len(unmatched_high),
        })
        return frame_output


class DepthAblationTracker(ByteTrackerLike):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def evaluate_mot(sequence, tracker_outputs, name, crowded_frames):
    acc_all = mm.MOTAccumulator(auto_id=False)
    acc_crowded = mm.MOTAccumulator(auto_id=False)
    frame_metrics = []
    for frame in sequence:
        fidx = frame["frame"]
        gt_ids = frame["gt_ids"]
        gt_boxes = [xyxy_to_tlwh(box) for box in frame["gt_bboxes"]]
        preds = tracker_outputs[fidx]
        pred_ids = [p["track_id"] for p in preds]
        pred_boxes = [xyxy_to_tlwh(p["bbox"]) for p in preds]
        dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc_all.update(gt_ids, pred_ids, dists, frameid=fidx)
        if fidx in crowded_frames:
            acc_crowded.update(gt_ids, pred_ids, dists, frameid=fidx)
        events = acc_all.mot_events.loc[acc_all.mot_events.index.get_level_values(0) == fidx] if acc_all.mot_events is not None else None
        idsw = int((events.Type == "SWITCH").sum()) if events is not None and not events.empty else 0
        matches = int((events.Type == "MATCH").sum()) if events is not None and not events.empty else 0
        misses = int((events.Type == "MISS").sum()) if events is not None and not events.empty else 0
        fps = int((events.Type == "FP").sum()) if events is not None and not events.empty else 0
        frame_metrics.append({"frame": fidx, "tracker": name, "matches": matches, "misses": misses, "false_positives": fps, "id_switches": idsw, "crowded": int(fidx in crowded_frames)})
    mh = mm.metrics.create()
    summary_all = mh.compute(acc_all, metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations"], name=name)
    summary_crowded = mh.compute(acc_crowded, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses", "num_fragmentations"], name=name)
    all_metrics = summary_all.loc[name].to_dict()
    crowded_metrics = summary_crowded.loc[name].to_dict() if len(summary_crowded) else {}
    return all_metrics, crowded_metrics, pd.DataFrame(frame_metrics)


def run_tracker(sequence, tracker):
    outputs = {}
    for frame in sequence:
        outputs[frame["frame"]] = tracker.step([dict(det) for det in frame["detections"]], frame["frame"])
    return outputs, pd.DataFrame(tracker.frame_logs)


def summarize_data(sequence):
    frame_rows = []
    all_scores = []
    overlap_rows = []
    for frame in sequence:
        gt_boxes = frame["gt_bboxes"]
        det_boxes = [d["bbox"] for d in frame["detections"]]
        overlaps = pairwise_iou(gt_boxes)
        det_overlaps = pairwise_iou(det_boxes)
        crowded_score = sum(v > 0.2 for v in overlaps)
        frame_rows.append({
            "frame": frame["frame"],
            "num_gt": len(gt_boxes),
            "num_detections": len(frame["detections"]),
            "mean_gt_iou": float(np.mean(overlaps) if overlaps else 0.0),
            "max_gt_iou": float(np.max(overlaps) if overlaps else 0.0),
            "mean_det_iou": float(np.mean(det_overlaps) if det_overlaps else 0.0),
            "crowding_pairs_gt_iou_gt_0.2": crowded_score,
        })
        for det in frame["detections"]:
            all_scores.append(det["score"])
        overlap_rows.extend([{ "frame": frame["frame"], "pair_iou": float(v)} for v in overlaps])
    frame_stats = pd.DataFrame(frame_rows)
    overlap_df = pd.DataFrame(overlap_rows, columns=["frame", "pair_iou"])
    crowded_frames = frame_stats[frame_stats["crowding_pairs_gt_iou_gt_0.2"] >= frame_stats["crowding_pairs_gt_iou_gt_0.2"].quantile(0.75)]["frame"].tolist()
    summary = {
        "num_frames": int(len(sequence)),
        "num_unique_gt_ids": int(len({gid for frame in sequence for gid in frame["gt_ids"]})),
        "gt_boxes_per_frame": int(len(sequence[0]["gt_bboxes"])),
        "mean_detections_per_frame": float(frame_stats["num_detections"].mean()),
        "detection_rate_proxy": float(frame_stats["num_detections"].mean() / frame_stats["num_gt"].mean()),
        "mean_detection_score": float(np.mean(all_scores)),
        "crowded_frame_count": int(len(crowded_frames)),
        "crowded_frames": crowded_frames,
    }
    return summary, frame_stats, overlap_df, crowded_frames


def threshold_sweep(sequence, crowded_frames):
    rows = []
    configs = [(0.5, 0.1), (0.6, 0.1), (0.7, 0.1), (0.6, 0.2)]
    for high, low in configs:
        tracker = ByteTrackerLike(high_thresh=high, low_thresh=low)
        outputs, _ = run_tracker(sequence, tracker)
        overall, crowded, _ = evaluate_mot(sequence, outputs, f"high{high}_low{low}", crowded_frames)
        rows.append({
            "high_thresh": high,
            "low_thresh": low,
            "idf1": overall["idf1"],
            "mota": overall["mota"],
            "num_switches": overall["num_switches"],
            "crowded_idf1": crowded.get("idf1", float("nan")),
            "crowded_switches": crowded.get("num_switches", float("nan")),
        })
    return pd.DataFrame(rows)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def generate_figures(frame_stats, overlap_df, metrics_df, per_frame_df, outputs_dir, report_img_dir):
    Path(report_img_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    frame_stats.plot(x="frame", y=["num_gt", "num_detections"], ax=ax1)
    ax1.set_title("Per-frame object and detection counts")
    ax1.set_ylabel("Count")
    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(overlap_df["pair_iou"], bins=30, ax=ax2)
    ax2.set_title("Distribution of GT pairwise IoU")
    ax2.set_xlabel("IoU")
    plt.tight_layout()
    plt.savefig(Path(report_img_dir) / "data_overview.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    metrics_melt = metrics_df.melt(id_vars=["method", "split"], value_vars=["idf1", "mota"], var_name="metric", value_name="value")
    sns.barplot(data=metrics_melt, x="metric", y="value", hue="method", palette="deep")
    plt.title("Overall tracking quality comparison")
    plt.tight_layout()
    plt.savefig(Path(report_img_dir) / "main_comparison.png", dpi=200)
    plt.close()

    crowded_pf = per_frame_df[per_frame_df["crowded"] == 1]
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=crowded_pf, x="frame", y="id_switches", hue="tracker", marker="o")
    plt.title("ID switches on crowded frames")
    plt.tight_layout()
    plt.savefig(Path(report_img_dir) / "occlusion_analysis.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=per_frame_df, x="frame", y="matches", hue="tracker")
    plt.title("Per-frame matches across trackers")
    plt.tight_layout()
    plt.savefig(Path(report_img_dir) / "per_frame_matches.png", dpi=200)
    plt.close()


def build_report_inputs(sequence_path, outputs_dir, report_img_dir):
    with open(sequence_path) as f:
        sequence = json.load(f)
    summary, frame_stats, overlap_df, crowded_frames = summarize_data(sequence)
    frame_stats.to_csv(Path(outputs_dir) / "frame_stats.csv", index=False)
    overlap_df.to_csv(Path(outputs_dir) / "overlap_stats.csv", index=False)
    save_json(Path(outputs_dir) / "data_summary.json", summary)

    sweep_df = threshold_sweep(sequence, crowded_frames)
    sweep_df.to_csv(Path(outputs_dir) / "baseline_threshold_sweep.csv", index=False)

    trackers = {
        "ByteTrack-like": ByteTrackerLike(high_thresh=0.6, low_thresh=0.1, iou_high=0.3, iou_low=0.15, max_age=10),
        "Sparse-Hierarchical": SparseHierarchicalTracker(high_thresh=0.6, low_thresh=0.1, iou_high=0.25, iou_low=0.12, max_age=12, depth_bins=4, depth_gate=0.12),
        "Ablation-NoHierarchy": DepthAblationTracker(high_thresh=0.6, low_thresh=0.1, iou_high=0.25, iou_low=0.12, max_age=12),
    }
    comparison_rows = []
    per_frame_tables = []
    for name, tracker in trackers.items():
        outputs, log_df = run_tracker(sequence, tracker)
        out_name = name.lower().replace("-", "_").replace(" ", "_")
        save_json(Path(outputs_dir) / f"{out_name}_tracks.json", outputs)
        log_df.to_csv(Path(outputs_dir) / f"{out_name}_association_log.csv", index=False)
        overall, crowded, per_frame = evaluate_mot(sequence, outputs, name, crowded_frames)
        save_json(Path(outputs_dir) / f"{out_name}_metrics.json", {"overall": overall, "crowded": crowded})
        overall_row = {"method": name, "split": "overall", **overall}
        crowded_row = {"method": name, "split": "crowded", **crowded}
        comparison_rows.extend([overall_row, crowded_row])
        per_frame_tables.append(per_frame)
    metrics_df = pd.DataFrame(comparison_rows)
    metrics_df.to_csv(Path(outputs_dir) / "comparison_table.csv", index=False)
    per_frame_df = pd.concat(per_frame_tables, ignore_index=True)
    per_frame_df.to_csv(Path(outputs_dir) / "per_frame_metrics.csv", index=False)
    generate_figures(frame_stats, overlap_df, metrics_df[metrics_df["split"] == "overall"], per_frame_df, outputs_dir, report_img_dir)
    return summary, frame_stats, sweep_df, metrics_df, per_frame_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all")
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    report_img_dir = Path("report/images")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_img_dir.mkdir(parents=True, exist_ok=True)

    build_report_inputs(args.data, outputs_dir, report_img_dir)
    print("Completed stage:", args.stage)


if __name__ == "__main__":
    main()
