import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from trackers import Detection, HierarchicalSparseTracker, ByteLikeTracker


DATA_PATH = Path("data/simulated_sequence.json")
OUT_DIR = Path("outputs")
FIG_DIR = Path("report/images")
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


def load_sequence() -> List[Dict]:
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def build_detections_per_frame(sequence: List[Dict]):
    per_frame = {}
    for frame_entry in sequence:
        f = frame_entry["frame"]
        per_frame[f] = []
        for det in frame_entry["detections"]:
            per_frame[f].append(
                Detection(
                    frame=f,
                    bbox=np.array(det["bbox"], dtype=float),
                    score=float(det["score"]),
                    gt_id=det.get("gt_id"),
                )
            )
    return per_frame


def build_gt_tracks(sequence: List[Dict]):
    gt_tracks: Dict[int, List[Detection]] = {}
    for frame_entry in sequence:
        f = frame_entry["frame"]
        for bbox, gid in zip(frame_entry["gt_bboxes"], frame_entry["gt_ids"]):
            det = Detection(frame=f, bbox=np.array(bbox, dtype=float), score=1.0, gt_id=int(gid))
            gt_tracks.setdefault(int(gid), []).append(det)
    return gt_tracks


def tracks_to_dataframe(tracks, algo_name: str) -> pd.DataFrame:
    rows = []
    for t in tracks:
        for d in t.detections:
            rows.append(
                {
                    "algo": algo_name,
                    "track_id": t.track_id,
                    "frame": d.frame,
                    "x1": d.bbox[0],
                    "y1": d.bbox[1],
                    "x2": d.bbox[2],
                    "y2": d.bbox[3],
                    "score": d.score,
                    "gt_id": d.gt_id,
                }
            )
    return pd.DataFrame(rows)


def mot_metrics(pred_df: pd.DataFrame, gt_tracks: Dict[int, List[Detection]], iou_thresh: float = 0.5):
    """Simple frame-wise MOT metrics using 1–1 matching via Hungarian.

    Returns IDF1-like association accuracy and basic counts.
    """

    from scipy.optimize import linear_sum_assignment

    # Build GT dataframe
    gt_rows = []
    for gid, dets in gt_tracks.items():
        for d in dets:
            gt_rows.append({"gt_id": gid, "frame": d.frame, "bbox": d.bbox})
    gt_df = pd.DataFrame(gt_rows)

    total_matches = 0
    total_gt = 0
    total_pred = 0
    id_switches = 0

    last_match_for_gt: Dict[int, int] = {}

    for frame in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        total_gt += len(gt_frame)
        total_pred += len(pred_frame)

        if len(gt_frame) == 0 or len(pred_frame) == 0:
            continue

        cost = np.ones((len(gt_frame), len(pred_frame)), dtype=float)
        gt_boxes = gt_frame["bbox"].tolist()
        pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values
        for i, gbox in enumerate(gt_boxes):
            for j, pbox in enumerate(pred_boxes):
                xa1, ya1, xa2, ya2 = gbox
                xb1, yb1, xb2, yb2 = pbox
                inter_x1 = max(xa1, xb1)
                inter_y1 = max(ya1, yb1)
                inter_x2 = min(xa2, xb2)
                inter_y2 = min(ya2, yb2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                if inter <= 0:
                    continue
                area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
                area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
                iou_val = inter / (area_a + area_b - inter + 1e-9)
                cost[i, j] = 1.0 - iou_val

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] < iou_thresh:
                continue
            gid = int(gt_frame.iloc[r]["gt_id"])
            tid = int(pred_frame.iloc[c]["track_id"])
            # ID switch detection
            if gid in last_match_for_gt and last_match_for_gt[gid] != tid:
                id_switches += 1
            last_match_for_gt[gid] = tid
            total_matches += 1

    recall = total_matches / total_gt if total_gt > 0 else 0.0
    precision = total_matches / total_pred if total_pred > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "matches": total_matches,
        "gt": total_gt,
        "pred": total_pred,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "id_switches": id_switches,
    }


def plot_data_overview(sequence: List[Dict]):
    frames = []
    num_gt = []
    num_det = []
    occ_counts = []
    for entry in sequence:
        frames.append(entry["frame"])
        num_gt.append(len(entry["gt_ids"]))
        num_det.append(len(entry["detections"]))
        occ_counts.append(len([d for d in entry["detections"] if d.get("occluded", False)]))

    df = pd.DataFrame({
        "frame": frames,
        "#GT": num_gt,
        "#Detections": num_det,
        "#OccludedDetections": occ_counts,
    })

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df.melt("frame", var_name="type", value_name="count"), x="frame", y="count", hue="type")
    plt.title("Data Overview: counts per frame")
    plt.tight_layout()
    fig_path = FIG_DIR / "data_overview_counts.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def plot_track_statistics(df: pd.DataFrame, algo_name: str):
    # Track length distribution
    lengths = df.groupby("track_id")["frame"].nunique().reset_index(name="length")
    plt.figure(figsize=(5, 4))
    sns.histplot(lengths["length"], bins=20, kde=False)
    plt.xlabel("Track length (frames)")
    plt.ylabel("Count")
    plt.title(f"Track length distribution – {algo_name}")
    plt.tight_layout()
    fig_path_len = FIG_DIR / f"track_length_{algo_name}.png"
    plt.savefig(fig_path_len, dpi=200)
    plt.close()

    # Per-frame target count
    per_frame = df.groupby("frame")["track_id"].nunique().reset_index(name="active_tracks")
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=per_frame, x="frame", y="active_tracks")
    plt.title(f"Active tracks per frame – {algo_name}")
    plt.tight_layout()
    fig_path_cnt = FIG_DIR / f"active_tracks_{algo_name}.png"
    plt.savefig(fig_path_cnt, dpi=200)
    plt.close()

    return fig_path_len, fig_path_cnt


def plot_metric_comparison(byte_metrics: Dict, sparse_metrics: Dict):
    df = pd.DataFrame([
        {"algo": "ByteLike", "metric": "F1", "value": byte_metrics["f1"]},
        {"algo": "SparseHier", "metric": "F1", "value": sparse_metrics["f1"]},
        {"algo": "ByteLike", "metric": "ID Switches", "value": byte_metrics["id_switches"]},
        {"algo": "SparseHier", "metric": "ID Switches", "value": sparse_metrics["id_switches"]},
    ])

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df[df["metric"] == "F1"], x="algo", y="value")
    plt.title("F1 comparison")
    plt.tight_layout()
    f1_path = FIG_DIR / "metric_f1_comparison.png"
    plt.savefig(f1_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df[df["metric"] == "ID Switches"], x="algo", y="value")
    plt.title("ID switch comparison (lower is better)")
    plt.tight_layout()
    ids_path = FIG_DIR / "metric_ids_comparison.png"
    plt.savefig(ids_path, dpi=200)
    plt.close()

    return f1_path, ids_path


def main():
    sequence = load_sequence()
    dets_per_frame = build_detections_per_frame(sequence)
    gt_tracks = build_gt_tracks(sequence)

    # Data overview figure
    data_overview_path = plot_data_overview(sequence)

    # Run trackers
    byte_tracker = ByteLikeTracker()
    sparse_tracker = HierarchicalSparseTracker()

    for frame in sorted(dets_per_frame.keys()):
        byte_tracker.step(frame, dets_per_frame[frame])
        sparse_tracker.step(frame, dets_per_frame[frame])

    byte_df = tracks_to_dataframe(byte_tracker.get_tracks(), "ByteLike")
    sparse_df = tracks_to_dataframe(sparse_tracker.get_tracks(), "SparseHier")

    byte_df.to_csv(OUT_DIR / "byte_tracks.csv", index=False)
    sparse_df.to_csv(OUT_DIR / "sparse_tracks.csv", index=False)

    # Track statistics plots
    byte_len_fig, byte_cnt_fig = plot_track_statistics(byte_df, "ByteLike")
    sparse_len_fig, sparse_cnt_fig = plot_track_statistics(sparse_df, "SparseHier")

    # Metrics
    byte_metrics = mot_metrics(byte_df, gt_tracks)
    sparse_metrics = mot_metrics(sparse_df, gt_tracks)

    metrics_df = pd.DataFrame([
        {"algo": "ByteLike", **byte_metrics},
        {"algo": "SparseHier", **sparse_metrics},
    ])
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)

    f1_fig, ids_fig = plot_metric_comparison(byte_metrics, sparse_metrics)

    print("Data overview figure:", data_overview_path)
    print("ByteTrack-like figures:", byte_len_fig, byte_cnt_fig)
    print("Sparse hierarchical figures:", sparse_len_fig, sparse_cnt_fig)
    print("Metric figures:", f1_fig, ids_fig)
    print("Metrics:\n", metrics_df)


if __name__ == "__main__":
    main()
