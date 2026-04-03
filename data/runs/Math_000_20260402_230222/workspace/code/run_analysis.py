#!/usr/bin/env python3
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'simulated_sequence.json'
OUTPUT_DIR = ROOT / 'outputs'
IMAGE_DIR = ROOT / 'report' / 'images'


@dataclass
class Track:
    track_id: int
    boxes: Dict[int, List[float]] = field(default_factory=dict)
    det_scores: Dict[int, float] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    last_frame: int = -1
    velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def last_box(self) -> List[float]:
        return self.boxes[self.last_frame]

    def predict_box(self, frame_idx: int) -> List[float]:
        if self.last_frame < 0:
            raise ValueError('Track has no observations')
        dt = max(0, frame_idx - self.last_frame)
        last = np.array(self.last_box(), dtype=float)
        vel = np.array(self.velocity, dtype=float)
        pred = last + dt * vel
        pred[0::2] = np.clip(pred[0::2], 0, 480)
        pred[1::2] = np.clip(pred[1::2], 0, 640)
        if pred[2] < pred[0]:
            pred[2] = pred[0] + 1
        if pred[3] < pred[1]:
            pred[3] = pred[1] + 1
        return pred.tolist()

    def update(self, frame_idx: int, bbox: List[float], score: float) -> None:
        bbox_arr = np.array(bbox, dtype=float)
        if self.last_frame >= 0:
            prev = np.array(self.last_box(), dtype=float)
            dt = max(1, frame_idx - self.last_frame)
            new_vel = (bbox_arr - prev) / dt
            self.velocity = tuple(0.65 * np.array(self.velocity) + 0.35 * new_vel)
        self.boxes[frame_idx] = bbox_arr.tolist()
        self.det_scores[frame_idx] = float(score)
        self.last_frame = frame_idx
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_sequence() -> List[dict]:
    with DATA_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def bbox_center(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_wh(b: List[float]) -> Tuple[float, float]:
    return (max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1]))


def bbox_area(b: List[float]) -> float:
    w, h = bbox_wh(b)
    return w * h


def bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / max(union, 1e-6)


def pseudo_depth(b: List[float]) -> float:
    _, cy = bbox_center(b)
    w, h = bbox_wh(b)
    bottom = b[3] / 640.0
    area_term = np.clip(math.log1p(w * h) / 10.0, 0, 1)
    height_term = np.clip(h / 220.0, 0, 1)
    return float(0.55 * bottom + 0.25 * area_term + 0.20 * height_term)


def track_detection_cost(track: Track, det_bbox: List[float], frame_idx: int) -> float:
    pred = track.predict_box(frame_idx)
    tcx, tcy = bbox_center(pred)
    dcx, dcy = bbox_center(det_bbox)
    tw, th = bbox_wh(pred)
    dw, dh = bbox_wh(det_bbox)
    norm = math.hypot(480, 640)
    dist = math.hypot(tcx - dcx, tcy - dcy) / norm
    size_delta = abs(math.log(dw / tw)) + abs(math.log(dh / th))
    depth_delta = abs(pseudo_depth(pred) - pseudo_depth(det_bbox))
    iou = bbox_iou(pred, det_bbox)
    stale_penalty = min(track.misses, 5) * 0.03
    return dist + 0.18 * size_delta + 0.35 * depth_delta + 0.6 * (1.0 - iou) + stale_penalty


def greedy_match(tracks: List[Track], detections: List[dict], frame_idx: int,
                 dist_gate: float = 0.55, cost_gate: float = 1.35) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    candidates = []
    for ti, track in enumerate(tracks):
        pred = track.predict_box(frame_idx)
        pcx, pcy = bbox_center(pred)
        for di, det in enumerate(detections):
            det_bbox = det['bbox']
            dcx, dcy = bbox_center(det_bbox)
            center_dist = math.hypot(pcx - dcx, pcy - dcy) / math.hypot(480, 640)
            if center_dist > dist_gate:
                continue
            cost = track_detection_cost(track, det_bbox, frame_idx)
            if cost <= cost_gate:
                candidates.append((cost, ti, di))
    candidates.sort(key=lambda x: x[0])
    matched_tracks = set()
    matched_dets = set()
    matches = []
    for cost, ti, di in candidates:
        if ti in matched_tracks or di in matched_dets:
            continue
        matches.append((ti, di))
        matched_tracks.add(ti)
        matched_dets.add(di)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def hierarchical_match(tracks: List[Track], detections: List[dict], frame_idx: int, depth_bins: int = 4) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    det_depths = np.array([pseudo_depth(d['bbox']) for d in detections])
    quantiles = np.quantile(det_depths, np.linspace(0, 1, depth_bins + 1))
    quantiles[0] -= 1e-6
    quantiles[-1] += 1e-6

    det_bin_map = {}
    for di, depth in enumerate(det_depths):
        bi = int(np.searchsorted(quantiles[1:-1], depth, side='right'))
        det_bin_map.setdefault(bi, []).append(di)

    track_depths = [pseudo_depth(t.predict_box(frame_idx)) for t in tracks]
    track_bin_map = {}
    for ti, depth in enumerate(track_depths):
        bi = int(np.searchsorted(quantiles[1:-1], depth, side='right'))
        track_bin_map.setdefault(bi, []).append(ti)

    matches = []
    globally_matched_tracks = set()
    globally_matched_dets = set()

    for bi in range(depth_bins):
        sub_track_ids = [ti for ti in track_bin_map.get(bi, []) if ti not in globally_matched_tracks]
        sub_det_ids = [di for di in det_bin_map.get(bi, []) if di not in globally_matched_dets]
        sub_tracks = [tracks[ti] for ti in sub_track_ids]
        sub_dets = [detections[di] for di in sub_det_ids]
        local_matches, _, _ = greedy_match(sub_tracks, sub_dets, frame_idx, dist_gate=0.38, cost_gate=1.0)
        for lti, ldi in local_matches:
            ti = sub_track_ids[lti]
            di = sub_det_ids[ldi]
            matches.append((ti, di))
            globally_matched_tracks.add(ti)
            globally_matched_dets.add(di)

    remaining_tracks = [i for i in range(len(tracks)) if i not in globally_matched_tracks]
    remaining_dets = [i for i in range(len(detections)) if i not in globally_matched_dets]
    if remaining_tracks and remaining_dets:
        rem_tracks = [tracks[i] for i in remaining_tracks]
        rem_dets = [detections[i] for i in remaining_dets]
        local_matches, _, _ = greedy_match(rem_tracks, rem_dets, frame_idx, dist_gate=0.5, cost_gate=1.2)
        for lti, ldi in local_matches:
            ti = remaining_tracks[lti]
            di = remaining_dets[ldi]
            matches.append((ti, di))
            globally_matched_tracks.add(ti)
            globally_matched_dets.add(di)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in globally_matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in globally_matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def run_tracker(sequence: List[dict], mode: str) -> dict:
    tracks: List[Track] = []
    all_tracks: List[Track] = []
    next_track_id = 0
    frame_assignments = []
    max_misses = 8 if mode == 'sparse_hierarchical' else 6
    creation_score = 0.16 if mode == 'sparse_hierarchical' else 0.18

    for frame_data in sequence:
        frame_idx = int(frame_data['frame'])
        detections = sorted(frame_data['detections'], key=lambda d: d['score'], reverse=True)

        if mode == 'sparse_hierarchical':
            matches, unmatched_tracks, unmatched_dets = hierarchical_match(tracks, detections, frame_idx)
        else:
            matches, unmatched_tracks, unmatched_dets = greedy_match(tracks, detections, frame_idx)

        assignment_rows = []
        for ti, di in matches:
            det = detections[di]
            tracks[ti].update(frame_idx, det['bbox'], det['score'])
            assignment_rows.append({
                'frame': frame_idx,
                'track_id': tracks[ti].track_id,
                'bbox': det['bbox'],
                'score': det['score'],
                'gt_id': det.get('gt_id', None),
                'source': 'matched_detection'
            })

        for ti in unmatched_tracks:
            tracks[ti].mark_missed()

        for di in unmatched_dets:
            det = detections[di]
            if det['score'] < creation_score:
                continue
            track = Track(track_id=next_track_id)
            next_track_id += 1
            track.update(frame_idx, det['bbox'], det['score'])
            tracks.append(track)
            all_tracks.append(track)
            assignment_rows.append({
                'frame': frame_idx,
                'track_id': track.track_id,
                'bbox': det['bbox'],
                'score': det['score'],
                'gt_id': det.get('gt_id', None),
                'source': 'new_track'
            })

        tracks = [t for t in tracks if t.misses <= max_misses]
        frame_assignments.extend(assignment_rows)

    trajectory_rows = []
    dense_trajectories: Dict[int, Dict[int, List[float]]] = {}
    for t in all_tracks:
        if len(t.boxes) < 2:
            continue
        frames = sorted(t.boxes)
        dense = {}
        for i, f in enumerate(frames):
            dense[f] = t.boxes[f]
            if i == len(frames) - 1:
                continue
            f_next = frames[i + 1]
            b1 = np.array(t.boxes[f], dtype=float)
            b2 = np.array(t.boxes[f_next], dtype=float)
            gap = f_next - f
            if gap <= 1 or gap > 5:
                continue
            for step in range(1, gap):
                alpha = step / gap
                dense[f + step] = ((1 - alpha) * b1 + alpha * b2).tolist()
        dense_trajectories[t.track_id] = {int(k): v for k, v in sorted(dense.items())}
        gt_counts = {}
        for f, score in t.det_scores.items():
            pass
        trajectory_rows.append({
            'track_id': t.track_id,
            'start_frame': min(frames),
            'end_frame': max(frames),
            'observed_length': len(frames),
            'dense_length': len(dense_trajectories[t.track_id]),
            'mean_score': float(np.mean([t.det_scores[f] for f in frames])),
            'mean_depth': float(np.mean([pseudo_depth(t.boxes[f]) for f in frames]))
        })

    return {
        'mode': mode,
        'frame_assignments': frame_assignments,
        'trajectory_summary': trajectory_rows,
        'dense_trajectories': dense_trajectories,
        'num_tracks': len(trajectory_rows)
    }


def summarize_data(sequence: List[dict]) -> dict:
    det_counts = [len(fr['detections']) for fr in sequence]
    gt_counts = [len(fr['gt_ids']) for fr in sequence]
    scores = [d['score'] for fr in sequence for d in fr['detections']]
    depths = [pseudo_depth(d['bbox']) for fr in sequence for d in fr['detections']]
    overlaps = []
    for fr in sequence[:20]:
        boxes = [d['bbox'] for d in fr['detections']]
        for i in range(min(len(boxes), 90)):
            for j in range(i + 1, min(len(boxes), 90)):
                overlaps.append(bbox_iou(boxes[i], boxes[j]))
    return {
        'num_frames': len(sequence),
        'unique_gt_ids': len({gid for fr in sequence for gid in fr['gt_ids']}),
        'mean_gt_per_frame': float(np.mean(gt_counts)),
        'mean_detections_per_frame': float(np.mean(det_counts)),
        'min_detections_per_frame': int(np.min(det_counts)),
        'max_detections_per_frame': int(np.max(det_counts)),
        'estimated_detection_rate': float(np.sum(det_counts) / np.sum(gt_counts)),
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'pseudo_depth_mean': float(np.mean(depths)),
        'pseudo_depth_std': float(np.std(depths)),
        'pairwise_iou_mean_sample': float(np.mean(overlaps)),
        'pairwise_iou_gt005_ratio_sample': float(np.mean(np.array(overlaps) > 0.05)),
    }


def evaluate_tracker(sequence: List[dict], result: dict) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    assign_df = pd.DataFrame(result['frame_assignments'])
    if assign_df.empty:
        raise RuntimeError(f'No assignments produced for {result["mode"]}')

    pred_lookup = {}
    for track_id, frame_map in result['dense_trajectories'].items():
        for frame_idx, bbox in frame_map.items():
            pred_lookup.setdefault(int(frame_idx), []).append((int(track_id), bbox))

    frame_metrics = []
    gt_to_tracks = {}
    det_rows = []
    prev_assignment_by_gt = {}
    id_switches = 0
    observed_gt_instances = 0

    assign_by_frame = {int(k): v for k, v in assign_df.groupby('frame')}

    for frame_data in sequence:
        frame_idx = int(frame_data['frame'])
        gt_ids = frame_data['gt_ids']
        gt_boxes = frame_data['gt_bboxes']
        preds = pred_lookup.get(frame_idx, [])
        matched_gt = 0
        pred_used = set()
        gt_assignment_this_frame = {}

        for gt_id, gt_box in zip(gt_ids, gt_boxes):
            observed_gt_instances += 1
            best = None
            best_iou = 0.0
            for pred_idx, (track_id, pred_box) in enumerate(preds):
                if pred_idx in pred_used:
                    continue
                iou = bbox_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best = (pred_idx, track_id)
            if best is not None and best_iou >= 0.5:
                matched_gt += 1
                pred_used.add(best[0])
                gt_assignment_this_frame[int(gt_id)] = int(best[1])

        if frame_idx in assign_by_frame:
            frame_assignments = assign_by_frame[frame_idx]
            for _, row in frame_assignments.iterrows():
                gt_id = row['gt_id']
                track_id = int(row['track_id'])
                if pd.isna(gt_id):
                    continue
                gt_id = int(gt_id)
                gt_to_tracks.setdefault(gt_id, []).append(track_id)
                det_rows.append({'frame': frame_idx, 'gt_id': gt_id, 'track_id': track_id, 'score': row['score']})

        for gt_id, track_id in gt_assignment_this_frame.items():
            prev = prev_assignment_by_gt.get(gt_id)
            if prev is not None and prev != track_id:
                id_switches += 1
            prev_assignment_by_gt[gt_id] = track_id

        precision = len(pred_used) / max(len(preds), 1)
        recall = matched_gt / max(len(gt_ids), 1)
        frame_metrics.append({
            'mode': result['mode'],
            'frame': frame_idx,
            'gt_count': len(gt_ids),
            'pred_count': len(preds),
            'matched_gt_iou50': matched_gt,
            'frame_precision_iou50': precision,
            'frame_recall_iou50': recall,
        })

    det_df = pd.DataFrame(det_rows)

    track_purity = []
    if not det_df.empty:
        for track_id, grp in det_df.groupby('track_id'):
            dominant = grp['gt_id'].value_counts().iloc[0]
            track_purity.append(dominant / len(grp))

    gt_fragmentation = []
    gt_recall = []
    for gt_id in sorted(gt_to_tracks):
        seq = gt_to_tracks[gt_id]
        gt_fragmentation.append(len(set(seq)))
        gt_recall.append(len(seq) / len(sequence))

    summary = {
        'mode': result['mode'],
        'num_tracks': int(result['num_tracks']),
        'mean_track_purity': float(np.mean(track_purity)) if track_purity else 0.0,
        'median_track_purity': float(np.median(track_purity)) if track_purity else 0.0,
        'mean_gt_fragmentation': float(np.mean(gt_fragmentation)) if gt_fragmentation else 0.0,
        'median_gt_fragmentation': float(np.median(gt_fragmentation)) if gt_fragmentation else 0.0,
        'mean_gt_detection_recall_proxy': float(np.mean(gt_recall)) if gt_recall else 0.0,
        'id_switches_iou50': int(id_switches),
        'mean_frame_recall_iou50': float(pd.DataFrame(frame_metrics)['frame_recall_iou50'].mean()),
        'mean_frame_precision_iou50': float(pd.DataFrame(frame_metrics)['frame_precision_iou50'].mean()),
    }
    return summary, pd.DataFrame(frame_metrics), det_df


def save_json(path: Path, payload: dict) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def make_figures(sequence: List[dict], data_summary: dict, evaluation_df: pd.DataFrame,
                 frame_metrics_df: pd.DataFrame, sparse_result: dict, sparse_det_df: pd.DataFrame) -> None:
    sns.set_theme(style='whitegrid')

    # Figure 1: data overview
    frames = [fr['frame'] for fr in sequence]
    det_counts = [len(fr['detections']) for fr in sequence]
    scores = [d['score'] for fr in sequence for d in fr['detections']]
    depths = [pseudo_depth(d['bbox']) for fr in sequence for d in fr['detections']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(frames, det_counts, color='#1f77b4', lw=2)
    axes[0].axhline(data_summary['mean_detections_per_frame'], color='black', ls='--', lw=1)
    axes[0].set_title('Detections per frame')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Count')

    axes[1].hist(scores, bins=30, color='#ff7f0e', alpha=0.85)
    axes[1].set_title('Detection score distribution')
    axes[1].set_xlabel('Score')

    axes[2].hist(depths, bins=30, color='#2ca02c', alpha=0.85)
    axes[2].set_title('Pseudo-depth distribution')
    axes[2].set_xlabel('Pseudo-depth')
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'data_overview.png', dpi=180)
    plt.close(fig)

    # Figure 2: comparison metrics
    metric_cols = [
        'mean_frame_recall_iou50',
        'mean_frame_precision_iou50',
        'mean_track_purity',
        'mean_gt_detection_recall_proxy'
    ]
    plot_df = evaluation_df.melt(id_vars=['mode'], value_vars=metric_cols, var_name='metric', value_name='value')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x='metric', y='value', hue='mode', ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    ax.set_title('Baseline vs sparse hierarchical tracking')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'tracking_comparison_metrics.png', dpi=180)
    plt.close(fig)

    # Figure 3: frame-level recall
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=frame_metrics_df, x='frame', y='frame_recall_iou50', hue='mode', ax=ax, lw=2)
    ax.set_ylim(0, 1.0)
    ax.set_title('Frame-level IoU@0.5 recall')
    ax.set_ylabel('Recall')
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'frame_level_recall.png', dpi=180)
    plt.close(fig)

    # Figure 4: sample trajectories for sparse tracker
    fig, ax = plt.subplots(figsize=(7, 9))
    summaries = pd.DataFrame(sparse_result['trajectory_summary']).sort_values('dense_length', ascending=False).head(12)
    cmap = plt.get_cmap('tab20')
    for idx, row in enumerate(summaries.itertuples(index=False)):
        track_id = int(row.track_id)
        traj = sparse_result['dense_trajectories'][track_id]
        centers = np.array([bbox_center(traj[f]) for f in sorted(traj)])
        ax.plot(centers[:, 0], centers[:, 1], marker='o', ms=2, lw=1.8, color=cmap(idx % 20), label=f'T{track_id}')
    ax.invert_yaxis()
    ax.set_xlim(0, 480)
    ax.set_ylim(640, 0)
    ax.set_title('Representative sparse-tracker trajectories')
    ax.set_xlabel('x center')
    ax.set_ylabel('y center')
    ax.legend(ncol=2, fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'trajectory_examples.png', dpi=180)
    plt.close(fig)

    # Figure 5: gt-track consistency heatmap for sparse tracker
    if not sparse_det_df.empty:
        top_gt = sparse_det_df['gt_id'].value_counts().head(15).index.tolist()
        top_tracks = sparse_det_df['track_id'].value_counts().head(15).index.tolist()
        ctab = pd.crosstab(
            sparse_det_df[sparse_det_df['gt_id'].isin(top_gt)]['gt_id'],
            sparse_det_df[sparse_det_df['track_id'].isin(top_tracks)]['track_id']
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(ctab, cmap='mako', ax=ax)
        ax.set_title('Sparse tracker identity consistency (top GT / track counts)')
        ax.set_xlabel('Predicted track ID')
        ax.set_ylabel('Ground-truth ID')
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / 'id_consistency_heatmap.png', dpi=180)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    sequence = load_sequence()

    data_summary = summarize_data(sequence)
    save_json(OUTPUT_DIR / 'data_summary.json', data_summary)

    baseline_result = run_tracker(sequence, mode='baseline_global')
    sparse_result = run_tracker(sequence, mode='sparse_hierarchical')

    save_json(OUTPUT_DIR / 'baseline_tracking.json', baseline_result)
    save_json(OUTPUT_DIR / 'sparse_tracking.json', sparse_result)

    baseline_eval, baseline_frame_metrics, baseline_det_df = evaluate_tracker(sequence, baseline_result)
    sparse_eval, sparse_frame_metrics, sparse_det_df = evaluate_tracker(sequence, sparse_result)

    evaluation_df = pd.DataFrame([baseline_eval, sparse_eval])
    frame_metrics_df = pd.concat([baseline_frame_metrics, sparse_frame_metrics], ignore_index=True)

    evaluation_df.to_csv(OUTPUT_DIR / 'evaluation_summary.csv', index=False)
    frame_metrics_df.to_csv(OUTPUT_DIR / 'frame_level_metrics.csv', index=False)
    baseline_det_df.to_csv(OUTPUT_DIR / 'baseline_detection_assignments.csv', index=False)
    sparse_det_df.to_csv(OUTPUT_DIR / 'sparse_detection_assignments.csv', index=False)

    sparse_traj_df = pd.DataFrame(sparse_result['trajectory_summary']).sort_values('dense_length', ascending=False)
    baseline_traj_df = pd.DataFrame(baseline_result['trajectory_summary']).sort_values('dense_length', ascending=False)
    sparse_traj_df.to_csv(OUTPUT_DIR / 'sparse_trajectory_summary.csv', index=False)
    baseline_traj_df.to_csv(OUTPUT_DIR / 'baseline_trajectory_summary.csv', index=False)

    make_figures(sequence, data_summary, evaluation_df, frame_metrics_df, sparse_result, sparse_det_df)

    combined_summary = {
        'data_summary': data_summary,
        'evaluation': [baseline_eval, sparse_eval],
        'generated_files': {
            'outputs': [
                'outputs/data_summary.json',
                'outputs/baseline_tracking.json',
                'outputs/sparse_tracking.json',
                'outputs/evaluation_summary.csv',
                'outputs/frame_level_metrics.csv',
                'outputs/baseline_detection_assignments.csv',
                'outputs/sparse_detection_assignments.csv',
                'outputs/baseline_trajectory_summary.csv',
                'outputs/sparse_trajectory_summary.csv'
            ],
            'figures': [
                'report/images/data_overview.png',
                'report/images/tracking_comparison_metrics.png',
                'report/images/frame_level_recall.png',
                'report/images/trajectory_examples.png',
                'report/images/id_consistency_heatmap.png'
            ]
        }
    }
    save_json(OUTPUT_DIR / 'analysis_summary.json', combined_summary)
    print('Analysis complete. Outputs written to outputs/ and report/images/.')


if __name__ == '__main__':
    main()
