"""
Full analysis script: SparseTrack vs ByteTrack comparison.
Generates all figures for the report.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns

from tracker import (ByteTrack, SparseTrack, run_tracker,
                     compute_metrics, compute_per_frame_metrics,
                     pseudo_depth, assign_depth_layer)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH    = '../data/simulated_sequence.json'
OUTPUT_DIR   = '../outputs'
FIGURES_DIR  = '../report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Colour / style ──────────────────────────────────────────────────────────
BT_COLOR  = '#2196F3'   # blue
ST_COLOR  = '#FF5722'   # red-orange
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False})

# ══════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════

with open(DATA_PATH) as f:
    data = json.load(f)

n_frames = len(data)
print(f"Loaded {n_frames} frames")
print(f"Objects per frame (GT): {len(data[0]['gt_ids'])}")
print(f"Detections per frame (avg): {np.mean([len(f['detections']) for f in data]):.1f}")

# ══════════════════════════════════════════════════════════════════════════
#  Run trackers
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Running ByteTrack ---")
bt_results = run_tracker(ByteTrack, data, high_thresh=0.5, low_thresh=0.1,
                         iou_thresh=0.3, min_hits=1)
bt_metrics = compute_metrics(data, bt_results)
bt_pf = compute_per_frame_metrics(data, bt_results)
print("ByteTrack metrics:", bt_metrics)

print("\n--- Running SparseTrack ---")
st_results = run_tracker(SparseTrack, data, high_thresh=0.5, low_thresh=0.1,
                         iou_thresh=0.3, n_layers=3, min_hits=1)
st_metrics = compute_metrics(data, st_results)
st_pf = compute_per_frame_metrics(data, st_results)
print("SparseTrack metrics:", st_metrics)

# Save raw metrics
with open(f'{OUTPUT_DIR}/metrics.json', 'w') as f:
    json.dump({'ByteTrack': bt_metrics, 'SparseTrack': st_metrics}, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
#  Figure 1 – Dataset overview
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1a: detections per frame
det_counts  = [len(f['detections']) for f in data]
gt_counts   = [len(f['gt_ids'])     for f in data]
axes[0].plot(det_counts, color=BT_COLOR, lw=1.5, label='Detections')
axes[0].axhline(np.mean(det_counts), color=BT_COLOR, ls='--', lw=1, alpha=0.6, label=f'Mean {np.mean(det_counts):.0f}')
axes[0].axhline(gt_counts[0], color='gray', ls=':', lw=1, label=f'GT ({gt_counts[0]})')
axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Count')
axes[0].set_title('Detections per Frame'); axes[0].legend(fontsize=9)

# 1b: score distribution
all_scores = [d['score'] for f in data for d in f['detections']]
axes[1].hist(all_scores, bins=40, color=BT_COLOR, alpha=0.8, edgecolor='white', lw=0.3)
axes[1].axvline(0.5, color='red', ls='--', lw=1.5, label='High thresh 0.5')
axes[1].axvline(0.1, color='orange', ls='--', lw=1.5, label='Low thresh 0.1')
axes[1].set_xlabel('Detection Score'); axes[1].set_ylabel('Count')
axes[1].set_title('Detection Score Distribution'); axes[1].legend(fontsize=9)

# 1c: pseudo-depth distribution
all_depths = [pseudo_depth(d['bbox']) for f in data for d in f['detections']]
axes[2].hist(all_depths, bins=40, color=ST_COLOR, alpha=0.8, edgecolor='white', lw=0.3)
for edge in np.linspace(0, 1, 4)[1:-1]:
    axes[2].axvline(edge, color='navy', ls='--', lw=1.2, alpha=0.7)
axes[2].set_xlabel('Pseudo-Depth (0=near, 1=far)')
axes[2].set_ylabel('Count')
axes[2].set_title('Pseudo-Depth Distribution of Detections\n(dashed lines = SparseTrack layer boundaries)')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_dataset_overview.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 2 – Metric comparison bar chart
# ══════════════════════════════════════════════════════════════════════════

metrics_to_plot = ['MOTA', 'MOTP', 'IDF1']
bt_vals = [bt_metrics[m] for m in metrics_to_plot]
st_vals = [st_metrics[m] for m in metrics_to_plot]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart for main metrics
x = np.arange(len(metrics_to_plot))
w = 0.35
bars1 = axes[0].bar(x - w/2, bt_vals, w, label='ByteTrack', color=BT_COLOR, alpha=0.85)
bars2 = axes[0].bar(x + w/2, st_vals, w, label='SparseTrack', color=ST_COLOR, alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics_to_plot)
axes[0].set_ylabel('Score'); axes[0].set_title('Tracking Accuracy Metrics')
axes[0].legend()
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

# Error metric comparison
error_metrics = ['FP', 'FN', 'ID_switches']
bt_err = [bt_metrics[m] for m in error_metrics]
st_err = [st_metrics[m] for m in error_metrics]
x2 = np.arange(len(error_metrics))
b1 = axes[1].bar(x2 - w/2, bt_err, w, label='ByteTrack', color=BT_COLOR, alpha=0.85)
b2 = axes[1].bar(x2 + w/2, st_err, w, label='SparseTrack', color=ST_COLOR, alpha=0.85)
axes[1].set_xticks(x2); axes[1].set_xticklabels(error_metrics)
axes[1].set_ylabel('Count'); axes[1].set_title('Error Metrics')
axes[1].legend()
for bar in list(b1) + list(b2):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig2_metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_metric_comparison.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 3 – Per-frame TP/FP/FN
# ══════════════════════════════════════════════════════════════════════════

frames = list(range(n_frames))
bt_tp = [r['TP'] for r in bt_pf]
bt_fp = [r['FP'] for r in bt_pf]
bt_fn = [r['FN'] for r in bt_pf]
bt_ids = [r['IDS'] for r in bt_pf]

st_tp = [r['TP'] for r in st_pf]
st_fp = [r['FP'] for r in st_pf]
st_fn = [r['FN'] for r in st_pf]
st_ids = [r['IDS'] for r in st_pf]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')

# TP
axes[0,0].plot(frames, bt_tp, color=BT_COLOR, lw=1.2, alpha=0.5, label='ByteTrack')
axes[0,0].plot(frames, smooth(bt_tp), color=BT_COLOR, lw=2, label='BT (smoothed)')
axes[0,0].plot(frames, st_tp, color=ST_COLOR, lw=1.2, alpha=0.5, label='SparseTrack')
axes[0,0].plot(frames, smooth(st_tp), color=ST_COLOR, lw=2, label='ST (smoothed)')
axes[0,0].set_title('True Positives per Frame'); axes[0,0].set_xlabel('Frame')
axes[0,0].set_ylabel('Count'); axes[0,0].legend(fontsize=8)

# FP
axes[0,1].plot(frames, bt_fp, color=BT_COLOR, lw=1.5, label='ByteTrack')
axes[0,1].plot(frames, st_fp, color=ST_COLOR, lw=1.5, label='SparseTrack')
axes[0,1].set_title('False Positives per Frame'); axes[0,1].set_xlabel('Frame')
axes[0,1].set_ylabel('Count'); axes[0,1].legend(fontsize=8)

# FN
axes[1,0].plot(frames, bt_fn, color=BT_COLOR, lw=1.5, label='ByteTrack')
axes[1,0].plot(frames, st_fn, color=ST_COLOR, lw=1.5, label='SparseTrack')
axes[1,0].set_title('False Negatives per Frame'); axes[1,0].set_xlabel('Frame')
axes[1,0].set_ylabel('Count'); axes[1,0].legend(fontsize=8)

# Active tracks per frame
bt_nt = [r['n_tracks'] for r in bt_pf]
st_nt = [r['n_tracks'] for r in st_pf]
axes[1,1].plot(frames, bt_nt, color=BT_COLOR, lw=1.5, label='ByteTrack')
axes[1,1].plot(frames, st_nt, color=ST_COLOR, lw=1.5, label='SparseTrack')
axes[1,1].set_title('Active Tracks per Frame'); axes[1,1].set_xlabel('Frame')
axes[1,1].set_ylabel('Count'); axes[1,1].legend(fontsize=8)

plt.suptitle('Per-Frame Tracking Metrics: ByteTrack vs SparseTrack', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig3_per_frame_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_per_frame_metrics.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 4 – ID switch analysis
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Cumulative ID switches
cum_bt = np.cumsum(bt_ids)
cum_st = np.cumsum(st_ids)
axes[0].plot(frames, cum_bt, color=BT_COLOR, lw=2, label=f'ByteTrack (total={bt_metrics["ID_switches"]})')
axes[0].plot(frames, cum_st, color=ST_COLOR, lw=2, label=f'SparseTrack (total={st_metrics["ID_switches"]})')
axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Cumulative ID Switches')
axes[0].set_title('Cumulative ID Switches over Time'); axes[0].legend(fontsize=9)

# TP rate and precision per frame
bt_prec = [tp/(tp+fp+1e-9) for tp, fp in zip(bt_tp, bt_fp)]
st_prec = [tp/(tp+fp+1e-9) for tp, fp in zip(st_tp, st_fp)]
bt_rec  = [tp/(tp+fn+1e-9) for tp, fn in zip(bt_tp, bt_fn)]
st_rec  = [tp/(tp+fn+1e-9) for tp, fn in zip(st_tp, st_fn)]

axes[1].plot(frames, smooth(bt_prec, 7), color=BT_COLOR, lw=2, ls='-',  label='BT Precision')
axes[1].plot(frames, smooth(st_prec, 7), color=ST_COLOR, lw=2, ls='-',  label='ST Precision')
axes[1].plot(frames, smooth(bt_rec, 7),  color=BT_COLOR, lw=2, ls='--', label='BT Recall')
axes[1].plot(frames, smooth(st_rec, 7),  color=ST_COLOR, lw=2, ls='--', label='ST Recall')
axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Score (smoothed)')
axes[1].set_title('Per-Frame Precision and Recall (7-frame smoothed)')
axes[1].legend(fontsize=8); axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig4_id_switches.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_id_switches.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 5 – Depth layer analysis
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
n_layers = 3
layer_colors = ['#1B5E20', '#FF6F00', '#B71C1C']

# Layer sizes per frame
layer_sizes = [[], [], []]
for frame in data:
    dets = frame['detections']
    high = [d for d in dets if d['score'] >= 0.5]
    for li in range(n_layers):
        edges = np.linspace(0, 1 + 1e-9, n_layers + 1)
        cnt = sum(1 for d in high if edges[li] <= pseudo_depth(d['bbox']) < edges[li+1])
        layer_sizes[li].append(cnt)

for li, (sizes, color) in enumerate(zip(layer_sizes, layer_colors)):
    axes[0].plot(frames, smooth(sizes, 5), color=color, lw=2, label=f'Layer {li}')
axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Count (smoothed)')
axes[0].set_title('High-Conf Detections per Depth Layer'); axes[0].legend()

# Depth histogram per layer with KDE
from scipy.stats import gaussian_kde
depths_all = [pseudo_depth(d['bbox']) for f in data for d in f['detections'] if d['score'] >= 0.5]
edges = np.linspace(0, 1, n_layers + 1)
for li, color in enumerate(layer_colors):
    layer_depths = [d for d in depths_all if edges[li] <= d < edges[li+1]]
    if layer_depths:
        axes[1].hist(layer_depths, bins=15, alpha=0.5, color=color, label=f'Layer {li}', density=True)
axes[1].set_xlabel('Pseudo-Depth'); axes[1].set_ylabel('Density')
axes[1].set_title('Depth Distribution per Layer'); axes[1].legend()

# Layer imbalance (std dev across layers per frame)
imbalances = [np.std([layer_sizes[li][fr] for li in range(n_layers)]) for fr in frames]
axes[2].plot(frames, smooth(imbalances, 5), color='purple', lw=2)
axes[2].fill_between(frames, 0, smooth(imbalances, 5), alpha=0.2, color='purple')
axes[2].set_xlabel('Frame'); axes[2].set_ylabel('Std Dev across Layers')
axes[2].set_title('Depth Layer Imbalance\n(Higher = more uneven distribution)')

plt.suptitle('Pseudo-Depth Decomposition Analysis', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig5_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_depth_analysis.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 6 – Trajectory visualization for a single frame
# ══════════════════════════════════════════════════════════════════════════

SAMPLE_FRAME = 30

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
frame_data = data[SAMPLE_FRAME]
W, H = 640, 600

def plot_frame(ax, title, detections, tracks, gt_bboxes):
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    # GT boxes (grey)
    for bb in gt_bboxes[:30]:  # show only first 30 for clarity
        r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                               linewidth=0.5, edgecolor='#666666', facecolor='none', ls=':')
        ax.add_patch(r)
    # Detections
    for d in detections:
        bb = d['bbox']
        col = '#00E676' if d['score'] >= 0.5 else '#FFD600'
        r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                               linewidth=1, edgecolor=col, facecolor=col, alpha=0.15)
        ax.add_patch(r)
        ax.plot([(bb[0]+bb[2])/2], [(bb[1]+bb[3])/2], 's', color=col, ms=3, alpha=0.7)
    # Tracks
    cmap_t = cm.get_cmap('tab20')
    for i, t in enumerate(tracks):
        bb = t.get_state()
        col = cmap_t(i % 20)
        r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                               linewidth=2, edgecolor=col, facecolor='none')
        ax.add_patch(r)
        ax.text(bb[0]+1, bb[1]-2, str(t.track_id), color=col, fontsize=6, fontweight='bold')
    ax.set_title(f'{title}\nFrame {SAMPLE_FRAME} | {len(tracks)} active tracks', fontsize=10)
    ax.set_xlabel('x (pixels)'); ax.set_ylabel('y (pixels)')

# GT reference
ax = axes[0]
ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.set_facecolor('#1a1a2e')
for bb in frame_data['gt_bboxes'][:40]:
    r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                           linewidth=1, edgecolor='#90CAF9', facecolor='#90CAF9', alpha=0.2)
    ax.add_patch(r)
for d in frame_data['detections']:
    bb = d['bbox']
    col = '#00E676' if d['score'] >= 0.5 else '#FFD600'
    r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                           linewidth=1, edgecolor=col, facecolor='none')
    ax.add_patch(r)
ax.set_title(f'Ground Truth + Detections\nFrame {SAMPLE_FRAME} | {len(frame_data["detections"])} dets', fontsize=10)
ax.set_xlabel('x (pixels)'); ax.set_ylabel('y (pixels)')

plot_frame(axes[1], 'ByteTrack', frame_data['detections'], bt_results[SAMPLE_FRAME], frame_data['gt_bboxes'])
plot_frame(axes[2], 'SparseTrack', frame_data['detections'], st_results[SAMPLE_FRAME], frame_data['gt_bboxes'])

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    patches.Patch(facecolor='#90CAF9', alpha=0.4, label='GT boxes'),
    patches.Patch(facecolor='#00E676', alpha=0.6, label='High-conf det (≥0.5)'),
    patches.Patch(facecolor='#FFD600', alpha=0.6, label='Low-conf det (<0.5)'),
    patches.Patch(facecolor='none', edgecolor='white', label='Track (colored by ID)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig6_frame_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_frame_visualization.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 7 – Score threshold sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
bt_mota_list, st_mota_list = [], []
bt_idf1_list, st_idf1_list = [], []
bt_ids_list,  st_ids_list  = [], []

for ht in thresholds:
    bt_r = run_tracker(ByteTrack, data, high_thresh=ht, low_thresh=0.1, iou_thresh=0.3, min_hits=1)
    st_r = run_tracker(SparseTrack, data, high_thresh=ht, low_thresh=0.1, iou_thresh=0.3, n_layers=3, min_hits=1)
    bm = compute_metrics(data, bt_r)
    sm = compute_metrics(data, st_r)
    bt_mota_list.append(bm['MOTA']); st_mota_list.append(sm['MOTA'])
    bt_idf1_list.append(bm['IDF1']); st_idf1_list.append(sm['IDF1'])
    bt_ids_list.append(bm['ID_switches']); st_ids_list.append(sm['ID_switches'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(thresholds, bt_mota_list, 'o-', color=BT_COLOR, lw=2, label='ByteTrack')
axes[0].plot(thresholds, st_mota_list, 's-', color=ST_COLOR, lw=2, label='SparseTrack')
axes[0].set_xlabel('High-Conf Threshold'); axes[0].set_ylabel('MOTA')
axes[0].set_title('MOTA vs Detection Threshold'); axes[0].legend(); axes[0].axhline(0, color='k', ls='--', lw=0.8)

axes[1].plot(thresholds, bt_idf1_list, 'o-', color=BT_COLOR, lw=2, label='ByteTrack')
axes[1].plot(thresholds, st_idf1_list, 's-', color=ST_COLOR, lw=2, label='SparseTrack')
axes[1].set_xlabel('High-Conf Threshold'); axes[1].set_ylabel('IDF1')
axes[1].set_title('IDF1 vs Detection Threshold'); axes[1].legend()

axes[2].plot(thresholds, bt_ids_list, 'o-', color=BT_COLOR, lw=2, label='ByteTrack')
axes[2].plot(thresholds, st_ids_list, 's-', color=ST_COLOR, lw=2, label='SparseTrack')
axes[2].set_xlabel('High-Conf Threshold'); axes[2].set_ylabel('ID Switches')
axes[2].set_title('ID Switches vs Detection Threshold'); axes[2].legend()

plt.suptitle('Sensitivity Analysis: High-Confidence Detection Threshold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig7_threshold_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_threshold_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 8 – IoU threshold sensitivity
# ══════════════════════════════════════════════════════════════════════════

iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
bt_mota_iou, st_mota_iou = [], []
bt_idf1_iou, st_idf1_iou = [], []

for ith in iou_thresholds:
    bt_r = run_tracker(ByteTrack, data, high_thresh=0.5, low_thresh=0.1, iou_thresh=ith, min_hits=1)
    st_r = run_tracker(SparseTrack, data, high_thresh=0.5, low_thresh=0.1, iou_thresh=ith, n_layers=3, min_hits=1)
    bm = compute_metrics(data, bt_r)
    sm = compute_metrics(data, st_r)
    bt_mota_iou.append(bm['MOTA']); st_mota_iou.append(sm['MOTA'])
    bt_idf1_iou.append(bm['IDF1']); st_idf1_iou.append(sm['IDF1'])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(iou_thresholds, bt_mota_iou, 'o-', color=BT_COLOR, lw=2, label='ByteTrack')
axes[0].plot(iou_thresholds, st_mota_iou, 's-', color=ST_COLOR, lw=2, label='SparseTrack')
axes[0].set_xlabel('IoU Association Threshold'); axes[0].set_ylabel('MOTA')
axes[0].set_title('MOTA vs Association IoU Threshold'); axes[0].legend(); axes[0].axhline(0, color='k', ls='--', lw=0.8)

axes[1].plot(iou_thresholds, bt_idf1_iou, 'o-', color=BT_COLOR, lw=2, label='ByteTrack')
axes[1].plot(iou_thresholds, st_idf1_iou, 's-', color=ST_COLOR, lw=2, label='SparseTrack')
axes[1].set_xlabel('IoU Association Threshold'); axes[1].set_ylabel('IDF1')
axes[1].set_title('IDF1 vs Association IoU Threshold'); axes[1].legend()

plt.suptitle('Sensitivity Analysis: IoU Association Threshold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig8_iou_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_iou_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 9 – n_layers sensitivity for SparseTrack
# ══════════════════════════════════════════════════════════════════════════

layers_range = [2, 3, 4, 5, 6]
st_mota_layers, st_idf1_layers, st_ids_layers = [], [], []

for nl in layers_range:
    st_r = run_tracker(SparseTrack, data, high_thresh=0.5, low_thresh=0.1,
                       iou_thresh=0.3, n_layers=nl, min_hits=1)
    sm = compute_metrics(data, st_r)
    st_mota_layers.append(sm['MOTA'])
    st_idf1_layers.append(sm['IDF1'])
    st_ids_layers.append(sm['ID_switches'])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(layers_range, st_mota_layers, 's-', color=ST_COLOR, lw=2)
axes[0].set_xlabel('Number of Depth Layers'); axes[0].set_ylabel('MOTA')
axes[0].set_title('SparseTrack MOTA vs Depth Layers')
axes[0].axhline(bt_metrics['MOTA'], color=BT_COLOR, ls='--', lw=1.5, label='ByteTrack')
axes[0].legend(); axes[0].axhline(0, color='k', ls=':', lw=0.8)

axes[1].plot(layers_range, st_idf1_layers, 's-', color=ST_COLOR, lw=2)
axes[1].set_xlabel('Number of Depth Layers'); axes[1].set_ylabel('IDF1')
axes[1].set_title('SparseTrack IDF1 vs Depth Layers')
axes[1].axhline(bt_metrics['IDF1'], color=BT_COLOR, ls='--', lw=1.5, label='ByteTrack')
axes[1].legend()

axes[2].plot(layers_range, st_ids_layers, 's-', color=ST_COLOR, lw=2)
axes[2].set_xlabel('Number of Depth Layers'); axes[2].set_ylabel('ID Switches')
axes[2].set_title('SparseTrack ID Switches vs Depth Layers')
axes[2].axhline(bt_metrics['ID_switches'], color=BT_COLOR, ls='--', lw=1.5, label='ByteTrack')
axes[2].legend()

plt.suptitle('SparseTrack Sensitivity to Number of Depth Layers', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig9_nlayers_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_nlayers_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════
#  Save summary table as JSON
# ══════════════════════════════════════════════════════════════════════════

summary = {
    'dataset': {
        'n_frames': n_frames,
        'n_objects_per_frame': len(data[0]['gt_ids']),
        'avg_detections_per_frame': float(np.mean([len(f['detections']) for f in data])),
        'detection_rate': float(np.mean([len(f['detections']) / len(f['gt_ids']) for f in data])),
    },
    'ByteTrack': bt_metrics,
    'SparseTrack': st_metrics,
    'threshold_sweep': {
        'thresholds': thresholds,
        'ByteTrack_MOTA': bt_mota_list,
        'SparseTrack_MOTA': st_mota_list,
        'ByteTrack_IDF1': bt_idf1_list,
        'SparseTrack_IDF1': st_idf1_list,
    },
    'nlayers_sweep': {
        'n_layers': layers_range,
        'SparseTrack_MOTA': st_mota_layers,
        'SparseTrack_IDF1': st_idf1_layers,
        'SparseTrack_ID_switches': st_ids_layers,
    }
}

with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== FINAL SUMMARY ===")
print(f"{'Metric':<15} {'ByteTrack':>12} {'SparseTrack':>12} {'Delta':>12}")
print("-" * 54)
for m in ['MOTA', 'MOTP', 'IDF1', 'TP', 'FP', 'FN', 'ID_switches']:
    bv = bt_metrics[m]
    sv = st_metrics[m]
    delta = sv - bv
    sign = '+' if delta > 0 else ''
    print(f"{m:<15} {bv:>12.4f} {sv:>12.4f} {sign+str(round(delta,4)):>12}")

print("\nAll figures saved to:", FIGURES_DIR)
print("Summary saved to:", f'{OUTPUT_DIR}/summary.json')
