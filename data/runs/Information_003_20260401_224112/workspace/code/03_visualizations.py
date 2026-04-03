"""
Visualization code for DIDS-MFL research report
"""
import sys
sys.path.insert(0, '/mnt/shared-storage-user/yetianlin/ResearchClawBench/.venv/lib/python3.13/site-packages')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import warnings
warnings.filterwarnings('ignore')

# Paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_003_20260401_224112')
OUTPUTS = WORKSPACE / 'outputs'
IMAGES = WORKSPACE / 'report' / 'images'
DATA_PATH = WORKSPACE / 'data' / 'NF-UNSW-NB15-v2_3d.pt'

CLASS_NAMES = ['Normal', 'Exploits', 'Recon', 'DoS', 'Generic',
               'Shellcode', 'Fuzzers', 'Analysis', 'Backdoors', 'Worms']

# Load data
from torch_geometric.data.temporal import TemporalData
data = torch.load(str(DATA_PATH), map_location='cpu', weights_only=False)
X = data.msg.numpy().astype(np.float32)
y_binary = data.label.numpy()
y_multi = data.attack.numpy()
t = data.t.numpy()
src = data.src.numpy()
dst = data.dst.numpy()

sort_idx = np.argsort(t)
X, y_binary, y_multi = X[sort_idx], y_binary[sort_idx], y_multi[sort_idx]
t = t[sort_idx]
attack_remap = {2: 0, 0: 1, 1: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
y_mc = np.array([attack_remap[a] for a in y_multi])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Load results
with open(OUTPUTS/'all_results.json') as f:
    results = json.load(f)

print("Data loaded. Generating figures...")

# Color scheme
DIDS_COLOR = '#1565C0'    # Deep blue for DIDS-MFL
BASE_COLOR = '#E53935'    # Red for baseline
PALETTE = ['#1565C0', '#E53935', '#2E7D32', '#F57F17', '#6A1B9A']

# =========================================================
# FIGURE 3: Classification Performance Comparison
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('DIDS-MFL vs Baseline Classification Performance', fontsize=14, fontweight='bold')

# 3a: Binary metrics comparison
ax = axes[0]
metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
dids_vals = [results['binary']['dids_mfl'][m] for m in metrics]
base_vals = [results['binary']['baseline'][m] for m in metrics]

x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, dids_vals, w, label='DIDS-MFL', color=DIDS_COLOR, alpha=0.85)
bars2 = ax.bar(x + w/2, base_vals, w, label='Baseline', color=BASE_COLOR, alpha=0.85)
ax.set_ylim(0.96, 1.005)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Binary Classification', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, dids_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
for bar, val in zip(bars2, base_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45, color='#B71C1C')

# 3b: Multi-class metrics
ax = axes[1]
mc_metrics = ['f1_macro', 'f1_weighted', 'accuracy']
mc_labels = ['F1 Macro', 'F1 Weighted', 'Accuracy']
mc_dids = [results['multiclass']['dids_mfl'][m] for m in mc_metrics]
mc_base = [results['multiclass']['baseline'][m] for m in mc_metrics]

x = np.arange(len(mc_metrics))
bars1 = ax.bar(x - w/2, mc_dids, w, label='DIDS-MFL', color=DIDS_COLOR, alpha=0.85)
bars2 = ax.bar(x + w/2, mc_base, w, label='Baseline', color=BASE_COLOR, alpha=0.85)
ax.set_ylim(0.7, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(mc_labels, rotation=15, ha='right', fontsize=10)
ax.set_ylabel('Score')
ax.set_title('Multi-class Classification', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, mc_dids):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, mc_base):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, color='#B71C1C')

# 3c: Unknown attack detection comparison
ax = axes[2]
unk_dids = results['unknown']['dids_mfl']
unk_base = results['unknown']['baseline']
unk_metrics = ['f1_all', 'f1_unknown', 'auc', 'detection_rate']
unk_labels = ['F1 (All)', 'F1 (Unknown)', 'AUC', 'Detection Rate']
unk_d_vals = [unk_dids.get(m, 0) for m in unk_metrics]
unk_b_vals = [unk_base.get(m, 0) for m in unk_metrics]

x = np.arange(len(unk_metrics))
bars1 = ax.bar(x - w/2, unk_d_vals, w, label='DIDS-MFL', color=DIDS_COLOR, alpha=0.85)
bars2 = ax.bar(x + w/2, unk_b_vals, w, label='Baseline', color=BASE_COLOR, alpha=0.85)
ax.set_ylim(0.9, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(unk_labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Unknown Attack Detection', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, unk_d_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
for bar, val in zip(bars2, unk_b_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45, color='#B71C1C')

plt.tight_layout()
plt.savefig(IMAGES/'fig3_classification_comparison.png', dpi=150, bbox_inches='tight')
print("Saved fig3_classification_comparison.png")
plt.close()

# =========================================================
# FIGURE 4: Confusion Matrix (DIDS-MFL Multi-class)
# =========================================================
y_true_cm = np.load(OUTPUTS/'cm_true.npy')
y_pred_cm = np.load(OUTPUTS/'cm_pred.npy')

# Map labels to class names
classes_present = sorted(list(set(y_true_cm)))
labels_present = [CLASS_NAMES[c] for c in classes_present]

cm = confusion_matrix(y_true_cm, y_pred_cm, labels=classes_present)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Confusion Matrices: DIDS-MFL Multi-class Classification', fontsize=14, fontweight='bold')

# Raw counts
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_present, yticklabels=labels_present,
            ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Class', fontsize=11)
ax.set_ylabel('True Class', fontsize=11)
ax.set_title('Confusion Matrix (Counts)', fontweight='bold')
ax.tick_params(axis='x', rotation=45)

# Normalized
ax = axes[1]
mask = np.zeros_like(cm_norm, dtype=bool)
# Annotate all cells
annot = np.empty_like(cm_norm, dtype=object)
for i in range(len(classes_present)):
    for j in range(len(classes_present)):
        if cm[i, j] > 0:
            annot[i, j] = f'{cm_norm[i, j]:.2f}\n({cm[i,j]})'
        else:
            annot[i, j] = '0'

sns.heatmap(cm_norm, annot=annot, fmt='', cmap='YlOrRd',
            xticklabels=labels_present, yticklabels=labels_present,
            ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Rate'})
ax.set_xlabel('Predicted Class', fontsize=11)
ax.set_ylabel('True Class', fontsize=11)
ax.set_title('Normalized Confusion Matrix', fontweight='bold')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(IMAGES/'fig4_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved fig4_confusion_matrix.png")
plt.close()

# =========================================================
# FIGURE 5: Few-Shot Learning Curves
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Few-Shot Attack Detection Performance', fontsize=14, fontweight='bold')

shot_vals = sorted([int(k) for k in results['few_shot']['dids_mfl'].keys()])
dids_f1 = [results['few_shot']['dids_mfl'][str(k)]['f1'] for k in shot_vals]
base_f1 = [results['few_shot']['baseline'][str(k)]['f1'] for k in shot_vals]
dids_acc = [results['few_shot']['dids_mfl'][str(k)]['accuracy'] for k in shot_vals]
base_acc = [results['few_shot']['baseline'][str(k)]['accuracy'] for k in shot_vals]

ax = axes[0]
ax.plot(shot_vals, dids_f1, 'o-', color=DIDS_COLOR, linewidth=2.5, markersize=8, label='DIDS-MFL')
ax.plot(shot_vals, base_f1, 's--', color=BASE_COLOR, linewidth=2.5, markersize=8, label='Baseline')
ax.fill_between(shot_vals, dids_f1, base_f1,
                where=[d >= b for d, b in zip(dids_f1, base_f1)],
                alpha=0.2, color=DIDS_COLOR, label='DIDS-MFL advantage')
ax.fill_between(shot_vals, dids_f1, base_f1,
                where=[d < b for d, b in zip(dids_f1, base_f1)],
                alpha=0.2, color=BASE_COLOR, label='Baseline advantage')
ax.set_xlabel('Number of Shots per Attack Class', fontsize=11)
ax.set_ylabel('F1-Score (Binary)', fontsize=11)
ax.set_title('Few-Shot F1 vs Number of Shots', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_ylim(0, 1.05)
for i, (s, d, b) in enumerate(zip(shot_vals, dids_f1, base_f1)):
    ax.annotate(f'{d:.3f}', (s, d), textcoords="offset points", xytext=(0, 8),
                ha='center', fontsize=8, color=DIDS_COLOR)

ax = axes[1]
# F1 gain at each shot count
gains = [d - b for d, b in zip(dids_f1, base_f1)]
colors = [DIDS_COLOR if g >= 0 else BASE_COLOR for g in gains]
ax.bar(range(len(shot_vals)), gains, color=colors, alpha=0.8)
ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.set_xticks(range(len(shot_vals)))
ax.set_xticklabels([f'{s}-shot' for s in shot_vals], rotation=30, ha='right')
ax.set_ylabel('F1 Gain (DIDS-MFL − Baseline)', fontsize=11)
ax.set_title('F1-Score Improvement Over Baseline', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, g in enumerate(gains):
    ax.text(i, g + 0.002 * np.sign(g), f'{g:+.4f}', ha='center',
            fontsize=9, fontweight='bold',
            color=DIDS_COLOR if g >= 0 else BASE_COLOR)

plt.tight_layout()
plt.savefig(IMAGES/'fig5_few_shot_curves.png', dpi=150, bbox_inches='tight')
print("Saved fig5_few_shot_curves.png")
plt.close()

# =========================================================
# FIGURE 6: Disentanglement Analysis
# =========================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Statistical Disentanglement Analysis', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# 6a: PCA with disentanglement coloring
ax1 = fig.add_subplot(gs[0, 0:2])
# Sample 8000 points
idx = np.random.choice(len(X), 8000, replace=False)
X_samp = X_norm[idx]
y_samp = y_binary[idx]
y_mc_samp = y_mc[idx]

pca = PCA(n_components=2, random_state=42)
X_pca2 = pca.fit_transform(X_samp)

# Color by attack class
cmap = plt.cm.tab10
for cls_id in range(10):
    mask = y_mc_samp == cls_id
    if mask.sum() > 0:
        ax1.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                    c=[cmap(cls_id/9)], alpha=0.4, s=6,
                    label=CLASS_NAMES[cls_id])
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax1.set_title('PCA Projection: Multi-class Traffic Distribution', fontweight='bold')
ax1.legend(fontsize=7, markerscale=2, ncol=2, loc='upper right')
ax1.grid(True, alpha=0.2)

# 6b: Fisher Discriminant Ratio per PC
ax2 = fig.add_subplot(gs[0, 2])
disc_scores = np.load(OUTPUTS/'disc_scores.npy')
x_pc = np.arange(1, len(disc_scores)+1)
colors_pc = ['#1565C0' if i < 10 else '#E53935' for i in range(len(disc_scores))]
ax2.bar(x_pc, disc_scores, color=colors_pc, alpha=0.85)
ax2.set_xlabel('Principal Component', fontsize=10)
ax2.set_ylabel('Fisher Discriminant Ratio', fontsize=10)
ax2.set_title('PC Discriminability\n(Blue=Attack, Red=Normal)', fontweight='bold')
ax2.axvline(10.5, color='black', linestyle='--', linewidth=1.5, label='Split')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# 6c: Feature importance (mean difference)
ax3 = fig.add_subplot(gs[1, 0])
mean_attack = X_norm[y_binary == 1].mean(axis=0)
mean_normal = X_norm[y_binary == 0].mean(axis=0)
mean_diff = np.abs(mean_attack - mean_normal)
top_k = 15
top_idx = np.argsort(mean_diff)[-top_k:]
colors_feat = plt.cm.viridis(np.linspace(0.2, 0.9, top_k))
ax3.barh(range(top_k), mean_diff[top_idx], color=colors_feat)
ax3.set_yticks(range(top_k))
ax3.set_yticklabels([f'Feature {i}' for i in top_idx], fontsize=8)
ax3.set_xlabel('|Mean Difference| (Normal vs Attack)')
ax3.set_title(f'Top {top_k} Discriminative\nFeatures', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 6d: Attack class feature heatmap (average feature values per class)
ax4 = fig.add_subplot(gs[1, 1:3])
n_feats_show = 10
feat_matrix = np.zeros((10, n_feats_show))
for cls_id in range(10):
    mask = y_mc == cls_id
    if mask.sum() > 0:
        feat_matrix[cls_id] = X_norm[mask][:, top_idx[-n_feats_show:]].mean(axis=0)

im = ax4.imshow(feat_matrix, aspect='auto', cmap='RdYlBu_r', vmin=-2, vmax=2)
ax4.set_yticks(range(10))
ax4.set_yticklabels(CLASS_NAMES, fontsize=9)
ax4.set_xticks(range(n_feats_show))
ax4.set_xticklabels([f'F{i}' for i in top_idx[-n_feats_show:]], fontsize=9)
ax4.set_title('Average Feature Activation per Attack Class\n(Top Discriminative Features)', fontweight='bold')
ax4.set_xlabel('Feature', fontsize=10)
ax4.set_ylabel('Attack Class', fontsize=10)
plt.colorbar(im, ax=ax4, label='Normalized Value')

plt.savefig(IMAGES/'fig6_disentanglement.png', dpi=150, bbox_inches='tight')
print("Saved fig6_disentanglement.png")
plt.close()

# =========================================================
# FIGURE 7: Per-class F1-score comparison
# =========================================================
with open(OUTPUTS/'classification_report.json') as f:
    cr = json.load(f)

class_f1 = []
for name in CLASS_NAMES:
    if name in cr:
        class_f1.append(cr[name]['f1-score'])
    else:
        class_f1.append(0.0)

# Also compute baseline per-class F1
from sklearn.ensemble import RandomForestClassifier

n_total = len(X)
n_train_split = int(n_total * 0.8)
X_tr_b = X_norm[:n_train_split]
X_te_b = X_norm[n_train_split:]
y_mc_tr_b = y_mc[:n_train_split]
y_mc_te_b = y_mc[n_train_split:]

train_classes = set(np.unique(y_mc_tr_b))
test_mask = np.array([y in train_classes for y in y_mc_te_b])

clf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
clf_base.fit(X_tr_b, y_mc_tr_b)
y_pred_base_mc = clf_base.predict(X_te_b[test_mask])
y_true_mc_b = y_mc_te_b[test_mask]

from sklearn.metrics import classification_report as cr_fn
cr_base = cr_fn(y_true_mc_b, y_pred_base_mc,
                target_names=CLASS_NAMES, output_dict=True, zero_division=0)
base_class_f1 = [cr_base.get(n, {}).get('f1-score', 0) for n in CLASS_NAMES]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Per-class F1-Score Analysis', fontsize=14, fontweight='bold')

# 7a: Side-by-side bar chart
ax = axes[0]
x = np.arange(len(CLASS_NAMES))
w = 0.35
bars1 = ax.bar(x - w/2, class_f1, w, label='DIDS-MFL', color=DIDS_COLOR, alpha=0.85)
bars2 = ax.bar(x + w/2, base_class_f1, w, label='Baseline', color=BASE_COLOR, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=40, ha='right', fontsize=9)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('Per-class F1-Score: DIDS-MFL vs Baseline', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')

for bar, val in zip(bars1, class_f1):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
for bar, val in zip(bars2, base_class_f1):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45, color='#B71C1C')

# 7b: Scatter plot of F1 values
ax = axes[1]
gains = [d - b for d, b in zip(class_f1, base_class_f1)]
colors_g = [DIDS_COLOR if g >= 0 else BASE_COLOR for g in gains]

ax.scatter(base_class_f1, class_f1, s=150, c=[DIDS_COLOR]*len(CLASS_NAMES),
           zorder=5, edgecolors='white', linewidth=0.5)
# Diagonal line
lim_min = min(min(class_f1), min(base_class_f1)) - 0.05
lim_max = max(max(class_f1), max(base_class_f1)) + 0.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, label='Equal performance')
ax.fill_between([lim_min, lim_max], [lim_min, lim_max], lim_max,
                alpha=0.1, color=DIDS_COLOR, label='DIDS-MFL advantage')
ax.fill_between([lim_min, lim_max], lim_min, [lim_min, lim_max],
                alpha=0.1, color=BASE_COLOR, label='Baseline advantage')

for i, name in enumerate(CLASS_NAMES):
    ax.annotate(name, (base_class_f1[i], class_f1[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_xlabel('Baseline F1-Score', fontsize=11)
ax.set_ylabel('DIDS-MFL F1-Score', fontsize=11)
ax.set_title('Per-class Performance Comparison', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)

plt.tight_layout()
plt.savefig(IMAGES/'fig7_perclass_f1.png', dpi=150, bbox_inches='tight')
print("Saved fig7_perclass_f1.png")
plt.close()

# =========================================================
# FIGURE 8: Temporal Attack Pattern and Graph Analysis
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Temporal Dynamics and Network Graph Analysis', fontsize=14, fontweight='bold')

# 8a: Temporal density of different attack types
ax = axes[0, 0]
n_bins = 24
time_bins = np.linspace(0, 86400, n_bins+1)
cmap_cls = plt.cm.tab10

# Stack plot of attack type proportions over time
attack_counts_time = np.zeros((10, n_bins))
for cls_id in range(10):
    times_cls = t[y_mc == cls_id]
    if len(times_cls) > 0:
        hist, _ = np.histogram(times_cls, bins=time_bins)
        attack_counts_time[cls_id] = hist

total_time = attack_counts_time.sum(axis=0)
total_time[total_time == 0] = 1

attack_pct_time = attack_counts_time / total_time

# Plot lines for each attack type
hours = np.arange(n_bins)
for cls_id in range(1, 10):  # skip normal (too dominant)
    vals = attack_pct_time[cls_id]
    if vals.sum() > 0:
        ax.plot(hours, vals * 100, label=CLASS_NAMES[cls_id],
                color=cmap_cls(cls_id/9), linewidth=1.5, marker='.')

ax.set_xlabel('Hour of Day', fontsize=10)
ax.set_ylabel('% of Total Traffic', fontsize=10)
ax.set_title('Attack Distribution by Hour', fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)

# 8b: dt (inter-arrival time) distribution by class
ax = axes[0, 1]
dt_sorted = data.dt.numpy()[sort_idx]

for cls_id in [0, 1, 3, 6, 7]:  # select a few representative classes
    dt_cls = dt_sorted[y_mc == cls_id]
    dt_cls_capped = np.clip(dt_cls, 0, 10)  # cap at 10 seconds
    if len(dt_cls) > 0:
        ax.hist(dt_cls_capped, bins=50, alpha=0.6,
                color=cmap_cls(cls_id/9), label=CLASS_NAMES[cls_id],
                density=True)

ax.set_xlabel('Inter-arrival Time (seconds, capped at 10)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Inter-arrival Time Distribution', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 8c: Degree distribution (src IP frequency)
ax = axes[1, 0]
src_counts = np.bincount(src)
src_counts_nz = src_counts[src_counts > 0]
ax.hist(np.log10(src_counts_nz + 1), bins=50, color='#1565C0', alpha=0.8, edgecolor='white')
ax.set_xlabel('log10(Number of Flows)', fontsize=10)
ax.set_ylabel('Number of Source IPs', fontsize=10)
ax.set_title('Source IP Flow Count Distribution', fontweight='bold')
ax.grid(True, alpha=0.3)

# Annotate: high-frequency IPs likely malicious
top_src_count = np.sort(src_counts_nz)[-5:]
ax.axvline(np.log10(top_src_count[-1] + 1), color='red', linestyle='--',
           label=f'Max: {top_src_count[-1]}', linewidth=2)
ax.legend(fontsize=9)

# 8d: Rolling attack rate over time
ax = axes[1, 1]
window_size = 1000
n_windows = len(t) // window_size
attack_rates = np.zeros(n_windows)
times_centers = np.zeros(n_windows)
for i in range(n_windows):
    start = i * window_size
    end = start + window_size
    attack_rates[i] = y_binary[start:end].mean()
    times_centers[i] = t[start:end].mean() / 3600

ax.plot(times_centers, attack_rates * 100, color='#E53935', linewidth=1.5, alpha=0.8)
ax.fill_between(times_centers, 0, attack_rates * 100, color='#E53935', alpha=0.2)
ax.set_xlabel('Time (hours)', fontsize=10)
ax.set_ylabel('Attack Rate (%)', fontsize=10)
ax.set_title('Rolling Attack Rate Over Time', fontweight='bold')
ax.grid(True, alpha=0.3)

ax_twin = ax.twinx()
# Normal traffic density
normal_hist, _ = np.histogram(t[y_binary == 0] / 3600, bins=24)
hours_bins = np.linspace(0, 24, 25)
ax_twin.bar(hours_bins[:-1], normal_hist, width=1.0, alpha=0.2, color='#2196F3', label='Normal traffic')
ax_twin.set_ylabel('Normal Flow Count', color='#2196F3', fontsize=9)

plt.tight_layout()
plt.savefig(IMAGES/'fig8_temporal_graph.png', dpi=150, bbox_inches='tight')
print("Saved fig8_temporal_graph.png")
plt.close()

# =========================================================
# FIGURE 9: ROC Curves and PR Curves
# =========================================================
# Rerun binary models to get probability scores for ROC curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

n_split = int(len(X) * 0.8)
X_tr_f = X_norm[:n_split]
X_te_f = X_norm[n_split:]
y_bin_tr_f = y_binary[:n_split]
y_bin_te_f = y_binary[n_split:]

# DIDS-MFL features
X_dids_full = np.load(OUTPUTS/'X_dids_mfl.npy')
X_tr_d = X_dids_full[:n_split]
X_te_d = X_dids_full[n_split:]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC Curve and Precision-Recall Curve', fontsize=14, fontweight='bold')

# Train RF for each
clf_roc_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
clf_roc_base.fit(X_tr_f, y_bin_tr_f)
prob_base = clf_roc_base.predict_proba(X_te_f)[:, 1]

clf_roc_dids = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
clf_roc_dids.fit(X_tr_d, y_bin_tr_f)
prob_dids = clf_roc_dids.predict_proba(X_te_d)[:, 1]

# ROC
ax = axes[0]
fpr_b, tpr_b, _ = roc_curve(y_bin_te_f, prob_base)
fpr_d, tpr_d, _ = roc_curve(y_bin_te_f, prob_dids)
auc_b = roc_auc_score(y_bin_te_f, prob_base)
auc_d = roc_auc_score(y_bin_te_f, prob_dids)

ax.plot(fpr_b, tpr_b, color=BASE_COLOR, linewidth=2, label=f'Baseline (AUC={auc_b:.4f})')
ax.plot(fpr_d, tpr_d, color=DIDS_COLOR, linewidth=2, label=f'DIDS-MFL (AUC={auc_d:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.fill_between(fpr_d, tpr_d, alpha=0.1, color=DIDS_COLOR)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve (Binary Classification)', fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

# PR Curve
ax = axes[1]
prec_b, rec_b, _ = precision_recall_curve(y_bin_te_f, prob_base)
prec_d, rec_d, _ = precision_recall_curve(y_bin_te_f, prob_dids)
ap_b = average_precision_score(y_bin_te_f, prob_base)
ap_d = average_precision_score(y_bin_te_f, prob_dids)

ax.plot(rec_b, prec_b, color=BASE_COLOR, linewidth=2, label=f'Baseline (AP={ap_b:.4f})')
ax.plot(rec_d, prec_d, color=DIDS_COLOR, linewidth=2, label=f'DIDS-MFL (AP={ap_d:.4f})')
ax.axhline(y_binary.mean(), color='gray', linestyle='--', label='Random classifier')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES/'fig9_roc_pr_curves.png', dpi=150, bbox_inches='tight')
print("Saved fig9_roc_pr_curves.png")
plt.close()

print("\nAll visualizations complete!")
print(f"Figures saved to: {IMAGES}")
