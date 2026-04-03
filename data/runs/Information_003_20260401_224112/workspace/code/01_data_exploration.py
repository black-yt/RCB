"""
Data Exploration for NF-UNSW-NB15-v2 Dataset
"""
import sys
sys.path.insert(0, '/mnt/shared-storage-user/yetianlin/ResearchClawBench/.venv/lib/python3.13/site-packages')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json

# Setup paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_003_20260401_224112')
DATA_PATH = WORKSPACE / 'data' / 'NF-UNSW-NB15-v2_3d.pt'
OUTPUTS = WORKSPACE / 'outputs'
IMAGES = WORKSPACE / 'report' / 'images'
OUTPUTS.mkdir(exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
from torch_geometric.data.temporal import TemporalData
data = torch.load(str(DATA_PATH), map_location='cpu', weights_only=False)
print(f"Loaded TemporalData with {len(data.t)} samples")

# Attack type names for UNSW-NB15
ATTACK_NAMES = {
    0: 'Exploits',
    1: 'Reconnaissance',
    2: 'Normal',
    3: 'DoS',
    4: 'Generic',
    5: 'Shellcode',
    6: 'Fuzzers',
    7: 'Analysis',
    8: 'Backdoors',
    9: 'Worms'
}

# Basic statistics
stats = {}
stats['n_samples'] = len(data.t)
stats['n_features'] = data.msg.shape[1]
stats['time_range'] = [data.t.min().item(), data.t.max().item()]
stats['n_src_nodes'] = data.src.unique().shape[0]
stats['n_dst_nodes'] = data.dst.unique().shape[0]

labels_np = data.label.numpy()
attack_np = data.attack.numpy()
msg_np = data.msg.numpy()
t_np = data.t.numpy()

# Label distribution
label_counts = {0: (labels_np == 0).sum(), 1: (labels_np == 1).sum()}
attack_counts = {int(k): int((attack_np == k).sum()) for k in np.unique(attack_np)}
stats['label_counts'] = label_counts
stats['attack_counts'] = attack_counts

print(f"Total samples: {stats['n_samples']}")
print(f"Features: {stats['n_features']}")
print(f"Binary labels: Benign={label_counts[0]}, Attack={label_counts[1]}")
print(f"Attack types: {attack_counts}")

# Save stats
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

with open(OUTPUTS / 'data_stats.json', 'w') as f:
    json.dump(convert_to_serializable(stats), f, indent=2)

# ===== FIGURE 1: Dataset Overview =====
fig = plt.figure(figsize=(18, 12))
fig.suptitle('NF-UNSW-NB15-v2 Dataset Overview', fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# 1a: Binary label distribution
ax1 = fig.add_subplot(gs[0, 0])
labels_list = ['Normal', 'Attack']
counts_list = [label_counts[0], label_counts[1]]
colors = ['#2196F3', '#F44336']
wedges, texts, autotexts = ax1.pie(counts_list, labels=labels_list, colors=colors,
                                    autopct='%1.1f%%', startangle=90)
for text in autotexts:
    text.set_fontsize(12)
ax1.set_title('Binary Label Distribution', fontweight='bold')

# 1b: Multi-class attack distribution
ax2 = fig.add_subplot(gs[0, 1])
attack_ids = sorted(attack_counts.keys())
attack_labels = [ATTACK_NAMES.get(i, f'Class {i}') for i in attack_ids]
attack_vals = [attack_counts[i] for i in attack_ids]
colors_bar = plt.cm.tab10(np.linspace(0, 1, len(attack_ids)))
bars = ax2.bar(range(len(attack_ids)), attack_vals, color=colors_bar)
ax2.set_xticks(range(len(attack_ids)))
ax2.set_xticklabels(attack_labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Count')
ax2.set_title('Multi-class Attack Distribution', fontweight='bold')
ax2.set_yscale('log')
for bar, val in zip(bars, attack_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
             f'{val:,}', ha='center', va='bottom', fontsize=7, rotation=45)

# 1c: Temporal distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(t_np[labels_np == 0], bins=50, alpha=0.6, color='#2196F3', label='Normal', density=True)
ax3.hist(t_np[labels_np == 1], bins=50, alpha=0.6, color='#F44336', label='Attack', density=True)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Density')
ax3.set_title('Temporal Distribution of Traffic', fontweight='bold')
ax3.legend()

# 1d: Feature correlation heatmap (first 20 features)
ax4 = fig.add_subplot(gs[1, 0:2])
corr_matrix = np.corrcoef(msg_np[:5000, :20].T)
im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax4.set_title('Feature Correlation Matrix (First 20 Features)', fontweight='bold')
ax4.set_xlabel('Feature Index')
ax4.set_ylabel('Feature Index')
plt.colorbar(im, ax=ax4)

# 1e: Feature value distributions (box plot for selected features)
ax5 = fig.add_subplot(gs[1, 2])
selected_feats = [0, 1, 2, 3, 4, 5, 6, 7]
feat_data_normal = [msg_np[labels_np == 0, f] for f in selected_feats]
feat_data_attack = [msg_np[labels_np == 1, f] for f in selected_feats]
positions_n = np.array(range(len(selected_feats))) * 2
positions_a = np.array(range(len(selected_feats))) * 2 + 0.8
bp1 = ax5.boxplot(feat_data_normal, positions=positions_n, widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor='#2196F3', alpha=0.7),
                   medianprops=dict(color='navy'), showfliers=False)
bp2 = ax5.boxplot(feat_data_attack, positions=positions_a, widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor='#F44336', alpha=0.7),
                   medianprops=dict(color='darkred'), showfliers=False)
ax5.set_xticks([i * 2 + 0.4 for i in range(len(selected_feats))])
ax5.set_xticklabels([f'F{i}' for i in selected_feats], fontsize=9)
ax5.set_ylabel('Feature Value')
ax5.set_title('Feature Distributions\n(Normal vs Attack)', fontweight='bold')
from matplotlib.patches import Patch
ax5.legend(handles=[Patch(facecolor='#2196F3', alpha=0.7, label='Normal'),
                     Patch(facecolor='#F44336', alpha=0.7, label='Attack')],
           loc='upper right')

plt.savefig(IMAGES / 'fig1_dataset_overview.png', dpi=150, bbox_inches='tight')
print("Saved fig1_dataset_overview.png")
plt.close()

# ===== FIGURE 2: Feature Analysis =====
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Feature Analysis and Attack Patterns', fontsize=14, fontweight='bold')

# 2a: PCA of features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample for speed
idx = np.random.choice(len(labels_np), 10000, replace=False)
X_sample = msg_np[idx]
y_sample = labels_np[idx]
attack_sample = attack_np[idx]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ax = axes[0, 0]
scatter = ax.scatter(X_pca[y_sample == 0, 0], X_pca[y_sample == 0, 1],
                     c='#2196F3', alpha=0.3, s=5, label='Normal')
scatter2 = ax.scatter(X_pca[y_sample == 1, 0], X_pca[y_sample == 1, 1],
                      c='#F44336', alpha=0.3, s=5, label='Attack')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA Projection of Traffic Features', fontweight='bold')
ax.legend(markerscale=3)

# Save PCA data
np.save(OUTPUTS / 'pca_components.npy', X_pca)
np.save(OUTPUTS / 'pca_labels.npy', y_sample)
print(f"PCA explained variance: {pca.explained_variance_ratio_[:2]}")

# 2b: Feature importance (variance-based)
ax = axes[0, 1]
feat_var_normal = msg_np[labels_np == 0].var(axis=0)
feat_var_attack = msg_np[labels_np == 1].var(axis=0)
feat_mean_diff = np.abs(msg_np[labels_np == 1].mean(axis=0) - msg_np[labels_np == 0].mean(axis=0))
feat_idx = np.argsort(feat_mean_diff)[-15:]
ax.barh(range(15), feat_mean_diff[feat_idx], color=plt.cm.viridis(feat_mean_diff[feat_idx]/feat_mean_diff.max()))
ax.set_yticks(range(15))
ax.set_yticklabels([f'Feature {i}' for i in feat_idx], fontsize=8)
ax.set_xlabel('|Mean Difference| (Normal vs Attack)')
ax.set_title('Top 15 Discriminative Features', fontweight='bold')

# 2c: Multi-class PCA
ax = axes[1, 0]
colors_multi = plt.cm.tab10(np.linspace(0, 1, 10))
for attack_id in np.unique(attack_sample):
    mask = attack_sample == attack_id
    if mask.sum() > 0:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors_multi[attack_id]], alpha=0.4, s=8,
                   label=ATTACK_NAMES.get(int(attack_id), f'C{attack_id}'))
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA: Multi-class Traffic Distribution', fontweight='bold')
ax.legend(fontsize=7, markerscale=2, loc='upper right')

# 2d: Temporal attack patterns
ax = axes[1, 1]
time_bins = np.linspace(0, 86400, 25)  # 1-hour bins
for attack_id in sorted(attack_counts.keys()):
    if attack_id == 2:  # skip normal (too many)
        continue
    times_attack = t_np[attack_np == attack_id]
    if len(times_attack) > 5:
        hist, _ = np.histogram(times_attack, bins=time_bins)
        ax.plot(time_bins[:-1]/3600, hist, label=ATTACK_NAMES.get(attack_id, f'C{attack_id}'),
                linewidth=1.5, marker='.')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Count')
ax.set_title('Temporal Attack Distribution', fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(0, 24)

plt.tight_layout()
plt.savefig(IMAGES / 'fig2_feature_analysis.png', dpi=150, bbox_inches='tight')
print("Saved fig2_feature_analysis.png")
plt.close()

print("\nData exploration complete!")
print(f"Stats saved to {OUTPUTS / 'data_stats.json'}")
