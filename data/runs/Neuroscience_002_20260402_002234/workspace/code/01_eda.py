"""
EDA: Neuron Segment Merge Prediction
Generates data overview plots for report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path('../data')
OUT_DIR = Path('../report/images')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train = pd.read_csv(DATA_DIR / 'train_simulated.csv')
test  = pd.read_csv(DATA_DIR / 'test_simulated.csv')
feat_cols = [str(i) for i in range(20)]

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Train label dist:\n{train['label'].value_counts()}")
print(f"Test  label dist:\n{test['label'].value_counts()}")

# ── Figure 1: Class imbalance + degradation breakdown ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1a: class balance (train)
counts = train['label'].value_counts().sort_index()
axes[0].bar(['No Merge (0)', 'Merge (1)'], counts.values, color=['#4C72B0', '#DD8452'])
axes[0].set_title('Class Distribution (Training Set)', fontsize=12)
axes[0].set_ylabel('Sample count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}\n({v/len(train)*100:.1f}%)', ha='center', fontsize=10)

# 1b: degradation distribution
deg_counts = train['degradation'].value_counts()
axes[1].bar(deg_counts.index, deg_counts.values, color=sns.color_palette('Set2', 4))
axes[1].set_title('Degradation Type Distribution (Training Set)', fontsize=12)
axes[1].set_ylabel('Sample count')
axes[1].tick_params(axis='x', rotation=20)
for i, v in enumerate(deg_counts.values):
    axes[1].text(i, v + 200, f'{v:,}', ha='center', fontsize=10)

# 1c: label rate by degradation
label_by_deg = train.groupby('degradation')['label'].mean().sort_values(ascending=False)
axes[2].bar(label_by_deg.index, label_by_deg.values * 100, color=sns.color_palette('Set2', 4))
axes[2].set_title('Merge Rate by Degradation Type', fontsize=12)
axes[2].set_ylabel('Merge rate (%)')
axes[2].tick_params(axis='x', rotation=20)
for i, v in enumerate(label_by_deg.values):
    axes[2].text(i, v * 100 + 0.1, f'{v*100:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_class_degradation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1")

# ── Figure 2: Feature distributions by label ──────────────────────────────
fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(feat_cols):
    ax = axes[i]
    for lbl, clr, nm in [(0, '#4C72B0', 'No Merge'), (1, '#DD8452', 'Merge')]:
        vals = train.loc[train['label'] == lbl, col]
        ax.hist(vals, bins=60, alpha=0.5, color=clr, label=nm, density=True)
    ax.set_title(f'Feature {col}', fontsize=9)
    ax.set_xlabel('Value', fontsize=7)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

plt.suptitle('Feature Distributions by Class Label', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2")

# ── Figure 3: Feature correlation heatmap ────────────────────────────────
corr = train[feat_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.3)
ax.set_title('Feature Correlation Matrix (Training Set)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3")

# ── Figure 4: Feature means by label ─────────────────────────────────────
means = train.groupby('label')[feat_cols].mean()
x = np.arange(20)
w = 0.35
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(x - w/2, means.loc[0.0], w, label='No Merge (0)', color='#4C72B0', alpha=0.8)
ax.bar(x + w/2, means.loc[1.0], w, label='Merge (1)',    color='#DD8452', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'F{i}' for i in range(20)], rotation=45, fontsize=9)
ax.set_ylabel('Mean feature value')
ax.set_title('Mean Feature Values by Class Label', fontsize=13)
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_feature_means_by_label.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ── Save EDA summary stats ────────────────────────────────────────────────
summary = {
    'n_train': len(train),
    'n_test': len(test),
    'n_features': 20,
    'train_pos_rate': train['label'].mean(),
    'test_pos_rate': test['label'].mean(),
}
import json
with open('../outputs/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("EDA done:", summary)
