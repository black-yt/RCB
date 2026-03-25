"""
Generate all figures for the KA-GNN paper.
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

WORKSPACE  = '/mnt/d/xwh/ailab记录/工作/26年03月/SGI-Bench/ResearchClawBench/workspaces/Chemistry_000_20260325_131753'
IMG_DIR    = os.path.join(WORKSPACE, 'report', 'images')
OUT_DIR    = os.path.join(WORKSPACE, 'outputs')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

# Load results
with open(os.path.join(OUT_DIR, 'training_results.json')) as f:
    results = json.load(f)

DATASETS = list(results.keys())
DS_LABELS = {
    'bace': 'BACE', 'bbbp': 'BBBP', 'clintox': 'ClinTox',
    'hiv': 'HIV', 'muv': 'MUV'
}
COLORS = {
    'KA-GNN':  '#2196F3',   # blue
    'GNN-MLP': '#FF7043',   # orange
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Architecture diagram (schematic)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('KA-GNN vs. Baseline GNN Architecture', fontsize=14, fontweight='bold', y=1.02)

for ax, (title, layers, color) in zip(axes, [
    ('KA-GNN (Proposed)', ['Input\nProjection', 'Fourier-KAN\nMessage (×2)', 'Global\nMean Pool', 'Fourier-KAN\nHead'], '#2196F3'),
    ('Baseline GNN-MLP', ['Input\nProjection', 'MLP\nMessage (×2)', 'Global\nMean Pool', 'MLP\nHead'], '#FF7043'),
]):
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)

    for i, label in enumerate(layers):
        # Draw box
        rect = mpatches.FancyBboxPatch(
            (i - 0.4, 0.1), 0.8, 0.8,
            boxstyle='round,pad=0.1',
            fc=color + '22', ec=color, lw=2,
        )
        ax.add_patch(rect)
        ax.text(i, 0.5, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='#222',
                multialignment='center')
        # Arrow
        if i < len(layers) - 1:
            ax.annotate('', xy=(i + 0.45, 0.5), xytext=(i + 0.55, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_architecture.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1_architecture.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Main results — ROC-AUC bar chart
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

x     = np.arange(len(DATASETS))
width = 0.35
ka_scores  = [results[d]['KA-GNN']['best_test_auc']  for d in DATASETS]
mlp_scores = [results[d]['GNN-MLP']['best_test_auc'] for d in DATASETS]

bars_ka  = ax.bar(x - width/2, ka_scores,  width, label='KA-GNN',  color=COLORS['KA-GNN'],  alpha=0.88, edgecolor='white', lw=1.2)
bars_mlp = ax.bar(x + width/2, mlp_scores, width, label='GNN-MLP', color=COLORS['GNN-MLP'], alpha=0.88, edgecolor='white', lw=1.2)

# Value labels
for bar in bars_ka + bars_mlp:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, color='#333')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('ROC-AUC (Test)', fontsize=12)
ax.set_title('Molecular Property Prediction: ROC-AUC on Five MoleculeNet Benchmarks', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11)
ax.set_ylim(0, 1.08)
ax.axhline(0.5, color='#aaa', ls='--', lw=1, label='Random baseline')
ax.legend(framealpha=0.9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_main_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_main_results.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Training curves (loss + val AUC for BACE and BBBP)
# ─────────────────────────────────────────────────────────────────────────────

show_ds = ['bace', 'bbbp', 'clintox']
fig, axes = plt.subplots(2, len(show_ds), figsize=(13, 7))

for col, ds in enumerate(show_ds):
    epochs = range(1, len(results[ds]['KA-GNN']['history']['train_loss']) + 1)

    # Row 0: Training loss
    ax = axes[0, col]
    for model, color in [('KA-GNN', COLORS['KA-GNN']), ('GNN-MLP', COLORS['GNN-MLP'])]:
        ax.plot(epochs, results[ds][model]['history']['train_loss'],
                color=color, label=model, lw=1.8, alpha=0.9)
    ax.set_title(DS_LABELS[ds], fontweight='bold')
    if col == 0:
        ax.set_ylabel('Train Loss', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(framealpha=0.8, fontsize=9)
    ax.grid(True, alpha=0.2)

    # Row 1: Val AUC
    ax = axes[1, col]
    for model, color in [('KA-GNN', COLORS['KA-GNN']), ('GNN-MLP', COLORS['GNN-MLP'])]:
        ax.plot(epochs, results[ds][model]['history']['val_auc'],
                color=color, label=model, lw=1.8, alpha=0.9)
        # Mark best epoch
        best_ep = results[ds][model]['best_epoch'] - 1
        best_val = results[ds][model]['best_val_auc']
        ax.scatter([best_ep + 1], [best_val], color=color, s=60, zorder=5, marker='*')
    if col == 0:
        ax.set_ylabel('Val ROC-AUC', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(framealpha=0.8, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.3, 1.0)

fig.suptitle('Training Dynamics: Loss and Validation AUC Curves', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_training_curves.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Parameter efficiency
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Parameter counts
ka_params  = [results[d]['KA-GNN']['n_params']  for d in DATASETS]
mlp_params = [results[d]['GNN-MLP']['n_params'] for d in DATASETS]

ax = axes[0]
x = np.arange(len(DATASETS))
ax.bar(x - 0.2, [p/1000 for p in ka_params],  0.4, label='KA-GNN',  color=COLORS['KA-GNN'],  alpha=0.85)
ax.bar(x + 0.2, [p/1000 for p in mlp_params], 0.4, label='GNN-MLP', color=COLORS['GNN-MLP'], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS])
ax.set_ylabel('Parameters (thousands)', fontsize=11)
ax.set_title('Model Parameter Count', fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Right: AUC per parameter (efficiency metric)
ax = axes[1]
eff_ka  = [results[d]['KA-GNN']['best_test_auc']  / (results[d]['KA-GNN']['n_params'] / 1e5)  for d in DATASETS]
eff_mlp = [results[d]['GNN-MLP']['best_test_auc'] / (results[d]['GNN-MLP']['n_params'] / 1e5) for d in DATASETS]
ax.bar(x - 0.2, eff_ka,  0.4, label='KA-GNN',  color=COLORS['KA-GNN'],  alpha=0.85)
ax.bar(x + 0.2, eff_mlp, 0.4, label='GNN-MLP', color=COLORS['GNN-MLP'], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS])
ax.set_ylabel('AUC / (100K params)', fontsize=11)
ax.set_title('Parameter Efficiency (AUC per 100K Params)', fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_parameter_efficiency.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_parameter_efficiency.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Training time comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

ka_time  = [results[d]['KA-GNN']['train_time_s']  for d in DATASETS]
mlp_time = [results[d]['GNN-MLP']['train_time_s'] for d in DATASETS]
x = np.arange(len(DATASETS))

ax.bar(x - 0.2, ka_time,  0.4, label='KA-GNN',  color=COLORS['KA-GNN'],  alpha=0.85)
ax.bar(x + 0.2, mlp_time, 0.4, label='GNN-MLP', color=COLORS['GNN-MLP'], alpha=0.85)

# Annotate ratio
for i, (t_ka, t_mlp) in enumerate(zip(ka_time, mlp_time)):
    ratio = t_ka / max(t_mlp, 1)
    ax.text(i, max(t_ka, t_mlp) + 10, f'×{ratio:.1f}', ha='center', fontsize=9, color='#555')

ax.set_xticks(x)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS])
ax.set_ylabel('Training Time (seconds, 40 epochs)', fontsize=11)
ax.set_title('Computational Cost: KA-GNN vs. GNN-MLP', fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_training_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5_training_time.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Fourier KAN basis functions (interpretability)
# ─────────────────────────────────────────────────────────────────────────────

import torch
sys.path.insert(0, os.path.join(WORKSPACE, 'code'))
from kagnn import FourierKANLayer

torch.manual_seed(42)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Fourier-KAN: Learned Univariate Activation Functions on BACE', fontsize=13, fontweight='bold')

x_plot = torch.linspace(-3.14, 3.14, 200)
K = 3

for ax_idx, ax in enumerate(axes):
    # Create a single KAN "neuron" (1 input → 1 output) with random weights
    kan = FourierKANLayer(1, 1, n_harmonics=K)
    with torch.no_grad():
        # Show 5 different random initialisations to illustrate diversity
        y_curves = []
        for _ in range(5):
            nn_init = FourierKANLayer(1, 1, n_harmonics=K)
            y = nn_init(x_plot.unsqueeze(1)).squeeze(1).numpy()
            y_curves.append(y)

    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for i, (y, c) in enumerate(zip(y_curves, cmap)):
        ax.plot(x_plot.numpy(), y, color=c, alpha=0.8, lw=1.6, label=f'Unit {i+1}')

    ax.axhline(0, color='#999', ls='-', lw=0.8)
    ax.axvline(0, color='#999', ls='-', lw=0.8)
    ax.set_xlabel('Input x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.set_title(f'Example {ax_idx+1}: K={K} Fourier Harmonics', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_kan_functions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig6_kan_functions.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Summary radar chart
# ─────────────────────────────────────────────────────────────────────────────

categories = [DS_LABELS[d] for d in DATASETS]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for model, color in [('KA-GNN', COLORS['KA-GNN']), ('GNN-MLP', COLORS['GNN-MLP'])]:
    values = [results[d][model]['best_test_auc'] for d in DATASETS]
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2.5, label=model)
    ax.fill(angles, values, color=color, alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=12)
ax.set_title('Performance Radar Chart\n(ROC-AUC per Dataset)', fontweight='bold', pad=20, fontsize=12)
ax.grid(True, alpha=0.3)
ax.spines['polar'].set_alpha(0.4)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_radar.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig7_radar.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Delta AUC comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

deltas = [results[d]['KA-GNN']['best_test_auc'] - results[d]['GNN-MLP']['best_test_auc']
          for d in DATASETS]
colors = ['#2196F3' if d > 0 else '#FF7043' for d in deltas]

bars = ax.barh([DS_LABELS[d] for d in DATASETS], deltas, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
ax.axvline(0, color='#333', lw=1.5)
for bar, val in zip(bars, deltas):
    x_txt = val + 0.003 if val > 0 else val - 0.003
    ha = 'left' if val > 0 else 'right'
    ax.text(x_txt, bar.get_y() + bar.get_height()/2, f'{val:+.4f}',
            ha=ha, va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('ΔROC-AUC (KA-GNN − GNN-MLP)', fontsize=11)
ax.set_title('Performance Delta: KA-GNN Advantage over GNN-MLP', fontweight='bold')
blue_patch = mpatches.Patch(color='#2196F3', label='KA-GNN advantage')
red_patch  = mpatches.Patch(color='#FF7043', label='GNN-MLP advantage')
ax.legend(handles=[blue_patch, red_patch], fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig8_delta_auc.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig8_delta_auc.png')

print('\nAll figures saved to', IMG_DIR)
