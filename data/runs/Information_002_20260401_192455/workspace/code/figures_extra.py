"""Additional figure: HF derivation pipeline visualization."""

import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BASE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_002_20260401_192455')
IMG_DIR = BASE / 'report/images'
OUTPUT_DIR = BASE / 'outputs'

# Reload task scores
df_tasks = pd.read_csv(OUTPUT_DIR / 'task_scores.csv')

# ─── Figure 7: HF Pipeline diagram with scores ──────────────────────────────
# Define pipeline stages
stages = [
    ("1. Kinetic\nHamiltonian\n(matrix)", 8),
    ("2. Kinetic H\nterms defined", 10),
    ("3. Potential\nHamiltonian\n(matrix)", 11),
    ("4. Potential H\nterms defined", 12),
    ("5. 2nd-Quantized\nH (matrix)", 12),
    ("6. 2nd-Quantized\nH (summation)", 10),
    ("7. FT to\nMomentum\nSpace", 11),
    ("8. Particle-hole\nTransformation", 10),
    ("9. Simplify\nHole Basis", 11),
    ("10. Interaction\nHamiltonian", 12),
    ("11. Wick's\nTheorem", 10),
    ("12. Extract\nQuadratic", 10),
    ("13. Swap Index\n(combine)", 11),
    ("14. Reduce\nHartree\nMomentum", 11),
    ("15. Reduce\nFock\nMomentum", 12),
    ("16. Combine\nHartree+Fock", 12),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Part A: Color-coded pipeline
N = len(stages)
cols = 8
rows = (N + cols - 1) // cols

def score_color(s):
    if s >= 11: return '#1B5E20', '#E8F5E9'  # dark/light green
    if s >= 9:  return '#E65100', '#FFF3E0'  # dark/light orange
    return       '#B71C1C', '#FFEBEE'         # dark/light red

for i, (label, score) in enumerate(stages):
    col = i % cols
    row = i // cols
    x = col * 2.1
    y = -(row * 2.8)
    fc, ec_light = score_color(score)
    box = FancyBboxPatch((x, y), 1.85, 2.3, boxstyle='round,pad=0.08',
                          facecolor=ec_light, edgecolor=fc, linewidth=1.5, transform=ax1.transData)
    ax1.add_patch(box)
    ax1.text(x + 0.925, y + 1.3, label, ha='center', va='center',
             fontsize=7.5, color=fc, fontweight='bold', multialignment='center')
    ax1.text(x + 0.925, y + 0.25, f"Score: {score}/12", ha='center', va='center',
             fontsize=8.5, color=fc, fontweight='bold')

# Connect arrows (within same row)
for i in range(N - 1):
    col_cur = i % cols
    row_cur = i // cols
    col_nxt = (i+1) % cols
    row_nxt = (i+1) // cols
    x_cur = col_cur * 2.1 + 1.85
    y_cur = -(row_cur * 2.8) + 1.15
    x_nxt = col_nxt * 2.1
    y_nxt = -(row_nxt * 2.8) + 1.15
    if row_cur == row_nxt:
        ax1.annotate('', xy=(x_nxt, y_nxt), xytext=(x_cur, y_cur),
                     arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
    else:
        # Row break: draw down then across
        ax1.annotate('', xy=(col_cur*2.1+1.85, y_nxt), xytext=(x_cur, y_cur),
                     arrowprops=dict(arrowstyle='->', color='#555', lw=1.0, connectionstyle='arc3,rad=0'))

ax1.set_xlim(-0.2, cols * 2.1 + 0.2)
ax1.set_ylim(-rows * 2.8 - 0.5, 2.8)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Hartree-Fock Derivation Pipeline with LLM Performance Scores\n(Paper 2111.01152: AB-stacked MoTe₂/WSe₂ Moiré System)', fontsize=12, pad=10)

# Legend
legend_patches = [
    mpatches.Patch(color='#E8F5E9', label='High (11–12)', ec='#1B5E20', lw=1.5),
    mpatches.Patch(color='#FFF3E0', label='Medium (9–10)', ec='#E65100', lw=1.5),
    mpatches.Patch(color='#FFEBEE', label='Low (<9)', ec='#B71C1C', lw=1.5),
]
ax1.legend(handles=legend_patches, loc='upper right', fontsize=9, framealpha=0.9)

# Part B: Score trend line
task_scores = [s for _, s in stages]
ax2.plot(range(1, N+1), task_scores, 'o-', color='#1565C0', linewidth=2, markersize=7, zorder=3)
ax2.fill_between(range(1, N+1), task_scores, alpha=0.15, color='#1565C0')
ax2.axhline(12, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Max score (12)')
ax2.axhline(np.mean(task_scores), color='red', linestyle='--', alpha=0.7, linewidth=1.5,
            label=f'Mean = {np.mean(task_scores):.1f}')
ax2.set_xlabel('HF Derivation Step', fontsize=11)
ax2.set_ylabel('Total Score (/12)', fontsize=11)
ax2.set_title('Score Trend Across HF Derivation Steps', fontsize=11)
ax2.set_xticks(range(1, N+1))
ax2.set_xticklabels([f'S{i}' for i in range(1, N+1)], fontsize=8)
ax2.set_ylim(0, 13)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Color-code x-axis labels
for tick_idx, tick in enumerate(ax2.get_xticklabels()):
    s = task_scores[tick_idx]
    if s >= 11: tick.set_color('#1B5E20')
    elif s >= 9: tick.set_color('#E65100')
    else: tick.set_color('#B71C1C')

plt.tight_layout(h_pad=2)
plt.savefig(IMG_DIR / 'fig7_hf_pipeline_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_hf_pipeline_scores.png")

# ─── Figure 8: Placeholder type analysis ────────────────────────────────────
df_ph = pd.read_csv(OUTPUT_DIR / 'placeholder_scores.csv')
ANNOTATORS = ['Haining', 'Will', 'Yasaman']

# Mean score per task, per annotator
task_means = df_ph.groupby(['task_index', 'annotator'])['score'].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
palette = {'Haining': '#1565C0', 'Will': '#2E7D32', 'Yasaman': '#C62828'}
offsets = {'Haining': -0.25, 'Will': 0, 'Yasaman': 0.25}

task_ids = sorted(task_means['task_index'].unique())
for ann in ANNOTATORS:
    sub = task_means[task_means['annotator'] == ann]
    sub = sub.set_index('task_index').reindex(task_ids)
    x_vals = [tid + offsets[ann] for tid in task_ids]
    ax.bar(x_vals, sub['score'].values, width=0.22, color=palette[ann], alpha=0.8,
           edgecolor='k', linewidth=0.4, label=ann)

ax.set_xticks(task_ids)
ax.set_xticklabels([f'T{tid}' for tid in task_ids], fontsize=8)
ax.set_xlabel('Task Index', fontsize=11)
ax.set_ylabel('Mean Placeholder Score (0–2)', fontsize=10)
ax.set_title('Mean Placeholder-Level Accuracy per Task and Annotator\n(LLM vs Ground Truth Comparison)', fontsize=12)
ax.legend(title='Annotator', fontsize=9)
ax.set_ylim(0, 2.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig8_per_task_annotator_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_per_task_annotator_scores.png")

print("All additional figures saved.")
