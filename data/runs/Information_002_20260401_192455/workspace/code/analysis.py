"""
Analysis of Hartree-Fock benchmark for paper 2111.01152 (AB-stacked MoTe2/WSe2 moiré system).
Parses YAML scores, computes statistics, and generates visualizations.
"""

import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Paths
BASE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_002_20260401_192455')
YAML_FILE = BASE / 'data/2111.01152/2111.01152.yaml'
OUTPUT_DIR = BASE / 'outputs'
IMG_DIR = BASE / 'report/images'

OUTPUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# ─── 1. Load YAML ───────────────────────────────────────────────────────────
with open(YAML_FILE) as f:
    raw = yaml.safe_load(f)

tasks = raw  # list of task dicts

# ─── 2. Extract per-task overall scores ─────────────────────────────────────
SCORE_DIMS = ['in_paper', 'prompt_quality', 'follow_instructions',
              'physics_logic', 'math_derivation', 'final_answer_accuracy']

task_records = []
for i, t in enumerate(tasks):
    name = t.get('task', f'Task {i+1}')
    sc = t.get('score', {})
    rec = {'task_index': i, 'task_name': name}
    total = 0
    n_dims = 0
    for dim in SCORE_DIMS:
        val = sc.get(dim, None)
        if isinstance(val, (int, float)):
            rec[dim] = val
            total += val
            n_dims += 1
        else:
            rec[dim] = np.nan
    rec['total_score'] = total
    rec['n_scored_dims'] = n_dims
    task_records.append(rec)

df_tasks = pd.DataFrame(task_records)
print("Task-level scores:\n", df_tasks[['task_name'] + SCORE_DIMS + ['total_score']].to_string())

# ─── 3. Extract placeholder-level scores (per annotator) ────────────────────
ph_records = []
ANNOTATORS = ['Haining', 'Will', 'Yasaman']

for i, t in enumerate(tasks):
    task_name = t.get('task', f'Task {i+1}')
    ph_dict = t.get('placeholder', {})
    if not ph_dict:
        continue
    for ph_key, ph_val in ph_dict.items():
        if not isinstance(ph_val, dict):
            continue
        sc_dict = ph_val.get('score', {})
        if not sc_dict:
            continue
        for ann in ANNOTATORS:
            raw_v = sc_dict.get(ann, None)
            # skip uncertain values like (?)
            if isinstance(raw_v, (int, float)):
                ph_records.append({
                    'task_index': i,
                    'task_name': task_name,
                    'placeholder': ph_key,
                    'annotator': ann,
                    'score': raw_v,
                })

df_ph = pd.DataFrame(ph_records)
print(f"\nPlaceholder-level records: {len(df_ph)}")
print(df_ph.groupby('annotator')['score'].describe())

# ─── 4. Figure 1: Overall task scores heatmap ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# Short task labels
short_labels = []
for nm in df_tasks['task_name']:
    words = nm.split()[:5]
    short_labels.append(' '.join(words))

heat_data = df_tasks[SCORE_DIMS].T
heat_data.columns = short_labels

# Mask NaN
mask = heat_data.isna()

sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=2,
            mask=mask, linewidths=0.5, ax=ax, cbar_kws={'label': 'Score (0–2)'})
ax.set_title('Per-Task Score Breakdown Across Quality Dimensions\n(Paper 2111.01152: MoTe₂/WSe₂ Hartree-Fock Benchmark)', fontsize=12)
ax.set_xlabel('Task (abbreviated)', fontsize=10)
ax.set_ylabel('Evaluation Dimension', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
ax.set_yticklabels([d.replace('_', ' ').title() for d in SCORE_DIMS], rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig1_task_score_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_task_score_heatmap.png")

# ─── 5. Figure 2: Total score per task (bar chart) ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#2196F3' if sc >= 10 else '#FF5722' if sc < 8 else '#FF9800'
          for sc in df_tasks['total_score']]
bars = ax.bar(range(len(df_tasks)), df_tasks['total_score'], color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
ax.axhline(df_tasks['total_score'].mean(), color='navy', linestyle='--', linewidth=1.5, label=f"Mean = {df_tasks['total_score'].mean():.1f}")
ax.set_xticks(range(len(df_tasks)))
ax.set_xticklabels([f"T{i+1}" for i in range(len(df_tasks))], fontsize=9)
ax.set_xlabel('Task Index', fontsize=11)
ax.set_ylabel('Total Score (sum over 6 dimensions, max=12)', fontsize=10)
ax.set_title('Total Evaluation Score per Hartree-Fock Derivation Step\n(Paper 2111.01152)', fontsize=12)
ax.set_ylim(0, 13)
ax.legend(fontsize=10)

# Add max possible marker
ax.axhline(12, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Max=12')

# Legend for colors
patch_high = mpatches.Patch(color='#2196F3', label='High (≥10)')
patch_mid  = mpatches.Patch(color='#FF9800', label='Medium (8–9)')
patch_low  = mpatches.Patch(color='#FF5722', label='Low (<8)')
ax.legend(handles=[patch_high, patch_mid, patch_low], loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig2_total_score_per_task.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_total_score_per_task.png")

# ─── 6. Figure 3: Score distribution per quality dimension ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for idx, dim in enumerate(SCORE_DIMS):
    ax = axes[idx]
    vals = df_tasks[dim].dropna()
    counts = vals.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='k', alpha=0.8)
    ax.set_title(dim.replace('_', ' ').title(), fontsize=11)
    ax.set_xlabel('Score', fontsize=9)
    ax.set_ylabel('Number of Tasks', fontsize=9)
    ax.set_xticks([0, 1, 2])
    mean_v = vals.mean()
    ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_v:.2f}')
    ax.legend(fontsize=8)

plt.suptitle('Score Distributions Across Quality Dimensions\n(15 HF Tasks, Paper 2111.01152)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig3_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_score_distributions.png")

# ─── 7. Figure 4: Annotator agreement on placeholder scores ─────────────────
if len(df_ph) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    palette = {'Haining': '#1565C0', 'Will': '#2E7D32', 'Yasaman': '#C62828'}

    for idx, ann in enumerate(ANNOTATORS):
        ax = axes[idx]
        sub = df_ph[df_ph['annotator'] == ann]['score']
        if len(sub) == 0:
            continue
        counts = sub.value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color=palette[ann], alpha=0.8, edgecolor='k')
        ax.set_title(f'Annotator: {ann}\n(n={len(sub)}, mean={sub.mean():.2f})', fontsize=11)
        ax.set_xlabel('Score (0=wrong, 1=partial, 2=correct)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xticks([0, 1, 2])
        # Add % labels
        total = len(sub)
        for bar in bars:
            h = bar.get_height()
            pct = 100 * h / total
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Placeholder-Level Score Distributions by Annotator\n(LLM vs Human Ground Truth)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'fig4_annotator_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_annotator_score_distributions.png")

# ─── 8. Figure 5: Annotator inter-rater analysis ────────────────────────────
if len(df_ph) > 0:
    # Pivot to get per-annotator score for each placeholder instance
    pivot = df_ph.pivot_table(index=['task_index', 'placeholder'], columns='annotator', values='score')
    pivot = pivot.dropna()

    if len(pivot) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        pairs = [('Haining', 'Will'), ('Haining', 'Yasaman'), ('Will', 'Yasaman')]
        pair_colors = ['#1565C0', '#C62828', '#2E7D32']

        for idx, (a1, a2) in enumerate(pairs):
            ax = axes[idx]
            if a1 not in pivot.columns or a2 not in pivot.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            x, y = pivot[a1].values, pivot[a2].values
            corr = np.corrcoef(x, y)[0, 1]
            # Jitter for visibility
            jx = x + np.random.uniform(-0.1, 0.1, len(x))
            jy = y + np.random.uniform(-0.1, 0.1, len(y))
            ax.scatter(jx, jy, alpha=0.5, color=pair_colors[idx], s=30)
            ax.set_xlabel(a1, fontsize=11)
            ax.set_ylabel(a2, fontsize=11)
            ax.set_title(f'{a1} vs {a2}\nr = {corr:.3f}', fontsize=11)
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            # Diagonal line
            ax.plot([0, 2], [0, 2], 'k--', alpha=0.4, linewidth=1)

        plt.suptitle('Inter-Annotator Agreement on Placeholder Scores\n(Jittered for Visibility)', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(IMG_DIR / 'fig5_inter_annotator_agreement.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved fig5_inter_annotator_agreement.png")

# ─── 9. Figure 6: Score radar chart per task type ───────────────────────────
# Group tasks by type
task_groups = {
    'Non-interacting H': [0, 1, 2, 3, 4, 5, 6, 7],  # kinetic, potential, 2nd quantized, FT, PH
    'Interaction & HF':  [8, 9, 10, 11, 12, 13, 14],  # interaction, Wick, extract, reduce, combine
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors_group = ['#1976D2', '#D32F2F']

for ax_idx, (group_name, indices) in enumerate(task_groups.items()):
    ax = axes[ax_idx]
    valid_idx = [i for i in indices if i < len(df_tasks)]
    sub = df_tasks.iloc[valid_idx]
    means = sub[SCORE_DIMS].mean()

    ax.bar(range(len(SCORE_DIMS)), means.values, color=colors_group[ax_idx], alpha=0.75, edgecolor='k')
    ax.axhline(2.0, color='green', linestyle=':', linewidth=1, alpha=0.6, label='Max=2')
    ax.set_xticks(range(len(SCORE_DIMS)))
    ax.set_xticklabels([d.replace('_', '\n') for d in SCORE_DIMS], fontsize=8)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel('Mean Score', fontsize=10)
    ax.set_title(f'{group_name}\n(n={len(valid_idx)} tasks)', fontsize=11)
    ax.legend(fontsize=9)
    for bar_i, val in enumerate(means.values):
        ax.text(bar_i, val + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Mean Scores by Task Group and Quality Dimension\n(Paper 2111.01152: MoTe₂/WSe₂ HF Benchmark)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig6_group_score_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_group_score_comparison.png")

# ─── 10. Save summary statistics ────────────────────────────────────────────
summary = {
    'n_tasks': len(df_tasks),
    'mean_total_score': float(df_tasks['total_score'].mean()),
    'std_total_score': float(df_tasks['total_score'].std()),
    'max_total_score': float(df_tasks['total_score'].max()),
    'min_total_score': float(df_tasks['total_score'].min()),
    'per_dim_means': {dim: float(df_tasks[dim].mean()) for dim in SCORE_DIMS},
    'annotator_mean_ph_score': {
        ann: float(df_ph[df_ph['annotator']==ann]['score'].mean())
        for ann in ANNOTATORS
    },
    'annotator_n_ph': {
        ann: int((df_ph['annotator']==ann).sum())
        for ann in ANNOTATORS
    },
}

with open(OUTPUT_DIR / 'summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

df_tasks.to_csv(OUTPUT_DIR / 'task_scores.csv', index=False)
df_ph.to_csv(OUTPUT_DIR / 'placeholder_scores.csv', index=False)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nAll outputs saved.")
