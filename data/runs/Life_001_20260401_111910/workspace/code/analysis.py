import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

base = Path(__file__).resolve().parents[1]
data_dir = base / 'data'
outputs_dir = base / 'outputs'
fig_dir = base / 'report' / 'images'
outputs_dir.mkdir(exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Load core data
cell_pops = pd.read_csv(data_dir / 'cell-populations.csv')
final_resp = pd.read_csv(data_dir / 'final-response-likelihoods.csv')
opt_runtime = pd.read_csv(data_dir / 'optimization_runtime_data.csv')
sel_vax_elems = pd.read_csv(data_dir / 'selected-vaccine-elements.budget-10.minsum.adaptive.csv')
sim_resp = pd.read_csv(data_dir / 'sim-specific-response-likelihoods.csv')
vax = pd.read_csv(data_dir / 'vaccine.budget-10.minsum.adaptive.csv')

# vaccine element score replicates
score_files = sorted(data_dir.glob('vaccine-elements.scores.100-cells.10x.rep-*.csv'))
score_dfs = []
for f in score_files:
    rep = int(f.stem.split('rep-')[-1])
    df = pd.read_csv(f)
    df['repetition'] = rep
    score_dfs.append(df)
score_all = pd.concat(score_dfs, ignore_index=True)

# 2. Data overview plots
sns.set(style='whitegrid')

# Distribution of p_response per population/vaccine
plt.figure(figsize=(6,4))
sns.histplot(final_resp['p_response'], bins=30, kde=True)
plt.xlabel('Per-cell immune response probability')
plt.ylabel('Count')
plt.title('Distribution of per-cell response probabilities')
plt.tight_layout()
plt.savefig(fig_dir / 'fig1_p_response_distribution.png', dpi=300)
plt.close()

# Coverage curve: fraction of cells above threshold
thresholds = np.linspace(0,1,101)
frac_above = [(final_resp['p_response'] >= t).mean() for t in thresholds]
plt.figure(figsize=(6,4))
plt.plot(thresholds, frac_above)
plt.xlabel('Response probability threshold')
plt.ylabel('Fraction of cells')
plt.title('Coverage as a function of response probability threshold')
plt.tight_layout()
plt.savefig(fig_dir / 'fig2_coverage_curve.png', dpi=300)
plt.close()

# Optimization runtime vs population size
plt.figure(figsize=(6,4))
for sid, sub in opt_runtime.groupby('SampleID'):
    plt.plot(sub['PopulationSize'], sub['RunTime'], marker='o', label=str(sid), alpha=0.6)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Population size (cells)')
plt.ylabel('Runtime (s)')
plt.title('Optimization runtime vs population size')
plt.legend(title='Sample', fontsize=6)
plt.tight_layout()
plt.savefig(fig_dir / 'fig3_runtime_scaling.png', dpi=300)
plt.close()

# 3. Vaccine efficacy metrics using selected elements (budget-10 minsum adaptive)
selected_peptides = set(vax['peptide'])

# Compute per-cell response probability under the 10-element vaccine for each repetition
# Assuming independence across vaccine elements and using p_no_response multiplicatively
sub_scores = score_all[score_all['vaccine_element'].isin(selected_peptides)].copy()
sub_scores['p_no_response'] = 1 - sub_scores['p_response']

agg = sub_scores.groupby(['repetition','cell_id'])['p_no_response'].prod().reset_index()
agg['p_response_combo'] = 1 - agg['p_no_response']

# Save per-cell combined probabilities
agg.to_csv(outputs_dir / 'per_cell_p_response_combo.csv', index=False)

# Summary statistics across cells and repetitions
summary = agg.groupby('repetition')['p_response_combo'].agg(['mean','median','min','max']).reset_index()
summary.to_csv(outputs_dir / 'per_cell_p_response_summary.csv', index=False)

# Plot distribution of combined response probabilities
plt.figure(figsize=(6,4))
sns.histplot(agg['p_response_combo'], bins=30, kde=True)
plt.xlabel('Per-cell response probability (combined vaccine)')
plt.ylabel('Count')
plt.title('Distribution of combined per-cell response probabilities')
plt.tight_layout()
plt.savefig(fig_dir / 'fig4_combo_p_response_distribution.png', dpi=300)
plt.close()

# Coverage ratio of tumor cells for a chosen threshold, e.g., 0.5
thr = 0.5
coverage_per_rep = agg.groupby('repetition').apply(lambda x: (x['p_response_combo']>=thr).mean()).reset_index(name='coverage')
coverage_per_rep.to_csv(outputs_dir / 'coverage_per_rep_thr0.5.csv', index=False)

plt.figure(figsize=(6,4))
sns.barplot(data=coverage_per_rep, x='repetition', y='coverage', color='steelblue')
plt.ylim(0,1)
plt.xlabel('Repetition')
plt.ylabel('Coverage (p_response_combo >= 0.5)')
plt.title('Coverage of tumor cells by optimized vaccine')
plt.tight_layout()
plt.savefig(fig_dir / 'fig5_coverage_bar_thr0.5.png', dpi=300)
plt.close()

# 4. IoU of optimal vaccine compositions across repetitions
sel_full = sel_vax_elems.groupby('repetition')['peptide'].apply(set).to_dict()
reps = sorted(sel_full.keys())

records = []
for i in range(len(reps)):
    for j in range(i+1, len(reps)):
        r1, r2 = reps[i], reps[j]
        s1, s2 = sel_full[r1], sel_full[r2]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        iou = inter / union if union>0 else np.nan
        records.append({'rep1': r1, 'rep2': r2, 'IoU': iou, 'intersection': inter, 'union': union})

ious = pd.DataFrame(records)
ious.to_csv(outputs_dir / 'vaccine_iou_pairs.csv', index=False)

plt.figure(figsize=(6,4))
plt.hist(ious['IoU'], bins=20)
plt.xlabel('IoU between optimal vaccine compositions')
plt.ylabel('Number of repetition pairs')
plt.title('Stability of optimized vaccine composition (IoU across repetitions)')
plt.tight_layout()
plt.savefig(fig_dir / 'fig6_iou_distribution.png', dpi=300)
plt.close()

# 5. Save high-level metrics
metrics = {
    'mean_per_cell_p_response_combo': float(agg['p_response_combo'].mean()),
    'median_per_cell_p_response_combo': float(agg['p_response_combo'].median()),
    'mean_coverage_thr0.5': float(coverage_per_rep['coverage'].mean()),
    'std_coverage_thr0.5': float(coverage_per_rep['coverage'].std()),
    'mean_iou': float(ious['IoU'].mean()),
    'std_iou': float(ious['IoU'].std()),
}

pd.Series(metrics).to_csv(outputs_dir / 'summary_metrics.csv', header=False)
