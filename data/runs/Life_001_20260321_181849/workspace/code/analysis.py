"""
NeoAgDT Analysis: Personalized Neoantigen Vaccine Optimization
Reproduces key analyses from the NeoAgDT paper using provided simulation data.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
DATA = WORKSPACE / "data"
IMAGES = WORKSPACE / "report" / "images"
OUTPUTS = WORKSPACE / "outputs"
IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

# Patient sample IDs (from the paper)
PATIENT_IDS = ['3812', '3942', '3948', '3978', '3995', '4007', '4032']

# ============================================================
# 1. Load all data
# ============================================================
print("Loading data...")

# Cell populations
cell_pop = pd.read_csv(DATA / "cell-populations.csv")
print(f"  cell-populations: {cell_pop.shape}")

# Final response likelihoods
final_resp = pd.read_csv(DATA / "final-response-likelihoods.csv")
print(f"  final-response-likelihoods: {final_resp.shape}")

# Simulation-specific response likelihoods
sim_resp = pd.read_csv(DATA / "sim-specific-response-likelihoods.csv")
print(f"  sim-specific-response-likelihoods: {sim_resp.shape}")

# Optimization runtime
runtime_data = pd.read_csv(DATA / "optimization_runtime_data.csv")
print(f"  optimization_runtime_data: {runtime_data.shape}")

# Selected vaccine elements
selected_elems = pd.read_csv(DATA / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
print(f"  selected-vaccine-elements: {selected_elems.shape}")

# Vaccine composition
vaccine_comp = pd.read_csv(DATA / "vaccine.budget-10.minsum.adaptive.csv")
print(f"  vaccine.budget-10: {vaccine_comp.shape}")

# Vaccine element scores (10 replicates)
rep_scores = {}
for i in range(10):
    fname = f"vaccine-elements.scores.100-cells.10x.rep-{i}.csv"
    rep_scores[i] = pd.read_csv(DATA / fname)
print(f"  vaccine-element scores: {len(rep_scores)} replicates, {rep_scores[0].shape[0]} rows each")

# ============================================================
# 2. Data Overview
# ============================================================
print("\n--- Data Overview ---")

# Cell population structure
n_reps = cell_pop['repetition'].nunique()
n_cells_per_rep = cell_pop.groupby('repetition')['cell_ids'].nunique().mean()
n_mutations = cell_pop['mutation'].nunique()
n_peptides = cell_pop['presented_peptides'].nunique()
n_hlas = cell_pop['presented_hlas'].nunique()

print(f"Repetitions: {n_reps}")
print(f"Avg cells per repetition: {n_cells_per_rep:.0f}")
print(f"Unique mutations: {n_mutations}")
print(f"Unique peptides: {n_peptides}")
print(f"Unique HLA alleles: {n_hlas}")

# Mutations in vaccine
vaccine_mutations = vaccine_comp['peptide'].tolist()
print(f"\nVaccine composition (budget=10): {vaccine_mutations}")

# Mutation frequency across cells
mut_freq = cell_pop.groupby(['repetition', 'mutation'])['cell_ids'].nunique().reset_index()
mut_freq.columns = ['repetition', 'mutation', 'n_cells']
avg_mut_freq = mut_freq.groupby('mutation')['n_cells'].mean().sort_values(ascending=False)
print(f"\nAverage mutation frequency (cells presenting):")
print(avg_mut_freq.to_string())

# Save overview
overview = {
    'n_repetitions': int(n_reps),
    'avg_cells_per_rep': float(n_cells_per_rep),
    'n_unique_mutations': int(n_mutations),
    'n_unique_peptides': int(n_peptides),
    'n_unique_hlas': int(n_hlas),
    'vaccine_elements': vaccine_mutations,
}
pd.DataFrame([overview]).to_csv(OUTPUTS / "data_overview.csv", index=False)

# ============================================================
# 3. Figure 1: Response Probability Distributions (Violin plots)
# ============================================================
print("\nGenerating Figure 1: Response probability distributions...")

# The data has 100 cells x 10 reps = 1000 total observations
# Parse population field to extract repetition info
final_resp['rep'] = final_resp['population'].str.extract(r',\s*(\d+)').astype(int)

fig, ax = plt.subplots(figsize=(10, 6))

# Create violin plot - data represents response probabilities across cells
# For the paper figure, each violin represents a "patient" but our data has one simulation
# We'll use the 10 repetitions to show distribution variability
vp = ax.violinplot(
    [final_resp[final_resp['rep'] == r]['p_response'].values for r in range(10)],
    positions=range(10),
    showmeans=True,
    showmedians=True,
)

for body in vp['bodies']:
    body.set_facecolor('#4C72B0')
    body.set_alpha(0.7)
vp['cmeans'].set_color('red')
vp['cmedians'].set_color('black')

ax.set_xlabel('Simulation Repetition')
ax.set_ylabel('Immune Response Probability')
ax.set_title('Distribution of Per-Cell Immune Response Probabilities\n(MinSum Objective, Budget=10, 100 Cells × 10 Repetitions)')
ax.set_xticks(range(10))
ax.set_xticklabels([f'Rep {i}' for i in range(10)], rotation=45)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(IMAGES / "response_distribution.png", bbox_inches='tight')
plt.close()
print("  Saved response_distribution.png")

# ============================================================
# 4. Figure 2: Coverage Curves
# ============================================================
print("\nGenerating Figure 2: Coverage curves...")

thresholds = np.linspace(0, 1, 101)

fig, ax = plt.subplots(figsize=(10, 6))

# Compute coverage for each repetition
coverage_by_rep = np.zeros((10, len(thresholds)))
for r in range(10):
    rep_data = final_resp[final_resp['rep'] == r]['p_response'].values
    n_cells = len(rep_data)
    for t_idx, thresh in enumerate(thresholds):
        coverage_by_rep[r, t_idx] = np.sum(rep_data >= thresh) / n_cells

# Mean and 95% CI
mean_coverage = coverage_by_rep.mean(axis=0)
std_coverage = coverage_by_rep.std(axis=0)
ci95 = 1.96 * std_coverage / np.sqrt(10)

ax.plot(thresholds, mean_coverage, 'b-', linewidth=2, label='Mean coverage')
ax.fill_between(thresholds, mean_coverage - ci95, mean_coverage + ci95,
                alpha=0.3, color='blue', label='95% CI')

# Also plot individual repetitions as thin lines
for r in range(10):
    ax.plot(thresholds, coverage_by_rep[r], 'b-', alpha=0.15, linewidth=0.5)

ax.set_xlabel('Response Probability Threshold')
ax.set_ylabel('Coverage Ratio (Fraction of Cells Above Threshold)')
ax.set_title('Coverage Curve: Fraction of Cells with Response Probability ≥ Threshold\n(MinSum, Budget=10, 10 Repetitions)')
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(IMAGES / "coverage_curve.png", bbox_inches='tight')
plt.close()
print("  Saved coverage_curve.png")

# ============================================================
# 5. Figure 3: Optimization Runtime vs Population Size
# ============================================================
print("\nGenerating Figure 3: Optimization runtime...")

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.Set2(np.linspace(0, 1, len(PATIENT_IDS)))
markers = ['o', 's', '^', 'D', 'v', 'P', '*']

for idx, sample_id in enumerate(PATIENT_IDS):
    sample_data = runtime_data[runtime_data['SampleID'] == int(sample_id)]
    ax.plot(sample_data['PopulationSize'], sample_data['RunTime'],
            marker=markers[idx], color=colors[idx], linewidth=2,
            markersize=8, label=f'Patient {sample_id}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Population Size (Number of Simulated Cells)')
ax.set_ylabel('Optimization Runtime (seconds)')
ax.set_title('NeoAgDT Optimization Runtime vs. Simulated Cell Population Size')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES / "runtime.png", bbox_inches='tight')
plt.close()
print("  Saved runtime.png")

# ============================================================
# 6. Figure 4: Vaccine Element Scores Heatmap
# ============================================================
print("\nGenerating Figure 4: Vaccine element scores heatmap...")

# Aggregate across replicates: mean response probability per vaccine element
all_scores = []
for i in range(10):
    df = rep_scores[i].copy()
    df['replicate'] = i
    all_scores.append(df)
all_scores = pd.concat(all_scores, ignore_index=True)

# Pivot: mean p_response per vaccine_element across cells
pivot = all_scores.groupby('vaccine_element')['p_response'].agg(['mean', 'std']).reset_index()
pivot.columns = ['vaccine_element', 'mean_p_response', 'std_p_response']
pivot = pivot.sort_values('mean_p_response', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(pivot['vaccine_element'], pivot['mean_p_response'],
               xerr=pivot['std_p_response'], color='#4C72B0', alpha=0.8,
               capsize=3, edgecolor='white')
ax.set_xlabel('Mean Response Probability')
ax.set_ylabel('Vaccine Element (Mutation)')
ax.set_title('Per-Vaccine-Element Mean Response Probability\n(Averaged Across All Cells and 10 Replicates)')
ax.set_xlim(0, max(pivot['mean_p_response'] + pivot['std_p_response']) * 1.1)

plt.tight_layout()
plt.savefig(IMAGES / "vaccine_element_scores.png", bbox_inches='tight')
plt.close()
print("  Saved vaccine_element_scores.png")

# ============================================================
# 7. Figure 5: Per-cell response heatmap (rep-0)
# ============================================================
print("\nGenerating Figure 5: Per-cell response heatmap...")

rep0 = rep_scores[0]
# Create a pivot table: cells × vaccine elements
heatmap_data = rep0.pivot_table(index='cell_id', columns='vaccine_element',
                                 values='p_response', aggfunc='first')
# Sort by mean response
heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]

# Subsample for visibility if needed
n_cells_show = min(50, len(heatmap_data))
heatmap_sub = heatmap_data.iloc[:n_cells_show]

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_sub, cmap='YlOrRd', vmin=0, vmax=1,
            xticklabels=True, yticklabels=5, ax=ax,
            cbar_kws={'label': 'Response Probability'})
ax.set_xlabel('Vaccine Element (Mutation)')
ax.set_ylabel('Cell ID')
ax.set_title('Per-Cell Response Probability Heatmap\n(Top 50 Cells by Mean Response, Replicate 0)')

plt.tight_layout()
plt.savefig(IMAGES / "cell_response_heatmap.png", bbox_inches='tight')
plt.close()
print("  Saved cell_response_heatmap.png")

# ============================================================
# 8. Figure 6: Vaccine composition stability across replicates
# ============================================================
print("\nGenerating Figure 6: Vaccine composition stability...")

# Check which mutations are selected in each replicate
rep_vaccine = selected_elems.groupby('repetition')['peptide'].apply(set).reset_index()
rep_vaccine.columns = ['repetition', 'elements']

# Compute IoU between all pairs of replicates
n = len(rep_vaccine)
iou_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        s1 = rep_vaccine.iloc[i]['elements']
        s2 = rep_vaccine.iloc[j]['elements']
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        iou_matrix[i, j] = intersection / union if union > 0 else 0

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(iou_matrix, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[f'Rep {i}' for i in range(n)],
            yticklabels=[f'Rep {i}' for i in range(n)],
            vmin=0, vmax=1, ax=ax)
ax.set_title('Vaccine Composition Stability: IoU Between Replicates\n(MinSum, Budget=10)')

plt.tight_layout()
plt.savefig(IMAGES / "vaccine_iou.png", bbox_inches='tight')
plt.close()
print("  Saved vaccine_iou.png")

# Compute average IoU (excluding diagonal)
mask = ~np.eye(n, dtype=bool)
avg_iou = iou_matrix[mask].mean()
print(f"  Average IoU between replicates: {avg_iou:.4f}")

# ============================================================
# 9. Figure 7: Mutation presentation landscape
# ============================================================
print("\nGenerating Figure 7: Mutation presentation landscape...")

# How many cells present each mutation, by HLA allele
mut_hla = cell_pop[cell_pop['repetition'] == 0].groupby(
    ['mutation', 'presented_hlas']
)['cell_ids'].nunique().reset_index()
mut_hla.columns = ['mutation', 'hla', 'n_cells']

pivot_mh = mut_hla.pivot_table(index='mutation', columns='hla',
                                values='n_cells', fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot_mh, cmap='YlGnBu', annot=True, fmt='.0f', ax=ax,
            cbar_kws={'label': 'Number of Cells'})
ax.set_title('Mutation × HLA Allele Presentation Landscape\n(Repetition 0, 100 Cells)')
ax.set_xlabel('HLA Allele')
ax.set_ylabel('Mutation')

plt.tight_layout()
plt.savefig(IMAGES / "mutation_hla_landscape.png", bbox_inches='tight')
plt.close()
print("  Saved mutation_hla_landscape.png")

# ============================================================
# 10. Figure 8: Peptide diversity per cell
# ============================================================
print("\nGenerating Figure 8: Peptide diversity per cell...")

peptides_per_cell = cell_pop[cell_pop['repetition'] == 0].groupby('cell_ids').agg(
    n_peptides=('presented_peptides', 'nunique'),
    n_mutations=('mutation', 'nunique'),
    n_hlas=('presented_hlas', 'nunique')
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(peptides_per_cell['n_peptides'], bins=20, color='#4C72B0', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Number of Presented Peptides')
axes[0].set_ylabel('Number of Cells')
axes[0].set_title('Peptide Diversity per Cell')

axes[1].hist(peptides_per_cell['n_mutations'], bins=range(0, peptides_per_cell['n_mutations'].max()+2),
             color='#55A868', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Number of Mutations Represented')
axes[1].set_ylabel('Number of Cells')
axes[1].set_title('Mutation Diversity per Cell')

axes[2].hist(peptides_per_cell['n_hlas'], bins=range(0, peptides_per_cell['n_hlas'].max()+2),
             color='#C44E52', edgecolor='white', alpha=0.8)
axes[2].set_xlabel('Number of HLA Alleles')
axes[2].set_ylabel('Number of Cells')
axes[2].set_title('HLA Allele Usage per Cell')

plt.suptitle('Cell-Level Antigen Presentation Diversity (Repetition 0)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(IMAGES / "cell_diversity.png", bbox_inches='tight')
plt.close()
print("  Saved cell_diversity.png")

# ============================================================
# 11. Quantitative Summaries
# ============================================================
print("\n--- Quantitative Summaries ---")

# Response probability statistics
resp_stats = final_resp['p_response'].describe()
print(f"\nResponse Probability Statistics:")
print(resp_stats)

# Coverage at key thresholds
key_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
print(f"\nCoverage at key thresholds:")
for t in key_thresholds:
    cov = (final_resp['p_response'] >= t).mean()
    print(f"  ≥{t:.2f}: {cov:.4f} ({cov*100:.1f}%)")

# Runtime summary
rt_summary = runtime_data.groupby('PopulationSize')['RunTime'].agg(['mean', 'std', 'min', 'max'])
print(f"\nRuntime Summary by Population Size:")
print(rt_summary)

# Save summaries
resp_stats.to_csv(OUTPUTS / "response_probability_stats.csv")
rt_summary.to_csv(OUTPUTS / "runtime_summary.csv")

# Coverage data
cov_df = pd.DataFrame({
    'threshold': key_thresholds,
    'coverage': [(final_resp['p_response'] >= t).mean() for t in key_thresholds]
})
cov_df.to_csv(OUTPUTS / "coverage_at_thresholds.csv", index=False)

# Vaccine element selection frequency
elem_freq = selected_elems.groupby('peptide')['repetition'].count().reset_index()
elem_freq.columns = ['mutation', 'times_selected']
elem_freq = elem_freq.sort_values('times_selected', ascending=False)
elem_freq.to_csv(OUTPUTS / "vaccine_element_frequency.csv", index=False)
print(f"\nVaccine Element Selection Frequency:")
print(elem_freq.to_string(index=False))

# IoU statistics
print(f"\nVaccine Composition IoU:")
print(f"  Mean IoU: {avg_iou:.4f}")
print(f"  Min IoU: {iou_matrix[mask].min():.4f}")
print(f"  Max IoU: {iou_matrix[mask].max():.4f}")

iou_stats = {
    'mean_iou': avg_iou,
    'min_iou': iou_matrix[mask].min(),
    'max_iou': iou_matrix[mask].max(),
}
pd.DataFrame([iou_stats]).to_csv(OUTPUTS / "iou_statistics.csv", index=False)

print("\n=== Analysis complete! ===")
print(f"Figures saved to: {IMAGES}")
print(f"Outputs saved to: {OUTPUTS}")
