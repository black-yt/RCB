"""
Personalized Neoantigen Vaccine Optimization Analysis
Analyzes vaccine composition, immune response probabilities, coverage, and runtime.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_001_20260401_200652"
DATA = os.path.join(WORKSPACE, "data")
OUTPUTS = os.path.join(WORKSPACE, "outputs")
IMG = os.path.join(WORKSPACE, "report", "images")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10")

# ==============================================================================
# 1. Load data
# ==============================================================================
print("Loading data...")

cell_pops = pd.read_csv(os.path.join(DATA, "cell-populations.csv"))
final_resp = pd.read_csv(os.path.join(DATA, "final-response-likelihoods.csv"))
runtime_df = pd.read_csv(os.path.join(DATA, "optimization_runtime_data.csv"))
sel_vax_elems = pd.read_csv(os.path.join(DATA, "selected-vaccine-elements.budget-10.minsum.adaptive.csv"))
sim_resp = pd.read_csv(os.path.join(DATA, "sim-specific-response-likelihoods.csv"))
vax_budget = pd.read_csv(os.path.join(DATA, "vaccine.budget-10.minsum.adaptive.csv"))

# Load all 10 replicate vaccine element score files
rep_dfs = []
for rep in range(10):
    fname = f"vaccine-elements.scores.100-cells.10x.rep-{rep}.csv"
    df = pd.read_csv(os.path.join(DATA, fname))
    df["rep"] = rep
    rep_dfs.append(df)
vax_scores_all = pd.concat(rep_dfs, ignore_index=True)

print(f"  cell_pops: {cell_pops.shape}")
print(f"  final_resp: {final_resp.shape}")
print(f"  runtime_df: {runtime_df.shape}")
print(f"  sel_vax_elems: {sel_vax_elems.shape}")
print(f"  sim_resp: {sim_resp.shape}")
print(f"  vax_budget: {vax_budget.shape}")
print(f"  vax_scores_all: {vax_scores_all.shape}")

# ==============================================================================
# 2. Descriptive statistics
# ==============================================================================
print("\nComputing descriptive statistics...")

# 2a. Mutation frequency across cells per repetition
mut_cell_counts = (
    cell_pops.groupby(["repetition", "mutation"])["cell_ids"]
    .nunique()
    .reset_index(name="n_cells")
)

# 2b. Number of presented peptides per cell
pep_per_cell = (
    cell_pops.groupby(["repetition", "cell_ids"])["presented_peptides"]
    .nunique()
    .reset_index(name="n_peptides")
)

# 2c. Vaccine element selection frequency across replicates
vax_elem_freq = (
    sel_vax_elems.groupby("peptide")["repetition"].count().reset_index(name="n_selected")
)
vax_elem_freq = vax_elem_freq.sort_values("n_selected", ascending=False)

# 2d. Per-cell response probability stats
resp_stats = final_resp["p_response"].describe()
print("\nResponse probability stats:")
print(resp_stats)

# Save summaries
mut_cell_counts.to_csv(os.path.join(OUTPUTS, "mutation_cell_counts.csv"), index=False)
vax_elem_freq.to_csv(os.path.join(OUTPUTS, "vaccine_element_frequency.csv"), index=False)
resp_stats.to_csv(os.path.join(OUTPUTS, "response_stats.csv"))

# ==============================================================================
# FIG 1 – Tumor Cell Mutation Landscape
# ==============================================================================
print("Generating Figure 1: Mutation landscape...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1a – Mutation occurrence: how many unique cells carry each mutation (avg over reps)
avg_mut = mut_cell_counts.groupby("mutation")["n_cells"].mean().sort_values(ascending=False)
ax = axes[0]
bars = ax.bar(range(len(avg_mut)), avg_mut.values, color=PALETTE[0], alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(avg_mut)))
ax.set_xticklabels(avg_mut.index, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("Mutation")
ax.set_ylabel("Mean number of cells (across 10 reps)")
ax.set_title("(A) Mutation Prevalence in Tumor Population")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1b – Distribution of peptides presented per cell
ax = axes[1]
pep_per_cell["n_peptides"].hist(ax=ax, bins=range(1, pep_per_cell["n_peptides"].max()+2),
                                 color=PALETTE[1], edgecolor="white", alpha=0.85)
ax.set_xlabel("Number of distinct peptides presented per cell")
ax.set_ylabel("Number of cells")
ax.set_title("(B) Presented Peptide Diversity per Cell")
mean_pep = pep_per_cell["n_peptides"].mean()
ax.axvline(mean_pep, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_pep:.1f}")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig1_mutation_landscape.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_mutation_landscape.png")

# ==============================================================================
# FIG 2 – Vaccine Element Response Scores
# ==============================================================================
print("Generating Figure 2: Vaccine element response scores...")

# Average p_response per vaccine element across all cells and replicates
elem_resp = (
    vax_scores_all.groupby("vaccine_element")["p_response"]
    .agg(["mean", "std", "median"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
elem_resp.to_csv(os.path.join(OUTPUTS, "element_response_summary.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2a – Box-plot of p_response per vaccine element
order = elem_resp.sort_values("mean", ascending=False)["vaccine_element"].tolist()
ax = axes[0]
sns.boxplot(data=vax_scores_all, x="vaccine_element", y="p_response",
            order=order, ax=ax, palette="tab10",
            flierprops=dict(marker=".", alpha=0.3, markersize=3))
ax.set_xlabel("Vaccine element (mutation)")
ax.set_ylabel("Per-cell response probability")
ax.set_title("(A) Response Probability Distribution per Vaccine Element")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 2b – Heatmap: mean p_response per (cell, vaccine_element) aggregated over reps
pivot = vax_scores_all.groupby(["cell_id", "vaccine_element"])["p_response"].mean().unstack()
# select a sample of 30 cells for readability
sample_cells = sorted(pivot.index)[:30]
pivot_sample = pivot.loc[sample_cells, order]
ax = axes[1]
sns.heatmap(pivot_sample, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "p_response"})
ax.set_xlabel("Vaccine element")
ax.set_ylabel("Cell ID (sample of 30)")
ax.set_title("(B) Per-Cell Response Heatmap (Mean across Replicates)")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig2_vaccine_element_scores.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_vaccine_element_scores.png")

# ==============================================================================
# FIG 3 – Per-Cell Immune Response Probability Distribution
# ==============================================================================
print("Generating Figure 3: Response probability distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 3a – Histogram + KDE of p_response across all cells and reps
ax = axes[0]
ax.hist(final_resp["p_response"], bins=40, density=True, color=PALETTE[0],
        alpha=0.6, edgecolor="white", label="Histogram")
kde = stats.gaussian_kde(final_resp["p_response"])
x_range = np.linspace(0, 1, 300)
ax.plot(x_range, kde(x_range), color=PALETTE[0], linewidth=2, label="KDE")
ax.axvline(final_resp["p_response"].mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Mean={final_resp['p_response'].mean():.3f}")
ax.axvline(final_resp["p_response"].median(), color="orange", linestyle=":",
           linewidth=1.5, label=f"Median={final_resp['p_response'].median():.3f}")
ax.set_xlabel("Per-cell immune response probability")
ax.set_ylabel("Density")
ax.set_title("(A) Distribution of Per-Cell Response Probabilities\n(MinSum budget-10 vaccine)")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 3b – Box-plots per repetition (population)
final_resp["rep"] = final_resp["population"].str.extract(r",\s*(\d+)").astype(int)
ax = axes[1]
rep_data = [final_resp[final_resp["rep"] == r]["p_response"].values for r in range(10)]
bp = ax.boxplot(rep_data, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", alpha=0.4, markersize=3))
for patch, color in zip(bp["boxes"], sns.color_palette("tab10", 10)):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xlabel("Simulation repetition")
ax.set_ylabel("Per-cell immune response probability")
ax.set_title("(B) Response Probability by Repetition")
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"Rep {i}" for i in range(10)], rotation=30, ha="right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig3_response_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_response_distribution.png")

# ==============================================================================
# FIG 4 – Coverage Curve: Fraction of Cells Responding vs. Threshold
# ==============================================================================
print("Generating Figure 4: Coverage curves...")

thresholds = np.linspace(0, 1, 200)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 4a – Aggregate coverage curve
def coverage_at_threshold(p_values, thresholds):
    return [np.mean(p_values >= t) for t in thresholds]

coverage_all = coverage_at_threshold(final_resp["p_response"].values, thresholds)
ax = axes[0]
ax.plot(thresholds, coverage_all, color=PALETTE[0], linewidth=2)
ax.fill_between(thresholds, coverage_all, alpha=0.15, color=PALETTE[0])

# Mark key thresholds
for th, ls in [(0.5, "--"), (0.8, ":")]:
    cov = np.mean(final_resp["p_response"] >= th)
    ax.axvline(th, color="gray", linestyle=ls, linewidth=1)
    ax.axhline(cov, color="gray", linestyle=ls, linewidth=1)
    ax.annotate(f"  {cov:.2f} cells @ ≥{th}", xy=(th, cov), fontsize=8, color="gray")

ax.set_xlabel("Immune response threshold")
ax.set_ylabel("Fraction of tumor cells covered")
ax.set_title("(A) Vaccine Coverage Curve\n(MinSum budget-10, all reps pooled)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 4b – Coverage curves per replicate
ax = axes[1]
for rep in range(10):
    rep_vals = final_resp[final_resp["rep"] == rep]["p_response"].values
    cov_rep = coverage_at_threshold(rep_vals, thresholds)
    ax.plot(thresholds, cov_rep, linewidth=1.2, alpha=0.7, label=f"Rep {rep}",
            color=sns.color_palette("tab10", 10)[rep])

# Mean across reps
mean_cov = []
for t in thresholds:
    per_rep = [np.mean(final_resp[final_resp["rep"] == r]["p_response"] >= t) for r in range(10)]
    mean_cov.append(np.mean(per_rep))
ax.plot(thresholds, mean_cov, color="black", linewidth=2.5, linestyle="-", label="Mean")

ax.set_xlabel("Immune response threshold")
ax.set_ylabel("Fraction of tumor cells covered")
ax.set_title("(B) Coverage Curves per Replicate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig4_coverage_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_coverage_curves.png")

# ==============================================================================
# FIG 5 – Vaccine Composition Consistency: IoU across Replicates
# ==============================================================================
print("Generating Figure 5: IoU analysis...")

# Get selected vaccine elements per repetition
rep_sets = {}
for rep in range(10):
    rep_elems = sel_vax_elems[sel_vax_elems["repetition"] == rep]["peptide"].tolist()
    rep_sets[rep] = set(rep_elems)

# Compute pairwise IoU
n_reps = 10
iou_matrix = np.zeros((n_reps, n_reps))
for i in range(n_reps):
    for j in range(n_reps):
        inter = len(rep_sets[i] & rep_sets[j])
        union = len(rep_sets[i] | rep_sets[j])
        iou_matrix[i, j] = inter / union if union > 0 else 1.0

# Frequency of each mutation being selected
all_mutations = sorted(set.union(*rep_sets.values()))
freq_dict = {m: sum(1 for r in range(n_reps) if m in rep_sets[r]) for m in all_mutations}
freq_df = pd.DataFrame(list(freq_dict.items()), columns=["mutation", "n_selected"])
freq_df = freq_df.sort_values("n_selected", ascending=False)

# Save IoU matrix
iou_df = pd.DataFrame(iou_matrix, index=[f"Rep{i}" for i in range(n_reps)],
                       columns=[f"Rep{i}" for i in range(n_reps)])
iou_df.to_csv(os.path.join(OUTPUTS, "iou_matrix.csv"))
freq_df.to_csv(os.path.join(OUTPUTS, "element_selection_frequency.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 5a – Heatmap of pairwise IoU
ax = axes[0]
mask = np.eye(n_reps, dtype=bool)
off_diag_iou = iou_matrix[~mask].flatten()
mean_iou = off_diag_iou.mean()
sns.heatmap(iou_matrix, ax=ax, cmap="Blues", vmin=0, vmax=1, annot=True,
            fmt=".2f", square=True, linewidths=0.5,
            xticklabels=[f"R{i}" for i in range(n_reps)],
            yticklabels=[f"R{i}" for i in range(n_reps)],
            cbar_kws={"label": "IoU"})
ax.set_title(f"(A) Pairwise IoU of Vaccine Compositions\n(Mean off-diagonal IoU = {mean_iou:.3f})")

# 5b – Bar chart: selection frequency per mutation
ax = axes[1]
colors = [PALETTE[2] if v == 10 else PALETTE[1] if v >= 7 else PALETTE[0]
          for v in freq_df["n_selected"]]
ax.barh(range(len(freq_df)), freq_df["n_selected"], color=colors, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(freq_df)))
ax.set_yticklabels(freq_df["mutation"], fontsize=9)
ax.set_xlabel("Number of replicates selected in (out of 10)")
ax.set_title("(B) Vaccine Element Selection Frequency\nacross 10 Replicates")
ax.axvline(10, color="gray", linestyle="--", linewidth=1)
ax.axvline(7, color="gray", linestyle=":", linewidth=1)

legend_handles = [
    mpatches.Patch(color=PALETTE[2], alpha=0.85, label="All 10 reps"),
    mpatches.Patch(color=PALETTE[1], alpha=0.85, label="7–9 reps"),
    mpatches.Patch(color=PALETTE[0], alpha=0.85, label="<7 reps"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig5_iou_vaccine_composition.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig5_iou_vaccine_composition.png")

# ==============================================================================
# FIG 6 – Optimization Runtime vs. Population Size
# ==============================================================================
print("Generating Figure 6: Runtime scaling...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

samples = runtime_df["SampleID"].unique()
pop_sizes = sorted(runtime_df["PopulationSize"].unique())

# 6a – Runtime curves per patient sample
ax = axes[0]
for i, sample in enumerate(samples):
    sub = runtime_df[runtime_df["SampleID"] == sample].sort_values("PopulationSize")
    ax.plot(sub["PopulationSize"], sub["RunTime"], marker="o", linewidth=2,
            markersize=5, label=f"Patient {sample}",
            color=sns.color_palette("tab10", len(samples))[i])

ax.set_xlabel("Cancer cell population size")
ax.set_ylabel("Optimization runtime (seconds)")
ax.set_title("(A) Runtime vs. Population Size\n(per patient sample)")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 6b – Log-log plot with linear fit
ax = axes[1]
log_pop = np.log10(runtime_df["PopulationSize"])
log_rt  = np.log10(runtime_df["RunTime"])
slope, intercept, r, p, se = stats.linregress(log_pop, log_rt)

x_fit = np.linspace(log_pop.min(), log_pop.max(), 100)
y_fit = slope * x_fit + intercept

for i, sample in enumerate(samples):
    sub = runtime_df[runtime_df["SampleID"] == sample].sort_values("PopulationSize")
    ax.scatter(np.log10(sub["PopulationSize"]), np.log10(sub["RunTime"]),
               color=sns.color_palette("tab10", len(samples))[i],
               s=50, zorder=5, label=f"Patient {sample}")

ax.plot(x_fit, y_fit, "k--", linewidth=1.5,
        label=f"Fit: slope={slope:.2f}, R²={r**2:.3f}")
ax.set_xlabel("log₁₀(Population size)")
ax.set_ylabel("log₁₀(Runtime, seconds)")
ax.set_title("(B) Log-Log Runtime Scaling\n(Empirical complexity analysis)")
ax.legend(fontsize=8, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig6_runtime_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig6_runtime_scaling.png")

# ==============================================================================
# FIG 7 – Vaccine Budget Composition (selected elements overview)
# ==============================================================================
print("Generating Figure 7: Vaccine budget composition...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 7a – Pie chart of final vaccine composition (vax_budget)
ax = axes[0]
vax_budget_sorted = vax_budget.sort_values("counts", ascending=False)
wedges, texts, autotexts = ax.pie(
    vax_budget_sorted["counts"],
    labels=vax_budget_sorted["peptide"],
    autopct="%1.0f%%",
    startangle=90,
    colors=sns.color_palette("tab10", len(vax_budget_sorted)),
    pctdistance=0.8,
)
for t in autotexts:
    t.set_fontsize(8)
ax.set_title("(A) Vaccine Composition\n(Budget=10, MinSum Adaptive)")

# 7b – Gantt-style: which mutations are selected in which replicates
sel_pivot = pd.DataFrame(False, index=range(10), columns=sorted(sel_vax_elems["peptide"].unique()))
for _, row in sel_vax_elems.iterrows():
    sel_pivot.loc[row["repetition"], row["peptide"]] = True

ax = axes[1]
sns.heatmap(sel_pivot.astype(int).T, ax=ax, cmap="Blues", vmin=0, vmax=1,
            linewidths=0.3, linecolor="lightgrey",
            xticklabels=[f"R{i}" for i in range(10)],
            yticklabels=sel_pivot.columns,
            cbar=False, annot=False)
ax.set_xlabel("Simulation replicate")
ax.set_ylabel("Mutation (vaccine element)")
ax.set_title("(B) Vaccine Element Selection Matrix\n(Blue = selected)")

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig7_vaccine_composition.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig7_vaccine_composition.png")

# ==============================================================================
# FIG 8 – Sim-Specific vs. Pooled Response Probability Comparison
# ==============================================================================
print("Generating Figure 8: Sim-specific vs pooled comparison...")

# Extract repetition number from vaccine name in sim_resp
sim_resp["rep"] = sim_resp["vaccine"].str.extract(r"rep-(\d+)").astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 8a – Correlation scatter: sim-specific vs final pooled response
merged = final_resp.merge(
    sim_resp, on=["name", "population"], suffixes=("_pooled", "_sim")
)
ax = axes[0]
ax.scatter(merged["p_response_pooled"], merged["p_response_sim"],
           alpha=0.3, s=10, color=PALETTE[0])
lims = [0, 1]
ax.plot(lims, lims, "r--", linewidth=1.5, label="y=x")
r_val, _ = stats.pearsonr(merged["p_response_pooled"], merged["p_response_sim"])
ax.set_xlabel("Pooled (final) response probability")
ax.set_ylabel("Sim-specific response probability")
ax.set_title(f"(A) Pooled vs. Sim-Specific Response Probabilities\n(Pearson r = {r_val:.4f})")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 8b – Distribution comparison: violin plots by rep
ax = axes[1]
plot_data = pd.concat([
    final_resp[["rep", "p_response"]].rename(columns={"p_response": "p"}
                                             ).assign(source="Pooled"),
    sim_resp[["rep", "p_response"]].rename(columns={"p_response": "p"}
                                           ).assign(source="Sim-specific"),
])
sns.violinplot(data=plot_data, x="rep", y="p", hue="source",
               split=True, inner="quart", ax=ax,
               palette={"Pooled": PALETTE[0], "Sim-specific": PALETTE[1]})
ax.set_xlabel("Simulation replicate")
ax.set_ylabel("Immune response probability")
ax.set_title("(B) Response Probability: Pooled vs. Sim-Specific\n(per replicate)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(IMG, "fig8_pooled_vs_sim_specific.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig8_pooled_vs_sim_specific.png")

# ==============================================================================
# Compute summary metrics for report
# ==============================================================================
print("\nComputing summary metrics...")

# Coverage at 0.5 threshold
cov_50 = np.mean(final_resp["p_response"] >= 0.5)
cov_80 = np.mean(final_resp["p_response"] >= 0.8)
cov_90 = np.mean(final_resp["p_response"] >= 0.9)

# Mean IoU
mean_iou_val = off_diag_iou.mean()
std_iou_val = off_diag_iou.std()

# Runtime scaling slope
print(f"\nRuntime log-log slope: {slope:.3f} (R²={r**2:.3f})")
print(f"Coverage at p≥0.5: {cov_50:.3f}")
print(f"Coverage at p≥0.8: {cov_80:.3f}")
print(f"Coverage at p≥0.9: {cov_90:.3f}")
print(f"Mean pairwise IoU: {mean_iou_val:.3f} ± {std_iou_val:.3f}")
print(f"Mean p_response: {final_resp['p_response'].mean():.4f}")
print(f"Median p_response: {final_resp['p_response'].median():.4f}")

# Core vaccines selected in all 10 reps
core = [m for m, c in freq_dict.items() if c == 10]
print(f"Core mutations (selected in all 10 reps): {sorted(core)}")
print(f"Budget compositions in vaccine.budget-10: {sorted(vax_budget['peptide'].tolist())}")

# Save summary metrics
summary = {
    "mean_p_response": final_resp["p_response"].mean(),
    "median_p_response": final_resp["p_response"].median(),
    "std_p_response": final_resp["p_response"].std(),
    "coverage_50": cov_50,
    "coverage_80": cov_80,
    "coverage_90": cov_90,
    "mean_iou": mean_iou_val,
    "std_iou": std_iou_val,
    "runtime_slope": slope,
    "runtime_r2": r**2,
    "n_core_mutations": len(core),
}
pd.Series(summary).to_csv(os.path.join(OUTPUTS, "summary_metrics.csv"), header=False)

print("\nAll figures saved. Analysis complete.")
