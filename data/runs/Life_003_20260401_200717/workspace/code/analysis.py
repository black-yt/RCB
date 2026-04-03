"""
Uncalled4 Research Analysis
Analyzes pore models, performance benchmarks, and m6A modification detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

# Paths
DATA_DIR = "../data"
OUT_DIR = "../outputs"
IMG_DIR = "../report/images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
COLORS = {"Uncalled4": "#2196F3", "f5c": "#FF9800", "Nanopolish": "#4CAF50", "Tombo": "#9C27B0"}

# ─────────────────────────────────────────────
# 1. Load pore models
# ─────────────────────────────────────────────
models = {
    "DNA r9.4.1\n(6-mer)":  pd.read_csv(os.path.join(DATA_DIR, "dna_r9.4.1_400bps_6mer_uncalled4.csv")),
    "DNA r10.4.1\n(9-mer)": pd.read_csv(os.path.join(DATA_DIR, "dna_r10.4.1_400bps_9mer_uncalled4.csv")),
    "RNA001\n(5-mer)":      pd.read_csv(os.path.join(DATA_DIR, "rna_r9.4.1_70bps_5mer_uncalled4.csv")),
    "RNA004\n(9-mer)":      pd.read_csv(os.path.join(DATA_DIR, "rna004_130bps_9mer_uncalled4.csv")),
}

# Save summary stats
summary_rows = []
for name, df in models.items():
    klen = len(df["kmer"].iloc[0])
    row = {
        "Model": name.replace("\n", " "),
        "k-mer length": klen,
        "N k-mers": len(df),
        "Mean current (mean)": df["current_mean"].mean(),
        "Mean current (std)": df["current_mean"].std(),
        "Std current (mean)": df["current_std"].mean(),
        "Dwell time (mean)": df["dwell_time"].mean(),
    }
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "pore_model_summary.csv"), index=False)
print("Pore model summary saved.")

# ─────────────────────────────────────────────
# Figure 1: Pore model current distributions
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
palette = sns.color_palette("husl", 4)

for i, (name, df) in enumerate(models.items()):
    ax = axes[i]
    ax.hist(df["current_mean"], bins=80, color=palette[i], alpha=0.8, edgecolor="none")
    ax.set_title(f"{name}", fontsize=12)
    ax.set_xlabel("Current Mean (pA, normalized)")
    ax.set_ylabel("Number of k-mers")
    ax.axvline(df["current_mean"].mean(), color="black", linestyle="--", lw=1.5, label=f"Mean={df['current_mean'].mean():.2f}")
    ax.legend(fontsize=9)

fig.suptitle("Distribution of Mean Ionic Current Across Pore Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig1_current_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 1 saved.")

# ─────────────────────────────────────────────
# Figure 2: Current mean vs std for each model
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for i, (name, df) in enumerate(models.items()):
    ax = axes[i]
    sample = df.sample(min(5000, len(df)), random_state=42)
    sc = ax.scatter(sample["current_mean"], sample["current_std"],
                    alpha=0.3, s=4, c=palette[i])
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Current Mean")
    ax.set_ylabel("Current Std")
    # Trend line
    z = np.polyfit(sample["current_mean"], sample["current_std"], 1)
    p = np.poly1d(z)
    xs = np.linspace(sample["current_mean"].min(), sample["current_mean"].max(), 100)
    ax.plot(xs, p(xs), "k--", lw=1.2, label=f"r={sample['current_mean'].corr(sample['current_std']):.2f}")
    ax.legend(fontsize=9)

fig.suptitle("Current Mean vs. Standard Deviation per Pore Model", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig2_mean_vs_std.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 2 saved.")

# ─────────────────────────────────────────────
# Figure 3: Base-position effects on current
# ─────────────────────────────────────────────
def base_position_effect(df, title):
    """Compute mean current for each base at each k-mer position."""
    klen = len(df["kmer"].iloc[0])
    bases = list("ACGT")
    records = []
    for pos in range(klen):
        for base in bases:
            mask = df["kmer"].str[pos] == base
            if mask.sum() > 0:
                records.append({
                    "Position": pos + 1,
                    "Base": base,
                    "Mean Current": df.loc[mask, "current_mean"].mean(),
                })
    return pd.DataFrame(records)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
base_colors = {"A": "#E74C3C", "C": "#3498DB", "G": "#2ECC71", "T": "#F39C12"}

for i, (name, df) in enumerate(models.items()):
    ax = axes[i]
    bpe = base_position_effect(df, name)
    for base in "ACGT":
        sub = bpe[bpe["Base"] == base]
        ax.plot(sub["Position"], sub["Mean Current"], marker="o", ms=5,
                color=base_colors[base], label=base, lw=2)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Position in k-mer")
    ax.set_ylabel("Mean Current")
    ax.legend(title="Base", fontsize=9)
    ax.set_xticks(range(1, len(df["kmer"].iloc[0]) + 1))

fig.suptitle("Base Identity Effect on Ionic Current by k-mer Position", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig3_base_position_effects.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 3 saved.")

# ─────────────────────────────────────────────
# Figure 4: Dwell time distributions
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (name, df) in enumerate(models.items()):
    ax = axes[i]
    dwell = df["dwell_time"]
    ax.hist(dwell, bins=50, color=palette[i], alpha=0.8, edgecolor="none")
    ax.axvline(dwell.mean(), color="black", linestyle="--", lw=1.5,
               label=f"Mean={dwell.mean():.1f}")
    ax.axvline(dwell.median(), color="gray", linestyle=":", lw=1.5,
               label=f"Median={dwell.median():.1f}")
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Dwell Time (samples)")
    ax.set_ylabel("Number of k-mers")
    ax.legend(fontsize=9)

fig.suptitle("Dwell Time Distributions Across Pore Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig4_dwell_time_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 4 saved.")

# ─────────────────────────────────────────────
# 2. Performance benchmarks
# ─────────────────────────────────────────────
perf = pd.read_csv(os.path.join(DATA_DIR, "performance_summary.csv"))
perf.to_csv(os.path.join(OUT_DIR, "performance_summary_clean.csv"), index=False)

# Compute speedup ratios relative to Uncalled4
speedup_rows = []
for chem in perf["Chemistry"].unique():
    sub = perf[perf["Chemistry"] == chem].copy()
    u4_time = sub.loc[sub["Tool"] == "Uncalled4", "Time_min"].values
    if len(u4_time) == 0 or np.isnan(u4_time[0]):
        continue
    u4_time = u4_time[0]
    for _, row in sub.iterrows():
        if row["Tool"] != "Uncalled4" and not np.isnan(row["Time_min"]):
            speedup_rows.append({
                "Chemistry": chem,
                "Tool": row["Tool"],
                "Speedup vs Uncalled4": row["Time_min"] / u4_time,
            })
speedup_df = pd.DataFrame(speedup_rows)
speedup_df.to_csv(os.path.join(OUT_DIR, "speedup_ratios.csv"), index=False)
print("Speedup ratios saved.")
print(speedup_df.to_string(index=False))

# ─────────────────────────────────────────────
# Figure 5: Performance comparison (time)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

chemistries = perf["Chemistry"].unique()
tools = ["Uncalled4", "f5c", "Nanopolish", "Tombo"]
x = np.arange(len(chemistries))
width = 0.2

ax = axes[0]
for j, tool in enumerate(tools):
    vals = []
    for chem in chemistries:
        row = perf[(perf["Chemistry"] == chem) & (perf["Tool"] == tool)]
        vals.append(row["Time_min"].values[0] if len(row) > 0 and not pd.isna(row["Time_min"].values[0]) else 0)
    bars = ax.bar(x + j * width, vals, width, label=tool, color=COLORS[tool], alpha=0.85)

ax.set_xlabel("Sequencing Chemistry")
ax.set_ylabel("Alignment Time (minutes)")
ax.set_title("Alignment Time Comparison\n(lower is better)", fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chemistries, fontsize=10)
ax.legend()
ax.set_yscale("log")
ax.yaxis.grid(True, alpha=0.4)

# File size
ax = axes[1]
for j, tool in enumerate(tools):
    vals = []
    for chem in chemistries:
        row = perf[(perf["Chemistry"] == chem) & (perf["Tool"] == tool)]
        vals.append(row["FileSize_MB"].values[0] if len(row) > 0 and not pd.isna(row["FileSize_MB"].values[0]) else 0)
    ax.bar(x + j * width, vals, width, label=tool, color=COLORS[tool], alpha=0.85)

ax.set_xlabel("Sequencing Chemistry")
ax.set_ylabel("Output File Size (MB)")
ax.set_title("Output File Size Comparison\n(lower is better)", fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chemistries, fontsize=10)
ax.legend()
ax.set_yscale("log")
ax.yaxis.grid(True, alpha=0.4)

fig.suptitle("Uncalled4 Performance Benchmarks vs. Competing Tools", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig5_performance_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 5 saved.")

# ─────────────────────────────────────────────
# Figure 6: Speedup heatmap
# ─────────────────────────────────────────────
pivot_time = perf.pivot(index="Chemistry", columns="Tool", values="Time_min")
pivot_size = perf.pivot(index="Chemistry", columns="Tool", values="FileSize_MB")

# Speedup: other / Uncalled4
speedup_time = pivot_time.div(pivot_time["Uncalled4"], axis=0).drop(columns=["Uncalled4"])
speedup_size = pivot_size.div(pivot_size["Uncalled4"], axis=0).drop(columns=["Uncalled4"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, data, title, fmt in zip(
    axes,
    [speedup_time, speedup_size],
    ["Time Speedup\n(X times slower than Uncalled4)", "File Size Ratio\n(X times larger than Uncalled4)"],
    [".1f", ".1f"]
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=1,
                cbar_kws={"label": "Fold increase vs Uncalled4"},
                mask=data.isna())
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Tool")
    ax.set_ylabel("Chemistry")

fig.suptitle("Performance Fold-Increase Relative to Uncalled4", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig6_speedup_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 6 saved.")

# ─────────────────────────────────────────────
# 3. m6A modification detection
# ─────────────────────────────────────────────
labels = pd.read_csv(os.path.join(DATA_DIR, "m6a_labels.csv"))
pred_u4 = pd.read_csv(os.path.join(DATA_DIR, "m6a_predictions_uncalled4.csv"))
pred_np = pd.read_csv(os.path.join(DATA_DIR, "m6a_predictions_nanopolish.csv"))

df_u4 = labels.merge(pred_u4, on="site_id")
df_np = labels.merge(pred_np, on="site_id")

y_true_u4 = df_u4["label"].values
y_score_u4 = df_u4["probability"].values
y_true_np = df_np["label"].values
y_score_np = df_np["probability"].values

# PR curves
prec_u4, rec_u4, thr_pr_u4 = precision_recall_curve(y_true_u4, y_score_u4)
prec_np, rec_np, thr_pr_np = precision_recall_curve(y_true_np, y_score_np)
ap_u4 = average_precision_score(y_true_u4, y_score_u4)
ap_np = average_precision_score(y_true_np, y_score_np)

# ROC curves
fpr_u4, tpr_u4, _ = roc_curve(y_true_u4, y_score_u4)
fpr_np, tpr_np, _ = roc_curve(y_true_np, y_score_np)
auc_u4 = auc(fpr_u4, tpr_u4)
auc_np = auc(fpr_np, tpr_np)

# Save metrics
m6a_metrics = pd.DataFrame({
    "Tool": ["Uncalled4", "Nanopolish"],
    "Average_Precision": [ap_u4, ap_np],
    "AUC_ROC": [auc_u4, auc_np],
    "N_positive": [y_true_u4.sum(), y_true_np.sum()],
    "N_total": [len(y_true_u4), len(y_true_np)],
})
m6a_metrics.to_csv(os.path.join(OUT_DIR, "m6a_metrics.csv"), index=False)
print(m6a_metrics.to_string(index=False))

# ─────────────────────────────────────────────
# Figure 7: PR and ROC curves
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# PR curve
ax = axes[0]
ax.plot(rec_u4, prec_u4, color=COLORS["Uncalled4"], lw=2.5,
        label=f"Uncalled4 (AP={ap_u4:.3f})")
ax.plot(rec_np, prec_np, color=COLORS["Nanopolish"], lw=2.5, linestyle="--",
        label=f"Nanopolish (AP={ap_np:.3f})")
baseline = y_true_u4.mean()
ax.axhline(baseline, color="gray", linestyle=":", lw=1.5, label=f"Baseline (prevalence={baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve\nfor m6A Detection", fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

# ROC curve
ax = axes[1]
ax.plot(fpr_u4, tpr_u4, color=COLORS["Uncalled4"], lw=2.5,
        label=f"Uncalled4 (AUC={auc_u4:.3f})")
ax.plot(fpr_np, tpr_np, color=COLORS["Nanopolish"], lw=2.5, linestyle="--",
        label=f"Nanopolish (AUC={auc_np:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve\nfor m6A Detection", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

fig.suptitle("m6A Modification Detection: Uncalled4 vs. Nanopolish Alignments", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig7_m6a_pr_roc.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 7 saved.")

# ─────────────────────────────────────────────
# Figure 8: Score distributions by label
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, df, tool, color in zip(
    axes,
    [df_u4, df_np],
    ["Uncalled4", "Nanopolish"],
    [COLORS["Uncalled4"], COLORS["Nanopolish"]]
):
    for lbl, ls, label in [(0, "--", "Unmodified (label=0)"), (1, "-", "Modified m6A (label=1)")]:
        sub = df[df["label"] == lbl]["probability"]
        ax.hist(sub, bins=50, alpha=0.6, density=True, linestyle=ls,
                label=f"{label} (n={len(sub)})")
    ax.set_xlabel("m6Anet Prediction Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{tool} — Score Distribution by Label", fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle("m6A Prediction Score Distributions: Modified vs. Unmodified Sites", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig8_m6a_score_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 8 saved.")

# ─────────────────────────────────────────────
# Figure 9: Pore model comparison heatmap (DNA chemistries)
# ─────────────────────────────────────────────
# Compare shared k-mer prefixes between r9.4 and r10.4 DNA models
df_r9 = models["DNA r9.4.1\n(6-mer)"].copy()
df_r10 = models["DNA r10.4.1\n(9-mer)"].copy()

# Use first 6 bases of r10 9-mers to match r9 6-mers
df_r10["kmer_6"] = df_r10["kmer"].str[:6]
merged = df_r9.merge(df_r10[["kmer_6", "current_mean", "current_std"]].rename(
    columns={"kmer_6": "kmer", "current_mean": "r10_mean", "current_std": "r10_std"}
), on="kmer", how="inner").rename(columns={"current_mean": "r9_mean", "current_std": "r9_std"})

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
sample_m = merged.sample(min(2000, len(merged)), random_state=42)
sc = ax.scatter(sample_m["r9_mean"], sample_m["r10_mean"], alpha=0.3, s=5, c="#1A237E")
corr = merged["r9_mean"].corr(merged["r10_mean"])
ax.set_xlabel("DNA r9.4.1 Current Mean")
ax.set_ylabel("DNA r10.4.1 Current Mean")
ax.set_title(f"Current Mean Correlation\nr9.4.1 vs r10.4.1 (r={corr:.3f})", fontweight="bold")
lims = [min(sample_m["r9_mean"].min(), sample_m["r10_mean"].min()),
        max(sample_m["r9_mean"].max(), sample_m["r10_mean"].max())]
ax.plot(lims, lims, "k--", lw=1.5, label="y=x")
ax.legend()

ax = axes[1]
diff = merged["r10_mean"] - merged["r9_mean"]
ax.hist(diff, bins=80, color="#1A237E", alpha=0.8, edgecolor="none")
ax.axvline(diff.mean(), color="red", lw=1.5, linestyle="--", label=f"Mean Δ={diff.mean():.3f}")
ax.axvline(0, color="black", lw=1, linestyle=":")
ax.set_xlabel("r10.4.1 − r9.4.1 Current Mean")
ax.set_ylabel("Count")
ax.set_title("Distribution of Current Mean Differences\nBetween r9.4.1 and r10.4.1 (shared 6-mer prefix)", fontweight="bold")
ax.legend()

fig.suptitle("Cross-Chemistry Pore Model Comparison: DNA r9.4.1 vs. r10.4.1", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig9_pore_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 9 saved.")

# ─────────────────────────────────────────────
# Figure 10: Summary overview (table-style)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

table_data = [
    ["Chemistry", "Uncalled4 Time", "Best Competitor Time", "Speedup", "Uncalled4 Size", "Competitor Size Ratio"],
]
for chem in perf["Chemistry"].unique():
    sub = perf[perf["Chemistry"] == chem]
    u4_t = sub.loc[sub["Tool"] == "Uncalled4", "Time_min"].values[0]
    u4_s = sub.loc[sub["Tool"] == "Uncalled4", "FileSize_MB"].values[0]
    competitors = sub[sub["Tool"] != "Uncalled4"].dropna(subset=["Time_min"])
    if len(competitors) == 0:
        continue
    best = competitors.loc[competitors["Time_min"].idxmin()]
    table_data.append([
        chem,
        f"{u4_t:.1f} min",
        f"{best['Time_min']:.1f} min ({best['Tool']})",
        f"{best['Time_min']/u4_t:.1f}×",
        f"{u4_s:.1f} MB",
        f"{competitors['FileSize_MB'].min()/u4_s:.1f}× smaller",
    ])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2196F3")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E3F2FD")

ax.set_title("Uncalled4 Performance Summary Across Sequencing Chemistries",
             fontsize=13, fontweight="bold", pad=20)
plt.savefig(os.path.join(IMG_DIR, "fig10_performance_table.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Figure 10 saved.")

print("\nAll analysis complete!")
