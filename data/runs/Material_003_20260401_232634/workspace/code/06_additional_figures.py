"""
Generate figures 8 and 9 for the report (fixing Chem.MolWt error).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

OUT_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/outputs")
IMG_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/report/images")

vitrimer     = pd.read_csv(OUT_DIR / "vitrimer_latent.csv")
candidates_df= pd.read_csv(OUT_DIR / "designed_candidates.csv")
tg_cal       = np.load(OUT_DIR / "tg_calibrated.npy")

colors = {'Low Tg (~300 K)': '#2196F3', 'Medium Tg (~380 K)': '#4CAF50', 'High Tg (~470 K)': '#F44336'}

def mol_wt(s):
    m = Chem.MolFromSmiles(s)
    return Descriptors.MolWt(m) if m else np.nan

# ── Figure 8 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
bg_mw = vitrimer.acid.apply(mol_wt) + vitrimer.epoxide.apply(mol_wt)
ax.scatter(bg_mw, tg_cal, c='lightgray', s=5, alpha=0.3, label='All vitrimers', rasterized=True)
for cat, ccolor in colors.items():
    sub = candidates_df[candidates_df.target_category == cat]
    ax.scatter(sub.total_MW, sub.tg_calibrated_K, c=ccolor, s=100, label=cat,
               edgecolors='k', lw=0.8, zorder=5)
ax.set_xlabel("Total MW (acid + epoxide) (g/mol)", fontsize=12)
ax.set_ylabel("Calibrated Tg (K)", fontsize=12)
ax.set_title("(a) MW vs. Tg for Designed Candidates", fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(vitrimer.tg_uncertainty, bins=40, alpha=0.5, color='gray', label='All vitrimers', edgecolor='k', lw=0.2)
for cat, ccolor in colors.items():
    sub = candidates_df[candidates_df.target_category == cat]
    ax.scatter(sub.tg_uncertainty_K, np.ones(len(sub)) * 10,
               c=ccolor, s=150, marker='|', linewidths=3, label=cat, zorder=5)
ax.set_xlabel("GP uncertainty on Tg (K)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("(b) Prediction Uncertainty of Candidates", fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG_DIR / "fig8_candidate_properties.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_candidate_properties.png")

# ── Figure 9: Top candidates showcase ────────────────────────────────────────
target_categories = ["Low Tg (~300 K)", "Medium Tg (~380 K)", "High Tg (~470 K)"]
top_cands = []
for cat in target_categories:
    sub = candidates_df[candidates_df.target_category == cat].sort_values('tg_uncertainty_K')
    top_cands.append(sub.iloc[0])
top_df = pd.DataFrame(top_cands)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for i, (_, row) in enumerate(top_df.iterrows()):
    ax = axes[i][0]
    mol_acid = Chem.MolFromSmiles(row.acid)
    if mol_acid:
        img = Draw.MolToImage(mol_acid, size=(300, 200))
        ax.imshow(img)
    ax.set_title(f"Acid — {target_categories[i]}", fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.text(0.5, -0.05, f"MW = {row.acid_MW:.0f} g/mol, LogP = {row.acid_LogP:.2f}",
            transform=ax.transAxes, ha='center', fontsize=8)

    ax = axes[i][1]
    mol_epx = Chem.MolFromSmiles(row.epoxide)
    if mol_epx:
        img = Draw.MolToImage(mol_epx, size=(300, 200))
        ax.imshow(img)
    ax.set_title(
        f"Epoxide — Tg_cal = {row.tg_calibrated_K:.1f} K\n"
        f"GP: {row.tg_gp_revalidated:.1f} ± {row.tg_gp_uncertainty:.1f} K",
        fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.text(0.5, -0.05, f"MW = {row.epx_MW:.0f} g/mol, LogP = {row.epx_LogP:.2f}",
            transform=ax.transAxes, ha='center', fontsize=8)

plt.suptitle("Top Designed Vitrimer Candidates per Target Tg Category", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(IMG_DIR / "fig9_top_candidates.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_top_candidates.png")

# ── Figure 10: Summary pipeline diagram ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

stages = [
    ("MD Simulation\nDatabase\n(N=8,424 vitrimers)", 0.05, 0.5, '#BBDEFB'),
    ("GP Calibration\nModel\n(MD→Exp Tg)\nMAE=24.6 K, R²=0.88", 0.25, 0.5, '#C8E6C9'),
    ("Molecular VAE\n(Fingerprint\nLatent Space,\ndim=64)", 0.50, 0.5, '#FFF9C4'),
    ("Latent Space\nOptimization\n(Target Tg)", 0.70, 0.5, '#F8BBD0'),
    ("Designed\nCandidates\n(15 novel vitrimers)", 0.90, 0.5, '#E1BEE7'),
]
for (label, x, y, col) in stages:
    bbox = dict(boxstyle='round,pad=0.4', facecolor=col, edgecolor='#333', lw=1.5)
    ax.text(x, y, label, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, fontweight='bold', bbox=bbox)

# Arrows
arrow_props = dict(arrowstyle='->', color='#333', lw=2)
for i in range(len(stages) - 1):
    x1, x2 = stages[i][1] + 0.08, stages[i+1][1] - 0.07
    ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=arrow_props)
ax.set_title("AI-Guided Inverse Design Pipeline for Recyclable Vitrimer Polymers",
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(IMG_DIR / "fig10_pipeline.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_pipeline.png")

print("\nAll additional figures saved.")
