"""
Data exploration for vitrimer Tg inverse design.
Generates overview figures for the report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/data")
OUT_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/outputs")
IMG_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/report/images")
OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
calib = pd.read_csv(DATA_DIR / "tg_calibration.csv")
vitrimer = pd.read_csv(DATA_DIR / "tg_vitrimer_MD.csv")

print("=== Calibration Dataset ===")
print(calib.shape)
print(calib.head())
print(calib.describe())
print(f"\nTg_exp range: {calib.tg_exp.min():.1f} – {calib.tg_exp.max():.1f} K")
print(f"Tg_md range:  {calib.tg_md.min():.1f} – {calib.tg_md.max():.1f} K")

print("\n=== Vitrimer MD Dataset ===")
print(vitrimer.shape)
print(vitrimer.head())
print(vitrimer.describe())
print(f"\nVitrimer Tg_MD range: {vitrimer.tg.min():.1f} – {vitrimer.tg.max():.1f} K")

# ── Figure 1: Calibration data overview ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) Tg exp vs MD scatter
ax = axes[0]
ax.scatter(calib.tg_md, calib.tg_exp, alpha=0.6, edgecolors='k', linewidths=0.3, s=40, c='steelblue')
mn = min(calib.tg_md.min(), calib.tg_exp.min())
mx = max(calib.tg_md.max(), calib.tg_exp.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='y = x')
ax.set_xlabel("MD-simulated Tg (K)", fontsize=12)
ax.set_ylabel("Experimental Tg (K)", fontsize=12)
ax.set_title("(a) MD vs. Experimental Tg", fontsize=13, fontweight='bold')
ax.legend()
from scipy.stats import pearsonr
r, _ = pearsonr(calib.tg_md, calib.tg_exp)
ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10)

# (b) Distribution of Tg (exp vs MD)
ax = axes[1]
ax.hist(calib.tg_exp, bins=25, alpha=0.6, label='Experimental', color='steelblue', edgecolor='k', lw=0.3)
ax.hist(calib.tg_md,  bins=25, alpha=0.6, label='MD-simulated', color='tomato', edgecolor='k', lw=0.3)
ax.set_xlabel("Tg (K)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("(b) Tg Distribution (Calibration)", fontsize=13, fontweight='bold')
ax.legend()

# (c) Vitrimer Tg_MD distribution
ax = axes[2]
ax.hist(vitrimer.tg, bins=40, color='seagreen', alpha=0.75, edgecolor='k', lw=0.3)
ax.set_xlabel("MD-simulated Tg (K)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"(c) Vitrimer Tg Distribution\n(N = {len(vitrimer):,})", fontsize=13, fontweight='bold')
ax.axvline(vitrimer.tg.mean(), color='red', linestyle='--', lw=1.5, label=f"Mean = {vitrimer.tg.mean():.0f} K")
ax.legend()

plt.tight_layout()
plt.savefig(IMG_DIR / "fig1_data_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_data_overview.png")

# ── Figure 2: Residuals & MD bias analysis ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

residual = calib.tg_md - calib.tg_exp
ax = axes[0]
ax.scatter(calib.tg_exp, residual, alpha=0.5, s=35, c='darkorange', edgecolors='k', lw=0.3)
ax.axhline(0, color='k', lw=1)
ax.axhline(residual.mean(), color='red', lw=1.5, linestyle='--', label=f"Mean bias = {residual.mean():.1f} K")
ax.set_xlabel("Experimental Tg (K)", fontsize=12)
ax.set_ylabel("Tg_MD − Tg_exp (K)", fontsize=12)
ax.set_title("(a) MD Bias vs. Experimental Tg", fontsize=13, fontweight='bold')
ax.legend()

ax = axes[1]
ax.hist(residual, bins=25, color='mediumpurple', alpha=0.75, edgecolor='k', lw=0.3)
ax.axvline(residual.mean(), color='red', linestyle='--', lw=1.5, label=f"Mean = {residual.mean():.1f} K")
ax.axvline(0, color='k', lw=1)
ax.set_xlabel("Tg_MD − Tg_exp (K)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"(b) MD Bias Distribution\n(std = {residual.std():.1f} K)", fontsize=13, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(IMG_DIR / "fig2_md_bias.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_md_bias.png")

# ── Compute basic molecular descriptors ──────────────────────────────────────
def get_mw_logp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.nan, np.nan
        return Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
    except:
        return np.nan, np.nan

calib[['MW', 'LogP']] = [get_mw_logp(s) for s in calib.smiles]
vitrimer[['acid_MW', 'acid_LogP']] = [get_mw_logp(s) for s in vitrimer.acid]
vitrimer[['epx_MW', 'epx_LogP']] = [get_mw_logp(s) for s in vitrimer.epoxide]

calib.to_csv(OUT_DIR / "calib_with_desc.csv", index=False)
vitrimer.to_csv(OUT_DIR / "vitrimer_with_desc.csv", index=False)

# ── Figure 3: Molecular property distributions ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(calib.MW, calib.tg_exp, alpha=0.5, s=35, c='teal', edgecolors='k', lw=0.3)
ax.set_xlabel("Molecular Weight (g/mol)", fontsize=12)
ax.set_ylabel("Experimental Tg (K)", fontsize=12)
ax.set_title("(a) MW vs. Experimental Tg", fontsize=13, fontweight='bold')

ax = axes[1]
ax.scatter(calib.LogP, calib.tg_exp, alpha=0.5, s=35, c='crimson', edgecolors='k', lw=0.3)
ax.set_xlabel("LogP", fontsize=12)
ax.set_ylabel("Experimental Tg (K)", fontsize=12)
ax.set_title("(b) LogP vs. Experimental Tg", fontsize=13, fontweight='bold')

ax = axes[2]
combined_mw = (vitrimer.acid_MW + vitrimer.epx_MW).dropna()
ax.hist(combined_mw, bins=35, color='goldenrod', alpha=0.75, edgecolor='k', lw=0.3)
ax.set_xlabel("Total MW (acid + epoxide, g/mol)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("(c) Vitrimer Combined MW", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(IMG_DIR / "fig3_molecular_properties.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_molecular_properties.png")

print("\nData exploration complete.")
print(f"Calibration set: {len(calib)} polymers")
print(f"Vitrimer set: {len(vitrimer)} systems")
print(f"MD bias (mean ± std): {residual.mean():.1f} ± {residual.std():.1f} K")
