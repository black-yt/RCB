"""
Apply GP calibration to vitrimer MD data to get calibrated Tg predictions.
Uses combined acid+epoxide fingerprints.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/data")
OUT_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/outputs")
IMG_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/report/images")

# ── Load GP model ─────────────────────────────────────────────────────────────
with open(OUT_DIR / "gp_calibration_model.pkl", "rb") as f:
    arts = pickle.load(f)
gp       = arts['gp']
pca      = arts['pca']
scaler_tg = arts['scaler_tg']
scaler_y  = arts['scaler_y']

# ── Load vitrimer data ────────────────────────────────────────────────────────
vitrimer = pd.read_csv(DATA_DIR / "tg_vitrimer_MD.csv")
print(f"Vitrimer systems: {len(vitrimer)}")

# ── Fingerprint function (same as in calibration) ────────────────────────────
def smiles_to_fp(smi, n_bits=512, radius=2):
    smi_clean = smi.replace("*", "[H]") if "*" in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=float)

# ── Build combined fingerprints for acid+epoxide pairs ────────────────────────
print("Computing fingerprints for vitrimer acid+epoxide pairs...")
fps_combined = []
valid_idx = []
for i, row in vitrimer.iterrows():
    fp_acid = smiles_to_fp(row.acid)
    fp_epx  = smiles_to_fp(row.epoxide)
    if fp_acid is None or fp_epx is None:
        continue
    # Combine by averaging (equivalent to symmetric combination)
    fp_combined = 0.5 * (fp_acid + fp_epx)
    fps_combined.append(fp_combined)
    valid_idx.append(i)

fp_matrix = np.array(fps_combined)
vitrimer_valid = vitrimer.iloc[valid_idx].copy().reset_index(drop=True)
print(f"Valid vitrimer pairs: {len(vitrimer_valid)}")

# ── Build feature matrix ──────────────────────────────────────────────────────
X_pca = pca.transform(fp_matrix)  # same PCA from calibration
Tg_md_scaled = scaler_tg.transform(vitrimer_valid.tg.values.reshape(-1, 1))
X = np.hstack([X_pca, Tg_md_scaled])

# ── GP prediction ─────────────────────────────────────────────────────────────
print("Predicting calibrated Tg for vitrimers (this may take ~1 min)...")
# Predict in batches to avoid memory issues
batch_size = 500
n = len(X)
y_pred_scaled = np.zeros(n)
y_std_scaled  = np.zeros(n)

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    mu, std = gp.predict(X[start:end], return_std=True)
    y_pred_scaled[start:end] = mu
    y_std_scaled[start:end]  = std
    if start % 2000 == 0:
        print(f"  Processed {start}/{n}")

# Back to Kelvin
y_pred_K = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_std_K  = y_std_scaled * scaler_y.scale_[0]

# Save results
vitrimer_valid['tg_calibrated'] = y_pred_K
vitrimer_valid['tg_uncertainty'] = y_std_K
vitrimer_valid.to_csv(OUT_DIR / "vitrimer_calibrated_tg.csv", index=False)
print(f"\nCalibrated Tg stats:")
print(f"  Mean: {y_pred_K.mean():.1f} K  (MD mean: {vitrimer_valid.tg.mean():.1f} K)")
print(f"  Std:  {y_pred_K.std():.1f} K   (MD std: {vitrimer_valid.tg.std():.1f} K)")
print(f"  Range: {y_pred_K.min():.1f} – {y_pred_K.max():.1f} K")
print(f"  Mean uncertainty: {y_std_K.mean():.1f} K")

# ── Figure 5: Calibrated Tg results ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) MD vs calibrated Tg
ax = axes[0]
ax.scatter(vitrimer_valid.tg, y_pred_K, alpha=0.15, s=5, c='steelblue', rasterized=True)
mn = min(vitrimer_valid.tg.min(), y_pred_K.min())
mx = max(vitrimer_valid.tg.max(), y_pred_K.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
ax.set_xlabel("MD-simulated Tg (K)", fontsize=12)
ax.set_ylabel("GP-calibrated Tg (K)", fontsize=12)
ax.set_title(f"(a) MD vs. Calibrated Tg\n(N = {len(vitrimer_valid):,})", fontsize=12, fontweight='bold')
bias = (y_pred_K - vitrimer_valid.tg.values).mean()
ax.text(0.05, 0.92, f"Mean shift: {bias:+.1f} K", transform=ax.transAxes, fontsize=10)

# (b) Distribution comparison
ax = axes[1]
ax.hist(vitrimer_valid.tg, bins=50, alpha=0.6, label='MD-simulated', color='tomato', edgecolor='k', lw=0.2)
ax.hist(y_pred_K, bins=50, alpha=0.6, label='GP-calibrated', color='steelblue', edgecolor='k', lw=0.2)
ax.set_xlabel("Tg (K)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("(b) Tg Distribution: MD vs. Calibrated", fontsize=12, fontweight='bold')
ax.legend()

# (c) Uncertainty map
ax = axes[2]
sc = ax.scatter(vitrimer_valid.tg, y_pred_K, c=y_std_K, cmap='plasma',
                alpha=0.2, s=5, rasterized=True)
plt.colorbar(sc, ax=ax, label='GP uncertainty (K)')
ax.set_xlabel("MD-simulated Tg (K)", fontsize=12)
ax.set_ylabel("GP-calibrated Tg (K)", fontsize=12)
ax.set_title("(c) Prediction Uncertainty Map", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(IMG_DIR / "fig5_vitrimer_calibrated.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_vitrimer_calibrated.png")
print("\nCalibration of vitrimer data complete.")
