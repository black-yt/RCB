"""
Gaussian Process calibration: MD-simulated Tg → experimental Tg.
Uses Morgan fingerprints + Tg_MD as features.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/data")
OUT_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/outputs")
IMG_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/report/images")

np.random.seed(42)

# ── Load calibration data ────────────────────────────────────────────────────
calib = pd.read_csv(DATA_DIR / "tg_calibration.csv")
print(f"Loaded {len(calib)} calibration molecules")

# ── Compute Morgan fingerprints (radius=2, 512 bits) ─────────────────────────
def smiles_to_fp(smi, n_bits=512, radius=2):
    """Return Morgan fingerprint as numpy array, or None on failure."""
    # Handle polymer SMILES with * wildcard atoms
    smi_clean = smi.replace("*", "[H]") if "*" in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

fps = []
valid_idx = []
for i, row in calib.iterrows():
    fp = smiles_to_fp(row.smiles)
    if fp is not None:
        fps.append(fp)
        valid_idx.append(i)
    else:
        print(f"  Warning: could not parse {row.smiles}")

fp_matrix = np.array(fps)
calib_valid = calib.iloc[valid_idx].copy().reset_index(drop=True)
print(f"Valid molecules: {len(calib_valid)}")

# ── Feature matrix: [fingerprint | Tg_MD_normalized] ─────────────────────────
X_fp  = fp_matrix.astype(float)
y     = calib_valid.tg_exp.values
Tg_md = calib_valid.tg_md.values

# Scale Tg_MD and concatenate
scaler_tg = StandardScaler()
Tg_md_scaled = scaler_tg.fit_transform(Tg_md.reshape(-1, 1))

# PCA on fingerprints for dimensionality reduction (keep 50 components)
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_fp)
print(f"PCA variance explained (50 components): {pca.explained_variance_ratio_.sum():.3f}")

X = np.hstack([X_pca, Tg_md_scaled])
print(f"Feature matrix shape: {X.shape}")

# ── GP model ─────────────────────────────────────────────────────────────────
# Scale y
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Kernel: RBF + white noise
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
    length_scale=np.ones(X.shape[1]),
    length_scale_bounds=(1e-2, 1e2)
) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=False,
    random_state=42
)

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\nRunning 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = np.zeros_like(y_scaled)
y_std_cv  = np.zeros_like(y_scaled)

for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr = y_scaled[tr_idx]
    gp_fold = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=False,
        random_state=42
    )
    gp_fold.fit(X_tr, y_tr)
    y_pred_cv[te_idx], y_std_cv[te_idx] = gp_fold.predict(X_te, return_std=True)
    mae_fold = mean_absolute_error(
        scaler_y.inverse_transform(y_scaled[te_idx].reshape(-1,1)),
        scaler_y.inverse_transform(y_pred_cv[te_idx].reshape(-1,1))
    )
    print(f"  Fold {fold+1}: MAE = {mae_fold:.2f} K")

# Convert back to K
y_pred_K = scaler_y.inverse_transform(y_pred_cv.reshape(-1, 1)).ravel()
y_std_K  = y_std_cv * scaler_y.scale_[0]

cv_mae = mean_absolute_error(y, y_pred_K)
cv_r2  = r2_score(y, y_pred_K)
r, _   = pearsonr(y, y_pred_K)
print(f"\nCV Results: MAE = {cv_mae:.2f} K, R² = {cv_r2:.4f}, r = {r:.4f}")

# Baseline: naive MD correction
baseline_pred = Tg_md - (Tg_md - y).mean()  # subtract mean bias
baseline_mae  = mean_absolute_error(y, baseline_pred)
print(f"Baseline (mean-bias subtraction): MAE = {baseline_mae:.2f} K")

# ── Fit final model on all data ───────────────────────────────────────────────
print("\nFitting final GP on all data...")
gp.fit(X, y_scaled)
print(f"Optimized kernel: {gp.kernel_}")

# ── Save model artifacts ──────────────────────────────────────────────────────
artifacts = {
    'gp': gp,
    'pca': pca,
    'scaler_tg': scaler_tg,
    'scaler_y':  scaler_y,
}
with open(OUT_DIR / "gp_calibration_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("Saved GP model to outputs/gp_calibration_model.pkl")

# Save CV predictions
cv_results = calib_valid.copy()
cv_results['tg_gp_pred'] = y_pred_K
cv_results['tg_gp_std']  = y_std_K
cv_results.to_csv(OUT_DIR / "gp_cv_predictions.csv", index=False)

# ── Figure 4: GP calibration performance ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) Parity plot: GP predictions vs experimental
ax = axes[0]
mn = min(y.min(), y_pred_K.min())
mx = max(y.max(), y_pred_K.max())
sc = ax.scatter(y, y_pred_K, c=y_std_K, cmap='viridis', alpha=0.7, s=40,
                edgecolors='k', lw=0.3)
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
plt.colorbar(sc, ax=ax, label='GP uncertainty (K)')
ax.set_xlabel("Experimental Tg (K)", fontsize=12)
ax.set_ylabel("GP-calibrated Tg prediction (K)", fontsize=12)
ax.set_title(f"(a) GP Calibration (5-fold CV)\nMAE = {cv_mae:.1f} K, R² = {cv_r2:.3f}", fontsize=12, fontweight='bold')

# (b) Compare: MD, GP, experimental
ax = axes[1]
residual_md = Tg_md - y
residual_gp = y_pred_K - y
ax.boxplot([residual_md, residual_gp], labels=['MD simulated', 'GP calibrated'],
           patch_artist=True,
           boxprops=dict(facecolor='lightsteelblue', color='navy'),
           medianprops=dict(color='red', lw=2))
ax.axhline(0, color='k', lw=1)
ax.set_ylabel("Predicted − Experimental Tg (K)", fontsize=12)
ax.set_title("(b) Residuals: MD vs. GP", fontsize=12, fontweight='bold')

# (c) GP uncertainty (std) vs absolute error
abs_err = np.abs(y_pred_K - y)
ax = axes[2]
ax.scatter(y_std_K, abs_err, alpha=0.5, s=35, c='purple', edgecolors='k', lw=0.3)
ax.set_xlabel("GP predictive std (K)", fontsize=12)
ax.set_ylabel("|Predicted − Experimental| (K)", fontsize=12)
ax.set_title("(c) Uncertainty Calibration", fontsize=12, fontweight='bold')
r_unc, _ = pearsonr(y_std_K, abs_err)
ax.text(0.05, 0.92, f"r = {r_unc:.3f}", transform=ax.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig(IMG_DIR / "fig4_gp_calibration.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_gp_calibration.png")

print("\nGP calibration complete.")
print(f"CV MAE: {cv_mae:.2f} K | R²: {cv_r2:.4f}")
print(f"MD baseline MAE: {baseline_mae:.2f} K")
print(f"GP improvement over MD baseline: {baseline_mae - cv_mae:.2f} K")
