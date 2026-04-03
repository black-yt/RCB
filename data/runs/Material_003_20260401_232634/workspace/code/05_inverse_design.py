"""
Inverse design: generate new vitrimer candidates with targeted Tg values.
Uses gradient-based optimization in the VAE latent space, then
nearest-neighbor decoding to find real vitrimer structures.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/data")
OUT_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/outputs")
IMG_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/report/images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load all artifacts ────────────────────────────────────────────────────────
with open(OUT_DIR / "gp_calibration_model.pkl", "rb") as f:
    gp_arts = pickle.load(f)
gp         = gp_arts['gp']
pca_gp     = gp_arts['pca']
scaler_tg  = gp_arts['scaler_tg']
scaler_y   = gp_arts['scaler_y']

with open(OUT_DIR / "pca_vae.pkl", "rb") as f:
    pca_vae = pickle.load(f)
with open(OUT_DIR / "scaler_X_vae.pkl", "rb") as f:
    scaler_X = pickle.load(f)

vitrimer = pd.read_csv(OUT_DIR / "vitrimer_latent.csv")
latent_codes = np.load(OUT_DIR / "latent_codes.npy")
tg_cal       = np.load(OUT_DIR / "tg_calibrated.npy")

tg_min = tg_cal.min()
tg_max = tg_cal.max()
print(f"Tg range (calibrated): {tg_min:.1f} – {tg_max:.1f} K")

# ── Reload VAE architecture ───────────────────────────────────────────────────
INPUT_DIM  = 256
HIDDEN_DIM = 512
LATENT_DIM = 64

class MolecularVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.LayerNorm(HIDDEN_DIM // 2), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.fc_var = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM // 2), nn.LayerNorm(HIDDEN_DIM // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM), nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    def decode(self, z):
        return self.decoder(z)
    def predict_tg(self, z):
        return self.predictor(z).squeeze(-1)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, self.predict_tg(z)

model = MolecularVAE().to(DEVICE)
model.load_state_dict(torch.load(OUT_DIR / "vae_model.pt", map_location=DEVICE))
model.eval()

# ── Helper: fingerprint functions ────────────────────────────────────────────
def smiles_to_fp(smi, n_bits=512, radius=2):
    smi_clean = smi.replace("*", "[H]") if "*" in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

def smi_to_latent(acid_smi, epx_smi):
    fp_acid = smiles_to_fp(acid_smi)
    fp_epx  = smiles_to_fp(epx_smi)
    if fp_acid is None or fp_epx is None:
        return None
    fp_cat = np.concatenate([fp_acid, fp_epx])
    fp_pca = pca_vae.transform(fp_cat.reshape(1, -1))
    fp_scl = scaler_X.transform(fp_pca).astype(np.float32)
    x = torch.tensor(fp_scl).to(DEVICE)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.cpu().numpy()

# ── GP prediction from fingerprint pair ──────────────────────────────────────
def predict_tg_gp(acid_smi, epx_smi, tg_md):
    fp_acid = smiles_to_fp(acid_smi)
    fp_epx  = smiles_to_fp(epx_smi)
    if fp_acid is None or fp_epx is None:
        return None, None
    fp_avg  = 0.5 * (fp_acid + fp_epx)
    fp_pca  = pca_gp.transform(fp_avg.reshape(1, -1))
    tg_scaled = scaler_tg.transform([[tg_md]])
    X = np.hstack([fp_pca, tg_scaled])
    y_scaled, y_std = gp.predict(X, return_std=True)
    tg_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
    tg_unc  = float(y_std[0]) * scaler_y.scale_[0]
    return tg_pred, tg_unc

# ── Gradient-based latent space optimization ─────────────────────────────────
def optimize_latent(target_tg_norm, n_steps=300, lr=0.05, init_z=None):
    """Optimize z to produce target Tg via VAE property predictor."""
    if init_z is None:
        z = torch.randn(1, LATENT_DIM, requires_grad=True, device=DEVICE)
    else:
        z = torch.tensor(init_z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z = z.detach().requires_grad_(True)
    target = torch.tensor([target_tg_norm], dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.Adam([z], lr=lr)
    for step in range(n_steps):
        optimizer.zero_grad()
        tg_p = model.predict_tg(z)
        # MSE to target + L2 regularization to stay near latent manifold
        loss = (tg_p - target).pow(2) + 0.01 * z.pow(2).mean()
        loss.backward()
        optimizer.step()
    return z.detach().cpu().numpy()[0]

# ── Nearest-neighbor matching in latent space ─────────────────────────────────
def find_nearest_vitrimers(z_query, top_k=10):
    """Find top-k vitrimers closest to z_query in latent space."""
    z_q = z_query.reshape(1, -1)
    dists = np.linalg.norm(latent_codes - z_q, axis=1)
    idxs  = np.argsort(dists)[:top_k]
    return vitrimer.iloc[idxs].copy().assign(latent_dist=dists[idxs])

# ── Design three target Tg categories ────────────────────────────────────────
# Low Tg (~300 K) – flexible vitrimers
# Medium Tg (~380 K) – typical service range
# High Tg (~470 K) – high-performance vitrimers
target_categories = {
    "Low Tg (~300 K)":    300.0,
    "Medium Tg (~380 K)": 380.0,
    "High Tg (~470 K)":   470.0,
}

all_candidates = []
print("Running inverse design via latent space optimization...")

for cat_name, tg_target in target_categories.items():
    tg_norm = (tg_target - tg_min) / (tg_max - tg_min)
    print(f"\n→ Target: {cat_name}  (Tg_target = {tg_target:.0f} K, norm = {tg_norm:.3f})")

    # Run several optimizations from different random initializations
    best_candidates = []
    for seed in range(10):
        torch.manual_seed(seed)
        z_opt = optimize_latent(tg_norm, n_steps=400, lr=0.03)
        neighbors = find_nearest_vitrimers(z_opt, top_k=5)
        # Use the nearest neighbor
        best = neighbors.iloc[0].to_dict()
        best['target_category'] = cat_name
        best['target_tg']       = tg_target
        best['z_query']         = z_opt
        best_candidates.append(best)

    # De-duplicate by acid SMILES and take top candidates
    seen = set()
    unique = []
    for c in best_candidates:
        k = (c['acid'], c['epoxide'])
        if k not in seen:
            seen.add(k)
            unique.append(c)
    all_candidates.extend(unique[:5])
    print(f"  Found {len(unique)} unique candidates")
    for c in unique[:3]:
        print(f"    Calibrated Tg = {c['tg_calibrated']:.1f} K  |  acid: {c['acid'][:40]}...")

# ── Build candidate DataFrame ─────────────────────────────────────────────────
candidates_df = pd.DataFrame([{
    'target_category': c['target_category'],
    'target_tg_K':     c['target_tg'],
    'acid':            c['acid'],
    'epoxide':         c['epoxide'],
    'tg_md_K':         c['tg'],
    'tg_calibrated_K': c['tg_calibrated'],
    'tg_uncertainty_K':c['tg_uncertainty'],
} for c in all_candidates])

# Add GP re-validation for each candidate
print("\nValidating candidates with GP calibration model...")
gp_vals = [predict_tg_gp(row.acid, row.epoxide, row.tg_md_K)
           for _, row in candidates_df.iterrows()]
candidates_df['tg_gp_revalidated'] = [v[0] for v in gp_vals]
candidates_df['tg_gp_uncertainty'] = [v[1] for v in gp_vals]

# Add molecular descriptors
def calc_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.nan, np.nan, np.nan
    return Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumRotatableBonds(mol)

candidates_df[['acid_MW','acid_LogP','acid_RotBonds']]   = [calc_descriptors(s) for s in candidates_df.acid]
candidates_df[['epx_MW', 'epx_LogP', 'epx_RotBonds']]   = [calc_descriptors(s) for s in candidates_df.epoxide]
candidates_df['total_MW'] = candidates_df.acid_MW + candidates_df.epx_MW

candidates_df.to_csv(OUT_DIR / "designed_candidates.csv", index=False)
print(f"\nTotal candidates: {len(candidates_df)}")
print(candidates_df[['target_category','tg_calibrated_K','tg_gp_revalidated','total_MW']].to_string())

# ── Figure 7: Inverse design results ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

colors = {'Low Tg (~300 K)': '#2196F3', 'Medium Tg (~380 K)': '#4CAF50', 'High Tg (~470 K)': '#F44336'}

# (a) Latent space PCA with designed candidates
from sklearn.decomposition import PCA as PCA2
pca2d = PCA2(n_components=2)
z2d_all = pca2d.fit_transform(latent_codes)

ax = axes[0]
sc = ax.scatter(z2d_all[:, 0], z2d_all[:, 1], c=tg_cal, cmap='coolwarm',
                s=3, alpha=0.15, rasterized=True, zorder=1)
plt.colorbar(sc, ax=ax, label='Calibrated Tg (K)', shrink=0.85)

# Plot designed candidates
for cat, ccolor in colors.items():
    sub = candidates_df[candidates_df.target_category == cat]
    # encode each candidate
    z_cands = []
    for _, row in sub.iterrows():
        z_c = smi_to_latent(row.acid, row.epoxide)
        if z_c is not None:
            z_cands.append(z_c[0])
    if z_cands:
        z_proj = pca2d.transform(np.array(z_cands))
        ax.scatter(z_proj[:, 0], z_proj[:, 1], c=ccolor, s=120, marker='*',
                   edgecolors='k', lw=0.5, zorder=5, label=cat)

ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_title("(a) Latent Space with Designed Candidates", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, markerscale=0.8)

# (b) Predicted vs target Tg for candidates
ax = axes[1]
for i, (cat, ccolor) in enumerate(colors.items()):
    sub = candidates_df[candidates_df.target_category == cat]
    x_pos = [i + 0.1 * j for j in range(len(sub))]
    ax.scatter(x_pos, sub.tg_calibrated_K, c=ccolor, s=80, label=cat, zorder=5, edgecolors='k', lw=0.5)
    ax.errorbar(x_pos, sub.tg_gp_revalidated, yerr=sub.tg_gp_uncertainty,
                fmt='D', color=ccolor, alpha=0.6, capsize=4, ms=5, zorder=4)
    target_val = sub.target_tg_K.iloc[0]
    ax.axhline(target_val, xmin=i/3+0.02, xmax=(i+1)/3-0.02,
               color=ccolor, linestyle='--', lw=2, alpha=0.7)

ax.set_xticks([0.2, 1.2, 2.2])
ax.set_xticklabels(['Low Tg\n(~300 K)', 'Medium Tg\n(~380 K)', 'High Tg\n(~470 K)'], fontsize=10)
ax.set_ylabel("Predicted Tg (K)", fontsize=12)
ax.set_title("(b) Designed Candidates:\nCalibrated (●) and Re-validated GP (◆)", fontsize=11, fontweight='bold')
circles = mpatches.Patch(color='gray', label='Latent-NN Tg')
diamonds= mpatches.Patch(color='gray', alpha=0.5, label='GP re-validated Tg')
ax.legend(handles=[circles, diamonds], fontsize=9)

# (c) Tg distribution of candidates vs full vitrimer dataset
ax = axes[2]
ax.hist(tg_cal, bins=50, alpha=0.4, label='All vitrimers (calibrated)', color='gray', edgecolor='k', lw=0.2)
for cat, ccolor in colors.items():
    sub = candidates_df[candidates_df.target_category == cat]
    ax.scatter(sub.tg_calibrated_K, np.ones(len(sub)) * 20, c=ccolor, s=150, marker='|',
               linewidths=3, label=cat, zorder=5)
ax.set_xlabel("Calibrated Tg (K)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("(c) Designed Candidates vs.\nFull Vitrimer Space", fontsize=11, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG_DIR / "fig7_inverse_design.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_inverse_design.png")

# ── Figure 8: Property analysis of candidates ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) MW vs Tg for candidates and background
ax = axes[0]
def mol_wt(s):
    m = Chem.MolFromSmiles(s)
    return Descriptors.MolWt(m) if m else np.nan
ax.scatter(vitrimer.acid.apply(mol_wt) + vitrimer.epoxide.apply(mol_wt),
           tg_cal, c='lightgray', s=5, alpha=0.3, label='All vitrimers', rasterized=True)
for cat, ccolor in colors.items():
    sub = candidates_df[candidates_df.target_category == cat]
    ax.scatter(sub.total_MW, sub.tg_calibrated_K, c=ccolor, s=100, label=cat,
               edgecolors='k', lw=0.8, zorder=5)
ax.set_xlabel("Total MW (acid + epoxide) (g/mol)", fontsize=12)
ax.set_ylabel("Calibrated Tg (K)", fontsize=12)
ax.set_title("(a) MW vs. Tg for Designed Candidates", fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# (b) Uncertainty comparison: candidates vs dataset
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

# ── Figure 9: Top candidate showcase ─────────────────────────────────────────
# Select top 3 candidates (one per category, highest confidence)
top_cands = []
for cat in target_categories:
    sub = candidates_df[candidates_df.target_category == cat].sort_values('tg_uncertainty_K')
    top_cands.append(sub.iloc[0])
top_df = pd.DataFrame(top_cands)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
cat_list = list(target_categories.keys())
for i, (_, row) in enumerate(top_df.iterrows()):
    # Acid
    ax = axes[i][0]
    mol_acid = Chem.MolFromSmiles(row.acid)
    if mol_acid:
        from rdkit.Chem import Draw
        from io import BytesIO
        from PIL import Image
        img = Draw.MolToImage(mol_acid, size=(300, 200))
        ax.imshow(img)
    ax.set_title(f"Acid component\n{cat_list[i]}", fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.text(0.5, -0.05, f"MW = {row.acid_MW:.0f} g/mol, LogP = {row.acid_LogP:.2f}",
            transform=ax.transAxes, ha='center', fontsize=8)

    # Epoxide
    ax = axes[i][1]
    mol_epx = Chem.MolFromSmiles(row.epoxide)
    if mol_epx:
        img = Draw.MolToImage(mol_epx, size=(300, 200))
        ax.imshow(img)
    ax.set_title(
        f"Epoxide component\nTg_cal = {row.tg_calibrated_K:.1f} K, "
        f"Tg_GP = {row.tg_gp_revalidated:.1f} ± {row.tg_gp_uncertainty:.1f} K",
        fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.text(0.5, -0.05, f"MW = {row.epx_MW:.0f} g/mol, LogP = {row.epx_LogP:.2f}",
            transform=ax.transAxes, ha='center', fontsize=8)

plt.suptitle("Top Designed Vitrimer Candidates per Target Tg Category", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(IMG_DIR / "fig9_top_candidates.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_top_candidates.png")

print("\nInverse design complete.")
print(candidates_df[['target_category','tg_calibrated_K','tg_gp_revalidated','tg_gp_uncertainty','total_MW']].to_string())
