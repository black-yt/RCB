"""
Molecular Variational Autoencoder for vitrimer inverse design.
Uses combined Morgan fingerprints of acid+epoxide pairs as input.
Trains a VAE to learn a smooth latent space, then performs
targeted optimization to generate new vitrimer candidates with desired Tg.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/data")
OUT_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/outputs")
IMG_DIR  = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260401_232634/report/images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load calibrated vitrimer data ─────────────────────────────────────────────
vitrimer = pd.read_csv(OUT_DIR / "vitrimer_calibrated_tg.csv")
print(f"Loaded {len(vitrimer)} calibrated vitrimer systems")

# ── Fingerprint function ──────────────────────────────────────────────────────
def smiles_to_fp(smi, n_bits=512, radius=2):
    smi_clean = smi.replace("*", "[H]") if "*" in smi else smi
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

# ── Build fingerprint matrix: concatenated acid || epoxide ────────────────────
print("Building fingerprint matrix (concat acid|epoxide)...")
fps_cat = []
valid_idx = []
for i, row in vitrimer.iterrows():
    fp_acid = smiles_to_fp(row.acid)
    fp_epx  = smiles_to_fp(row.epoxide)
    if fp_acid is None or fp_epx is None:
        continue
    fps_cat.append(np.concatenate([fp_acid, fp_epx]))  # 1024-dim
    valid_idx.append(i)

X = np.array(fps_cat, dtype=np.float32)  # (N, 1024)
vit_sub = vitrimer.iloc[valid_idx].reset_index(drop=True)
tg_cal  = vit_sub.tg_calibrated.values.astype(np.float32)
print(f"Fingerprint matrix: {X.shape}")

# ── Reduce to manageable size with PCA ───────────────────────────────────────
print("PCA reduction (256 components)...")
pca_vae = PCA(n_components=256, random_state=42)
X_pca = pca_vae.fit_transform(X).astype(np.float32)
print(f"PCA variance explained: {pca_vae.explained_variance_ratio_.sum():.3f}")
# Save PCA
with open(OUT_DIR / "pca_vae.pkl", "wb") as f:
    pickle.dump(pca_vae, f)

# Scale features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_pca).astype(np.float32)
with open(OUT_DIR / "scaler_X_vae.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

# Scale Tg to [0,1] for property predictor
tg_min, tg_max = tg_cal.min(), tg_cal.max()
tg_norm = ((tg_cal - tg_min) / (tg_max - tg_min)).astype(np.float32)
print(f"Tg range (calibrated): {tg_min:.1f} – {tg_max:.1f} K")

# ── VAE architecture ──────────────────────────────────────────────────────────
INPUT_DIM  = 256
HIDDEN_DIM = 512
LATENT_DIM = 64   # latent space dimensionality

class MolecularVAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # input is [0,1] scaled
        )

        # Property predictor (from latent z → Tg)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def predict_tg(self, z):
        return self.predictor(z).squeeze(-1)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        tg_pred = self.predict_tg(z)
        return recon, mu, log_var, tg_pred

# ── Training ──────────────────────────────────────────────────────────────────
model = MolecularVAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

X_tensor  = torch.tensor(X_scaled, dtype=torch.float32)
tg_tensor = torch.tensor(tg_norm,  dtype=torch.float32)
dataset   = TensorDataset(X_tensor, tg_tensor)
loader    = DataLoader(dataset, batch_size=256, shuffle=True)

def vae_loss(recon, x, mu, log_var, tg_pred, tg_true, beta=0.01, gamma=5.0):
    """ELBO + property prediction loss"""
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl_div     = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    prop_loss  = nn.functional.mse_loss(tg_pred, tg_true)
    return recon_loss + beta * kl_div + gamma * prop_loss, recon_loss, kl_div, prop_loss

print("\nTraining VAE...")
train_losses = []
EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, tgb in loader:
        xb, tgb = xb.to(DEVICE), tgb.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, log_var, tg_pred = model(xb)
        loss, rl, kl, pl = vae_loss(recon, xb, mu, log_var, tg_pred, tgb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    scheduler.step()
    avg_loss = total_loss / len(dataset)
    train_losses.append(avg_loss)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), OUT_DIR / "vae_model.pt")
np.save(OUT_DIR / "vae_train_losses.npy", np.array(train_losses))
print(f"VAE trained. Final loss: {train_losses[-1]:.4f}")

# ── Encode all vitrimer data to latent space ──────────────────────────────────
model.eval()
with torch.no_grad():
    X_t = X_tensor.to(DEVICE)
    mu_all, _ = model.encode(X_t)
    mu_all = mu_all.cpu().numpy()
    tg_pred_all = model.predict_tg(torch.tensor(mu_all).to(DEVICE)).cpu().numpy()

# De-normalize Tg prediction
tg_pred_K = tg_pred_all * (tg_max - tg_min) + tg_min

vit_sub['tg_vae_pred'] = tg_pred_K
vit_sub['latent_idx']  = range(len(vit_sub))
for j in range(LATENT_DIM):
    vit_sub[f'z_{j}'] = mu_all[:, j]

vit_sub.to_csv(OUT_DIR / "vitrimer_latent.csv", index=False)
print(f"\nVAE property prediction correlation with calibrated Tg:")
from scipy.stats import pearsonr
r, _ = pearsonr(tg_cal, tg_pred_K)
mae  = np.mean(np.abs(tg_pred_K - tg_cal))
print(f"  r = {r:.4f}, MAE = {mae:.2f} K")
np.save(OUT_DIR / "latent_codes.npy",  mu_all)
np.save(OUT_DIR / "tg_calibrated.npy", tg_cal)

# ── Figure 6: VAE training and latent space ───────────────────────────────────
from sklearn.decomposition import PCA as PCA_viz

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) Training loss curve
ax = axes[0]
ax.plot(train_losses, color='royalblue', lw=1.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Total loss", fontsize=12)
ax.set_title("(a) VAE Training Loss", fontsize=12, fontweight='bold')
ax.set_yscale('log')

# (b) PCA of latent space colored by calibrated Tg
pca2 = PCA_viz(n_components=2)
z2d = pca2.fit_transform(mu_all)
ax = axes[1]
sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=tg_cal, cmap='coolwarm', s=3, alpha=0.3, rasterized=True)
plt.colorbar(sc, ax=ax, label='Calibrated Tg (K)')
ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_title("(b) Latent Space (PCA)\nColored by Tg", fontsize=12, fontweight='bold')

# (c) VAE-predicted vs calibrated Tg
ax = axes[2]
ax.scatter(tg_cal, tg_pred_K, alpha=0.2, s=5, c='seagreen', rasterized=True)
mn, mx = tg_cal.min(), tg_cal.max()
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
ax.set_xlabel("GP-calibrated Tg (K)", fontsize=12)
ax.set_ylabel("VAE-predicted Tg (K)", fontsize=12)
ax.set_title(f"(c) VAE Property Prediction\nr = {r:.3f}, MAE = {mae:.1f} K",
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(IMG_DIR / "fig6_vae_latent.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_vae_latent.png")
print("\nVAE training complete.")
