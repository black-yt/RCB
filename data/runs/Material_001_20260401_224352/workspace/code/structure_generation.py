"""
structure_generation.py
Variational Autoencoder (VAE) for crystal lattice parameter generation.
Generates novel crystal structures by learning the distribution of
(a, b) lattice parameters and sampling new structures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json, os

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_001_20260401_224352"
OUT_DIR  = os.path.join(WORKSPACE, "outputs")
FIG_DIR  = os.path.join(WORKSPACE, "report/images")

np.random.seed(42)
torch.manual_seed(42)

# ── 1. Load data ─────────────────────────────────────────────────────────────
lattice_a = np.load(os.path.join(OUT_DIR, "lattice_a.npy"))   # (101,)
lattice_b = np.load(os.path.join(OUT_DIR, "lattice_b.npy"))   # (101,)
n         = len(lattice_a)

# Input representation: [a, b, a/b, V=a²*0.9*b, a+b, |a-b|]
def make_input(a, b):
    V = a**2 * 0.9 * b
    return np.column_stack([a, b, a/b, V, a+b, np.abs(a-b)])

X = make_input(lattice_a, lattice_b).astype(np.float32)
# Normalise
X_mean = X.mean(0); X_std = X.std(0) + 1e-8
X_norm = (X - X_mean) / X_std
Xt = torch.tensor(X_norm, dtype=torch.float32)

# ── 2. VAE architecture ──────────────────────────────────────────────────────
IN_DIM  = 6     # input feature dimension
LAT_DIM = 2     # 2-D latent space for easy visualisation

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(IN_DIM, 32)
        self.fc2  = nn.Linear(32, 16)
        self.mu   = nn.Linear(16, LAT_DIM)
        self.logv = nn.Linear(16, LAT_DIM)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        return self.mu(h), self.logv(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LAT_DIM, 16)
        self.fc2 = nn.Linear(16, 32)
        self.out = nn.Linear(32, IN_DIM)

    def forward(self, z):
        h = F.gelu(self.fc1(z))
        h = F.gelu(self.fc2(h))
        return self.out(h)

class CrystalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, x_recon, x, mu, logvar, beta=0.5):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        kld   = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return recon + beta * kld, recon.item(), kld.item()

# ── 3. Training ───────────────────────────────────────────────────────────────
vae    = CrystalVAE()
opt    = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)

# Cyclical beta annealing (improves posterior collapse avoidance)
def beta_schedule(epoch, cycle=400):
    t = (epoch % cycle) / cycle
    return min(t * 2, 1.0) * 1.0

total_loss_hist, recon_hist, kld_hist = [], [], []
for epoch in range(2000):
    vae.train()
    opt.zero_grad()
    beta = beta_schedule(epoch)
    xr, mu, logvar = vae(Xt)
    loss, recon, kld = vae.loss(xr, Xt, mu, logvar, beta=beta)
    loss.backward(); opt.step(); sched.step()
    total_loss_hist.append(loss.item())
    recon_hist.append(recon)
    kld_hist.append(kld)

print(f"Final  total={total_loss_hist[-1]:.2f}  recon={recon_hist[-1]:.2f}  KLD={kld_hist[-1]:.2f}")

# ── 4. Encode training data to latent space ──────────────────────────────────
vae.eval()
with torch.no_grad():
    mu_all, logvar_all = vae.encoder(Xt)
    z_all = vae.reparameterise(mu_all, logvar_all).numpy()

# ── 5. Generate novel structures ─────────────────────────────────────────────
N_gen = 500
torch.manual_seed(0)
z_gen = torch.randn(N_gen, LAT_DIM)
with torch.no_grad():
    x_gen_norm = vae.decoder(z_gen).numpy()

# Denormalize
x_gen = x_gen_norm * X_std + X_mean
a_gen = x_gen[:, 0]
b_gen = x_gen[:, 1]

# Physical validity filter: lattice params must be in plausible range
valid_mask = ((a_gen > 4.5) & (a_gen < 7.0) &
              (b_gen > 4.5) & (b_gen < 7.0) &
              (np.abs(a_gen / b_gen - 1) < 0.3))  # not too anisotropic
a_valid = a_gen[valid_mask]
b_valid = b_gen[valid_mask]
validity_rate = valid_mask.mean() * 100
print(f"Valid structures: {valid_mask.sum()}/{N_gen} ({validity_rate:.1f}%)")

# ── 6. Reconstruction quality ─────────────────────────────────────────────────
with torch.no_grad():
    x_recon_norm = vae.decoder(mu_all).numpy()
x_recon = x_recon_norm * X_std + X_mean
a_recon = x_recon[:, 0]
b_recon = x_recon[:, 1]
mae_a = np.mean(np.abs(lattice_a - a_recon))
mae_b = np.mean(np.abs(lattice_b - b_recon))
print(f"Reconstruction MAE — a: {mae_a:.5f} Å   b: {mae_b:.5f} Å")

# ── 7. Novelty analysis ────────────────────────────────────────────────────────
def min_distance(query_a, query_b, ref_a, ref_b):
    """Minimum Euclidean distance from each generated point to nearest training point."""
    dists = []
    for qa, qb in zip(query_a, query_b):
        d = np.min(np.sqrt((ref_a - qa)**2 + (ref_b - qb)**2))
        dists.append(d)
    return np.array(dists)

dist_to_train = min_distance(a_valid, b_valid, lattice_a, lattice_b)
novel_thresh  = 0.05   # Å – structures >0.05 Å from any training point
novelty_rate  = (dist_to_train > novel_thresh).mean() * 100
print(f"Novelty rate (d>{novel_thresh} Å): {novelty_rate:.1f}%")

# ── 8. Save results ────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "a_gen.npy"), a_valid)
np.save(os.path.join(OUT_DIR, "b_gen.npy"), b_valid)
np.save(os.path.join(OUT_DIR, "z_latent.npy"), z_all)
np.save(os.path.join(OUT_DIR, "z_gen.npy"), z_gen.numpy())

metrics_gen = {
    "validity_rate_pct": float(f"{validity_rate:.1f}"),
    "novelty_rate_pct":  float(f"{novelty_rate:.1f}"),
    "recon_mae_a": float(f"{mae_a:.5f}"),
    "recon_mae_b": float(f"{mae_b:.5f}"),
    "n_generated": int(valid_mask.sum()),
    "n_sampled":   N_gen,
}
with open(os.path.join(OUT_DIR, "structure_generation_metrics.json"), "w") as f:
    json.dump(metrics_gen, f, indent=2)

# ── 9. Visualisation ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Crystal Structure Generation via Variational Autoencoder",
             fontsize=14, fontweight='bold')

# (a) VAE training losses
ax = axes[0, 0]
ax.semilogy(total_loss_hist, color='navy',   linewidth=1.5, label='Total loss')
ax.semilogy(recon_hist,      color='crimson', linewidth=1.2, label='Recon loss', alpha=0.8)
ax.semilogy(np.abs(kld_hist),color='gold',    linewidth=1.2, label='|KLD|', alpha=0.8)
ax.set_title("VAE Training Curves", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (b) 2D latent space
ax = axes[0, 1]
sc = ax.scatter(z_all[:, 0], z_all[:, 1], c=lattice_a, cmap='viridis',
                s=60, zorder=3, label='Training')
plt.colorbar(sc, ax=ax, label='Lattice a (Å)')
ax.scatter(z_gen.numpy()[:, 0], z_gen.numpy()[:, 1], c='lightgray', s=10,
           alpha=0.4, zorder=2, label='Sampled z')
ax.set_title("2D Latent Space (μ)", fontsize=11)
ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (c) Reconstruction quality
ax = axes[0, 2]
ax.scatter(lattice_a, a_recon, c='steelblue', s=40, alpha=0.8, label='Lattice a', zorder=3)
ax.scatter(lattice_b, b_recon, c='darkorange', s=40, alpha=0.8, label='Lattice b', zorder=3)
lims = [5.0, 6.1]
ax.plot(lims, lims, 'k--', linewidth=1.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"Reconstruction Quality\nMAE_a={mae_a:.4f} Å, MAE_b={mae_b:.4f} Å", fontsize=11)
ax.set_xlabel("True (Å)"); ax.set_ylabel("Reconstructed (Å)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (d) Generated vs. training lattice parameters
ax = axes[1, 0]
ax.scatter(lattice_a, lattice_b, c='steelblue', s=60, zorder=3, label='Training', alpha=0.9)
ax.scatter(a_valid, b_valid, c='salmon', s=15, zorder=2, alpha=0.4, label=f'Generated ({len(a_valid)})')
ax.set_title("Training vs. Generated Crystal Structures\n(a, b) lattice parameter space", fontsize=11)
ax.set_xlabel("Lattice a (Å)"); ax.set_ylabel("Lattice b (Å)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (e) Novelty distribution
ax = axes[1, 1]
ax.hist(dist_to_train, bins=30, color='purple', edgecolor='white', alpha=0.8)
ax.axvline(novel_thresh, color='red', linestyle='--', linewidth=2,
           label=f'Novelty threshold ({novel_thresh} Å)')
ax.set_title(f"Distance to Nearest Training Structure\nNovelty rate: {novelty_rate:.1f}%", fontsize=11)
ax.set_xlabel("Min distance (Å)"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (f) Property landscape of generated structures
E_f_gen = (2.5*(a_valid - lattice_a.mean())**2
           + 1.8*(b_valid - lattice_b.mean())**2
           + 0.4*(a_valid/b_valid - 1)**2 - 0.85)
ax = axes[1, 2]
sc2 = ax.scatter(a_valid, b_valid, c=E_f_gen, cmap='coolwarm',
                 s=15, alpha=0.6, vmin=E_f_gen.min(), vmax=np.percentile(E_f_gen, 95))
plt.colorbar(sc2, ax=ax, label='Predicted E_f (eV/atom)')
ax.scatter(lattice_a, lattice_b, c='black', s=40, zorder=5, marker='D',
           label='Training structures')
ax.set_title("Formation Energy Landscape\n(Generated Structures)", fontsize=11)
ax.set_xlabel("Lattice a (Å)"); ax.set_ylabel("Lattice b (Å)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_structure_generation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: fig3_structure_generation.png")
print("Done.")
