"""
Optimized LES analysis for Random Charges dataset.
Pre-computes features and Coulomb matrices to speed up training.
"""
import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, compute_total_energy_rc
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt
import time

BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_003_20260401_190857'
DATA_DIR = f'{BASE}/data'
OUT_DIR = f'{BASE}/outputs'
IMG_DIR = f'{BASE}/report/images'

DEVICE = 'cpu'  # 128-atom Coulomb fits in CPU


def precompute_features(positions_list, cutoff=6.0, n_rbf=16):
    """Pre-compute local features for all configs. Returns list of (N, n_rbf) arrays."""
    from fast_les_model import compute_features_fast
    features = []
    for i, pos in enumerate(positions_list):
        pos_t = torch.tensor(pos, dtype=torch.float32)
        f = compute_features_fast(pos_t, cutoff=cutoff, n_rbf=n_rbf)
        features.append(f.detach())
        if (i + 1) % 20 == 0:
            print(f'    Features computed: {i+1}/{len(positions_list)}')
    return features


def precompute_coulomb_matrices(positions_list, lj_sigma=2.0, lj_eps=1.0):
    """
    Pre-compute:
    - Coulomb matrix A_ij = 1/r_ij (i≠j), A_ii = 0
    - LJ repulsion energy (doesn't depend on charges)
    """
    coulomb_mats = []
    lj_energies = []

    for pos in positions_list:
        pos_t = torch.tensor(pos, dtype=torch.float32)
        N = pos_t.shape[0]
        diff = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)

        # Coulomb matrix (1/r, zero diagonal)
        with torch.no_grad():
            inv_r = torch.where(dist > 1e-6, 1.0 / dist, torch.zeros_like(dist))
            inv_r.fill_diagonal_(0.0)
        coulomb_mats.append(inv_r.detach())

        # LJ repulsive energy
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        r_ij = dist[mask]
        lj_e = lj_eps * (lj_sigma / r_ij).pow(12)
        lj_energies.append(lj_e.sum().item())

    return coulomb_mats, np.array(lj_energies)


class LESNetPrecomputed(nn.Module):
    """
    LES model using pre-computed features and Coulomb matrices.
    Forward pass is just NN + matrix multiply.
    """
    def __init__(self, n_rbf=16, n_hidden=32):
        super().__init__()
        self.charge_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        self.sr_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, features, coulomb_mat, lj_energy, total_charge=0.0):
        """
        features: (N, n_rbf)
        coulomb_mat: (N, N) precomputed 1/r matrix
        lj_energy: scalar
        Returns: total energy, latent charges
        """
        N = features.shape[0]
        # Predict charges
        raw_q = self.charge_net(features).squeeze(-1)
        correction = (total_charge - raw_q.sum()) / N
        latent_q = raw_q + correction

        # Coulomb energy: E = 0.5 * q^T @ A @ q
        E_lr = 0.5 * (latent_q @ (coulomb_mat @ latent_q))

        # Short-range energy
        E_sr = self.sr_net(features).squeeze(-1).sum()

        return E_lr + E_sr + lj_energy, latent_q


class SRNetPrecomputed(nn.Module):
    """Short-range model using pre-computed features."""
    def __init__(self, n_rbf=16, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, features, coulomb_mat=None, lj_energy=0.0, total_charge=0.0):
        """Only uses features, ignores Coulomb matrix."""
        return self.net(features).squeeze(-1).sum() + lj_energy, None


def train_precomputed(model, features_list, coulomb_mats, lj_energies,
                      energies, total_charges=None,
                      n_epochs=200, lr=1e-3, verbose=True):
    """Fast training using pre-computed quantities."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n = len(features_list)
    if total_charges is None:
        total_charges = [0.0] * n

    e_ref = torch.tensor(energies, dtype=torch.float32)
    e_std = e_ref.std().item() + 1e-8

    losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        perm = np.random.permutation(n)

        for idx in perm:
            optimizer.zero_grad()
            features = features_list[idx].clone().requires_grad_(False)
            cm = coulomb_mats[idx]
            lj_e = float(lj_energies[idx])
            tc = float(total_charges[idx])

            e_pred, q = model(features, cm, lj_e, tc)
            loss = ((e_pred - e_ref[idx]) ** 2) / (e_std ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / n
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 25 == 0:
            print(f'    Epoch {epoch+1:4d}: loss={avg_loss:.6f}')

    return losses


def evaluate_precomputed(model, features_list, coulomb_mats, lj_energies,
                          energies, total_charges=None, return_charges=False):
    model.eval()
    n = len(features_list)
    if total_charges is None:
        total_charges = [0.0] * n

    preds, all_q = [], []
    with torch.no_grad():
        for i in range(n):
            e, q = model(features_list[i], coulomb_mats[i], float(lj_energies[i]), float(total_charges[i]))
            preds.append(e.item())
            if q is not None:
                all_q.append(q.numpy().copy())

    preds = np.array(preds)
    refs = np.array(energies)
    mae = np.mean(np.abs(preds - refs))
    rmse = np.sqrt(np.mean((preds - refs) ** 2))
    r2 = 1 - np.mean((preds - refs) ** 2) / (np.var(refs) + 1e-10)

    res = {'energies_pred': preds, 'mae': mae, 'rmse': rmse, 'r2': r2}
    if return_charges:
        res['latent_charges'] = all_q
    return res


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Analysis 1: Random Charges Benchmark (Optimized)")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    frames = parse_extxyz(f'{DATA_DIR}/random_charges.xyz')
    n = len(frames)
    print(f"  {n} configs of {len(frames[0]['positions'])} atoms")

    positions_list = [fr['positions'] for fr in frames]
    true_charges_list = [fr['true_charges'] for fr in frames]

    # 2. Compute reference energies
    print("\n[2] Computing reference Coulomb+LJ energies...")
    sigma = 2.0; epsilon = 1.0
    energies_total = []
    energies_coulomb = []
    energies_rep = []
    for fr in frames:
        e_tot, e_c, e_r = compute_total_energy_rc(fr['positions'], fr['true_charges'],
                                                    sigma=sigma, epsilon=epsilon)
        energies_total.append(e_tot)
        energies_coulomb.append(e_c)
        energies_rep.append(e_r)

    energies_total = np.array(energies_total)
    energies_coulomb = np.array(energies_coulomb)
    energies_rep = np.array(energies_rep)
    print(f"  Coulomb energy range: {energies_coulomb.min():.2f} to {energies_coulomb.max():.2f}")
    print(f"  LJ energy range:      {energies_rep.min():.2f} to {energies_rep.max():.2f}")
    print(f"  Total energy range:   {energies_total.min():.2f} to {energies_total.max():.2f}")

    # 3. Pre-compute features and Coulomb matrices
    print("\n[3] Pre-computing features (cutoff=6Å)...")
    t0 = time.time()
    features_list = precompute_features(positions_list, cutoff=6.0, n_rbf=16)
    print(f"  Features computed in {time.time()-t0:.1f}s")

    print("\n[4] Pre-computing Coulomb matrices...")
    t0 = time.time()
    coulomb_mats, lj_energies_precomp = precompute_coulomb_matrices(
        positions_list, lj_sigma=sigma, lj_eps=epsilon)
    print(f"  Coulomb matrices computed in {time.time()-t0:.1f}s")

    # Verify that precomputed LJ energies match
    lj_err = np.abs(lj_energies_precomp - energies_rep).max()
    print(f"  LJ energy max discrepancy: {lj_err:.2e}")

    # 4. Train/test split
    idx = np.random.permutation(n)
    n_train = 70
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    f_train = [features_list[i] for i in train_idx]
    f_test = [features_list[i] for i in test_idx]
    cm_train = [coulomb_mats[i] for i in train_idx]
    cm_test = [coulomb_mats[i] for i in test_idx]
    lj_train = lj_energies_precomp[train_idx]
    lj_test = lj_energies_precomp[test_idx]
    e_train = energies_total[train_idx]
    e_test = energies_total[test_idx]
    tc_train = np.zeros(n_train)  # All neutral

    # 5. Train SR model
    print("\n[5] Training Short-Range (SR) baseline model...")
    t0 = time.time()
    sr_model = SRNetPrecomputed(n_rbf=16, n_hidden=64)
    sr_losses = train_precomputed(sr_model, f_train, cm_train, lj_train, e_train,
                                   n_epochs=300, lr=1e-3, verbose=True)
    print(f"  Training time: {time.time()-t0:.1f}s")
    sr_tr = evaluate_precomputed(sr_model, f_train, cm_train, lj_train, e_train)
    sr_te = evaluate_precomputed(sr_model, f_test, cm_test, lj_test, e_test)
    print(f"  SR Train MAE={sr_tr['mae']:.4f}, R²={sr_tr['r2']:.4f}")
    print(f"  SR Test  MAE={sr_te['mae']:.4f}, R²={sr_te['r2']:.4f}")

    # 6. Train LES model
    print("\n[6] Training LES model...")
    t0 = time.time()
    les_model = LESNetPrecomputed(n_rbf=16, n_hidden=64)
    les_losses = train_precomputed(les_model, f_train, cm_train, lj_train, e_train,
                                    n_epochs=300, lr=1e-3, verbose=True)
    print(f"  Training time: {time.time()-t0:.1f}s")
    les_tr = evaluate_precomputed(les_model, f_train, cm_train, lj_train, e_train, return_charges=True)
    les_te = evaluate_precomputed(les_model, f_test, cm_test, lj_test, e_test, return_charges=True)
    print(f"  LES Train MAE={les_tr['mae']:.4f}, R²={les_tr['r2']:.4f}")
    print(f"  LES Test  MAE={les_te['mae']:.4f}, R²={les_te['r2']:.4f}")

    # 7. Save results
    np.save(f'{OUT_DIR}/rc_results.npy', {
        'sr_train': sr_tr, 'sr_test': sr_te, 'les_train': les_tr, 'les_test': les_te,
        'sr_losses': sr_losses, 'les_losses': les_losses,
        'train_idx': train_idx, 'test_idx': test_idx,
        'energies_total': energies_total, 'energies_coulomb': energies_coulomb,
        'energies_rep': energies_rep,
        'true_charges_list': [true_charges_list[i] for i in train_idx],
    }, allow_pickle=True)

    # 8. Figures
    print("\n[7] Generating figures...")
    fig_rc_data_overview(frames, energies_coulomb, energies_rep)
    fig_rc_energy_parity(sr_te, les_te, e_test)
    fig_rc_latent_charges(les_tr, [true_charges_list[i] for i in train_idx])
    fig_rc_training_curves(sr_losses, les_losses)

    print(f"\n[Summary]")
    print(f"  SR  MAE (test): {sr_te['mae']:.4f}")
    print(f"  LES MAE (test): {les_te['mae']:.4f}")
    improvement = (1 - les_te['mae'] / (sr_te['mae'] + 1e-10)) * 100
    print(f"  Improvement:    {improvement:.1f}%")

    return sr_te, les_te, les_tr


def fig_rc_data_overview(frames, energies_coulomb, energies_rep):
    """Figure 1: Random charges dataset overview."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: 3D projection
    ax = axes[0]
    pos = frames[0]['positions']
    tc = frames[0]['true_charges']
    ax.scatter(pos[tc > 0, 0], pos[tc > 0, 1], c='crimson', s=30, alpha=0.6,
               label='+1e (n=64)', edgecolors='darkred', lw=0.3)
    ax.scatter(pos[tc < 0, 0], pos[tc < 0, 1], c='royalblue', s=30, alpha=0.6,
               label='−1e (n=64)', edgecolors='darkblue', lw=0.3)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_title('(a) Random Charges Configuration\n(x-y projection, 128 atoms)')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    # Panel 2: Min pairwise distance distribution
    ax = axes[1]
    min_dists = []
    for fr in frames:
        pos = fr['positions']
        dm = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        np.fill_diagonal(dm, np.inf)
        min_dists.append(dm.min(axis=1))
    min_dists = np.concatenate(min_dists)
    ax.hist(min_dists, bins=30, color='steelblue', alpha=0.7, density=True, edgecolor='k', lw=0.3)
    ax.axvline(x=2.0, color='r', linestyle='--', label='LJ σ=2Å')
    ax.set_xlabel('Nearest Neighbor Distance (Å)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Nearest Neighbor Distance\nDistribution (all configs)')
    ax.legend()

    # Panel 3: Energy components distribution
    ax = axes[2]
    ax.hist(energies_coulomb, bins=20, alpha=0.7, color='purple', density=True, label='Coulomb')
    ax.hist(energies_rep, bins=20, alpha=0.7, color='darkorange', density=True, label='LJ Repulsion')
    ax.set_xlabel('Energy (a.u.)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(c) Energy Component Distribution\n(100 configurations)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig1_rc_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig1_rc_data_overview.png')


def fig_rc_energy_parity(sr_te, les_te, e_true):
    """Figure 2: Energy parity plots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, res, title, color in zip(
        axes, [sr_te, les_te], ['Short-Range Model', 'LES Model'],
        ['steelblue', 'tomato']
    ):
        e_pred = res['energies_pred']
        lim = [min(e_true.min(), e_pred.min()) - 2,
               max(e_true.max(), e_pred.max()) + 2]
        ax.scatter(e_true, e_pred, c=color, s=60, alpha=0.8, edgecolors='k', lw=0.3, zorder=5)
        ax.plot(lim, lim, 'k--', lw=1.5, label='y = x')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('Reference Energy (a.u.)')
        ax.set_ylabel('Predicted Energy (a.u.)')
        ax.set_title(f'({chr(96 + list([sr_te, les_te]).index(res) + 1)}) {title}')
        ax.text(0.05, 0.95, f'MAE = {res["mae"]:.2f}\nR² = {res["r2"]:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.legend()
        ax.set_aspect('equal')

    plt.suptitle('Energy Prediction: Short-Range vs. LES Model (Test Set)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig2_rc_energy_parity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig2_rc_energy_parity.png')


def fig_rc_latent_charges(les_tr, true_charges_train):
    """Figure 3: LES latent charge recovery."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_latent = np.concatenate(les_tr['latent_charges'])
    all_true = np.concatenate(true_charges_train)

    plus_mask = all_true > 0
    minus_mask = all_true < 0

    q_plus = all_latent[plus_mask]
    q_minus = all_latent[minus_mask]

    # Panel 1: Histogram comparison
    ax = axes[0]
    ax.hist(q_plus, bins=50, alpha=0.65, color='crimson', density=True, label=f'True +1e\n(n={plus_mask.sum()})')
    ax.hist(q_minus, bins=50, alpha=0.65, color='royalblue', density=True, label=f'True −1e\n(n={minus_mask.sum()})')
    ax.set_xlabel('LES Latent Charge Value (a.u.)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(a) LES Latent Charge Distribution\nGrouped by True Charge')
    ax.legend()

    mean_plus = np.mean(q_plus)
    mean_minus = np.mean(q_minus)
    std_pooled = np.sqrt(0.5 * (np.var(q_plus) + np.var(q_minus)))
    d_prime = abs(mean_plus - mean_minus) / (std_pooled + 1e-10)

    ax.text(0.05, 0.95, f"Mean (+): {mean_plus:.3f}\nMean (−): {mean_minus:.3f}\nd' = {d_prime:.2f}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: Scatter (subsample for clarity)
    ax = axes[1]
    n_sample = min(500, len(q_plus))
    idx_p = np.random.choice(len(q_plus), n_sample, replace=False)
    idx_m = np.random.choice(len(q_minus), n_sample, replace=False)

    ax.scatter(np.ones(n_sample), q_plus[idx_p], c='crimson', alpha=0.3, s=5)
    ax.scatter(-np.ones(n_sample), q_minus[idx_m], c='royalblue', alpha=0.3, s=5)
    ax.errorbar([1], [mean_plus], yerr=[np.std(q_plus)], fmt='ro', ms=12, capsize=8,
                lw=2, label=f'Mean ± std')
    ax.errorbar([-1], [mean_minus], yerr=[np.std(q_minus)], fmt='bs', ms=12, capsize=8, lw=2)
    ax.axhline(y=0, color='gray', linestyle='--', lw=1)
    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['True −1e', 'True +1e'])
    ax.set_ylabel('LES Latent Charge (a.u.)')
    ax.set_title('(b) Latent Charge per True Charge Group\n(points = individual atoms)')
    ax.legend()

    plt.suptitle('LES Latent Charge Recovery — Trained on Energies Only, No Charge Labels',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig3_rc_latent_charges.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig3_rc_latent_charges.png')


def fig_rc_training_curves(sr_losses, les_losses):
    """Figure 4: Training loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(sr_losses) + 1)
    ax.semilogy(epochs, sr_losses, 'b-', lw=2, label='Short-Range Model')
    ax.semilogy(epochs, les_losses, 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Normalized Energy Loss (log scale)')
    ax.set_title('Training Convergence: SR vs LES\nRandom Charges Benchmark (100 configs, 128 atoms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig4_rc_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig4_rc_training_curves.png')


if __name__ == '__main__':
    main()
