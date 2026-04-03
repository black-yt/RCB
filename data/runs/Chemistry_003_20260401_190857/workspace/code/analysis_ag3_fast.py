"""
Analysis 3 (Fast): Ag3 Charge State Discrimination
"""
import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, pairwise_distances_vectorized
from fast_les_model import FastLESModel, FastSRModel, rbf_expansion
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_003_20260401_190857'
DATA_DIR = f'{BASE}/data'
OUT_DIR = f'{BASE}/outputs'
IMG_DIR = f'{BASE}/report/images'


def compute_ag3_features(positions_list, cutoff=5.0, n_rbf=16):
    """Pre-compute features for Ag3."""
    feats = []
    for pos in positions_list:
        pos_t = torch.tensor(pos, dtype=torch.float32)
        N = 3
        diff = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 1e-6) & (dist < cutoff)
        row_idx = mask.nonzero(as_tuple=True)[0]
        d_flat = dist[mask]
        rbf_flat = rbf_expansion(d_flat, n_rbf=n_rbf, r_max=cutoff)
        f = torch.zeros(N, n_rbf)
        f.scatter_add_(0, row_idx.unsqueeze(-1).expand(-1, n_rbf), rbf_flat)
        feats.append(f.detach())
    return feats


class Ag3SRModel(nn.Module):
    """SR Ag3 model - no charge state info."""
    def __init__(self, n_rbf=16, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, features, charge_state=None):
        return self.net(features).squeeze(-1).sum(), None


class Ag3CSModel(nn.Module):
    """Ag3 model with explicit charge state embedding."""
    def __init__(self, n_rbf=16, n_hidden=32):
        super().__init__()
        self.cs_embed = nn.Embedding(2, 8)  # 2 charge states, 8-dim embedding
        self.net = nn.Sequential(
            nn.Linear(n_rbf + 8, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, features, charge_state=1):
        cs_idx = torch.tensor(0 if charge_state < 0 else 1)
        embed = self.cs_embed(cs_idx).unsqueeze(0).expand(3, -1)
        combined = torch.cat([features, embed], dim=-1)
        return self.net(combined).squeeze(-1).sum(), None


class Ag3LESModel(nn.Module):
    """Ag3 LES model with latent charges."""
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
        self.cutoff = 5.0

    def forward(self, features, positions, charge_state=1):
        """
        features: (3, n_rbf) precomputed
        positions: (3, 3) tensor
        """
        N = 3
        raw_q = self.charge_net(features).squeeze(-1)
        correction = (charge_state - raw_q.sum()) / N
        latent_q = raw_q + correction

        # Coulomb energy
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        r_ij = dist[mask]
        q_outer = latent_q.unsqueeze(1) * latent_q.unsqueeze(0)
        E_lr = (q_outer[mask] / r_ij).sum()

        E_sr = self.sr_net(features).squeeze(-1).sum()
        return E_lr + E_sr, latent_q


def train_ag3(model, features_list, positions_list, energies, forces_list,
              charge_states, n_epochs=500, lr=5e-4, verbose=True):
    """Train Ag3 model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    e_std = np.std(energies) + 1e-8
    losses = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        perm = np.random.permutation(len(features_list))

        for idx in perm:
            feats = features_list[idx]
            pos = torch.tensor(positions_list[idx], dtype=torch.float32, requires_grad=True)
            cs = int(charge_states[idx])
            e_ref = torch.tensor(energies[idx], dtype=torch.float32)

            optimizer.zero_grad()

            if isinstance(model, Ag3LESModel):
                e_pred, q = model(feats, pos, cs)
            else:
                e_pred, q = model(feats, cs)

            # Force loss via autograd if positions require grad
            if isinstance(model, Ag3LESModel):
                # Re-run with grad
                pos_g = torch.tensor(positions_list[idx], dtype=torch.float32, requires_grad=True)
                e_g, _ = model(feats, pos_g, cs)
                f_pred = -torch.autograd.grad(e_g, pos_g, create_graph=False)[0]
                f_ref = torch.tensor(forces_list[idx], dtype=torch.float32)
                f_loss = ((f_pred - f_ref) ** 2).mean()
            else:
                # SR/CS model: use position-dependent forward
                pos_g = torch.tensor(positions_list[idx], dtype=torch.float32, requires_grad=True)
                # Need position-dependent features for grad
                from fast_les_model import compute_features_fast
                feats_g = compute_features_fast(pos_g, cutoff=5.0, n_rbf=feats.shape[-1])
                e_g, _ = model(feats_g, cs)
                f_pred = -torch.autograd.grad(e_g, pos_g, create_graph=False)[0]
                f_ref = torch.tensor(forces_list[idx], dtype=torch.float32)
                f_loss = ((f_pred - f_ref) ** 2).mean()

            e_loss = ((e_pred - e_ref) ** 2) / (e_std ** 2)
            loss = e_loss + 0.2 * f_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(features_list)
        losses.append(avg)
        if verbose and (epoch + 1) % 100 == 0:
            print(f'  Epoch {epoch+1}: loss={avg:.4f}')

    return losses


def evaluate_ag3(model, features_list, positions_list, energies, charge_states):
    """Evaluate Ag3 model."""
    model.eval()
    preds, all_q = [], []
    with torch.no_grad():
        for i in range(len(features_list)):
            pos = torch.tensor(positions_list[i], dtype=torch.float32)
            cs = int(charge_states[i])
            if isinstance(model, Ag3LESModel):
                e, q = model(features_list[i], pos, cs)
                all_q.append(q.numpy())
            else:
                e, q = model(features_list[i], cs)
            preds.append(e.item())

    preds = np.array(preds)
    refs = np.array(energies)
    mae = np.mean(np.abs(preds - refs))
    r2 = 1 - np.mean((preds - refs) ** 2) / (np.var(refs) + 1e-10)
    rmse = np.sqrt(np.mean((preds - refs) ** 2))
    return {'energies_pred': preds, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'latent_charges': all_q if all_q else None}


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Analysis 3: Ag3 Charge State Discrimination")
    print("=" * 60)

    frames = parse_extxyz(f'{DATA_DIR}/ag3_chargestates.xyz')
    positions_list = [fr['positions'] for fr in frames]
    energies = np.array([fr['energy'] for fr in frames])
    forces_list = [fr.get('forces') for fr in frames]
    charge_states = np.array([fr['charge_state'] for fr in frames])

    print(f"  {len(frames)} configs: {(charge_states==1).sum()} with +1, {(charge_states==-1).sum()} with -1")

    # Pre-compute features
    print("\n[2] Pre-computing features...")
    features_list = compute_ag3_features(positions_list, cutoff=5.0, n_rbf=16)

    # Split
    train_i, test_i = train_test_split(np.arange(len(frames)), test_size=0.25,
                                        stratify=charge_states, random_state=42)

    f_tr = [features_list[i] for i in train_i]
    f_te = [features_list[i] for i in test_i]
    p_tr = [positions_list[i] for i in train_i]
    p_te = [positions_list[i] for i in test_i]
    e_tr = energies[train_i]; e_te = energies[test_i]
    cs_tr = charge_states[train_i]; cs_te = charge_states[test_i]
    fo_tr = [forces_list[i] for i in train_i]

    # SR model
    print("\n[3] Training SR model (no charge state)...")
    sr = Ag3SRModel(n_rbf=16, n_hidden=32)
    sr_l = train_ag3(sr, f_tr, p_tr, e_tr, fo_tr, cs_tr, n_epochs=500, verbose=True)
    sr_tr = evaluate_ag3(sr, f_tr, p_tr, e_tr, cs_tr)
    sr_te = evaluate_ag3(sr, f_te, p_te, e_te, cs_te)
    print(f"  SR  Train MAE={sr_tr['mae']:.4f}, R²={sr_tr['r2']:.4f}")
    print(f"  SR  Test  MAE={sr_te['mae']:.4f}, R²={sr_te['r2']:.4f}")

    # Charge state model
    print("\n[4] Training Charge State embedding model...")
    cs_m = Ag3CSModel(n_rbf=16, n_hidden=32)
    cs_l = train_ag3(cs_m, f_tr, p_tr, e_tr, fo_tr, cs_tr, n_epochs=500, verbose=True)
    cs_m_tr = evaluate_ag3(cs_m, f_tr, p_tr, e_tr, cs_tr)
    cs_m_te = evaluate_ag3(cs_m, f_te, p_te, e_te, cs_te)
    print(f"  CS  Train MAE={cs_m_tr['mae']:.4f}, R²={cs_m_tr['r2']:.4f}")
    print(f"  CS  Test  MAE={cs_m_te['mae']:.4f}, R²={cs_m_te['r2']:.4f}")

    # LES model
    print("\n[5] Training LES model...")
    les = Ag3LESModel(n_rbf=16, n_hidden=32)
    les_l = train_ag3(les, f_tr, p_tr, e_tr, fo_tr, cs_tr, n_epochs=500, verbose=True)
    les_tr = evaluate_ag3(les, f_tr, p_tr, e_tr, cs_tr)
    les_te = evaluate_ag3(les, f_te, p_te, e_te, cs_te)
    print(f"  LES Train MAE={les_tr['mae']:.4f}, R²={les_tr['r2']:.4f}")
    print(f"  LES Test  MAE={les_te['mae']:.4f}, R²={les_te['r2']:.4f}")

    # Save
    np.save(f'{OUT_DIR}/ag3_results.npy', {
        'sr_train': sr_tr, 'sr_test': sr_te,
        'cs_train': cs_m_tr, 'cs_test': cs_m_te,
        'les_train': les_tr, 'les_test': les_te,
        'sr_losses': sr_l, 'cs_losses': cs_l, 'les_losses': les_l,
        'test_idx': test_i, 'charge_states_test': cs_te
    }, allow_pickle=True)

    print("\n[6] Generating figures...")
    fig_ag3_overview(frames, energies, charge_states)
    fig_ag3_parity(sr_te, cs_m_te, les_te, e_te, cs_te)
    fig_ag3_latent_charges(les_tr, cs_tr, f_tr)
    fig_ag3_training(sr_l, cs_l, les_l)
    fig_ag3_comparison(sr_te, cs_m_te, les_te, e_te, cs_te)

    print(f"\n[Summary]")
    for name, res in [('SR', sr_te), ('CS', cs_m_te), ('LES', les_te)]:
        print(f"  {name}: MAE={res['mae']:.4f} eV, R²={res['r2']:.4f}")

    return sr_te, cs_m_te, les_te


def fig_ag3_overview(frames, energies, charge_states):
    """Overview figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Ag3 geometry
    ax = axes[0]
    pos = frames[0]['positions']
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'k-', lw=2.5, zorder=3)
    ax.scatter(pos[:,0], pos[:,1], c='silver', s=400, edgecolors='gray', lw=2, zorder=5)
    for i in range(3):
        r = np.linalg.norm(pos[(i+1)%3] - pos[i])
        mid = 0.5*(pos[i] + pos[(i+1)%3])
        ax.text(mid[0], mid[1]+0.1, f'{r:.2f}Å', ha='center', fontsize=8, color='navy')
    ax.set_title('(a) Ag₃ Cluster (first config)')
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    ax.set_aspect('equal')
    m = 1.0
    ax.set_xlim(pos[:,0].min()-m, pos[:,0].max()+m)
    ax.set_ylim(pos[:,1].min()-m, pos[:,1].max()+m)

    # Panel 2: Energy distribution by charge state
    ax = axes[1]
    for cs, color in [(1, 'crimson'), (-1, 'royalblue')]:
        e = energies[charge_states == cs]
        ax.hist(e, bins=12, alpha=0.65, color=color, density=True, label=f'q={cs:+d}e')
    ax.set_xlabel('Energy (eV)'); ax.set_ylabel('Probability Density')
    ax.set_title('(b) Energy Distribution\nby Charge State')
    ax.legend()

    # Panel 3: Scatter plot: energy vs bond lengths
    ax = axes[2]
    from data_utils import pairwise_distances_vectorized
    for cs, color, marker in [(1, 'crimson', '^'), (-1, 'royalblue', 'o')]:
        mask = charge_states == cs
        mean_bonds = []
        for fr in [f for i, f in enumerate(frames) if mask[i]]:
            dm = pairwise_distances_vectorized(fr['positions'])
            bonds = [dm[i,j] for i in range(3) for j in range(i+1,3)]
            mean_bonds.append(np.mean(bonds))
        ax.scatter(mean_bonds, energies[mask], c=color, s=20, alpha=0.6,
                   marker=marker, label=f'q={cs:+d}e')
    ax.set_xlabel('Mean Bond Length (Å)'); ax.set_ylabel('Energy (eV)')
    ax.set_title('(c) Energy vs. Bond Length\n(both charge states)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig8_ag3_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig8_ag3_overview.png')


def fig_ag3_parity(sr_te, cs_te, les_te, e_ref, charge_states):
    """Energy parity plots."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    models = [('SR Model\n(no CS)', sr_te, 'steelblue'),
              ('CS Embed Model', cs_te, 'seagreen'),
              ('LES Model', les_te, 'tomato')]

    for ax, (title, res, color) in zip(axes, models):
        e_pred = res['energies_pred']
        lim = [min(e_ref.min(), e_pred.min())-0.1, max(e_ref.max(), e_pred.max())+0.1]

        for cs, marker, col in [(1, '^', 'red'), (-1, 'o', 'blue')]:
            mask = charge_states == cs
            ax.scatter(e_ref[mask], e_pred[mask], marker=marker, c=col, alpha=0.7,
                       s=50, edgecolors='k', lw=0.3, label=f'q={cs:+d}e')
        ax.plot(lim, lim, 'k--', lw=1.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('Reference Energy (eV)')
        ax.set_ylabel('Predicted Energy (eV)')
        ax.set_title(f'{title}\nMAE={res["mae"]:.4f} eV, R²={res["r2"]:.4f}')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    plt.suptitle('Ag₃ PES Prediction: Model Comparison (Test Set)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig9_ag3_parity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig9_ag3_parity.png')


def fig_ag3_latent_charges(les_tr, charge_states_train, features_list):
    """Latent charge analysis."""
    if les_tr['latent_charges'] is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lqs = les_tr['latent_charges']  # list of (3,) arrays
    cs_arr = np.array(charge_states_train)

    # Panel 1: mean latent charge per atom by CS
    ax = axes[0]
    for cs, color, marker in [(1, 'crimson', '^'), (-1, 'royalblue', 'o')]:
        mask = cs_arr == cs
        qs = np.array([lqs[i] for i in np.where(mask)[0]])  # (n_cs, 3)
        for atom_i in range(3):
            q_m = np.mean(qs[:, atom_i])
            q_s = np.std(qs[:, atom_i])
            ax.errorbar(atom_i + (0.1 if cs == 1 else -0.1), q_m, yerr=q_s,
                       fmt=marker, color=color, ms=10, capsize=6,
                       label=f'q={cs:+d}e' if atom_i == 0 else '')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['Ag1','Ag2','Ag3'])
    ax.set_ylabel('Latent Charge (a.u.)')
    ax.set_title('(a) Per-Atom Latent Charges\nby Charge State')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--', lw=1)

    # Panel 2: distribution
    ax = axes[1]
    for cs, color in [(1, 'crimson'), (-1, 'royalblue')]:
        mask = cs_arr == cs
        total_q = np.array([np.sum(lqs[i]) for i in np.where(mask)[0]])
        ax.hist(total_q, bins=12, alpha=0.65, color=color, density=True,
                label=f'q={cs:+d}e (mean={np.mean(total_q):.3f})')
    ax.set_xlabel('Total Latent Charge (a.u.)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Total LES Latent Charge\nDistribution by Charge State')
    ax.legend()

    plt.suptitle('Ag₃ LES Latent Charge Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig10_ag3_latent_charges.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig10_ag3_latent_charges.png')


def fig_ag3_training(sr_l, cs_l, les_l):
    """Training curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(sr_l)+1)
    ax.semilogy(epochs, sr_l, 'b-', lw=2, label='SR Model')
    ax.semilogy(epochs, cs_l, 'g-', lw=2, label='CS Embedding')
    ax.semilogy(epochs, les_l, 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Ag₃ Training Curves: Three Model Variants')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig11_ag3_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig11_ag3_training.png')


def fig_ag3_comparison(sr_te, cs_te, les_te, e_ref, charge_states):
    """Summary comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['SR Model', 'CS Embed', 'LES Model']
    results = [sr_te, cs_te, les_te]
    colors = ['steelblue', 'seagreen', 'tomato']

    ax = axes[0]
    maes = [r['mae'] for r in results]
    bars = ax.bar(models, maes, color=colors, alpha=0.8, edgecolor='k', lw=0.8)
    ax.set_ylabel('MAE (eV)')
    ax.set_title('(a) Energy MAE Comparison\n(Test Set)')
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1]
    r2s = [r['r2'] for r in results]
    bars = ax.bar(models, r2s, color=colors, alpha=0.8, edgecolor='k', lw=0.8)
    ax.set_ylabel('R² Score')
    ax.set_title('(b) R² Score Comparison\n(Test Set)')
    ax.set_ylim([min(r2s) - 0.05, 1.01])
    for bar, r2 in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Ag₃ Model Comparison Summary', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig12_ag3_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig12_ag3_comparison.png')


if __name__ == '__main__':
    main()
