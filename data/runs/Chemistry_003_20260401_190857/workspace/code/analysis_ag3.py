"""
Analysis 3: Ag3 Charge State Discrimination

Tests whether a model can distinguish PES for Ag3 in different charge
states (+1 and -1) with and without charge state embedding.
Reproduces the analysis from Fig. 5e and Table 1 of the LES paper.

Dataset: Ag3 trimers in charge states +1 and -1.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, pairwise_distances_vectorized
from les_model import LESModel, ShortRangeModel, train_model, evaluate_model, AtomicNetwork, RadialBasisLayer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_003_20260401_190857'
DATA_DIR = f'{BASE}/data'
OUT_DIR = f'{BASE}/outputs'
IMG_DIR = f'{BASE}/report/images'


class Ag3SRModel(nn.Module):
    """SR model for Ag3: ignores charge state."""

    def __init__(self, n_rbf=16, n_hidden=32, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RadialBasisLayer(n_rbf=n_rbf, r_max=cutoff)
        self.atomic_net = AtomicNetwork(n_rbf, n_hidden=n_hidden, n_output=1)

    def compute_features(self, positions):
        n = positions.shape[0]
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 0) & (dist < self.cutoff)
        features = torch.zeros(n, self.rbf.centers.shape[0], device=positions.device)
        for i in range(n):
            nbr_dists = dist[i][mask[i]]
            if nbr_dists.numel() > 0:
                features[i] = self.rbf(nbr_dists).sum(dim=0)
        return features

    def forward(self, positions, charge_state=None):
        features = self.compute_features(positions)
        return self.atomic_net(features).squeeze(-1).sum()


class Ag3ChargeStateModel(nn.Module):
    """SR model for Ag3 WITH charge state embedding."""

    def __init__(self, n_rbf=16, n_hidden=32, cutoff=5.0, n_charge_states=2):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RadialBasisLayer(n_rbf=n_rbf, r_max=cutoff)
        # Charge state is encoded as a learned embedding
        self.charge_embedding = nn.Embedding(n_charge_states, n_hidden)
        # Network takes local features + charge state embedding
        self.atomic_net = AtomicNetwork(n_rbf + n_hidden, n_hidden=n_hidden, n_output=1)

    def compute_features(self, positions):
        n = positions.shape[0]
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 0) & (dist < self.cutoff)
        features = torch.zeros(n, self.rbf.centers.shape[0], device=positions.device)
        for i in range(n):
            nbr_dists = dist[i][mask[i]]
            if nbr_dists.numel() > 0:
                features[i] = self.rbf(nbr_dists).sum(dim=0)
        return features

    def forward(self, positions, charge_state=1):
        n = positions.shape[0]
        features = self.compute_features(positions)
        # Map charge state (-1, +1) to index (0, 1)
        cs_idx = torch.tensor(0 if charge_state < 0 else 1, device=positions.device)
        cs_embed = self.charge_embedding(cs_idx).unsqueeze(0).expand(n, -1)  # (N, n_hidden)
        combined = torch.cat([features, cs_embed], dim=-1)  # (N, n_rbf + n_hidden)
        return self.atomic_net(combined).squeeze(-1).sum()


class Ag3LESModel(nn.Module):
    """LES model for Ag3: predicts latent charges using total_charge constraint."""

    def __init__(self, n_rbf=16, n_hidden=32, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RadialBasisLayer(n_rbf=n_rbf, r_max=cutoff)
        self.charge_net = AtomicNetwork(n_rbf, n_hidden=n_hidden, n_output=1)
        self.energy_net = AtomicNetwork(n_rbf, n_hidden=n_hidden, n_output=1)

    def compute_features(self, positions):
        n = positions.shape[0]
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 0) & (dist < self.cutoff)
        features = torch.zeros(n, self.rbf.centers.shape[0], device=positions.device)
        for i in range(n):
            nbr_dists = dist[i][mask[i]]
            if nbr_dists.numel() > 0:
                features[i] = self.rbf(nbr_dists).sum(dim=0)
        return features

    def forward(self, positions, charge_state=1):
        """Forward pass with total charge constraint."""
        n = positions.shape[0]
        features = self.compute_features(positions)

        # Predict latent charges constrained to sum to total_charge
        raw_q = self.charge_net(features).squeeze(-1)
        correction = (charge_state - raw_q.sum()) / n
        latent_q = raw_q + correction

        # Coulomb energy
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        r_ij = dist[mask]
        q_i = latent_q.unsqueeze(1).expand(n, n)
        q_j = latent_q.unsqueeze(0).expand(n, n)
        E_lr = ((q_i * q_j)[mask] / r_ij).sum()

        # Short-range energy
        E_sr = self.energy_net(features).squeeze(-1).sum()

        return E_lr + E_sr, latent_q


def train_ag3_model(model, frames, n_epochs=500, lr=1e-3, verbose=True):
    """Train Ag3 model on energy + force data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    energies = np.array([f['energy'] for f in frames])
    e_std = energies.std() + 1e-8

    losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for fr in frames:
            pos = torch.tensor(fr['positions'], dtype=torch.float32, requires_grad=True)
            cs = fr['charge_state']

            optimizer.zero_grad()

            if isinstance(model, Ag3LESModel):
                e_pred, _ = model(pos, charge_state=cs)
            else:
                e_pred = model(pos, charge_state=cs)

            e_ref = torch.tensor(fr['energy'], dtype=torch.float32)
            e_loss = ((e_pred - e_ref) ** 2) / (e_std ** 2)

            # Force loss
            forces_pred = -torch.autograd.grad(e_pred, pos, create_graph=True)[0]
            f_ref = torch.tensor(fr['forces'], dtype=torch.float32)
            f_loss = ((forces_pred - f_ref) ** 2).mean()

            loss = e_loss + 0.5 * f_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(frames)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f'  Epoch {epoch+1:4d}: loss={avg_loss:.4f}')

    return losses


def evaluate_ag3(model, frames):
    """Evaluate Ag3 model predictions."""
    model.eval()
    energies_ref = np.array([f['energy'] for f in frames])
    energies_pred = []
    charges_pred = []
    charge_states = np.array([f['charge_state'] for f in frames])

    with torch.no_grad():
        for fr in frames:
            pos = torch.tensor(fr['positions'], dtype=torch.float32)
            cs = fr['charge_state']
            if isinstance(model, Ag3LESModel):
                e, q = model(pos, charge_state=cs)
                charges_pred.append(q.numpy())
            else:
                e = model(pos, charge_state=cs)
            energies_pred.append(e.item())

    energies_pred = np.array(energies_pred)
    mae = np.mean(np.abs(energies_pred - energies_ref))
    rmse = np.sqrt(np.mean((energies_pred - energies_ref) ** 2))
    r2 = 1 - np.sum((energies_pred - energies_ref) ** 2) / np.sum((energies_ref - energies_ref.mean()) ** 2)

    # Separate by charge state
    metrics = {}
    for cs in sorted(set(charge_states)):
        mask = charge_states == cs
        ep = energies_pred[mask]
        er = energies_ref[mask]
        metrics[cs] = {
            'mae': np.mean(np.abs(ep - er)),
            'r2': 1 - np.sum((ep - er) ** 2) / np.sum((er - er.mean()) ** 2)
        }

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'energies_pred': energies_pred,
        'charge_state_metrics': metrics,
        'latent_charges': charges_pred if charges_pred else None
    }


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Analysis 3: Ag3 Charge State Discrimination")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading ag3_chargestates.xyz...")
    frames = parse_extxyz(f'{DATA_DIR}/ag3_chargestates.xyz')
    print(f"  Loaded {len(frames)} configurations of {len(frames[0]['positions'])} atoms")
    states = [f['charge_state'] for f in frames]
    for cs in sorted(set(states)):
        print(f"  Charge state {cs:+d}: {states.count(cs)} configs")

    # 2. Compute bond lengths
    print("\n[2] Structural analysis...")
    bond_lengths_by_state = {cs: [] for cs in sorted(set(states))}
    for fr in frames:
        pos = fr['positions']
        dm = pairwise_distances_vectorized(pos)
        bonds = [dm[i, j] for i in range(3) for j in range(i+1, 3)]
        bond_lengths_by_state[fr['charge_state']].extend(bonds)

    for cs, bls in bond_lengths_by_state.items():
        print(f"  State {cs:+d}: bond length {np.mean(bls):.3f} ± {np.std(bls):.3f} Å")

    energies = np.array([f['energy'] for f in frames])
    cs_arr = np.array([f['charge_state'] for f in frames])
    for cs in sorted(set(states)):
        e = energies[cs_arr == cs]
        print(f"  State {cs:+d}: energy {e.min():.4f} to {e.max():.4f}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    train_frames, test_frames = train_test_split(frames, test_size=0.2,
                                                  stratify=cs_arr, random_state=42)

    # 3. Train SR model (no charge state)
    print("\n[3] Training SR model (ignores charge state)...")
    sr_model = Ag3SRModel(n_rbf=16, n_hidden=32, cutoff=5.0)
    sr_losses = train_ag3_model(sr_model, train_frames, n_epochs=600, lr=1e-3, verbose=True)
    sr_results_train = evaluate_ag3(sr_model, train_frames)
    sr_results_test = evaluate_ag3(sr_model, test_frames)
    print(f"  SR  Train: MAE={sr_results_train['mae']:.4f}, R²={sr_results_train['r2']:.4f}")
    print(f"  SR  Test:  MAE={sr_results_test['mae']:.4f}, R²={sr_results_test['r2']:.4f}")

    # 4. Train Charge State model (uses explicit charge state)
    print("\n[4] Training Charge State model...")
    cs_model = Ag3ChargeStateModel(n_rbf=16, n_hidden=32, cutoff=5.0)
    cs_losses = train_ag3_model(cs_model, train_frames, n_epochs=600, lr=1e-3, verbose=True)
    cs_results_train = evaluate_ag3(cs_model, train_frames)
    cs_results_test = evaluate_ag3(cs_model, test_frames)
    print(f"  CS  Train: MAE={cs_results_train['mae']:.4f}, R²={cs_results_train['r2']:.4f}")
    print(f"  CS  Test:  MAE={cs_results_test['mae']:.4f}, R²={cs_results_test['r2']:.4f}")

    # 5. Train LES model
    print("\n[5] Training LES model (latent charges)...")
    les_model = Ag3LESModel(n_rbf=16, n_hidden=32, cutoff=5.0)
    les_losses = train_ag3_model(les_model, train_frames, n_epochs=600, lr=1e-3, verbose=True)
    les_results_train = evaluate_ag3(les_model, train_frames)
    les_results_test = evaluate_ag3(les_model, test_frames)
    print(f"  LES Train: MAE={les_results_train['mae']:.4f}, R²={les_results_train['r2']:.4f}")
    print(f"  LES Test:  MAE={les_results_test['mae']:.4f}, R²={les_results_test['r2']:.4f}")

    # 6. Generate figures
    print("\n[6] Generating figures...")
    fig_ag3_overview(frames, energies, cs_arr, bond_lengths_by_state)
    fig_ag3_pes_comparison(frames, sr_results_test, cs_results_test, les_results_test, test_frames)
    fig_ag3_charge_states_comparison(sr_results_train, les_results_train, train_frames)
    fig_ag3_summary_table(sr_results_test, cs_results_test, les_results_test)

    # Save results
    np.save(f'{OUT_DIR}/ag3_results.npy', {
        'sr': sr_results_test, 'cs': cs_results_test, 'les': les_results_test,
        'sr_losses': sr_losses, 'cs_losses': cs_losses, 'les_losses': les_losses
    }, allow_pickle=True)

    return sr_results_test, cs_results_test, les_results_test


def fig_ag3_overview(frames, energies, cs_arr, bond_lengths_by_state):
    """Figure: Overview of Ag3 dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Ag3 geometry
    ax = axes[0]
    pos = frames[0]['positions']
    ax.scatter(pos[:, 0], pos[:, 1], c=['silver', 'silver', 'silver'],
               s=300, edgecolors='gray', lw=2, zorder=5)
    for i, p in enumerate(pos):
        ax.text(p[0], p[1] + 0.15, f'Ag{i+1}', ha='center', va='bottom', fontsize=10)
    # Draw bonds
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    'k-', lw=2, zorder=3)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_title('(a) Ag₃ Cluster Geometry\n(one configuration)')
    ax.set_aspect('equal')
    margin = 1.0
    ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
    ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)

    # Panel 2: Energy distributions by charge state
    ax = axes[1]
    for cs, color in [(1, 'red'), (-1, 'blue')]:
        mask = cs_arr == cs
        ax.hist(energies[mask], bins=15, alpha=0.6, color=color, density=True,
                label=f'q = {cs:+d}e')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Energy Distributions\nby Charge State')
    ax.legend()

    # Panel 3: Bond length distributions
    ax = axes[2]
    for cs, color in [(1, 'red'), (-1, 'blue')]:
        bls = bond_lengths_by_state[cs]
        ax.hist(bls, bins=20, alpha=0.6, color=color, density=True,
                label=f'q = {cs:+d}e')
    ax.set_xlabel('Bond Length (Å)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(c) Bond Length Distributions\nby Charge State')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig8_ag3_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig8_ag3_overview.png")


def fig_ag3_pes_comparison(frames_all, sr_results, cs_results, les_results, test_frames):
    """Figure: PES comparison - predicted vs true energies."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    cs_arr = np.array([f['charge_state'] for f in test_frames])
    e_ref = np.array([f['energy'] for f in test_frames])

    model_results = [
        (sr_results, 'SR Model\n(No charge state)', 'steelblue'),
        (cs_results, 'Charge State Model\n(Explicit CS embed)', 'green'),
        (les_results, 'LES Model\n(Latent charges)', 'tomato'),
    ]

    for ax, (results, title, color) in zip(axes, model_results):
        e_pred = results['energies_pred']
        mae = results['mae']
        r2 = results['r2']

        lim = [min(e_ref.min(), e_pred.min()) - 0.1,
               max(e_ref.max(), e_pred.max()) + 0.1]

        for cs, marker, col in [(1, '^', 'red'), (-1, 'o', 'blue')]:
            mask = cs_arr == cs
            ax.scatter(e_ref[mask], e_pred[mask], alpha=0.7, s=50,
                       marker=marker, color=col, edgecolors='k', lw=0.3,
                       label=f'q={cs:+d}e')

        ax.plot(lim, lim, 'k--', lw=1.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('Reference Energy (eV)')
        ax.set_ylabel('Predicted Energy (eV)')
        ax.set_title(f'{title}\nMAE={mae:.4f} eV, R²={r2:.4f}')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    plt.suptitle('Ag₃ PES Prediction: Model Comparison (Test Set)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig9_ag3_pes_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig9_ag3_pes_comparison.png")


def fig_ag3_charge_states_comparison(sr_results, les_results, train_frames):
    """Figure: Show how LES latent charges differ between charge states."""
    if les_results['latent_charges'] is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cs_arr = np.array([f['charge_state'] for f in train_frames])
    latent_charges = les_results['latent_charges']

    # Panel 1: Mean latent charge per atom by charge state
    ax = axes[0]
    for cs, color, marker in [(1, 'red', '^'), (-1, 'blue', 'o')]:
        mask = cs_arr == cs
        qs = np.array([latent_charges[i] for i in np.where(mask)[0]])  # (N_cs, 3)
        for atom_idx in range(3):
            q_mean = np.mean(qs[:, atom_idx])
            q_std = np.std(qs[:, atom_idx])
            ax.errorbar(atom_idx + (0.1 if cs == 1 else -0.1),
                       q_mean, yerr=q_std,
                       fmt=marker, color=color, ms=8, capsize=5,
                       label=f'q={cs:+d}e' if atom_idx == 0 else '')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Ag1', 'Ag2', 'Ag3'])
    ax.set_ylabel('Latent Charge (a.u.)')
    ax.set_title('(a) Mean Latent Charges by Atom\nSeparated by Charge State')
    ax.axhline(y=0, color='gray', linestyle='--', lw=1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Distribution of total predicted charge
    ax = axes[1]
    for cs, color in [(1, 'red'), (-1, 'blue')]:
        mask = cs_arr == cs
        total_qs = np.array([np.sum(latent_charges[i]) for i in np.where(mask)[0]])
        ax.hist(total_qs, bins=15, alpha=0.6, color=color, density=True,
                label=f'q={cs:+d}e (mean={np.mean(total_qs):.3f})')

    ax.set_xlabel('Sum of Latent Charges (a.u.)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Total LES Latent Charge\nby Charge State')
    ax.legend()

    plt.suptitle('Ag₃ LES Latent Charge Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig10_ag3_latent_charges.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig10_ag3_latent_charges.png")


def fig_ag3_summary_table(sr_results, cs_results, les_results):
    """Figure: Summary table of Ag3 model comparison."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    models = ['Short-Range\n(no CS info)', 'Charge State\nEmbedding', 'LES\n(latent charges)']
    results_list = [sr_results, cs_results, les_results]

    table_data = []
    for name, res in zip(models, results_list):
        row = [name, f'{res["mae"]:.4f}', f'{res["rmse"]:.4f}', f'{res["r2"]:.4f}']
        for cs in [1, -1]:
            if cs in res.get('charge_state_metrics', {}):
                m = res['charge_state_metrics'][cs]
                row.append(f'{m["mae"]:.4f}')
            else:
                row.append('N/A')
        table_data.append(row)

    col_labels = ['Model', 'MAE (eV)', 'RMSE (eV)', 'R²',
                  'MAE q=+1', 'MAE q=−1']

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2.0)

    # Color headers
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color LES row
    for j in range(len(col_labels)):
        table[3, j].set_facecolor('#FFE0E0')

    ax.set_title('Ag₃ Model Comparison (Test Set)', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig11_ag3_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig11_ag3_summary_table.png")


if __name__ == '__main__':
    main()
