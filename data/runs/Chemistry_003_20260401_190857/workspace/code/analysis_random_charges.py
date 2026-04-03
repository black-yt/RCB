"""
Analysis 1: Random Charges Benchmark

Tests whether the LES model can recover true atomic charges (+1/-1)
from energy supervision alone, without any charge labels during training.
Reproduces the benchmark from Fig. 1 of the LES paper.

Physics: E = sum_{i<j} q_i*q_j/r_ij + epsilon*(sigma/r_ij)^12
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, compute_total_energy_rc
from les_model import LESModel, ShortRangeModel, train_model, evaluate_model
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_003_20260401_190857'
DATA_DIR = f'{BASE}/data'
OUT_DIR = f'{BASE}/outputs'
IMG_DIR = f'{BASE}/report/images'


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Analysis 1: Random Charges Benchmark")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading random_charges.xyz...")
    frames = parse_extxyz(f'{DATA_DIR}/random_charges.xyz')
    print(f"  Loaded {len(frames)} configurations of {len(frames[0]['positions'])} atoms")
    print(f"  True charges: {np.unique(frames[0]['true_charges'])} (unique values)")

    # 2. Compute reference energies from Coulomb + repulsive LJ
    print("\n[2] Computing reference energies from Coulomb + repulsive LJ...")
    sigma = 2.0  # Angstrom
    epsilon = 1.0

    energies_total = []
    energies_coulomb = []
    energies_rep = []

    for fr in frames:
        e_tot, e_c, e_r = compute_total_energy_rc(
            fr['positions'], fr['true_charges'],
            sigma=sigma, epsilon=epsilon
        )
        energies_total.append(e_tot)
        energies_coulomb.append(e_c)
        energies_rep.append(e_r)

    energies_total = np.array(energies_total)
    energies_coulomb = np.array(energies_coulomb)
    energies_rep = np.array(energies_rep)

    print(f"  Energy range: {energies_total.min():.3f} to {energies_total.max():.3f}")
    print(f"  Coulomb part: {energies_coulomb.min():.3f} to {energies_coulomb.max():.3f}")
    print(f"  Repulsive part: {energies_rep.min():.3f} to {energies_rep.max():.3f}")

    # Save energies
    np.save(f'{OUT_DIR}/rc_energies_total.npy', energies_total)
    np.save(f'{OUT_DIR}/rc_energies_coulomb.npy', energies_coulomb)
    np.save(f'{OUT_DIR}/rc_true_charges.npy',
            np.array([fr['true_charges'] for fr in frames]))

    positions_list = [fr['positions'] for fr in frames]
    true_charges_list = [fr['true_charges'] for fr in frames]

    # 3. Train/test split
    n_train = 80
    n_test = 20
    idx = np.random.permutation(len(frames))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]

    pos_train = [positions_list[i] for i in train_idx]
    pos_test = [positions_list[i] for i in test_idx]
    e_train = energies_total[train_idx]
    e_test = energies_total[test_idx]
    tc_train = [true_charges_list[i] for i in train_idx]

    # 4. Train Short-Range baseline model
    print("\n[3] Training Short-Range (SR) baseline model...")
    sr_model = ShortRangeModel(n_rbf=16, n_hidden=64, cutoff=5.0)
    sr_losses = train_model(sr_model, pos_train, e_train,
                            n_epochs=300, lr=1e-3, verbose=True)

    sr_results_train = evaluate_model(sr_model, pos_train, e_train)
    sr_results_test = evaluate_model(sr_model, pos_test, e_test)
    print(f"  SR Train MAE: {sr_results_train['mae']:.4f}, R²: {sr_results_train['r2']:.4f}")
    print(f"  SR Test  MAE: {sr_results_test['mae']:.4f}, R²: {sr_results_test['r2']:.4f}")

    # 5. Train LES model
    print("\n[4] Training LES model (latent charges + Coulomb)...")
    les_model = LESModel(n_rbf=16, n_hidden=64, cutoff=5.0)
    les_losses = train_model(les_model, pos_train, e_train,
                             n_epochs=300, lr=1e-3, verbose=True)

    les_results_train = evaluate_model(les_model, pos_train, e_train, return_charges=True)
    les_results_test = evaluate_model(les_model, pos_test, e_test, return_charges=True)
    print(f"  LES Train MAE: {les_results_train['mae']:.4f}, R²: {les_results_train['r2']:.4f}")
    print(f"  LES Test  MAE: {les_results_test['mae']:.4f}, R²: {les_results_test['r2']:.4f}")

    # 6. Save results
    results = {
        'sr_train': sr_results_train,
        'sr_test': sr_results_test,
        'les_train': les_results_train,
        'les_test': les_results_test,
        'sr_losses': sr_losses,
        'les_losses': les_losses,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'energies_total': energies_total,
        'energies_coulomb': energies_coulomb,
    }
    np.save(f'{OUT_DIR}/rc_results.npy', results, allow_pickle=True)

    # 7. Generate figures
    print("\n[5] Generating figures...")
    fig_data_overview(frames, energies_coulomb, energies_rep)
    fig_energy_correlation(sr_results_test, les_results_test, e_test)
    fig_latent_charges(les_results_train, true_charges_list, train_idx)
    fig_training_curves(sr_losses, les_losses)

    print(f"\n[6] Summary:")
    print(f"  SR  model: MAE={sr_results_test['mae']:.4f}, R²={sr_results_test['r2']:.4f}")
    print(f"  LES model: MAE={les_results_test['mae']:.4f}, R²={les_results_test['r2']:.4f}")
    print(f"  Improvement: {(1 - les_results_test['mae']/sr_results_test['mae'])*100:.1f}% MAE reduction")

    return results


def fig_data_overview(frames, energies_coulomb, energies_rep):
    """Figure: Overview of random charges dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: 3D structure of first config
    ax = axes[0]
    pos = frames[0]['positions']
    charges = frames[0]['true_charges']
    pos_plus = pos[charges > 0]
    pos_minus = pos[charges < 0]
    ax.scatter(pos_plus[:, 0], pos_plus[:, 1], c='red', alpha=0.5, s=20, label='+1e')
    ax.scatter(pos_minus[:, 0], pos_minus[:, 1], c='blue', alpha=0.5, s=20, label='−1e')
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_title('(a) Random Charges Structure\n(x-y projection, config 1)')
    ax.legend(framealpha=0.8)
    ax.set_aspect('equal')

    # Panel 2: Pairwise distance distribution
    ax = axes[1]
    from data_utils import pairwise_distances_vectorized
    pos = frames[0]['positions']
    dist_mat = pairwise_distances_vectorized(pos)
    # Upper triangle
    mask = np.triu(np.ones_like(dist_mat, dtype=bool), k=1)
    dists = dist_mat[mask]
    charges = frames[0]['true_charges']
    q_i = charges[:, np.newaxis]
    q_j = charges[np.newaxis, :]
    qq = (q_i * q_j)[mask]

    ax.hist(dists[qq > 0], bins=40, alpha=0.6, color='red', density=True, label='Same charge (+)')
    ax.hist(dists[qq < 0], bins=40, alpha=0.6, color='blue', density=True, label='Opp. charge (−)')
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('Probability density')
    ax.set_title('(b) Pairwise Distance Distribution\n(by charge product)')
    ax.legend()

    # Panel 3: Energy distribution
    ax = axes[2]
    ax.hist(energies_coulomb, bins=20, alpha=0.7, color='purple', label='Coulomb')
    ax.hist(energies_rep, bins=20, alpha=0.7, color='orange', label='Repulsive LJ')
    ax.set_xlabel('Energy (a.u.)')
    ax.set_ylabel('Count')
    ax.set_title('(c) Energy Distribution\nAcross 100 Configurations')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig1_rc_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_rc_data_overview.png")


def fig_energy_correlation(sr_results, les_results, e_true):
    """Figure: Predicted vs. true energy correlation for SR and LES models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, results, title, color in zip(
        axes,
        [sr_results, les_results],
        ['Short-Range Model', 'LES Model'],
        ['steelblue', 'tomato']
    ):
        e_pred = results['energies_pred']
        mae = results['mae']
        r2 = results['r2']

        lim = [min(e_true.min(), e_pred.min()) - 0.5,
               max(e_true.max(), e_pred.max()) + 0.5]

        ax.scatter(e_true, e_pred, alpha=0.7, color=color, s=40, edgecolors='k', lw=0.3)
        ax.plot(lim, lim, 'k--', lw=1.5, label='y = x')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('Reference Energy (a.u.)')
        ax.set_ylabel('Predicted Energy (a.u.)')
        ax.set_title(f'{title}\nMAE={mae:.4f}, R²={r2:.4f}')
        ax.legend()
        ax.set_aspect('equal')

    plt.suptitle('Energy Prediction: SR vs LES on Random Charges Dataset', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig2_rc_energy_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_rc_energy_correlation.png")


def fig_latent_charges(les_results, true_charges_list, train_idx):
    """Figure: Correlation between LES latent charges and true charges."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    latent_charges = les_results['latent_charges']  # list of (N,) arrays
    true_charges = [true_charges_list[i] for i in train_idx]

    # Flatten all charges
    all_latent = np.concatenate(latent_charges)
    all_true = np.concatenate(true_charges)

    # Panel 1: Scatter of all latent vs true charges
    ax = axes[0]
    plus_mask = all_true > 0
    minus_mask = all_true < 0

    ax.scatter(all_true[plus_mask], all_latent[plus_mask],
               alpha=0.1, c='red', s=5, label='+1e atoms')
    ax.scatter(all_true[minus_mask], all_latent[minus_mask],
               alpha=0.1, c='blue', s=5, label='−1e atoms')

    # Reference line
    ax.axhline(y=np.mean(all_latent[plus_mask]), color='red',
               linestyle='--', lw=2, label=f'Mean +: {np.mean(all_latent[plus_mask]):.3f}')
    ax.axhline(y=np.mean(all_latent[minus_mask]), color='blue',
               linestyle='--', lw=2, label=f'Mean −: {np.mean(all_latent[minus_mask]):.3f}')

    ax.set_xlabel('True Charge (e)')
    ax.set_ylabel('Latent Charge (a.u.)')
    ax.set_title('(a) Latent vs. True Charges\n(all training atoms)')
    ax.legend(fontsize=9, framealpha=0.8)

    # Panel 2: Histogram of latent charges for each true charge value
    ax = axes[1]
    ax.hist(all_latent[plus_mask], bins=40, alpha=0.6, color='red',
            density=True, label='+1e true charge')
    ax.hist(all_latent[minus_mask], bins=40, alpha=0.6, color='blue',
            density=True, label='−1e true charge')

    ax.set_xlabel('Latent Charge Value (a.u.)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Distribution of LES Latent Charges\nby True Charge Assignment')
    ax.legend()

    # Compute discrimination statistics
    mean_plus = np.mean(all_latent[plus_mask])
    mean_minus = np.mean(all_latent[minus_mask])
    std_plus = np.std(all_latent[plus_mask])
    std_minus = np.std(all_latent[minus_mask])
    separation = abs(mean_plus - mean_minus) / (0.5 * (std_plus + std_minus))

    ax.text(0.05, 0.95, f'Separation: {separation:.2f}σ\n'
            f'Mean +: {mean_plus:.3f}\nMean −: {mean_minus:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('LES Latent Charge Recovery (Trained on Energies Only)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig3_rc_latent_charges.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_rc_latent_charges.png")


def fig_training_curves(sr_losses, les_losses):
    """Figure: Training loss curves for SR and LES models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sr_e = [l['energy'] for l in sr_losses]
    les_e = [l['energy'] for l in les_losses]

    epochs = np.arange(1, len(sr_e) + 1)
    ax.semilogy(epochs, sr_e, 'b-', lw=2, label='Short-Range Model')
    ax.semilogy(epochs, les_e, 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Normalized Energy Loss (log scale)')
    ax.set_title('Training Convergence: SR vs LES\nRandom Charges Benchmark')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig4_rc_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_rc_training_curves.png")


if __name__ == '__main__':
    results = main()
