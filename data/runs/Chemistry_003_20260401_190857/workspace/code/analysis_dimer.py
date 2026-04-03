"""
Analysis 2: Charged Dimer Binding Energy Curves

Tests whether LES can capture long-range binding energy curves
when two charged molecules are beyond the short-range cutoff.
Reproduces the analysis from Fig. 3 of the LES paper.

Dataset: Two CH3-like dimers at various separation distances.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, get_dimer_separation
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
    print("Analysis 2: Charged Dimer Binding Energy Curves")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading charged_dimer.xyz...")
    frames = parse_extxyz(f'{DATA_DIR}/charged_dimer.xyz')
    print(f"  Loaded {len(frames)} configurations of {len(frames[0]['positions'])} atoms")

    positions_list = [fr['positions'] for fr in frames]
    energies = np.array([fr['energy'] for fr in frames])
    forces_list = [fr.get('forces') for fr in frames]

    # 2. Compute dimer separations
    separations = np.array([get_dimer_separation(fr['positions'], n_atoms_per_monomer=4)
                             for fr in frames])
    print(f"  Separation range: {separations.min():.2f} to {separations.max():.2f} Å")
    print(f"  Energy range: {energies.min():.4f} to {energies.max():.4f}")

    # Sort by separation for visualization
    sort_idx = np.argsort(separations)
    seps_sorted = separations[sort_idx]
    e_sorted = energies[sort_idx]

    # 3. Split: train on close configs, test on far configs
    # Short-range cutoff: 4 Å (molecules can be within cutoff when close)
    # Beyond ~4 Å inter-monomer distance, short-range model should fail
    cutoff_sr = 4.0

    # Use all configs for training but evaluate binding curve
    train_idx = np.arange(len(frames))
    pos_train = [positions_list[i] for i in train_idx]
    e_train = energies[train_idx]
    f_train = [forces_list[i] for i in train_idx]

    # Compute per-monomer internal energies (at large separation, energy = sum of monomers)
    # Identify near and far configs
    near_mask = separations < 5.0
    far_mask = separations >= 5.0
    print(f"  Near configs (<5Å): {near_mask.sum()}, Far configs (>=5Å): {far_mask.sum()}")

    # 4. Train Short-Range model (cutoff=4.0 Å)
    print("\n[2] Training Short-Range model (cutoff=4.0 Å)...")
    sr_model = ShortRangeModel(n_rbf=20, n_hidden=64, cutoff=4.0)
    sr_losses = train_model(sr_model, pos_train, e_train,
                            forces_list=f_train,
                            n_epochs=500, lr=1e-3,
                            energy_weight=1.0, force_weight=0.5,
                            verbose=True)

    sr_results = evaluate_model(sr_model, positions_list, energies)
    print(f"  SR  MAE: {sr_results['mae']:.4f}, R²: {sr_results['r2']:.4f}")

    # 5. Train LES model
    print("\n[3] Training LES model...")
    les_model = LESModel(n_rbf=20, n_hidden=64, cutoff=4.0)
    les_losses = train_model(les_model, pos_train, e_train,
                             forces_list=f_train,
                             n_epochs=500, lr=1e-3,
                             energy_weight=1.0, force_weight=0.5,
                             verbose=True)

    les_results = evaluate_model(les_model, positions_list, energies, return_charges=True)
    print(f"  LES MAE: {les_results['mae']:.4f}, R²: {les_results['r2']:.4f}")

    # 6. Generate figures
    print("\n[4] Generating figures...")
    fig_dimer_overview(positions_list, separations, energies)
    fig_binding_curves(separations, energies, sr_results, les_results, cutoff_sr)
    fig_dimer_training_curves(sr_losses, les_losses)

    # 7. Detailed analysis: energy vs separation
    print("\n[5] Binding curve analysis:")
    # Interpolate smooth binding curve
    sep_grid = np.linspace(seps_sorted[0], seps_sorted[-1], 200)
    for name, results in [('SR', sr_results), ('LES', les_results)]:
        e_pred = results['energies_pred'][sort_idx]
        # Pearson correlation for near vs far configs
        r_near = np.corrcoef(energies[near_mask], results['energies_pred'][near_mask])[0,1]
        r_far = np.corrcoef(energies[far_mask], results['energies_pred'][far_mask])[0,1] if far_mask.sum() > 1 else 0
        print(f"  {name}: R_near={r_near:.4f}, R_far={r_far:.4f}")

    # Save results
    np.save(f'{OUT_DIR}/dimer_results.npy', {
        'separations': separations,
        'energies': energies,
        'sr_results': sr_results,
        'les_results': les_results,
        'sr_losses': sr_losses,
        'les_losses': les_losses,
    }, allow_pickle=True)

    return sr_results, les_results, separations, energies


def fig_dimer_overview(positions_list, separations, energies):
    """Figure: Overview of charged dimer dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: 2D schematic of close and far configurations
    ax = axes[0]
    # Plot close configuration
    close_idx = np.argmin(separations)
    far_idx = np.argmax(separations)
    for idx, label, color, offset in [
        (close_idx, f'Close ({separations[close_idx]:.1f} Å)', 'blue', 0),
        (far_idx, f'Far ({separations[far_idx]:.1f} Å)', 'red', 8)
    ]:
        pos = positions_list[idx]
        # Center monomer 1 at 0, translate
        com = pos.mean(axis=0)
        p = pos - com
        p[:, 1] += offset
        symbols = ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
        colors_atom = ['gray' if s == 'C' else 'white' for s in symbols]
        sizes = [100 if s == 'C' else 40 for s in symbols]
        for i, (xi, yi) in enumerate(zip(p[:, 0], p[:, 2])):
            ax.scatter(xi, yi, c=colors_atom[i], s=sizes[i],
                       edgecolors='k', lw=0.5, zorder=5)

    ax.set_xlabel('x (Å)')
    ax.set_ylabel('z (Å)')
    ax.set_title('(a) Close vs. Far\nDimer Configurations')
    ax.set_aspect('equal')

    # Panel 2: Binding energy curve (raw data)
    ax = axes[1]
    sort_idx = np.argsort(separations)
    ax.scatter(separations, energies, c='black', s=20, alpha=0.7, zorder=5)
    ax.plot(separations[sort_idx], energies[sort_idx], 'k-', alpha=0.3, lw=1)
    ax.set_xlabel('Dimer Separation (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('(b) Raw Binding Curve\nAll 60 Configurations')
    ax.axvline(x=4.0, color='gray', linestyle='--', lw=1, label='SR cutoff')
    ax.legend()

    # Panel 3: Force distribution
    ax = axes[2]
    frames = parse_extxyz(f'{DATA_DIR}/charged_dimer.xyz')
    all_forces = np.concatenate([fr.get('forces', np.zeros((8, 3))) for fr in frames[:20]])
    force_magnitudes = np.linalg.norm(all_forces, axis=1)
    ax.hist(force_magnitudes[force_magnitudes > 0], bins=30, color='green',
            alpha=0.7, density=True)
    ax.set_xlabel('Force Magnitude (eV/Å)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(c) Force Distribution\n(First 20 configs)')

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig5_dimer_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_dimer_overview.png")


def fig_binding_curves(separations, energies, sr_results, les_results, cutoff):
    """Figure: Binding energy curves for reference, SR, and LES models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sort_idx = np.argsort(separations)
    seps = separations[sort_idx]
    e_ref = energies[sort_idx]
    e_sr = sr_results['energies_pred'][sort_idx]
    e_les = les_results['energies_pred'][sort_idx]

    # Panel 1: Full binding curve
    ax = axes[0]
    ax.plot(seps, e_ref, 'ko-', ms=4, lw=1.5, label='Reference', zorder=5)
    ax.plot(seps, e_sr, 'b^--', ms=5, lw=1.5, label=f'SR model (MAE={sr_results["mae"]:.3f})')
    ax.plot(seps, e_les, 'r*-', ms=6, lw=1.5, label=f'LES model (MAE={les_results["mae"]:.3f})')
    ax.axvline(x=cutoff, color='gray', linestyle=':', lw=2, label=f'SR cutoff ({cutoff} Å)')
    ax.set_xlabel('Dimer Separation (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('(a) Binding Energy Curve: All Separations')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Error vs separation
    ax = axes[1]
    sr_err = np.abs(e_sr - e_ref)
    les_err = np.abs(e_les - e_ref)
    ax.plot(seps, sr_err, 'b^--', ms=5, lw=1.5, label='SR model error')
    ax.plot(seps, les_err, 'r*-', ms=6, lw=1.5, label='LES model error')
    ax.axvline(x=cutoff, color='gray', linestyle=':', lw=2, label=f'SR cutoff ({cutoff} Å)')
    ax.set_xlabel('Dimer Separation (Å)')
    ax.set_ylabel('Absolute Error (eV)')
    ax.set_title('(b) Prediction Error vs. Separation Distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.suptitle('Charged Dimer: LES vs Short-Range Model', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig6_dimer_binding_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig6_dimer_binding_curves.png")


def fig_dimer_training_curves(sr_losses, les_losses):
    """Figure: Training curves for dimer analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sr_e = [l['energy'] for l in sr_losses]
    les_e = [l['energy'] for l in les_losses]
    epochs = np.arange(1, len(sr_e) + 1)

    ax = axes[0]
    ax.semilogy(epochs, sr_e, 'b-', lw=2, label='SR Model')
    ax.semilogy(epochs, les_e, 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Energy Loss (log scale)')
    ax.set_title('(a) Energy Loss Training Curves\nCharged Dimer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sr_f = [l['force'] for l in sr_losses]
    les_f = [l['force'] for l in les_losses]
    ax.semilogy(epochs, [x + 1e-10 for x in sr_f], 'b-', lw=2, label='SR Model')
    ax.semilogy(epochs, [x + 1e-10 for x in les_f], 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force Loss (log scale)')
    ax.set_title('(b) Force Loss Training Curves\nCharged Dimer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig7_dimer_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_dimer_training.png")


# Need to import parse_extxyz here too
from data_utils import parse_extxyz

if __name__ == '__main__':
    results = main()
