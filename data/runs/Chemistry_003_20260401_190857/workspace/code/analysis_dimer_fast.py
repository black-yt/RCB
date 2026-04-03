"""
Analysis 2 (Fast): Charged Dimer Binding Energy Curves
Uses force information for better training.
"""
import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, get_dimer_separation
from fast_les_model import FastLESModel, FastSRModel, train_fast, evaluate_fast
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt

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

    frames = parse_extxyz(f'{DATA_DIR}/charged_dimer.xyz')
    positions_list = [fr['positions'] for fr in frames]
    energies = np.array([fr['energy'] for fr in frames])
    forces_list = [fr.get('forces') for fr in frames]
    separations = np.array([get_dimer_separation(fr['positions']) for fr in frames])

    print(f"  Loaded {len(frames)} configs")
    print(f"  Separation: {separations.min():.2f} to {separations.max():.2f} Å")
    print(f"  Energy: {energies.min():.4f} to {energies.max():.4f} eV")

    # Sort by separation
    sort_idx = np.argsort(separations)

    # We use all data for training since the goal is to capture the binding curve
    pos_train = positions_list
    e_train = energies
    f_train = forces_list

    # SR model with small cutoff (4 Å) - can't see across dimer gap at large separations
    print("\n[2] Training SR model (cutoff=4.0Å)...")
    sr_model = FastSRModel(n_rbf=16, n_hidden=32, cutoff=4.0)
    sr_losses = train_fast(sr_model, pos_train, e_train, forces_list=f_train,
                           n_epochs=300, lr=5e-4, e_weight=1.0, f_weight=0.1, verbose=True)
    sr_res = evaluate_fast(sr_model, positions_list, energies)
    print(f"  SR MAE={sr_res['mae']:.4f}, R²={sr_res['r2']:.4f}")

    # LES model
    print("\n[3] Training LES model (cutoff=4.0Å)...")
    les_model = FastLESModel(n_rbf=16, n_hidden=32, cutoff=4.0)
    les_losses = train_fast(les_model, pos_train, e_train, forces_list=f_train,
                            n_epochs=300, lr=5e-4, e_weight=1.0, f_weight=0.1, verbose=True)
    les_res = evaluate_fast(les_model, positions_list, energies, return_charges=True)
    print(f"  LES MAE={les_res['mae']:.4f}, R²={les_res['r2']:.4f}")

    np.save(f'{OUT_DIR}/dimer_results.npy', {
        'sr': sr_res, 'les': les_res, 'separations': separations,
        'energies': energies, 'sr_losses': sr_losses, 'les_losses': les_losses
    }, allow_pickle=True)

    print("\n[4] Generating figures...")
    fig_dimer_data(frames, separations, energies)
    fig_dimer_curves(separations, energies, sr_res, les_res, sort_idx)
    fig_dimer_error(separations, energies, sr_res, les_res, sort_idx)

    print(f"\n[Summary]")
    print(f"  SR  MAE: {sr_res['mae']:.4f} eV")
    print(f"  LES MAE: {les_res['mae']:.4f} eV")
    return sr_res, les_res, separations, energies


def fig_dimer_data(frames, separations, energies):
    """Overview figure for dimer dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Example dimer structure
    ax = axes[0]
    close_i = np.argmin(separations)
    far_i = np.argmax(separations)
    for idx, yoff, label in [(close_i, 0, 'Close'), (far_i, 3, 'Far')]:
        pos = frames[idx]['positions'].copy()
        pos -= pos.mean(axis=0)
        pos[:, 1] += yoff
        symbols = ['C','H','H','H','C','H','H','H']
        for i, (xi, zi) in enumerate(zip(pos[:, 0], pos[:, 2])):
            clr = '#404040' if symbols[i] == 'C' else '#DDDDDD'
            sz = 180 if symbols[i] == 'C' else 70
            ax.scatter(xi, zi, c=clr, s=sz, edgecolors='k', lw=0.8, zorder=5)
    ax.text(0, 1.5, f'Sep={separations[far_i]:.1f}Å', ha='center', fontsize=9, color='red')
    ax.set_xlabel('x (Å)'); ax.set_ylabel('z (Å)')
    ax.set_title('(a) Dimer Configurations\n(Close and Far)')
    ax.set_aspect('equal')

    # Panel 2: E vs separation (all data)
    ax = axes[1]
    sort_i = np.argsort(separations)
    ax.scatter(separations, energies, c='black', s=20, alpha=0.7)
    ax.set_xlabel('Monomer Separation (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('(b) Binding Energy vs. Separation\n(60 configurations)')
    ax.axvline(x=4.0, color='gray', linestyle='--', lw=1.5, label='SR cutoff (4Å)')
    ax.legend()

    # Panel 3: Force distribution
    ax = axes[2]
    all_f = np.concatenate([fr.get('forces', np.zeros((8, 3))) for fr in frames])
    fmag = np.linalg.norm(all_f, axis=1)
    ax.hist(fmag, bins=30, color='seagreen', alpha=0.7, density=True, edgecolor='k', lw=0.3)
    ax.set_xlabel('Force Magnitude (eV/Å)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(c) Force Distribution\n(all 60 configs)')

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig5_dimer_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig5_dimer_overview.png')


def fig_dimer_curves(separations, energies, sr_res, les_res, sort_idx):
    """Binding energy curves."""
    fig, ax = plt.subplots(figsize=(9, 6))
    seps = separations[sort_idx]
    e_ref = energies[sort_idx]
    e_sr = sr_res['energies_pred'][sort_idx]
    e_les = les_res['energies_pred'][sort_idx]

    ax.plot(seps, e_ref, 'ko-', ms=5, lw=2, label='Reference', zorder=5)
    ax.plot(seps, e_sr, 'b^--', ms=6, lw=2, label=f'SR model (MAE={sr_res["mae"]:.3f} eV)')
    ax.plot(seps, e_les, 'r*-', ms=8, lw=2, label=f'LES model (MAE={les_res["mae"]:.3f} eV)')
    ax.axvline(x=4.0, color='gray', linestyle=':', lw=2, alpha=0.7, label='SR cutoff (4Å)')
    ax.set_xlabel('Dimer Separation (Å)', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Charged Dimer Binding Curve: SR vs LES', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig6_dimer_binding_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig6_dimer_binding_curves.png')


def fig_dimer_error(separations, energies, sr_res, les_res, sort_idx):
    """Error analysis for dimer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    seps = separations[sort_idx]
    e_ref = energies[sort_idx]
    e_sr = sr_res['energies_pred'][sort_idx]
    e_les = les_res['energies_pred'][sort_idx]

    ax = axes[0]
    ax.plot(seps, np.abs(e_sr - e_ref), 'b^--', ms=6, lw=2, label='SR error')
    ax.plot(seps, np.abs(e_les - e_ref), 'r*-', ms=7, lw=2, label='LES error')
    ax.axvline(x=4.0, color='gray', linestyle=':', lw=2, alpha=0.7, label='SR cutoff')
    ax.set_xlabel('Dimer Separation (Å)', fontsize=12)
    ax.set_ylabel('|Error| (eV)', fontsize=12)
    ax.set_title('(a) Prediction Error vs. Separation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    # Energy parity
    e_all = np.concatenate([e_ref, e_sr, e_les])
    lim = [e_all.min() - 0.05, e_all.max() + 0.05]
    ax.scatter(e_ref, e_sr, c='steelblue', s=40, alpha=0.7, label=f'SR (R²={sr_res["r2"]:.4f})')
    ax.scatter(e_ref, e_les, c='tomato', marker='^', s=40, alpha=0.7, label=f'LES (R²={les_res["r2"]:.4f})')
    ax.plot(lim, lim, 'k--', lw=1.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Reference Energy (eV)', fontsize=12)
    ax.set_ylabel('Predicted Energy (eV)', fontsize=12)
    ax.set_title('(b) Energy Parity Plot')
    ax.legend()
    ax.set_aspect('equal')

    plt.suptitle('Charged Dimer: Error Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig7_dimer_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig7_dimer_error_analysis.png')


if __name__ == '__main__':
    main()
