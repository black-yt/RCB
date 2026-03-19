"""
Main analysis script: Bayesian constraints on ultralight bosons from BH superradiance.

Processes M33 X-7 (stellar-mass) and IRAS 09149-6206 (supermassive) posterior samples
to derive exclusion limits on ULB masses and self-interaction couplings.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import sys
import json

# Add code directory to path
sys.path.insert(0, os.path.dirname(__file__))
from superradiance import (
    alpha, omega_plus, superradiance_condition, superradiance_rate_nlm,
    superradiance_timescale, critical_spin_for_superradiance,
    exclusion_probability_single_sample, bayesian_exclusion,
    compute_exclusion_curve, bosenova_critical_mass_fraction,
    self_interaction_constraint, load_samples,
    G_N, c, hbar, M_sun, M_Pl, yr_to_s, age_universe
)

# Paths
BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, 'data')
OUTPUTS = os.path.join(BASE, 'outputs')
IMAGES = os.path.join(BASE, 'report', 'images')

os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(IMAGES, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


def main():
    print("Loading data...")
    M_m33, a_m33 = load_samples(os.path.join(DATA, 'M33_X-7_samples.dat'))
    M_iras, a_iras = load_samples(os.path.join(DATA, 'IRAS_09149-6206_samples.dat'))

    print(f"M33 X-7: {len(M_m33)} samples, M=[{M_m33.min():.2f}, {M_m33.max():.2f}] Msun, "
          f"a*=[{a_m33.min():.3f}, {a_m33.max():.3f}]")
    print(f"IRAS 09149-6206: {len(M_iras)} samples, M=[{M_iras.min():.2e}, {M_iras.max():.2e}] Msun, "
          f"a*=[{a_iras.min():.3f}, {a_iras.max():.3f}]")

    # =========================================================================
    # Figure 1: Data overview - posterior distributions
    # =========================================================================
    print("\nGenerating Figure 1: Posterior distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # M33 X-7
    axes[0, 0].hist(M_m33, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='navy')
    axes[0, 0].set_xlabel(r'$M_{\rm BH}$ [$M_\odot$]')
    axes[0, 0].set_ylabel('Probability density')
    axes[0, 0].set_title('M33 X-7: Mass')

    axes[0, 1].hist(a_m33, bins=50, density=True, color='coral', alpha=0.7, edgecolor='darkred')
    axes[0, 1].set_xlabel(r'$a_*$')
    axes[0, 1].set_ylabel('Probability density')
    axes[0, 1].set_title('M33 X-7: Spin')

    axes[0, 2].scatter(M_m33[::5], a_m33[::5], s=1, alpha=0.3, c='purple')
    axes[0, 2].set_xlabel(r'$M_{\rm BH}$ [$M_\odot$]')
    axes[0, 2].set_ylabel(r'$a_*$')
    axes[0, 2].set_title('M33 X-7: Joint posterior')

    # IRAS 09149-6206
    axes[1, 0].hist(M_iras / 1e7, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='navy')
    axes[1, 0].set_xlabel(r'$M_{\rm BH}$ [$10^7 M_\odot$]')
    axes[1, 0].set_ylabel('Probability density')
    axes[1, 0].set_title('IRAS 09149-6206: Mass')

    axes[1, 1].hist(a_iras, bins=50, density=True, color='coral', alpha=0.7, edgecolor='darkred')
    axes[1, 1].set_xlabel(r'$a_*$')
    axes[1, 1].set_ylabel('Probability density')
    axes[1, 1].set_title('IRAS 09149-6206: Spin')

    axes[1, 2].scatter(M_iras[::5] / 1e7, a_iras[::5], s=1, alpha=0.3, c='purple')
    axes[1, 2].set_xlabel(r'$M_{\rm BH}$ [$10^7 M_\odot$]')
    axes[1, 2].set_ylabel(r'$a_*$')
    axes[1, 2].set_title('IRAS 09149-6206: Joint posterior')

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, 'fig1_posteriors.png'))
    plt.close()

    # =========================================================================
    # Figure 2: Regge plane with data
    # =========================================================================
    print("Generating Figure 2: Regge plane...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel a: stellar-mass BH regime ---
    ax = axes[0]
    # Representative boson masses for stellar-mass BH
    mu_stellar = [1e-12, 3e-12, 1e-11, 3e-11, 1e-10]
    colors_mu = plt.cm.viridis(np.linspace(0.1, 0.9, len(mu_stellar)))

    M_grid = np.linspace(3, 20, 500)
    for idx, mu in enumerate(mu_stellar):
        for m_mode in [1, 2, 3]:
            alpha_grid = alpha(mu, M_grid)
            a_regge = critical_spin_for_superradiance(alpha_grid, m=m_mode)
            valid = (alpha_grid > 0.01) & (alpha_grid < m_mode * 0.5)
            if np.any(valid):
                label = rf'$\mu_b={mu:.0e}$ eV' if m_mode == 1 else None
                ls = ['-', '--', ':'][m_mode - 1]
                ax.plot(M_grid[valid], a_regge[valid], color=colors_mu[idx],
                       ls=ls, lw=1.5, label=label)

    # Overlay M33 X-7 data
    ax.scatter(M_m33[::3], a_m33[::3], s=2, alpha=0.15, c='red', zorder=5)
    ax.scatter([np.median(M_m33)], [np.median(a_m33)], s=100, c='red',
              marker='*', edgecolors='black', zorder=10, label='M33 X-7')

    ax.set_xlabel(r'$M_{\rm BH}$ [$M_\odot$]')
    ax.set_ylabel(r'$a_*$')
    ax.set_title('Stellar-mass BH: Regge Plane')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')

    # Line style legend
    legend_lines = [Line2D([0], [0], color='gray', ls='-', lw=1.5),
                    Line2D([0], [0], color='gray', ls='--', lw=1.5),
                    Line2D([0], [0], color='gray', ls=':', lw=1.5)]
    ax.legend(legend_lines, ['l=m=1', 'l=m=2', 'l=m=3'],
             loc='upper right', fontsize=8, title='Modes')

    # --- Panel b: supermassive BH regime ---
    ax = axes[1]
    mu_smbh = [1e-19, 3e-19, 1e-18, 3e-18, 1e-17]
    colors_mu2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(mu_smbh)))

    M_grid2 = np.logspace(6.5, 8.5, 500)
    for idx, mu in enumerate(mu_smbh):
        for m_mode in [1, 2]:
            alpha_grid = alpha(mu, M_grid2)
            a_regge = critical_spin_for_superradiance(alpha_grid, m=m_mode)
            valid = (alpha_grid > 0.01) & (alpha_grid < m_mode * 0.5)
            if np.any(valid):
                label = rf'$\mu_b={mu:.0e}$ eV' if m_mode == 1 else None
                ls = ['-', '--'][m_mode - 1]
                ax.plot(M_grid2[valid] / 1e7, a_regge[valid], color=colors_mu2[idx],
                       ls=ls, lw=1.5, label=label)

    ax.scatter(M_iras[::10] / 1e7, a_iras[::10], s=2, alpha=0.15, c='red', zorder=5)
    ax.scatter([np.median(M_iras) / 1e7], [np.median(a_iras)], s=100, c='red',
              marker='*', edgecolors='black', zorder=10, label='IRAS 09149-6206')

    ax.set_xlabel(r'$M_{\rm BH}$ [$10^7 M_\odot$]')
    ax.set_ylabel(r'$a_*$')
    ax.set_title('Supermassive BH: Regge Plane')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, 'fig2_regge_plane.png'))
    plt.close()

    # =========================================================================
    # Figure 3: Superradiance timescales
    # =========================================================================
    print("Generating Figure 3: Superradiance timescales...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stellar-mass
    ax = axes[0]
    mu_grid_stellar = np.logspace(-13, -9, 200)
    M_bh_ref = np.median(M_m33)
    a_ref = np.median(a_m33)
    for l_mode in [1, 2, 3, 4]:
        taus = []
        for mu in mu_grid_stellar:
            tau = superradiance_timescale(alpha(mu, M_bh_ref), a_ref, M_bh_ref,
                                          l=l_mode, m=l_mode, n=0)
            taus.append(tau / yr_to_s)
        taus = np.array(taus)
        valid = np.isfinite(taus) & (taus > 0) & (taus < 1e20)
        if np.any(valid):
            ax.plot(mu_grid_stellar[valid], taus[valid], label=f'l=m={l_mode}', lw=2)

    ax.axhline(age_universe / yr_to_s, color='gray', ls='--', lw=1, label='Age of Universe')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mu_b$ [eV]')
    ax.set_ylabel(r'$\tau_{\rm sr}$ [yr]')
    ax.set_title(f'M33 X-7 (M={M_bh_ref:.1f} $M_\\odot$, $a_*$={a_ref:.3f})')
    ax.legend()
    ax.set_ylim(1e-5, 1e18)

    # Supermassive
    ax = axes[1]
    mu_grid_smbh = np.logspace(-21, -16, 200)
    M_bh_ref2 = np.median(M_iras)
    a_ref2 = np.median(a_iras)
    for l_mode in [1, 2, 3]:
        taus = []
        for mu in mu_grid_smbh:
            tau = superradiance_timescale(alpha(mu, M_bh_ref2), a_ref2, M_bh_ref2,
                                          l=l_mode, m=l_mode, n=0)
            taus.append(tau / yr_to_s)
        taus = np.array(taus)
        valid = np.isfinite(taus) & (taus > 0) & (taus < 1e20)
        if np.any(valid):
            ax.plot(mu_grid_smbh[valid], taus[valid], label=f'l=m={l_mode}', lw=2)

    ax.axhline(age_universe / yr_to_s, color='gray', ls='--', lw=1, label='Age of Universe')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mu_b$ [eV]')
    ax.set_ylabel(r'$\tau_{\rm sr}$ [yr]')
    ax.set_title(f'IRAS 09149-6206 (M={M_bh_ref2:.2e} $M_\\odot$, $a_*$={a_ref2:.3f})')
    ax.legend()
    ax.set_ylim(1e-5, 1e18)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, 'fig3_timescales.png'))
    plt.close()

    # =========================================================================
    # Figure 4: Main result - Bayesian exclusion curves
    # =========================================================================
    print("Computing Bayesian exclusion curves (this may take a while)...")

    # Stellar-mass BH: sensitive to mu ~ 1e-12 to 1e-10 eV
    mu_grid_m33 = np.logspace(-13, -9, 100)
    # Subsample for speed
    n_sub = min(500, len(M_m33))
    idx_sub = np.random.choice(len(M_m33), n_sub, replace=False)
    M_m33_sub, a_m33_sub = M_m33[idx_sub], a_m33[idx_sub]

    p_excl_m33 = compute_exclusion_curve(M_m33_sub, a_m33_sub, mu_grid_m33, l_max=5)

    # Supermassive BH: sensitive to mu ~ 1e-20 to 1e-16 eV
    mu_grid_iras = np.logspace(-21, -16, 100)
    n_sub2 = min(500, len(M_iras))
    idx_sub2 = np.random.choice(len(M_iras), n_sub2, replace=False)
    M_iras_sub, a_iras_sub = M_iras[idx_sub2], a_iras[idx_sub2]

    p_excl_iras = compute_exclusion_curve(M_iras_sub, a_iras_sub, mu_grid_iras, l_max=4)

    # Save intermediate results
    np.savez(os.path.join(OUTPUTS, 'exclusion_curves.npz'),
             mu_m33=mu_grid_m33, p_excl_m33=p_excl_m33,
             mu_iras=mu_grid_iras, p_excl_iras=p_excl_iras)

    print("Generating Figure 4: Exclusion curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(mu_grid_m33, p_excl_m33, 'b-', lw=2, label='M33 X-7')
    ax.axhline(0.95, color='red', ls='--', lw=1, label='95% CL')
    ax.axhline(0.99, color='darkred', ls=':', lw=1, label='99% CL')
    ax.fill_between(mu_grid_m33, p_excl_m33, alpha=0.2, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu_b$ [eV]')
    ax.set_ylabel('Exclusion probability')
    ax.set_title('Stellar-mass BH: M33 X-7')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(mu_grid_iras, p_excl_iras, 'r-', lw=2, label='IRAS 09149-6206')
    ax.axhline(0.95, color='red', ls='--', lw=1, label='95% CL')
    ax.axhline(0.99, color='darkred', ls=':', lw=1, label='99% CL')
    ax.fill_between(mu_grid_iras, p_excl_iras, alpha=0.2, color='red')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu_b$ [eV]')
    ax.set_ylabel('Exclusion probability')
    ax.set_title('Supermassive BH: IRAS 09149-6206')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, 'fig4_exclusion_curves.png'))
    plt.close()

    # Extract 95% bounds
    excl_95_m33 = mu_grid_m33[p_excl_m33 >= 0.95] if np.any(p_excl_m33 >= 0.95) else []
    excl_95_iras = mu_grid_iras[p_excl_iras >= 0.95] if np.any(p_excl_iras >= 0.95) else []

    print(f"\n--- 95% CL Exclusion Ranges ---")
    if len(excl_95_m33) > 0:
        print(f"M33 X-7: mu_b in [{excl_95_m33.min():.2e}, {excl_95_m33.max():.2e}] eV")
    else:
        print("M33 X-7: No 95% exclusion achieved")
    if len(excl_95_iras) > 0:
        print(f"IRAS 09149-6206: mu_b in [{excl_95_iras.min():.2e}, {excl_95_iras.max():.2e}] eV")
    else:
        print("IRAS 09149-6206: No 95% exclusion achieved")

    # =========================================================================
    # Figure 5: Combined exclusion plot
    # =========================================================================
    print("\nGenerating Figure 5: Combined exclusion...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mu_grid_m33, p_excl_m33, 'b-', lw=2.5, label='M33 X-7 (stellar-mass)')
    ax.plot(mu_grid_iras, p_excl_iras, 'r-', lw=2.5, label='IRAS 09149-6206 (supermassive)')
    ax.axhline(0.95, color='gray', ls='--', lw=1, label='95% CL threshold')
    ax.set_xscale('log')
    ax.set_xlabel(r'Ultralight Boson Mass $\mu_b$ [eV]', fontsize=14)
    ax.set_ylabel('Bayesian Exclusion Probability', fontsize=14)
    ax.set_title('Combined ULB Mass Exclusion from BH Superradiance', fontsize=15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Shade excluded regions
    if len(excl_95_m33) > 0:
        ax.axvspan(excl_95_m33.min(), excl_95_m33.max(), alpha=0.1, color='blue')
    if len(excl_95_iras) > 0:
        ax.axvspan(excl_95_iras.min(), excl_95_iras.max(), alpha=0.1, color='red')

    plt.savefig(os.path.join(IMAGES, 'fig5_combined_exclusion.png'))
    plt.close()

    # =========================================================================
    # Figure 6: Self-interaction constraints
    # =========================================================================
    print("Computing self-interaction constraints...")

    # For selected boson masses in the excluded range, vary fa and see how exclusion changes
    # Pick representative masses
    if len(excl_95_m33) > 0:
        mu_test_m33 = np.exp(np.mean(np.log(excl_95_m33)))  # geometric mean
    else:
        mu_test_m33 = 3e-11  # fallback

    if len(excl_95_iras) > 0:
        mu_test_iras = np.exp(np.mean(np.log(excl_95_iras)))
    else:
        mu_test_iras = 1e-18

    fa_grid = np.logspace(14, 19, 30)  # in eV (10^14 to 10^19 eV)
    p_excl_fa_m33 = []
    p_excl_fa_iras = []

    for fa in fa_grid:
        p1 = self_interaction_constraint(M_m33_sub, a_m33_sub, mu_test_m33, fa, l_max=5)
        p2 = self_interaction_constraint(M_iras_sub, a_iras_sub, mu_test_iras, fa, l_max=4)
        p_excl_fa_m33.append(p1)
        p_excl_fa_iras.append(p2)

    p_excl_fa_m33 = np.array(p_excl_fa_m33)
    p_excl_fa_iras = np.array(p_excl_fa_iras)

    np.savez(os.path.join(OUTPUTS, 'self_interaction_constraints.npz'),
             fa_grid=fa_grid,
             mu_test_m33=mu_test_m33, p_excl_fa_m33=p_excl_fa_m33,
             mu_test_iras=mu_test_iras, p_excl_fa_iras=p_excl_fa_iras)

    print("Generating Figure 6: Self-interaction constraints...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(fa_grid, p_excl_fa_m33, 'b-o', lw=2, ms=4,
           label=f'M33 X-7, $\\mu_b$={mu_test_m33:.1e} eV')
    ax.plot(fa_grid, p_excl_fa_iras, 'r-s', lw=2, ms=4,
           label=f'IRAS 09149-6206, $\\mu_b$={mu_test_iras:.1e} eV')
    ax.axhline(0.95, color='gray', ls='--', lw=1, label='95% CL')
    ax.set_xscale('log')
    ax.set_xlabel(r'Decay Constant $f_a$ [eV]', fontsize=14)
    ax.set_ylabel('Exclusion Probability', fontsize=14)
    ax.set_title('Constraints on Self-Interaction Coupling Strength', fontsize=15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark GUT and Planck scales
    ax.axvline(2e16, color='green', ls=':', lw=1.5, alpha=0.7)
    ax.text(2e16, 0.5, r'$M_{\rm GUT}$', rotation=90, va='center', fontsize=10, color='green')
    ax.axvline(M_Pl, color='orange', ls=':', lw=1.5, alpha=0.7)
    ax.text(M_Pl * 0.7, 0.5, r'$M_{\rm Pl}$', rotation=90, va='center', fontsize=10, color='orange')

    plt.savefig(os.path.join(IMAGES, 'fig6_self_interaction.png'))
    plt.close()

    # =========================================================================
    # Figure 7: 2D exclusion in (mu, fa) plane
    # =========================================================================
    print("Computing 2D exclusion map...")

    mu_2d_m33 = np.logspace(-13, -9, 40)
    fa_2d = np.logspace(14, 19, 30)

    excl_2d_m33 = np.zeros((len(fa_2d), len(mu_2d_m33)))
    for i, fa in enumerate(fa_2d):
        for j, mu in enumerate(mu_2d_m33):
            excl_2d_m33[i, j] = self_interaction_constraint(
                M_m33_sub, a_m33_sub, mu, fa, l_max=5)
        print(f"  fa row {i+1}/{len(fa_2d)} done")

    np.savez(os.path.join(OUTPUTS, 'exclusion_2d_m33.npz'),
             mu_grid=mu_2d_m33, fa_grid=fa_2d, excl_2d=excl_2d_m33)

    print("Generating Figure 7: 2D exclusion map...")
    fig, ax = plt.subplots(figsize=(10, 7))
    MU, FA = np.meshgrid(mu_2d_m33, fa_2d)
    pcm = ax.pcolormesh(MU, FA, excl_2d_m33, cmap='RdYlBu_r',
                         vmin=0, vmax=1, shading='auto')
    ax.contour(MU, FA, excl_2d_m33, levels=[0.95], colors='black', linewidths=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Ultralight Boson Mass $\mu_b$ [eV]', fontsize=14)
    ax.set_ylabel(r'Decay Constant $f_a$ [eV]', fontsize=14)
    ax.set_title(r'M33 X-7: Exclusion in ($\mu_b$, $f_a$) Plane', fontsize=15)
    plt.colorbar(pcm, ax=ax, label='Exclusion probability')
    ax.axhline(2e16, color='green', ls=':', lw=1.5, alpha=0.7)
    ax.text(mu_2d_m33[1], 2e16 * 1.3, r'$M_{\rm GUT}$', fontsize=10, color='green')

    plt.savefig(os.path.join(IMAGES, 'fig7_2d_exclusion.png'))
    plt.close()

    # =========================================================================
    # Figure 8: Validation - check superradiance rates
    # =========================================================================
    print("Generating Figure 8: Validation plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel a: Superradiance rates vs alpha for different l
    ax = axes[0]
    alpha_grid = np.linspace(0.01, 2.5, 300)
    a_star_val = 0.999  # near-extremal

    for l_mode in [1, 2, 3, 4, 5]:
        rates = []
        for alp in alpha_grid:
            r = superradiance_rate_nlm(alp, a_star_val, l=l_mode, m=l_mode, n=0)
            rates.append(r)
        rates = np.array(rates)
        valid = rates > 0
        if np.any(valid):
            ax.plot(alpha_grid[valid], rates[valid], lw=2, label=f'l=m={l_mode}')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha = \mu_b r_g$')
    ax.set_ylabel(r'$\Gamma_{\rm sr}$ [$r_g^{-1}$]')
    ax.set_title(r'Superradiance Rates ($a_*/r_g = 0.999$)')
    ax.legend()
    ax.set_ylim(1e-18, 1e-5)
    ax.grid(True, alpha=0.3)

    # Panel b: Rates for l=1 at different spins
    ax = axes[1]
    spins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    for a_val in spins:
        rates = []
        for alp in alpha_grid[:150]:  # l=1 only needs small alpha
            r = superradiance_rate_nlm(alp, a_val, l=1, m=1, n=0)
            rates.append(r)
        rates = np.array(rates)
        valid = rates > 0
        if np.any(valid):
            ax.plot(alpha_grid[:150][valid], rates[valid], lw=2,
                   label=f'$a_*={a_val}$')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha = \mu_b r_g$')
    ax.set_ylabel(r'$\Gamma_{\rm sr}$ [$r_g^{-1}$]')
    ax.set_title(r'$l=m=1$ Rates at Different Spins')
    ax.legend(fontsize=9)
    ax.set_ylim(1e-18, 1e-5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, 'fig8_validation.png'))
    plt.close()

    # =========================================================================
    # Summary output
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    summary = {
        'M33_X-7': {
            'n_samples': len(M_m33),
            'M_median': float(np.median(M_m33)),
            'M_std': float(np.std(M_m33)),
            'a_median': float(np.median(a_m33)),
            'a_std': float(np.std(a_m33)),
        },
        'IRAS_09149-6206': {
            'n_samples': len(M_iras),
            'M_median': float(np.median(M_iras)),
            'M_std': float(np.std(M_iras)),
            'a_median': float(np.median(a_iras)),
            'a_std': float(np.std(a_iras)),
        },
        'exclusion_95CL': {}
    }

    if len(excl_95_m33) > 0:
        summary['exclusion_95CL']['M33_X-7'] = {
            'mu_min_eV': float(excl_95_m33.min()),
            'mu_max_eV': float(excl_95_m33.max()),
        }
    if len(excl_95_iras) > 0:
        summary['exclusion_95CL']['IRAS_09149-6206'] = {
            'mu_min_eV': float(excl_95_iras.min()),
            'mu_max_eV': float(excl_95_iras.max()),
        }

    with open(os.path.join(OUTPUTS, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved to: {IMAGES}")
    print(f"Outputs saved to: {OUTPUTS}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
