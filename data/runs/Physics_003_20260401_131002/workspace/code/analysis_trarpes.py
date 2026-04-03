#!/usr/bin/env python
"""Analysis of Floquet-Bloch states in pumped graphene tr-ARPES data.

This script:
1. Loads raw tr-ARPES spectra from HDF5.
2. Visualizes equilibrium Dirac cone and pumped spectra at selected polarization angles.
3. Uses processed band data to overlay extracted Dirac and replica bands.
4. Quantifies polarization dependence of replica-band intensity.
5. Saves figures to report/images and intermediate data products to outputs.

Reproducible: run from workspace root as
    python code/analysis_trarpes.py
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="talk", style="white")

BASE = Path('.')
DATA_DIR = BASE / 'data'
OUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

RAW_PATH = DATA_DIR / 'raw_trARPES_data.h5'
PROC_PATH = DATA_DIR / 'processed_band_data.json'
POL_PATH = DATA_DIR / 'polarization_dependence_data.csv'


def load_raw():
    with h5py.File(RAW_PATH, 'r') as f:
        energy = f['energy_axis'][...]
        kx = f['kx_axis'][...]
        time_delays = f['time_delays'][...]
        pol_angles = f['polarization_angles'][...]
        pump_off = f['pump_off_spectrum'][...]
        pump_on = {}
        for ang in pol_angles:
            dname = f'pump_on_angle_{int(ang)}'
            if dname in f:
                pump_on[int(ang)] = f[dname][...]
    return {
        'energy': energy,
        'kx': kx,
        'time_delays': time_delays,
        'pol_angles': pol_angles,
        'pump_off': pump_off,
        'pump_on': pump_on,
    }


def load_processed():
    with open(PROC_PATH, 'r') as f:
        proc = json.load(f)
    return proc


def load_polarization():
    return pd.read_csv(POL_PATH)


def plot_equilibrium_spectrum(raw):
    energy = raw['energy']
    kx = raw['kx']
    I0 = raw['pump_off']

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(kx, energy, I0, shading='auto', cmap='inferno')
    ax.set_xlabel(r'$k_x$ (1/Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Equilibrium ARPES spectrum (pump off)')
    fig.colorbar(im, ax=ax, label='Intensity (arb. units)')
    fig.tight_layout()
    out = FIG_DIR / 'fig_equilibrium_spectrum.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_pumped_spectra(raw):
    energy = raw['energy']
    kx = raw['kx']
    pump_on = raw['pump_on']

    # Sort angles for consistent layout
    angles = sorted(pump_on.keys())
    n = len(angles)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    vmin = min(I.min() for I in pump_on.values())
    vmax = max(I.max() for I in pump_on.values())

    for ax, ang in zip(axes.flat, angles):
        I = pump_on[ang]
        im = ax.pcolormesh(kx, energy, I, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(f'Pump on, $\\theta_p$ = {ang}°')
        ax.set_xlabel(r'$k_x$ (1/Å)')
        ax.set_ylabel('Energy (eV)')

    # Hide any unused axes
    for j in range(len(angles), nrows*ncols):
        axes.flat[j].axis('off')

    fig.suptitle('Pumped tr-ARPES spectra vs polarization angle', y=0.93)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label='Intensity (arb. units)')
    out = FIG_DIR / 'fig_pumped_spectra_vs_angle.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_bands_with_replicas(raw, proc):
    energy = np.array(proc['energy_axis'])
    kx = np.array(proc['kx_axis'])

    # Construct main Dirac band from dirac_indices along energy axis
    e_idx, k_idx = proc['dirac_indices']
    main_band = np.full_like(energy, np.nan, dtype=float)
    main_band[e_idx] = kx[k_idx]

    # Collect replica points by order
    replicas = {+1: [], -1: []}
    for pt in proc['replica_bands']:
        order = int(pt.get('order', 0))
        if order in replicas:
            replicas[order].append((pt['energy'], pt['kx']))

    replica_plus = np.array(sorted(replicas[+1])) if replicas[+1] else np.empty((0, 2))
    replica_minus = np.array(sorted(replicas[-1])) if replicas[-1] else np.empty((0, 2))

    # Use one representative pumped spectrum, e.g. 0°
    I = raw['pump_on'][sorted(raw['pump_on'].keys())[0]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(kx, energy, I, shading='auto', cmap='inferno')

    # Overlay bands
    if np.isfinite(main_band).any():
        ax.plot(main_band, energy, color='cyan', lw=2, label='Dirac cone')
    if replica_plus.size:
        ax.plot(replica_plus[:,1], replica_plus[:,0], color='lime', lw=1.5, ls='--', label='Floquet +1 replica')
    if replica_minus.size:
        ax.plot(replica_minus[:,1], replica_minus[:,0], color='magenta', lw=1.5, ls='--', label='Floquet -1 replica')

    ax.set_xlabel(r'$k_x$ (1/Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Floquet-Bloch replica bands in pumped graphene')
    ax.legend(loc='best', frameon=True)
    fig.colorbar(im, ax=ax, label='Intensity (arb. units)')
    fig.tight_layout()
    out = FIG_DIR / 'fig_floquet_replicas_overlay.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_polarization_dependence(pol_df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    theta = pol_df['angle_degrees']
    intensity = pol_df['intensity']

    ax.plot(theta, intensity, 'o-', color='C3', label='Replica intensity')

    # Fit simple cos^2 model: I = A * cos^2(theta - theta0) + B
    theta_rad = np.deg2rad(theta.values)
    X = np.column_stack([
        np.cos(theta_rad)**2,
        np.sin(theta_rad)**2,
        np.ones_like(theta_rad),
    ])
    y = intensity.values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    A_c, A_s, B = coeffs

    theta_fit = np.linspace(0, 180, 361)
    theta_fit_rad = np.deg2rad(theta_fit)
    I_fit = A_c * np.cos(theta_fit_rad)**2 + A_s * np.sin(theta_fit_rad)**2 + B
    ax.plot(theta_fit, I_fit, '-', color='k', alpha=0.7, label='Anisotropic cos$^2$ fit')

    ax.set_xlabel('Pump polarization \u03b8_p (deg)')
    ax.set_ylabel('Replica-band intensity (arb. units)')
    ax.set_title('Polarization dependence of Floquet replica intensity')
    ax.legend(frameon=True)
    fig.tight_layout()
    out = FIG_DIR / 'fig_polarization_dependence.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)

    # Save fit parameters
    fit_info = {
        'A_cos2': float(A_c),
        'A_sin2': float(A_s),
        'B_offset': float(B),
    }
    with open(OUT_DIR / 'polarization_fit_parameters.json', 'w') as f:
        json.dump(fit_info, f, indent=2)

    return out, fit_info


def main():
    raw = load_raw()
    proc = load_processed()
    pol_df = load_polarization()

    eq_fig = plot_equilibrium_spectrum(raw)
    pumped_fig = plot_pumped_spectra(raw)
    replica_fig = plot_bands_with_replicas(raw, proc)
    pol_fig, fit_info = plot_polarization_dependence(pol_df)

    summary = {
        'equilibrium_spectrum_figure': str(eq_fig),
        'pumped_spectra_figure': str(pumped_fig),
        'floquet_replicas_figure': str(replica_fig),
        'polarization_dependence_figure': str(pol_fig),
        'polarization_fit_parameters': fit_info,
    }
    with open(OUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
