import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit


ROOT = Path('.')
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
IMAGES = ROOT / 'report' / 'images'

OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def load_data():
    with h5py.File(DATA / 'raw_trARPES_data.h5', 'r') as f:
        raw = {
            'energy_axis': f['energy_axis'][:],
            'kx_axis': f['kx_axis'][:],
            'time_delays': f['time_delays'][:],
            'polarization_angles': f['polarization_angles'][:],
            'pump_off_spectrum': f['pump_off_spectrum'][:],
        }
        pump_on = {}
        for key in f.keys():
            if key.startswith('pump_on_angle_'):
                angle = int(key.split('_')[-1])
                pump_on[angle] = f[key][:]
        raw['pump_on'] = pump_on

    with open(DATA / 'processed_band_data.json', 'r', encoding='utf-8') as fh:
        processed = json.load(fh)

    pol = pd.read_csv(DATA / 'polarization_dependence_data.csv')
    return raw, processed, pol


def cosine2(theta, a0, a1, phi):
    return a0 + a1 * np.cos(2 * (theta - phi))


def summarize_data(raw, processed, pol):
    photon_energy_ev = 1.239841984 / 5.0
    dirac_kx, dirac_energy = processed['dirac_point']
    replicas = processed['replica_bands']

    replica_summary = []
    spacings = []
    for rep in replicas:
        delta_e = rep['energy'] - dirac_energy
        spacing_error = delta_e - rep['order'] * photon_energy_ev
        spacings.append(delta_e)
        replica_summary.append({
            'order': rep['order'],
            'kx': rep['kx'],
            'energy': rep['energy'],
            'intensity': rep['intensity'],
            'delta_energy_from_dirac_eV': delta_e,
            'expected_order_energy_eV': rep['order'] * photon_energy_ev,
            'spacing_error_eV': spacing_error,
        })

    dispersion = pd.DataFrame(processed['band_dispersion'])
    coeffs = np.polyfit(np.abs(dispersion['kx']), np.abs(dispersion['energy'] - dirac_energy), 1)
    slope_eV_per_Ainv = float(coeffs[0])

    popt, pcov = curve_fit(
        cosine2,
        pol['angle_radians'].values,
        pol['intensity'].values,
        p0=[pol['intensity'].mean(), (pol['intensity'].max() - pol['intensity'].min()) / 2, 0.0],
    )
    pred = cosine2(pol['angle_radians'].values, *popt)
    ss_res = float(np.sum((pol['intensity'].values - pred) ** 2))
    ss_tot = float(np.sum((pol['intensity'].values - pol['intensity'].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    summary = {
        'raw_data': {
            'energy_points': int(len(raw['energy_axis'])),
            'kx_points': int(len(raw['kx_axis'])),
            'time_delays': raw['time_delays'].tolist(),
            'polarization_angles': raw['polarization_angles'].tolist(),
            'energy_range_eV': [float(raw['energy_axis'].min()), float(raw['energy_axis'].max())],
            'kx_range_Ainv': [float(raw['kx_axis'].min()), float(raw['kx_axis'].max())],
            'pump_off_mean': float(raw['pump_off_spectrum'].mean()),
            'pump_off_std': float(raw['pump_off_spectrum'].std()),
        },
        'processed_data': {
            'dirac_point_kx_energy': [float(dirac_kx), float(dirac_energy)],
            'replica_count': len(replicas),
            'replica_summary': replica_summary,
            'estimated_dirac_cone_slope_eV_per_Ainv': slope_eV_per_Ainv,
        },
        'physics_metrics': {
            'pump_wavelength_um': 5.0,
            'photon_energy_eV': photon_energy_ev,
            'mean_abs_replica_spacing_eV': float(np.mean(np.abs(spacings))),
            'max_abs_spacing_error_eV': float(np.max(np.abs([r['spacing_error_eV'] for r in replica_summary]))),
            'replica_intensity_mean': float(np.mean([r['intensity'] for r in replicas])),
            'replica_intensity_std': float(np.std([r['intensity'] for r in replicas])),
        },
        'polarization_fit': {
            'offset_a0': float(popt[0]),
            'amplitude_a1': float(popt[1]),
            'phase_phi_rad': float(popt[2]),
            'r_squared': float(r2),
            'modulation_depth_percent': float(100 * (pol['intensity'].max() - pol['intensity'].min()) / pol['intensity'].mean()),
        },
    }
    return summary, dispersion, np.array(pred)


def plot_raw_spectra(raw, processed):
    energy = raw['energy_axis']
    kx = raw['kx_axis']
    pump_off = raw['pump_off_spectrum']
    angle0 = raw['pump_on'][0]
    diff0 = angle0 - pump_off
    dirac_kx, dirac_energy = processed['dirac_point']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    datasets = [
        (pump_off, 'Pump off'),
        (angle0, 'Pump on (0° polarization)'),
        (diff0, 'Pump on - pump off (0°)'),
    ]
    for ax, (spec, title) in zip(axes, datasets):
        im = ax.imshow(
            spec,
            aspect='auto',
            origin='lower',
            extent=[kx.min(), kx.max(), energy.min(), energy.max()],
            cmap='magma',
        )
        ax.scatter([dirac_kx], [dirac_energy], c='cyan', s=50, label='Dirac point')
        ax.set_title(title)
        ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
        ax.set_ylabel('Energy (eV)')
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].legend(loc='upper right', frameon=True)
    fig.savefig(IMAGES / 'raw_spectra_overview.png')
    plt.close(fig)


def plot_dispersion_and_replicas(processed):
    dispersion = pd.DataFrame(processed['band_dispersion'])
    replicas = pd.DataFrame(processed['replica_bands'])
    dirac_kx, dirac_energy = processed['dirac_point']

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        dispersion['kx'],
        dispersion['energy'],
        c=dispersion['intensity'],
        cmap='viridis',
        s=28,
        label='Main cone dispersion',
    )
    ax.scatter(replicas['kx'], replicas['energy'], s=140, c=replicas['order'], cmap='coolwarm', marker='X', label='Replica bands')
    ax.scatter([dirac_kx], [dirac_energy], c='black', s=80, marker='*', label='Dirac point')
    for _, row in replicas.iterrows():
        ax.plot([dirac_kx, row['kx']], [dirac_energy, row['energy']], '--', color='gray', alpha=0.6)
        ax.text(row['kx'] + 0.005, row['energy'], f"n={int(row['order'])}", fontsize=10)
    ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Extracted Dirac-cone dispersion and Floquet replica bands')
    ax.legend(loc='best', frameon=True)
    fig.colorbar(sc, ax=ax, label='Extracted intensity')
    fig.savefig(IMAGES / 'dispersion_and_replicas.png')
    plt.close(fig)


def plot_replica_energy_alignment(processed):
    replicas = pd.DataFrame(processed['replica_bands'])
    dirac_energy = processed['dirac_point'][1]
    photon_energy = 1.239841984 / 5.0
    observed = replicas.groupby('order')['energy'].mean().sort_index()
    orders = observed.index.to_numpy(dtype=float)
    expected = dirac_energy + orders * photon_energy

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(dirac_energy, color='black', linestyle=':', label='Dirac point energy')
    ax.plot(orders, observed.values, 'o-', label='Observed replica energies')
    ax.plot(orders, expected, 's--', label=r'Expected $E_D + n\hbar\omega$')
    for x, y_obs, y_exp in zip(orders, observed.values, expected):
        ax.vlines(x, y_exp, y_obs, color='tab:red', alpha=0.6)
        ax.text(x + 0.03, y_obs, f"Δ={y_obs - y_exp:+.2e} eV", fontsize=10)
    ax.set_xlabel('Replica order n')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Replica energies align with the 5 μm pump photon energy')
    ax.legend(frameon=True)
    fig.savefig(IMAGES / 'replica_energy_alignment.png')
    plt.close(fig)


def plot_polarization_dependence(pol, pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].plot(pol['angle_degrees'], pol['intensity'], 'o', ms=8, label='Measured replica intensity')
    dense_deg = np.linspace(pol['angle_degrees'].min(), pol['angle_degrees'].max(), 361)
    dense_rad = np.deg2rad(dense_deg)
    dense_pred = cosine2(dense_rad, *curve_fit(
        cosine2,
        pol['angle_radians'].values,
        pol['intensity'].values,
        p0=[pol['intensity'].mean(), (pol['intensity'].max() - pol['intensity'].min()) / 2, 0.0],
    )[0])
    axes[0].plot(dense_deg, dense_pred, '-', label=r'Fit: $I(\theta)=a_0+a_1\cos 2(\theta-\phi)$')
    axes[0].set_xlabel('Pump polarization angle (deg)')
    axes[0].set_ylabel('Replica intensity (a.u.)')
    axes[0].set_title('Polarization dependence of replica-band intensity')
    axes[0].legend(frameon=True)

    polar_ax = fig.add_subplot(1, 2, 2, projection='polar')
    polar_ax.plot(pol['angle_radians'], pol['intensity'], 'o-', label='Measured')
    polar_ax.plot(dense_rad, dense_pred, '--', label='Fit')
    polar_ax.set_title('Polar representation')
    polar_ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))

    fig.savefig(IMAGES / 'polarization_dependence_fit.png')
    plt.close(fig)


def plot_angle_comparison(raw, processed):
    energy = raw['energy_axis']
    kx = raw['kx_axis']
    probe_angles = [0, 30, 90, 150]
    dirac_kx, dirac_energy = processed['dirac_point']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, angle in zip(axes, probe_angles):
        spec = raw['pump_on'][angle] - raw['pump_off_spectrum']
        im = ax.imshow(
            spec,
            aspect='auto',
            origin='lower',
            extent=[kx.min(), kx.max(), energy.min(), energy.max()],
            cmap='coolwarm',
        )
        ax.scatter([dirac_kx], [dirac_energy], c='black', s=30)
        ax.set_title(f'Pump-induced change at {angle}°')
        ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
        ax.set_ylabel('Energy (eV)')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(IMAGES / 'angle_resolved_difference_maps.png')
    plt.close(fig)


def main():
    raw, processed, pol = load_data()
    summary, dispersion, pred = summarize_data(raw, processed, pol)

    with open(OUTPUTS / 'data_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary['raw_data'], fh, indent=2)
    with open(OUTPUTS / 'analysis_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    dispersion.to_csv(OUTPUTS / 'band_dispersion_table.csv', index=False)
    pol.assign(fit_prediction=pred).to_csv(OUTPUTS / 'polarization_fit_table.csv', index=False)

    plot_raw_spectra(raw, processed)
    plot_dispersion_and_replicas(processed)
    plot_replica_energy_alignment(processed)
    plot_polarization_dependence(pol, pred)
    plot_angle_comparison(raw, processed)


if __name__ == '__main__':
    main()
