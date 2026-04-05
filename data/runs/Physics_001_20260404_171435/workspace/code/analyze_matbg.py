import ast
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('.')
DATA_PATH = ROOT / 'data' / 'MATBG Superfluid Stiffness Core Dataset.txt'
OUTPUT_DIR = ROOT / 'outputs'
IMAGE_DIR = ROOT / 'report' / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def parse_dataset(path: Path):
    text = path.read_text()
    sections = {}
    current_file = None
    i = 0
    lines = text.splitlines()
    while i < len(lines):
        line = lines[i].strip()
        file_match = re.match(r'\*\*File\s+(\d+):\s+(.+?)\*\*', line)
        if file_match:
            current_file = file_match.group(2)
            sections[current_file] = {'parameters': {}, 'arrays': {}}
            i += 1
            continue
        if current_file is None:
            i += 1
            continue
        param_match = re.match(r'([A-Za-z0-9_]+)\s*=\s*(.+)', line)
        if param_match and not line.startswith('**'):
            key, value = param_match.groups()
            try:
                sections[current_file]['parameters'][key] = ast.literal_eval(value)
            except Exception:
                sections[current_file]['parameters'][key] = value
            i += 1
            continue
        label_match = re.match(r'\*\*(.+?):\*\*', line)
        if label_match:
            label = label_match.group(1)
            array_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    break
                if next_line.startswith('**') and not next_line.startswith('**Fixed Parameters'):
                    break
                if re.match(r'[A-Za-z0-9_]+\s*=\s*', next_line) and not next_line.startswith('['):
                    break
                array_lines.append(next_line)
                if next_line.endswith(']'):
                    break
                j += 1
            joined = ' '.join(array_lines).strip()
            if joined.startswith('[') and joined.endswith(']'):
                arr = np.fromstring(joined.strip('[]'), sep=' ')
                sections[current_file]['arrays'][label] = arr
                i = j + 1
                continue
        i += 1
    return sections


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def fit_power_law_temperature(T, y, Tc, powers=(2.0, 2.5, 3.0)):
    results = {}
    for n in powers:
        pred = np.maximum(100.0 * (1 - (T / Tc) ** n), 0)
        results[str(n)] = {'rmse': rmse(y, pred), 'r2': r2_score(y, pred)}
    return results


def fit_low_current_quadratic(I, y, current_max=30.0):
    mask = I <= current_max
    x = I[mask] ** 2
    coeffs = np.polyfit(x, y[mask], deg=1)
    pred = coeffs[0] * x + coeffs[1]
    return {
        'current_max_nA': current_max,
        'slope_vs_I2': float(coeffs[0]),
        'intercept': float(coeffs[1]),
        'rmse': rmse(y[mask], pred),
        'r2': r2_score(y[mask], pred),
    }


def fit_low_current_linear(I, y, current_max=30.0):
    mask = I <= current_max
    coeffs = np.polyfit(I[mask], y[mask], deg=1)
    pred = np.polyval(coeffs, I[mask])
    return {
        'current_max_nA': current_max,
        'slope': float(coeffs[0]),
        'intercept': float(coeffs[1]),
        'rmse': rmse(y[mask], pred),
        'r2': r2_score(y[mask], pred),
    }


def align_to_min_length(named_arrays):
    min_len = min(len(v) for v in named_arrays.values())
    return {k: np.asarray(v)[:min_len] for k, v in named_arrays.items()}, min_len


def main():
    sections = parse_dataset(DATA_PATH)

    carrier = sections['superfluid_stiffness_measurement.py']
    temp = sections['temperature_dependence.py']
    curr = sections['current_dependence.py']

    carrier_df = pd.DataFrame({
        'n_eff_m^-2': carrier['arrays']['Carrier Density Data (n_eff in m^-2)'],
        'D_s_conv': carrier['arrays']['Conventional Superfluid Stiffness (D_s_conv)'],
        'D_s_geom': carrier['arrays']['Quantum Geometric Superfluid Stiffness (D_s_geom)'],
        'D_s_exp_hole': carrier['arrays']['Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole)'],
        'D_s_exp_electron': carrier['arrays']['Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron)'],
    })
    carrier_df['enhancement_hole_vs_conv'] = carrier_df['D_s_exp_hole'] / carrier_df['D_s_conv']
    carrier_df['enhancement_electron_vs_conv'] = carrier_df['D_s_exp_electron'] / carrier_df['D_s_conv']
    carrier_df['enhancement_hole_vs_geom'] = carrier_df['D_s_exp_hole'] / carrier_df['D_s_geom']
    carrier_df['enhancement_electron_vs_geom'] = carrier_df['D_s_exp_electron'] / carrier_df['D_s_geom']
    carrier_df['hole_electron_ratio'] = carrier_df['D_s_exp_hole'] / carrier_df['D_s_exp_electron']
    carrier_df.to_csv(OUTPUT_DIR / 'carrier_density_data.csv', index=False)

    temp_arrays, temp_min_len = align_to_min_length({
        'T_K': temp['arrays']['Temperature Array (T in K)'],
        'D_s_bcs': temp['arrays']['BCS Model Data (D_s_bcs)'],
        'D_s_nodal': temp['arrays']['Nodal Superconductor Data (D_s_nodal)'],
        'D_s_power_n2': temp['arrays']['Power Law n=2.0 Data (D_s_power_n2)'],
        'D_s_power_n2_5': temp['arrays']['Power Law n=2.5 Data (D_s_power_n2_5)'],
        'D_s_power_n3': temp['arrays']['Power Law n=3.0 Data (D_s_power_n3)'],
        'D_s_experimental': temp['arrays']['Experimental Data with Noise (D_s_experimental)'],
    })
    temp_df = pd.DataFrame(temp_arrays)
    temp_df.to_csv(OUTPUT_DIR / 'temperature_data.csv', index=False)

    current_dc = curr['arrays']['DC Current Array (I_dc in nA)']
    D_s_dc = curr['arrays']['Experimental DC Data (D_s_dc_exp)']
    gl_model = curr['arrays']['Ginzburg-Landau Model (D_s_gl)']
    linear_model = curr['arrays']['Linear Meissner Model (D_s_linear)']
    P_mw = curr['arrays']['Microwave Power Array (P_mw normalized)']
    I_mw = curr['arrays']['Microwave Current Amplitude (I_mw_amplitude in nA)']
    D_s_mw = curr['arrays']['Experimental Microwave Data (D_s_mw_exp)']

    dc_arrays, dc_min_len = align_to_min_length({
        'I_dc_nA': current_dc,
        'D_s_gl': gl_model,
        'D_s_linear': linear_model,
        'D_s_dc_exp': D_s_dc,
    })
    dc_df = pd.DataFrame(dc_arrays)
    dc_df.to_csv(OUTPUT_DIR / 'current_data_dc.csv', index=False)

    mw_arrays, mw_min_len = align_to_min_length({
        'P_mw_norm': P_mw,
        'I_mw_amplitude_nA': I_mw,
        'D_s_mw_exp': D_s_mw,
    })
    mw_df = pd.DataFrame(mw_arrays)
    mw_df.to_csv(OUTPUT_DIR / 'current_data_mw.csv', index=False)

    density_summary = {
        'mean_hole_vs_conventional_enhancement': float(carrier_df['enhancement_hole_vs_conv'].mean()),
        'mean_electron_vs_conventional_enhancement': float(carrier_df['enhancement_electron_vs_conv'].mean()),
        'mean_hole_vs_geometric_enhancement': float(carrier_df['enhancement_hole_vs_geom'].mean()),
        'mean_electron_vs_geometric_enhancement': float(carrier_df['enhancement_electron_vs_geom'].mean()),
        'mean_hole_electron_ratio': float(carrier_df['hole_electron_ratio'].mean()),
        'max_hole_stiffness': float(carrier_df['D_s_exp_hole'].max()),
        'max_electron_stiffness': float(carrier_df['D_s_exp_electron'].max()),
        'carrier_density_at_max_hole': float(carrier_df.loc[carrier_df['D_s_exp_hole'].idxmax(), 'n_eff_m^-2']),
    }

    temp_model_metrics = {
        'bcs': {'rmse': rmse(temp_df['D_s_experimental'], temp_df['D_s_bcs']), 'r2': r2_score(temp_df['D_s_experimental'], temp_df['D_s_bcs'])},
        'nodal_linear': {'rmse': rmse(temp_df['D_s_experimental'], temp_df['D_s_nodal']), 'r2': r2_score(temp_df['D_s_experimental'], temp_df['D_s_nodal'])},
        'power_n2': {'rmse': rmse(temp_df['D_s_experimental'], temp_df['D_s_power_n2']), 'r2': r2_score(temp_df['D_s_experimental'], temp_df['D_s_power_n2'])},
        'power_n2_5': {'rmse': rmse(temp_df['D_s_experimental'], temp_df['D_s_power_n2_5']), 'r2': r2_score(temp_df['D_s_experimental'], temp_df['D_s_power_n2_5'])},
        'power_n3': {'rmse': rmse(temp_df['D_s_experimental'], temp_df['D_s_power_n3']), 'r2': r2_score(temp_df['D_s_experimental'], temp_df['D_s_power_n3'])},
    }
    best_temp_model = min(temp_model_metrics.items(), key=lambda kv: kv[1]['rmse'])[0]

    dc_metrics = {
        'gl_model': {'rmse': rmse(dc_df['D_s_dc_exp'], dc_df['D_s_gl']), 'r2': r2_score(dc_df['D_s_dc_exp'], dc_df['D_s_gl'])},
        'linear_model': {'rmse': rmse(dc_df['D_s_dc_exp'], dc_df['D_s_linear']), 'r2': r2_score(dc_df['D_s_dc_exp'], dc_df['D_s_linear'])},
        'low_current_quadratic_fit': fit_low_current_quadratic(dc_df['I_dc_nA'].values, dc_df['D_s_dc_exp'].values, current_max=30.0),
        'low_current_linear_fit': fit_low_current_linear(dc_df['I_dc_nA'].values, dc_df['D_s_dc_exp'].values, current_max=30.0),
        'minimum_dc_stiffness': float(dc_df['D_s_dc_exp'].min()),
        'current_at_minimum_dc_stiffness': float(dc_df.loc[dc_df['D_s_dc_exp'].idxmin(), 'I_dc_nA']),
    }

    mw_linear_fit = fit_low_current_linear(mw_df['I_mw_amplitude_nA'].values, mw_df['D_s_mw_exp'].values, current_max=15.0)
    mw_quadratic_fit = fit_low_current_quadratic(mw_df['I_mw_amplitude_nA'].values, mw_df['D_s_mw_exp'].values, current_max=15.0)
    mw_metrics = {
        'monotonic_decrease': bool(np.all(np.diff(mw_df['D_s_mw_exp']) <= 1e-9)),
        'low_current_linear_fit': mw_linear_fit,
        'low_current_quadratic_fit': mw_quadratic_fit,
        'fractional_drop_full_range': float((mw_df['D_s_mw_exp'].iloc[0] - mw_df['D_s_mw_exp'].iloc[-1]) / mw_df['D_s_mw_exp'].iloc[0]),
    }

    summary = {
        'dataset_checks': {
            'carrier_rows': int(len(carrier_df)),
            'temperature_rows': int(len(temp_df)),
            'temperature_common_aligned_length': int(temp_min_len),
            'temperature_original_lengths': {k: int(len(v)) for k, v in {
                'T_K': temp['arrays']['Temperature Array (T in K)'],
                'D_s_bcs': temp['arrays']['BCS Model Data (D_s_bcs)'],
                'D_s_nodal': temp['arrays']['Nodal Superconductor Data (D_s_nodal)'],
                'D_s_power_n2': temp['arrays']['Power Law n=2.0 Data (D_s_power_n2)'],
                'D_s_power_n2_5': temp['arrays']['Power Law n=2.5 Data (D_s_power_n2_5)'],
                'D_s_power_n3': temp['arrays']['Power Law n=3.0 Data (D_s_power_n3)'],
                'D_s_experimental': temp['arrays']['Experimental Data with Noise (D_s_experimental)'],
            }.items()},
            'dc_current_rows': int(len(dc_df)),
            'dc_common_aligned_length': int(dc_min_len),
            'dc_original_lengths': {'I_dc_nA': int(len(current_dc)), 'D_s_gl': int(len(gl_model)), 'D_s_linear': int(len(linear_model)), 'D_s_dc_exp': int(len(D_s_dc))},
            'mw_rows': int(len(mw_df)),
            'mw_common_aligned_length': int(mw_min_len),
            'mw_original_lengths': {'P_mw_norm': int(len(P_mw)), 'I_mw_amplitude_nA': int(len(I_mw)), 'D_s_mw_exp': int(len(D_s_mw))},
        },
        'density_dependence': density_summary,
        'temperature_dependence': {
            'model_metrics': temp_model_metrics,
            'best_model_by_rmse': best_temp_model,
        },
        'current_dependence': {
            'dc': dc_metrics,
            'microwave': mw_metrics,
        },
    }

    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    metrics_df = pd.DataFrame([
        {'experiment': 'temperature', 'model': model, **vals}
        for model, vals in temp_model_metrics.items()
    ] + [
        {'experiment': 'dc_current', 'model': 'gl_model', **dc_metrics['gl_model']},
        {'experiment': 'dc_current', 'model': 'linear_model', **dc_metrics['linear_model']},
        {'experiment': 'mw_current', 'model': 'linear_low_current', 'rmse': mw_linear_fit['rmse'], 'r2': mw_linear_fit['r2']},
        {'experiment': 'mw_current', 'model': 'quadratic_low_current', 'rmse': mw_quadratic_fit['rmse'], 'r2': mw_quadratic_fit['r2']},
    ])
    metrics_df.to_csv(OUTPUT_DIR / 'model_comparison_metrics.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['D_s_conv'] / 1e9, label='Conventional theory', lw=2.5)
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['D_s_geom'] / 1e9, label='Quantum geometric theory', lw=2.5)
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['D_s_exp_hole'] / 1e9, label='Experiment (hole)', lw=3)
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['D_s_exp_electron'] / 1e9, label='Experiment (electron)', lw=3)
    ax.set_xlabel(r'Effective carrier density $n_{eff}$ ($10^{15}$ m$^{-2}$)')
    ax.set_ylabel(r'Superfluid stiffness $D_s$ ($10^9$ arb. units)')
    ax.set_title('Carrier-density dependence of MATBG superfluid stiffness')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'density_dependence.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['enhancement_hole_vs_conv'], label='Hole / conventional', lw=2.5)
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['enhancement_hole_vs_geom'], label='Hole / geometric', lw=2.5)
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['enhancement_electron_vs_conv'], label='Electron / conventional', lw=2.5)
    ax.plot(carrier_df['n_eff_m^-2'] / 1e15, carrier_df['enhancement_electron_vs_geom'], label='Electron / geometric', lw=2.5)
    ax.set_xlabel(r'Effective carrier density $n_{eff}$ ($10^{15}$ m$^{-2}$)')
    ax.set_ylabel('Enhancement factor')
    ax.set_title('Enhancement of experimental stiffness relative to theory')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'density_enhancement.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_df['T_K'], temp_df['D_s_experimental'], label='Experimental', lw=3, color='black')
    ax.plot(temp_df['T_K'], temp_df['D_s_bcs'], label='BCS-like', lw=2)
    ax.plot(temp_df['T_K'], temp_df['D_s_power_n2'], label='Power law n=2', lw=2)
    ax.plot(temp_df['T_K'], temp_df['D_s_power_n2_5'], label='Power law n=2.5', lw=2)
    ax.plot(temp_df['T_K'], temp_df['D_s_power_n3'], label='Power law n=3', lw=2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Normalized superfluid stiffness')
    ax.set_title('Temperature dependence favors a power-law stiffness decay')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'temperature_dependence.png')
    plt.close(fig)

    residual_df = pd.DataFrame({
        'T_K': temp_df['T_K'],
        'BCS-like': temp_df['D_s_experimental'] - temp_df['D_s_bcs'],
        'Power law n=2.5': temp_df['D_s_experimental'] - temp_df['D_s_power_n2_5'],
        'Power law n=3': temp_df['D_s_experimental'] - temp_df['D_s_power_n3'],
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in residual_df.columns[1:]:
        ax.plot(residual_df['T_K'], residual_df[col], label=col, lw=2)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Residual (experiment - model)')
    ax.set_title('Residual comparison for temperature models')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'temperature_residuals.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dc_df['I_dc_nA'], dc_df['D_s_dc_exp'], label='Experimental DC', lw=3, color='black')
    ax.plot(dc_df['I_dc_nA'], dc_df['D_s_gl'], label='GL quadratic suppression', lw=2.5)
    ax.plot(dc_df['I_dc_nA'], dc_df['D_s_linear'], label='Linear model', lw=2.5)
    ax.set_xlabel('DC bias current (nA)')
    ax.set_ylabel('Normalized superfluid stiffness')
    ax.set_title('Current-driven suppression under DC bias')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'dc_current_dependence.png')
    plt.close(fig)

    mask = dc_df['I_dc_nA'] <= 30
    xfit = np.linspace(0, 30, 200)
    quad = dc_metrics['low_current_quadratic_fit']['slope_vs_I2'] * xfit**2 + dc_metrics['low_current_quadratic_fit']['intercept']
    lin = dc_metrics['low_current_linear_fit']['slope'] * xfit + dc_metrics['low_current_linear_fit']['intercept']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dc_df.loc[mask, 'I_dc_nA'], dc_df.loc[mask, 'D_s_dc_exp'], label='Experimental DC (I≤30 nA)', s=55, color='black')
    ax.plot(xfit, quad, label='Quadratic fit', lw=2.5)
    ax.plot(xfit, lin, label='Linear fit', lw=2.5)
    ax.set_xlabel('DC bias current (nA)')
    ax.set_ylabel('Normalized superfluid stiffness')
    ax.set_title('Low-current DC response is better described by a quadratic trend')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'dc_low_current_fit.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mw_df['I_mw_amplitude_nA'], mw_df['D_s_mw_exp'], lw=3, color='purple', label='Microwave response')
    ax.set_xlabel('Microwave current amplitude (nA)')
    ax.set_ylabel('Normalized superfluid stiffness')
    ax.set_title('Microwave probing causes a weaker monotonic stiffness suppression')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'microwave_current_dependence.png')
    plt.close(fig)

    overview = pd.DataFrame({
        'carrier_density_points': [len(carrier_df)],
        'temperature_points': [len(temp_df)],
        'dc_current_points': [len(dc_df)],
        'mw_points': [len(mw_df)],
        'carrier_density_range_1e15_m^-2': [carrier_df['n_eff_m^-2'].max() / 1e15 - carrier_df['n_eff_m^-2'].min() / 1e15],
        'temperature_range_K': [temp_df['T_K'].max() - temp_df['T_K'].min()],
        'dc_current_range_nA': [dc_df['I_dc_nA'].max() - dc_df['I_dc_nA'].min()],
        'mw_current_range_nA': [mw_df['I_mw_amplitude_nA'].max() - mw_df['I_mw_amplitude_nA'].min()],
    })
    overview.to_csv(OUTPUT_DIR / 'dataset_overview.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    axes[0].hist(carrier_df['n_eff_m^-2'] / 1e15, bins=12, color='steelblue')
    axes[0].set_title('Carrier density coverage')
    axes[0].set_xlabel(r'$n_{eff}$ ($10^{15}$ m$^{-2}$)')
    axes[1].hist(temp_df['T_K'], bins=12, color='darkorange')
    axes[1].set_title('Temperature coverage')
    axes[1].set_xlabel('T (K)')
    axes[2].hist(dc_df['I_dc_nA'], bins=12, color='seagreen')
    axes[2].set_title('DC current coverage')
    axes[2].set_xlabel('I_dc (nA)')
    axes[3].hist(mw_df['I_mw_amplitude_nA'], bins=12, color='mediumpurple')
    axes[3].set_title('Microwave current coverage')
    axes[3].set_xlabel('I_mw (nA)')
    for ax in axes:
        ax.set_ylabel('Count')
    fig.suptitle('Dataset overview', y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'dataset_overview.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
