#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'MATBG Superfluid Stiffness Core Dataset.txt'
OUTPUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'report' / 'images'


plt.style.use('seaborn-v0_8-whitegrid')


SECTION_RE = re.compile(r"^\*\*(.+?):\*\*$")
ARRAY_RE = re.compile(r"\[.*?\]", re.S)
SCALAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_numeric_array(text: str) -> np.ndarray:
    cleaned = text.strip().strip('[]')
    if not cleaned:
        return np.array([], dtype=float)
    return np.fromstring(cleaned.replace('\n', ' '), sep=' ')


def parse_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = path.read_text(encoding='utf-8')
    blocks = re.split(r"\n(?=\*\*File \d+:)", raw.strip())
    parsed: Dict[str, Dict[str, Any]] = {}

    for block in blocks:
        file_match = re.search(r"\*\*(File \d+: .*?)\*\*", block)
        if not file_match:
            continue
        file_key = file_match.group(1)
        parsed[file_key] = {'scalars': {}, 'arrays': {}}

        lines = block.splitlines()
        current_section = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            sec_match = SECTION_RE.match(line)
            if sec_match:
                current_section = sec_match.group(1)
                if current_section == 'Fixed Parameters':
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if not next_line:
                            i += 1
                            continue
                        if SECTION_RE.match(next_line):
                            i -= 1
                            break
                        scalar_match = SCALAR_RE.match(next_line)
                        if scalar_match:
                            key, value = scalar_match.groups()
                            try:
                                parsed[file_key]['scalars'][key] = ast.literal_eval(value)
                            except Exception:
                                parsed[file_key]['scalars'][key] = value
                        i += 1
                else:
                    chunk_lines = []
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        if SECTION_RE.match(next_line.strip()):
                            i -= 1
                            break
                        if next_line.strip():
                            chunk_lines.append(next_line)
                        i += 1
                    chunk_text = '\n'.join(chunk_lines)
                    arr_match = ARRAY_RE.search(chunk_text)
                    if arr_match:
                        parsed[file_key]['arrays'][current_section] = parse_numeric_array(arr_match.group(0))
            i += 1

    return parsed


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def save_csv(path: Path, header: str, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=',', header=header, comments='')


def common_prefix_length(*arrays: np.ndarray) -> int:
    return int(min(len(np.asarray(arr)) for arr in arrays))


def reconstruct_resistance_from_stiffness(ds: np.ndarray, scale_ohm: float = 1000.0) -> np.ndarray:
    ds = np.asarray(ds)
    ds_norm = ds / np.nanmax(ds)
    return scale_ohm / np.clip(ds_norm, 1e-6, None)


def reconstruct_frequency_from_stiffness(ds: np.ndarray, f0_ghz: float = 6.0) -> np.ndarray:
    ds = np.asarray(ds)
    ds_norm = ds / np.nanmax(ds)
    return f0_ghz * np.sqrt(np.clip(ds_norm, 0.0, None))


def analyze_density(block: Dict[str, Any]) -> Dict[str, Any]:
    arrays = block['arrays']
    n = arrays['Carrier Density Data (n_eff in m^-2)']
    ds_conv = arrays['Conventional Superfluid Stiffness (D_s_conv)']
    ds_geom = arrays['Quantum Geometric Superfluid Stiffness (D_s_geom)']
    ds_hole = arrays['Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole)']
    ds_elec = arrays['Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron)']
    ds_exp_avg = 0.5 * (ds_hole + ds_elec)

    ratio_vs_conv = ds_exp_avg / ds_conv
    ratio_vs_geom = ds_exp_avg / ds_geom
    asymmetry = (ds_hole - ds_elec) / ds_exp_avg

    peak_idx = int(np.argmax(ds_exp_avg))

    density_table = np.column_stack([
        n, ds_conv, ds_geom, ds_hole, ds_elec, ds_exp_avg,
        ratio_vs_conv, ratio_vs_geom, asymmetry,
        reconstruct_resistance_from_stiffness(ds_exp_avg),
        reconstruct_frequency_from_stiffness(ds_exp_avg),
    ])
    save_csv(
        OUTPUT_DIR / 'density_analysis.csv',
        'n_eff_m^-2,D_s_conv,D_s_geom,D_s_exp_hole,D_s_exp_electron,D_s_exp_avg,ratio_exp_to_conv,ratio_exp_to_geom,hole_electron_asymmetry,resistance_proxy_ohm,resonance_freq_proxy_GHz',
        density_table,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    n_1e12 = n / 1e12
    axes[0].plot(n_1e12, ds_conv / 1e9, label='Conventional FL', lw=2)
    axes[0].plot(n_1e12, ds_geom / 1e9, label='Quantum geometric', lw=2)
    axes[0].plot(n_1e12, ds_hole / 1e9, label='Experiment (hole)', lw=2)
    axes[0].plot(n_1e12, ds_elec / 1e9, label='Experiment (electron)', lw=2)
    axes[0].set_xlabel('Carrier density n_eff (1e12 m^-2)')
    axes[0].set_ylabel('D_s (1e9 arb. units)')
    axes[0].set_title('Carrier-density dependence of superfluid stiffness')
    axes[0].legend(frameon=True)

    axes[1].plot(n_1e12, ratio_vs_conv, label='Experimental / conventional', lw=2)
    axes[1].plot(n_1e12, ratio_vs_geom, label='Experimental / geometric', lw=2)
    axes[1].axhline(1.0, color='black', ls='--', lw=1)
    axes[1].set_xlabel('Carrier density n_eff (1e12 m^-2)')
    axes[1].set_ylabel('Enhancement ratio')
    axes[1].set_title('Quantum-geometry benchmark ratios')
    axes[1].legend(frameon=True)
    fig.savefig(FIG_DIR / 'density_dependence.png', dpi=220)
    plt.close(fig)

    return {
        'n_points': int(n.size),
        'peak_density_m^-2': float(n[peak_idx]),
        'peak_experimental_avg_stiffness': float(ds_exp_avg[peak_idx]),
        'mean_exp_to_conventional_ratio': float(np.mean(ratio_vs_conv)),
        'mean_exp_to_geometric_ratio': float(np.mean(ratio_vs_geom)),
        'max_hole_electron_fractional_asymmetry': float(np.max(np.abs(asymmetry))),
    }


def fit_power_law_temperature(T: np.ndarray, Ds: np.ndarray, Tc: float, Ds0: float) -> Dict[str, float]:
    mask = (T > 0) & (T < 0.95 * Tc) & (Ds < Ds0)
    x = np.log(T[mask] / Tc)
    y = np.log(np.clip(1 - Ds[mask] / Ds0, 1e-10, None))
    slope, intercept = np.polyfit(x, y, 1)
    n_fit = float(slope)
    prefactor = float(np.exp(intercept))
    Ds_fit = Ds0 * (1 - prefactor * (np.clip(T / Tc, 0, None) ** n_fit))
    Ds_fit[T >= Tc] = 0.0
    Ds_fit = np.clip(Ds_fit, 0.0, Ds0)
    return {
        'n_fit': n_fit,
        'prefactor': prefactor,
        'rmse_full_range': rmse(Ds, Ds_fit),
        'r2_full_range': r2_score(Ds, Ds_fit),
    }


def analyze_temperature(block: Dict[str, Any]) -> Dict[str, Any]:
    scalars = block['scalars']
    arrays = block['arrays']
    T = arrays['Temperature Array (T in K)']
    ds_bcs = arrays['BCS Model Data (D_s_bcs)']
    ds_nodal = arrays['Nodal Superconductor Data (D_s_nodal)']
    ds_n2 = arrays['Power Law n=2.0 Data (D_s_power_n2)']
    ds_n25 = arrays['Power Law n=2.5 Data (D_s_power_n2_5)']
    ds_n3 = arrays['Power Law n=3.0 Data (D_s_power_n3)']
    ds_exp = arrays['Experimental Data with Noise (D_s_experimental)']
    Tc = float(scalars['T_c'])
    Ds0 = float(scalars['D_s0'])

    n_common = common_prefix_length(T, ds_bcs, ds_nodal, ds_n2, ds_n25, ds_n3, ds_exp)
    T = T[:n_common]
    ds_bcs = ds_bcs[:n_common]
    ds_nodal = ds_nodal[:n_common]
    ds_n2 = ds_n2[:n_common]
    ds_n25 = ds_n25[:n_common]
    ds_n3 = ds_n3[:n_common]
    ds_exp = ds_exp[:n_common]

    candidate_models = {
        'BCS_like_n2': ds_bcs,
        'nodal_linear': ds_nodal,
        'power_n2': ds_n2,
        'power_n2.5': ds_n25,
        'power_n3': ds_n3,
    }
    model_scores = {
        name: {
            'rmse': rmse(ds_exp, arr),
            'r2': r2_score(ds_exp, arr),
        }
        for name, arr in candidate_models.items()
    }
    best_model = min(model_scores.items(), key=lambda kv: kv[1]['rmse'])[0]
    fitted = fit_power_law_temperature(T, ds_exp, Tc, Ds0)

    temperature_table = np.column_stack([
        T, ds_exp, ds_bcs, ds_nodal, ds_n2, ds_n25, ds_n3,
        reconstruct_resistance_from_stiffness(ds_exp, scale_ohm=2000.0),
        reconstruct_frequency_from_stiffness(ds_exp, f0_ghz=6.5),
    ])
    save_csv(
        OUTPUT_DIR / 'temperature_analysis.csv',
        'temperature_K,D_s_experimental,D_s_bcs,D_s_nodal,D_s_power_n2,D_s_power_n2_5,D_s_power_n3,resistance_proxy_ohm,resonance_freq_proxy_GHz',
        temperature_table,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    axes[0].plot(T, ds_exp, label='Experimental', color='black', lw=2)
    axes[0].plot(T, ds_bcs, label='BCS / n=2', lw=1.8)
    axes[0].plot(T, ds_nodal, label='Nodal linear', lw=1.8)
    axes[0].plot(T, ds_n25, label='Power law n=2.5', lw=1.8)
    axes[0].plot(T, ds_n3, label='Power law n=3', lw=1.8)
    axes[0].axvline(Tc, color='gray', ls='--', lw=1)
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Superfluid stiffness (arb. units)')
    axes[0].set_title('Temperature dependence and model comparison')
    axes[0].legend(frameon=True)

    low_t = (T > 0) & (T < 0.9 * Tc)
    x = np.log(T[low_t] / Tc)
    y = np.log(np.clip(1 - ds_exp[low_t] / Ds0, 1e-8, None))
    slope, intercept = np.polyfit(x, y, 1)
    axes[1].scatter(x, y, s=18, color='black', alpha=0.7, label='Experimental transform')
    x_line = np.linspace(x.min(), x.max(), 200)
    axes[1].plot(x_line, intercept + slope * x_line, color='tab:red', lw=2, label=f'Fit slope n={slope:.2f}')
    axes[1].set_xlabel('log(T/T_c)')
    axes[1].set_ylabel('log(1 - D_s/D_s0)')
    axes[1].set_title('Power-law exponent extraction')
    axes[1].legend(frameon=True)
    fig.savefig(FIG_DIR / 'temperature_dependence.png', dpi=220)
    plt.close(fig)

    return {
        'Tc_K': Tc,
        'Ds0': Ds0,
        'best_reference_model': best_model,
        'reference_model_scores': model_scores,
        'fitted_power_law': fitted,
    }


def analyze_current(block: Dict[str, Any]) -> Dict[str, Any]:
    scalars = block['scalars']
    arrays = block['arrays']
    I_dc = arrays['DC Current Array (I_dc in nA)']
    ds_gl = arrays['Ginzburg-Landau Model (D_s_gl)']
    ds_linear = arrays['Linear Meissner Model (D_s_linear)']
    ds_dc_exp = arrays['Experimental DC Data (D_s_dc_exp)']
    P_mw = arrays['Microwave Power Array (P_mw normalized)']
    I_mw = arrays['Microwave Current Amplitude (I_mw_amplitude in nA)']
    ds_mw_exp = arrays['Experimental Microwave Data (D_s_mw_exp)']
    Ic = float(scalars['I_c'])
    Ds0 = float(scalars['D_s0'])

    n_dc_common = common_prefix_length(I_dc, ds_gl, ds_linear, ds_dc_exp)
    I_dc = I_dc[:n_dc_common]
    ds_gl = ds_gl[:n_dc_common]
    ds_linear = ds_linear[:n_dc_common]
    ds_dc_exp = ds_dc_exp[:n_dc_common]

    n_mw_common = common_prefix_length(P_mw, I_mw, ds_mw_exp)
    P_mw = P_mw[:n_mw_common]
    I_mw = I_mw[:n_mw_common]
    ds_mw_exp = ds_mw_exp[:n_mw_common]

    low_current_mask = I_dc <= 0.7 * Ic
    quad_coeff = np.polyfit(I_dc[low_current_mask] ** 2, Ds0 - ds_dc_exp[low_current_mask], 1)
    lin_coeff = np.polyfit(I_dc[low_current_mask], Ds0 - ds_dc_exp[low_current_mask], 1)
    dc_quad_pred = Ds0 - np.polyval(quad_coeff, I_dc ** 2)
    dc_lin_pred = Ds0 - np.polyval(lin_coeff, I_dc)

    mw_quad_coeff = np.polyfit(I_mw ** 2, Ds0 - ds_mw_exp, 1)
    mw_quad_pred = Ds0 - np.polyval(mw_quad_coeff, I_mw ** 2)

    current_table = np.column_stack([
        I_dc, ds_dc_exp, ds_gl, ds_linear,
        reconstruct_resistance_from_stiffness(np.clip(ds_dc_exp, 1e-3, None), scale_ohm=1500.0),
        reconstruct_frequency_from_stiffness(np.clip(ds_dc_exp, 1e-3, None), f0_ghz=5.8),
    ])
    save_csv(
        OUTPUT_DIR / 'current_dc_analysis.csv',
        'I_dc_nA,D_s_dc_experimental,D_s_gl,D_s_linear,resistance_proxy_ohm,resonance_freq_proxy_GHz',
        current_table,
    )

    mw_table = np.column_stack([
        P_mw, I_mw, ds_mw_exp,
        reconstruct_frequency_from_stiffness(ds_mw_exp, f0_ghz=5.8),
    ])
    save_csv(
        OUTPUT_DIR / 'current_microwave_analysis.csv',
        'P_mw_normalized,I_mw_amplitude_nA,D_s_mw_experimental,resonance_freq_proxy_GHz',
        mw_table,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    axes[0].plot(I_dc, ds_dc_exp, label='Experimental DC', color='black', lw=2)
    axes[0].plot(I_dc, ds_gl, label='GL model', lw=1.8)
    axes[0].plot(I_dc, ds_linear, label='Linear model', lw=1.8)
    axes[0].plot(I_dc, dc_quad_pred, label='Low-current quadratic fit', lw=1.8, ls='--')
    axes[0].axvline(Ic, color='gray', ls='--', lw=1)
    axes[0].set_xlabel('DC current (nA)')
    axes[0].set_ylabel('Superfluid stiffness (arb. units)')
    axes[0].set_title('DC current suppression of superfluid stiffness')
    axes[0].legend(frameon=True)

    axes[1].plot(I_mw, ds_mw_exp, label='Experimental microwave', color='tab:blue', lw=2)
    axes[1].plot(I_mw, mw_quad_pred, label='Quadratic fit vs I_mw^2', color='tab:red', ls='--', lw=2)
    axes[1].set_xlabel('Microwave current amplitude (nA)')
    axes[1].set_ylabel('Superfluid stiffness (arb. units)')
    axes[1].set_title('Microwave-current dependence')
    axes[1].legend(frameon=True)
    fig.savefig(FIG_DIR / 'current_dependence.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.scatter(I_dc ** 2, Ds0 - ds_dc_exp, s=24, label='DC suppression vs I^2', color='black')
    x_dc = np.linspace(0, np.max(I_dc ** 2), 200)
    ax.plot(x_dc, np.polyval(quad_coeff, x_dc), label='DC quadratic fit', lw=2)
    ax.scatter(I_mw ** 2, Ds0 - ds_mw_exp, s=24, label='Microwave suppression vs I_mw^2', color='tab:blue')
    x_mw = np.linspace(0, np.max(I_mw ** 2), 200)
    ax.plot(x_mw, np.polyval(mw_quad_coeff, x_mw), label='Microwave quadratic fit', lw=2, color='tab:red')
    ax.set_xlabel('Current squared (nA^2)')
    ax.set_ylabel('D_s0 - D_s')
    ax.set_title('Quadratic suppression consistency check')
    ax.legend(frameon=True)
    fig.savefig(FIG_DIR / 'quadratic_suppression_validation.png', dpi=220)
    plt.close(fig)

    return {
        'Ic_nA': Ic,
        'Ds0': Ds0,
        'dc_model_rmse': {
            'GL': rmse(ds_dc_exp, ds_gl),
            'linear': rmse(ds_dc_exp, ds_linear),
            'quadratic_low_current_fit': rmse(ds_dc_exp, dc_quad_pred),
            'linear_low_current_fit': rmse(ds_dc_exp, dc_lin_pred),
        },
        'dc_quadratic_fit_coefficients_for_Ds0_minus_Ds_vs_I2': [float(c) for c in quad_coeff],
        'mw_quadratic_fit_coefficients_for_Ds0_minus_Ds_vs_I2': [float(c) for c in mw_quad_coeff],
        'dc_quadratic_r2': r2_score(ds_dc_exp, dc_quad_pred),
        'mw_quadratic_r2': r2_score(ds_mw_exp, mw_quad_pred),
    }


def create_overview_figure(density: Dict[str, Any], temperature: Dict[str, Any], current: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = ['<Exp/Conv>\n(density)', '<Exp/Geom>\n(density)', 'n_fit\n(temp)', 'DC quad R^2', 'MW quad R^2']
    values = [
        density['mean_exp_to_conventional_ratio'],
        density['mean_exp_to_geometric_ratio'],
        temperature['fitted_power_law']['n_fit'],
        current['dc_quadratic_r2'],
        current['mw_quadratic_r2'],
    ]
    colors = ['tab:purple', 'tab:green', 'tab:orange', 'tab:blue', 'tab:red']
    ax.bar(labels, values, color=colors)
    ax.set_title('MATBG superfluid-stiffness analysis summary metrics')
    ax.set_ylabel('Metric value')
    fig.savefig(FIG_DIR / 'analysis_overview_metrics.png', dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    dataset = parse_dataset(DATA_FILE)

    density_summary = analyze_density(dataset['File 1: superfluid_stiffness_measurement.py'])
    temperature_summary = analyze_temperature(dataset['File 2: temperature_dependence.py'])
    current_summary = analyze_current(dataset['File 3: current_dependence.py'])
    create_overview_figure(density_summary, temperature_summary, current_summary)

    summary = {
        'task': 'MATBG superfluid stiffness analysis',
        'data_source': str(DATA_FILE.relative_to(ROOT)),
        'outputs_generated': [
            'outputs/density_analysis.csv',
            'outputs/temperature_analysis.csv',
            'outputs/current_dc_analysis.csv',
            'outputs/current_microwave_analysis.csv',
            'outputs/analysis_summary.json',
            'report/images/density_dependence.png',
            'report/images/temperature_dependence.png',
            'report/images/current_dependence.png',
            'report/images/quadratic_suppression_validation.png',
            'report/images/analysis_overview_metrics.png',
        ],
        'notes': {
            'reconstructed_observables': 'DC resistance and resonance frequency are reported as stiffness-derived proxies because the provided dataset directly contains superfluid-stiffness traces rather than raw transport/resonator readout channels.',
            'model_interpretation': 'Use the temperature and current fit metrics to assess unconventional power-law behavior and quadratic current suppression.',
        },
        'density_summary': density_summary,
        'temperature_summary': temperature_summary,
        'current_summary': current_summary,
    }
    (OUTPUT_DIR / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
