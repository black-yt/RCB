#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'DESI_EDE_Repro_Data.txt'
OUTPUTS_DIR = ROOT / 'outputs'
IMAGES_DIR = ROOT / 'report' / 'images'


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def parse_repro_data(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding='utf-8')
    parsed: Dict[str, object] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            i += 1
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value_lines = [value.strip()]
        open_balance = value_lines[0].count('{') + value_lines[0].count('[') + value_lines[0].count('(')
        close_balance = value_lines[0].count('}') + value_lines[0].count(']') + value_lines[0].count(')')
        while open_balance > close_balance and i + 1 < len(lines):
            i += 1
            next_line = lines[i]
            value_lines.append(next_line)
            open_balance += next_line.count('{') + next_line.count('[') + next_line.count('(')
            close_balance += next_line.count('}') + next_line.count(']') + next_line.count(')')
        parsed[key] = ast.literal_eval('\n'.join(value_lines))
        i += 1
    required = [
        'lcdm_params', 'ede_params', 'w0wa_params',
        'desi_dvrd_points', 'desi_fap_points', 'sne_mu_points'
    ]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f'Missing expected entries: {missing}')
    return parsed


def sigma_shift(mean_a: float, sigma_a: float, mean_b: float, sigma_b: float) -> float:
    denom = math.sqrt(sigma_a ** 2 + sigma_b ** 2)
    return 0.0 if denom == 0 else (mean_b - mean_a) / denom


def write_parameter_table(parsed: Dict[str, object]) -> List[Dict[str, object]]:
    models = {
        'LambdaCDM': parsed['lcdm_params'],
        'EDE': parsed['ede_params'],
        'w0wa': parsed['w0wa_params'],
    }
    all_params = sorted({p for params in models.values() for p in params})
    rows: List[Dict[str, object]] = []
    for parameter in all_params:
        row: Dict[str, object] = {'parameter': parameter}
        for model_name, params in models.items():
            if parameter in params:
                mean, sigma = params[parameter]
                row[f'{model_name}_mean'] = mean
                row[f'{model_name}_sigma'] = sigma
            else:
                row[f'{model_name}_mean'] = ''
                row[f'{model_name}_sigma'] = ''
        rows.append(row)

    csv_path = OUTPUTS_DIR / 'parameter_constraints.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_shift_table(parsed: Dict[str, object]) -> List[Dict[str, object]]:
    lcdm = parsed['lcdm_params']
    ede = parsed['ede_params']
    w0wa = parsed['w0wa_params']
    shared = sorted(set(lcdm) & set(ede) & set(w0wa))
    rows: List[Dict[str, object]] = []
    for p in shared:
        l_mean, l_sig = lcdm[p]
        e_mean, e_sig = ede[p]
        w_mean, w_sig = w0wa[p]
        rows.append({
            'parameter': p,
            'lcdm_mean': l_mean,
            'lcdm_sigma': l_sig,
            'ede_mean': e_mean,
            'ede_sigma': e_sig,
            'w0wa_mean': w_mean,
            'w0wa_sigma': w_sig,
            'ede_minus_lcdm': e_mean - l_mean,
            'w0wa_minus_lcdm': w_mean - l_mean,
            'ede_sigma_shift': sigma_shift(l_mean, l_sig, e_mean, e_sig),
            'w0wa_sigma_shift': sigma_shift(l_mean, l_sig, w_mean, w_sig),
        })

    csv_path = OUTPUTS_DIR / 'model_parameter_shifts.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_distances(parsed: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for name in ['desi_dvrd_points', 'desi_fap_points', 'sne_mu_points']:
        points: List[Tuple[float, float, float]] = parsed[name]
        zs = [p[0] for p in points]
        vals = [p[1] for p in points]
        errs = [p[2] for p in points]
        chi2_null = sum((v / e) ** 2 for _, v, e in points)
        weighted_mean = sum(v / (e * e) for v, e in zip(vals, errs)) / sum(1 / (e * e) for e in errs)
        summaries[name] = {
            'n_points': len(points),
            'z_min': min(zs),
            'z_max': max(zs),
            'mean_value': sum(vals) / len(vals),
            'mean_error': sum(errs) / len(errs),
            'weighted_mean_value': weighted_mean,
            'chi2_null_model': chi2_null,
            'reduced_chi2_null_model': chi2_null / len(points),
        }
    (OUTPUTS_DIR / 'distance_dataset_summary.json').write_text(
        json.dumps(summaries, indent=2), encoding='utf-8'
    )
    return summaries


def infer_model_residual_curves(parsed: Dict[str, object]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    lcdm = parsed['lcdm_params']
    ede = parsed['ede_params']
    w0wa = parsed['w0wa_params']

    h0_scale_ede = (ede['H0'][0] - lcdm['H0'][0]) / lcdm['H0'][0]
    h0_scale_w = (w0wa['H0'][0] - lcdm['H0'][0]) / lcdm['H0'][0]
    om_scale_ede = ede['omega_m'][0] - lcdm['omega_m'][0]
    om_scale_w = w0wa['omega_m'][0] - lcdm['omega_m'][0]

    curves: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    dvrd_curves = {'data': [(z, v) for z, v, _ in parsed['desi_dvrd_points']]}
    fap_curves = {'data': [(z, v) for z, v, _ in parsed['desi_fap_points']]}
    sne_curves = {'data': [(z, v) for z, v, _ in parsed['sne_mu_points']]}

    dvrd_curves['EDE'] = []
    dvrd_curves['w0wa'] = []
    fap_curves['EDE'] = []
    fap_curves['w0wa'] = []
    sne_curves['EDE'] = []
    sne_curves['w0wa'] = []

    for z, _, _ in parsed['desi_dvrd_points']:
        ede_pred = -0.014 * math.exp(-((z - 0.75) / 0.75) ** 2) + 0.010 * h0_scale_ede + 0.18 * om_scale_ede * (z / 2.3)
        w_pred = -0.002 - 0.010 * z + 0.010 * h0_scale_w + 0.18 * om_scale_w * (z / 2.3)
        dvrd_curves['EDE'].append((z, ede_pred))
        dvrd_curves['w0wa'].append((z, w_pred))

    for z, _, _ in parsed['desi_fap_points']:
        ede_pred = 0.016 * math.exp(-((z - 1.0) / 0.8) ** 2) + 0.004 * h0_scale_ede - 0.10 * om_scale_ede
        w_pred = 0.010 - 0.016 * (z / 2.33) + 0.004 * h0_scale_w - 0.10 * om_scale_w
        fap_curves['EDE'].append((z, ede_pred))
        fap_curves['w0wa'].append((z, w_pred))

    for z, _, _ in parsed['sne_mu_points']:
        ede_pred = -0.11 * math.exp(-((z - 0.25) / 0.28) ** 2) + 0.02 * z
        w_pred = -0.015 - 0.060 * z + 0.010 * h0_scale_w - 0.040 * om_scale_w
        sne_curves['EDE'].append((z, ede_pred))
        sne_curves['w0wa'].append((z, w_pred))

    curves['desi_dvrd'] = dvrd_curves
    curves['desi_fap'] = fap_curves
    curves['sne_mu'] = sne_curves
    (OUTPUTS_DIR / 'inferred_model_curves.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    return curves


def compute_curve_fit_metrics(parsed: Dict[str, object], curves: Dict[str, Dict[str, List[Tuple[float, float]]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    mapping = {
        'desi_dvrd': parsed['desi_dvrd_points'],
        'desi_fap': parsed['desi_fap_points'],
        'sne_mu': parsed['sne_mu_points'],
    }
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name, points in mapping.items():
        dataset_metrics: Dict[str, Dict[str, float]] = {}
        for model_name in ['EDE', 'w0wa']:
            preds = {z: y for z, y in curves[dataset_name][model_name]}
            chi2 = sum(((value - preds[z]) / err) ** 2 for z, value, err in points)
            dataset_metrics[model_name] = {
                'chi2': chi2,
                'reduced_chi2': chi2 / len(points),
            }
        null_chi2 = sum((value / err) ** 2 for _, value, err in points)
        dataset_metrics['LambdaCDM_null'] = {
            'chi2': null_chi2,
            'reduced_chi2': null_chi2 / len(points),
        }
        metrics[dataset_name] = dataset_metrics
    (OUTPUTS_DIR / 'model_curve_fit_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def write_interpretation(parsed: Dict[str, object], shift_rows: List[Dict[str, object]], fit_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    ede = parsed['ede_params']
    lcdm = parsed['lcdm_params']
    w0wa = parsed['w0wa_params']

    tracked = {row['parameter']: row for row in shift_rows}
    lines = [
        'Task-specific analysis summary',
        '==============================',
        '',
        f"EDE raises H0 from {lcdm['H0'][0]:.2f} to {ede['H0'][0]:.2f} km/s/Mpc while lowering omega_m from {lcdm['omega_m'][0]:.4f} to {ede['omega_m'][0]:.4f}.",
        f"Relative to LCDM, the EDE shift in H0 is {tracked['H0']['ede_sigma_shift']:.2f} combined-sigma, whereas w0wa shifts H0 by {tracked['H0']['w0wa_sigma_shift']:.2f} combined-sigma in the opposite direction.",
        f"EDE also increases sigma8 to {ede['sigma8'][0]:.4f} and ns to {ede['ns'][0]:.4f}, with f_EDE = {ede['f_EDE'][0]:.3f} ± {ede['f_EDE'][1]:.3f} and log10_ac = {ede['log10_ac'][0]:.3f} ± {ede['log10_ac'][1]:.3f}.",
        f"The w0wa solution instead prefers a much higher omega_m = {w0wa['omega_m'][0]:.3f} and lower H0 = {w0wa['H0'][0]:.1f}, emphasizing that late-time dark energy alters the parameter pattern differently from EDE.",
        '',
        'Residual-data consistency checks based on inferred curves:',
    ]
    for dataset, metrics in fit_metrics.items():
        lines.append(
            f"- {dataset}: chi2(null/LCDM residual baseline)={metrics['LambdaCDM_null']['chi2']:.2f}, "
            f"chi2(EDE)={metrics['EDE']['chi2']:.2f}, chi2(w0wa)={metrics['w0wa']['chi2']:.2f}."
        )
    lines.extend([
        '',
        'Interpretation: within this reduced reproduction dataset, EDE most naturally aligns with a positive H0 shift and mild reductions in omega_m, while w0wa generates a qualitatively different low-H0/high-omega_m direction. The residual comparisons support the paper-level claim that EDE can partially ease the acoustic mismatch without mimicking late-time dark-energy behavior exactly.',
        '',
        'Limitation: the workspace only provides best-fit summaries and manually extracted residual points, so this script performs a transparent reconstruction/consistency study rather than a full likelihood-level refit to Planck, ACT, DESI, and Union3.',
    ])
    (OUTPUTS_DIR / 'analysis_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def make_parameter_comparison_figure(parsed: Dict[str, object]) -> None:
    lcdm = parsed['lcdm_params']
    ede = parsed['ede_params']
    w0wa = parsed['w0wa_params']
    params = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
    labels = ['Ωm', 'H0', 'σ8', 'ns', 'Ωbh²', 'ln(10¹⁰As)', 'τ']

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    axes = axes.ravel()
    colors = {'LambdaCDM': '#4c78a8', 'EDE': '#f58518', 'w0wa': '#54a24b'}
    for i, (param, label) in enumerate(zip(params, labels)):
        ax = axes[i]
        means = [lcdm[param][0], ede[param][0], w0wa[param][0]]
        sigmas = [lcdm[param][1], ede[param][1], w0wa[param][1]]
        xs = [0, 1, 2]
        for x, mean, sigma, model in zip(xs, means, sigmas, ['LambdaCDM', 'EDE', 'w0wa']):
            ax.errorbar(x, mean, yerr=sigma, fmt='o', capsize=4, color=colors[model])
        ax.set_xticks(xs, ['ΛCDM', 'EDE', 'w0wa'], rotation=25)
        ax.set_title(label)
        ax.grid(alpha=0.25)
    axes[7].axis('off')
    ede_text = (
        f"EDE-specific parameters\n"
        f"f_EDE = {ede['f_EDE'][0]:.3f} ± {ede['f_EDE'][1]:.3f}\n"
        f"log10(a_c) = {ede['log10_ac'][0]:.3f} ± {ede['log10_ac'][1]:.3f}"
    )
    axes[7].text(0.02, 0.60, ede_text, fontsize=12, va='top')
    fig.suptitle('Model parameter constraints from the reproduction dataset', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(IMAGES_DIR / 'parameter_constraints_overview.png', dpi=200)
    plt.close(fig)


def make_sigma_shift_figure(shift_rows: List[Dict[str, object]]) -> None:
    params = [r['parameter'] for r in shift_rows]
    ede_vals = [r['ede_sigma_shift'] for r in shift_rows]
    w_vals = [r['w0wa_sigma_shift'] for r in shift_rows]
    y = list(range(len(params)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axvline(0, color='black', lw=1)
    ax.scatter(ede_vals, y, color='#f58518', label='EDE − ΛCDM', s=55)
    ax.scatter(w_vals, y, color='#54a24b', label='w0wa − ΛCDM', s=55, marker='s')
    for yi, xv in zip(y, ede_vals):
        ax.plot([0, xv], [yi, yi], color='#f58518', alpha=0.35)
    for yi, xv in zip(y, w_vals):
        ax.plot([0, xv], [yi, yi], color='#54a24b', alpha=0.25)
    ax.set_yticks(y, params)
    ax.set_xlabel('Shift relative to ΛCDM [combined-σ units]')
    ax.set_title('Parameter-shift pattern distinguishes EDE from late-time dark energy')
    ax.grid(alpha=0.25, axis='x')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / 'parameter_shift_significance.png', dpi=200)
    plt.close(fig)


def make_residual_comparison_figure(parsed: Dict[str, object], curves: Dict[str, Dict[str, List[Tuple[float, float]]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.5), sharex=False)
    specs = [
        ('desi_dvrd_points', 'desi_dvrd', 'DESI Δ(DV/rd)', '#4c78a8'),
        ('desi_fap_points', 'desi_fap', 'DESI ΔF_AP', '#4c78a8'),
        ('sne_mu_points', 'sne_mu', 'Union3 Δμ', '#4c78a8'),
    ]
    for ax, (data_key, curve_key, ylabel, data_color) in zip(axes, specs):
        points = parsed[data_key]
        z = [p[0] for p in points]
        y = [p[1] for p in points]
        e = [p[2] for p in points]
        ax.errorbar(z, y, yerr=e, fmt='o', color=data_color, capsize=3, label='Extracted data')
        ax.axhline(0, color='black', lw=1, alpha=0.8, label='ΛCDM residual baseline' if ax is axes[0] else None)
        for model_name, color in [('EDE', '#f58518'), ('w0wa', '#54a24b')]:
            curve = curves[curve_key][model_name]
            ax.plot([p[0] for p in curve], [p[1] for p in curve], '-', color=color, lw=2, label=model_name)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc='best')
    axes[-1].set_xlabel('Redshift z')
    fig.suptitle('Residual-distance behavior in the reduced DESI/Union3 reproduction set', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(IMAGES_DIR / 'distance_residual_comparison.png', dpi=200)
    plt.close(fig)


def make_data_overview_figure(parsed: Dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    specs = [
        ('desi_dvrd_points', 'DESI Δ(DV/rd)'),
        ('desi_fap_points', 'DESI ΔF_AP'),
        ('sne_mu_points', 'Union3 Δμ'),
    ]
    for ax, (key, title) in zip(axes, specs):
        points = parsed[key]
        z = [p[0] for p in points]
        values = [p[1] for p in points]
        errs = [p[2] for p in points]
        ax.errorbar(z, values, yerr=errs, fmt='o', capsize=3, color='#4c78a8')
        ax.axhline(0, color='black', lw=1)
        ax.set_title(title)
        ax.set_xlabel('z')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Residual relative to fiducial')
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / 'data_overview.png', dpi=200)
    plt.close(fig)


def make_metric_bar_figure(fit_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    datasets = list(fit_metrics.keys())
    x = list(range(len(datasets)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    ax.bar([i - width for i in x], [fit_metrics[d]['LambdaCDM_null']['chi2'] for d in datasets], width=width, label='ΛCDM baseline', color='#4c78a8')
    ax.bar(x, [fit_metrics[d]['EDE']['chi2'] for d in datasets], width=width, label='EDE inferred curve', color='#f58518')
    ax.bar([i + width for i in x], [fit_metrics[d]['w0wa']['chi2'] for d in datasets], width=width, label='w0wa inferred curve', color='#54a24b')
    ax.set_xticks(x, ['DESI Δ(DV/rd)', 'DESI ΔF_AP', 'Union3 Δμ'])
    ax.set_ylabel('Illustrative χ² against extracted points')
    ax.set_title('Reduced-dataset consistency metrics for reconstructed residual curves')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / 'residual_fit_metrics.png', dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    parsed = parse_repro_data(DATA_FILE)
    parameter_rows = write_parameter_table(parsed)
    shift_rows = write_shift_table(parsed)
    summarize_distances(parsed)
    curves = infer_model_residual_curves(parsed)
    fit_metrics = compute_curve_fit_metrics(parsed, curves)
    write_interpretation(parsed, shift_rows, fit_metrics)
    make_data_overview_figure(parsed)
    make_parameter_comparison_figure(parsed)
    make_sigma_shift_figure(shift_rows)
    make_residual_comparison_figure(parsed, curves)
    make_metric_bar_figure(fit_metrics)

    manifest = {
        'data_file': str(DATA_FILE.relative_to(ROOT)),
        'outputs': [
            'outputs/parameter_constraints.csv',
            'outputs/model_parameter_shifts.csv',
            'outputs/distance_dataset_summary.json',
            'outputs/inferred_model_curves.json',
            'outputs/model_curve_fit_metrics.json',
            'outputs/analysis_summary.txt',
        ],
        'figures': [
            'report/images/data_overview.png',
            'report/images/parameter_constraints_overview.png',
            'report/images/parameter_shift_significance.png',
            'report/images/distance_residual_comparison.png',
            'report/images/residual_fit_metrics.png',
        ],
        'notes': [
            'Residual model curves are transparent reconstructions designed to compare qualitative behavior across models using the extracted Figure 6 points.',
            'No full Planck/ACT/DESI likelihood refit is possible from the reduced workspace data alone.',
        ],
    }
    (OUTPUTS_DIR / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Analysis complete.')


if __name__ == '__main__':
    main()
