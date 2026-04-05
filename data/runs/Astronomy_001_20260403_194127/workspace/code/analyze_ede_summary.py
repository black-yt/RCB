import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / 'data' / 'DESI_EDE_Repro_Data.txt'
OUT_DIR = BASE / 'outputs'
IMG_DIR = BASE / 'report' / 'images'

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def parse_data_file(path: Path):
    text = path.read_text()
    env = {}
    filtered_lines = []
    for line in text.splitlines():
        if line.strip().startswith('#') or not line.strip():
            continue
        filtered_lines.append(line)
    code = '\n'.join(filtered_lines)
    exec(code, {}, env)
    return env


def gaussian_overlap(mu1, sig1, mu2, sig2, n_grid=2000):
    lo = min(mu1 - 6 * sig1, mu2 - 6 * sig2)
    hi = max(mu1 + 6 * sig1, mu2 + 6 * sig2)
    grid = np.linspace(lo, hi, n_grid)
    pdf1 = np.exp(-0.5 * ((grid - mu1) / sig1) ** 2) / (sig1 * np.sqrt(2 * np.pi))
    pdf2 = np.exp(-0.5 * ((grid - mu2) / sig2) ** 2) / (sig2 * np.sqrt(2 * np.pi))
    return float(np.trapezoid(np.minimum(pdf1, pdf2), grid))


def weighted_linear_fit(x, y, sigma):
    w = 1.0 / np.square(sigma)
    X = np.vstack([np.ones_like(x), x]).T
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)
    beta = np.linalg.solve(XtWX, XtWy)
    cov = np.linalg.inv(XtWX)
    intercept, slope = beta
    intercept_err, slope_err = np.sqrt(np.diag(cov))
    return intercept, intercept_err, slope, slope_err


def weighted_mean(y, sigma):
    w = 1.0 / np.square(sigma)
    mu = np.sum(w * y) / np.sum(w)
    err = np.sqrt(1.0 / np.sum(w))
    return float(mu), float(err)


def validate_points(name, pts):
    zs = [p[0] for p in pts]
    errs = [p[2] for p in pts]
    monotonic = all(z2 > z1 for z1, z2 in zip(zs[:-1], zs[1:]))
    positive_errs = all(e > 0 for e in errs)
    return {
        'dataset': name,
        'n_points': len(pts),
        'z_min': min(zs),
        'z_max': max(zs),
        'monotonic_redshift': monotonic,
        'positive_errors': positive_errs,
    }


def make_manifest(env):
    model_names = ['lcdm_params', 'ede_params', 'w0wa_params']
    rows = []
    for name in model_names:
        for param in env[name].keys():
            rows.append({'model': name.replace('_params', ''), 'parameter': param})
    manifest_df = pd.DataFrame(rows)
    pivot = manifest_df.assign(value=1).pivot_table(index='parameter', columns='model', values='value', fill_value=0)
    pivot.to_csv(OUT_DIR / 'parameter_manifest.csv')
    with open(OUT_DIR / 'data_manifest.md', 'w') as f:
        f.write('# Data Manifest\n\n')
        f.write('## Model parameter availability\n\n')
        f.write(pivot.to_string())
        f.write('\n\n## Residual datasets\n')
        for key in ['desi_dvrd_points', 'desi_fap_points', 'sne_mu_points']:
            f.write(f'- `{key}`: {len(env[key])} points\n')
    return pivot


def build_parameter_table(env):
    model_map = {
        'lcdm': env['lcdm_params'],
        'ede': env['ede_params'],
        'w0wa': env['w0wa_params'],
    }
    rows = []
    for model, params in model_map.items():
        for param, (mean, sigma) in params.items():
            rows.append({'model': model, 'parameter': param, 'mean': mean, 'sigma': sigma})
    df = pd.DataFrame(rows).sort_values(['parameter', 'model'])
    df.to_csv(OUT_DIR / 'parameter_summary_table.csv', index=False)
    return df, model_map


def compute_pairwise(model_map):
    comparisons = [('lcdm', 'ede'), ('lcdm', 'w0wa'), ('ede', 'w0wa')]
    pull_rows, overlap_rows = [], []
    for a, b in comparisons:
        shared = sorted(set(model_map[a]).intersection(model_map[b]))
        for p in shared:
            mu1, s1 = model_map[a][p]
            mu2, s2 = model_map[b][p]
            z = (mu2 - mu1) / math.sqrt(s1 ** 2 + s2 ** 2)
            overlap = gaussian_overlap(mu1, s1, mu2, s2)
            pull_rows.append({
                'comparison': f'{a}_vs_{b}',
                'parameter': p,
                'mean_a': mu1,
                'sigma_a': s1,
                'mean_b': mu2,
                'sigma_b': s2,
                'delta_b_minus_a': mu2 - mu1,
                'standardized_shift_z': z,
                'abs_z': abs(z),
            })
            overlap_rows.append({
                'comparison': f'{a}_vs_{b}',
                'parameter': p,
                'overlap_coefficient': overlap,
            })
    pull_df = pd.DataFrame(pull_rows).sort_values(['comparison', 'abs_z'], ascending=[True, False])
    overlap_df = pd.DataFrame(overlap_rows).sort_values(['comparison', 'overlap_coefficient'])
    pull_df.to_csv(OUT_DIR / 'pull_difference_table.csv', index=False)
    overlap_df.to_csv(OUT_DIR / 'overlap_metrics_table.csv', index=False)
    return pull_df, overlap_df


def summarize_residuals(points, name):
    arr = np.array(points, dtype=float)
    z, y, err = arr[:, 0], arr[:, 1], arr[:, 2]
    mu, mu_err = weighted_mean(y, err)
    intercept, intercept_err, slope, slope_err = weighted_linear_fit(z, y, err)
    frac_pos = float(np.mean(y > 0))
    frac_neg = float(np.mean(y < 0))
    summary = pd.DataFrame([{
        'dataset': name,
        'n_points': len(z),
        'weighted_mean': mu,
        'weighted_mean_err': mu_err,
        'intercept': intercept,
        'intercept_err': intercept_err,
        'slope_per_redshift': slope,
        'slope_err': slope_err,
        'frac_positive': frac_pos,
        'frac_negative': frac_neg,
        'mean_raw': float(np.mean(y)),
        'std_raw': float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
    }])
    summary.to_csv(OUT_DIR / f'{name}_summary.csv', index=False)
    detailed = pd.DataFrame({'z': z, 'value': y, 'error': err})
    detailed.to_csv(OUT_DIR / f'{name}_points.csv', index=False)
    return summary.iloc[0].to_dict(), detailed


def plot_parameters(df):
    params = ['H0', 'omega_m', 'sigma8', 'ns', 'ombh2', 'tau']
    models = ['lcdm', 'ede', 'w0wa']
    colors = {'lcdm': '#4c72b0', 'ede': '#dd8452', 'w0wa': '#55a868'}
    fig, axes = plt.subplots(len(params), 1, figsize=(7.5, 10.5), sharex=False)
    for ax, param in zip(axes, params):
        sub = df[df['parameter'] == param].set_index('model').loc[models].reset_index()
        y = np.arange(len(models))
        ax.errorbar(sub['mean'], y, xerr=sub['sigma'], fmt='o', color='black', ecolor='black', capsize=3)
        for yy, (_, row) in zip(y, sub.iterrows()):
            ax.scatter(row['mean'], yy, s=70, color=colors[row['model']], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub['model'])
        ax.set_title(param)
        ax.grid(alpha=0.3, axis='x')
    fig.suptitle('Parameter constraints from extracted CMB+DESI summaries', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(IMG_DIR / 'parameter_forest.png', dpi=200)
    plt.close(fig)


def plot_pull_heatmap(pull_df):
    heat = pull_df.pivot(index='parameter', columns='comparison', values='standardized_shift_z')
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heat.values, cmap='coolwarm', aspect='auto', vmin=-3.5, vmax=3.5)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title('Standardized parameter shifts between models')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Z-shift')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'pull_heatmap.png', dpi=200)
    plt.close(fig)


def plot_residual_series(points, ylabel, title, filename):
    arr = np.array(points, dtype=float)
    z, y, err = arr[:, 0], arr[:, 1], arr[:, 2]
    intercept, intercept_err, slope, slope_err = weighted_linear_fit(z, y, err)
    zz = np.linspace(z.min(), z.max(), 200)
    yy = intercept + slope * zz
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(z, y, yerr=err, fmt='o', capsize=3, color='#4c72b0')
    ax.plot(zz, yy, color='#dd8452', lw=2, label=f'weighted trend: slope={slope:.3f}±{slope_err:.3f}')
    ax.axhline(0.0, color='black', ls='--', lw=1)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_DIR / filename, dpi=200)
    plt.close(fig)


def plot_combined_desi(dvrd_points, fap_points):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for ax, points, ylabel, title in [
        (axes[0], dvrd_points, 'Δ(D_V/r_d)', 'DESI BAO volume-distance residuals'),
        (axes[1], fap_points, 'ΔF_AP', 'DESI BAO Alcock–Paczynski residuals'),
    ]:
        arr = np.array(points, dtype=float)
        z, y, err = arr[:, 0], arr[:, 1], arr[:, 2]
        intercept, intercept_err, slope, slope_err = weighted_linear_fit(z, y, err)
        zz = np.linspace(z.min(), z.max(), 200)
        ax.errorbar(z, y, yerr=err, fmt='o', capsize=3, color='#4c72b0')
        ax.plot(zz, intercept + slope * zz, color='#dd8452', lw=2)
        ax.axhline(0.0, color='black', ls='--', lw=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Redshift z')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'desi_residuals.png', dpi=200)
    plt.close(fig)


def write_qc(env):
    qc_rows = [
        validate_points('desi_dvrd', env['desi_dvrd_points']),
        validate_points('desi_fap', env['desi_fap_points']),
        validate_points('sne_mu', env['sne_mu_points']),
    ]
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(OUT_DIR / 'data_qc.csv', index=False)
    with open(OUT_DIR / 'data_qc_summary.txt', 'w') as f:
        for row in qc_rows:
            f.write(str(row) + '\n')
    return qc_df


def write_highlights(model_map, pull_df, residual_summaries):
    h0 = {m: model_map[m]['H0'] for m in model_map}
    top_lcdm_ede = pull_df[pull_df['comparison'] == 'lcdm_vs_ede'].sort_values('abs_z', ascending=False).head(5)
    top_lcdm_w0wa = pull_df[pull_df['comparison'] == 'lcdm_vs_w0wa'].sort_values('abs_z', ascending=False).head(5)
    lines = []
    lines.append('# Key numerical highlights\n')
    lines.append(f"- H0 (LambdaCDM): {h0['lcdm'][0]:.2f} ± {h0['lcdm'][1]:.2f}")
    lines.append(f"- H0 (EDE): {h0['ede'][0]:.2f} ± {h0['ede'][1]:.2f}")
    lines.append(f"- H0 (w0wa): {h0['w0wa'][0]:.2f} ± {h0['w0wa'][1]:.2f}\n")
    lines.append('## Largest standardized shifts: LambdaCDM vs EDE')
    for _, row in top_lcdm_ede.iterrows():
        lines.append(f"- {row['parameter']}: Δ={row['delta_b_minus_a']:.4f}, Z={row['standardized_shift_z']:.2f}")
    lines.append('\n## Largest standardized shifts: LambdaCDM vs w0wa')
    for _, row in top_lcdm_w0wa.iterrows():
        lines.append(f"- {row['parameter']}: Δ={row['delta_b_minus_a']:.4f}, Z={row['standardized_shift_z']:.2f}")
    lines.append('\n## Residual weighted means')
    for name, summary in residual_summaries.items():
        lines.append(f"- {name}: {summary['weighted_mean']:.4f} ± {summary['weighted_mean_err']:.4f}, slope={summary['slope_per_redshift']:.4f} ± {summary['slope_err']:.4f}")
    (OUT_DIR / 'highlights.md').write_text('\n'.join(lines) + '\n')


def main():
    env = parse_data_file(DATA_FILE)
    make_manifest(env)
    write_qc(env)
    df, model_map = build_parameter_table(env)
    pull_df, overlap_df = compute_pairwise(model_map)

    residual_summaries = {}
    for key, name in [('desi_dvrd_points', 'desi_dvrd_residual'), ('desi_fap_points', 'desi_fap_residual'), ('sne_mu_points', 'union3_mu_residual')]:
        summary, detailed = summarize_residuals(env[key], name)
        residual_summaries[name] = summary

    plot_parameters(df)
    plot_pull_heatmap(pull_df)
    plot_residual_series(env['desi_dvrd_points'], 'Δ(D_V/r_d)', 'DESI BAO volume-distance residuals', 'desi_dvrd.png')
    plot_residual_series(env['desi_fap_points'], 'ΔF_AP', 'DESI BAO Alcock–Paczynski residuals', 'desi_fap.png')
    plot_residual_series(env['sne_mu_points'], 'Δμ', 'Union3 supernova distance-modulus residuals', 'union3_mu.png')
    plot_combined_desi(env['desi_dvrd_points'], env['desi_fap_points'])
    write_highlights(model_map, pull_df, residual_summaries)

    # Also save compact executive tables for report use.
    h0_table = pd.DataFrame([
        {'model': m, 'H0_mean': model_map[m]['H0'][0], 'H0_sigma': model_map[m]['H0'][1]}
        for m in ['lcdm', 'ede', 'w0wa']
    ])
    h0_table.to_csv(OUT_DIR / 'h0_comparison.csv', index=False)

    ede_specific = pd.DataFrame([
        {'parameter': 'f_EDE', 'mean': model_map['ede']['f_EDE'][0], 'sigma': model_map['ede']['f_EDE'][1]},
        {'parameter': 'log10_ac', 'mean': model_map['ede']['log10_ac'][0], 'sigma': model_map['ede']['log10_ac'][1]},
    ])
    ede_specific.to_csv(OUT_DIR / 'ede_specific_parameters.csv', index=False)

    print('Analysis complete.')


if __name__ == '__main__':
    main()
