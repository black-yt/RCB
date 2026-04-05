from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 12345
rng = np.random.default_rng(SEED)

ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.bbox'] = 'tight'


def bootstrap_ci(values, stat_func=np.median, n_boot=4000, alpha=0.05, random_state=SEED):
    values = np.asarray(values, dtype=float)
    local_rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = local_rng.choice(values, size=n, replace=True)
        stats[i] = stat_func(sample)
    lower = np.quantile(stats, alpha / 2)
    upper = np.quantile(stats, 1 - alpha / 2)
    return float(lower), float(upper)


def summarize_series(name: str, series: pd.Series) -> dict:
    s = series.astype(float)
    ci_low, ci_high = bootstrap_ci(s.to_numpy(), np.median)
    return {
        'name': name,
        'count': int(s.shape[0]),
        'mean': float(s.mean()),
        'std': float(s.std(ddof=1)),
        'min': float(s.min()),
        'q25': float(s.quantile(0.25)),
        'median': float(s.median()),
        'q75': float(s.quantile(0.75)),
        'q90': float(s.quantile(0.90)),
        'q95': float(s.quantile(0.95)),
        'max': float(s.max()),
        'median_ci_low': ci_low,
        'median_ci_high': ci_high,
    }


def main():
    fig6 = pd.read_csv(DATA_DIR / 'fig6_data.csv')
    fig7 = pd.read_csv(DATA_DIR / 'fig7_data.csv')
    fig8 = pd.read_csv(DATA_DIR / 'fig8_data.csv')

    validation = {
        'fig6_shape': fig6.shape,
        'fig7_shape': fig7.shape,
        'fig8_shape': fig8.shape,
        'fig6_missing': int(fig6.isna().sum().sum()),
        'fig7_missing': int(fig7.isna().sum().sum()),
        'fig8_missing': int(fig8.isna().sum().sum()),
        'seed': SEED,
    }

    resolution = fig6.iloc[:, 0].rename('waveform_difference')
    resolution_summary = summarize_series('resolution_difference', resolution)
    thresholds = [1e-4, 5e-4, 1e-3, 1e-2]
    resolution_thresholds = {
        f'frac_below_{thr:.0e}': float((resolution < thr).mean()) for thr in thresholds
    }
    resolution_tail = {
        'frac_above_1e-2': float((resolution > 1e-2).mean()),
        'frac_above_5e-3': float((resolution > 5e-3).mean()),
        'top_1pct_threshold': float(resolution.quantile(0.99)),
    }

    mode_rows = []
    for col in fig7.columns:
        row = summarize_series(col, fig7[col])
        row['ell'] = int(col.replace('ell', ''))
        mode_rows.append(row)
    mode_summary = pd.DataFrame(mode_rows).sort_values('ell')
    mode_summary['median_ratio_to_ell2'] = mode_summary['median'] / mode_summary.loc[mode_summary['ell'] == 2, 'median'].iloc[0]

    long_mode = fig7.melt(var_name='mode', value_name='difference')
    long_mode['ell'] = long_mode['mode'].str.replace('ell', '', regex=False).astype(int)

    n23 = fig8['N2vsN3'].astype(float)
    n24 = fig8['N2vsN4'].astype(float)
    paired_log_diff = np.log10(n24) - np.log10(n23)
    ratio = n24 / n23
    log_ratio_ci = bootstrap_ci(np.log10(ratio.to_numpy()), np.median)
    extrapolation_summary = pd.DataFrame([
        summarize_series('N2vsN3', n23),
        summarize_series('N2vsN4', n24),
    ])
    extrapolation_summary['comparison'] = extrapolation_summary['name']

    quality_metrics = {
        'validation': validation,
        'resolution_summary': resolution_summary,
        'resolution_thresholds': resolution_thresholds,
        'resolution_tail': resolution_tail,
        'mode_monotonic_median_increase': bool(np.all(np.diff(mode_summary['median']) > 0)),
        'mode_monotonic_q75_increase': bool(np.all(np.diff(mode_summary['q75']) > 0)),
        'extrapolation_median_ratio_N24_to_N23': float(np.median(ratio)),
        'extrapolation_median_log10_ratio_ci': [float(log_ratio_ci[0]), float(log_ratio_ci[1])],
        'extrapolation_frac_N24_gt_N23': float((n24 > n23).mean()),
        'extrapolation_wilcoxon_proxy_sign_fraction': float((paired_log_diff > 0).mean()),
    }

    pd.DataFrame([resolution_summary]).to_csv(OUTPUT_DIR / 'summary_stats.csv', index=False)
    mode_summary.to_csv(OUTPUT_DIR / 'mode_summary.csv', index=False)
    extrapolation_summary.to_csv(OUTPUT_DIR / 'extrapolation_summary.csv', index=False)
    with open(OUTPUT_DIR / 'quality_metrics.json', 'w') as f:
        json.dump(quality_metrics, f, indent=2)

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(resolution, bins=40, log_scale=(True, False), color='#4C72B0', edgecolor='white')
    ax.axvline(resolution.median(), color='black', linestyle='--', linewidth=2, label=f"Median = {resolution.median():.2e}")
    for thr, color in zip([1e-3, 1e-2], ['#DD8452', '#C44E52']):
        ax.axvline(thr, color=color, linestyle=':', linewidth=2, label=f"{thr:.0e} threshold")
    ax.set_xlabel('Minimal-alignment waveform difference')
    ax.set_ylabel('Simulation count')
    ax.set_title('Distribution of highest-resolution waveform differences')
    ax.legend(frameon=True)
    plt.savefig(IMG_DIR / 'fig_resolution_distribution.png')
    plt.close()

    plt.figure(figsize=(11, 6))
    ax = sns.boxplot(data=long_mode, x='ell', y='difference', color='#55A868', showfliers=False)
    ax.set_yscale('log')
    ax.set_xlabel('Spherical-harmonic mode index $\\ell$')
    ax.set_ylabel('Waveform difference')
    ax.set_title('Mode-resolved waveform disagreement broadens toward higher $\\ell$')
    plt.savefig(IMG_DIR / 'fig_mode_distributions.png')
    plt.close()

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.errorbar(
        mode_summary['ell'],
        mode_summary['median'],
        yerr=[mode_summary['median'] - mode_summary['q25'], mode_summary['q75'] - mode_summary['median']],
        fmt='o-',
        color='#8172B2',
        capsize=5,
        linewidth=2,
    )
    ax.set_yscale('log')
    ax.set_xticks(mode_summary['ell'])
    ax.set_xlabel('Mode index $\\ell$')
    ax.set_ylabel('Median waveform difference (IQR error bars)')
    ax.set_title('Median modal error increases systematically with harmonic order')
    plt.savefig(IMG_DIR / 'fig_mode_growth.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_df = fig8.melt(var_name='comparison', value_name='difference')
    ax = sns.violinplot(
        data=plot_df,
        x='comparison',
        y='difference',
        hue='comparison',
        inner='quartile',
        palette=['#64B5CD', '#CCB974'],
        legend=False,
    )
    ax.set_yscale('log')
    ax.set_xlabel('Extrapolation-order comparison')
    ax.set_ylabel('Waveform difference')
    ax.set_title('Higher-order extrapolation comparisons show larger disagreement')
    plt.savefig(IMG_DIR / 'fig_extrapolation_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sample = pd.DataFrame({'N2vsN3': n23, 'N2vsN4': n24}).sample(n=min(400, len(fig8)), random_state=SEED)
    ax = sns.scatterplot(data=sample, x='N2vsN3', y='N2vsN4', s=45, alpha=0.7, color='#C44E52')
    lims = [min(sample.min()) * 0.8, max(sample.max()) * 1.2]
    ax.plot(lims, lims, linestyle='--', color='black', linewidth=1.5, label='Parity line')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('N=2 vs N=3 difference')
    ax.set_ylabel('N=2 vs N=4 difference')
    ax.set_title('Paired extrapolation discrepancies mostly lie above parity')
    ax.legend(frameon=True)
    plt.savefig(IMG_DIR / 'fig_extrapolation_scatter.png')
    plt.close()


if __name__ == '__main__':
    main()
