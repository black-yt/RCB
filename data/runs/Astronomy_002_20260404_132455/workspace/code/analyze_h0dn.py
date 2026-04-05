import ast
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

C_KM = 299792.458
PLANCK_H0 = 67.4
PLANCK_H0_ERR = 0.5
SEED = 20260404
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'H0DN_MinimalDataset.txt'
OUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def load_dataset(path: Path):
    namespace = {}
    exec(path.read_text(), {}, namespace)
    return namespace


def build_host_dataframe(ds):
    rows = []
    for host, method, anchor, mu_meas, err_meas in ds['host_measurements']:
        rows.append({
            'host': host,
            'method': method,
            'anchor': anchor,
            'mu_meas': float(mu_meas),
            'err_meas': float(err_meas),
            'anchor_mu': float(ds['anchors'][anchor]['mu']),
            'anchor_err': float(ds['anchors'][anchor]['err']),
            'method_anchor_err': float(ds['method_anchor_err'].get((method, anchor), 0.0)),
        })
    return pd.DataFrame(rows)


def host_covariance(subdf: pd.DataFrame, include_shared=True):
    n = len(subdf)
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_i = subdf.iloc[i]
        base_var = row_i.err_meas ** 2
        cov[i, i] += base_var
        if include_shared:
            cov[i, i] += row_i.anchor_err ** 2 + row_i.method_anchor_err ** 2
        for j in range(i + 1, n):
            row_j = subdf.iloc[j]
            shared = 0.0
            if include_shared and row_i.anchor == row_j.anchor:
                shared += row_i.anchor_err ** 2
            if include_shared and row_i.method == row_j.method and row_i.anchor == row_j.anchor:
                shared += min(row_i.method_anchor_err, row_j.method_anchor_err) ** 2
            cov[i, j] = cov[j, i] = shared
    return cov


def gls_host_estimates(host_df: pd.DataFrame, include_shared=True, methods=None, anchors=None):
    df = host_df.copy()
    if methods is not None:
        df = df[df['method'].isin(methods)].copy()
    if anchors is not None:
        df = df[df['anchor'].isin(anchors)].copy()
    host_rows = []
    for host, subdf in df.groupby('host'):
        y = subdf['mu_meas'].to_numpy()
        cov = host_covariance(subdf, include_shared=include_shared)
        ones = np.ones((len(subdf), 1))
        cov_inv = np.linalg.inv(cov)
        fisher = float((ones.T @ cov_inv @ ones)[0, 0])
        mu_hat = float(((ones.T @ cov_inv @ y.reshape(-1, 1))[0, 0]) / fisher)
        mu_err = math.sqrt(1.0 / fisher)
        chi2 = float((y - mu_hat) @ cov_inv @ (y - mu_hat))
        dof = max(len(subdf) - 1, 0)
        host_rows.append({
            'host': host,
            'mu_hat': mu_hat,
            'mu_err': mu_err,
            'n_measurements': len(subdf),
            'methods_used': ','.join(sorted(subdf['method'].unique())),
            'anchors_used': ','.join(sorted(subdf['anchor'].unique())),
            'chi2': chi2,
            'dof': dof,
            'reduced_chi2': chi2 / dof if dof > 0 else np.nan,
        })
    return pd.DataFrame(host_rows).sort_values('host').reset_index(drop=True)


def calibrate_absolute_magnitude(calibrators, host_estimates, mag_col='mB', err_col='err_mB', id_col='host'):
    host_map = host_estimates.set_index('host')[['mu_hat', 'mu_err']]
    rows = []
    for item in calibrators:
        host, mag, err = item
        if host not in host_map.index:
            continue
        mu_hat = host_map.loc[host, 'mu_hat']
        mu_err = host_map.loc[host, 'mu_err']
        abs_mag = mag - mu_hat
        abs_err = math.sqrt(err ** 2 + mu_err ** 2)
        rows.append({
            id_col: host,
            mag_col: mag,
            err_col: err,
            'mu_hat': mu_hat,
            'mu_err': mu_err,
            'abs_mag': abs_mag,
            'abs_mag_err': abs_err,
            'weight': 1.0 / abs_err ** 2,
        })
    df = pd.DataFrame(rows)
    w = df['weight'].to_numpy()
    mean = float(np.sum(w * df['abs_mag']) / np.sum(w))
    err = math.sqrt(1.0 / np.sum(w))
    chi2 = float(np.sum(((df['abs_mag'] - mean) / df['abs_mag_err']) ** 2))
    dof = max(len(df) - 1, 0)
    stats = {
        'abs_mag': mean,
        'abs_mag_err': err,
        'n_calibrators': int(len(df)),
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': chi2 / dof if dof > 0 else np.nan,
    }
    return df, stats


def build_sbf_calibrators(ds):
    rows = []
    anchors = ds['anchors']
    group_mu = {
        'Fornax': anchors['LMC']['mu'] + 12.94,
        'Virgo': anchors['LMC']['mu'] + 12.50,
    }
    group_mu_err = {
        'Fornax': math.sqrt(0.10 ** 2 + anchors['LMC']['err'] ** 2),
        'Virgo': math.sqrt(0.12 ** 2 + anchors['LMC']['err'] ** 2),
    }
    depth_scatter = float(ds['depth_scatter'])
    for host, mag, err in ds['sbf_calibrators']:
        group = ds['host_group'][host]
        mu_hat = group_mu[group]
        mu_err = math.sqrt(group_mu_err[group] ** 2 + depth_scatter ** 2)
        rows.append((host, mag, math.sqrt(err ** 2 + mu_err ** 2), mu_hat, mu_err, group))
    return rows


def calibrate_sbf_absolute_magnitude(ds):
    rows = []
    for host, mag, total_err, mu_hat, mu_err, group in build_sbf_calibrators(ds):
        abs_mag = mag - mu_hat
        rows.append({
            'host': host,
            'group': group,
            'm_sbf': mag,
            'm_sbf_err': total_err,
            'mu_hat': mu_hat,
            'mu_err': mu_err,
            'abs_mag': abs_mag,
            'abs_mag_err': total_err,
            'weight': 1.0 / total_err ** 2,
        })
    df = pd.DataFrame(rows)
    w = df['weight'].to_numpy()
    mean = float(np.sum(w * df['abs_mag']) / np.sum(w))
    err = math.sqrt(1.0 / np.sum(w))
    chi2 = float(np.sum(((df['abs_mag'] - mean) / df['abs_mag_err']) ** 2))
    dof = max(len(df) - 1, 0)
    stats = {
        'abs_mag': mean,
        'abs_mag_err': err,
        'n_calibrators': int(len(df)),
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': chi2 / dof if dof > 0 else np.nan,
    }
    return df, stats


def hubble_constant_from_flow(flow_rows, abs_mag, abs_mag_err, mag_key='mB', err_key='err_mB'):
    rows = []
    logh_vals = []
    logh_vars = []
    for z, mag, err, sigma_v in flow_rows:
        sigma_mu_pv = 5.0 / math.log(10.0) * sigma_v / (C_KM * z)
        sigma_tot = math.sqrt(err ** 2 + sigma_mu_pv ** 2 + abs_mag_err ** 2)
        logh = math.log10(C_KM * z) - 0.2 * (mag - abs_mag - 25.0)
        rows.append({
            'z': z,
            mag_key: mag,
            err_key: err,
            'sigma_v': sigma_v,
            'sigma_mu_pv': sigma_mu_pv,
            'sigma_tot_mag': sigma_tot,
            'log10_H0': logh,
            'H0': 10 ** logh,
        })
        var_logh = (0.2 ** 2) * sigma_tot ** 2
        logh_vals.append(logh)
        logh_vars.append(var_logh)
    df = pd.DataFrame(rows)
    weights = 1.0 / np.array(logh_vars)
    mean_logh = float(np.sum(weights * np.array(logh_vals)) / np.sum(weights))
    err_logh = math.sqrt(1.0 / np.sum(weights))
    h0 = 10 ** mean_logh
    h0_err = math.log(10.0) * h0 * err_logh
    return df, {
        'H0': h0,
        'H0_err': h0_err,
        'log10_H0': mean_logh,
        'log10_H0_err': err_logh,
        'n_flow': int(len(df)),
    }


def monte_carlo_h0(ds, host_df, methods=None, anchors=None, include_shared=True, n_draws=20000):
    host_est = gls_host_estimates(host_df, include_shared=include_shared, methods=methods, anchors=anchors)
    sn_cal_df, sn_stats = calibrate_absolute_magnitude(ds['sneia_calibrators'], host_est)
    flow_df, flow_stats = hubble_constant_from_flow(ds['hubble_flow_sneia'], sn_stats['abs_mag'], sn_stats['abs_mag_err'])

    host_map = host_est.set_index('host')
    mb_draws = []
    for _, row in sn_cal_df.iterrows():
        mb_draws.append(RNG.normal(row['abs_mag'], row['abs_mag_err'], size=n_draws))
    mb_mean_draws = np.average(np.vstack(mb_draws), axis=0, weights=sn_cal_df['weight'].to_numpy())

    logh_draws = []
    flow_weights = []
    for z, mag, err, sigma_v in ds['hubble_flow_sneia']:
        sigma_mu_pv = 5.0 / math.log(10.0) * sigma_v / (C_KM * z)
        mag_draw = RNG.normal(mag, math.sqrt(err ** 2 + sigma_mu_pv ** 2), size=n_draws)
        logh = np.log10(C_KM * z) - 0.2 * (mag_draw - mb_mean_draws - 25.0)
        logh_draws.append(logh)
        flow_weights.append(1.0 / ((0.2 ** 2) * (err ** 2 + sigma_mu_pv ** 2 + sn_stats['abs_mag_err'] ** 2)))
    flow_weights = np.array(flow_weights)
    logh_draws = np.vstack(logh_draws)
    mean_logh_draws = np.average(logh_draws, axis=0, weights=flow_weights)
    h0_draws = 10 ** mean_logh_draws
    ci16, ci50, ci84 = np.percentile(h0_draws, [16, 50, 84])
    mc_stats = {
        'median': float(ci50),
        'p16': float(ci16),
        'p84': float(ci84),
        'std': float(np.std(h0_draws, ddof=1)),
    }
    return host_est, sn_cal_df, sn_stats, flow_df, flow_stats, mc_stats, h0_draws


def run_variants(ds, host_df):
    variants = [
        ('baseline_gls', dict(methods=None, anchors=None, include_shared=True)),
        ('diag_only', dict(methods=None, anchors=None, include_shared=False)),
        ('cepheid_only', dict(methods=['Cepheid'], anchors=None, include_shared=True)),
        ('trgb_only', dict(methods=['TRGB'], anchors=None, include_shared=True)),
        ('n4258_only', dict(methods=None, anchors=['N4258'], include_shared=True)),
        ('lmc_only', dict(methods=None, anchors=['LMC'], include_shared=True)),
        ('cepheid_n4258', dict(methods=['Cepheid'], anchors=['N4258'], include_shared=True)),
    ]
    results = []
    draw_cache = {}
    for name, kwargs in variants:
        host_est, sn_cal_df, sn_stats, flow_df, flow_stats, mc_stats, h0_draws = monte_carlo_h0(ds, host_df, **kwargs)
        results.append({
            'variant': name,
            'methods': 'all' if kwargs['methods'] is None else ','.join(kwargs['methods']),
            'anchors': 'all' if kwargs['anchors'] is None else ','.join(kwargs['anchors']),
            'include_shared': kwargs['include_shared'],
            'n_hosts': int(len(host_est)),
            'M_B': sn_stats['abs_mag'],
            'M_B_err': sn_stats['abs_mag_err'],
            'H0': flow_stats['H0'],
            'H0_err_analytic': flow_stats['H0_err'],
            'H0_mc_median': mc_stats['median'],
            'H0_mc_std': mc_stats['std'],
            'H0_mc_p16': mc_stats['p16'],
            'H0_mc_p84': mc_stats['p84'],
            'delta_vs_baseline': np.nan,
        })
        draw_cache[name] = h0_draws
    results_df = pd.DataFrame(results)
    baseline = float(results_df.loc[results_df['variant'] == 'baseline_gls', 'H0'].iloc[0])
    results_df['delta_vs_baseline'] = results_df['H0'] - baseline
    return results_df, draw_cache


def fit_sbf_h0(ds):
    sbf_cal_df, sbf_stats = calibrate_sbf_absolute_magnitude(ds)
    flow_df, flow_stats = hubble_constant_from_flow(ds['hubble_flow_sbf'], sbf_stats['abs_mag'], sbf_stats['abs_mag_err'], mag_key='m_sbf', err_key='err_m_sbf')
    return sbf_cal_df, sbf_stats, flow_df, flow_stats


def make_figures(host_df, host_est, sn_cal_df, flow_df, variants_df, sbf_flow_df, sbf_stats, baseline_stats):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=host_df, x='method', hue='anchor')
    plt.title('Primary-indicator measurements by method and anchor')
    plt.ylabel('Number of host measurements')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'data_overview_measurements.png', dpi=200)
    plt.close()

    host_plot_df = host_df.merge(host_est[['host', 'mu_hat', 'mu_err']], on='host', how='left')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=host_plot_df, x='mu_meas', y='mu_hat', hue='method', style='anchor', s=120)
    lims = [host_plot_df[['mu_meas', 'mu_hat']].min().min() - 0.2, host_plot_df[['mu_meas', 'mu_hat']].max().max() + 0.2]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.xlabel('Individual host distance modulus measurements')
    plt.ylabel('Covariance-weighted host distance modulus')
    plt.title('GLS synthesis of host distances')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'host_modulus_gls.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    order = sn_cal_df.sort_values('abs_mag')['host']
    sns.pointplot(data=sn_cal_df.sort_values('abs_mag'), x='host', y='abs_mag', linestyle='none', order=order)
    plt.errorbar(x=np.arange(len(sn_cal_df)), y=sn_cal_df.sort_values('abs_mag')['abs_mag'], yerr=sn_cal_df.sort_values('abs_mag')['abs_mag_err'], fmt='none', c='black', capsize=4)
    plt.axhline(baseline_stats['M_B'], color='crimson', linestyle='--', label=f"Weighted mean = {baseline_stats['M_B']:.2f}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('SN Ia absolute magnitude $M_B$')
    plt.xlabel('Calibrator host')
    plt.title('SN Ia calibrator absolute magnitudes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'snia_calibrators.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=flow_df, x='z', y='H0', s=120)
    plt.axhline(baseline_stats['H0'], color='crimson', linestyle='--', label=f"Baseline H0 = {baseline_stats['H0']:.2f}")
    plt.ylabel('$H_0$ implied by each Hubble-flow SN')
    plt.xlabel('Redshift z')
    plt.title('Hubble-flow SN Ia constraints on $H_0$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'hubble_flow_snia.png', dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    ordered = variants_df.sort_values('H0')
    plt.errorbar(ordered['H0'], ordered['variant'], xerr=ordered['H0_mc_std'], fmt='o', color='navy', capsize=4)
    plt.axvline(baseline_stats['H0'], color='crimson', linestyle='--', label='Baseline GLS')
    plt.axvline(PLANCK_H0, color='darkgreen', linestyle=':', label=r'Planck $\Lambda$CDM reference')
    plt.xlabel('$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    plt.ylabel('Analysis variant')
    plt.title('Sensitivity of $H_0$ to analysis choices')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'variant_comparison.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sbf_flow_df, x='z', y='H0', s=120, color='purple')
    plt.axhline(sbf_stats['H0'], color='purple', linestyle='--', label=f"SBF-only H0 = {sbf_stats['H0']:.2f}")
    plt.axhline(baseline_stats['H0'], color='crimson', linestyle=':', label='SN Ia baseline')
    plt.ylabel('$H_0$ implied by each Hubble-flow SBF galaxy')
    plt.xlabel('Redshift z')
    plt.title('SBF cross-check for the local distance network')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'sbf_crosscheck.png', dpi=200)
    plt.close()


def main():
    ds = load_dataset(DATA_PATH)
    host_df = build_host_dataframe(ds)

    host_est, sn_cal_df, sn_stats, flow_df, flow_stats, mc_stats, h0_draws = monte_carlo_h0(ds, host_df)
    variants_df, draw_cache = run_variants(ds, host_df)
    sbf_cal_df, sbf_mag_stats, sbf_flow_df, sbf_flow_stats = fit_sbf_h0(ds)

    baseline_stats = {
        'M_B': sn_stats['abs_mag'],
        'M_B_err': sn_stats['abs_mag_err'],
        'H0': flow_stats['H0'],
        'H0_err': flow_stats['H0_err'],
        'H0_mc_std': mc_stats['std'],
        'H0_mc_p16': mc_stats['p16'],
        'H0_mc_p84': mc_stats['p84'],
    }
    baseline_stats['tension_sigma_vs_planck'] = (baseline_stats['H0'] - PLANCK_H0) / math.sqrt(baseline_stats['H0_err'] ** 2 + PLANCK_H0_ERR ** 2)
    baseline_stats['difference_vs_reference_73p5'] = baseline_stats['H0'] - 73.50
    baseline_stats['interpretation'] = (
        'The minimal dataset does not reproduce the published consensus normalization; '
        'its absolute SN and SBF calibrations imply a much higher H0 than the target 73.5 km s^-1 Mpc^-1.'
    )

    host_df.to_csv(OUT_DIR / 'host_measurements_tidy.csv', index=False)
    host_est.to_csv(OUT_DIR / 'host_estimates_baseline.csv', index=False)
    sn_cal_df.to_csv(OUT_DIR / 'snia_calibrator_estimates.csv', index=False)
    flow_df.to_csv(OUT_DIR / 'snia_hubble_flow_estimates.csv', index=False)
    sbf_cal_df.to_csv(OUT_DIR / 'sbf_calibrator_estimates.csv', index=False)
    sbf_flow_df.to_csv(OUT_DIR / 'sbf_hubble_flow_estimates.csv', index=False)
    variants_df.to_csv(OUT_DIR / 'analysis_variants.csv', index=False)

    summary = {
        'seed': SEED,
        'baseline': baseline_stats,
        'baseline_monte_carlo': mc_stats,
        'snia_calibration': sn_stats,
        'sbf_calibration': sbf_mag_stats,
        'sbf_flow': sbf_flow_stats,
        'dataset_counts': {
            'host_measurements': int(len(host_df)),
            'unique_hosts': int(host_df['host'].nunique()),
            'snia_calibrators': len(ds['sneia_calibrators']),
            'snia_hubble_flow': len(ds['hubble_flow_sneia']),
            'sbf_calibrators': len(ds['sbf_calibrators']),
            'sbf_hubble_flow': len(ds['hubble_flow_sbf']),
        },
    }
    (OUT_DIR / 'summary_results.json').write_text(json.dumps(summary, indent=2))

    draws_df = pd.DataFrame({name: vals[:5000] for name, vals in draw_cache.items()})
    draws_df.to_csv(OUT_DIR / 'h0_variant_draws_preview.csv', index=False)

    make_figures(host_df, host_est, sn_cal_df, flow_df, variants_df, sbf_flow_df, {**sbf_flow_stats, **{'H0': sbf_flow_stats['H0']}}, baseline_stats)


if __name__ == '__main__':
    main()
