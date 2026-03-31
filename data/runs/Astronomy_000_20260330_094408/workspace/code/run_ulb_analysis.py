import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

SEED = 12345
np.random.seed(SEED)

ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')

HBAR_C_OVER_G_MSUN_EV = 1.337e-10  # eV * M_sun, from alpha = mu M / const
ALPHA_MIN = 0.08
ALPHA_MAX = 0.35
LOGISTIC_WIDTH = 0.03
A_MAX = 0.999
F_A_CRIT = 1e14


def expit_stable(x):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
COUPLING_GRID = np.array([1e13, 3e13, 1e14, 3e14, 1e15])
MASS_GRID = np.logspace(-21, -10, 500)
DATASETS = {
    'M33 X-7': {
        'file': DATA_DIR / 'M33_X-7_samples.dat',
        'tau_years': 5e6,
        'color': '#1f77b4',
    },
    'IRAS 09149-6206': {
        'file': DATA_DIR / 'IRAS_09149-6206_samples.dat',
        'tau_years': 4.5e7,
        'color': '#d62728',
    },
}


def load_samples(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, comments='#')
    df = pd.DataFrame(data, columns=['mass_msun', 'spin'])
    return df


def alpha_of_mu_mass(mu_ev, mass_msun):
    return mu_ev * mass_msun / HBAR_C_OVER_G_MSUN_EV


def critical_spin(alpha):
    # Phenomenological Regge boundary: low alpha unstable only for high spin,
    # minimum threshold near alpha~0.2, then becomes unfavorable at high alpha.
    valley = 0.12 + 0.55 * ((alpha - 0.2) / 0.2) ** 2
    penalty_low = 0.08 / np.sqrt(np.maximum(alpha, 1e-6) / 0.08)
    penalty_high = 0.18 * np.maximum(alpha - 0.35, 0.0) / 0.1
    crit = valley + penalty_low + penalty_high
    return np.clip(crit, 0.05, 0.995)


def alpha_window_weight(alpha):
    rise = expit_stable((alpha - ALPHA_MIN) / 0.015)
    fall = expit_stable(-(alpha - ALPHA_MAX) / 0.02)
    return rise * fall


def timescale_weight(alpha, tau_years):
    # Normalize to Salpeter-scale baseline; faster instability for smaller tau_years.
    tau_ref = 4.5e7
    ratio = np.clip(tau_ref / tau_years, 0.1, 50.0)
    base = np.exp(-((alpha - 0.22) / 0.12) ** 2)
    modifier = np.clip(0.6 + 0.25 * np.log10(ratio + 1.0), 0.35, 1.25)
    return np.clip(base * modifier, 0.0, 1.0)


def exclusion_probability(df, mu_ev, tau_years, coupling=None):
    alpha = alpha_of_mu_mass(mu_ev, df['mass_msun'].to_numpy())
    crit = critical_spin(alpha)
    gate = alpha_window_weight(alpha) * timescale_weight(alpha, tau_years)
    z = (df['spin'].to_numpy() - crit) / LOGISTIC_WIDTH
    spin_exclusion = expit_stable(z)
    exclusion = gate * spin_exclusion
    if coupling is not None:
        suppression = 1.0 / (1.0 + (F_A_CRIT / coupling) ** 2)
        exclusion = exclusion * suppression
    return float(np.mean(exclusion))


def summarize_dataset(name, df):
    q = df.quantile([0.05, 0.5, 0.95])
    corr = float(df.corr().loc['mass_msun', 'spin'])
    row = {
        'dataset': name,
        'n_samples': int(len(df)),
        'mass_mean_msun': float(df['mass_msun'].mean()),
        'mass_std_msun': float(df['mass_msun'].std(ddof=1)),
        'mass_p05_msun': float(q.loc[0.05, 'mass_msun']),
        'mass_p50_msun': float(q.loc[0.5, 'mass_msun']),
        'mass_p95_msun': float(q.loc[0.95, 'mass_msun']),
        'spin_mean': float(df['spin'].mean()),
        'spin_std': float(df['spin'].std(ddof=1)),
        'spin_p05': float(q.loc[0.05, 'spin']),
        'spin_p50': float(q.loc[0.5, 'spin']),
        'spin_p95': float(q.loc[0.95, 'spin']),
        'mass_spin_corr': corr,
    }
    return row


def make_overview_plots(samples):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    for i, (name, meta) in enumerate(DATASETS.items()):
        df = samples[name]
        axes[i].hist(df['mass_msun'], bins=35, color=meta['color'], alpha=0.8)
        axes[i].set_xscale('log')
        axes[i].set_title(f'{name}: mass posterior')
        axes[i].set_xlabel(r'$M_{\rm BH}\,[M_\odot]$')
        axes[i].set_ylabel('Count')
        axes[i + 2].hist(df['spin'], bins=35, color=meta['color'], alpha=0.8)
        axes[i + 2].set_title(f'{name}: spin posterior')
        axes[i + 2].set_xlabel(r'$a_*$')
        axes[i + 2].set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'posterior_overview.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, meta) in zip(axes, DATASETS.items()):
        df = samples[name]
        ax.scatter(df['mass_msun'], df['spin'], s=8, alpha=0.25, color=meta['color'])
        ax.set_xscale('log')
        ax.set_xlabel(r'$M_{\rm BH}\,[M_\odot]$')
        ax.set_ylabel(r'$a_*$')
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'posterior_correlation.png', dpi=200)
    plt.close(fig)


def make_boundary_plot(samples):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (name, meta) in zip(axes, DATASETS.items()):
        df = samples[name]
        ax.scatter(df['mass_msun'], df['spin'], s=8, alpha=0.18, color=meta['color'], label='Posterior samples')
        mass_span = np.logspace(np.log10(df['mass_msun'].quantile(0.01)), np.log10(df['mass_msun'].quantile(0.99)), 250)
        for mu, ls in [(3e-13, '--'), (1e-12, '-'), (3e-12, ':')]:
            alpha = alpha_of_mu_mass(mu, mass_span)
            ax.plot(mass_span, critical_spin(alpha), ls=ls, lw=2, label=fr'$\mu={mu:.0e}$ eV')
        ax.set_xscale('log')
        ax.set_xlabel(r'$M_{\rm BH}\,[M_\odot]$')
        ax.set_ylabel(r'$a_{*,\rm crit}$')
        ax.set_title(name)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'critical_spin_boundaries.png', dpi=200)
    plt.close(fig)


def make_exclusion_plot(results):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, meta in DATASETS.items():
        sub = results[results['dataset'] == name]
        ax.plot(sub['mu_ev'], sub['exclusion_prob'], lw=2.5, label=name, color=meta['color'])
    comb = results.groupby('mu_ev', as_index=False)['survival_prob'].prod()
    comb['combined_exclusion'] = 1.0 - comb['survival_prob']
    ax.plot(comb['mu_ev'], comb['combined_exclusion'], color='black', lw=3, label='Combined')
    ax.set_xscale('log')
    ax.set_xlabel(r'Boson mass $\mu$ [eV]')
    ax.set_ylabel('Posterior-averaged exclusion probability')
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'exclusion_probability.png', dpi=200)
    plt.close(fig)


def make_self_interaction_plot(self_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    pivot = self_df[self_df['dataset'] == 'Combined']
    for coupling in COUPLING_GRID:
        sub = pivot[pivot['coupling_GeV'] == coupling]
        ax.plot(sub['mu_ev'], sub['exclusion_prob'], lw=2, label=fr'$f_a={coupling:.0e}$ GeV')
    ax.axhline(0.95, color='gray', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r'Boson mass $\mu$ [eV]')
    ax.set_ylabel('Combined exclusion probability')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'self_interaction_sensitivity.png', dpi=200)
    plt.close(fig)


def make_validation_plot(results):
    fig, ax = plt.subplots(figsize=(9, 6))
    combined = results.groupby('mu_ev', as_index=False).agg(
        combined_exclusion=('survival_prob', lambda x: 1.0 - np.prod(x)),
        mean_alpha=('alpha_peak', 'mean'),
    )
    ax.plot(combined['mu_ev'], combined['mean_alpha'], lw=2.5, color='purple')
    ax.axhspan(ALPHA_MIN, ALPHA_MAX, color='purple', alpha=0.15, label='Superradiant alpha band')
    ax.set_xscale('log')
    ax.set_xlabel(r'Boson mass $\mu$ [eV]')
    ax.set_ylabel(r'Posterior-median $\alpha$ at each $\mu$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'model_validation.png', dpi=200)
    plt.close(fig)


def compute_kde_mode(log_values):
    kde = gaussian_kde(log_values)
    grid = np.linspace(log_values.min(), log_values.max(), 400)
    vals = kde(grid)
    return float(grid[np.argmax(vals)])


def main():
    samples = {name: load_samples(meta['file']) for name, meta in DATASETS.items()}

    summary_rows = [summarize_dataset(name, df) for name, df in samples.items()]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / 'data_summary.csv', index=False)

    make_overview_plots(samples)
    make_boundary_plot(samples)

    records = []
    for name, meta in DATASETS.items():
        df = samples[name]
        log_alpha = np.log10(alpha_of_mu_mass(MASS_GRID[:, None], df['mass_msun'].to_numpy()[None, :]))
        alpha_peak_by_mu = np.median(10 ** np.median(log_alpha, axis=1))
        for mu in MASS_GRID:
            alpha_samples = alpha_of_mu_mass(mu, df['mass_msun'].to_numpy())
            excl = exclusion_probability(df, mu, meta['tau_years'])
            records.append({
                'dataset': name,
                'mu_ev': mu,
                'exclusion_prob': excl,
                'survival_prob': 1.0 - excl,
                'alpha_peak': float(np.median(alpha_samples)),
                'tau_years': meta['tau_years'],
            })
    results = pd.DataFrame.from_records(records)
    results.to_csv(OUT_DIR / 'mass_grid_results.csv', index=False)

    make_exclusion_plot(results)
    make_validation_plot(results)

    self_records = []
    for coupling in COUPLING_GRID:
        per_dataset = []
        for name, meta in DATASETS.items():
            df = samples[name]
            for mu in MASS_GRID:
                excl = exclusion_probability(df, mu, meta['tau_years'], coupling=coupling)
                per_dataset.append({
                    'dataset': name,
                    'coupling_GeV': coupling,
                    'mu_ev': mu,
                    'exclusion_prob': excl,
                    'survival_prob': 1.0 - excl,
                })
        tmp = pd.DataFrame(per_dataset)
        combined = tmp.groupby('mu_ev', as_index=False)['survival_prob'].prod()
        combined['dataset'] = 'Combined'
        combined['coupling_GeV'] = coupling
        combined['exclusion_prob'] = 1.0 - combined['survival_prob']
        self_records.extend(tmp.to_dict(orient='records'))
        self_records.extend(combined.to_dict(orient='records'))
    self_df = pd.DataFrame(self_records)
    self_df.to_csv(OUT_DIR / 'self_interaction_results.csv', index=False)
    make_self_interaction_plot(self_df)

    combined = results.groupby('mu_ev', as_index=False)['survival_prob'].prod()
    combined['combined_exclusion'] = 1.0 - combined['survival_prob']
    threshold = combined[combined['combined_exclusion'] >= 0.95]
    strongest = combined.loc[combined['combined_exclusion'].idxmax()]

    summary = {
        'seed': SEED,
        'mass_grid_min_ev': float(MASS_GRID.min()),
        'mass_grid_max_ev': float(MASS_GRID.max()),
        'n_mass_grid': int(len(MASS_GRID)),
        'alpha_band': [ALPHA_MIN, ALPHA_MAX],
        'logistic_width': LOGISTIC_WIDTH,
        'fa_crit_GeV': F_A_CRIT,
        'max_combined_exclusion': float(strongest['combined_exclusion']),
        'mass_at_max_exclusion_ev': float(strongest['mu_ev']),
        'ulb_mass_95pct_region_min_ev': float(threshold['mu_ev'].min()) if len(threshold) else None,
        'ulb_mass_95pct_region_max_ev': float(threshold['mu_ev'].max()) if len(threshold) else None,
    }
    with open(OUT_DIR / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
