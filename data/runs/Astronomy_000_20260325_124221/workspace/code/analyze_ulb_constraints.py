
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

MSUN_SEC = 4.92549095e-6  # GM/c^3 in seconds for one solar mass
HBAR_EV_S = 6.582119569e-16

def omega_horizon(a):
    a = np.clip(a, 1e-6, 0.999999)
    return a / (2.0 * (1.0 + np.sqrt(1.0 - a * a)))

def alpha_from_mu(M_msun, mu_ev):
    return M_msun * MSUN_SEC * mu_ev / HBAR_EV_S

def critical_spin(alpha, m=1):
    x = 4.0 * alpha / m
    x = np.clip(x, 1e-9, 0.999999)
    return 4.0 * x / (1.0 + x * x)

def exclusion_probability(M, a, mu_ev, delta=0.03, m=1):
    alpha = alpha_from_mu(M, mu_ev)
    acrit = critical_spin(alpha, m=m)
    z = (a - acrit) / delta
    return 1.0 / (1.0 + np.exp(z))

def coupling_penalty(g, g0=1e-17, beta=2.0):
    g = np.asarray(g)
    return 1.0 / (1.0 + (g / g0) ** beta)

def credible_upper(x, p=0.95):
    x = np.asarray(x)
    y = np.sort(x)
    idx = int(np.clip(np.ceil(p * len(y)) - 1, 0, len(y) - 1))
    return float(y[idx])

def credible_interval(x, p=0.9):
    lo = (1-p)/2
    hi = 1-lo
    return np.quantile(x, [lo, 0.5, hi]).tolist()

def summarize_samples(name, arr):
    q = np.quantile(arr, [0.05,0.16,0.5,0.84,0.95])
    return {
        'name': name,
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)),
        'q05': float(q[0]),
        'q16': float(q[1]),
        'q50': float(q[2]),
        'q84': float(q[3]),
        'q95': float(q[4]),
    }

def make_kde_grid(x, y, gridsize=160):
    xmin, xmax = np.quantile(x, [0.01, 0.99])
    ymin, ymax = np.quantile(y, [0.01, 0.99])
    xpad = 0.1*(xmax-xmin)
    ypad = 0.1*(ymax-ymin)
    xs = np.linspace(max(0, xmin-xpad), xmax+xpad, gridsize)
    ys = np.linspace(max(0, ymin-ypad), min(0.999, ymax+ypad), gridsize)
    X,Y = np.meshgrid(xs, ys)
    if HAVE_SCIPY:
        kde = gaussian_kde(np.vstack([x,y]))
        Z = kde(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape)
    else:
        Z = np.zeros_like(X)
    return X,Y,Z

def main():
    base = Path('.')
    data_dir = base/'data'
    out_dir = base/'outputs'
    img_dir = base/'report'/'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='whitegrid', context='talk')

    datasets = {
        'IRAS 09149-6206': np.loadtxt(data_dir/'IRAS_09149-6206_samples.dat'),
        'M33 X-7': np.loadtxt(data_dir/'M33_X-7_samples.dat'),
    }

    mu_grid = np.logspace(-21, -9, 500)
    g_grid = np.logspace(-20, -15, 240)
    results = {'datasets': {}, 'combined': {}}
    combined_survival = np.ones_like(mu_grid)

    fig, axes = plt.subplots(1,2, figsize=(14,6), constrained_layout=True)
    for ax, (name, data) in zip(axes, datasets.items()):
        M = data[:,0]
        a = data[:,1]
        ax.scatter(M, a, s=10, alpha=0.25)
        ax.set_title(name)
        ax.set_xlabel(r'$M_{\rm BH}\,[M_\odot]$')
        ax.set_ylabel(r'$a_\ast$')
        ax.set_xscale('log')
        ax.set_ylim(0,1)
    fig.savefig(img_dir/'posterior_samples.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1,2, figsize=(14,6), constrained_layout=True)
    dataset_rows = []

    for ax, (name, data) in zip(axes, datasets.items()):
        M = data[:,0]
        a = data[:,1]
        surv = np.array([np.mean(1.0 - exclusion_probability(M, a, mu)) for mu in mu_grid])
        post = surv / np.trapz(surv, np.log(mu_grid))
        combined_survival *= surv

        upper95 = credible_upper(np.repeat(mu_grid, np.maximum((post/post.max()*300).astype(int),1)), 0.95)
        q = np.quantile(M, [0.16,0.5,0.84]).tolist() + np.quantile(a,[0.16,0.5,0.84]).tolist()
        dataset_rows.append({
            'dataset': name,
            'n_samples': int(len(M)),
            'mass_q16': q[0], 'mass_q50': q[1], 'mass_q84': q[2],
            'spin_q16': q[3], 'spin_q50': q[4], 'spin_q84': q[5],
            'corr_M_a': float(np.corrcoef(M,a)[0,1]),
            'mu95_upper_eV': upper95,
            'most_excluded_mu_eV': float(mu_grid[np.argmin(surv)]),
            'min_survival': float(np.min(surv)),
        })

        ax.plot(mu_grid, surv, lw=2, label='survival probability')
        ax.axhline(0.05, color='k', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1.1)
        ax.set_xlabel(r'$\mu\,[\mathrm{eV}]$')
        ax.set_ylabel('Allowed probability')
        ax.set_title(name)
        ax.legend()

        X,Y,Z = make_kde_grid(M,a)
        fig2, ax2 = plt.subplots(figsize=(7,6), constrained_layout=True)
        ax2.scatter(M, a, s=8, alpha=0.15, color='tab:blue')
        if Z.max() > 0:
            levs = np.quantile(Z[Z>0], [0.70,0.88,0.96])
            ax2.contour(X, Y, Z, levels=np.unique(levs), colors='tab:blue', linewidths=1.5)
        for mu, color in [(dataset_rows[-1]['most_excluded_mu_eV'], 'crimson'), (3e-12 if name=='M33 X-7' else 3e-18, 'orange')]:
            mg = np.logspace(np.log10(max(M.min()*0.6,1e-1)), np.log10(M.max()*1.6), 300)
            ag = critical_spin(alpha_from_mu(mg, mu))
            ax2.plot(mg, ag, color=color, lw=2, label=fr'$\mu={mu:.1e}\,$eV')
        ax2.set_xscale('log')
        ax2.set_ylim(0,1)
        ax2.set_xlabel(r'$M_{\rm BH}\,[M_\odot]$')
        ax2.set_ylabel(r'$a_\ast$')
        ax2.set_title(f'{name}: posterior + Regge trajectories')
        ax2.legend(fontsize=10)
        fig2.savefig(img_dir/f"regge_{name.lower().replace(' ','_').replace('-','_')}.png", dpi=200)
        plt.close(fig2)

        G,MU = np.meshgrid(g_grid, mu_grid)
        like2d = surv[:,None] * coupling_penalty(G)
        fig3, ax3 = plt.subplots(figsize=(7,6), constrained_layout=True)
        pcm = ax3.pcolormesh(mu_grid, g_grid, like2d.T, shading='auto', cmap='viridis')
        ax3.set_xscale('log'); ax3.set_yscale('log')
        ax3.set_xlabel(r'$\mu\,[\mathrm{eV}]$')
        ax3.set_ylabel(r'$g\,[\mathrm{arbitrary}]$')
        ax3.set_title(f'{name}: joint posterior proxy')
        cb = fig3.colorbar(pcm, ax=ax3)
        cb.set_label('Relative allowed probability')
        fig3.savefig(img_dir/f"coupling_{name.lower().replace(' ','_').replace('-','_')}.png", dpi=200)
        plt.close(fig3)

        # Coupling upper envelope from P(g|mu)=surv(mu)*penalty(g)
        g95_by_mu = []
        for s in surv:
            pg = s * coupling_penalty(g_grid)
            cdf = np.cumsum(pg)
            cdf = cdf / cdf[-1]
            g95_by_mu.append(float(g_grid[np.searchsorted(cdf, 0.95)]))
        results['datasets'][name] = {
            'summary': dataset_rows[-1],
            'survival_mu': surv.tolist(),
            'mu_grid': mu_grid.tolist(),
            'g_grid': g_grid.tolist(),
            'g95_by_mu': g95_by_mu,
        }

    fig.savefig(img_dir/'allowed_probability_vs_mass.png', dpi=200)
    plt.close(fig)

    comb = combined_survival / np.trapz(combined_survival, np.log(mu_grid))
    mu95_comb = credible_upper(np.repeat(mu_grid, np.maximum((comb/comb.max()*400).astype(int),1)), 0.95)

    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    for name in datasets:
        ax.plot(mu_grid, np.array(results['datasets'][name]['survival_mu']), lw=2, alpha=0.8, label=name)
    ax.plot(mu_grid, combined_survival, lw=3, color='k', label='Combined')
    ax.axhline(0.05, color='k', ls='--', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-6,1.1)
    ax.set_xlabel(r'$\mu\,[\mathrm{eV}]$'); ax.set_ylabel('Allowed probability')
    ax.set_title('Combined exclusion from full posterior samples')
    ax.legend()
    fig.savefig(img_dir/'combined_allowed_probability.png', dpi=200)
    plt.close(fig)

    # validation with point-estimate approximation
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    for name, data in datasets.items():
        M = data[:,0]; a = data[:,1]
        surv = np.array(results['datasets'][name]['survival_mu'])
        M0 = np.median(M); a0 = np.median(a)
        surv_point = 1.0 - exclusion_probability(M0, a0, mu_grid)
        ax.plot(mu_grid, surv, lw=2, label=f'{name} full posterior')
        ax.plot(mu_grid, surv_point, lw=2, ls='--', label=f'{name} median-only')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-4,1.1)
    ax.set_xlabel(r'$\mu\,[\mathrm{eV}]$'); ax.set_ylabel('Allowed probability')
    ax.set_title('Validation: full posterior versus point-estimate analysis')
    ax.legend(fontsize=10)
    fig.savefig(img_dir/'validation_full_vs_point.png', dpi=200)
    plt.close(fig)

    # combined 2D proxy
    G,MU = np.meshgrid(g_grid, mu_grid)
    joint = combined_survival[:,None] * coupling_penalty(G)
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    pcm = ax.pcolormesh(mu_grid, g_grid, joint.T, shading='auto', cmap='magma')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\mu\,[\mathrm{eV}]$'); ax.set_ylabel(r'$g\,[\mathrm{arbitrary}]$')
    ax.set_title('Combined posterior proxy in boson mass–coupling space')
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Relative allowed probability')
    fig.savefig(img_dir/'combined_mass_coupling_posterior.png', dpi=200)
    plt.close(fig)

    results['combined'] = {
        'mu_grid': mu_grid.tolist(),
        'allowed_probability': combined_survival.tolist(),
        'mu95_upper_eV': mu95_comb,
        'most_excluded_mu_eV': float(mu_grid[np.argmin(combined_survival)]),
        'min_survival': float(np.min(combined_survival)),
        'dataset_table': dataset_rows,
        'model_notes': {
            'critical_spin_formula': 'a_crit = 4 x / (1 + x^2), x = 4 alpha / m, alpha = G M mu / (hbar c^3)',
            'softening_delta': 0.03,
            'coupling_penalty': '1 / (1 + (g/g0)^beta), g0=1e-17, beta=2',
            'interpretation': 'Posterior proxy, not a fully calibrated physical self-interaction likelihood.'
        }
    }

    pd.DataFrame(dataset_rows).to_csv(out_dir/'dataset_summary.csv', index=False)
    with open(out_dir/'results.json','w',encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results['combined'], indent=2)[:4000])

if __name__ == '__main__':
    main()
