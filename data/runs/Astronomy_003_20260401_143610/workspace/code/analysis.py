"""
SXS Binary Black Hole Catalog: Waveform Accuracy Analysis
==========================================================
Reproduces and extends the waveform-error analysis from the SXS third catalog
paper, focusing on three aspects:
  - Fig 6: overall numerical resolution error distribution (1500 simulations)
  - Fig 7: per-mode (ell=2..8) waveform difference distributions
  - Fig 8: extrapolation-order comparison (N2vsN3, N2vsN4)

Outputs:
  outputs/summary_statistics.csv
  report/images/fig6_resolution_error.png
  report/images/fig7_modal_errors.png
  report/images/fig8_extrapolation_order.png
  report/images/fig_combined_overview.png
  report/images/fig_cdf_comparison.png
  report/images/fig_modal_scaling.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_003_20260401_143610'
DATA = os.path.join(BASE, 'data')
OUT  = os.path.join(BASE, 'outputs')
IMG  = os.path.join(BASE, 'report', 'images')

for d in (OUT, IMG):
    os.makedirs(d, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df6 = pd.read_csv(os.path.join(DATA, 'fig6_data.csv'))
df7 = pd.read_csv(os.path.join(DATA, 'fig7_data.csv'))
df8 = pd.read_csv(os.path.join(DATA, 'fig8_data.csv'))

vals6   = df6['waveform_difference'].values
modes7  = {f'ℓ={l}': df7[f'ell{l}'].values for l in range(2, 9)}
n2n3    = df8['N2vsN3'].values
n2n4    = df8['N2vsN4'].values

print(f"fig6: {len(vals6)} simulations")
print(f"fig7: {df7.shape[0]} simulations × {df7.shape[1]} modes")
print(f"fig8: {len(n2n3)} simulations")

# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def lognormal_fit(arr):
    """Return (mu, sigma) of the underlying normal distribution."""
    log_arr = np.log(arr[arr > 0])
    return log_arr.mean(), log_arr.std()

def percentile_table(arr, label=''):
    mu, sig = lognormal_fit(arr)
    med = np.exp(mu)
    return {
        'label': label,
        'n': len(arr),
        'median': np.median(arr),
        'lognormal_median': med,
        'lognormal_sigma': sig,
        'p10': np.percentile(arr, 10),
        'p25': np.percentile(arr, 25),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
        'p99': np.percentile(arr, 99),
        'frac_lt_1e3': (arr < 1e-3).mean(),
        'frac_lt_1e4': (arr < 1e-4).mean(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics → CSV
# ══════════════════════════════════════════════════════════════════════════════
rows = []
rows.append(percentile_table(vals6, 'Overall (all modes)'))
for key, v in modes7.items():
    rows.append(percentile_table(v, f'Mode {key}'))
rows.append(percentile_table(n2n3, 'Extrap N2vsN3'))
rows.append(percentile_table(n2n4, 'Extrap N2vsN4'))

stats_df = pd.DataFrame(rows)
stats_df.to_csv(os.path.join(OUT, 'summary_statistics.csv'), index=False)
print("\nSummary statistics saved.")
print(stats_df[['label','median','lognormal_sigma','frac_lt_1e3']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = plt.cm.plasma(np.linspace(0.1, 0.85, 7))

def log_hist(ax, arr, color, label=None, alpha=0.75, lw=1.5, bins=None):
    """Plot a log-spaced histogram as a step function."""
    arr = arr[arr > 0]
    if bins is None:
        bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 45)
    counts, edges = np.histogram(arr, bins=bins)
    ax.step(edges[:-1], counts, where='post', color=color, label=label,
            alpha=alpha, linewidth=lw)

def add_vline(ax, val, color, ls='--', lw=1.4, label=None):
    ax.axvline(val, color=color, linestyle=ls, linewidth=lw, label=label)

def lognormal_pdf_overlay(ax, arr, color, n_pts=300, scale=1.0):
    """Overlay the best-fit log-normal PDF (scaled to histogram counts)."""
    arr = arr[arr > 0]
    mu, sig = lognormal_fit(arr)
    xs = np.logspace(np.log10(arr.min()), np.log10(arr.max()), n_pts)
    pdf = stats.lognorm.pdf(xs, s=sig, scale=np.exp(mu))
    # scale to histogram bin area
    bin_width = np.diff(np.logspace(np.log10(arr.min()), np.log10(arr.max()), 45))
    total_area = (pdf * np.interp(xs, xs, np.ones_like(xs))).sum() * (xs[1]-xs[0])
    norm = len(arr) / (n_pts / (np.log(arr.max()/arr.min())))
    ax.plot(xs, pdf * len(arr) * np.mean(bin_width), color=color,
            linewidth=2, linestyle='-', alpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6  –  Overall waveform resolution error distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

bins6 = np.logspace(np.log10(vals6.min()*0.8), np.log10(vals6.max()*1.2), 50)
counts6, edges6 = np.histogram(vals6, bins=bins6)
ax.bar(edges6[:-1], counts6, width=np.diff(edges6), align='edge',
       color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85,
       label='SXS simulations (N=1500)')

# log-normal fit overlay
mu6, sig6 = lognormal_fit(vals6)
xs6 = np.logspace(np.log10(vals6.min()*0.8), np.log10(vals6.max()*1.2), 400)
pdf6 = stats.lognorm.pdf(xs6, s=sig6, scale=np.exp(mu6))
bw6  = np.mean(np.diff(edges6))
ax.plot(xs6, pdf6 * len(vals6) * bw6, color='navy', linewidth=2.2,
        label=fr'Log-normal fit ($\mu={mu6:.2f}$, $\sigma={sig6:.2f}$)')

med6 = np.median(vals6)
add_vline(ax, med6, color='tomato', ls='--', lw=2,
          label=fr'Median = {med6:.1e}')
add_vline(ax, 1e-3, color='gray', ls=':', lw=1.5, label='$10^{-3}$ threshold')

ax.set_xscale('log')
ax.set_xlabel('Waveform difference $\\delta h$ (minimal alignment)', fontsize=12)
ax.set_ylabel('Number of simulations', fontsize=12)
ax.set_title('Fig. 6  —  Numerical resolution error: SXS BBH catalog\n'
             r'$\delta h$ between two highest-resolution runs (all $\ell$ modes combined)',
             fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, which='both', alpha=0.25)

frac_good = (vals6 < 1e-3).mean() * 100
ax.text(0.97, 0.95,
        f'{frac_good:.0f}% of simulations\nhave $\\delta h < 10^{{-3}}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.85))

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig6_resolution_error.png'), dpi=150)
plt.close()
print("Saved fig6_resolution_error.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7  –  Per-mode (ℓ = 2 … 8) waveform difference distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=False, sharey=False)
axes_flat = axes.flatten()

ell_labels = [f'ℓ={l}' for l in range(2, 9)]
x_all = np.concatenate([v for v in modes7.values()])
global_bins = np.logspace(np.log10(x_all[x_all>0].min()*0.5),
                          np.log10(x_all.max()*1.5), 50)

for idx, (key, arr) in enumerate(modes7.items()):
    ax = axes_flat[idx]
    col = PALETTE[idx]
    counts, edges = np.histogram(arr, bins=global_bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
           color=col, edgecolor='white', linewidth=0.3, alpha=0.85)
    mu, sig = lognormal_fit(arr)
    xs = np.logspace(np.log10(arr.min()*0.8), np.log10(arr.max()*1.2), 400)
    pdf = stats.lognorm.pdf(xs, s=sig, scale=np.exp(mu))
    bw  = np.mean(np.diff(edges))
    ax.plot(xs, pdf * len(arr) * bw, color='k', linewidth=1.8)
    med = np.median(arr)
    ax.axvline(med, color='red', linestyle='--', linewidth=1.5,
               label=f'med = {med:.1e}')
    ax.set_xscale('log')
    ax.set_title(key, fontsize=11, color=col, fontweight='bold')
    ax.set_xlabel('$\\delta h$', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, which='both', alpha=0.2)

# hide unused panel
axes_flat[-1].set_visible(False)

fig.suptitle('Fig. 7  —  Per-mode waveform resolution error distributions\n'
             r'SXS BBH catalog (1500 simulations), $\ell = 2$ to $\ell = 8$',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig7_modal_errors.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved fig7_modal_errors.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 8  –  Extrapolation-order comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

for arr, color, label in [
    (n2n3, '#2196F3', 'N=2 vs N=3'),
    (n2n4, '#FF5722', 'N=2 vs N=4'),
]:
    bins = np.logspace(np.log10(arr[arr>0].min()*0.7),
                       np.log10(arr.max()*1.5), 50)
    counts, edges = np.histogram(arr, bins=bins)
    ax.step(edges[:-1], counts, where='post', color=color,
            linewidth=2.2, label=label, alpha=0.9)
    mu, sig = lognormal_fit(arr)
    xs = np.logspace(np.log10(arr[arr>0].min()*0.7),
                     np.log10(arr.max()*1.5), 400)
    pdf = stats.lognorm.pdf(xs, s=sig, scale=np.exp(mu))
    bw  = np.mean(np.diff(bins))
    ax.plot(xs, pdf * len(arr) * bw, color=color, linewidth=1.5,
            linestyle='--', alpha=0.7)
    med = np.median(arr)
    ax.axvline(med, color=color, linestyle=':', linewidth=1.5,
               label=f'med({label.split()[-1]}) = {med:.1e}')

ax.set_xscale('log')
ax.set_xlabel('Waveform difference $\\delta h$', fontsize=12)
ax.set_ylabel('Number of simulations', fontsize=12)
ax.set_title('Fig. 8  —  Waveform differences by extrapolation order\n'
             'SXS BBH catalog: N=2 vs N=3 and N=2 vs N=4 comparison',
             fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, which='both', alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig8_extrapolation_order.png'), dpi=150)
plt.close()
print("Saved fig8_extrapolation_order.png")

# ══════════════════════════════════════════════════════════════════════════════
# Additional Figure: Cumulative Distribution Functions (CDF) comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: CDF per mode
ax = axes[0]
for idx, (key, arr) in enumerate(modes7.items()):
    arr_s = np.sort(arr)
    cdf   = np.arange(1, len(arr_s)+1) / len(arr_s)
    ax.plot(arr_s, cdf, color=PALETTE[idx], linewidth=1.8, label=key)
# overlay overall
arr_s = np.sort(vals6)
cdf   = np.arange(1, len(arr_s)+1) / len(arr_s)
ax.plot(arr_s, cdf, color='black', linewidth=2.2, linestyle='--',
        label='All modes combined')
ax.set_xscale('log')
ax.set_xlabel('Waveform difference $\\delta h$', fontsize=11)
ax.set_ylabel('Cumulative fraction of simulations', fontsize=11)
ax.set_title('CDF: per-mode vs combined\nresolution error', fontsize=11)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.axvline(1e-3, color='gray', linestyle=':', linewidth=1.2)
ax.grid(True, which='both', alpha=0.25)

# Right: CDF for extrapolation orders
ax = axes[1]
for arr, color, label in [
    (n2n3, '#2196F3', 'N=2 vs N=3'),
    (n2n4, '#FF5722', 'N=2 vs N=4'),
    (vals6, 'black',  'Resolution error (all modes)'),
]:
    arr_s = np.sort(arr)
    cdf   = np.arange(1, len(arr_s)+1) / len(arr_s)
    ax.plot(arr_s, cdf, color=color, linewidth=1.8, label=label)
ax.set_xscale('log')
ax.set_xlabel('Waveform difference $\\delta h$', fontsize=11)
ax.set_ylabel('Cumulative fraction of simulations', fontsize=11)
ax.set_title('CDF: extrapolation order errors\nvs resolution error', fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.axvline(1e-3, color='gray', linestyle=':', linewidth=1.2)
ax.grid(True, which='both', alpha=0.25)

fig.suptitle('Cumulative Distribution Functions of Waveform Errors — SXS BBH Catalog',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_cdf_comparison.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved fig_cdf_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# Additional Figure: Modal median scaling and scatter
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ell_vals  = np.arange(2, 9)
medians7  = [np.median(modes7[f'ℓ={l}']) for l in ell_vals]
p25_7     = [np.percentile(modes7[f'ℓ={l}'], 25) for l in ell_vals]
p75_7     = [np.percentile(modes7[f'ℓ={l}'], 75) for l in ell_vals]
sigmas7   = [lognormal_fit(modes7[f'ℓ={l}'])[1] for l in ell_vals]

# Left: median vs ℓ with IQR band
ax = axes[0]
ax.fill_between(ell_vals, p25_7, p75_7, alpha=0.25, color='purple',
                label='IQR (25th–75th %ile)')
ax.plot(ell_vals, medians7, 'o-', color='purple', linewidth=2.2,
        markersize=7, label='Median $\\delta h$')
ax.set_yscale('log')
ax.set_xlabel('Spherical harmonic mode $\\ell$', fontsize=12)
ax.set_ylabel('Waveform difference $\\delta h$', fontsize=12)
ax.set_title('Median waveform error vs mode $\\ell$\nwith interquartile range', fontsize=11)
ax.set_xticks(ell_vals)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, which='both', alpha=0.3)

# Right: log-normal sigma vs ℓ
ax = axes[1]
ax.bar(ell_vals, sigmas7, color=PALETTE, edgecolor='white', linewidth=0.5,
       alpha=0.9)
ax.set_xlabel('Spherical harmonic mode $\\ell$', fontsize=12)
ax.set_ylabel('Log-normal $\\sigma$ of $\\delta h$', fontsize=12)
ax.set_title('Spread of waveform error distribution vs mode $\\ell$\n'
             '(log-normal shape parameter $\\sigma$)', fontsize=11)
ax.set_xticks(ell_vals)
ax.grid(True, axis='y', alpha=0.3)
for i, (l, s) in enumerate(zip(ell_vals, sigmas7)):
    ax.text(l, s + 0.01, f'{s:.2f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Mode-Dependent Waveform Error Statistics — SXS BBH Catalog',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_modal_scaling.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved fig_modal_scaling.png")

# ══════════════════════════════════════════════════════════════════════════════
# Additional Figure: Combined four-panel overview
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)

# Panel A: overall resolution error (violin + histogram overlay)
ax_a = fig.add_subplot(gs[0, 0])
bins_a = np.logspace(np.log10(vals6.min()*0.8), np.log10(vals6.max()*1.2), 50)
counts_a, edges_a = np.histogram(vals6, bins=bins_a)
ax_a.bar(edges_a[:-1], counts_a, width=np.diff(edges_a), align='edge',
         color='steelblue', edgecolor='white', linewidth=0.3, alpha=0.8)
mu, sig = lognormal_fit(vals6)
xs = np.logspace(np.log10(vals6.min()*0.8), np.log10(vals6.max()*1.2), 400)
pdf = stats.lognorm.pdf(xs, s=sig, scale=np.exp(mu))
ax_a.plot(xs, pdf*len(vals6)*np.mean(np.diff(edges_a)), 'navy', lw=2.2)
ax_a.axvline(np.median(vals6), color='tomato', ls='--', lw=2,
             label=f'Median={np.median(vals6):.1e}')
ax_a.set_xscale('log')
ax_a.set_xlabel('$\\delta h$', fontsize=10)
ax_a.set_ylabel('Count', fontsize=10)
ax_a.set_title('(A) Overall resolution error\nall modes, 1500 simulations', fontsize=10)
ax_a.legend(fontsize=8)
ax_a.grid(True, which='both', alpha=0.2)

# Panel B: mode-by-mode medians with IQR
ax_b = fig.add_subplot(gs[0, 1])
ax_b.fill_between(ell_vals, p25_7, p75_7, alpha=0.25, color='purple')
ax_b.plot(ell_vals, medians7, 'o-', color='purple', lw=2, ms=7,
          label='Median $\\delta h$')
ax_b.axhline(np.median(vals6), color='steelblue', ls='--', lw=1.5,
             label=f'Overall median={np.median(vals6):.1e}')
ax_b.set_yscale('log')
ax_b.set_xticks(ell_vals)
ax_b.set_xlabel('Mode $\\ell$', fontsize=10)
ax_b.set_ylabel('$\\delta h$', fontsize=10)
ax_b.set_title('(B) Median error per mode $\\ell$\nwith IQR band', fontsize=10)
ax_b.legend(fontsize=8)
ax_b.grid(True, which='both', alpha=0.2)

# Panel C: extrapolation order comparison
ax_c = fig.add_subplot(gs[1, 0])
for arr, color, label in [(n2n3, '#2196F3', 'N2 vs N3'), (n2n4, '#FF5722', 'N2 vs N4')]:
    bins_c = np.logspace(np.log10(arr[arr>0].min()*0.7), np.log10(arr.max()*1.5), 50)
    counts_c, edges_c = np.histogram(arr, bins=bins_c)
    ax_c.step(edges_c[:-1], counts_c, where='post', color=color, lw=2, label=label)
    ax_c.axvline(np.median(arr), color=color, ls=':', lw=1.5)
ax_c.set_xscale('log')
ax_c.set_xlabel('$\\delta h$', fontsize=10)
ax_c.set_ylabel('Count', fontsize=10)
ax_c.set_title('(C) Extrapolation-order errors\nN2 vs N3 and N2 vs N4', fontsize=10)
ax_c.legend(fontsize=8)
ax_c.grid(True, which='both', alpha=0.2)

# Panel D: CDF comparison
ax_d = fig.add_subplot(gs[1, 1])
for arr, color, label in [
    (vals6, 'steelblue', 'Resolution (all modes)'),
    (modes7['ℓ=2'], PALETTE[0], 'Mode ℓ=2'),
    (modes7['ℓ=8'], PALETTE[6], 'Mode ℓ=8'),
    (n2n3, '#2196F3', 'Extrap N2vsN3'),
    (n2n4, '#FF5722', 'Extrap N2vsN4'),
]:
    arr_s = np.sort(arr)
    cdf = np.arange(1, len(arr_s)+1)/len(arr_s)
    ax_d.plot(arr_s, cdf, color=color, lw=1.8, label=label)
ax_d.set_xscale('log')
ax_d.set_xlabel('$\\delta h$', fontsize=10)
ax_d.set_ylabel('Cumulative fraction', fontsize=10)
ax_d.set_title('(D) CDF comparison across\nerror sources', fontsize=10)
ax_d.legend(fontsize=7.5, framealpha=0.9)
ax_d.axvline(1e-3, color='gray', ls=':', lw=1.2)
ax_d.grid(True, which='both', alpha=0.2)

fig.suptitle('SXS Binary Black Hole Catalog — Waveform Error Analysis Overview',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig(os.path.join(IMG, 'fig_combined_overview.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved fig_combined_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# Violin plot for per-mode distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

data_violin = [np.log10(modes7[f'ℓ={l}']) for l in range(2, 9)]
parts = ax.violinplot(data_violin, positions=ell_vals, showmedians=True,
                      showextrema=True)

# color each violin
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(PALETTE[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
parts['cmedians'].set_color('red')
parts['cmedians'].set_linewidth(2.5)
parts['cmaxes'].set_linewidth(1)
parts['cmins'].set_linewidth(1)
parts['cbars'].set_linewidth(1)

# overlay scatter (subsampled)
rng = np.random.default_rng(42)
for i, l in enumerate(range(2, 9)):
    yv = np.log10(modes7[f'ℓ={l}'])
    jitter = rng.uniform(-0.15, 0.15, size=min(200, len(yv)))
    sub = rng.choice(len(yv), size=min(200, len(yv)), replace=False)
    ax.scatter(l + jitter, yv[sub], s=4, color=PALETTE[i], alpha=0.35, zorder=3)

ax.set_xticks(ell_vals)
ax.set_xticklabels([f'ℓ={l}' for l in ell_vals], fontsize=10)
ax.set_ylabel(r'$\log_{10}(\delta h)$', fontsize=12)
ax.set_xlabel('Spherical harmonic mode $\\ell$', fontsize=12)
ax.set_title('Per-mode waveform error distributions (violin plot)\nSXS BBH catalog, 1500 simulations',
             fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

legend_elements = [Line2D([0], [0], color='red', linewidth=2.5, label='Median')]
ax.legend(handles=legend_elements, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig7_violin.png'), dpi=150)
plt.close()
print("Saved fig7_violin.png")

# ══════════════════════════════════════════════════════════════════════════════
# Print final summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nOverall resolution error (fig6):")
print(f"  N simulations: {len(vals6)}")
print(f"  Median δh: {np.median(vals6):.3e}")
mu6, sig6 = lognormal_fit(vals6)
print(f"  Log-normal fit: median={np.exp(mu6):.3e}, σ={sig6:.3f}")
print(f"  Fraction with δh < 10^-3: {(vals6<1e-3).mean()*100:.1f}%")
print(f"  Fraction with δh < 10^-4: {(vals6<1e-4).mean()*100:.1f}%")

print(f"\nPer-mode medians (fig7):")
for l in range(2, 9):
    v = modes7[f'ℓ={l}']
    print(f"  ℓ={l}: median={np.median(v):.3e}, σ_lognorm={lognormal_fit(v)[1]:.3f}")

print(f"\nExtrapolation-order comparison (fig8):")
print(f"  N2vsN3: median={np.median(n2n3):.3e}, σ_lognorm={lognormal_fit(n2n3)[1]:.3f}")
print(f"  N2vsN4: median={np.median(n2n4):.3e}, σ_lognorm={lognormal_fit(n2n4)[1]:.3f}")

print("\nAll figures and statistics saved successfully.")
