import json
from pathlib import Path
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

ROOT = Path('.')
DATA = ROOT / 'data' / 'glambie'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def region_title(region: str) -> str:
    return region.replace('_', ' ').title()


def parse_method_group_from_filename(stem: str):
    region_names = [
        'alaska', 'western_canada_us', 'arctic_canada_north', 'arctic_canada_south',
        'greenland_periphery', 'iceland', 'svalbard', 'scandinavia', 'russian_arctic',
        'north_asia', 'central_europe', 'caucasus_middle_east', 'central_asia',
        'south_asia_west', 'south_asia_east', 'low_latitudes', 'southern_andes',
        'new_zealand', 'antarctic_and_subantarctic'
    ]
    for region in sorted(region_names, key=len, reverse=True):
        prefix = region + '_'
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            return region, rest.split('_')[0]
    parts = stem.split('_')
    return '_'.join(parts[:-3]), parts[-3]

def read_input_inventory():
    rows = []
    for f in sorted(glob.glob(str(DATA / 'input' / '*' / '*.csv'))):
        path = Path(f)
        region_folder = path.parent.name
        fname = path.stem
        region, method_group = parse_method_group_from_filename(fname)
        df = pd.read_csv(path)
        unit = str(df['unit'].iloc[0]) if len(df) else np.nan
        rows.append({
            'file': str(path.relative_to(ROOT)),
            'region_folder': region_folder,
            'region': region,
            'method_group_raw': method_group,
            'author': df['author'].iloc[0] if 'author' in df.columns and len(df) else np.nan,
            'unit': unit,
            'n_rows': len(df),
            'start_min': df['start_dates'].min() if 'start_dates' in df.columns else np.nan,
            'end_max': df['end_dates'].max() if 'end_dates' in df.columns else np.nan,
        })
    inv = pd.DataFrame(rows)
    mapping = {
        'glaciological': 'glaciological',
        'demdiff': 'dem_differencing',
        'altimetry': 'altimetry',
        'gravimetry': 'gravimetry',
        'combined': 'combined',
    }
    inv['method_group'] = inv['method_group_raw'].map(mapping).fillna(inv['method_group_raw'])
    return inv


def read_calendar_results():
    dfs = []
    for f in sorted((DATA / 'results' / 'calendar_years').glob('*.csv')):
        df = pd.read_csv(f)
        df['source_file'] = f.name
        dfs.append(df)
    cal = pd.concat(dfs, ignore_index=True)
    return cal


def read_hydro_results():
    dfs = []
    for f in sorted((DATA / 'results' / 'hydrological_years').glob('*.csv')):
        df = pd.read_csv(f)
        df['source_file'] = f.name
        dfs.append(df)
    hydro = pd.concat(dfs, ignore_index=True)
    return hydro


def add_cumulative(df, value_col, group_col='region'):
    out = df.sort_values([group_col, 'start_dates']).copy()
    out[f'cumulative_{value_col}'] = out.groupby(group_col)[value_col].cumsum()
    return out


def summarize_trends(calendar):
    reg = calendar[calendar['region'] != 'global'].copy()
    rows = []
    for region, g in reg.groupby('region'):
        x = g['start_dates'].values
        y = g['combined_gt'].values
        lr = linregress(x, y)
        total_gt = g['combined_gt'].sum()
        total_mwe = g['combined_mwe'].sum()
        mean_gt = g['combined_gt'].mean()
        mean_mwe = g['combined_mwe'].mean()
        err_rss_gt = float(np.sqrt(np.square(g['combined_gt_errors']).sum()))
        err_rss_mwe = float(np.sqrt(np.square(g['combined_mwe_errors']).sum()))
        rows.append({
            'region': region,
            'start_year': g['start_dates'].min(),
            'end_year': g['end_dates'].max(),
            'mean_annual_gt': mean_gt,
            'mean_annual_mwe': mean_mwe,
            'cumulative_gt_2000_2023': total_gt,
            'cumulative_gt_rss_error': err_rss_gt,
            'cumulative_mwe_2000_2023': total_mwe,
            'cumulative_mwe_rss_error': err_rss_mwe,
            'annual_gt_trend_per_year': lr.slope,
            'annual_gt_trend_pvalue': lr.pvalue,
            'min_annual_gt': g['combined_gt'].min(),
            'max_annual_gt': g['combined_gt'].max(),
            'mean_area_km2': g['glacier_area'].mean(),
            'area_change_pct': 100 * (g['glacier_area'].iloc[-1] - g['glacier_area'].iloc[0]) / g['glacier_area'].iloc[0],
        })
    return pd.DataFrame(rows).sort_values('cumulative_gt_2000_2023')


def compute_global_stats(calendar):
    g = calendar[calendar['region'] == 'global'].sort_values('start_dates').copy()
    lr = linregress(g['start_dates'], g['combined_gt'])
    stats = {
        'period': f"{int(g['start_dates'].min())}-{int(g['end_dates'].max())}",
        'mean_annual_gt': float(g['combined_gt'].mean()),
        'mean_annual_gt_unc_mean': float(g['combined_gt_errors'].mean()),
        'mean_annual_mwe': float(g['combined_mwe'].mean()),
        'cumulative_gt': float(g['combined_gt'].sum()),
        'cumulative_gt_rss_error': float(np.sqrt(np.square(g['combined_gt_errors']).sum())),
        'cumulative_mwe': float(g['combined_mwe'].sum()),
        'cumulative_mwe_rss_error': float(np.sqrt(np.square(g['combined_mwe_errors']).sum())),
        'annual_gt_trend_per_year': float(lr.slope),
        'annual_gt_trend_pvalue': float(lr.pvalue),
        'max_loss_year': int(g.loc[g['combined_gt'].idxmin(), 'start_dates']),
        'max_loss_gt': float(g['combined_gt'].min()),
        'least_loss_year': int(g.loc[g['combined_gt'].idxmax(), 'start_dates']),
        'least_loss_gt': float(g['combined_gt'].max()),
        'area_change_pct': float(100 * (g['glacier_area'].iloc[-1] - g['glacier_area'].iloc[0]) / g['glacier_area'].iloc[0]),
    }
    return stats


def compare_method_groups(hydro):
    rows = []
    method_prefixes = ['altimetry', 'gravimetry', 'demdiff_and_glaciological']
    for region, g in hydro.groupby('region'):
        for prefix in method_prefixes:
            gt = f'{prefix}_gt'
            err = f'{prefix}_gt_errors'
            if gt not in g.columns:
                continue
            sub = g[['start_dates', 'combined_gt', 'combined_gt_errors', gt, err]].dropna()
            if len(sub) == 0:
                continue
            diff = sub[gt] - sub['combined_gt']
            mae = np.abs(diff).mean()
            bias = diff.mean()
            rmse = np.sqrt(np.mean(diff ** 2))
            corr = sub[[gt, 'combined_gt']].corr().iloc[0, 1] if len(sub) > 1 else np.nan
            rows.append({
                'region': region,
                'method': prefix,
                'n_years': len(sub),
                'bias_gt': bias,
                'mae_gt': mae,
                'rmse_gt': rmse,
                'corr_with_combined': corr,
                'mean_method_error_gt': sub[err].mean(),
            })
    return pd.DataFrame(rows)


def make_figures(calendar, hydro, region_summary, inventory):
    reg = add_cumulative(calendar[calendar['region'] != 'global'].copy(), 'combined_gt')
    glob_df = add_cumulative(calendar[calendar['region'] == 'global'].copy(), 'combined_gt')

    # Figure 1: data overview
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = inventory.groupby('region')['file'].count().sort_values(ascending=False).index
    sns.countplot(data=inventory, y='region', hue='method_group', order=order, ax=axes[0])
    axes[0].set_title('Input estimates by region and method group')
    axes[0].set_xlabel('Number of input datasets')
    axes[0].set_ylabel('Region')
    axes[0].legend(title='Method group', fontsize=10, title_fontsize=11)

    unit_counts = inventory.groupby(['method_group', 'unit']).size().reset_index(name='count')
    sns.barplot(data=unit_counts, x='method_group', y='count', hue='unit', ax=axes[1])
    axes[1].set_title('Input dataset units by method group')
    axes[1].set_xlabel('Method group')
    axes[1].set_ylabel('Number of files')
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].legend(title='Unit')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_data_overview.png')
    plt.close(fig)

    # Figure 2: global annual + cumulative
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    g = glob_df.sort_values('start_dates')
    axes[0].bar(g['start_dates'], g['combined_gt'], color=np.where(g['combined_gt'] < 0, '#2166ac', '#b2182b'), width=0.8)
    axes[0].fill_between(g['start_dates'], g['combined_gt'] - g['combined_gt_errors'], g['combined_gt'] + g['combined_gt_errors'], color='gray', alpha=0.25)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_ylabel('Annual mass change (Gt yr$^{-1}$)')
    axes[0].set_title('Global glacier mass change, calendar years')

    axes[1].plot(g['end_dates'], g['cumulative_combined_gt'], color='black', linewidth=2.5)
    axes[1].fill_between(g['end_dates'], g['cumulative_combined_gt'] - np.sqrt(np.cumsum(g['combined_gt_errors'].values**2)), g['cumulative_combined_gt'] + np.sqrt(np.cumsum(g['combined_gt_errors'].values**2)), color='gray', alpha=0.25)
    axes[1].set_ylabel('Cumulative mass change since 2000 (Gt)')
    axes[1].set_xlabel('Year')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_global_timeseries.png')
    plt.close(fig)

    # Figure 3: regional cumulative contributions
    latest = reg.groupby('region').tail(1).sort_values('cumulative_combined_gt')
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#2166ac' if x < 0 else '#b2182b' for x in latest['cumulative_combined_gt']]
    ax.barh(latest['region'], latest['cumulative_combined_gt'], color=colors)
    ax.set_title('Regional cumulative glacier mass change, 2000–2023')
    ax.set_xlabel('Cumulative mass change (Gt)')
    ax.set_ylabel('Region')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_regional_cumulative_gt.png')
    plt.close(fig)

    # Figure 4: specific mass change heatmap
    pivot = calendar[calendar['region'] != 'global'].pivot(index='region', columns='start_dates', values='combined_mwe')
    pivot = pivot.loc[region_summary.sort_values('cumulative_mwe_2000_2023').region]
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Specific mass change (m w.e. yr$^{-1}$)'})
    ax.set_title('Regional annual specific mass change (calendar years)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Region')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_specific_mass_change_heatmap.png')
    plt.close(fig)

    # Figure 5: method comparison scatter by method
    method_prefixes = ['altimetry', 'gravimetry', 'demdiff_and_glaciological']
    method_names = {'altimetry': 'Altimetry', 'gravimetry': 'Gravimetry', 'demdiff_and_glaciological': 'DEM differencing + glaciological'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    for ax, prefix in zip(axes, method_prefixes):
        gt = f'{prefix}_gt'
        subset = hydro[['region', 'combined_gt', gt]].dropna()
        if len(subset) == 0:
            ax.axis('off')
            continue
        sns.scatterplot(data=subset, x='combined_gt', y=gt, hue='region', legend=False, ax=ax, s=50)
        lim = max(np.abs(subset['combined_gt']).max(), np.abs(subset[gt]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], linestyle='--', color='black', linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(method_names[prefix])
        ax.set_xlabel('Combined estimate (Gt yr$^{-1}$)')
        ax.set_ylabel('Method estimate (Gt yr$^{-1}$)')
    fig.suptitle('Agreement between method-group estimates and the combined solution', y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / 'figure_method_comparison.png')
    plt.close(fig)

    # Figure 6: uncertainty vs magnitude
    r = calendar[calendar['region'] != 'global'].copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=r, x=np.abs(r['combined_gt']), y=r['combined_gt_errors'], hue='region', ax=ax)
    ax.set_xlabel('Absolute annual mass change (Gt yr$^{-1}$)')
    ax.set_ylabel('Annual uncertainty (Gt yr$^{-1}$)')
    ax.set_title('Regional uncertainty scales with signal magnitude')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_uncertainty_vs_signal.png')
    plt.close(fig)


def main():
    inventory = read_input_inventory()
    calendar = read_calendar_results()
    hydro = read_hydro_results()
    calendar = add_cumulative(calendar, 'combined_gt')
    calendar = add_cumulative(calendar, 'combined_mwe')

    region_summary = summarize_trends(calendar)
    global_stats = compute_global_stats(calendar)
    method_comp = compare_method_groups(hydro)

    inventory.to_csv(OUT / 'input_inventory.csv', index=False)
    calendar.to_csv(OUT / 'calendar_results_with_cumulative.csv', index=False)
    hydro.to_csv(OUT / 'hydrological_results.csv', index=False)
    region_summary.to_csv(OUT / 'regional_summary.csv', index=False)
    method_comp.to_csv(OUT / 'method_comparison_summary.csv', index=False)
    with open(OUT / 'global_stats.json', 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, indent=2)

    # additional summary tables for reporting
    reg = calendar[calendar['region'] != 'global'].copy()
    annual_rank = reg.groupby('region')['combined_gt'].mean().sort_values()
    annual_specific_rank = reg.groupby('region')['combined_mwe'].mean().sort_values()
    pd.DataFrame({'region': annual_rank.index, 'mean_annual_gt': annual_rank.values}).to_csv(OUT / 'rank_mean_annual_gt.csv', index=False)
    pd.DataFrame({'region': annual_specific_rank.index, 'mean_annual_mwe': annual_specific_rank.values}).to_csv(OUT / 'rank_mean_annual_mwe.csv', index=False)

    make_figures(calendar, hydro, region_summary, inventory)

    # concise metrics for report drafting
    metrics = {
        'n_input_files': int(len(inventory)),
        'n_regions': int(calendar['region'].nunique() - 1),
        'n_calendar_years_global': int((calendar['region'] == 'global').sum()),
        'method_counts': inventory['method_group'].value_counts().to_dict(),
        'unit_counts': inventory['unit'].value_counts().to_dict(),
        'top_5_cumulative_loss_regions': region_summary[['region', 'cumulative_gt_2000_2023']].head(5).to_dict(orient='records'),
        'least_negative_regions_by_gt': region_summary[['region', 'cumulative_gt_2000_2023']].tail(5).to_dict(orient='records'),
    }
    with open(OUT / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('Analysis complete')
    print(json.dumps(global_stats, indent=2))
    print(region_summary[['region', 'cumulative_gt_2000_2023', 'cumulative_mwe_2000_2023']].head(10).to_string(index=False))
    print(method_comp.groupby('method')[['bias_gt', 'mae_gt', 'rmse_gt', 'corr_with_combined']].mean().to_string())


if __name__ == '__main__':
    main()
