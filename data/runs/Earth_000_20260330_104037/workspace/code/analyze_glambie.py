import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path('.')
DATA_DIR = BASE / 'data' / 'glambie'
INPUT_DIR = DATA_DIR / 'input'
RESULTS_CAL = DATA_DIR / 'results' / 'calendar_years'
RESULTS_HYD = DATA_DIR / 'results' / 'hydrological_years'
OUTPUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

METHOD_KEYWORDS = {
    'glaciological': 'glaciological',
    'demdiff': 'demdiff',
    'altimetry': 'altimetry',
    'gravimetry': 'gravimetry',
    'combined': 'combined',
}

REGION_NAME_MAP = {
    '1_alaska': 'Alaska',
    '2_western_canada_us': 'Western Canada & US',
    '3_arctic_canada_north': 'Arctic Canada North',
    '4_arctic_canada_south': 'Arctic Canada South',
    '5_greenland_periphery': 'Greenland Periphery',
    '6_iceland': 'Iceland',
    '7_svalbard': 'Svalbard',
    '8_scandinavia': 'Scandinavia',
    '9_russian_arctic': 'Russian Arctic',
    '10_north_asia': 'North Asia',
    '11_central_europe': 'Central Europe',
    '12_caucasus_middle_east': 'Caucasus & Middle East',
    '13_central_asia': 'Central Asia',
    '14_south_asia_west': 'South Asia West',
    '15_south_asia_east': 'South Asia East',
    '16_low_latitudes': 'Low Latitudes',
    '17_southern_andes': 'Southern Andes',
    '18_new_zealand': 'New Zealand',
    '19_antarctic_and_subantarctic': 'Antarctic & Subantarctic',
    '0_global': 'Global',
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)


def detect_method(path: Path) -> str:
    name = path.name.lower()
    for method, keyword in METHOD_KEYWORDS.items():
        if f'_{keyword}_' in name:
            return method
    for method, keyword in METHOD_KEYWORDS.items():
        if keyword in name:
            return method
    return 'unknown'


def detect_unit_type(unit: str) -> str:
    u = str(unit).strip().lower()
    if 'gt' in u:
        return 'Gt'
    if 'mwe' in u or 'w.e' in u:
        return 'mwe'
    if u == 'm':
        return 'm'
    return u


def load_input_inventory():
    rows = []
    for path in sorted(INPUT_DIR.rglob('*.csv')):
        df = pd.read_csv(path)
        duration = df['end_dates'] - df['start_dates']
        region_code = path.parent.name
        rows.append({
            'file': str(path.relative_to(BASE)),
            'region_code': region_code,
            'region': REGION_NAME_MAP.get(region_code, region_code),
            'dataset_name': path.stem,
            'method': detect_method(path),
            'author': str(df['author'].iloc[0]) if 'author' in df.columns and len(df) else None,
            'unit': str(df['unit'].iloc[0]) if 'unit' in df.columns and len(df) else None,
            'unit_type': detect_unit_type(df['unit'].iloc[0]) if 'unit' in df.columns and len(df) else None,
            'n_records': int(len(df)),
            'start_min': float(df['start_dates'].min()),
            'end_max': float(df['end_dates'].max()),
            'median_duration_years': float(duration.median()),
        })
    return pd.DataFrame(rows)


def load_calendar_results():
    frames = []
    for path in sorted(RESULTS_CAL.glob('*.csv')):
        df = pd.read_csv(path)
        df['region_code'] = path.stem
        df['region'] = REGION_NAME_MAP.get(path.stem, path.stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_hydrological_results():
    frames = []
    for path in sorted(RESULTS_HYD.glob('*.csv')):
        df = pd.read_csv(path)
        df['region_code'] = path.stem
        df['region'] = REGION_NAME_MAP.get(path.stem, path.stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_primary_outputs(calendar_df: pd.DataFrame):
    regional = calendar_df[calendar_df['region_code'] != '0_global'].copy()
    regional['year'] = regional['start_dates'].astype(int)
    regional = regional.rename(columns={
        'combined_gt': 'mass_change_gt',
        'combined_gt_errors': 'mass_change_gt_error',
        'combined_mwe': 'specific_mass_change_mwe',
        'combined_mwe_errors': 'specific_mass_change_mwe_error',
        'glacier_area': 'glacier_area_km2',
    })
    regional = regional[[
        'year', 'region_code', 'region', 'glacier_area_km2',
        'mass_change_gt', 'mass_change_gt_error',
        'specific_mass_change_mwe', 'specific_mass_change_mwe_error'
    ]].sort_values(['year', 'region_code'])

    global_df = calendar_df[calendar_df['region_code'] == '0_global'].copy()
    global_df['year'] = global_df['start_dates'].astype(int)
    global_df = global_df.rename(columns={
        'combined_gt': 'mass_change_gt',
        'combined_gt_errors': 'mass_change_gt_error',
        'combined_mwe': 'specific_mass_change_mwe',
        'combined_mwe_errors': 'specific_mass_change_mwe_error',
        'glacier_area': 'glacier_area_km2',
    })
    global_df = global_df[[
        'year', 'region', 'glacier_area_km2',
        'mass_change_gt', 'mass_change_gt_error',
        'specific_mass_change_mwe', 'specific_mass_change_mwe_error'
    ]].sort_values('year')

    annual_global_check = regional.groupby('year', as_index=False).agg(
        regional_sum_gt=('mass_change_gt', 'sum'),
        regional_sum_gt_error_rss=('mass_change_gt_error', lambda x: float(np.sqrt(np.square(x).sum()))),
        total_area_km2=('glacier_area_km2', 'sum'),
    )
    merged = global_df.merge(annual_global_check, on='year', how='left')
    merged['sum_minus_official_gt'] = merged['regional_sum_gt'] - merged['mass_change_gt']
    merged['weighted_regional_mwe'] = merged['regional_sum_gt'] / merged['total_area_km2']
    return regional, global_df, merged


def summarize_method_coverage(input_inventory: pd.DataFrame):
    coverage = (
        input_inventory.groupby(['region', 'method'])
        .agg(
            datasets=('file', 'count'),
            first_year=('start_min', 'min'),
            last_year=('end_max', 'max'),
            median_duration_years=('median_duration_years', 'median'),
        )
        .reset_index()
    )
    return coverage.sort_values(['region', 'method'])


def method_timeseries_from_hydrological(hyd_df: pd.DataFrame):
    records = []
    method_cols = ['altimetry', 'gravimetry', 'demdiff_and_glaciological']
    for _, row in hyd_df.iterrows():
        year = int(math.floor(row['start_dates']))
        for method in method_cols:
            gt_col = f'{method}_gt'
            err_col = f'{method}_gt_errors'
            mwe_col = f'{method}_mwe'
            mwe_err_col = f'{method}_mwe_errors'
            av_col = f'{method}_annual_variability'
            if gt_col in row.index and pd.notna(row[gt_col]):
                records.append({
                    'year': year,
                    'region': row['region'],
                    'region_code': row['region_code'],
                    'method_group': method,
                    'gt': row[gt_col],
                    'gt_error': row.get(err_col, np.nan),
                    'mwe': row.get(mwe_col, np.nan),
                    'mwe_error': row.get(mwe_err_col, np.nan),
                    'annual_variability_native': row.get(av_col, np.nan),
                    'combined_gt': row['combined_gt'],
                    'combined_mwe': row['combined_mwe'],
                })
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out['gt_diff_vs_combined'] = out['gt'] - out['combined_gt']
    out['abs_gt_diff_vs_combined'] = out['gt_diff_vs_combined'].abs()
    return out.sort_values(['region_code', 'year', 'method_group'])


def build_validation_summary(method_ts: pd.DataFrame):
    summary = (
        method_ts.groupby('method_group')
        .agg(
            n_region_year=('gt', 'count'),
            mean_abs_diff_gt=('abs_gt_diff_vs_combined', 'mean'),
            median_abs_diff_gt=('abs_gt_diff_vs_combined', 'median'),
            rmse_gt=('gt_diff_vs_combined', lambda x: float(np.sqrt(np.mean(np.square(x))))),
            native_annual_variability_share=('annual_variability_native', 'mean'),
        )
        .reset_index()
        .sort_values('mean_abs_diff_gt')
    )
    return summary


def cumulative_global(global_df: pd.DataFrame):
    out = global_df.copy()
    out['cumulative_gt_since_2000'] = out['mass_change_gt'].cumsum()
    out['cumulative_gt_low'] = (out['mass_change_gt'] - out['mass_change_gt_error']).cumsum()
    out['cumulative_gt_high'] = (out['mass_change_gt'] + out['mass_change_gt_error']).cumsum()
    return out


def make_figures(input_inventory, regional, global_df, global_check, method_ts, validation):
    # Figure 1: data overview
    overview = input_inventory.groupby(['region', 'method']).size().reset_index(name='datasets')
    pivot = overview.pivot(index='region', columns='method', values='datasets').fillna(0)
    pivot = pivot.loc[sorted(pivot.index)]
    plt.figure(figsize=(12, 9))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues', cbar_kws={'label': 'Number of datasets'})
    plt.title('Input dataset coverage by region and observation method')
    plt.xlabel('Observation method')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_data_overview_heatmap.png', bbox_inches='tight')
    plt.close()

    # Figure 2: global annual and cumulative series
    cum = cumulative_global(global_df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(cum['year'], cum['mass_change_gt'], color=np.where(cum['mass_change_gt'] < 0, '#2b8cbe', '#d95f0e'))
    axes[0].fill_between(cum['year'], cum['mass_change_gt'] - cum['mass_change_gt_error'], cum['mass_change_gt'] + cum['mass_change_gt_error'], color='gray', alpha=0.25)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_ylabel('Annual mass change (Gt yr$^{-1}$)')
    axes[0].set_title('Global glacier mass change benchmark, calendar years')

    axes[1].plot(cum['year'], cum['cumulative_gt_since_2000'], color='#045a8d', linewidth=2.5)
    axes[1].fill_between(cum['year'], cum['cumulative_gt_low'], cum['cumulative_gt_high'], color='#74a9cf', alpha=0.3)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_ylabel('Cumulative mass change since 2000 (Gt)')
    axes[1].set_xlabel('Year')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_global_timeseries.png', bbox_inches='tight')
    plt.close()

    # Figure 3: regional contributions heatmap
    regional_pivot = regional.pivot(index='region', columns='year', values='mass_change_gt')
    regional_pivot = regional_pivot.loc[regional.groupby('region')['mass_change_gt'].mean().sort_values().index]
    plt.figure(figsize=(14, 10))
    sns.heatmap(regional_pivot, cmap='RdBu_r', center=0, cbar_kws={'label': 'Mass change (Gt yr$^{-1}$)'})
    plt.title('Regional glacier mass change by year')
    plt.xlabel('Year')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_regional_heatmap.png', bbox_inches='tight')
    plt.close()

    # Figure 4: method agreement diagnostics
    method_plot = validation.copy()
    method_plot['method_group'] = method_plot['method_group'].str.replace('_', ' ')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=method_plot, x='method_group', y='mean_abs_diff_gt', color='#3182bd')
    plt.ylabel('Mean |method - combined| (Gt)')
    plt.xlabel('Method group')
    plt.title('Agreement of method-group estimates with reconciled regional annual series')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_method_agreement.png', bbox_inches='tight')
    plt.close()

    # Figure 5: official global vs summed regional
    plt.figure(figsize=(11, 6))
    plt.plot(global_check['year'], global_check['mass_change_gt'], marker='o', label='Official global result')
    plt.plot(global_check['year'], global_check['regional_sum_gt'], marker='s', label='Sum of 19 regional results')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Mass change (Gt yr$^{-1}$)')
    plt.xlabel('Year')
    plt.title('Internal consistency of released calendar-year benchmark')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_global_consistency.png', bbox_inches='tight')
    plt.close()


def write_summary_json(input_inventory, regional, global_df, global_check, validation):
    worst = global_df.nsmallest(3, 'mass_change_gt')[['year', 'mass_change_gt']].to_dict(orient='records')
    best = global_df.nlargest(3, 'mass_change_gt')[['year', 'mass_change_gt']].to_dict(orient='records')
    summary = {
        'n_input_datasets': int(len(input_inventory)),
        'n_regions': int(regional['region'].nunique()),
        'years': [int(global_df['year'].min()), int(global_df['year'].max())],
        'global_total_2000_2023_gt': float(global_df['mass_change_gt'].sum()),
        'mean_global_annual_gt': float(global_df['mass_change_gt'].mean()),
        'mean_global_specific_mwe': float(global_df['specific_mass_change_mwe'].mean()),
        'max_abs_global_consistency_gap_gt': float(global_check['sum_minus_official_gt'].abs().max()),
        'best_years_by_mass_change_gt': best,
        'worst_years_by_mass_change_gt': worst,
        'method_agreement': validation.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    ensure_dirs()
    input_inventory = load_input_inventory()
    calendar_df = load_calendar_results()
    hyd_df = load_hydrological_results()

    regional, global_df, global_check = build_primary_outputs(calendar_df)
    method_coverage = summarize_method_coverage(input_inventory)
    method_ts = method_timeseries_from_hydrological(hyd_df)
    validation = build_validation_summary(method_ts)

    input_inventory.to_csv(OUTPUT_DIR / 'input_dataset_inventory.csv', index=False)
    method_coverage.to_csv(OUTPUT_DIR / 'method_coverage_by_region.csv', index=False)
    regional.to_csv(OUTPUT_DIR / 'regional_annual_timeseries.csv', index=False)
    global_df.to_csv(OUTPUT_DIR / 'global_annual_timeseries.csv', index=False)
    global_check.to_csv(OUTPUT_DIR / 'global_consistency_check.csv', index=False)
    method_ts.to_csv(OUTPUT_DIR / 'method_agreement.csv', index=False)
    validation.to_csv(OUTPUT_DIR / 'validation_summary.csv', index=False)

    top_losses = (
        regional.groupby('region', as_index=False)['mass_change_gt'].sum()
        .sort_values('mass_change_gt')
        .rename(columns={'mass_change_gt': 'total_mass_change_2000_2023_gt'})
    )
    top_losses.to_csv(OUTPUT_DIR / 'regional_total_change_ranked.csv', index=False)

    make_figures(input_inventory, regional, global_df, global_check, method_ts, validation)
    write_summary_json(input_inventory, regional, global_df, global_check, validation)

    print('Analysis complete.')
    print(f'Regional rows: {len(regional)} | Global rows: {len(global_df)} | Input datasets: {len(input_inventory)}')
    print('Outputs written to outputs/ and figures to report/images/.')


if __name__ == '__main__':
    main()
