import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
REPORT_DIR = ROOT / 'report'
IMG_DIR = REPORT_DIR / 'images'
CODE_DIR = ROOT / 'code'

INPUT_PATH = DATA_DIR / '20231012-06_input_netcdf.nc'
FORECAST_PATH = DATA_DIR / '006.nc'

UPPER_VARS = ['Z', 'T', 'U', 'V', 'R']
PRESSURE_LEVELS = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']
SURFACE_VARS = ['T2M', 'U10', 'V10', 'MSL', 'TP']

CHANNEL_LABELS = [f'{var}{lev}' for var in UPPER_VARS for lev in PRESSURE_LEVELS] + SURFACE_VARS
PLOT_CHANNELS = ['Z500', 'T850', 'U500', 'V500', 'R850', 'T2M', 'MSL', 'TP']


def ensure_dirs():
    for d in [OUTPUT_DIR, REPORT_DIR, IMG_DIR, CODE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def area_weights(lat):
    weights = np.cos(np.deg2rad(lat))
    weights = np.clip(weights, 0.0, None)
    return weights / weights.mean()


def weighted_stats(field, lat):
    arr = np.asarray(field, dtype=np.float64)
    w_lat = area_weights(lat)[:, None]
    w = np.broadcast_to(w_lat, arr.shape)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return {k: np.nan for k in ['mean', 'std', 'min', 'max', 'weighted_rmse_zero', 'abs_mean']}
    arrv = arr[mask]
    wv = w[mask]
    wsum = wv.sum()
    mean = float((arrv * wv).sum() / wsum)
    var = float((wv * (arrv - mean) ** 2).sum() / wsum)
    abs_mean = float((np.abs(arrv) * wv).sum() / wsum)
    rmse_zero = float(np.sqrt((wv * arrv ** 2).sum() / wsum))
    return {
        'mean': mean,
        'std': np.sqrt(max(var, 0.0)),
        'min': float(np.nanmin(arr)),
        'max': float(np.nanmax(arr)),
        'weighted_rmse_zero': rmse_zero,
        'abs_mean': abs_mean,
    }


def parse_channels(level_values):
    rows = []
    for idx, label in enumerate(level_values):
        label = str(label)
        if label in SURFACE_VARS:
            rows.append({'channel_index': idx, 'channel_label': label, 'group': 'surface', 'variable': label, 'pressure_hpa': np.nan})
        else:
            prefix = ''.join([c for c in label if c.isalpha()])
            suffix = ''.join([c for c in label if c.isdigit()])
            rows.append({'channel_index': idx, 'channel_label': label, 'group': 'upper_air', 'variable': prefix, 'pressure_hpa': float(suffix) if suffix else np.nan})
    return pd.DataFrame(rows)


def dataset_inventory(ds, name):
    info = {
        'name': name,
        'dims': {k: int(v) for k, v in ds.sizes.items()},
        'coords': {},
        'data_vars': {},
        'attrs': {k: str(v) for k, v in ds.attrs.items()},
    }
    for c in ds.coords:
        arr = ds[c]
        info['coords'][c] = {
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'attrs': {k: str(v) for k, v in arr.attrs.items()},
        }
        values = arr.values
        if values.ndim == 1 and values.size <= 10:
            info['coords'][c]['values'] = [str(v) for v in values.tolist()]
        elif values.ndim == 1 and values.size > 10:
            info['coords'][c]['head'] = [str(v) for v in values[:5].tolist()]
            info['coords'][c]['tail'] = [str(v) for v in values[-5:].tolist()]
    for v in ds.data_vars:
        arr = ds[v]
        info['data_vars'][v] = {
            'dims': list(arr.dims),
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'attrs': {k: str(val) for k, val in arr.attrs.items()},
        }
    return info


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def plot_selected_maps(input_ds, forecast_ds, tendency_ds):
    fig, axes = plt.subplots(len(PLOT_CHANNELS), 3, figsize=(15, 3.2 * len(PLOT_CHANNELS)), constrained_layout=True)
    lon = input_ds['lon'].values
    lat = input_ds['lat'].values
    for i, ch in enumerate(PLOT_CHANNELS):
        init_field = input_ds['data'].sel(time=input_ds.time.values[-1], level=ch).values
        tendency_field = tendency_ds.sel(level=ch).values
        if ch in forecast_ds.level.values:
            forecast_field = forecast_ds['data'].isel(time=0, step=0).sel(level=ch).values
        else:
            forecast_field = np.full_like(init_field, np.nan)
        panels = [init_field, tendency_field, forecast_field]
        titles = [f'{ch}: input t1', f'{ch}: input tendency', f'{ch}: FuXi +6h']
        cmaps = ['coolwarm', 'RdBu_r', 'coolwarm']
        for j, (field, title, cmap) in enumerate(zip(panels, titles, cmaps)):
            ax = axes[i, j]
            im = ax.imshow(field, origin='upper', aspect='auto', cmap=cmap,
                           extent=[lon.min(), lon.max(), lat.min(), lat.max()])
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle('Selected global fields from input tendency and 6-hour forecast', fontsize=16)
    fig.savefig(IMG_DIR / 'selected_global_maps.png', dpi=180)
    plt.close(fig)


def plot_channel_group_counts(channel_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    group_counts = channel_df.groupby('variable').size().reset_index(name='count')
    sns.barplot(data=group_counts, x='variable', y='count', hue='variable', dodge=False, legend=False, ax=axes[0], palette='viridis')
    axes[0].set_title('Channel count by variable family')
    axes[0].set_ylabel('Number of channels')
    axes[0].set_xlabel('Variable family')

    upper = channel_df[channel_df['group'] == 'upper_air'].copy()
    pressure_counts = upper.groupby('pressure_hpa').size().reset_index(name='count')
    sns.lineplot(data=pressure_counts, x='pressure_hpa', y='count', marker='o', ax=axes[1])
    axes[1].set_title('Upper-air channels by pressure level')
    axes[1].set_ylabel('Number of variables')
    axes[1].set_xlabel('Pressure level (hPa)')
    axes[1].invert_xaxis()
    fig.savefig(IMG_DIR / 'channel_structure.png', dpi=180)
    plt.close(fig)


def plot_summary_distributions(stats_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    sns.histplot(stats_df['latest_abs_mean'], bins=20, ax=axes[0, 0], color='tab:blue')
    axes[0, 0].set_title('Distribution of absolute area-weighted means')
    axes[0, 0].set_xlabel('Absolute area-weighted mean')

    sns.histplot(stats_df['tendency_rmse'], bins=20, ax=axes[0, 1], color='tab:orange')
    axes[0, 1].set_title('Distribution of 6-hour input tendency RMSE')
    axes[0, 1].set_xlabel('RMSE between input times')

    top_tend = stats_df.sort_values('tendency_rmse', ascending=False).head(15)
    sns.barplot(data=top_tend, y='channel_label', x='tendency_rmse', hue='channel_label', dodge=False, legend=False, ax=axes[1, 0], palette='rocket')
    axes[1, 0].set_title('Top channels by input tendency RMSE')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].set_ylabel('Channel')

    aligned = stats_df.dropna(subset=['forecast_minus_latest_rmse']).sort_values('forecast_minus_latest_rmse', ascending=False).head(15)
    sns.barplot(data=aligned, y='channel_label', x='forecast_minus_latest_rmse', hue='channel_label', dodge=False, legend=False, ax=axes[1, 1], palette='mako')
    axes[1, 1].set_title('Top channels by |FuXi - latest input| RMSE')
    axes[1, 1].set_xlabel('RMSE')
    axes[1, 1].set_ylabel('Channel')

    fig.savefig(IMG_DIR / 'channel_distributions.png', dpi=180)
    plt.close(fig)


def plot_tendency_vs_forecast(stats_df):
    aligned = stats_df.dropna(subset=['forecast_minus_latest_rmse']).copy()
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.scatterplot(
        data=aligned,
        x='tendency_rmse',
        y='forecast_minus_latest_rmse',
        hue='variable',
        style='group',
        s=80,
        ax=ax,
    )
    for _, row in aligned.nlargest(10, 'forecast_minus_latest_rmse').iterrows():
        ax.text(row['tendency_rmse'], row['forecast_minus_latest_rmse'], row['channel_label'], fontsize=8)
    ax.set_title('Relation between recent tendency and FuXi 6-hour increment')
    ax.set_xlabel('Input tendency RMSE (t0→t1)')
    ax.set_ylabel('RMSE(FuXi +6h − latest input)')
    fig.savefig(IMG_DIR / 'tendency_vs_forecast_scatter.png', dpi=180)
    plt.close(fig)


def create_markdown_audit(input_info, forecast_info, compatibility, related_work_notes):
    lines = []
    lines.append('# Data Audit and Feasibility Notes')
    lines.append('')
    lines.append('## Available files')
    lines.append(f"- Input file: `{INPUT_PATH}`")
    lines.append(f"- Forecast file: `{FORECAST_PATH}`")
    lines.append('')
    lines.append('## Input summary')
    lines.append(f"- Dimensions: {input_info['dims']}")
    lines.append(f"- Coordinates: {list(input_info['coords'].keys())}")
    lines.append(f"- Data variables: {list(input_info['data_vars'].keys())}")
    lines.append('')
    lines.append('## Forecast summary')
    lines.append(f"- Dimensions: {forecast_info['dims']}")
    lines.append(f"- Coordinates: {list(forecast_info['coords'].keys())}")
    lines.append(f"- Data variables: {list(forecast_info['data_vars'].keys())}")
    lines.append('')
    lines.append('## Compatibility assessment')
    for k, v in compatibility.items():
        lines.append(f'- **{k}**: {v}')
    lines.append('')
    lines.append('## Interpretation')
    lines.append('- The available data support only a single-case diagnostic study.')
    lines.append('- The stated 15-day cascade objective cannot be validated because there is no training corpus, no truth trajectory, and no benchmark ensemble data.')
    lines.append('- The forecast file provides one 6-hour model output, which is useful for structural comparison but not for forecast-skill evaluation.')
    lines.append('')
    lines.append('## Related-work guidance from local PDFs')
    for item in related_work_notes:
        lines.append(f'- {item}')
    return '\n'.join(lines)


def main():
    ensure_dirs()
    input_ds = xr.open_dataset(INPUT_PATH)
    forecast_ds = xr.open_dataset(FORECAST_PATH)

    input_info = dataset_inventory(input_ds, 'input')
    forecast_info = dataset_inventory(forecast_ds, 'forecast')
    save_json({'input': input_info, 'forecast': forecast_info}, OUTPUT_DIR / 'data_inventory.json')

    channel_df = parse_channels(input_ds['level'].values)
    channel_df.to_csv(OUTPUT_DIR / 'channel_summary.csv', index=False)

    latest_input = input_ds['data'].isel(time=1)
    prev_input = input_ds['data'].isel(time=0)
    tendency = latest_input - prev_input
    forecast_field = forecast_ds['data'].isel(time=0, step=0)

    input_channels = [str(v) for v in input_ds['level'].values.tolist()]
    forecast_channels = [str(v) for v in forecast_ds['level'].values.tolist()]
    compatibility = {
        'same_channel_labels': input_channels == forecast_channels,
        'same_latitudes': bool(np.array_equal(input_ds['lat'].values, forecast_ds['lat'].values)),
        'same_longitudes': bool(np.array_equal(input_ds['lon'].values, forecast_ds['lon'].values)),
        'input_time_count': int(input_ds.sizes['time']),
        'forecast_step_count': int(forecast_ds.sizes['step']),
        'forecast_step_hours': [int(v) for v in np.atleast_1d(forecast_ds['step'].values).tolist()],
        'can_evaluate_15_day_skill': False,
    }
    save_json(compatibility, OUTPUT_DIR / 'compatibility_summary.json')

    stats_rows = []
    lat = input_ds['lat'].values
    for ch in input_channels:
        latest = latest_input.sel(level=ch).values
        prev = prev_input.sel(level=ch).values
        tend = tendency.sel(level=ch).values
        row = {
            'channel_label': ch,
            'group': channel_df.loc[channel_df['channel_label'] == ch, 'group'].iloc[0],
            'variable': channel_df.loc[channel_df['channel_label'] == ch, 'variable'].iloc[0],
            'pressure_hpa': channel_df.loc[channel_df['channel_label'] == ch, 'pressure_hpa'].iloc[0],
            'nan_fraction_latest': float(np.isnan(latest).mean()),
            'nan_fraction_tendency': float(np.isnan(tend).mean()),
        }
        prev_stats = weighted_stats(prev, lat)
        latest_stats = weighted_stats(latest, lat)
        tend_stats = weighted_stats(tend, lat)
        row.update({f'prev_{k}': v for k, v in prev_stats.items()})
        row.update({f'latest_{k}': v for k, v in latest_stats.items()})
        row.update({f'tendency_{k}': v for k, v in tend_stats.items()})
        if ch in forecast_channels:
            fc = forecast_field.sel(level=ch).values
            row['nan_fraction_forecast'] = float(np.isnan(fc).mean())
            diff = fc - latest
            row.update({f'forecast_{k}': v for k, v in weighted_stats(fc, lat).items()})
            row.update({f'forecast_minus_latest_{k}': v for k, v in weighted_stats(diff, lat).items()})
        else:
            row['nan_fraction_forecast'] = np.nan
            for prefix in ['forecast', 'forecast_minus_latest']:
                for k in ['mean', 'std', 'min', 'max', 'weighted_rmse_zero', 'abs_mean']:
                    row[f'{prefix}_{k}'] = np.nan
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.rename(columns={'tendency_weighted_rmse_zero': 'tendency_rmse', 'forecast_minus_latest_weighted_rmse_zero': 'forecast_minus_latest_rmse'}, inplace=True)
    stats_df.to_csv(OUTPUT_DIR / 'statistics.csv', index=False)

    global_metrics = {
        'n_channels': int(len(stats_df)),
        'n_surface_channels': int((stats_df['group'] == 'surface').sum()),
        'n_upper_air_channels': int((stats_df['group'] == 'upper_air').sum()),
        'mean_input_tendency_rmse': float(stats_df['tendency_rmse'].mean()),
        'median_input_tendency_rmse': float(stats_df['tendency_rmse'].median()),
        'mean_forecast_increment_rmse': float(stats_df['forecast_minus_latest_rmse'].mean()),
        'median_forecast_increment_rmse': float(stats_df['forecast_minus_latest_rmse'].median()),
        'corr_tendency_vs_forecast_increment': float(stats_df[['tendency_rmse', 'forecast_minus_latest_rmse']].dropna().corr().iloc[0, 1]),
    }
    save_json(global_metrics, OUTPUT_DIR / 'global_metrics.json')

    feasibility_lines = [
        '# Feasibility Notes',
        '',
        '- Full 15-day cascade model development is not executable from the provided sample alone.',
        '- The available sample is sufficient for data auditing, structural compatibility checks, and single-case descriptive diagnostics.',
        '- Published weather-forecast evaluations typically rely on lead-time RMSE/ACC curves over many initializations; those ingredients are absent here.',
        '- The present analysis therefore treats the FuXi file as a sample forecast field rather than ground truth.',
    ]
    (OUTPUT_DIR / 'feasibility_notes.md').write_text('\n'.join(feasibility_lines), encoding='utf-8')

    related_work_notes = [
        'Local related-work PDFs emphasize latitude-weighted RMSE and anomaly correlation coefficient as standard medium-range metrics.',
        'The references repeatedly warn that autoregressive inference accumulates error with lead time, so single-step evidence cannot justify 15-day claims.',
        'Transformer-based weather models in the local literature rely on multi-case evaluation sets and often use modality-specific encoders or spectral token mixing.',
        'Given only one input case and one 6-hour output, the most defensible contribution is a feasibility and diagnostic study rather than a forecast-skill benchmark.',
    ]
    audit_md = create_markdown_audit(input_info, forecast_info, compatibility, related_work_notes)
    (OUTPUT_DIR / 'data_audit.md').write_text(audit_md, encoding='utf-8')

    plot_channel_group_counts(channel_df)
    plot_summary_distributions(stats_df)
    plot_tendency_vs_forecast(stats_df)
    plot_selected_maps(input_ds, forecast_ds, tendency)

    top_tables = {
        'top_input_tendency_rmse': stats_df.sort_values('tendency_rmse', ascending=False).head(10)[['channel_label', 'variable', 'tendency_rmse']].to_dict(orient='records'),
        'top_forecast_increment_rmse': stats_df.sort_values('forecast_minus_latest_rmse', ascending=False).head(10)[['channel_label', 'variable', 'forecast_minus_latest_rmse']].to_dict(orient='records'),
    }
    save_json(top_tables, OUTPUT_DIR / 'top_channel_rankings.json')

    print('Analysis complete.')
    print('Wrote:', OUTPUT_DIR / 'data_inventory.json')
    print('Wrote:', OUTPUT_DIR / 'channel_summary.csv')
    print('Wrote:', OUTPUT_DIR / 'statistics.csv')
    print('Wrote:', OUTPUT_DIR / 'global_metrics.json')
    print('Wrote figures to', IMG_DIR)


if __name__ == '__main__':
    main()
