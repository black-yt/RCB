import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.neighbors import BallTree

sns.set_theme(style="whitegrid", context="talk")
R_EARTH_KM = 6371.0088
WORKSPACE_SEED = 42


def ensure_dirs():
    for path in [Path('outputs'), Path('report/images')]:
        path.mkdir(parents=True, exist_ok=True)


def winsorized_minmax(series: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    s = series.astype(float).copy()
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(s)), index=s.index)
    clipped = s.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


def load_mangroves() -> gpd.GeoDataFrame:
    gdf = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
    gdf = gdf.to_crs(4326)
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    # Dataset is a 10% sample of mangrove reference points; assign equal area weight.
    total_area_2020 = 14879481.0  # ha, sum across country dataset
    point_area_ha = total_area_2020 / len(gdf) / 0.10
    gdf['sample_area_ha'] = point_area_ha
    return gdf


def load_country_bounds() -> gpd.GeoDataFrame:
    cb = gpd.read_file('data/ecosystem/UCSC_CWON_countrybounds.gpkg').to_crs(4326)
    keep = [
        'Country', 'ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020',
        'Ben_Pop_2020', 'Ben_Stock_2020', 'geometry'
    ]
    return cb[keep].copy()


def spatial_join_countries(mangroves: gpd.GeoDataFrame, countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(mangroves, countries, how='left', predicate='within')
    joined = joined.drop(columns=['index_right'])
    # Fallback for points near coastlines / geometry gaps using projected nearest-neighbor join.
    missing = joined['ISO3'].isna()
    if missing.any():
        missing_pts = joined.loc[missing, ['uid', 'geometry']].copy().to_crs(3857)
        countries_proj = countries.to_crs(3857)
        nearest = gpd.sjoin_nearest(missing_pts, countries_proj, how='left', distance_col='dist_m')
        nearest = nearest.drop_duplicates('uid').set_index('uid')
        for col in ['Country', 'ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020']:
            joined.loc[missing, col] = joined.loc[missing, 'uid'].map(nearest[col])
    return joined


def load_slr_scenario(path: str, scenario_name: str) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    rate = ds['sea_level_change_rate'].sel(quantiles=0.5)
    rate_2020 = rate.sel(years=2020)
    rate_2100 = rate.sel(years=2100)
    cumulative_cm = (rate_2100 - rate_2020).values.astype(float)
    annual_cm_per_year = cumulative_cm / (2100 - 2020)
    out = pd.DataFrame({
        'slr_lat': ds['lat'].values.astype(float),
        'slr_lon': ds['lon'].values.astype(float),
        f'slr_{scenario_name}_cm_2100': cumulative_cm,
        f'slr_{scenario_name}_cm_per_year': annual_cm_per_year,
    })
    return out


def _haversine_array(lat_deg, lon_deg):
    return np.deg2rad(np.c_[lat_deg, lon_deg])


def nearest_join_balltree(points_df: pd.DataFrame, ref_df: pd.DataFrame, point_lat='lat', point_lon='lon', ref_lat='lat', ref_lon='lon') -> np.ndarray:
    tree = BallTree(_haversine_array(ref_df[ref_lat].values, ref_df[ref_lon].values), metric='haversine')
    dist, idx = tree.query(_haversine_array(points_df[point_lat].values, points_df[point_lon].values), k=1)
    return idx[:, 0], dist[:, 0] * R_EARTH_KM


def attach_slr(mangroves: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        'ssp245': 'data/slr/total_ssp245_medium_confidence_rates.nc',
        'ssp370': 'data/slr/total_ssp370_medium_confidence_rates.nc',
        'ssp585': 'data/slr/total_ssp585_medium_confidence_rates.nc',
    }
    df = mangroves.copy()
    for scenario, path in scenarios.items():
        slr = load_slr_scenario(path, scenario)
        idx, dist_km = nearest_join_balltree(df[['lat', 'lon']], slr.rename(columns={'slr_lat': 'lat', 'slr_lon': 'lon'}))
        matched = slr.iloc[idx].reset_index(drop=True)
        df[f'{scenario}_nearest_slr_km'] = dist_km
        for col in matched.columns:
            if col not in {'slr_lat', 'slr_lon'}:
                df[col] = matched[col].values
    return df


def load_tc_points() -> pd.DataFrame:
    ds = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
    tc = pd.DataFrame({
        'lat': ds['lat'].values.astype(float),
        'lon': ds['lon'].values.astype(float),
        'wind': ds['wind'].values.astype(float),
    }).dropna()
    tc['abs_lat'] = tc['lat'].abs()
    tc['severe'] = (tc['wind'] >= 50).astype(int)
    return tc


def compute_tc_features(mangroves: pd.DataFrame, tc: pd.DataFrame, radius_km: float = 150.0) -> pd.DataFrame:
    pts_rad = _haversine_array(mangroves['lat'].values, mangroves['lon'].values)
    tc_rad = _haversine_array(tc['lat'].values, tc['lon'].values)
    tree = BallTree(tc_rad, metric='haversine')
    neighbors = tree.query_radius(pts_rad, r=radius_km / R_EARTH_KM, return_distance=True, sort_results=True)

    counts = np.zeros(len(mangroves), dtype=float)
    severe_counts = np.zeros(len(mangroves), dtype=float)
    mean_wind = np.full(len(mangroves), np.nan)
    max_wind = np.full(len(mangroves), np.nan)
    invdist_intensity = np.zeros(len(mangroves), dtype=float)
    storm_abs_lat = np.full(len(mangroves), np.nan)

    tc_wind = tc['wind'].values
    tc_severe = tc['severe'].values
    tc_abs_lat = tc['abs_lat'].values

    for i, (inds, dists) in enumerate(zip(*neighbors)):
        if len(inds) == 0:
            continue
        counts[i] = len(inds)
        severe_counts[i] = tc_severe[inds].sum()
        mean_wind[i] = tc_wind[inds].mean()
        max_wind[i] = tc_wind[inds].max()
        weights = 1.0 / (1.0 + dists * R_EARTH_KM)
        invdist_intensity[i] = np.sum(tc_wind[inds] * weights)
        storm_abs_lat[i] = tc_abs_lat[inds].mean()

    df = mangroves.copy()
    df['tc_count_150km'] = counts
    df['tc_severe_count_150km'] = severe_counts
    df['tc_mean_wind_150km'] = np.nan_to_num(mean_wind, nan=0.0)
    df['tc_max_wind_150km'] = np.nan_to_num(max_wind, nan=0.0)
    df['tc_invdist_intensity_150km'] = invdist_intensity
    df['tc_mean_abs_lat_150km'] = np.nan_to_num(storm_abs_lat, nan=df['lat'].abs())
    df['tc_lat_shift_proxy'] = np.maximum(0.0, df['tc_mean_abs_lat_150km'] - df['lat'].abs())
    df['tc_regime_shift_proxy'] = 0.5 * winsorized_minmax(df['tc_severe_count_150km']) + 0.5 * winsorized_minmax(df['tc_lat_shift_proxy'])
    df['tc_baseline_hazard'] = 0.4 * winsorized_minmax(df['tc_count_150km']) + 0.6 * winsorized_minmax(df['tc_invdist_intensity_150km'])
    df['tc_total_hazard'] = 0.6 * df['tc_baseline_hazard'] + 0.4 * df['tc_regime_shift_proxy']
    return df


def build_service_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Risk_Pop_2020', 'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020', 'Mang_Ha_2020']:
        df[col] = df[col].fillna(0)
    df['service_risk_index'] = (
        0.3 * winsorized_minmax(np.log1p(df['Risk_Pop_2020'])) +
        0.3 * winsorized_minmax(np.log1p(df['Ben_Pop_2020'])) +
        0.2 * winsorized_minmax(np.log1p(df['Risk_Stock_2020'])) +
        0.2 * winsorized_minmax(np.log1p(df['Ben_Stock_2020']))
    )
    return df


def compute_scenario_risk(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()
    slr_col = f'slr_{scenario}_cm_2100'
    out[f'{scenario}_slr_norm'] = winsorized_minmax(out[slr_col])
    out[f'{scenario}_hazard_index'] = 0.5 * out['tc_total_hazard'] + 0.5 * out[f'{scenario}_slr_norm']
    out[f'{scenario}_composite_risk'] = (
        0.5 * out[f'{scenario}_hazard_index'] + 0.3 * out['service_risk_index'] + 0.2 * winsorized_minmax(np.log1p(out['sample_area_ha']))
    )
    return out


def summarize_country(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    comp = f'{scenario}_composite_risk'
    haz = f'{scenario}_hazard_index'
    slr = f'slr_{scenario}_cm_2100'

    def _agg(x: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'sample_count': len(x),
            'estimated_mangrove_area_ha': x['sample_area_ha'].sum(),
            'mean_composite_risk': np.average(x[comp], weights=x['sample_area_ha']),
            'mean_hazard_index': np.average(x[haz], weights=x['sample_area_ha']),
            'mean_tc_hazard': np.average(x['tc_total_hazard'], weights=x['sample_area_ha']),
            'mean_slr_cm_2100': np.average(x[slr], weights=x['sample_area_ha']),
            'mean_service_index': np.average(x['service_risk_index'], weights=x['sample_area_ha']),
            'p90_composite_risk': np.quantile(x[comp], 0.9),
            'high_risk_area_ha': x.loc[x[comp] >= 0.67, 'sample_area_ha'].sum(),
        })

    grouped = df.groupby(['ISO3', 'Country'], dropna=False)[df.columns].apply(_agg).reset_index()
    grouped['high_risk_area_share'] = grouped['high_risk_area_ha'] / grouped['estimated_mangrove_area_ha']
    grouped['scenario'] = scenario
    return grouped.sort_values('mean_composite_risk', ascending=False)


def latitudinal_profile(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    temp = df.copy()
    temp['lat_band'] = (np.floor(temp['lat'] / 5) * 5).astype(int)
    comp = f'{scenario}_composite_risk'
    slr = f'slr_{scenario}_cm_2100'
    haz = f'{scenario}_hazard_index'

    def _agg(x: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'area_ha': x['sample_area_ha'].sum(),
            'mean_composite_risk': np.average(x[comp], weights=x['sample_area_ha']),
            'mean_tc_hazard': np.average(x['tc_total_hazard'], weights=x['sample_area_ha']),
            'mean_slr_cm_2100': np.average(x[slr], weights=x['sample_area_ha']),
            'mean_hazard_index': np.average(x[haz], weights=x['sample_area_ha']),
        })

    out = temp.groupby('lat_band')[temp.columns].apply(_agg).reset_index().sort_values('lat_band')
    out['scenario'] = scenario
    return out


def save_inventory(mangroves, countries, tc):
    inventory = {
        'mangrove_points': int(len(mangroves)),
        'mangrove_bounds': [float(x) for x in mangroves.total_bounds],
        'country_polygons': int(len(countries)),
        'country_fields': list(countries.columns),
        'tc_records': int(len(tc)),
        'tc_wind_min': float(tc['wind'].min()),
        'tc_wind_max': float(tc['wind'].max()),
    }
    Path('outputs/data_inventory.json').write_text(json.dumps(inventory, indent=2))


def make_figures(df: pd.DataFrame, country_summary: pd.DataFrame):
    img_dir = Path('report/images')

    # Figure 1: data overview.
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    axes[0].hist(df['lat'], bins=40, color='#2a9d8f', alpha=0.9)
    axes[0].set_title('Mangrove sample latitude distribution')
    axes[0].set_xlabel('Latitude')
    axes[0].set_ylabel('Count')

    axes[1].hist(df['tc_invdist_intensity_150km'], bins=40, color='#e76f51', alpha=0.9)
    axes[1].set_title('TC exposure intensity (150 km)')
    axes[1].set_xlabel('Inverse-distance weighted wind')

    axes[2].hist(df['slr_ssp585_cm_2100'], bins=40, color='#264653', alpha=0.9)
    axes[2].set_title('SLR increment to 2100 (SSP5-8.5)')
    axes[2].set_xlabel('cm relative change, 2020-2100')
    plt.tight_layout()
    fig.savefig(img_dir / 'figure_1_data_overview.png', dpi=220)
    plt.close(fig)

    # Figure 2: global scatter map.
    fig, ax = plt.subplots(figsize=(14, 7))
    sc = ax.scatter(df['lon'], df['lat'], c=df['ssp585_composite_risk'], s=4, cmap='magma', alpha=0.6, linewidths=0)
    ax.set_title('Global mangrove composite risk under SSP5-8.5')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax, label='Composite risk index')
    plt.tight_layout()
    fig.savefig(img_dir / 'figure_2_global_risk_map.png', dpi=220)
    plt.close(fig)

    # Figure 3: scenario comparison boxplot.
    box_df = pd.concat([
        df[['ssp245_composite_risk']].rename(columns={'ssp245_composite_risk': 'risk'}).assign(scenario='SSP2-4.5'),
        df[['ssp370_composite_risk']].rename(columns={'ssp370_composite_risk': 'risk'}).assign(scenario='SSP3-7.0'),
        df[['ssp585_composite_risk']].rename(columns={'ssp585_composite_risk': 'risk'}).assign(scenario='SSP5-8.5'),
    ])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=box_df, x='scenario', y='risk', hue='scenario', dodge=False, ax=ax, palette='viridis', legend=False)
    ax.set_title('Composite risk distribution across SLR scenarios')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Composite risk index')
    plt.tight_layout()
    fig.savefig(img_dir / 'figure_3_scenario_comparison.png', dpi=220)
    plt.close(fig)

    # Figure 4: hazard component relationship.
    fig, ax = plt.subplots(figsize=(8, 7))
    sample = df.sample(min(15000, len(df)), random_state=WORKSPACE_SEED)
    sns.scatterplot(
        data=sample, x='tc_total_hazard', y='ssp585_slr_norm', hue='service_risk_index',
        palette='viridis', s=18, alpha=0.45, linewidth=0, ax=ax
    )
    ax.set_title('Joint cyclone and SLR hazard space')
    ax.set_xlabel('TC hazard index')
    ax.set_ylabel('Normalized SLR hazard (SSP5-8.5)')
    plt.legend(title='Service index', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(img_dir / 'figure_4_hazard_space.png', dpi=220)
    plt.close(fig)

    # Figure 5: top countries.
    top = country_summary[country_summary['scenario'] == 'ssp585'].head(15).sort_values('mean_composite_risk')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['Country'], top['mean_composite_risk'], color='#8d99ae')
    ax.set_title('Top 15 countries by area-weighted mangrove risk (SSP5-8.5)')
    ax.set_xlabel('Mean composite risk index')
    ax.set_ylabel('Country')
    plt.tight_layout()
    fig.savefig(img_dir / 'figure_5_top_countries.png', dpi=220)
    plt.close(fig)

    # Figure 6: latitudinal profile.
    latprof = pd.read_csv('outputs/global_latitudinal_profile.csv')
    latprof['scenario_label'] = latprof['scenario'].map({'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=latprof, x='lat_band', y='mean_composite_risk', hue='scenario_label', marker='o', ax=ax)
    ax.set_title('Latitudinal profile of mangrove risk')
    ax.set_xlabel('Latitude band (5°)')
    ax.set_ylabel('Area-weighted composite risk')
    plt.tight_layout()
    fig.savefig(img_dir / 'figure_6_latitudinal_profile.png', dpi=220)
    plt.close(fig)


def inspect_mode():
    ensure_dirs()
    mangroves = load_mangroves()
    countries = load_country_bounds()
    tc = load_tc_points()
    save_inventory(mangroves, countries, tc)
    print(json.dumps(json.loads(Path('outputs/data_inventory.json').read_text()), indent=2))


def run_mode():
    ensure_dirs()
    mangroves = load_mangroves()
    countries = load_country_bounds()
    tc = load_tc_points()
    save_inventory(mangroves, countries, tc)

    df = spatial_join_countries(mangroves, countries)
    df = attach_slr(df)
    df = compute_tc_features(df, tc)
    df = build_service_index(df)
    for scenario in ['ssp245', 'ssp370', 'ssp585']:
        df = compute_scenario_risk(df, scenario)

    keep_cols = [
        'uid', 'lon', 'lat', 'sample_area_ha', 'Country', 'ISO3', 'Mang_Ha_2020',
        'Risk_Pop_2020', 'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020',
        'tc_count_150km', 'tc_severe_count_150km', 'tc_mean_wind_150km', 'tc_max_wind_150km',
        'tc_invdist_intensity_150km', 'tc_lat_shift_proxy', 'tc_regime_shift_proxy',
        'tc_baseline_hazard', 'tc_total_hazard', 'service_risk_index',
        'slr_ssp245_cm_2100', 'slr_ssp370_cm_2100', 'slr_ssp585_cm_2100',
        'ssp245_slr_norm', 'ssp370_slr_norm', 'ssp585_slr_norm',
        'ssp245_hazard_index', 'ssp370_hazard_index', 'ssp585_hazard_index',
        'ssp245_composite_risk', 'ssp370_composite_risk', 'ssp585_composite_risk'
    ]
    out_df = pd.DataFrame(df[keep_cols])
    out_df.to_parquet('outputs/mangrove_risk_samples.parquet', index=False)
    out_df.sample(5000, random_state=WORKSPACE_SEED).to_csv('outputs/mangrove_risk_samples_preview.csv', index=False)

    country_frames = [summarize_country(out_df, s) for s in ['ssp245', 'ssp370', 'ssp585']]
    country_summary = pd.concat(country_frames, ignore_index=True)
    country_summary.to_csv('outputs/country_risk_summary.csv', index=False)

    lat_frames = [latitudinal_profile(out_df, s) for s in ['ssp245', 'ssp370', 'ssp585']]
    lat_summary = pd.concat(lat_frames, ignore_index=True)
    lat_summary.to_csv('outputs/global_latitudinal_profile.csv', index=False)

    scenario_summary = []
    for scenario in ['ssp245', 'ssp370', 'ssp585']:
        comp = out_df[f'{scenario}_composite_risk']
        scenario_summary.append({
            'scenario': scenario,
            'mean': comp.mean(),
            'median': comp.median(),
            'p90': comp.quantile(0.9),
            'high_risk_share': float((comp >= 0.67).mean()),
            'slr_mean_cm_2100': out_df[f'slr_{scenario}_cm_2100'].mean(),
        })
    pd.DataFrame(scenario_summary).to_csv('outputs/scenario_risk_summary.csv', index=False)

    corr = out_df[[
        'tc_total_hazard', 'service_risk_index', 'slr_ssp245_cm_2100', 'slr_ssp370_cm_2100',
        'slr_ssp585_cm_2100', 'ssp245_composite_risk', 'ssp370_composite_risk', 'ssp585_composite_risk'
    ]].corr()
    corr.to_csv('outputs/risk_correlation_matrix.csv')

    make_figures(out_df, country_summary)
    print('Completed analysis. Main outputs written to outputs/ and report/images/.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inspect', 'run'], default='run')
    args = parser.parse_args()

    if args.mode == 'inspect':
        inspect_mode()
    else:
        run_mode()
