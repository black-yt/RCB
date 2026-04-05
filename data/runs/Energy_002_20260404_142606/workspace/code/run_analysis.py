import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

BASE = Path('.')
DATA = BASE / 'data'
OUT = BASE / 'outputs'
REPORT_IMG = BASE / 'report' / 'images'

OUT.mkdir(exist_ok=True, parents=True)
REPORT_IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

SCENARIOS = {
    'baseline': {
        'wacc': 0.10,
        'electrolyzer_capex_per_kw': 700,
        'ammonia_capex_per_kg_h2_yr': 1.2,
        'reconversion_capex_per_kg_h2_yr': 0.9,
        'shipping_per_1000km': 0.22,
        'europe_h2_cost': 7.2,
        'policy_credit': 0.0,
        'description': 'African emerging-market finance conditions, moderate 2030 technology progress.',
    },
    'derisked': {
        'wacc': 0.06,
        'electrolyzer_capex_per_kw': 650,
        'ammonia_capex_per_kg_h2_yr': 1.1,
        'reconversion_capex_per_kg_h2_yr': 0.85,
        'shipping_per_1000km': 0.21,
        'europe_h2_cost': 7.4,
        'policy_credit': 0.35,
        'description': 'Concessional/de-risked finance with contracts-for-difference or export support.',
    },
    'high_rate': {
        'wacc': 0.14,
        'electrolyzer_capex_per_kw': 760,
        'ammonia_capex_per_kg_h2_yr': 1.25,
        'reconversion_capex_per_kg_h2_yr': 0.95,
        'shipping_per_1000km': 0.24,
        'europe_h2_cost': 7.6,
        'policy_credit': 0.0,
        'description': 'Tight global capital markets with higher project finance spreads.',
    },
}

MODEL_ASSUMPTIONS = {
    'electrolyzer_efficiency_kwh_per_kg': 52,
    'water_desalination_kwh_per_kg': 1.2,
    'ammonia_synthesis_kwh_per_kg': 8.0,
    'reconversion_kwh_per_kg': 10.5,
    'buffer_storage_kwh_per_kg': 1.0,
    'solar_lcoe_min': 18,
    'solar_lcoe_max': 32,
    'wind_lcoe_min': 20,
    'wind_lcoe_max': 42,
    'mix_penalty_max': 6.0,
    'base_cf': 0.48,
    'pv_cf_bonus': 0.20,
    'wind_cf_bonus': 0.25,
    'grid_connection_cost_per_km': 0.0020,
    'road_connection_cost_per_km': 0.0012,
    'water_pipeline_cost_per_km': 0.0015,
    'port_connection_cost_per_km': 0.0028,
    'africa_to_europe_distance_km': 9000,
    'fixed_port_cost': 0.12,
    'fixed_terminal_cost': 0.18,
    'om_fraction': 0.035,
    'plant_life_years': 20,
    'ammonia_conversion_loss': 0.12,
    'reconversion_loss': 0.03,
    'europe_wacc': 0.05,
}


def capital_recovery_factor(wacc: float, years: int) -> float:
    return (wacc * (1 + wacc) ** years) / (((1 + wacc) ** years) - 1)


def normalize(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if np.isclose(smin, smax):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - smin) / (smax - smin)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA / 'hex_final_NA_min.csv')
    return df


def add_resource_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['pv_share'] = out['theo_pv'] / (out['theo_pv'] + out['theo_wind'])
    out['wind_share'] = 1 - out['pv_share']
    out['solar_lcoe'] = MODEL_ASSUMPTIONS['solar_lcoe_max'] - out['theo_pv'] * (
        MODEL_ASSUMPTIONS['solar_lcoe_max'] - MODEL_ASSUMPTIONS['solar_lcoe_min']
    )
    out['wind_lcoe'] = MODEL_ASSUMPTIONS['wind_lcoe_max'] - out['theo_wind'] * (
        MODEL_ASSUMPTIONS['wind_lcoe_max'] - MODEL_ASSUMPTIONS['wind_lcoe_min']
    )
    out['renewable_lcoe'] = out['pv_share'] * out['solar_lcoe'] + out['wind_share'] * out['wind_lcoe']
    balancing_penalty = MODEL_ASSUMPTIONS['mix_penalty_max'] * np.abs(out['pv_share'] - 0.5) * 2
    out['delivered_power_cost_per_mwh'] = out['renewable_lcoe'] + balancing_penalty
    cf = (
        MODEL_ASSUMPTIONS['base_cf']
        + MODEL_ASSUMPTIONS['pv_cf_bonus'] * out['theo_pv']
        + MODEL_ASSUMPTIONS['wind_cf_bonus'] * out['theo_wind']
    )
    out['effective_cf'] = cf.clip(0.42, 0.82)
    return out


def compute_site_costs(df: pd.DataFrame, scenario_name: str, scenario: dict) -> pd.DataFrame:
    out = add_resource_metrics(df)
    crf = capital_recovery_factor(scenario['wacc'], MODEL_ASSUMPTIONS['plant_life_years'])
    annual_h2_per_kw = 8760 * out['effective_cf'] / MODEL_ASSUMPTIONS['electrolyzer_efficiency_kwh_per_kg']
    annualized_capex_per_kw = scenario['electrolyzer_capex_per_kw'] * (crf + MODEL_ASSUMPTIONS['om_fraction'])
    out['electrolyzer_cost'] = annualized_capex_per_kw / annual_h2_per_kw
    electricity_kwh = (
        MODEL_ASSUMPTIONS['electrolyzer_efficiency_kwh_per_kg']
        + MODEL_ASSUMPTIONS['water_desalination_kwh_per_kg']
        + MODEL_ASSUMPTIONS['ammonia_synthesis_kwh_per_kg']
        + MODEL_ASSUMPTIONS['buffer_storage_kwh_per_kg']
    )
    out['electricity_cost'] = out['delivered_power_cost_per_mwh'] * electricity_kwh / 1000
    out['infrastructure_cost'] = (
        out['grid_dist_km'] * MODEL_ASSUMPTIONS['grid_connection_cost_per_km']
        + out['road_dist_km'] * MODEL_ASSUMPTIONS['road_connection_cost_per_km']
        + out['waterbody_dist_km'] * MODEL_ASSUMPTIONS['water_pipeline_cost_per_km']
        + out['ocean_dist_km'] * MODEL_ASSUMPTIONS['port_connection_cost_per_km']
    )
    out['ammonia_conversion_cost'] = (
        scenario['ammonia_capex_per_kg_h2_yr'] * (crf + 0.04)
        + 0.18
        + MODEL_ASSUMPTIONS['ammonia_synthesis_kwh_per_kg'] * out['delivered_power_cost_per_mwh'] / 1000
    )
    shipped_h2_factor = 1 / (1 - MODEL_ASSUMPTIONS['ammonia_conversion_loss'])
    out['shipping_cost'] = shipped_h2_factor * (
        MODEL_ASSUMPTIONS['fixed_port_cost']
        + MODEL_ASSUMPTIONS['fixed_terminal_cost']
        + scenario['shipping_per_1000km'] * MODEL_ASSUMPTIONS['africa_to_europe_distance_km'] / 1000
    )
    out['reconversion_cost'] = (
        scenario['reconversion_capex_per_kg_h2_yr'] * (crf + 0.04)
        + 0.22
        + MODEL_ASSUMPTIONS['reconversion_kwh_per_kg'] * 65 / 1000
    )
    gross = (
        out['electrolyzer_cost']
        + out['electricity_cost']
        + out['infrastructure_cost']
        + out['ammonia_conversion_cost']
        + out['shipping_cost']
        + out['reconversion_cost']
    )
    out['delivered_cost_eur_per_kg'] = gross / (1 - MODEL_ASSUMPTIONS['reconversion_loss']) - scenario['policy_credit']
    out['scenario'] = scenario_name
    out['europe_h2_cost'] = scenario['europe_h2_cost']
    out['cost_gap_vs_europe'] = out['delivered_cost_eur_per_kg'] - out['europe_h2_cost']
    out['competitive_vs_europe'] = out['cost_gap_vs_europe'] <= 0
    return out


def europe_benchmark() -> pd.DataFrame:
    rows = []
    euro_power = {'baseline': 62, 'derisked': 58, 'high_rate': 68}
    for name, sc in SCENARIOS.items():
        europe_crf = capital_recovery_factor(MODEL_ASSUMPTIONS['europe_wacc'], MODEL_ASSUMPTIONS['plant_life_years'])
        annual_h2_per_kw = 8760 * 0.52 / MODEL_ASSUMPTIONS['electrolyzer_efficiency_kwh_per_kg']
        capex = sc['electrolyzer_capex_per_kw'] * 1.1
        electrolyzer_cost = capex * (europe_crf + 0.035) / annual_h2_per_kw
        electricity_cost = euro_power[name] * (MODEL_ASSUMPTIONS['electrolyzer_efficiency_kwh_per_kg'] + 2.0) / 1000
        total = electrolyzer_cost + electricity_cost + 0.25
        rows.append({
            'scenario': name,
            'europe_h2_cost_modelled': total,
            'europe_h2_cost_assumed': sc['europe_h2_cost'],
        })
    return pd.DataFrame(rows)


def make_overview_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['theo_pv'], kde=True, ax=axes[0, 0], color='#f4a300')
    axes[0, 0].set_title('PV potential distribution')
    sns.histplot(df['theo_wind'], kde=True, ax=axes[0, 1], color='#2a9d8f')
    axes[0, 1].set_title('Wind potential distribution')
    sns.scatterplot(data=df, x='ocean_dist_km', y='grid_dist_km', hue='theo_pv', palette='viridis', ax=axes[1, 0])
    axes[1, 0].set_title('Distance to ocean vs grid')
    sns.scatterplot(data=df, x='lat', y='lon', size='theo_wind', hue='theo_pv', palette='magma', ax=axes[1, 1])
    axes[1, 1].set_title('Site geography and resource quality')
    plt.tight_layout()
    fig.savefig(REPORT_IMG / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)


def make_map(best_sites: pd.DataFrame) -> None:
    world = gpd.read_file(DATA / 'africa_map' / 'ne_10m_admin_0_countries.shp')
    africa = world[world['CONTINENT'].isin(['Africa', 'Europe'])].copy()
    gdf = gpd.GeoDataFrame(best_sites, geometry=gpd.points_from_xy(best_sites['lon'], best_sites['lat']), crs='EPSG:4326')
    fig, ax = plt.subplots(figsize=(12, 10))
    africa.plot(ax=ax, color='#f0f0f0', edgecolor='#999999', linewidth=0.6)
    gdf.plot(ax=ax, column='delivered_cost_eur_per_kg', cmap='viridis_r', markersize=60, legend=True)
    for _, row in gdf.head(5).iterrows():
        ax.annotate(row['hex_id'], (row['lon'], row['lat']), fontsize=8, xytext=(3, 3), textcoords='offset points')
    ax.set_title('Least-cost African export locations (baseline scenario)')
    ax.set_xlim(-20, 40)
    ax.set_ylim(-36, 42)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    fig.savefig(REPORT_IMG / 'baseline_map.png', bbox_inches='tight')
    plt.close(fig)


def make_scenario_plots(all_results: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=all_results, x='scenario', y='delivered_cost_eur_per_kg', hue='scenario', ax=ax, palette='Set2', legend=False)
    sns.stripplot(data=all_results, x='scenario', y='delivered_cost_eur_per_kg', ax=ax, color='black', alpha=0.5, size=4)
    ax.set_ylabel('Delivered hydrogen cost to Europe (€/kg-H2)')
    ax.set_title('Scenario distribution of delivered African green hydrogen cost')
    plt.tight_layout()
    fig.savefig(REPORT_IMG / 'scenario_comparison.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    tmp = summary.melt(id_vars='scenario', value_vars=['mean_cost', 'p10_cost', 'min_cost', 'europe_benchmark'],
                       var_name='metric', value_name='value')
    sns.barplot(data=tmp, x='scenario', y='value', hue='metric', ax=ax)
    ax.set_ylabel('€/kg-H2')
    ax.set_title('African delivered costs versus European benchmark')
    plt.tight_layout()
    fig.savefig(REPORT_IMG / 'benchmark_comparison.png', bbox_inches='tight')
    plt.close(fig)


def make_tornado(summary: pd.DataFrame) -> None:
    base = summary.set_index('scenario').loc['baseline', 'min_cost']
    comp = summary.set_index('scenario')['min_cost'] - base
    tornado = comp.drop('baseline').sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d95f02' if v > 0 else '#1b9e77' for v in tornado.values]
    ax.barh(tornado.index, tornado.values, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Change in least-cost delivered cost vs baseline (€/kg-H2)')
    ax.set_title('Financing scenario sensitivity of best-site economics')
    plt.tight_layout()
    fig.savefig(REPORT_IMG / 'financing_sensitivity.png', bbox_inches='tight')
    plt.close(fig)


def save_outputs(df: pd.DataFrame, all_results: pd.DataFrame, summary: pd.DataFrame, top_sites: pd.DataFrame, competitiveness: pd.DataFrame, euro: pd.DataFrame) -> None:
    overview = df.describe().transpose()
    overview.to_csv(OUT / 'data_overview.csv')
    all_results.to_csv(OUT / 'site_results.csv', index=False)
    summary.to_csv(OUT / 'scenario_summary.csv', index=False)
    top_sites.to_csv(OUT / 'top_sites_by_scenario.csv', index=False)
    competitiveness.to_csv(OUT / 'competitiveness_summary.csv', index=False)
    euro.to_csv(OUT / 'europe_benchmark.csv', index=False)


def main(stage: str = 'all') -> None:
    df = load_data()
    make_overview_plots(df)
    if stage == 'eda':
        df.describe().transpose().to_csv(OUT / 'data_overview.csv')
        return

    results = []
    for name, sc in SCENARIOS.items():
        results.append(compute_site_costs(df, name, sc))
    all_results = pd.concat(results, ignore_index=True)

    summary = all_results.groupby('scenario').agg(
        min_cost=('delivered_cost_eur_per_kg', 'min'),
        mean_cost=('delivered_cost_eur_per_kg', 'mean'),
        median_cost=('delivered_cost_eur_per_kg', 'median'),
        p10_cost=('delivered_cost_eur_per_kg', lambda s: np.quantile(s, 0.10)),
        competitive_sites=('competitive_vs_europe', 'sum'),
        total_sites=('hex_id', 'count'),
        mean_gap_vs_europe=('cost_gap_vs_europe', 'mean'),
    ).reset_index()
    summary['competitive_share'] = summary['competitive_sites'] / summary['total_sites']
    summary['europe_benchmark'] = summary['scenario'].map({k: v['europe_h2_cost'] for k, v in SCENARIOS.items()})

    top_sites = all_results.sort_values(['scenario', 'delivered_cost_eur_per_kg']).groupby('scenario').head(5).copy()
    top_sites['rank'] = top_sites.groupby('scenario')['delivered_cost_eur_per_kg'].rank(method='first')

    competitiveness = all_results.groupby('scenario').agg(
        sites_below_europe=('competitive_vs_europe', 'sum'),
        sites_below_europe_plus_10pct=('delivered_cost_eur_per_kg', lambda s: int((s <= 1.1 * all_results.loc[s.index, 'europe_h2_cost']).sum())),
        best_site_gap=('cost_gap_vs_europe', 'min'),
        worst_site_gap=('cost_gap_vs_europe', 'max'),
    ).reset_index()

    euro = europe_benchmark()
    save_outputs(df, all_results, summary, top_sites, competitiveness, euro)
    make_map(top_sites[top_sites['scenario'] == 'baseline'])
    make_scenario_plots(all_results, summary)
    make_tornado(summary)

    assumptions = pd.DataFrame([
        {'parameter': k, 'value': v} for k, v in MODEL_ASSUMPTIONS.items()
    ])
    assumptions.to_csv(OUT / 'model_assumptions.csv', index=False)
    pd.DataFrame([
        {'scenario': k, **v} for k, v in SCENARIOS.items()
    ]).to_csv(OUT / 'scenario_assumptions.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all', choices=['all', 'eda'])
    args = parser.parse_args()
    main(stage=args.stage)
