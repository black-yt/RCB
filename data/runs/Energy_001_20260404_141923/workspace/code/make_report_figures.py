import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')


def load_summary():
    return pd.read_csv(OUTPUT_DIR / 'scenario_summary.csv')


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_demand_and_wind_overview():
    demand = pd.read_csv(DATA_DIR / 'demand.csv')
    wind_cf = pd.read_csv(DATA_DIR / 'wind_cf.csv')
    hourly = pd.DataFrame({
        'hour': np.arange(len(demand)),
        'total_demand_gw': demand.sum(axis=1) / 1000.0,
        'mean_wind_cf': wind_cf.mean(axis=1),
    })
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(hourly['hour'], hourly['total_demand_gw'], color='tab:blue', linewidth=2)
    ax1.set_ylabel('Demand (GW)', color='tab:blue')
    ax1.set_xlabel('Hour')
    ax2 = ax1.twinx()
    ax2.plot(hourly['hour'], hourly['mean_wind_cf'], color='tab:green', linewidth=2)
    ax2.set_ylabel('Mean wind capacity factor', color='tab:green')
    plt.title('System demand and mean wind availability over the modeled week')
    savefig(IMG_DIR / 'overview_demand_wind.png')


def plot_network_map():
    buses = pd.read_csv(DATA_DIR / 'buses.csv')
    links = pd.read_csv(DATA_DIR / 'links.csv')
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for _, link in links.iterrows():
        b0 = buses[buses['name'] == link['bus0']].iloc[0]
        b1 = buses[buses['name'] == link['bus1']].iloc[0]
        ax.plot([b0['x'], b1['x']], [b0['y'], b1['y']], color='lightgray', linewidth=0.6 + link['p_nom'] / 2500.0, alpha=0.7)
    ax.scatter(buses['x'], buses['y'], s=50, color='tab:red', zorder=3)
    for _, row in buses.iterrows():
        ax.text(row['x'] + 0.05, row['y'] + 0.05, row['name'], fontsize=8)
    ax.set_xlabel('Longitude-like coordinate')
    ax.set_ylabel('Latitude-like coordinate')
    ax.set_title('Synthetic 20-bus GB-like network topology')
    savefig(IMG_DIR / 'network_topology.png')


def plot_capacity_mix():
    generators = pd.read_csv(DATA_DIR / 'generators.csv')
    capacities = generators.groupby('carrier')['p_nom'].sum().sort_values(ascending=False) / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=capacities.index, y=capacities.values, ax=ax, palette='deep')
    ax.set_ylabel('Installed capacity (GW)')
    ax.set_xlabel('Generator type')
    ax.set_title('Installed generation capacity mix')
    savefig(IMG_DIR / 'capacity_mix.png')


def plot_baseline_dispatch():
    metrics = pd.read_csv(OUTPUT_DIR / 'baseline' / 'metrics.csv')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        metrics['hour'],
        metrics['nuclear_mw'] / 1000.0,
        metrics['wind_mw'] / 1000.0,
        metrics['gas_mw'] / 1000.0,
        metrics['storage_discharge_mw'] / 1000.0,
        labels=['Nuclear', 'Wind', 'Gas', 'Storage discharge'],
        alpha=0.85,
    )
    ax.plot(metrics['hour'], metrics['demand_mw'] / 1000.0, color='black', linewidth=2, label='Demand')
    ax.plot(metrics['hour'], metrics['storage_charge_mw'] / 1000.0, color='tab:purple', linewidth=1.5, linestyle='--', label='Storage charge')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power (GW)')
    ax.set_title('Baseline hourly dispatch composition')
    ax.legend(ncol=3, fontsize=10)
    savefig(IMG_DIR / 'baseline_dispatch.png')


def plot_scenario_comparison(summary):
    plot_df = summary[['scenario', 'objective_gbp', 'wind_curtailment_share', 'max_line_utilization', 'unserved_share']].copy()
    plot_df['cost_million_gbp'] = plot_df['objective_gbp'] / 1e6
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.barplot(data=plot_df, x='scenario', y='cost_million_gbp', ax=axes[0, 0], palette='Blues_d')
    axes[0, 0].set_title('Total operating cost')
    axes[0, 0].set_ylabel('Million GBP')
    sns.barplot(data=plot_df, x='scenario', y='wind_curtailment_share', ax=axes[0, 1], palette='Greens_d')
    axes[0, 1].set_title('Wind curtailment share')
    axes[0, 1].set_ylabel('Share')
    sns.barplot(data=plot_df, x='scenario', y='max_line_utilization', ax=axes[1, 0], palette='Oranges_d')
    axes[1, 0].set_title('Maximum line utilization')
    axes[1, 0].set_ylabel('Per-unit of limit')
    sns.barplot(data=plot_df, x='scenario', y='unserved_share', ax=axes[1, 1], palette='Reds_d')
    axes[1, 1].set_title('Unserved energy share')
    axes[1, 1].set_ylabel('Share')
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=20)
    savefig(IMG_DIR / 'scenario_comparison.png')


def plot_congestion_duration():
    util = pd.read_csv(OUTPUT_DIR / 'baseline' / 'line_utilization.csv')
    sorted_vals = np.sort(util.max(axis=1).to_numpy())[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(sorted_vals) + 1), sorted_vals, linewidth=2)
    ax.axhline(0.8, color='tab:red', linestyle='--', label='0.8 p.u.')
    ax.set_xlabel('Hour rank')
    ax.set_ylabel('Maximum hourly line utilization')
    ax.set_title('Baseline congestion duration curve')
    ax.legend()
    savefig(IMG_DIR / 'congestion_duration.png')


def plot_storage_soc():
    path = OUTPUT_DIR / 'baseline' / 'storage_soc.csv'
    if not path.exists():
        return
    soc = pd.read_csv(path)
    soc['hour'] = np.arange(len(soc))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in soc.columns:
        if col != 'hour':
            ax.plot(soc['hour'], soc[col], linewidth=2, label=col)
    ax.set_xlabel('Hour')
    ax.set_ylabel('State of charge (MWh)')
    ax.set_title('Baseline storage state of charge trajectories')
    ax.legend(ncol=3, fontsize=9)
    savefig(IMG_DIR / 'storage_soc.png')


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    plot_demand_and_wind_overview()
    plot_network_map()
    plot_capacity_mix()
    plot_baseline_dispatch()
    plot_scenario_comparison(summary)
    plot_congestion_duration()
    plot_storage_soc()


if __name__ == '__main__':
    main()
