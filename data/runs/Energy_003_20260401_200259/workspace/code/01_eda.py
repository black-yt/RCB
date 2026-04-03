"""
Script 01: Exploratory Data Analysis (EDA) for HEEW Mini-Dataset
Analyzes hourly energy and weather data from ASU Campus (2014)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE   = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_003_20260401_200259")
DATA   = BASE / "data" / "HEEW_Mini-Dataset"
OUT    = BASE / "outputs"
IMGS   = BASE / "report" / "images"
OUT.mkdir(exist_ok=True)
IMGS.mkdir(exist_ok=True)

# ─── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    'electricity': '#2196F3',
    'heat':        '#FF5722',
    'cooling':     '#00BCD4',
    'pv':          '#FFC107',
    'ghg':         '#9C27B0',
}

# ─── 1. Load data ─────────────────────────────────────────────────────────────
def load_energy(csv_path):
    df = pd.read_csv(csv_path)
    # rename to short names
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour',
        'Electricity [kW]': 'electricity',
        'Heat [mmBTU]': 'heat',
        'Cooling Energy [Ton]': 'cooling',
        'PV Power Generation [kW]': 'pv',
        'Greenhouse Gas Emission [Ton]': 'ghg',
    }
    df = df.rename(columns=col_map)
    df['datetime'] = pd.to_datetime(dict(year=df.year, month=df.month,
                                         day=df.day, hour=df.hour))
    df = df.set_index('datetime').sort_index()
    return df[['electricity', 'heat', 'cooling', 'pv', 'ghg']]


def load_weather(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # first column is datetime
    dt_col = df.columns[0]
    df = df.rename(columns={dt_col: 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    col_map = {
        'Temperature [°F]':  'temperature',
        'Dew Point [°F]':    'dew_point',
        'Humidity [%]':      'humidity',
        'Wind Speed [mph]':  'wind_speed',
        'Wind Gust [mph]':   'wind_gust',
        'Pressure [in]':     'pressure',
        'Precipitation [in]':'precipitation',
    }
    df = df.rename(columns=col_map)
    return df[list(col_map.values())]


print("Loading data ...")
total_energy  = load_energy(DATA / "Total_energy.csv")
cn01_energy   = load_energy(DATA / "CN01_energy.csv")
weather       = load_weather(DATA / "Total_weather.csv")

buildings = {}
for i in range(1, 11):
    name = f"BN{i:03d}"
    buildings[name] = load_energy(DATA / f"{name}_energy.csv")

print(f"Total energy shape  : {total_energy.shape}")
print(f"Weather shape       : {weather.shape}")
print(f"Buildings loaded    : {len(buildings)}")

# ─── 2. Summary statistics ───────────────────────────────────────────────────
summary = total_energy.describe().T.round(3)
summary.to_csv(OUT / "total_energy_stats.csv")
weather.describe().T.round(3).to_csv(OUT / "weather_stats.csv")
print("\nTotal energy summary:\n", summary)

# ─── 3. Missing-value audit ───────────────────────────────────────────────────
missing = pd.DataFrame({
    'count':   total_energy.isnull().sum(),
    'pct':     (total_energy.isnull().mean() * 100).round(2),
})
missing.to_csv(OUT / "missing_values.csv")
print("\nMissing values:\n", missing)

# ─── 4. Plot 1: Full-year time-series overview ────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
labels = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling [Ton]',
          'PV Generation [kW]', 'GHG Emission [Ton]']
keys   = ['electricity', 'heat', 'cooling', 'pv', 'ghg']
cols   = [COLORS[k] for k in keys]

for ax, key, label, color in zip(axes, keys, labels, cols):
    ax.plot(total_energy.index, total_energy[key], color=color, lw=0.6, alpha=0.8)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
    # monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

axes[-1].set_xlabel('Month (2014)', fontsize=10)
fig.suptitle('HEEW Mini-Dataset — Total Campus Energy & GHG (2014)', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(IMGS / "fig01_timeseries_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig01")

# ─── 5. Plot 2: Monthly box-plots ────────────────────────────────────────────
total_energy['month'] = total_energy.index.month
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, key, label, color in zip(axes,
                                  ['electricity','heat','cooling'],
                                  ['Electricity [kW]','Heat [mmBTU]','Cooling [Ton]'],
                                  [COLORS['electricity'], COLORS['heat'], COLORS['cooling']]):
    data_by_month = [total_energy[total_energy.month == m][key].dropna() for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Monthly Distribution — {label}', fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('HEEW — Monthly Distributions of Energy Loads (2014)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(IMGS / "fig02_monthly_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig02")

# clean up temp column
total_energy.drop(columns=['month'], inplace=True)

# ─── 6. Plot 3: Average diurnal profiles ────────────────────────────────────
total_energy['hour'] = total_energy.index.hour
diurnal = total_energy.groupby('hour')[['electricity','heat','cooling','pv']].mean()
total_energy.drop(columns=['hour'], inplace=True)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(diurnal.index, diurnal['electricity'], color=COLORS['electricity'], lw=2.5, marker='o', ms=4, label='Electricity [kW]')
ax1.plot(diurnal.index, diurnal['cooling'],     color=COLORS['cooling'],     lw=2.5, marker='s', ms=4, label='Cooling [Ton]')
ax1.plot(diurnal.index, diurnal['heat'],        color=COLORS['heat'],        lw=2.5, marker='^', ms=4, label='Heat [mmBTU]')
ax2.plot(diurnal.index, diurnal['pv'],          color=COLORS['pv'],          lw=2.5, linestyle='--', marker='D', ms=4, label='PV [kW]')

ax1.set_xlabel('Hour of Day', fontsize=11)
ax1.set_ylabel('Electricity / Heat / Cooling', fontsize=11)
ax2.set_ylabel('PV Generation [kW]', fontsize=11, color=COLORS['pv'])
ax1.set_xticks(range(0, 24))
ax1.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax1.set_title('Average Diurnal Load Profiles — Total Campus (2014)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMGS / "fig03_diurnal_profiles.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig03")

# ─── 7. Plot 4: Weather time series ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
weather_keys = ['temperature', 'humidity', 'wind_speed']
weather_labels = ['Temperature [°F]', 'Humidity [%]', 'Wind Speed [mph]']
weather_colors = ['#E53935', '#43A047', '#1E88E5']

for ax, key, label, color in zip(axes, weather_keys, weather_labels, weather_colors):
    ax.plot(weather.index, weather[key], color=color, lw=0.5, alpha=0.8)
    # 7-day rolling mean
    roll = weather[key].rolling(window=168, center=True).mean()
    ax.plot(weather.index, roll, color='black', lw=1.5, label='7-day mean')
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Month (2014)', fontsize=10)
fig.suptitle('HEEW — Meteorological Observations (2014)', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(IMGS / "fig04_weather_timeseries.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig04")

# ─── 8. Plot 5: Building-level energy comparison ────────────────────────────
bldg_means = {name: df['electricity'].mean() for name, df in buildings.items()}
bldg_maxes = {name: df['electricity'].max() for name, df in buildings.items()}

names = list(bldg_means.keys())
means = list(bldg_means.values())
maxes = list(bldg_maxes.values())

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, means, width, label='Mean', color=COLORS['electricity'], alpha=0.8)
bars2 = ax.bar(x + width/2, maxes, width, label='Max',  color='#1565C0', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, fontsize=9)
ax.set_ylabel('Electricity [kW]', fontsize=10)
ax.set_title('Mean vs. Maximum Electricity Demand per Building (2014)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(IMGS / "fig05_building_electricity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig05")

# ─── 9. Save processed data ──────────────────────────────────────────────────
total_energy.to_csv(OUT / "total_energy_processed.csv")
weather.to_csv(OUT / "weather_processed.csv")
print("\nAll EDA outputs saved.")
print("Script 01 complete.")
