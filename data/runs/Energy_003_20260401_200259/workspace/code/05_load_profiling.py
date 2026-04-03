"""
Script 05: Load Profiling — PV Generation, GHG Emissions, Temporal Patterns
"""

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

BASE = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_003_20260401_200259")
DATA = BASE / "data" / "HEEW_Mini-Dataset"
OUT  = BASE / "outputs"
IMGS = BASE / "report" / "images"

COLORS = {
    'electricity': '#2196F3',
    'heat':        '#FF5722',
    'cooling':     '#00BCD4',
    'pv':          '#FFC107',
    'ghg':         '#9C27B0',
}

def load_energy(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        'year':'year','month':'month','day':'day','hour':'hour',
        'Electricity [kW]':'electricity',
        'Heat [mmBTU]':'heat',
        'Cooling Energy [Ton]':'cooling',
        'PV Power Generation [kW]':'pv',
        'Greenhouse Gas Emission [Ton]':'ghg',
    }
    df = df.rename(columns=col_map)
    df['datetime'] = pd.to_datetime(dict(year=df.year, month=df.month,
                                         day=df.day, hour=df.hour))
    return df.set_index('datetime').sort_index()[['electricity','heat','cooling','pv','ghg']]

def load_weather(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    dt_col = df.columns[0]
    df = df.rename(columns={dt_col: 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    col_map = {
        'Temperature [°F]':'temperature','Dew Point [°F]':'dew_point',
        'Humidity [%]':'humidity','Wind Speed [mph]':'wind_speed',
        'Wind Gust [mph]':'wind_gust','Pressure [in]':'pressure',
        'Precipitation [in]':'precipitation',
    }
    df = df.rename(columns=col_map)
    return df[list(col_map.values())]

total   = load_energy(DATA / "Total_energy.csv")
weather = load_weather(DATA / "Total_weather.csv")

# Add derived temporal features
total['hour']    = total.index.hour
total['month']   = total.index.month
total['weekday'] = total.index.dayofweek   # 0=Mon, 6=Sun
total['is_weekend'] = total['weekday'].isin([5,6])

# ─── Plot 15: Heatmap — electricity by hour × month ──────────────────────────
pivot_elec = total.groupby(['month','hour'])['electricity'].mean().unstack('hour')
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_elec, cmap='YlOrRd', annot=False, ax=ax, linewidths=0.2,
            cbar_kws={'label': 'Mean Electricity [kW]'})
ax.set_yticklabels(month_labels, rotation=0)
ax.set_xlabel('Hour of Day', fontsize=10)
ax.set_ylabel('Month', fontsize=10)
ax.set_title('Mean Electricity Demand: Hour × Month Heatmap (2014)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMGS / "fig15_electricity_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig15")

# ─── Plot 16: Heatmap — PV generation by hour × month ───────────────────────
pivot_pv = total.groupby(['month','hour'])['pv'].mean().unstack('hour')

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_pv, cmap='YlGnBu', annot=False, ax=ax, linewidths=0.2,
            cbar_kws={'label': 'Mean PV Generation [kW]'})
ax.set_yticklabels(month_labels, rotation=0)
ax.set_xlabel('Hour of Day', fontsize=10)
ax.set_ylabel('Month', fontsize=10)
ax.set_title('Mean PV Generation: Hour × Month Heatmap (2014)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMGS / "fig16_pv_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig16")

# ─── Plot 17: GHG emissions — monthly violin ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data_by_month = [total[total.month == m]['ghg'].dropna() for m in range(1, 13)]
vp = ax.violinplot(data_by_month, positions=range(1, 13), showmedians=True,
                   showextrema=True)
for pc in vp['bodies']:
    pc.set_facecolor(COLORS['ghg'])
    pc.set_alpha(0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels, rotation=30)
ax.set_ylabel('GHG Emission [Ton]', fontsize=10)
ax.set_title('Monthly GHG Emission Distribution (2014)', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(IMGS / "fig17_ghg_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig17")

# ─── Plot 18: Weekday vs Weekend profile ─────────────────────────────────────
weekday_profile = total[~total.is_weekend].groupby('hour')[['electricity','cooling','heat']].mean()
weekend_profile = total[total.is_weekend].groupby('hour')[['electricity','cooling','heat']].mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color, label in zip(axes,
                                   ['electricity','cooling','heat'],
                                   [COLORS['electricity'],COLORS['cooling'],COLORS['heat']],
                                   ['Electricity [kW]','Cooling [Ton]','Heat [mmBTU]']):
    ax.plot(weekday_profile.index, weekday_profile[col], color=color, lw=2.5,
            marker='o', ms=4, label='Weekday')
    ax.plot(weekend_profile.index, weekend_profile[col], color=color, lw=2.5,
            linestyle='--', marker='s', ms=4, alpha=0.7, label='Weekend')
    ax.fill_between(weekday_profile.index,
                    weekday_profile[col], weekend_profile[col],
                    alpha=0.15, color=color)
    ax.set_xlabel('Hour of Day', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Weekday vs. Weekend — {label.split("[")[0]}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 3))

fig.suptitle('Weekday vs. Weekend Diurnal Load Profiles (2014)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(IMGS / "fig18_weekday_weekend.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig18")

# ─── Plot 19: Monthly energy balance (pie chart) ─────────────────────────────
monthly_mean = total.groupby('month')[['electricity','heat','cooling','pv','ghg']].mean()
summer_mean  = monthly_mean.loc[[6,7,8]].mean()
winter_mean  = monthly_mean.loc[[12,1,2]].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mean_vals, title in zip(axes, [summer_mean, winter_mean], ['Summer (Jun-Aug)', 'Winter (Dec-Feb)']):
    cols_plot = ['electricity','heat','cooling','pv']
    sizes     = [abs(mean_vals[c]) for c in cols_plot]
    colors_p  = [COLORS[c] for c in cols_plot]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=['Electricity','Heat','Cooling','PV'],
        colors=colors_p, autopct='%1.1f%%', startangle=90,
        pctdistance=0.8, textprops={'fontsize':9}
    )
    ax.set_title(f'Average Load Composition — {title}', fontsize=11)

fig.suptitle('Energy Load Composition: Summer vs. Winter (2014)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMGS / "fig19_load_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig19")

# ─── Plot 20: PV self-sufficiency ratio ──────────────────────────────────────
total['pv_ratio'] = total['pv'] / (total['electricity'] + 1e-9)
monthly_pv_ratio  = total.groupby('month')['pv_ratio'].mean() * 100

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(range(1, 13), monthly_pv_ratio.values,
              color=[COLORS['pv']]*12, alpha=0.8, edgecolor='black', lw=0.5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=30)
ax.set_ylabel('PV / Electricity Ratio [%]', fontsize=10)
ax.set_title('Monthly PV Self-Sufficiency Ratio (PV/Electricity, 2014)', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, monthly_pv_ratio.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
fig.savefig(IMGS / "fig20_pv_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig20")

# ─── Save load profile statistics ────────────────────────────────────────────
monthly_mean.to_csv(OUT / "monthly_mean_loads.csv")
weekday_profile.to_csv(OUT / "weekday_diurnal_profile.csv")
weekend_profile.to_csv(OUT / "weekend_diurnal_profile.csv")
print("Script 05 complete.")
