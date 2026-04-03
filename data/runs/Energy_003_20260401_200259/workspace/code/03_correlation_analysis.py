"""
Script 03: Correlation Analysis — Weather vs. Energy Loads
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_003_20260401_200259")
DATA = BASE / "data" / "HEEW_Mini-Dataset"
OUT  = BASE / "outputs"
IMGS = BASE / "report" / "images"

# ─── Load processed data ─────────────────────────────────────────────────────
energy  = pd.read_csv(OUT / "total_energy_processed.csv", index_col=0, parse_dates=True)
weather = pd.read_csv(OUT / "weather_processed.csv",      index_col=0, parse_dates=True)

# Align on common index
merged = pd.concat([energy, weather], axis=1).dropna()
print(f"Merged shape: {merged.shape}")

# ─── Plot 8: Pearson correlation heatmap ─────────────────────────────────────
corr = merged.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Pearson Correlation Matrix — Energy & Weather Variables (2014)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMGS / "fig08_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig08")
corr.to_csv(OUT / "correlation_matrix.csv")

# ─── Plot 9: Scatter — Temperature vs. each energy variable ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [
    ('temperature', 'electricity', '#2196F3', 'Electricity [kW]'),
    ('temperature', 'cooling',     '#00BCD4', 'Cooling [Ton]'),
    ('temperature', 'heat',        '#FF5722', 'Heat [mmBTU]'),
]
for ax, (x_col, y_col, color, ylabel) in zip(axes, pairs):
    ax.scatter(merged[x_col], merged[y_col], c=color, alpha=0.2, s=5)
    # linear fit
    m, b = np.polyfit(merged[x_col], merged[y_col], 1)
    xr = np.linspace(merged[x_col].min(), merged[x_col].max(), 100)
    ax.plot(xr, m*xr + b, color='black', lw=2, label=f'r={corr.loc[y_col, x_col]:.2f}')
    ax.set_xlabel('Temperature [°F]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'Temperature vs. {ylabel.split("[")[0]}', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('Temperature–Load Scatter Plots (hourly, 2014)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(IMGS / "fig09_temp_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig09")

# ─── Plot 10: Seasonal correlation breakdown ──────────────────────────────────
merged['season'] = merged.index.month.map(
    lambda m: 'Winter' if m in [12,1,2] else
              'Spring' if m in [3,4,5]  else
              'Summer' if m in [6,7,8]  else 'Fall'
)

season_order  = ['Winter','Spring','Summer','Fall']
season_colors = {'Winter':'#1E88E5','Spring':'#43A047','Summer':'#E53935','Fall':'#FF8F00'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, y_col, ylabel in zip(axes,
                               ['electricity','cooling','heat'],
                               ['Electricity [kW]','Cooling [Ton]','Heat [mmBTU]']):
    for season in season_order:
        sub = merged[merged.season == season]
        ax.scatter(sub['temperature'], sub[y_col],
                   color=season_colors[season], alpha=0.3, s=6, label=season)
    ax.set_xlabel('Temperature [°F]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'Season-stratified: Temp vs {ylabel.split("[")[0]}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Seasonal Stratification of Temperature–Load Relationships', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(IMGS / "fig10_seasonal_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10")

# ─── Plot 11: Pairplot (subset) ───────────────────────────────────────────────
subset_cols = ['electricity','cooling','pv','temperature','humidity']
sample = merged[subset_cols].sample(n=min(2000, len(merged)), random_state=42)
pg = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha':0.3,'s':5},
                  diag_kws={'fill':True})
pg.fig.suptitle('Pairplot — Energy and Weather Subset (sampled 2000 points)', y=1.01, fontsize=11)
pg.fig.savefig(IMGS / "fig11_pairplot.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved fig11")

# ─── Report top correlations ─────────────────────────────────────────────────
energy_cols  = ['electricity','heat','cooling','pv','ghg']
weather_cols = ['temperature','dew_point','humidity','wind_speed','wind_gust','pressure','precipitation']

cross_corr = corr.loc[energy_cols, weather_cols]
print("\nEnergy–Weather Pearson correlation:\n", cross_corr.round(3))
cross_corr.to_csv(OUT / "energy_weather_correlation.csv")
print("Script 03 complete.")
