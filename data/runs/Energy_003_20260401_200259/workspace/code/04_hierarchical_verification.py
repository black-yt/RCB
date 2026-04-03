"""
Script 04: Hierarchical Aggregation Consistency Verification
Validates that: Total ≈ CN01 + (BN001..BN010 - CN01)
and that CN01 ≈ sum(BN001..BN010)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


total = load_energy(DATA / "Total_energy.csv")
cn01  = load_energy(DATA / "CN01_energy.csv")
buildings = {f"BN{i:03d}": load_energy(DATA / f"BN{i:03d}_energy.csv") for i in range(1, 11)}

# ─── 1. Sum of individual buildings ──────────────────────────────────────────
bldg_sum = sum(buildings.values())

# ─── 2. Relative error: |bldg_sum - cn01| / cn01 ────────────────────────────
rel_err_cn01 = ((bldg_sum - cn01).abs() / (cn01.abs() + 1e-9)).describe()
print("Relative error: sum(BN001..BN010) vs CN01")
print(rel_err_cn01.round(6))

# ─── 3. Relative error: cn01 vs total ────────────────────────────────────────
# Note: CN01 may be a community subset; compare directly
rel_err_total = ((cn01 - total).abs() / (total.abs() + 1e-9)).describe()
print("\nRelative error: CN01 vs Total")
print(rel_err_total.round(6))

# ─── 4. MAPE, RMSE, R2 ───────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, r2_score

def mape(y_true, y_pred):
    mask = y_true.abs() > 1e-6
    return (((y_true - y_pred).abs() / y_true.abs())[mask]).mean() * 100

results = {}
for col in ['electricity','heat','cooling','pv','ghg']:
    y_true = cn01[col].values
    y_pred = bldg_sum[col].values
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    mape_v = mape(cn01[col], bldg_sum[col])
    results[col] = {'RMSE': round(rmse, 4), 'R2': round(r2, 6), 'MAPE(%)': round(mape_v, 4)}

results_df = pd.DataFrame(results).T
print("\nAggregation consistency metrics (BN sum vs CN01):")
print(results_df)
results_df.to_csv(OUT / "hierarchical_consistency_metrics.csv")

# ─── Plot 12: Scatter — sum(BN) vs CN01 for electricity ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, color, label in zip(axes,
                                   ['electricity','heat','cooling'],
                                   [COLORS['electricity'],COLORS['heat'],COLORS['cooling']],
                                   ['Electricity [kW]','Heat [mmBTU]','Cooling [Ton]']):
    x = cn01[col].values
    y = bldg_sum[col].values
    ax.scatter(x, y, c=color, alpha=0.3, s=5)
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([mn,mx],[mn,mx], 'k--', lw=1.5, label='1:1 line')
    r2 = r2_score(x, y)
    ax.set_xlabel(f'CN01 {label}', fontsize=9)
    ax.set_ylabel(f'Sum(BN001-BN010) {label}', fontsize=9)
    ax.set_title(f'{label.split("[")[0]}: R²={r2:.4f}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Hierarchical Aggregation Consistency: Sum(Buildings) vs. Community (CN01)',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(IMGS / "fig12_hierarchical_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12")

# ─── Plot 13: Time-series residuals ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for ax, col, color, label in zip(axes,
                                   ['electricity','heat','cooling'],
                                   [COLORS['electricity'],COLORS['heat'],COLORS['cooling']],
                                   ['Electricity [kW]','Heat [mmBTU]','Cooling [Ton]']):
    residual = bldg_sum[col] - cn01[col]
    ax.plot(residual.index, residual, color=color, lw=0.5, alpha=0.8)
    ax.axhline(0, color='black', lw=1.5, linestyle='--')
    ax.set_ylabel(f'Residual {label}', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_title(f'{col.capitalize()} — Aggregation Residual (Sum-BN minus CN01)', fontsize=9)

axes[-1].set_xlabel('Month (2014)', fontsize=10)
fig.suptitle('Temporal Aggregation Residuals: Σ(Buildings) − Community (CN01)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(IMGS / "fig13_hierarchical_residuals.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13")

# ─── Plot 14: Building electricity share stacked area ────────────────────────
# Resample to daily for visibility
daily = {name: df['electricity'].resample('D').mean() for name, df in buildings.items()}
daily_df = pd.DataFrame(daily)

fig, ax = plt.subplots(figsize=(16, 6))
ax.stackplot(daily_df.index, daily_df.T.values,
             labels=daily_df.columns,
             alpha=0.85,
             colors=plt.cm.tab10(np.linspace(0,1,10)))
ax.set_ylabel('Mean Daily Electricity [kW]', fontsize=10)
ax.set_xlabel('Month (2014)', fontsize=10)
ax.legend(loc='upper left', ncol=2, fontsize=8)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.grid(True, alpha=0.3)
ax.set_title('Stacked Daily Electricity — Individual Buildings (BN001–BN010)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMGS / "fig14_stacked_area.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig14")

print("Script 04 complete.")
