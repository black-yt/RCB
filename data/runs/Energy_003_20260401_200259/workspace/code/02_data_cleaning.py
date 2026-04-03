"""
Script 02: Data Cleaning Algorithms for HEEW Mini-Dataset
Implements outlier detection (IQR, Z-score), anomaly flagging, and interpolation.
"""

import os
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

COLORS = {
    'electricity': '#2196F3',
    'heat':        '#FF5722',
    'cooling':     '#00BCD4',
    'pv':          '#FFC107',
    'ghg':         '#9C27B0',
}

# ─── load ─────────────────────────────────────────────────────────────────────
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
                                         day=df.day,  hour=df.hour))
    return df.set_index('datetime').sort_index()[['electricity','heat','cooling','pv','ghg']]

total = load_energy(DATA / "Total_energy.csv")

# ─── Algorithm 1: Physical bounds check ───────────────────────────────────────
# Non-negative energy; PV is zero at night (hours 20-05)
def physical_bounds_check(df):
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        flags[col] |= (df[col] < 0)          # negative values impossible
    # PV should be zero outside [06, 19]
    night_mask = ~df.index.hour.isin(range(6, 20))
    flags['pv'] |= (df['pv'] > 1) & night_mask
    return flags

phys_flags = physical_bounds_check(total)
print("Physical-bound violations per column:")
print(phys_flags.sum())

# ─── Algorithm 2: IQR-based outlier detection ─────────────────────────────────
def iqr_outlier(series, k=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - k*IQR) | (series > Q3 + k*IQR)

iqr_flags = pd.DataFrame({col: iqr_outlier(total[col]) for col in total.columns})
print("\nIQR outlier counts (k=3):")
print(iqr_flags.sum())

# ─── Algorithm 3: Z-score outlier detection ───────────────────────────────────
def zscore_outlier(series, threshold=3.5):
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold

zs_flags = pd.DataFrame({col: zscore_outlier(total[col]) for col in total.columns})
print("\nZ-score outlier counts (|z|>3.5):")
print(zs_flags.sum())

# ─── Algorithm 4: Rolling window anomaly detection ────────────────────────────
def rolling_anomaly(series, window=24, n_sigma=3):
    roll_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    roll_std  = series.rolling(window=window, center=True, min_periods=1).std().fillna(1e-6)
    z = (series - roll_mean) / roll_std
    return z.abs() > n_sigma

roll_flags = pd.DataFrame({col: rolling_anomaly(total[col]) for col in total.columns})
print("\nRolling-window anomaly counts:")
print(roll_flags.sum())

# ─── Combine flags and compute anomaly score ─────────────────────────────────
combined = (iqr_flags.astype(int) + zs_flags.astype(int) + roll_flags.astype(int))
combined.to_csv(OUT / "anomaly_scores.csv")

# ─── Algorithm 5: Linear interpolation for flagged points ─────────────────────
def clean_series(series, flag_series):
    cleaned = series.copy()
    cleaned[flag_series] = np.nan
    cleaned = cleaned.interpolate(method='time')
    return cleaned

total_cleaned = total.copy()
any_flag = iqr_flags | zs_flags | phys_flags
for col in total.columns:
    total_cleaned[col] = clean_series(total[col], any_flag[col])
total_cleaned.to_csv(OUT / "total_energy_cleaned.csv")
print("\nCleaned data saved.")

# ─── Plot 6: Anomaly detection summary ───────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Top: electricity with IQR and Z-score flags
ax = axes[0]
ax.plot(total.index, total['electricity'], color='#90CAF9', lw=0.6, label='Raw', alpha=0.8)
ax.plot(total_cleaned.index, total_cleaned['electricity'], color=COLORS['electricity'],
        lw=0.8, label='Cleaned', alpha=0.9)
outlier_idx = iqr_flags[iqr_flags['electricity']].index
ax.scatter(outlier_idx, total.loc[outlier_idx, 'electricity'],
           color='red', s=20, zorder=5, label='IQR outlier')
ax.set_ylabel('Electricity [kW]')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Outlier Detection & Interpolation — Electricity', fontsize=11)
ax.grid(True, alpha=0.3)

# Bottom: PV with physical-bound flag
ax = axes[1]
ax.plot(total.index, total['pv'], color='#FFE082', lw=0.6, label='Raw', alpha=0.8)
ax.plot(total_cleaned.index, total_cleaned['pv'], color=COLORS['pv'],
        lw=0.8, label='Cleaned', alpha=0.9)
pv_flag_idx = phys_flags[phys_flags['pv']].index
ax.scatter(pv_flag_idx, total.loc[pv_flag_idx, 'pv'],
           color='red', s=20, zorder=5, label='Nighttime violation')
ax.set_ylabel('PV Generation [kW]')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Physical-Bound Anomaly Detection — PV Generation', fontsize=11)
ax.grid(True, alpha=0.3)
import matplotlib.dates as mdates
axes[-1].set_xlabel('Month (2014)')
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout()
fig.savefig(IMGS / "fig06_anomaly_detection.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig06")

# ─── Plot 7: Before/After cleaning comparison ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, color in zip(axes,
                           ['electricity', 'heat', 'cooling'],
                           [COLORS['electricity'], COLORS['heat'], COLORS['cooling']]):
    ax.hist(total[col].dropna(), bins=60, alpha=0.5, color='grey', label='Raw')
    ax.hist(total_cleaned[col].dropna(), bins=60, alpha=0.7, color=color, label='Cleaned')
    ax.set_xlabel(col.capitalize(), fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{col.capitalize()} Distribution', fontsize=10)

fig.suptitle('Data Cleaning — Before vs. After Distributions', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(IMGS / "fig07_cleaning_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig07")

# ─── Summary table ────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'IQR_outliers':        iqr_flags.sum(),
    'Zscore_outliers':     zs_flags.sum(),
    'Physical_violations': phys_flags.sum(),
    'Rolling_anomalies':   roll_flags.sum(),
})
summary['Total_flagged'] = (any_flag).sum()
summary['Flag_pct'] = (summary['Total_flagged'] / len(total) * 100).round(3)
print("\nCleaning summary:\n", summary)
summary.to_csv(OUT / "cleaning_summary.csv")
print("Script 02 complete.")
