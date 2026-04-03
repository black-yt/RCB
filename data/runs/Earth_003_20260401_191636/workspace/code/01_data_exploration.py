"""
Data Exploration and Overview Analysis
Explores ERA5 input data and FuXi 6h forecast output
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

# Load data
print("Loading data...")
ds_input = nc.Dataset('../data/20231012-06_input_netcdf.nc', 'r')
ds_forecast = nc.Dataset('../data/006.nc', 'r')

lat = ds_input.variables['lat'][:]         # (181,) from 90 to -90
lon = ds_input.variables['lon'][:]         # (360,) from 0 to 359
levels_raw = ds_input.variables['level'][:]
level_names = [''.join(row.tolist()) for row in levels_raw.astype(str)]

input_data = ds_input.variables['data'][:]    # (2, 70, 181, 360) - t0 and t+6h
forecast_data = ds_forecast.variables['data'][:]  # (1, 1, 70, 181, 360) - 6h forecast

# Squeeze forecast
forecast_6h = forecast_data[0, 0]  # (70, 181, 360)
truth_6h = input_data[1]           # (70, 181, 360) - actual t+6h
init_state = input_data[0]         # (70, 181, 360) - initial conditions

print(f"Level names: {level_names}")
print(f"Input data shape: {input_data.shape}")
print(f"Forecast data shape: {forecast_6h.shape}")
print(f"Lat range: {lat[0]:.1f} to {lat[-1]:.1f}")
print(f"Lon range: {lon[0]:.1f} to {lon[-1]:.1f}")

# Variable groups
VAR_GROUPS = {
    'Geopotential (Z)': [i for i, n in enumerate(level_names) if n.startswith('Z')],
    'Temperature (T)': [i for i, n in enumerate(level_names) if n.startswith('T') and n != 'T2M' and n != 'TP'],
    'U-wind': [i for i, n in enumerate(level_names) if n.startswith('U') and n != 'U10'],
    'V-wind': [i for i, n in enumerate(level_names) if n.startswith('V') and n != 'V10'],
    'Relative Humidity (R)': [i for i, n in enumerate(level_names) if n.startswith('R')],
    'Surface': [i for i, n in enumerate(level_names) if n in ['T2M', 'U10', 'V10', 'MSL', 'TP']],
}

print("\nVariable groups:")
for name, indices in VAR_GROUPS.items():
    print(f"  {name}: indices {indices}, names {[level_names[i] for i in indices]}")

# Save processed data to outputs
np.save('../outputs/level_names.npy', np.array(level_names))
np.save('../outputs/lat.npy', np.array(lat))
np.save('../outputs/lon.npy', np.array(lon))
np.save('../outputs/init_state.npy', np.array(init_state))
np.save('../outputs/truth_6h.npy', np.array(truth_6h))
np.save('../outputs/forecast_6h.npy', np.array(forecast_6h))

# =====================
# Figure 1: Data Overview - Global Maps of Key Variables
# =====================
print("\nGenerating Figure 1: Data overview...")

# Key variables to display
key_vars = {
    'Z500 (Geopotential 500 hPa)': 7,
    'T850 (Temperature 850 hPa)': 23,
    'U850 (U-wind 850 hPa)': 36,
    'MSL (Mean Sea Level Pressure)': 68,
    'T2M (2m Temperature)': 65,
    'TP (Total Precipitation)': 69,
}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('ERA5 Reanalysis: Initial State\n(2023-10-12 00:00 UTC)',
             fontsize=14, fontweight='bold', y=0.98)

LON, LAT = np.meshgrid(lon, lat)

for idx, (varname, var_idx) in enumerate(key_vars.items()):
    ax = axes[idx // 2, idx % 2]
    field = init_state[var_idx]

    if 'TP' in varname:
        # Precipitation - use sequential colormap
        im = ax.contourf(LON, LAT, field, levels=20, cmap='Blues')
    elif 'T' in varname:
        # Temperature - diverging
        vmax = np.percentile(np.abs(field), 97)
        im = ax.contourf(LON, LAT, field, levels=20, cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax)
    else:
        im = ax.contourf(LON, LAT, field, levels=20, cmap='viridis')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Normalized value')
    ax.set_title(varname, fontsize=10, fontweight='bold')
    ax.set_xlabel('Longitude (°)', fontsize=8)
    ax.set_ylabel('Latitude (°)', fontsize=8)
    ax.set_xlim([0, 360])
    ax.set_ylim([-90, 90])
    ax.grid(True, alpha=0.3)

    # Add some annotation
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('../report/images/fig1_data_overview.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved fig1_data_overview.png")

# =====================
# Figure 2: Variable Distribution Analysis
# =====================
print("Generating Figure 2: Variable distributions...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of ERA5 Variables (Normalized)\nInitial State vs. 6h Later',
             fontsize=13, fontweight='bold')

dist_vars = [
    ('Z500', 7, 'Z500'),
    ('T850', 23, 'T850'),
    ('U850', 36, 'U850'),
    ('T2M', 65, 'T2M'),
    ('MSL', 68, 'MSL'),
    ('TP', 69, 'TP'),
]

for idx, (name, var_idx, label) in enumerate(dist_vars):
    ax = axes[idx // 3, idx % 3]

    v0 = init_state[var_idx].flatten()
    v1 = truth_6h[var_idx].flatten()
    vf = forecast_6h[var_idx].flatten()

    bins = np.linspace(min(v0.min(), v1.min()), max(v0.max(), v1.max()), 60)

    ax.hist(v0, bins=bins, alpha=0.5, label='t=0h (ERA5)', density=True, color='blue')
    ax.hist(v1, bins=bins, alpha=0.5, label='t=6h (ERA5)', density=True, color='green')
    ax.hist(vf, bins=bins, alpha=0.5, label='t=6h (FuXi)', density=True, color='red')

    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized value', fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../report/images/fig2_variable_distributions.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved fig2_variable_distributions.png")

# =====================
# Figure 3: Vertical profile of RMSE
# =====================
print("Generating Figure 3: Pressure level RMSE profiles...")

pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Compute RMSE for upper-air variables
def rmse(pred, truth):
    return np.sqrt(np.nanmean((pred - truth)**2))

# For each variable group (Z, T, U, V, R), compute RMSE at each pressure level
upper_air_groups = {
    'Z': [i for i, n in enumerate(level_names) if n.startswith('Z')],
    'T': [i for i, n in enumerate(level_names) if n.startswith('T') and n not in ('T2M', 'TP')],
    'U': [i for i, n in enumerate(level_names) if n.startswith('U') and n != 'U10'],
    'V': [i for i, n in enumerate(level_names) if n.startswith('V') and n != 'V10'],
    'R': [i for i, n in enumerate(level_names) if n.startswith('R')],
}

colors = {'Z': 'navy', 'T': 'firebrick', 'U': 'green', 'V': 'darkorange', 'R': 'purple'}

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: RMSE vs pressure level
ax = axes[0]
for var_name, indices in upper_air_groups.items():
    rmse_vals = []
    for idx in indices:
        r = rmse(forecast_6h[idx], truth_6h[idx])
        rmse_vals.append(r)
    ax.plot(rmse_vals, pressure_levels, 'o-', color=colors[var_name],
            label=var_name, linewidth=2, markersize=6)

ax.set_ylabel('Pressure Level (hPa)', fontsize=11)
ax.set_xlabel('RMSE (normalized units)', fontsize=11)
ax.set_title('6h Forecast RMSE by Pressure Level\n(FuXi vs ERA5)', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.set_yscale('log')
ax.set_yticks(pressure_levels)
ax.set_yticklabels(pressure_levels, fontsize=8)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Right: Bias (mean error) by pressure level
ax = axes[1]
for var_name, indices in upper_air_groups.items():
    bias_vals = []
    for idx in indices:
        bias = np.nanmean(forecast_6h[idx] - truth_6h[idx])
        bias_vals.append(bias)
    ax.plot(bias_vals, pressure_levels, 'o-', color=colors[var_name],
            label=var_name, linewidth=2, markersize=6)

ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylabel('Pressure Level (hPa)', fontsize=11)
ax.set_xlabel('Bias (normalized units)', fontsize=11)
ax.set_title('6h Forecast Bias by Pressure Level\n(FuXi vs ERA5)', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.set_yscale('log')
ax.set_yticks(pressure_levels)
ax.set_yticklabels(pressure_levels, fontsize=8)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('../report/images/fig3_rmse_profiles.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved fig3_rmse_profiles.png")

# =====================
# Save RMSE statistics to outputs
# =====================
rmse_all = {}
bias_all = {}
for i, name in enumerate(level_names):
    rmse_all[name] = float(rmse(forecast_6h[i], truth_6h[i]))
    bias_all[name] = float(np.nanmean(forecast_6h[i] - truth_6h[i]))

import json
with open('../outputs/rmse_6h.json', 'w') as f:
    json.dump(rmse_all, f, indent=2)
with open('../outputs/bias_6h.json', 'w') as f:
    json.dump(bias_all, f, indent=2)

print("\nRMSE statistics saved to outputs/rmse_6h.json")
print("\nTop 10 highest RMSE variables:")
for k, v in sorted(rmse_all.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k}: RMSE={v:.4f}, Bias={bias_all[k]:.4f}")

print("\nTop 10 lowest RMSE variables:")
for k, v in sorted(rmse_all.items(), key=lambda x: x[1])[:10]:
    print(f"  {k}: RMSE={v:.4f}, Bias={bias_all[k]:.4f}")

ds_input.close()
ds_forecast.close()
print("\nData exploration complete!")
