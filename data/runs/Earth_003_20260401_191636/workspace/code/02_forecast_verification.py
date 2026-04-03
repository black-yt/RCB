"""
Forecast Verification and Error Analysis
Computes detailed skill metrics for the FuXi 6h forecast
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

# Load pre-computed arrays
print("Loading arrays...")
level_names = list(np.load('../outputs/level_names.npy'))
lat = np.load('../outputs/lat.npy')
lon = np.load('../outputs/lon.npy')
init_state = np.load('../outputs/init_state.npy')
truth_6h = np.load('../outputs/truth_6h.npy')
forecast_6h = np.load('../outputs/forecast_6h.npy')

n_vars, n_lat, n_lon = truth_6h.shape

# Compute latitude weights for area-weighted metrics
lat_weights = np.cos(np.deg2rad(lat))
lat_weights = lat_weights / lat_weights.mean()
lat_weights_2d = lat_weights[:, np.newaxis]  # (181, 1)

def weighted_rmse(pred, truth, w):
    """Area-weighted RMSE"""
    return np.sqrt(np.average((pred - truth)**2, weights=w * np.ones_like(pred)))

def weighted_mae(pred, truth, w):
    """Area-weighted MAE"""
    return np.average(np.abs(pred - truth), weights=w * np.ones_like(pred))

def weighted_corr(pred, truth, w):
    """Area-weighted correlation"""
    w_flat = (w * np.ones_like(pred)).flatten()
    p = pred.flatten()
    t = truth.flatten()
    mean_p = np.average(p, weights=w_flat)
    mean_t = np.average(t, weights=w_flat)
    cov = np.average((p - mean_p) * (t - mean_t), weights=w_flat)
    std_p = np.sqrt(np.average((p - mean_p)**2, weights=w_flat))
    std_t = np.sqrt(np.average((t - mean_t)**2, weights=w_flat))
    if std_p * std_t == 0:
        return 0.0
    return cov / (std_p * std_t)

def anomaly_correlation(pred, truth, climo, w):
    """Anomaly Correlation Coefficient (ACC)"""
    pred_anom = pred - climo
    truth_anom = truth - climo
    w_flat = (w * np.ones_like(pred)).flatten()
    p = pred_anom.flatten()
    t = truth_anom.flatten()
    num = np.average(p * t, weights=w_flat)
    denom = np.sqrt(np.average(p**2, weights=w_flat) * np.average(t**2, weights=w_flat))
    if denom == 0:
        return 0.0
    return num / denom

print("Computing forecast verification metrics...")

# Use climatology as the initial state (persistence)
climo = init_state  # Use t=0 as climatology proxy

metrics = {}
for i, name in enumerate(level_names):
    pred = forecast_6h[i]
    truth = truth_6h[i]
    init = init_state[i]
    w = lat_weights_2d

    rmse_fcst = weighted_rmse(pred, truth, w)
    rmse_pers = weighted_rmse(init, truth, w)  # Persistence forecast RMSE
    mae_fcst = weighted_mae(pred, truth, w)
    corr = weighted_corr(pred, truth, w)
    acc = anomaly_correlation(pred, truth, init, w)

    # Skill score relative to persistence
    skill = 1.0 - rmse_fcst / (rmse_pers + 1e-10)

    metrics[name] = {
        'rmse': float(rmse_fcst),
        'rmse_persistence': float(rmse_pers),
        'mae': float(mae_fcst),
        'corr': float(corr),
        'acc': float(acc),
        'skill_vs_persistence': float(skill),
    }

with open('../outputs/verification_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics computed. Sample (Z500):")
print(json.dumps(metrics.get('Z500', {}), indent=2))

# =====================
# Figure 4: Forecast Error Maps for Key Variables
# =====================
print("\nGenerating Figure 4: Error maps...")

key_vars = [
    ('Z500', 7, 'Z500 500hPa Geopotential'),
    ('T850', 23, 'T850 850hPa Temperature'),
    ('U850', 36, 'U850 850hPa U-wind'),
    ('T2M', 65, 'T2M 2m Temperature'),
    ('MSL', 68, 'MSL Mean Sea Level Pressure'),
    ('TP', 69, 'TP Total Precipitation'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('FuXi 6h Forecast Error Maps (Forecast − ERA5)\nInitialized: 2023-10-12 00:00 UTC, Valid: 2023-10-12 06:00 UTC',
             fontsize=12, fontweight='bold')

LON, LAT = np.meshgrid(lon, lat)

for idx, (varname, var_idx, title) in enumerate(key_vars):
    ax = axes[idx // 3, idx % 3]
    error = forecast_6h[var_idx] - truth_6h[var_idx]

    vmax = np.percentile(np.abs(error), 98)
    if vmax == 0:
        vmax = 0.1

    levels_contour = np.linspace(-vmax, vmax, 21)
    im = ax.contourf(LON, LAT, error, levels=levels_contour,
                     cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Error (norm. units)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Lon (°)', fontsize=8)
    ax.set_ylabel('Lat (°)', fontsize=8)
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.2)

    # Add RMSE annotation
    rmse_val = metrics[varname]['rmse']
    acc_val = metrics[varname]['acc']
    ax.text(0.02, 0.98, f'RMSE={rmse_val:.3f}\nACC={acc_val:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('../report/images/fig4_error_maps.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved fig4_error_maps.png")

# =====================
# Figure 5: Skill Score Summary by Variable Group
# =====================
print("Generating Figure 5: Skill scores...")

pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

upper_air_groups = {
    'Z (Geopotential)': [f'Z{p}' for p in pressure_levels],
    'T (Temperature)': [f'T{p}' for p in pressure_levels],
    'U (U-wind)': [f'U{p}' for p in pressure_levels],
    'V (V-wind)': [f'V{p}' for p in pressure_levels],
    'R (Rel. Humidity)': [f'R{p}' for p in pressure_levels],
}

colors = {
    'Z (Geopotential)': 'navy',
    'T (Temperature)': 'firebrick',
    'U (U-wind)': 'forestgreen',
    'V (V-wind)': 'darkorange',
    'R (Rel. Humidity)': 'purple',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('FuXi 6h Forecast Skill by Pressure Level', fontsize=13, fontweight='bold')

# Panel 1: RMSE
ax = axes[0]
for group_name, var_list in upper_air_groups.items():
    vals = [metrics[v]['rmse'] for v in var_list if v in metrics]
    ax.plot(vals, pressure_levels, 'o-', color=colors[group_name],
            label=group_name, linewidth=2, markersize=6)
ax.set_ylabel('Pressure Level (hPa)', fontsize=11)
ax.set_xlabel('Weighted RMSE', fontsize=11)
ax.set_title('RMSE (area-weighted)', fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.set_yscale('log')
ax.set_yticks(pressure_levels)
ax.set_yticklabels(pressure_levels, fontsize=8)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, which='both')

# Panel 2: ACC
ax = axes[1]
for group_name, var_list in upper_air_groups.items():
    vals = [metrics[v]['acc'] for v in var_list if v in metrics]
    ax.plot(vals, pressure_levels, 'o-', color=colors[group_name],
            label=group_name, linewidth=2, markersize=6)
ax.axvline(x=0.6, color='k', linestyle='--', linewidth=1.5, alpha=0.7,
           label='ACC=0.6 threshold')
ax.set_ylabel('Pressure Level (hPa)', fontsize=11)
ax.set_xlabel('Anomaly Correlation (ACC)', fontsize=11)
ax.set_title('ACC (Anomaly Correlation)', fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.set_yscale('log')
ax.set_yticks(pressure_levels)
ax.set_yticklabels(pressure_levels, fontsize=8)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, which='both')

# Panel 3: Skill vs Persistence
ax = axes[2]
for group_name, var_list in upper_air_groups.items():
    vals = [metrics[v]['skill_vs_persistence'] for v in var_list if v in metrics]
    ax.plot(vals, pressure_levels, 'o-', color=colors[group_name],
            label=group_name, linewidth=2, markersize=6)
ax.axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Skill=0 (persistence)')
ax.set_ylabel('Pressure Level (hPa)', fontsize=11)
ax.set_xlabel('Skill Score vs. Persistence', fontsize=11)
ax.set_title('Skill Score vs. Persistence', fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.set_yscale('log')
ax.set_yticks(pressure_levels)
ax.set_yticklabels(pressure_levels, fontsize=8)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('../report/images/fig5_skill_scores.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved fig5_skill_scores.png")

# =====================
# Figure 6: Surface Variables Comparison
# =====================
print("Generating Figure 6: Surface variables bar chart...")

surface_vars = ['T2M', 'U10', 'V10', 'MSL', 'TP']
rmse_fcst = [metrics[v]['rmse'] for v in surface_vars]
rmse_pers = [metrics[v]['rmse_persistence'] for v in surface_vars]
skill = [metrics[v]['skill_vs_persistence'] for v in surface_vars]
acc = [metrics[v]['acc'] for v in surface_vars]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Surface Variable Forecast Performance (6h Lead Time)', fontsize=12, fontweight='bold')

x = np.arange(len(surface_vars))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, rmse_fcst, width, label='FuXi Forecast', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, rmse_pers, width, label='Persistence', color='tomato', alpha=0.85)
ax.set_xlabel('Variable', fontsize=11)
ax.set_ylabel('Weighted RMSE', fontsize=11)
ax.set_title('RMSE Comparison', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(surface_vars)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
colors_skill = ['forestgreen' if s > 0 else 'tomato' for s in skill]
ax.bar(surface_vars, skill, color=colors_skill, alpha=0.85)
ax.axhline(y=0, color='k', linewidth=1.5)
ax.set_xlabel('Variable', fontsize=11)
ax.set_ylabel('Skill Score', fontsize=11)
ax.set_title('Skill Score vs. Persistence', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[2]
ax.bar(surface_vars, acc, color='darkorange', alpha=0.85)
ax.axhline(y=0.6, color='k', linewidth=1.5, linestyle='--', label='ACC=0.6 threshold')
ax.set_xlabel('Variable', fontsize=11)
ax.set_ylabel('ACC', fontsize=11)
ax.set_title('Anomaly Correlation Coefficient', fontsize=10, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../report/images/fig6_surface_vars.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved fig6_surface_vars.png")

# Print summary statistics
print("\n=== Surface Variable Metrics ===")
print(f"{'Variable':<10} {'RMSE':<10} {'RMSE_pers':<12} {'Skill':<10} {'ACC':<10}")
for v in surface_vars:
    m = metrics[v]
    print(f"{v:<10} {m['rmse']:<10.4f} {m['rmse_persistence']:<12.4f} {m['skill_vs_persistence']:<10.4f} {m['acc']:<10.4f}")

print("\n=== Key Upper-Air Metrics ===")
key_upper = ['Z500', 'T850', 'U850', 'Z250', 'T500']
print(f"{'Variable':<10} {'RMSE':<10} {'RMSE_pers':<12} {'Skill':<10} {'ACC':<10}")
for v in key_upper:
    if v in metrics:
        m = metrics[v]
        print(f"{v:<10} {m['rmse']:<10.4f} {m['rmse_persistence']:<12.4f} {m['skill_vs_persistence']:<10.4f} {m['acc']:<10.4f}")

print("\nForecast verification complete!")
