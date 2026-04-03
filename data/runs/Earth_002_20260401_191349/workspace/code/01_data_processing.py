"""
Step 1: Data Processing
- Load mangrove locations (GMW v4)
- Compute TC frequency grid from historical MIT tracks
- Extract SLR rates (SSP2-4.5, SSP3-7.0, SSP5-8.5) for mangrove locations
- Save processed outputs
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

WS = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_002_20260401_191349'

# ─────────────────────────────────────────────────
# 1. Mangrove locations
# ─────────────────────────────────────────────────
print("Loading mangrove data...")
gdf = gpd.read_file(f'{WS}/data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
gdf['lon'] = gdf.geometry.x
gdf['lat'] = gdf.geometry.y
mangroves = gdf[['uid', 'lon', 'lat']].copy()
print(f"  Mangrove points: {len(mangroves)}")

# ─────────────────────────────────────────────────
# 2. TC frequency grid from historical tracks
# ─────────────────────────────────────────────────
print("Processing TC tracks...")
tc_ds = xr.open_dataset(f'{WS}/data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lat = tc_ds['lat'].values
tc_lon = tc_ds['lon'].values
tc_wind = tc_ds['wind'].values

# Normalize lon to [-180, 180]
tc_lon = np.where(tc_lon > 180, tc_lon - 360, tc_lon)

# Saffir-Simpson categories (wind in m/s)
# Cat 1: 33-42.5, Cat 2: 42.5-49.2, Cat 3: 49.2-58.1, Cat 4: 58.1-69.4, Cat 5: ≥69.4
# "Major" TCs = Cat 3+: wind >= 49.2 m/s
MAJOR_WIND_THRESH = 49.2   # m/s, Category 3+
ALL_TC_THRESH = 33.0       # m/s, all TCs (already filtered in file)

# Build 1° resolution frequency grid
GRID_RES = 1.0
lat_bins = np.arange(-40, 50 + GRID_RES, GRID_RES)
lon_bins = np.arange(-180, 181 + GRID_RES, GRID_RES)
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

# All TCs (wind >= 33 m/s)
all_mask = tc_wind >= ALL_TC_THRESH
H_all, _, _ = np.histogram2d(tc_lat[all_mask], tc_lon[all_mask],
                              bins=[lat_bins, lon_bins])

# Major TCs (Cat 3+, wind >= 49.2 m/s)
major_mask = tc_wind >= MAJOR_WIND_THRESH
H_major, _, _ = np.histogram2d(tc_lat[major_mask], tc_lon[major_mask],
                                bins=[lat_bins, lon_bins])

# Historical period: 1850-2014 = 165 years
# The dataset was downsampled (max_points=200000), so we need to account for that
# Total points recorded: 200000; dataset has 200000 points
# Scale factor to get annual frequency density
N_YEARS = 165.0  # 1850-2014

# Annual frequency per grid cell
freq_all = H_all / N_YEARS
freq_major = H_major / N_YEARS

print(f"  TC grid shape: {H_all.shape}")
print(f"  Max annual frequency (all TCs): {freq_all.max():.2f}")
print(f"  Max annual frequency (major TCs): {freq_major.max():.2f}")

# Save TC grids
np.savez(f'{WS}/outputs/tc_frequency_grid.npz',
         lat_centers=lat_centers, lon_centers=lon_centers,
         freq_all=freq_all, freq_major=freq_major,
         lat_bins=lat_bins, lon_bins=lon_bins)
print("  Saved TC frequency grid.")

# ─────────────────────────────────────────────────
# 3. Extract SLR rates at mangrove locations
# ─────────────────────────────────────────────────
print("Extracting SLR rates...")

SCENARIOS = {
    'ssp245': f'{WS}/data/slr/total_ssp245_medium_confidence_rates.nc',
    'ssp370': f'{WS}/data/slr/total_ssp370_medium_confidence_rates.nc',
    'ssp585': f'{WS}/data/slr/total_ssp585_medium_confidence_rates.nc',
}

slr_results = {}

for ssp, fpath in SCENARIOS.items():
    print(f"  Processing {ssp}...")
    ds = xr.open_dataset(fpath)
    slr_lat = ds['lat'].values
    slr_lon = ds['lon'].values
    rates = ds['sea_level_change_rate']  # (quantiles, years, locations)

    # Get years 2020-2100
    years = ds['years'].values
    yr_mask = (years >= 2020) & (years <= 2100)

    # Get median quantile (0.5)
    quantiles = ds['quantiles'].values
    q50_idx = np.argmin(np.abs(quantiles - 0.5))
    q17_idx = np.argmin(np.abs(quantiles - 0.17))  # 17th percentile
    q83_idx = np.argmin(np.abs(quantiles - 0.83))  # 83rd percentile

    print(f"    Median quantile index: {q50_idx}, value: {quantiles[q50_idx]}")

    # Extract rates for 2020-2100 at median, 17th, 83rd quantiles
    # Average over years to get mean rate 2020-2100
    med_rates = rates.isel(quantiles=q50_idx, years=np.where(yr_mask)[0]).mean(dim='years').values
    lo_rates  = rates.isel(quantiles=q17_idx,  years=np.where(yr_mask)[0]).mean(dim='years').values
    hi_rates  = rates.isel(quantiles=q83_idx,  years=np.where(yr_mask)[0]).mean(dim='years').values

    print(f"    Median SLR rate range: {med_rates.min():.1f} to {med_rates.max():.1f} mm/yr")

    # Build KDTree for nearest-neighbor interpolation to mangrove points
    # Filter to valid lat/lon (non-NaN, relevant lat range)
    valid = np.isfinite(slr_lat) & np.isfinite(slr_lon) & np.isfinite(med_rates)
    tree = cKDTree(np.column_stack([slr_lat[valid], slr_lon[valid]]))

    mg_coords = np.column_stack([mangroves['lat'].values, mangroves['lon'].values])
    dist, idx = tree.query(mg_coords, k=1)

    slr_results[ssp] = {
        'median': med_rates[valid][idx],
        'lo':     lo_rates[valid][idx],
        'hi':     hi_rates[valid][idx],
        'dist':   dist,
    }

    ds.close()
    print(f"    Done. Mean SLR at mangroves: {slr_results[ssp]['median'].mean():.2f} mm/yr")

# ─────────────────────────────────────────────────
# 4. Extract TC frequency at mangrove locations
# ─────────────────────────────────────────────────
print("Extracting TC frequency at mangrove locations...")

# Use KDTree on grid centers
lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing='ij')
grid_coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
tc_tree = cKDTree(grid_coords)

mg_coords = np.column_stack([mangroves['lat'].values, mangroves['lon'].values])
_, tc_idx = tc_tree.query(mg_coords, k=1)

tc_freq_all_mg   = freq_all.ravel()[tc_idx]
tc_freq_major_mg = freq_major.ravel()[tc_idx]

print(f"  TC freq all range: {tc_freq_all_mg.min():.3f} to {tc_freq_all_mg.max():.3f}")
print(f"  TC freq major range: {tc_freq_major_mg.min():.3f} to {tc_freq_major_mg.max():.3f}")

# ─────────────────────────────────────────────────
# 5. Assemble master DataFrame
# ─────────────────────────────────────────────────
print("Assembling master DataFrame...")

df = mangroves.copy()
df['tc_freq_all']   = tc_freq_all_mg
df['tc_freq_major'] = tc_freq_major_mg

for ssp in SCENARIOS:
    df[f'slr_median_{ssp}'] = slr_results[ssp]['median']
    df[f'slr_lo_{ssp}']     = slr_results[ssp]['lo']
    df[f'slr_hi_{ssp}']     = slr_results[ssp]['hi']

df.to_csv(f'{WS}/outputs/mangrove_risk_data.csv', index=False)
print(f"  Saved master DataFrame: {df.shape}")
print(df.describe())
