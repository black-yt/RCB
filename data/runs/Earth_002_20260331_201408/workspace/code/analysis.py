
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path('.')

mg_path = BASE/'data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg'
slr_base = BASE/'data/slr'
tc_path = BASE/'data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc'
outputs_dir = BASE/'outputs'
fig_dir = BASE/'report/images'
outputs_dir.mkdir(exist_ok=True)
fig_dir.mkdir(exist_ok=True)

print('Loading mangrove points...')
mg = gpd.read_file(mg_path)
mg = mg.to_crs(4326)
mg['lon'] = mg.geometry.x
mg['lat'] = mg.geometry.y

print('Loading TC tracks...')
tc = xr.open_dataset(tc_path)
tc_df = tc[['lat','lon','wind']].to_dataframe().reset_index()
tc_df = tc_df[(tc_df['wind']>=33)]
tc_df['lat_bin'] = tc_df['lat'].round(1)
tc_df['lon_bin'] = tc_df['lon'].round(1)

print('Computing TC frequency map...')
# Approximate storms per grid cell: count normalized by years (1850-2014 ~165y)
years = 2014-1850+1
freq = tc_df.groupby(['lat_bin','lon_bin']).size().rename('count').reset_index()
freq['tc_per_yr'] = freq['count']/years

# Plot global TC frequency heatmap
plt.figure(figsize=(10,4))
plt.scatter(freq['lon_bin'], freq['lat_bin'], c=freq['tc_per_yr'], s=4, cmap='magma', vmin=0)
plt.colorbar(label='TC points per year (>=33 m/s)')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Historical intense TC occurrence (MPI-ESM1-2-HR)')
plt.tight_layout(); plt.savefig(fig_dir/'tc_frequency_map.png', dpi=300); plt.close()

print('Extracting median SLR rates 2020-2100...')
slr_summaries = []
for ssp in ['245','370','585']:
    path = slr_base/f'total_ssp{ssp}_medium_confidence_rates.nc'
    ds = xr.open_dataset(path)
    # select 2020-2100 subset
    ds_sub = ds.sel(years=slice(2020,2100))
    # median over quantiles and years
    rate_med = ds_sub['sea_level_change_rate'].median(dim=('quantiles','years'))
    slr_df = rate_med.to_dataframe(name='slr_rate_mm_per_yr').reset_index()
    # attach lat/lon from locations dimension
    loc = ds_sub['locations']
    lat = ds_sub['lat']
    lon = ds_sub['lon']
    lat_lon = lat.to_dataframe(name='lat').merge(lon.to_dataframe(name='lon'), left_index=True, right_index=True)
    lat_lon = lat_lon.reset_index()
    slr_df = slr_df.merge(lat_lon, on='locations', how='left')
    slr_df = slr_df[['lat','lon','slr_rate_mm_per_yr']]
    slr_df['ssp'] = ssp
    slr_summaries.append(slr_df)

slr_all = pd.concat(slr_summaries, ignore_index=True)
slr_all.to_parquet(outputs_dir/'slr_rates_2020_2100_median.parquet')

# quick global SLR map for SSP585
slr585 = slr_all[slr_all['ssp']=='585']
plt.figure(figsize=(10,4))
plt.scatter(slr585['lon'], slr585['lat'], c=slr585['slr_rate_mm_per_yr'], s=2, cmap='viridis')
plt.colorbar(label='Median SLR rate 2020-2100 (mm/yr)')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Relative sea-level rise rate (SSP5-8.5, 2020-2100 median)')
plt.tight_layout(); plt.savefig(fig_dir/'slr_rate_ssp585_map.png', dpi=300); plt.close()

print('Linking mangroves to nearest SLR grid point (SSP585)...')
# simple nearest-neighbour via rounding to 0.25 deg (approx spacing)
mg['lat_round'] = (mg['lat']/0.25).round()*0.25
mg['lon_round'] = (mg['lon']/0.25).round()*0.25
slr585['lat_round'] = (slr585['lat']/0.25).round()*0.25
slr585['lon_round'] = (slr585['lon']/0.25).round()*0.25

slr585_small = slr585[['lat_round','lon_round','slr_rate_mm_per_yr']].drop_duplicates()
mg_slr = mg.merge(slr585_small, on=['lat_round','lon_round'], how='left')

print('Linking mangroves to TC frequency via 1-degree bins...')
mg['lat_bin'] = mg['lat'].round(1)
mg['lon_bin'] = mg['lon'].round(1)
mg_tc = mg.merge(freq[['lat_bin','lon_bin','tc_per_yr']], on=['lat_bin','lon_bin'], how='left')

# combine
mg_risk = mg_tc.merge(mg_slr[['uid','slr_rate_mm_per_yr']], on='uid', how='left', suffixes=('','_slr'))

# normalise indicators across mangroves
for col in ['tc_per_yr','slr_rate_mm_per_yr']:
    mg_risk[col] = mg_risk[col].fillna(0)
    if mg_risk[col].max()>0:
        mg_risk[col+'_norm'] = mg_risk[col]/mg_risk[col].max()
    else:
        mg_risk[col+'_norm'] = 0

# composite index with equal weights
mg_risk['risk_index'] = 0.5*mg_risk['tc_per_yr_norm'] + 0.5*mg_risk['slr_rate_mm_per_yr_norm']

mg_risk.to_file(outputs_dir/'mangrove_tc_slr_risk.gpkg', driver='GPKG')

print('Plotting mangrove risk index map...')
plt.figure(figsize=(10,4))
sc = plt.scatter(mg_risk['lon'], mg_risk['lat'], c=mg_risk['risk_index'], s=4, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(sc, label='Composite risk index')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Composite risk index for sampled mangroves (SSP5-8.5, 2100)')
plt.tight_layout(); plt.savefig(fig_dir/'mangrove_risk_index_map.png', dpi=300); plt.close()

print('Generating validation plots...')

plt.figure(figsize=(6,4))
sns.histplot(mg_risk['tc_per_yr'], bins=40)
plt.xlabel('TC occurrences per year (grid cell)'); plt.ylabel('Count'); plt.title('Distribution of TC exposure across mangroves')
plt.tight_layout(); plt.savefig(fig_dir/'hist_tc_exposure.png', dpi=300); plt.close()

plt.figure(figsize=(6,4))
sns.histplot(mg_risk['slr_rate_mm_per_yr'], bins=40)
plt.xlabel('SLR rate 2020-2100 (mm/yr)'); plt.ylabel('Count'); plt.title('Distribution of SLR exposure across mangroves (SSP5-8.5)')
plt.tight_layout(); plt.savefig(fig_dir/'hist_slr_exposure.png', dpi=300); plt.close()

plt.figure(figsize=(6,4))
sns.kdeplot(data=mg_risk, x='slr_rate_mm_per_yr', y='tc_per_yr', fill=True, thresh=0.05)
plt.xlabel('SLR rate (mm/yr)'); plt.ylabel('TC occurrences per year'); plt.title('Joint distribution of SLR and TC exposure at mangrove locations')
plt.tight_layout(); plt.savefig(fig_dir/'joint_slr_tc_exposure.png', dpi=300); plt.close()

print('Analysis complete.')
