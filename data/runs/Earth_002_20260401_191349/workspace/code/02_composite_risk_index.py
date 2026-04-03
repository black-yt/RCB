"""
Step 2: Composite Risk Index Construction
- Normalize TC frequency and SLR rates
- Build composite risk index for each SSP scenario
- Apply thresholds from literature (SLR: 4 mm/yr moderate, 7 mm/yr high)
- Add ecosystem service component from country bounds
- Classify risk levels
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings('ignore')

WS = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_002_20260401_191349'

# ─────────────────────────────────────────────────
# Load processed data
# ─────────────────────────────────────────────────
print("Loading processed data...")
df = pd.read_csv(f'{WS}/outputs/mangrove_risk_data.csv')
print(f"  Loaded {len(df)} mangrove locations")

# ─────────────────────────────────────────────────
# 1. Normalize SLR component
# ─────────────────────────────────────────────────
# Based on Saintilan et al. (2023):
#   - SLR < 4 mm/yr: low risk
#   - SLR 4-7 mm/yr: moderate risk (likely elevation deficit)
#   - SLR > 7 mm/yr: high risk (very likely elevation deficit)
# Normalize to 0-1 using min-max scaled from 0-15 mm/yr (practical range for mangroves)

SLR_MIN = 0.0   # mm/yr (zero or negative = accretion possible)
SLR_MAX = 15.0  # mm/yr (upper realistic bound for high-emissions mangrove zones)

def normalize_slr(rates, slr_min=SLR_MIN, slr_max=SLR_MAX):
    """Normalize SLR rates to 0-1 risk score. Negative rates = 0 risk."""
    clipped = np.clip(rates, slr_min, slr_max)
    return (clipped - slr_min) / (slr_max - slr_min)

for ssp in ['ssp245', 'ssp370', 'ssp585']:
    col = f'slr_median_{ssp}'
    df[f'slr_risk_{ssp}'] = normalize_slr(df[col].values)

print("  SLR risk scores computed.")
print(df[['slr_risk_ssp245','slr_risk_ssp370','slr_risk_ssp585']].describe())

# ─────────────────────────────────────────────────
# 2. Normalize TC frequency component
# ─────────────────────────────────────────────────
# Use rank-based normalization (percentile) for TC frequency
# to handle heavy-tailed distribution

def normalize_tc_rank(freq):
    """Rank-based normalization to 0-1."""
    ranks = pd.Series(freq).rank(pct=True).values
    return ranks

df['tc_risk_all']   = normalize_tc_rank(df['tc_freq_all'].values)
df['tc_risk_major'] = normalize_tc_rank(df['tc_freq_major'].values)

print("\n  TC risk scores computed.")
print(df[['tc_risk_all','tc_risk_major']].describe())

# ─────────────────────────────────────────────────
# 3. Composite Risk Index (CRI)
# ─────────────────────────────────────────────────
# CRI = w_SLR * SLR_risk + w_TC * TC_risk
# Equal weighting: w_SLR = w_TC = 0.5
# Using major TC risk (Cat 3+, most damaging per Mo et al. 2023)

W_SLR = 0.5
W_TC  = 0.5

for ssp in ['ssp245', 'ssp370', 'ssp585']:
    cri_col = f'cri_{ssp}'
    df[cri_col] = W_SLR * df[f'slr_risk_{ssp}'] + W_TC * df['tc_risk_major']

# Also compute TC-only and SLR-only risk for sensitivity
df['cri_tc_only']  = df['tc_risk_major']
df['cri_slr_only_ssp585'] = df['slr_risk_ssp585']

print("\n  CRI computed for all scenarios.")
print(df[['cri_ssp245','cri_ssp370','cri_ssp585']].describe())

# ─────────────────────────────────────────────────
# 4. Risk Classification
# ─────────────────────────────────────────────────
# Quintile-based risk classification:
# 0.0-0.2: Very Low; 0.2-0.4: Low; 0.4-0.6: Moderate; 0.6-0.8: High; 0.8-1.0: Very High

def classify_risk(cri):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    return pd.cut(cri, bins=bins, labels=labels, right=True, include_lowest=True)

for ssp in ['ssp245', 'ssp370', 'ssp585']:
    df[f'risk_class_{ssp}'] = classify_risk(df[f'cri_{ssp}'])

print("\n  Risk classification for SSP5-8.5:")
print(df['risk_class_ssp585'].value_counts())

# ─────────────────────────────────────────────────
# 5. Add ecosystem service context from country bounds
# ─────────────────────────────────────────────────
print("\nJoining ecosystem service data...")
eco = gpd.read_file(f'{WS}/data/ecosystem/UCSC_CWON_countrybounds.gpkg')
eco_df = eco[['Country', 'ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020',
              'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020', 'geometry']].copy()

# Convert mangroves to GeoDataFrame for spatial join
import geopandas as gpd
from shapely.geometry import Point

gdf_mg = gpd.GeoDataFrame(df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs='EPSG:4326')

# Spatial join with country bounds
gdf_joined = gpd.sjoin(gdf_mg, eco_df[['ISO3','Country','Mang_Ha_2020',
                                         'Risk_Pop_2020','Risk_Stock_2020',
                                         'Ben_Pop_2020','Ben_Stock_2020','geometry']],
                       how='left', predicate='within')

print(f"  Joined: {len(gdf_joined)} records")
print(f"  Points with country match: {gdf_joined['ISO3'].notna().sum()}")

# Fill unmatched with nearest country using centroids
unmatched = gdf_joined['ISO3'].isna()
if unmatched.sum() > 0:
    print(f"  Unmatched points: {unmatched.sum()} - filling with nearest country")
    eco_centroids = eco_df.copy()
    eco_centroids['geometry'] = eco_df.geometry.centroid

    from scipy.spatial import cKDTree
    eco_coords = np.column_stack([
        eco_centroids.geometry.y.values,
        eco_centroids.geometry.x.values
    ])
    eco_tree = cKDTree(eco_coords)

    unmatched_coords = np.column_stack([
        gdf_joined.loc[unmatched, 'lat'].values,
        gdf_joined.loc[unmatched, 'lon'].values
    ])
    _, near_idx = eco_tree.query(unmatched_coords, k=1)

    for col in ['ISO3','Country','Mang_Ha_2020','Risk_Pop_2020',
                'Risk_Stock_2020','Ben_Pop_2020','Ben_Stock_2020']:
        gdf_joined.loc[unmatched, col] = eco_df.iloc[near_idx][col].values

print(f"  Final unmatched: {gdf_joined['ISO3'].isna().sum()}")

# Drop duplicate geometry columns
cols_keep = [c for c in gdf_joined.columns if c not in ['geometry', 'index_right']]
df_final = gdf_joined[cols_keep].copy()

# ─────────────────────────────────────────────────
# 6. Save final dataset
# ─────────────────────────────────────────────────
df_final.to_csv(f'{WS}/outputs/mangrove_composite_risk.csv', index=False)
print(f"\nSaved final dataset: {df_final.shape}")

# Country-level summary
country_risk = df_final.groupby(['ISO3','Country']).agg(
    n_points=('uid','count'),
    mang_ha=('Mang_Ha_2020','first'),
    risk_pop=('Risk_Pop_2020','first'),
    risk_stock=('Risk_Stock_2020','first'),
    ben_pop=('Ben_Pop_2020','first'),
    ben_stock=('Ben_Stock_2020','first'),
    cri_ssp245_mean=('cri_ssp245','mean'),
    cri_ssp370_mean=('cri_ssp370','mean'),
    cri_ssp585_mean=('cri_ssp585','mean'),
    slr_median_ssp585=('slr_median_ssp585','mean'),
    tc_freq_major=('tc_freq_major','mean'),
).reset_index()

country_risk['cri_ssp585_pct'] = (country_risk['cri_ssp585_mean']
    .rank(pct=True) * 100).round(1)

country_risk.to_csv(f'{WS}/outputs/country_risk_summary.csv', index=False)
print(f"Saved country risk summary: {country_risk.shape}")
print("\nTop 20 countries by CRI (SSP5-8.5):")
print(country_risk.nlargest(20, 'cri_ssp585_mean')[
    ['Country','mang_ha','cri_ssp585_mean','slr_median_ssp585','tc_freq_major','risk_pop','ben_stock']
].to_string())
