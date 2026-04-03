"""
Step 3: Generate Figures
- Fig 1: Global TC frequency map
- Fig 2: Global SLR rate map (SSP5-8.5)
- Fig 3: Global composite risk index map
- Fig 4: Risk component scatter (SLR vs TC)
- Fig 5: Country bar chart - top at-risk countries
- Fig 6: Risk class distribution by ocean basin
- Fig 7: Scenario comparison (SSP2-4.5 vs SSP3-7.0 vs SSP5-8.5)
- Fig 8: Ecosystem services at risk
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

WS = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_002_20260401_191349'
IMGDIR = f'{WS}/report/images'

# ─────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────
df = pd.read_csv(f'{WS}/outputs/mangrove_composite_risk.csv')
country_risk = pd.read_csv(f'{WS}/outputs/country_risk_summary.csv')
tc_grid = np.load(f'{WS}/outputs/tc_frequency_grid.npz')

lat_centers = tc_grid['lat_centers']
lon_centers = tc_grid['lon_centers']
freq_major  = tc_grid['freq_major']

print(f"Loaded {len(df)} mangrove points, {len(country_risk)} countries")

# ─────────────────────────────────────────────────
# Helper: Simple world map background
# ─────────────────────────────────────────────────
def world_background(ax):
    """Draw simple country outlines using coastline data."""
    import geopandas as gpd
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.plot(ax=ax, color='#e8e8e8', edgecolor='#aaaaaa', linewidth=0.3, zorder=0)
    except Exception:
        ax.set_facecolor('#d4e8f0')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-35, 50)

# Custom colormap for risk
risk_colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']
risk_cmap = LinearSegmentedColormap.from_list('risk', risk_colors, N=256)

# ─────────────────────────────────────────────────
# FIGURE 1: TC Frequency Map (Major TCs, Cat 3+)
# ─────────────────────────────────────────────────
print("Generating Fig 1: TC frequency map...")
fig, ax = plt.subplots(figsize=(14, 6))
world_background(ax)

# Plot TC frequency grid as heatmap
lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
freq_plot = np.ma.masked_where(freq_major == 0, freq_major)
pcm = ax.pcolormesh(lon_centers, lat_centers, freq_plot,
                    cmap='YlOrRd', vmin=0, vmax=0.15,
                    alpha=0.75, zorder=1)
cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Annual TC frequency\n(Category 3+, tracks per grid cell)', fontsize=10)

# Overlay mangrove locations
ax.scatter(df['lon'], df['lat'], s=0.3, c='#006400', alpha=0.4, zorder=2,
           label='Mangroves (GMW v4)')

ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('Historical Tropical Cyclone Frequency (Category 3+) and Global Mangrove Distribution\n'
             'MIT model, MPI-ESM1-2-HR historical (1850–2014)', fontsize=12, fontweight='bold')
ax.legend(loc='lower left', markerscale=8, fontsize=9)
ax.grid(True, alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig1_tc_frequency_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1")

# ─────────────────────────────────────────────────
# FIGURE 2: SLR Rate Map at Mangrove Locations
# ─────────────────────────────────────────────────
print("Generating Fig 2: SLR rate maps...")
fig, axes = plt.subplots(3, 1, figsize=(14, 16))

scenarios = [('ssp245', 'SSP2-4.5'), ('ssp370', 'SSP3-7.0'), ('ssp585', 'SSP5-8.5')]
vmin, vmax = 2, 14

for ax, (ssp, label) in zip(axes, scenarios):
    world_background(ax)
    sc = ax.scatter(df['lon'], df['lat'],
                    c=df[f'slr_median_{ssp}'],
                    s=0.5, cmap='RdYlBu_r',
                    vmin=vmin, vmax=vmax, alpha=0.8, zorder=2)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Median SLR rate (mm yr⁻¹)\n2020–2100', fontsize=9)
    ax.axhline(y=0, color='k', linewidth=0.3, linestyle=':')
    ax.set_title(f'{label} — Median Relative Sea Level Rise Rate at Mangrove Locations',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    # Add threshold lines in colorbar
    for thresh, col, lbl in [(4, 'orange', '4 mm/yr'), (7, 'red', '7 mm/yr')]:
        cbar.ax.axhline(y=(thresh - vmin)/(vmax - vmin), color=col, linewidth=2, linestyle='--')

plt.suptitle('Projected Relative Sea Level Rise Rates at Global Mangrove Locations\n'
             '(IPCC AR6, medium confidence, median, 2020–2100 mean)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig2_slr_rates_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2")

# ─────────────────────────────────────────────────
# FIGURE 3: Composite Risk Index Map (all 3 scenarios)
# ─────────────────────────────────────────────────
print("Generating Fig 3: CRI maps...")
fig, axes = plt.subplots(3, 1, figsize=(14, 16))

for ax, (ssp, label) in zip(axes, scenarios):
    world_background(ax)
    sc = ax.scatter(df['lon'], df['lat'],
                    c=df[f'cri_{ssp}'],
                    s=0.5, cmap=risk_cmap,
                    vmin=0.1, vmax=1.0, alpha=0.85, zorder=2)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Composite Risk Index\n(0 = Low, 1 = High)', fontsize=9)
    ax.set_title(f'{label} — Composite Risk Index (CRI) for Global Mangroves',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

plt.suptitle('Global Composite Risk Index (CRI) Combining Sea Level Rise and\n'
             'Tropical Cyclone Regime Shifts for Mangrove Ecosystems',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig3_cri_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3")

# ─────────────────────────────────────────────────
# FIGURE 4: SLR vs TC Risk Scatter + Quadrant Analysis
# ─────────────────────────────────────────────────
print("Generating Fig 4: Risk component scatter...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (ssp, label) in zip(axes, scenarios):
    # Subsample for visualization
    idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
    sub = df.iloc[idx]

    sc = ax.scatter(sub['tc_risk_major'], sub[f'slr_risk_{ssp}'],
                    c=sub[f'cri_{ssp}'], cmap=risk_cmap,
                    s=2, alpha=0.5, vmin=0.1, vmax=1.0)

    # Quadrant lines (0.5 threshold)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Label quadrants
    ax.text(0.25, 0.95, 'SLR\ndominant', transform=ax.transAxes,
            ha='center', va='top', fontsize=8, color='gray', style='italic')
    ax.text(0.75, 0.95, 'Dual\nhigh risk', transform=ax.transAxes,
            ha='center', va='top', fontsize=8, color='darkred', style='italic')
    ax.text(0.25, 0.05, 'Low risk', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8, color='gray', style='italic')
    ax.text(0.75, 0.05, 'TC\ndominant', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8, color='gray', style='italic')

    ax.set_xlabel('TC Risk Score (Cat 3+, normalized)', fontsize=10)
    ax.set_ylabel('SLR Risk Score (normalized)', fontsize=10)
    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='CRI')

plt.suptitle('Risk Component Space: Sea Level Rise vs. Tropical Cyclone Risk\n'
             'Each point = one mangrove sample location', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig4_risk_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4")

# ─────────────────────────────────────────────────
# FIGURE 5: Top 25 countries by CRI (SSP5-8.5)
# ─────────────────────────────────────────────────
print("Generating Fig 5: Country bar chart...")

top25 = country_risk.nlargest(25, 'cri_ssp585_mean').copy()
top25 = top25.sort_values('cri_ssp585_mean', ascending=True)

# Color bars by risk level
colors = [risk_cmap((v - 0.1) / 0.9) for v in top25['cri_ssp585_mean']]

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(top25['Country'], top25['cri_ssp585_mean'], color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(x=0.7, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='High risk threshold (0.7)')
ax.axvline(x=0.5, color='orange', linewidth=1.5, linestyle='--', alpha=0.7, label='Moderate risk threshold (0.5)')
ax.set_xlabel('Mean Composite Risk Index (SSP5-8.5)', fontsize=11)
ax.set_title('Top 25 Countries with Highest Mean Composite Risk Index\n'
             'for Mangrove Ecosystems (SSP5-8.5, end-of-century)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0.3, 1.0)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add mangrove area annotation
for i, (_, row) in enumerate(top25.iterrows()):
    ha = int(row['mang_ha']) if pd.notna(row['mang_ha']) else 0
    ax.text(row['cri_ssp585_mean'] + 0.005, i, f'{ha/1000:.0f}k ha',
            va='center', fontsize=7, color='#333333')

plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig5_country_cri_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5")

# ─────────────────────────────────────────────────
# FIGURE 6: Risk class distribution by ocean basin
# ─────────────────────────────────────────────────
print("Generating Fig 6: Basin risk distribution...")

# Assign ocean basins based on longitude/latitude
def assign_basin(lon, lat):
    if lon < -20:
        if lat >= 0:
            return 'N. Atlantic'
        else:
            return 'S. Atlantic'
    elif lon < 70:
        if lat >= 0:
            return 'N. Indian'
        else:
            return 'S. Indian'
    elif lon < 140:
        if lat >= 0:
            return 'NW Pacific'
        else:
            return 'SW Pacific'
    else:
        if lat >= 0:
            return 'NE Pacific'
        else:
            return 'SE Pacific / Oceania'

df['basin'] = [assign_basin(r.lon, r.lat) for _, r in df[['lon','lat']].iterrows()]

# Risk class distribution per basin (SSP5-8.5)
risk_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
risk_pal = {'Very Low': '#2b83ba', 'Low': '#abdda4', 'Moderate': '#ffffbf',
            'High': '#fdae61', 'Very High': '#d7191c'}

basin_counts = df.groupby(['basin', 'risk_class_ssp585']).size().unstack(fill_value=0)
# Keep only risk classes present
risk_cols = [r for r in risk_order if r in basin_counts.columns]
basin_counts = basin_counts[risk_cols]
basin_pct = basin_counts.div(basin_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(basin_pct))
for r in risk_cols:
    vals = basin_pct[r].values
    bars = ax.bar(basin_pct.index, vals, bottom=bottom,
                  color=risk_pal.get(r, '#aaaaaa'), label=r, edgecolor='white', linewidth=0.5)
    bottom += vals

ax.set_xlabel('Ocean Basin', fontsize=11)
ax.set_ylabel('Percentage of Mangrove Points (%)', fontsize=11)
ax.set_title('Composite Risk Index Distribution by Ocean Basin\n'
             '(SSP5-8.5 scenario, mangrove sample points)', fontsize=12, fontweight='bold')
ax.legend(title='Risk Class', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig6_basin_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6")

# ─────────────────────────────────────────────────
# FIGURE 7: Scenario comparison violin plot
# ─────────────────────────────────────────────────
print("Generating Fig 7: Scenario comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: CRI distributions by scenario
ax = axes[0]
cri_data = [df['cri_ssp245'].values, df['cri_ssp370'].values, df['cri_ssp585'].values]
labels = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
colors_scen = ['#2171b5', '#fd8d3c', '#cb181d']

parts = ax.violinplot(cri_data, positions=[1, 2, 3], showmedians=True, showextrema=True)
for pc, c in zip(parts['bodies'], colors_scen):
    pc.set_facecolor(c)
    pc.set_alpha(0.7)
parts['cmedians'].set_colors('black')
parts['cmaxes'].set_colors('gray')
parts['cmins'].set_colors('gray')
parts['cbars'].set_colors('gray')

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Composite Risk Index', fontsize=11)
ax.set_title('CRI Distribution by Emission Scenario', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.6, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='High risk (0.6)')
ax.legend(fontsize=9)

# Right: Fraction of mangroves in each risk class by scenario
ax = axes[1]
scenario_labels = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
ssps = ['ssp245', 'ssp370', 'ssp585']
risk_fracs = {}
for ssp in ssps:
    vc = df[f'risk_class_{ssp}'].value_counts(normalize=True) * 100
    risk_fracs[ssp] = {r: vc.get(r, 0) for r in risk_order}

x = np.arange(len(scenario_labels))
width = 0.15
for i, r in enumerate(risk_order):
    vals = [risk_fracs[ssp][r] for ssp in ssps]
    ax.bar(x + i * width, vals, width, label=r,
           color=risk_pal.get(r, '#aaaaaa'), edgecolor='white', linewidth=0.5)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(scenario_labels, fontsize=11)
ax.set_ylabel('Percentage of Mangrove Points (%)', fontsize=11)
ax.set_title('Risk Class Distribution by Emission Scenario', fontsize=12, fontweight='bold')
ax.legend(title='Risk Class', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Comparison of Composite Risk Index Across SSP Emission Scenarios\n'
             'Equal-weight combination of SLR and TC hazard components',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig7_scenario_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7")

# ─────────────────────────────────────────────────
# FIGURE 8: Ecosystem services at risk
# ─────────────────────────────────────────────────
print("Generating Fig 8: Ecosystem services at risk...")

# Filter to countries with complete data
eco_risk = country_risk.dropna(subset=['risk_pop', 'ben_stock', 'mang_ha']).copy()
eco_risk = eco_risk[eco_risk['mang_ha'] > 1000].copy()  # countries with >1000 ha

# Sort by CRI
eco_risk = eco_risk.sort_values('cri_ssp585_mean', ascending=False).head(30)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Panel a: Mangrove area by CRI
ax = axes[0]
colors_cri = [risk_cmap((v - 0.1) / 0.9) for v in eco_risk['cri_ssp585_mean']]
sc = ax.scatter(eco_risk['cri_ssp585_mean'], eco_risk['mang_ha'] / 1000,
                c=eco_risk['cri_ssp585_mean'], cmap=risk_cmap,
                s=np.sqrt(eco_risk['mang_ha'] / 100) * 5,
                vmin=0.1, vmax=1.0, alpha=0.8, edgecolors='gray', linewidth=0.5)
for _, row in eco_risk.iterrows():
    if row['mang_ha'] > 50000 or row['cri_ssp585_mean'] > 0.75:
        ax.annotate(row['Country'], (row['cri_ssp585_mean'], row['mang_ha']/1000),
                    fontsize=6.5, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('CRI (SSP5-8.5)', fontsize=11)
ax.set_ylabel('Mangrove Area (thousands ha)', fontsize=11)
ax.set_title('Mangrove Area at Risk', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Panel b: Population at risk vs CRI
ax = axes[1]
valid_pop = eco_risk[eco_risk['risk_pop'] > 0]
sc2 = ax.scatter(valid_pop['cri_ssp585_mean'], valid_pop['risk_pop'] / 1000,
                 c=valid_pop['cri_ssp585_mean'], cmap=risk_cmap,
                 s=np.sqrt(valid_pop['risk_pop'] / 50) * 5,
                 vmin=0.1, vmax=1.0, alpha=0.8, edgecolors='gray', linewidth=0.5)
for _, row in valid_pop.iterrows():
    if row['risk_pop'] > 50000 or row['cri_ssp585_mean'] > 0.75:
        ax.annotate(row['Country'], (row['cri_ssp585_mean'], row['risk_pop']/1000),
                    fontsize=6.5, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('CRI (SSP5-8.5)', fontsize=11)
ax.set_ylabel('Coastal Population at Risk (thousands)', fontsize=11)
ax.set_title('Population at Risk', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Panel c: Economic stock at risk vs CRI
ax = axes[2]
valid_stock = eco_risk[eco_risk['ben_stock'] > 0]
sc3 = ax.scatter(valid_stock['cri_ssp585_mean'], valid_stock['ben_stock'] / 1e9,
                 c=valid_stock['cri_ssp585_mean'], cmap=risk_cmap,
                 s=np.sqrt(valid_stock['ben_stock'] / 1e8) * 5,
                 vmin=0.1, vmax=1.0, alpha=0.8, edgecolors='gray', linewidth=0.5)
for _, row in valid_stock.iterrows():
    if row['ben_stock'] > 1e10 or row['cri_ssp585_mean'] > 0.75:
        ax.annotate(row['Country'], (row['cri_ssp585_mean'], row['ben_stock']/1e9),
                    fontsize=6.5, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('CRI (SSP5-8.5)', fontsize=11)
ax.set_ylabel('Coastal Asset Value at Risk (billion USD)', fontsize=11)
ax.set_title('Coastal Asset Value at Risk', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

plt.colorbar(sc3, ax=axes[2], label='CRI (SSP5-8.5)')
plt.suptitle('Ecosystem Services and Socioeconomic Values at Risk\n'
             'Countries with >1,000 ha of Mangroves, ranked by CRI (SSP5-8.5)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig8_ecosystem_services_risk.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig8")

# ─────────────────────────────────────────────────
# FIGURE 9: Data overview - histogram of risk components
# ─────────────────────────────────────────────────
print("Generating Fig 9: Data overview histograms...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: SLR distributions
for ax, (ssp, label) in zip(axes[0], scenarios):
    vals = df[f'slr_median_{ssp}'].values
    ax.hist(vals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=4, color='orange', linewidth=2, linestyle='--', label='4 mm/yr')
    ax.axvline(x=7, color='red', linewidth=2, linestyle='--', label='7 mm/yr')
    ax.set_xlabel('Median SLR Rate (mm yr⁻¹)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'SLR Distribution — {label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # Percentage above thresholds
    pct4 = (vals > 4).sum() / len(vals) * 100
    pct7 = (vals > 7).sum() / len(vals) * 100
    ax.text(0.98, 0.97, f'>4 mm/yr: {pct4:.0f}%\n>7 mm/yr: {pct7:.0f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Row 2: TC frequency, and CRI
ax = axes[1][0]
vals_tc = df['tc_freq_major'].values[df['tc_freq_major'].values > 0]
ax.hist(vals_tc, bins=60, color='#d62728', edgecolor='white', alpha=0.8)
ax.set_xlabel('Annual TC Frequency (Cat 3+)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('TC Frequency Distribution\n(Non-zero locations only)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
pct_tc = (df['tc_freq_major'].values > 0).sum() / len(df) * 100
ax.text(0.98, 0.97, f'TC-exposed: {pct_tc:.0f}%', transform=ax.transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax = axes[1][1]
for ssp, label, c in zip(['ssp245','ssp370','ssp585'], ['SSP2-4.5','SSP3-7.0','SSP5-8.5'],
                          ['#2171b5','#fd8d3c','#cb181d']):
    ax.hist(df[f'cri_{ssp}'].values, bins=60, alpha=0.5, label=label, color=c, edgecolor='none')
ax.set_xlabel('Composite Risk Index', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('CRI Distribution by Scenario', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1][2]
# Mangrove point density map (2D histogram)
h, xedge, yedge = np.histogram2d(df['lon'], df['lat'], bins=[72, 35])
ax.imshow(h.T, origin='lower', aspect='auto',
          extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
          cmap='Greens', interpolation='bilinear')
ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude', fontsize=10)
ax.set_title('Mangrove Sample Density\n(GMW v4, 10% sample)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, color='white')

plt.suptitle('Data Overview: Risk Component Distributions and Mangrove Coverage',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig9_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig9")

# ─────────────────────────────────────────────────
# FIGURE 10: Sensitivity analysis — SLR thresholds
# ─────────────────────────────────────────────────
print("Generating Fig 10: Sensitivity analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel a: fraction above SLR thresholds by scenario
ax = axes[0]
thresholds = np.arange(1, 15, 0.5)
for ssp, label, c in zip(['ssp245','ssp370','ssp585'],
                          ['SSP2-4.5','SSP3-7.0','SSP5-8.5'],
                          ['#2171b5','#fd8d3c','#cb181d']):
    fracs = [(df[f'slr_median_{ssp}'] > t).mean() * 100 for t in thresholds]
    ax.plot(thresholds, fracs, label=label, color=c, linewidth=2)

ax.axvline(x=4, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(x=7, color='gray', linestyle=':', alpha=0.7, linewidth=1)
ax.text(4.1, 85, '4 mm/yr\n(moderate risk)', fontsize=8, color='gray')
ax.text(7.1, 85, '7 mm/yr\n(high risk)', fontsize=8, color='gray')
ax.set_xlabel('SLR Rate Threshold (mm yr⁻¹)', fontsize=11)
ax.set_ylabel('% of Mangrove Points Exceeding Threshold', fontsize=11)
ax.set_title('Mangroves Exposed to SLR Above Ecological Thresholds', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(1, 14)
ax.set_ylim(0, 100)

# Panel b: CRI sensitivity to component weights
ax = axes[1]
w_tc_vals = np.linspace(0, 1, 21)
for ssp, label, c in zip(['ssp245','ssp370','ssp585'],
                          ['SSP2-4.5','SSP3-7.0','SSP5-8.5'],
                          ['#2171b5','#fd8d3c','#cb181d']):
    mean_cri = []
    for w_tc in w_tc_vals:
        w_slr = 1 - w_tc
        cri = w_slr * df[f'slr_risk_{ssp}'] + w_tc * df['tc_risk_major']
        mean_cri.append(cri.mean())
    ax.plot(w_tc_vals, mean_cri, label=label, color=c, linewidth=2)

ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.text(0.51, ax.get_ylim()[0] * 1.02, 'Equal weights\n(w=0.5)', fontsize=8, color='gray')
ax.set_xlabel('Weight of TC Component (w_TC)', fontsize=11)
ax.set_ylabel('Mean Composite Risk Index', fontsize=11)
ax.set_title('Sensitivity to TC/SLR Component Weighting', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(0, 1)

plt.suptitle('Sensitivity Analysis: SLR Thresholds and Component Weighting Effects',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMGDIR}/fig10_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig10")

print("\nAll figures generated successfully!")
