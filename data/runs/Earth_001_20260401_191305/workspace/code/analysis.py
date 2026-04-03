"""
Cloud Seeding US 2000-2025: Comprehensive Analysis
Reproduces key empirical conclusions from published structured dataset.
Analyzes: spatial concentration, annual activity dynamics, purpose composition,
and agent-apparatus deployment patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_001_20260401_191305"
DATA_PATH  = f"{WORKSPACE}/data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv"
GEOJSON    = f"{WORKSPACE}/data/dataset1_cloud_seeding_records/us_states.geojson"
OUT_DIR    = f"{WORKSPACE}/outputs"
IMG_DIR    = f"{WORKSPACE}/report/images"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
PALETTE = sns.color_palette("tab10", 14)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD & CLEAN DATA
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} records, {df['year'].min()}–{df['year'].max()}")

# Normalise text fields
for col in ['state','agent','apparatus','purpose','season','operator_affiliation']:
    df[col] = df[col].str.strip().str.lower()

# Canonical purpose: keep first stated purpose only for simple categorisation
def primary_purpose(s):
    return s.split(',')[0].strip()

df['purpose_primary'] = df['purpose'].apply(primary_purpose)

# Canonical apparatus
def canonical_apparatus(s):
    if pd.isna(s): return 'unknown'
    s = str(s).strip().lower()
    if 'ground' in s and 'airborne' in s:
        return 'ground & airborne'
    elif 'airborne' in s:
        return 'airborne'
    elif 'ground' in s:
        return 'ground'
    return s

df['apparatus_cat'] = df['apparatus'].apply(canonical_apparatus)

# Canonical agent: broad categories
def agent_category(s):
    s = str(s).lower()
    if 'silver iodide' in s and 'sodium iodide' in s:
        return 'silver iodide + sodium iodide'
    elif 'silver iodide' in s and 'ammonium iodide' in s:
        return 'silver iodide + ammonium iodide'
    elif 'silver iodide' in s and ('hygroscopic' in s or 'calcium chloride' in s or 'sodium chloride' in s or 'potassium chloride' in s):
        return 'silver iodide + hygroscopic'
    elif 'silver iodide' in s:
        return 'silver iodide (pure)'
    elif 'ionized air' in s:
        return 'ionized air'
    elif 'carbon dioxide' in s or 'dry ice' in s:
        return 'dry ice / CO₂'
    elif 'calcium chloride' in s or 'sodium chloride' in s:
        return 'hygroscopic salts'
    else:
        return 'other'

df['agent_cat'] = df['agent'].apply(agent_category)

# Save cleaned dataset
df.to_csv(f"{OUT_DIR}/cloud_seeding_cleaned.csv", index=False)
print("Cleaned dataset saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
summary = {
    "total_records": len(df),
    "year_range": f"{df['year'].min()}–{df['year'].max()}",
    "n_states": df['state'].nunique(),
    "states": sorted(df['state'].unique().tolist()),
    "n_operators": df['operator_affiliation'].nunique(),
    "top_states": df['state'].value_counts().head(5).to_dict(),
    "top_operators": df['operator_affiliation'].value_counts().head(5).to_dict(),
    "purpose_distribution": df['purpose_primary'].value_counts().to_dict(),
    "apparatus_distribution": df['apparatus_cat'].value_counts().to_dict(),
    "agent_cat_distribution": df['agent_cat'].value_counts().to_dict(),
    "season_distribution": df['season'].value_counts().head(8).to_dict(),
    "annual_counts": df['year'].value_counts().sort_index().to_dict(),
}
with open(f"{OUT_DIR}/summary_statistics.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary statistics saved.")

# Also build summary table
state_summary = df.groupby('state').agg(
    total_projects=('project', 'count'),
    year_first=('year', 'min'),
    year_last=('year', 'max'),
    n_operators=('operator_affiliation', 'nunique'),
    n_purposes=('purpose_primary', 'nunique'),
).sort_values('total_projects', ascending=False)
state_summary.to_csv(f"{OUT_DIR}/state_summary.csv")

annual_summary = df.groupby('year').agg(
    total_projects=('project', 'count'),
    n_states=('state', 'nunique'),
    n_operators=('operator_affiliation', 'nunique'),
).reset_index()
annual_summary.to_csv(f"{OUT_DIR}/annual_summary.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 3.  FIGURE 1: Annual Activity Dynamics (Bar chart with trend line)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

annual = df.groupby('year').size().reset_index(name='count')
all_years = pd.DataFrame({'year': range(df['year'].min(), df['year'].max()+1)})
annual = all_years.merge(annual, on='year', how='left').fillna(0)

ax = axes[0]
bars = ax.bar(annual['year'], annual['count'], color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5)
# Trend line
z = np.polyfit(annual['year'], annual['count'], 1)
p = np.poly1d(z)
ax.plot(annual['year'], p(annual['year']), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.1f}/yr)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Annual Cloud-Seeding Project Count (2000–2025)')
ax.legend()
ax.set_xticks(range(2000, 2026, 2))
ax.set_xticklabels(range(2000, 2026, 2), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Stacked bar: Annual count by apparatus
ax2 = axes[1]
app_annual = df.groupby(['year', 'apparatus_cat']).size().unstack(fill_value=0)
app_annual = all_years.set_index('year').join(app_annual).fillna(0)
colors_app = {'ground': '#2196F3', 'airborne': '#FF9800', 'ground & airborne': '#4CAF50', 'unknown': '#9E9E9E'}
bottom = np.zeros(len(app_annual))
for col in ['ground', 'airborne', 'ground & airborne']:
    if col in app_annual.columns:
        ax2.bar(app_annual.index, app_annual[col], bottom=bottom,
                label=col.capitalize(), color=colors_app.get(col, 'gray'), alpha=0.85, edgecolor='white', linewidth=0.3)
        bottom += app_annual[col].values
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Projects')
ax2.set_title('Annual Projects by Deployment Apparatus')
ax2.legend(loc='upper left')
ax2.set_xticks(range(2000, 2026, 2))
ax2.set_xticklabels(range(2000, 2026, 2), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig1_annual_activity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  FIGURE 2: Spatial Concentration (Choropleth map)
# ═══════════════════════════════════════════════════════════════════════════
try:
    import geopandas as gpd
    gdf = gpd.read_file(GEOJSON)
    print("GeoJSON columns:", gdf.columns.tolist())
    # Identify state name column
    name_col = None
    for c in ['NAME', 'name', 'STATE_NAME', 'state_name', 'STUSPS']:
        if c in gdf.columns:
            name_col = c
            break
    if name_col:
        gdf[name_col] = gdf[name_col].str.lower()
        state_counts = df['state'].value_counts().reset_index()
        state_counts.columns = ['state', 'count']
        gdf = gdf.merge(state_counts, left_on=name_col, right_on='state', how='left')
        gdf['count'] = gdf['count'].fillna(0)

        # Filter to contiguous US for better visualization
        exclude = ['alaska', 'hawaii', 'puerto rico', 'guam', 'american samoa',
                   'united states virgin islands', 'commonwealth of the northern mariana islands']
        gdf_cont = gdf[~gdf[name_col].isin(exclude)]

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        gdf_cont.plot(column='count', ax=ax, cmap='YlOrRd', legend=True,
                      legend_kwds={'label': 'Number of Projects', 'orientation': 'horizontal', 'shrink': 0.6},
                      missing_kwds={'color': 'lightgrey'},
                      edgecolor='white', linewidth=0.5)
        # Annotate states with projects
        for _, row in gdf_cont[gdf_cont['count'] > 0].iterrows():
            centroid = row.geometry.centroid
            ax.annotate(f"{int(row['count'])}", xy=(centroid.x, centroid.y),
                       ha='center', va='center', fontsize=7, fontweight='bold', color='black')
        ax.set_title('Geographic Distribution of Cloud-Seeding Projects (2000–2025)', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/fig2_spatial_map.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Figure 2 (map) saved.")
    else:
        raise ValueError("No name column found")
except Exception as e:
    print(f"Map error: {e}. Creating bar chart alternative.")
    fig, ax = plt.subplots(figsize=(12, 6))
    state_counts = df['state'].value_counts()
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(state_counts)))[::-1]
    bars = ax.bar(state_counts.index, state_counts.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, state_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('State')
    ax.set_ylabel('Number of Projects')
    ax.set_title('Cloud-Seeding Project Count by State (2000–2025)')
    ax.set_xticklabels([s.title() for s in state_counts.index], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig2_spatial_map.png", dpi=150, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 5.  FIGURE 3: Purpose Composition
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Pie chart – primary purpose
purpose_counts = df['purpose_primary'].value_counts()
purpose_labels = [p.replace('augment snowpack', 'Augment Snowpack')
                   .replace('increase precipitation', 'Increase Precipitation')
                   .replace('suppress hail', 'Suppress Hail')
                   .replace('suppress fog', 'Suppress Fog')
                   .replace('research', 'Research')
                   .replace('increase runoff', 'Increase Runoff')
                   for p in purpose_counts.index]
colors_pie = plt.cm.Set2(np.linspace(0, 1, len(purpose_counts)))
wedges, texts, autotexts = axes[0].pie(
    purpose_counts.values,
    labels=purpose_labels,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
    colors=colors_pie,
    startangle=140,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(9)
axes[0].set_title('Primary Purpose Distribution\n(All Records, 2000–2025)', fontsize=12)

# Stacked bar – purpose by state (top 8 states)
top_states = df['state'].value_counts().head(8).index.tolist()
purpose_by_state = df[df['state'].isin(top_states)].groupby(
    ['state', 'purpose_primary']).size().unstack(fill_value=0)
purpose_by_state = purpose_by_state.loc[top_states]  # preserve order
purpose_colors = {
    'augment snowpack': '#1565C0',
    'increase precipitation': '#2196F3',
    'suppress hail': '#F44336',
    'suppress fog': '#9C27B0',
    'research': '#4CAF50',
    'increase runoff': '#FF9800',
    'augment snowpack, increase runoff': '#00BCD4',
}
bottom = np.zeros(len(purpose_by_state))
for purp in purpose_by_state.columns:
    color = purpose_colors.get(purp, '#9E9E9E')
    label = purp.replace('augment snowpack', 'Aug. Snowpack').replace('increase precipitation', 'Inc. Precipitation')
    axes[1].bar(purpose_by_state.index, purpose_by_state[purp],
                bottom=bottom, color=color, label=label, alpha=0.85, edgecolor='white', linewidth=0.5)
    bottom += purpose_by_state[purp].values
axes[1].set_xlabel('State')
axes[1].set_ylabel('Number of Projects')
axes[1].set_title('Purpose Composition by State\n(Top 8 States)', fontsize=12)
axes[1].set_xticklabels([s.title() for s in purpose_by_state.index], rotation=30, ha='right')
axes[1].legend(loc='upper right', fontsize=8, ncol=1)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig3_purpose_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  FIGURE 4: Agent-Apparatus Deployment Patterns
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Agent category distribution – horizontal bar
agent_counts = df['agent_cat'].value_counts()
colors_agent = plt.cm.Paired(np.linspace(0, 1, len(agent_counts)))
bars = axes[0].barh(agent_counts.index[::-1], agent_counts.values[::-1],
                     color=colors_agent, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, agent_counts.values[::-1]):
    axes[0].text(val + 2, bar.get_y() + bar.get_height()/2.,
                 f'{val} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
axes[0].set_xlabel('Number of Projects')
axes[0].set_title('Seeding Agent Categories\n(2000–2025)', fontsize=12)
axes[0].set_xlim(0, agent_counts.max() * 1.25)
axes[0].grid(axis='x', alpha=0.3)

# Cross-tabulation: apparatus vs agent_cat (heatmap)
cross = df.groupby(['apparatus_cat', 'agent_cat']).size().unstack(fill_value=0)
# Reorder
app_order = ['ground', 'airborne', 'ground & airborne']
app_order = [a for a in app_order if a in cross.index]
cross = cross.loc[app_order]
sns.heatmap(cross, ax=axes[1], cmap='Blues', annot=True, fmt='d',
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
axes[1].set_title('Agent × Apparatus Cross-Tabulation', fontsize=12)
axes[1].set_xlabel('Seeding Agent Category')
axes[1].set_ylabel('Deployment Apparatus')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=9)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig4_agent_apparatus.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  FIGURE 5: Seasonal Patterns & Purpose Over Time
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Season distribution
def primary_season(s):
    return s.split(',')[0].strip()
df['season_primary'] = df['season'].apply(primary_season)
season_counts = df['season_primary'].value_counts()
season_colors = {'winter': '#90CAF9', 'summer': '#FFCC80', 'spring': '#A5D6A7',
                 'fall': '#FFAB91', 'winter,spring': '#CE93D8'}
sc = [season_colors.get(s, '#E0E0E0') for s in season_counts.index]
wedges, texts, autotexts = axes[0].pie(
    season_counts.values, labels=[s.title() for s in season_counts.index],
    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
    colors=sc, startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts: at.set_fontsize(9)
axes[0].set_title('Seasonal Distribution of\nCloud-Seeding Projects', fontsize=12)

# Purpose over time (stacked area)
purpose_year = df.groupby(['year', 'purpose_primary']).size().unstack(fill_value=0)
top_purposes = df['purpose_primary'].value_counts().head(5).index.tolist()
purpose_year_top = purpose_year[top_purposes].reindex(range(df['year'].min(), df['year'].max()+1), fill_value=0)
purpose_nice = {
    'augment snowpack': 'Augment Snowpack',
    'increase precipitation': 'Increase Precipitation',
    'suppress hail': 'Suppress Hail',
    'suppress fog': 'Suppress Fog',
    'research': 'Research',
}
colors_stack = ['#1565C0', '#2196F3', '#F44336', '#9C27B0', '#4CAF50']
axes[1].stackplot(purpose_year_top.index,
                  [purpose_year_top[p] for p in top_purposes],
                  labels=[purpose_nice.get(p, p.title()) for p in top_purposes],
                  colors=colors_stack[:len(top_purposes)], alpha=0.8)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Projects')
axes[1].set_title('Purpose Composition Over Time\n(Top 5 Purposes)', fontsize=12)
axes[1].legend(loc='upper left', fontsize=9)
axes[1].set_xticks(range(2000, 2026, 2))
axes[1].set_xticklabels(range(2000, 2026, 2), rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig5_seasonal_purpose_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  FIGURE 6: Top Operators Analysis
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Top operators by count
top_ops = df['operator_affiliation'].value_counts().head(10)
colors_ops = plt.cm.tab10(np.linspace(0, 1, 10))
bars = axes[0].barh(top_ops.index[::-1], top_ops.values[::-1],
                     color=colors_ops, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top_ops.values[::-1]):
    axes[0].text(val + 1, bar.get_y() + bar.get_height()/2.,
                 str(val), va='center', fontsize=9)
axes[0].set_xlabel('Number of Projects')
axes[0].set_title('Top 10 Cloud-Seeding Operators\n(2000–2025)', fontsize=12)
axes[0].set_xlim(0, top_ops.max() * 1.18)
# Wrap long labels
labels = [l.get_text().title() for l in axes[0].get_yticklabels()]
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].grid(axis='x', alpha=0.3)

# Operator activity over time (top 5)
top5_ops = df['operator_affiliation'].value_counts().head(5).index.tolist()
op_year = df[df['operator_affiliation'].isin(top5_ops)].groupby(
    ['year', 'operator_affiliation']).size().unstack(fill_value=0)
op_year = op_year.reindex(range(df['year'].min(), df['year'].max()+1), fill_value=0)
for i, op in enumerate(top5_ops):
    if op in op_year.columns:
        axes[1].plot(op_year.index, op_year[op], marker='o', markersize=4,
                     label=op.title(), linewidth=2, color=colors_ops[i])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Projects')
axes[1].set_title('Annual Activity of Top 5 Operators\n(2000–2025)', fontsize=12)
axes[1].legend(loc='upper right', fontsize=8)
axes[1].set_xticks(range(2000, 2026, 2))
axes[1].set_xticklabels(range(2000, 2026, 2), rotation=45, ha='right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig6_operators.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  FIGURE 7: State-level heat-map (year × state)
# ═══════════════════════════════════════════════════════════════════════════
pivot = df.groupby(['year', 'state']).size().unstack(fill_value=0)
# order states by total
state_order = pivot.sum(axis=0).sort_values(ascending=False).index
pivot = pivot[state_order]

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot.T, ax=ax, cmap='YlOrRd', linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Number of Projects'})
ax.set_title('Cloud-Seeding Activity Heatmap: Year × State (2000–2025)', fontsize=13)
ax.set_xlabel('Year')
ax.set_ylabel('State')
ax.set_yticklabels([s.title() for s in pivot.T.index], rotation=0, fontsize=9)
ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig7_heatmap_year_state.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 10.  FIGURE 8: Data Overview / Dashboard
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.suptitle('U.S. Cloud-Seeding Activity Overview (2000–2025)', fontsize=15, fontweight='bold', y=0.98)
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

# (a) Total records per state
ax_a = fig.add_subplot(gs[0, 0])
sc = df['state'].value_counts()
ax_a.barh(sc.index[::-1], sc.values[::-1], color='steelblue', alpha=0.85)
ax_a.set_title('(a) Projects by State')
ax_a.set_xlabel('Count')
ax_a.set_yticklabels([s.title() for s in sc.index[::-1]], fontsize=8)

# (b) Apparatus distribution
ax_b = fig.add_subplot(gs[0, 1])
app_c = df['apparatus_cat'].value_counts()
ax_b.pie(app_c.values, labels=[s.title() for s in app_c.index],
         autopct='%1.1f%%', startangle=90, colors=['#2196F3', '#FF9800', '#4CAF50'],
         wedgeprops={'edgecolor': 'white'})
ax_b.set_title('(b) Apparatus Type')

# (c) Season distribution
ax_c = fig.add_subplot(gs[0, 2])
def classify_season(s):
    s = str(s).lower()
    if 'winter' in s and 'spring' not in s and 'summer' not in s and 'fall' not in s:
        return 'Winter only'
    elif 'summer' in s and 'winter' not in s and 'spring' not in s and 'fall' not in s:
        return 'Summer only'
    elif 'winter' in s and 'spring' in s:
        return 'Winter+Spring'
    elif 'spring' in s or 'summer' in s or 'fall' in s:
        return 'Multi-season'
    return 'Other'
df['season_class'] = df['season'].apply(classify_season)
sc2 = df['season_class'].value_counts()
ax_c.bar(sc2.index, sc2.values, color=['#90CAF9','#FFCC80','#CE93D8','#A5D6A7','#E0E0E0'], alpha=0.85)
ax_c.set_title('(c) Season Classification')
ax_c.set_xticklabels(sc2.index, rotation=30, ha='right', fontsize=9)
ax_c.grid(axis='y', alpha=0.3)

# (d) Annual trend
ax_d = fig.add_subplot(gs[1, 0:2])
yr_c = df.groupby('year').size().reindex(range(2000, 2026), fill_value=0)
ax_d.bar(yr_c.index, yr_c.values, color='steelblue', alpha=0.8)
z2 = np.polyfit(yr_c.index, yr_c.values, 1)
ax_d.plot(yr_c.index, np.poly1d(z2)(yr_c.index), 'r--', lw=2, label=f'Trend: {z2[0]:+.1f}/yr')
ax_d.set_title('(d) Annual Project Count with Trend')
ax_d.set_xlabel('Year')
ax_d.set_ylabel('Projects')
ax_d.legend()
ax_d.grid(axis='y', alpha=0.3)
ax_d.set_xticks(range(2000, 2026, 2))
ax_d.set_xticklabels(range(2000, 2026, 2), rotation=45, ha='right', fontsize=8)

# (e) Agent top 6
ax_e = fig.add_subplot(gs[1, 2])
ag_c = df['agent_cat'].value_counts().head(6)
ax_e.barh(ag_c.index[::-1], ag_c.values[::-1], color=plt.cm.Paired(np.linspace(0,1,6)))
ax_e.set_title('(e) Agent Categories')
ax_e.set_xlabel('Count')
ax_e.set_yticklabels(ag_c.index[::-1], fontsize=8)

plt.savefig(f"{IMG_DIR}/fig8_overview_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 11.  SUMMARY TABLES (CSV + print)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TABLE 1: State Summary ===")
print(state_summary.to_string())

print("\n=== TABLE 2: Purpose Distribution ===")
print(df['purpose_primary'].value_counts().to_string())

print("\n=== TABLE 3: Agent Distribution ===")
print(df['agent_cat'].value_counts().to_string())

print("\n=== TABLE 4: Apparatus Distribution ===")
print(df['apparatus_cat'].value_counts().to_string())

print("\n=== TABLE 5: Top Operators ===")
print(df['operator_affiliation'].value_counts().head(10).to_string())

print("\nAll analyses complete.")
print(f"Outputs: {OUT_DIR}")
print(f"Figures: {IMG_DIR}")
