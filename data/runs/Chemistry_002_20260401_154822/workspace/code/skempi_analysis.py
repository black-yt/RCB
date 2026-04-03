"""
SKEMPI 2.0 analysis for the barnase-barstar complex (1BRS_A_D).
Computes DDG values, identifies hotspots, analyzes mutation types,
and generates all related figures.
"""
import os, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_002_20260401_154822'
SKEMPI    = os.path.join(WORKSPACE, 'data', 'skempi_v2.csv')
OUTPUT    = os.path.join(WORKSPACE, 'outputs')
IMAGES    = os.path.join(WORKSPACE, 'report', 'images')

RT298 = 0.5923  # kcal/mol at 298 K  (R × 298 K)
HOTSPOT_CUTOFF = 1.5  # kcal/mol DDG threshold for hotspot

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(SKEMPI, sep=';')
print(f"Total SKEMPI entries: {len(df)}")

# Filter 1BRS barnase-barstar
brs = df[df['#Pdb'].str.contains('1BRS', na=False)].copy()
print(f"1BRS entries: {len(brs)}")

# Parse chains
brs['pdb_id']  = brs['#Pdb'].str.split('_').str[0]
brs['chains']  = brs['#Pdb'].str.split('_').str[1:].apply(lambda x: '_'.join(x))

# Keep only A_D (barnase–barstar pair)
brs_AD = brs[brs['#Pdb'] == '1BRS_A_D'].copy()
print(f"1BRS_A_D entries: {len(brs_AD)}")

# Compute DDG
brs_AD['ddg'] = RT298 * np.log(brs_AD['Affinity_mut_parsed'] / brs_AD['Affinity_wt_parsed'])

# Number of mutations
brs_AD['n_muts'] = brs_AD['Mutation(s)_cleaned'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

single = brs_AD[brs_AD['n_muts'] == 1].copy()
multi  = brs_AD[brs_AD['n_muts'] > 1].copy()
print(f"Single mutants: {len(single)}, Multi mutants: {len(multi)}")

# ── parse mutation codes ─────────────────────────────────────────────────────
# Format: <1-letter-WT><chain><res_num><1-letter-MUT>   e.g. KA25A
def parse_mutation(mut_str):
    """Parse 'KA25A' -> {wt:'K', chain:'A', resnum:25, mt:'A'}"""
    m = re.match(r'^([A-Z])([A-Z])(\d+)([A-Z])$', mut_str.strip())
    if m:
        return {'wt': m.group(1), 'chain': m.group(2),
                'resnum': int(m.group(3)), 'mt': m.group(4)}
    return None

# Parse single mutations
single = single.copy()
single['mut_parsed'] = single['Mutation(s)_cleaned'].apply(
    lambda x: parse_mutation(x) if pd.notna(x) else None)
single = single[single['mut_parsed'].notna()].copy()
single['wt_aa']   = single['mut_parsed'].apply(lambda x: x['wt'])
single['chain']   = single['mut_parsed'].apply(lambda x: x['chain'])
single['resnum']  = single['mut_parsed'].apply(lambda x: x['resnum'])
single['mut_aa']  = single['mut_parsed'].apply(lambda x: x['mt'])

print(f"\nParsed single mutations: {len(single)}")
print(f"Residues covered: {sorted(single['resnum'].unique())}")

# Average DDG per residue (some residues have multiple measurements)
resnum_ddg = single.groupby('resnum').agg(
    ddg_mean = ('ddg','mean'),
    ddg_std  = ('ddg','std'),
    n_meas   = ('ddg','count'),
    wt_aa    = ('wt_aa','first'),
    chain    = ('chain','first'),
).reset_index()
resnum_ddg['label'] = resnum_ddg['wt_aa'] + resnum_ddg['resnum'].astype(str)
resnum_ddg = resnum_ddg.sort_values('resnum')

print(f"\nUnique residues with measurements: {len(resnum_ddg)}")
hotspots = resnum_ddg[resnum_ddg['ddg_mean'] >= HOTSPOT_CUTOFF]
print(f"Hotspot residues (DDG ≥ {HOTSPOT_CUTOFF} kcal/mol): {len(hotspots)}")
print(hotspots[['label','ddg_mean','ddg_std','chain']].to_string())

# Save
resnum_ddg.to_csv(os.path.join(OUTPUT, 'residue_ddg.csv'), index=False)
single.to_csv(os.path.join(OUTPUT, 'single_mutations.csv'), index=False)
brs_AD.to_csv(os.path.join(OUTPUT, 'brs_AD_all_mutations.csv'), index=False)

# ── categorise all single mutations ──────────────────────────────────────────
# Alanine scanning vs other
single['is_ala'] = single['mut_aa'] == 'A'
ala_scan = single[single['is_ala']]
non_ala  = single[~single['is_ala']]

# By chain
chain_A_muts = single[single['chain'] == 'A']  # barnase mutations
chain_D_muts = single[single['chain'] == 'D']  # barstar mutations

print(f"\nAlanine scanning mutations: {len(ala_scan)}")
print(f"Barnase (A) mutations: {len(chain_A_muts)}")
print(f"Barstar (D) mutations: {len(chain_D_muts)}")

# ── FIGURE 4: DDG distribution (histogram + KDE) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: all mutations
ax = axes[0]
ddg_vals = brs_AD['ddg'].dropna()
ax.hist(ddg_vals, bins=30, color='steelblue', edgecolor='white', alpha=0.8,
        label=f'All mutations (n={len(ddg_vals)})')
ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label='ΔΔG = 0')
ax.axvline(HOTSPOT_CUTOFF, color='red', linewidth=1.5, linestyle='-',
           label=f'Hotspot threshold ({HOTSPOT_CUTOFF} kcal/mol)')
ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Distribution of ΔΔG Values\n1BRS_A_D (all {len(ddg_vals)} mutations)',
             fontsize=12)
ax.legend(fontsize=9)
ax.text(0.98, 0.97, f'Mean = {ddg_vals.mean():.2f}\nMedian = {ddg_vals.median():.2f}\nSD = {ddg_vals.std():.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: single mutations by chain
ax = axes[1]
colors = {'A': 'steelblue', 'D': 'darkorange'}
for ch, label, data in [('A','Barnase (A)', chain_A_muts),
                         ('D','Barstar (D)',  chain_D_muts)]:
    ax.hist(data['ddg'].dropna(), bins=20, alpha=0.7, color=colors[ch],
            label=f'{label} (n={len(data)})', edgecolor='white')
ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label='ΔΔG = 0')
ax.axvline(HOTSPOT_CUTOFF, color='red', linewidth=1.5, linestyle='-',
           label=f'Hotspot ({HOTSPOT_CUTOFF} kcal/mol)')
ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('ΔΔG Distribution by Protein Chain\n(Single mutations only)', fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig4_ddg_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_ddg_distribution.png")

# ── FIGURE 5: Hotspot residue bar plot ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

colors_bar = ['red' if d >= HOTSPOT_CUTOFF else 'steelblue' if d > 0 else '#2ecc71'
              for d in resnum_ddg['ddg_mean']]
bars = ax.bar(resnum_ddg['label'], resnum_ddg['ddg_mean'],
              color=colors_bar, edgecolor='white', linewidth=0.5)
ax.errorbar(x=range(len(resnum_ddg)),
            y=resnum_ddg['ddg_mean'],
            yerr=resnum_ddg['ddg_std'].fillna(0),
            fmt='none', color='black', capsize=3, linewidth=1)
ax.axhline(HOTSPOT_CUTOFF, color='red', linestyle='--', linewidth=1.5,
           label=f'Hotspot threshold ({HOTSPOT_CUTOFF} kcal/mol)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Mutated Residue', fontsize=11)
ax.set_ylabel('Mean ΔΔG (kcal/mol)', fontsize=11)
ax.set_title('Mean ΔΔG per Residue in Barnase-Barstar Complex (1BRS_A_D)\n'
             'Error bars = SD across replicate measurements', fontsize=12)
ax.set_xticklabels(resnum_ddg['label'], rotation=90, fontsize=9)
red_p  = mpatches.Patch(color='red',      label='Hotspot (ΔΔG ≥ 1.5)')
blue_p = mpatches.Patch(color='steelblue',label='Destabilising (ΔΔG > 0)')
green_p= mpatches.Patch(color='#2ecc71',  label='Neutral/stabilising (ΔΔG ≤ 0)')
ax.legend(handles=[red_p, blue_p, green_p], fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig5_hotspot_residues.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_hotspot_residues.png")

# ── FIGURE 6: Alanine scanning heatmap ───────────────────────────────────────
# DDG for each position × mutation type
pivot_data = single.pivot_table(values='ddg', index='resnum', columns='mut_aa', aggfunc='mean')
fig, ax = plt.subplots(figsize=(max(10, len(pivot_data.columns)*0.6), max(6, len(pivot_data)*0.3)))
sns.heatmap(pivot_data, cmap='RdYlGn_r', center=0, ax=ax,
            cbar_kws={'label': 'ΔΔG (kcal/mol)'},
            linewidths=0.3, linecolor='gray')
ax.set_xlabel('Mutant Amino Acid', fontsize=11)
ax.set_ylabel('Residue Number (Wild-type)', fontsize=11)
ax.set_yticklabels([f"{row['wt_aa']}{row['resnum']}"
                    for _, row in single.drop_duplicates('resnum').sort_values('resnum').iterrows()
                    if row['resnum'] in pivot_data.index],
                   rotation=0, fontsize=8)
ax.set_title('ΔΔG Heatmap: Residue × Mutation Type\nRed = destabilising, Green = stabilising/neutral',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig6_ddg_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_ddg_heatmap.png")

# ── FIGURE 7: Additivity of double mutations ──────────────────────────────────
# Compare observed DDG of double mutants vs sum of individual DDG
double = brs_AD[brs_AD['n_muts'] == 2].copy()
double_observed = double['ddg'].dropna().values

# For each double mutant, look up individual DDG sums
def get_single_ddg(mut_code, single_df):
    p = re.match(r'^([A-Z])([A-Z])(\d+)([A-Z])$', mut_code.strip())
    if not p:
        return np.nan
    wt, ch, rn, mt = p.group(1), p.group(2), int(p.group(3)), p.group(4)
    mask = (single_df['wt_aa']==wt) & (single_df['chain']==ch) & \
           (single_df['resnum']==rn) & (single_df['mut_aa']==mt)
    if mask.sum() == 0:
        return np.nan
    return single_df[mask]['ddg'].mean()

additive_ddg = []
obs_ddg_double = []
for _, row in double.iterrows():
    muts = [m.strip() for m in str(row['Mutation(s)_cleaned']).split(',')]
    indiv = [get_single_ddg(m, single) for m in muts]
    if all(np.isfinite(v) for v in indiv):
        additive_ddg.append(sum(indiv))
        obs_ddg_double.append(row['ddg'])

additive_ddg   = np.array(additive_ddg)
obs_ddg_double = np.array(obs_ddg_double)
print(f"\nDouble mutants with matched single DDG: {len(additive_ddg)}")

if len(additive_ddg) > 2:
    slope, intercept, r, p_val, se = stats.linregress(additive_ddg, obs_ddg_double)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(additive_ddg, obs_ddg_double, alpha=0.7, color='steelblue', s=60, zorder=3)
    xmin, xmax = additive_ddg.min()-0.5, additive_ddg.max()+0.5
    xs = np.linspace(xmin, xmax, 100)
    ax.plot(xs, slope*xs + intercept, 'r-', linewidth=2,
            label=f'Linear fit (r={r:.2f}, p={p_val:.3f})')
    ax.plot([xmin, xmax],[xmin, xmax], 'k--', linewidth=1, label='y = x (perfect additivity)')
    ax.set_xlabel('Sum of Single Mutant ΔΔG (kcal/mol)', fontsize=12)
    ax.set_ylabel('Observed Double Mutant ΔΔG (kcal/mol)', fontsize=12)
    ax.set_title('Additivity of Double Mutations\nBarnase-Barstar (1BRS_A_D)', fontsize=12)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.95, f'n = {len(additive_ddg)}\nSlope = {slope:.2f}\nR² = {r**2:.2f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, 'fig7_double_mutant_additivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig7_double_mutant_additivity.png")

# ── FIGURE 8: Database-wide overview ─────────────────────────────────────────
# Compare 1BRS to the rest of SKEMPI
df_all = pd.read_csv(SKEMPI, sep=';')
df_all['ddg'] = RT298 * np.log(df_all['Affinity_mut_parsed'] / df_all['Affinity_wt_parsed'])
df_all['n_muts'] = df_all['Mutation(s)_cleaned'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DDG distribution: SKEMPI overall vs 1BRS
ax = axes[0]
ddg_all  = df_all['ddg'].dropna()
ddg_1brs = brs_AD['ddg'].dropna()
ax.hist(ddg_all.clip(-5,15), bins=60, density=True, alpha=0.6, color='gray',
        label=f'All SKEMPI (n={len(ddg_all):,})')
ax.hist(ddg_1brs.clip(-5,15), bins=20, density=True, alpha=0.8, color='steelblue',
        label=f'1BRS_A_D (n={len(ddg_1brs)})')
ax.axvline(HOTSPOT_CUTOFF, color='red', linestyle='--', linewidth=1.5,
           label='Hotspot threshold')
ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('ΔΔG Distribution: 1BRS vs All SKEMPI', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(-5, 15)

# Number of mutations per complex (top 15 + 1BRS highlighted)
ax = axes[1]
pdb_counts = df_all.groupby('#Pdb').size().sort_values(ascending=False)
top_15 = pdb_counts.head(15)
colors_bar2 = ['red' if pid == '1BRS_A_D' else 'steelblue' for pid in top_15.index]
ax.barh(range(len(top_15)), top_15.values, color=colors_bar2)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of mutations', fontsize=11)
ax.set_title('Top 15 Complexes by Mutation Count\n(1BRS_A_D highlighted in red)', fontsize=12)
# Add 1BRS if not in top 15
if '1BRS_A_D' not in top_15.index:
    ax.text(0.98, 0.02, f'1BRS_A_D: {pdb_counts["1BRS_A_D"]} entries',
            transform=ax.transAxes, ha='right', fontsize=9, color='red')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig8_skempi_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_skempi_overview.png")

# ── save summary stats ────────────────────────────────────────────────────────
stats_out = {
    'n_total_1brs': int(len(brs_AD)),
    'n_single': int(len(single)),
    'n_multi':  int(len(multi)),
    'ddg_mean': float(brs_AD['ddg'].mean()),
    'ddg_median': float(brs_AD['ddg'].median()),
    'ddg_std':  float(brs_AD['ddg'].std()),
    'n_hotspots': int(len(hotspots)),
    'hotspot_residues': hotspots['label'].tolist(),
    'hotspot_ddg_mean': hotspots['ddg_mean'].tolist(),
    'n_stabilising': int((brs_AD['ddg'] < 0).sum()),
    'n_neutral': int(((brs_AD['ddg'] >= 0) & (brs_AD['ddg'] < HOTSPOT_CUTOFF)).sum()),
    'n_hotspot_cat': int((brs_AD['ddg'] >= HOTSPOT_CUTOFF).sum()),
}
with open(os.path.join(OUTPUT, 'skempi_summary.json'), 'w') as f:
    json.dump(stats_out, f, indent=2)
print("\nSaved skempi_summary.json")
print(json.dumps(stats_out, indent=2))
print("\nDone — skempi_analysis.py")
