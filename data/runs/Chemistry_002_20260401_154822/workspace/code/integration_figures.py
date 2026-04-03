"""
Final integration figure: SKEMPI overview, structure-mutation correlation,
and HADDOCK workflow schematic.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_002_20260401_154822'
OUTPUT    = os.path.join(WORKSPACE, 'outputs')
IMAGES    = os.path.join(WORKSPACE, 'report', 'images')

# Load data
df_brs    = pd.read_csv(os.path.join(OUTPUT, 'brs_AD_all_mutations.csv'))
single    = pd.read_csv(os.path.join(OUTPUT, 'single_mutations.csv'))
res_ddg   = pd.read_csv(os.path.join(OUTPUT, 'residue_ddg.csv'))
bsa_df    = pd.read_csv(os.path.join(OUTPUT, 'bsa_per_residue.csv'))
iface_df  = pd.read_csv(os.path.join(OUTPUT, 'interface_residues.csv'))

with open(os.path.join(OUTPUT, 'structural_summary.json')) as f:
    struct_sum = json.load(f)
with open(os.path.join(OUTPUT, 'skempi_summary.json')) as f:
    skempi_sum = json.load(f)
with open(os.path.join(OUTPUT, 'air_summary.json')) as f:
    air_sum = json.load(f)

RT298 = 0.5923
HOTSPOT_CUTOFF = 1.5

# ── FIGURE 12: HADDOCK workflow schematic ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_facecolor('#f8f9fa')

# Workflow steps
steps = [
    (1.0, 'SKEMPI 2.0\nDatabase\n(7,085 entries)', '#3498db'),
    (3.5, 'ΔΔG Calculation\nfrom Kd ratios\n(RTln[Kd_mut/Kd_wt])', '#9b59b6'),
    (6.0, 'Hotspot\nIdentification\n(ΔΔG ≥ 1.5 kcal/mol)', '#e74c3c'),
    (8.5, 'SASA Filtering\n(rSASA ≥ 10%)\nActive/Passive\nassignment', '#e67e22'),
    (11.0,'HADDOCK3\nAIR-driven\nDocking\n(25 restraint pairs)', '#27ae60'),
    (13.0,'Ranked\nEnsemble of\nStructures', '#2c3e50'),
]

for x, label, color in steps:
    box = FancyBboxPatch((x-0.85, 1.2), 1.7, 2.6,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', alpha=0.85, linewidth=2)
    ax.add_patch(box)
    ax.text(x, 2.5, label, ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', multialignment='center')

# Arrows
for i in range(len(steps)-1):
    x0 = steps[i][0] + 0.85
    x1 = steps[i+1][0] - 0.85
    ax.annotate('', xy=(x1, 2.5), xytext=(x0, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Annotate with numbers
annotations = [
    (2.25, 3.9, '94 barnase-barstar\nmutations'),
    (4.75, 3.9, '49 single / 45 multi'),
    (7.25, 3.9, '12 hotspot\nresidues'),
    (9.75, 3.9, '10 active residues\n26 passive residues'),
    (12.0, 3.9, '25 AIR pairs'),
]
for x, y, txt in annotations:
    ax.text(x, y, txt, ha='center', va='bottom', fontsize=8,
            color='#2c3e50', style='italic')
    ax.annotate('', xy=(x, 3.8), xytext=(x, 3.6),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1))

ax.set_title('HADDOCK3 AIR Design Pipeline: From SKEMPI Mutagenesis Data to Docking Restraints\n'
             'Applied to the Barnase-Barstar Complex (PDB: 1BRS_A_D)',
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig12_workflow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12_workflow.png")

# ── FIGURE 13: comprehensive residue analysis ─────────────────────────────────
# For each residue with DDG data, show DDG, BSA, and AIR status together
active_A_nums  = {int(r['resnum']) for r in air_sum['active_barnase']}
active_D_nums  = {int(r['resnum']) for r in air_sum['active_barstar']}
passive_A_nums = {int(r['resnum']) for r in air_sum['passive_barnase']}
passive_D_nums = {int(r['resnum']) for r in air_sum['passive_barstar']}

# Build per-residue merged dataframe
bsa_A_dict = dict(zip(bsa_df[bsa_df['chain']=='A']['res_num'],
                      bsa_df[bsa_df['chain']=='A']['bsa']))
bsa_D_dict = dict(zip(bsa_df[bsa_df['chain']=='D']['res_num'],
                      bsa_df[bsa_df['chain']=='D']['bsa']))

def get_air_status(chain, resnum):
    rn = int(resnum)
    if chain == 'A' and rn in active_A_nums:  return 'active'
    if chain == 'D' and rn in active_D_nums:  return 'active'
    if chain == 'A' and rn in passive_A_nums: return 'passive'
    if chain == 'D' and rn in passive_D_nums: return 'passive'
    return 'not selected'

res_ddg['air_status'] = res_ddg.apply(
    lambda r: get_air_status(r['chain'], r['resnum']), axis=1)
res_ddg['bsa'] = res_ddg.apply(
    lambda r: bsa_A_dict.get(r['resnum'], 0) if r['chain']=='A'
              else bsa_D_dict.get(r['resnum'], 0), axis=1)

color_map = {'active': 'red', 'passive': 'orange', 'not selected': 'steelblue'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: BSA vs DDG, colored by AIR status
ax = axes[0]
for status, grp in res_ddg.groupby('air_status'):
    ax.scatter(grp['bsa'], grp['ddg_mean'], color=color_map[status],
               s=80, alpha=0.85, zorder=3, label=status.capitalize())
    for _, row in grp.iterrows():
        ax.annotate(row['label'], (row['bsa'], row['ddg_mean']),
                    fontsize=8, xytext=(3, 3), textcoords='offset points')
ax.axhline(HOTSPOT_CUTOFF, color='red', linestyle='--', linewidth=1, alpha=0.6)
ax.set_xlabel('Buried Surface Area (Å²)', fontsize=11)
ax.set_ylabel('Mean ΔΔG (kcal/mol)', fontsize=11)
ax.set_title('BSA vs ΔΔG for All Mutated Residues\nColored by HADDOCK AIR Status', fontsize=12)
ax.legend(fontsize=9)

# Bar chart: number of residues in each category by chain
ax = axes[1]
cats = ['active','passive','not selected']
cols = [color_map[c] for c in cats]
grp_A = res_ddg[res_ddg['chain']=='A']['air_status'].value_counts().reindex(cats, fill_value=0)
grp_D = res_ddg[res_ddg['chain']=='D']['air_status'].value_counts().reindex(cats, fill_value=0)
x = np.arange(len(cats))
w = 0.35
ax.bar(x-w/2, grp_A.values, w, color=cols, alpha=0.85, label='Barnase (A)',
       edgecolor='white', linewidth=0.5)
ax.bar(x+w/2, grp_D.values, w, color=cols, alpha=0.5, label='Barstar (D)',
       edgecolor='black', linewidth=0.5, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in cats], fontsize=11)
ax.set_ylabel('Number of residues with ΔΔG data', fontsize=11)
ax.set_title('AIR Residue Classification by Chain', fontsize=12)
# Add value labels
for i, (va, vd) in enumerate(zip(grp_A.values, grp_D.values)):
    ax.text(i-w/2, va+0.05, str(va), ha='center', fontsize=10)
    ax.text(i+w/2, vd+0.05, str(vd), ha='center', fontsize=10)

# Manual legend
blue_p  = mpatches.Patch(facecolor='gray', alpha=0.85, label='Barnase (A)')
empty_p = mpatches.Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Barstar (D)')
ax.legend(handles=[blue_p, empty_p], fontsize=9)

plt.suptitle('Residue-Level Analysis: Structural and Biochemical Properties',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig13_residue_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13_residue_analysis.png")

# ── Verify all figures ────────────────────────────────────────────────────────
figures = sorted([f for f in os.listdir(IMAGES) if f.endswith('.png')])
print(f"\nAll figures ({len(figures)}):")
for f in figures:
    size_kb = os.path.getsize(os.path.join(IMAGES, f)) // 1024
    print(f"  {f}  ({size_kb} KB)")

print("\nDone — integration_figures.py")
