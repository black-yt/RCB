"""
HADDOCK AIR design from SKEMPI mutagenesis data for barnase-barstar (1BRS_A_D).
Also produces the integrated structure-affinity figure.
"""
import os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_002_20260401_154822'
PDB_FILE  = os.path.join(WORKSPACE, 'data', '1brs_AD.pdb')
OUTPUT    = os.path.join(WORKSPACE, 'outputs')
IMAGES    = os.path.join(WORKSPACE, 'report', 'images')

RT298          = 0.5923
HOTSPOT_CUTOFF = 1.5   # kcal/mol — HADDOCK active residue threshold
PASSIVE_SA_CUTOFF = 50  # minimum % SASA for passive residues

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V',
}

# ── reload data ───────────────────────────────────────────────────────────────
df_brs = pd.read_csv(os.path.join(OUTPUT, 'brs_AD_all_mutations.csv'))
single  = pd.read_csv(os.path.join(OUTPUT, 'single_mutations.csv'))
res_ddg = pd.read_csv(os.path.join(OUTPUT, 'residue_ddg.csv'))
iface   = pd.read_csv(os.path.join(OUTPUT, 'interface_residues.csv'))
bsa_df  = pd.read_csv(os.path.join(OUTPUT, 'bsa_per_residue.csv'))
with open(os.path.join(OUTPUT, 'structural_summary.json')) as f:
    struct_sum = json.load(f)

# ── compute SASA for individual chains ───────────────────────────────────────
parser = PDBParser(QUIET=True)
structure = parser.get_structure('1BRS', PDB_FILE)
model = structure[0]

from copy import deepcopy
from Bio.PDB import Structure, Model

# SASA for chain A alone
struct_A = Structure.Structure('A')
ma = Model.Model(0)
ma.add(deepcopy(model['A']))
struct_A.add(ma)

struct_D = Structure.Structure('D')
md = Model.Model(0)
md.add(deepcopy(model['D']))
struct_D.add(md)

sr = ShrakeRupley()
sr.compute(struct_A[0], level='R')
sasa_A = {r.get_id()[1]: r.sasa for r in struct_A[0]['A'] if r.get_resname() in AA3TO1}

sr.compute(struct_D[0], level='R')
sasa_D = {r.get_id()[1]: r.sasa for r in struct_D[0]['D'] if r.get_resname() in AA3TO1}

# Max possible SASA per residue (approximate, Gly=167, Trp=334)
MAX_SASA = {
    'ALA':129,'ARG':274,'ASN':195,'ASP':193,'CYS':167,'GLN':225,
    'GLU':223,'GLY':167,'HIS':224,'ILE':197,'LEU':201,'LYS':236,
    'MET':224,'PHE':240,'PRO':159,'SER':155,'THR':172,'TRP':334,
    'TYR':263,'VAL':174,
}

# Relative SASA
res_A = [r for r in model['A'] if r.get_resname() in AA3TO1]
res_D = [r for r in model['D'] if r.get_resname() in AA3TO1]

rSASA_A = {}
for r in res_A:
    rid, rname = r.get_id()[1], r.get_resname()
    max_s = MAX_SASA.get(rname, 200)
    rSASA_A[rid] = (sasa_A.get(rid, 0) / max_s) * 100

rSASA_D = {}
for r in res_D:
    rid, rname = r.get_id()[1], r.get_resname()
    max_s = MAX_SASA.get(rname, 200)
    rSASA_D[rid] = (sasa_D.get(rid, 0) / max_s) * 100

# ── define AIRs from SKEMPI data ─────────────────────────────────────────────
# HADDOCK protocol:
# Active residues = mutations with DDG >= threshold AND solvent accessible
# Passive residues = surface neighbors of active residues

hotspot_A = res_ddg[(res_ddg['chain'] == 'A') & (res_ddg['ddg_mean'] >= HOTSPOT_CUTOFF)]
hotspot_D = res_ddg[(res_ddg['chain'] == 'D') & (res_ddg['ddg_mean'] >= HOTSPOT_CUTOFF)]

print(f"Hotspot barnase residues: {hotspot_A['label'].tolist()}")
print(f"Hotspot barstar residues: {hotspot_D['label'].tolist()}")

# Filter for solvent accessibility (apply loosely to ensure some active residues)
# Use 10% threshold (consistent with HADDOCK paper approach)
SA_ACT_CUTOFF = 10.0

active_A = []
for _, row in hotspot_A.iterrows():
    rnum = int(row['resnum'])
    sasa_pct = rSASA_A.get(rnum, 0)
    if sasa_pct >= SA_ACT_CUTOFF:
        active_A.append({'resnum': rnum, 'label': row['label'],
                         'ddg_mean': row['ddg_mean'], 'rSASA': sasa_pct})

active_D = []
for _, row in hotspot_D.iterrows():
    rnum = int(row['resnum'])
    sasa_pct = rSASA_D.get(rnum, 0)
    if sasa_pct >= SA_ACT_CUTOFF:
        active_D.append({'resnum': rnum, 'label': row['label'],
                         'ddg_mean': row['ddg_mean'], 'rSASA': sasa_pct})

df_active_A = pd.DataFrame(active_A)
df_active_D = pd.DataFrame(active_D)
print(f"\nActive residues (AIR-eligible):")
print(f"  Barnase: {df_active_A['label'].tolist()}")
print(f"  Barstar: {df_active_D['label'].tolist()}")

# Passive residues: interface residues that are NOT active but surface-exposed
iface_A_nums = set(iface[iface['chain']=='A']['res_num'].astype(int))
iface_D_nums = set(iface[iface['chain']=='D']['res_num'].astype(int))
active_A_nums = set(df_active_A['resnum'].astype(int)) if len(df_active_A) else set()
active_D_nums = set(df_active_D['resnum'].astype(int)) if len(df_active_D) else set()

passive_A = []
for r in res_A:
    rnum = r.get_id()[1]
    rname = r.get_resname()
    sasa_pct = rSASA_A.get(rnum, 0)
    if rnum in iface_A_nums and rnum not in active_A_nums and sasa_pct >= SA_ACT_CUTOFF:
        passive_A.append({'resnum': rnum, 'label': f"{AA3TO1.get(rname,'?')}{rnum}",
                          'rSASA': sasa_pct})

passive_D = []
for r in res_D:
    rnum = r.get_id()[1]
    rname = r.get_resname()
    sasa_pct = rSASA_D.get(rnum, 0)
    if rnum in iface_D_nums and rnum not in active_D_nums and sasa_pct >= SA_ACT_CUTOFF:
        passive_D.append({'resnum': rnum, 'label': f"{AA3TO1.get(rname,'?')}{rnum}",
                          'rSASA': sasa_pct})

df_passive_A = pd.DataFrame(passive_A)
df_passive_D = pd.DataFrame(passive_D)
print(f"\nPassive residues (interface, surface-exposed):")
print(f"  Barnase: {df_passive_A['label'].tolist() if len(df_passive_A) else []}")
print(f"  Barstar: {df_passive_D['label'].tolist() if len(df_passive_D) else []}")

# ── write HADDOCK-style AIR table ─────────────────────────────────────────────
air_records = []
for role, df_r, chain in [('active', df_active_A,'A'), ('active', df_active_D,'D'),
                            ('passive', df_passive_A,'A'), ('passive', df_passive_D,'D')]:
    for _, row in df_r.iterrows():
        air_records.append({'role': role, 'chain': chain, **row})

df_air = pd.DataFrame(air_records)
df_air.to_csv(os.path.join(OUTPUT, 'haddock_airs.csv'), index=False)
print(f"\nAIR table saved with {len(df_air)} entries")

# Also write HADDOCK .tbl format (ambiguous restraints file)
air_text = "! HADDOCK Ambiguous Interaction Restraints (AIRs)\n"
air_text += "! Generated from SKEMPI 2.0 mutagenesis data for 1BRS_A_D\n"
air_text += f"! Active threshold: DDG >= {HOTSPOT_CUTOFF} kcal/mol AND rSASA >= {SA_ACT_CUTOFF}%\n\n"

for _, row_a in df_active_A.iterrows():
    for _, row_d in df_active_D.iterrows():
        air_text += (f"assign (segid A and resi {int(row_a['resnum'])})\n"
                     f"       (segid D and resi {int(row_d['resnum'])}) 3.0 3.0 0.0\n\n")

with open(os.path.join(OUTPUT, 'barnase_barstar.tbl'), 'w') as f:
    f.write(air_text)
print(f"Written HADDOCK .tbl file with {len(df_active_A)*len(df_active_D)} AIR pairs")

# ── FIGURE 9: AIR map – structural context ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def make_linear_map(ax, res_list, rSASA_dict, active_nums, passive_nums, ddg_dict,
                    title, color_act='red', color_pass='orange'):
    resnums = [r.get_id()[1] for r in res_list]
    res_names = [r.get_resname() for r in res_list]
    sasa_vals = [rSASA_dict.get(n, 0) for n in resnums]

    # Background SASA profile
    ax.fill_between(range(len(resnums)), sasa_vals, alpha=0.2, color='gray', label='rSASA (%)')
    ax.plot(range(len(resnums)), sasa_vals, color='gray', linewidth=0.8)

    # Mark active residues
    for i, (rn, rname) in enumerate(zip(resnums, res_names)):
        if rn in active_nums:
            ddg_val = ddg_dict.get(rn, 0)
            ax.axvline(i, color=color_act, alpha=0.8, linewidth=2)
            ax.text(i, sasa_vals[i]+3, f"{AA3TO1.get(rname,'?')}{rn}\n({ddg_val:.1f})",
                    fontsize=7, ha='center', va='bottom', color='darkred',
                    rotation=90)
        elif rn in passive_nums:
            ax.axvline(i, color=color_pass, alpha=0.6, linewidth=1.5, linestyle='--')

    ax.axhline(SA_ACT_CUTOFF, color='navy', linestyle=':', linewidth=1,
               label=f'SA cutoff ({SA_ACT_CUTOFF}%)')
    ax.set_xlabel('Residue Index', fontsize=11)
    ax.set_ylabel('Relative SASA (%)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1, len(resnums))
    ax.set_ylim(0, max(sasa_vals)+20 if sasa_vals else 100)

    red_p  = mpatches.Patch(color=color_act,   label='Active residue (HADDOCK AIR)')
    org_p  = mpatches.Patch(color=color_pass,  label='Passive residue')
    gray_p = mpatches.Patch(color='gray',       alpha=0.3, label='Relative SASA')
    ax.legend(handles=[red_p, org_p, gray_p], fontsize=8, loc='upper right')

ddg_A_map = dict(zip(res_ddg['resnum'].astype(int), res_ddg['ddg_mean']))
ddg_D_map = dict(zip(res_ddg[res_ddg['chain']=='D']['resnum'].astype(int),
                      res_ddg[res_ddg['chain']=='D']['ddg_mean']))

make_linear_map(axes[0], res_A, rSASA_A,
                active_A_nums, set(df_passive_A['resnum'].astype(int)) if len(df_passive_A) else set(),
                ddg_A_map,
                'Barnase (Chain A)\nAIR Residue Map with SASA Profile')

make_linear_map(axes[1], res_D, rSASA_D,
                active_D_nums, set(df_passive_D['resnum'].astype(int)) if len(df_passive_D) else set(),
                ddg_A_map,
                'Barstar (Chain D)\nAIR Residue Map with SASA Profile')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig9_air_map.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_air_map.png")

# ── FIGURE 10: DDG vs BSA correlation ────────────────────────────────────────
# For residues with both DDG and BSA data
bsa_A_df = bsa_df[bsa_df['chain']=='A'][['res_num','bsa']].rename(columns={'res_num':'resnum'})
bsa_D_df = bsa_df[bsa_df['chain']=='D'][['res_num','bsa']].rename(columns={'res_num':'resnum'})

# Merge DDG with BSA
ddg_bsa = res_ddg.copy()
ddg_bsa['resnum'] = ddg_bsa['resnum'].astype(int)
bsa_all = pd.concat([
    bsa_A_df.assign(chain='A'),
    bsa_D_df.assign(chain='D'),
])
merged = ddg_bsa.merge(bsa_all, on=['resnum','chain'], how='inner')
merged = merged[merged['bsa'] > 0]  # only interface residues
print(f"\nResidues with both DDG and BSA: {len(merged)}")

fig, ax = plt.subplots(figsize=(8, 6))
colors_ch = {'A': 'steelblue', 'D': 'darkorange'}
for ch, grp in merged.groupby('chain'):
    ax.scatter(grp['bsa'], grp['ddg_mean'],
               color=colors_ch[ch],
               label=f"{'Barnase' if ch=='A' else 'Barstar'} (n={len(grp)})",
               s=70, alpha=0.85, zorder=3)
    # Label points
    for _, row in grp.iterrows():
        ax.annotate(row['label'], (row['bsa'], row['ddg_mean']),
                    fontsize=7, xytext=(4, 2), textcoords='offset points')

# Linear regression on full merged
if len(merged) > 3:
    from scipy.stats import pearsonr, spearmanr
    r_p, pv_p = pearsonr(merged['bsa'], merged['ddg_mean'])
    r_s, pv_s = spearmanr(merged['bsa'], merged['ddg_mean'])
    xs = np.linspace(merged['bsa'].min(), merged['bsa'].max(), 100)
    from numpy.polynomial import polynomial as P
    coefs = np.polyfit(merged['bsa'], merged['ddg_mean'], 1)
    ax.plot(xs, np.polyval(coefs, xs), 'k--', linewidth=1.5,
            label=f'Linear fit (Pearson r={r_p:.2f}, p={pv_p:.3f})')
    ax.text(0.98, 0.05, f"Pearson r = {r_p:.2f}, p = {pv_p:.3f}\nSpearman ρ = {r_s:.2f}, p = {pv_s:.3f}",
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.axhline(HOTSPOT_CUTOFF, color='red', linestyle='--', linewidth=1, label='Hotspot threshold')
ax.set_xlabel('Buried Surface Area (Å²)', fontsize=12)
ax.set_ylabel('Mean ΔΔG (kcal/mol)', fontsize=12)
ax.set_title('Buried Surface Area vs ΔΔG\nBarnase-Barstar Interface Residues', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig10_bsa_vs_ddg.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_bsa_vs_ddg.png")

# ── FIGURE 11: Comprehensive AIR summary panel ────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

ax1 = fig.add_subplot(gs[0, :2])  # bar chart of DDG per residue
ax2 = fig.add_subplot(gs[0, 2])   # pie chart of AIR categories
ax3 = fig.add_subplot(gs[1, :])   # lollipop of selected active/passive

# Ax1: DDG per residue with AIR status
ddg_mean = res_ddg['ddg_mean'].values
labels_r = res_ddg['label'].values
col_r = []
for _, row in res_ddg.iterrows():
    rn, ch = int(row['resnum']), row['chain']
    if (ch == 'A' and rn in active_A_nums) or (ch == 'D' and rn in active_D_nums):
        col_r.append('red')
    elif ((ch == 'A' and rn in set(df_passive_A['resnum'].astype(int)) if len(df_passive_A) else False) or
          (ch == 'D' and rn in set(df_passive_D['resnum'].astype(int)) if len(df_passive_D) else False)):
        col_r.append('orange')
    else:
        col_r.append('steelblue')

ax1.bar(range(len(labels_r)), ddg_mean, color=col_r, edgecolor='white')
ax1.errorbar(range(len(labels_r)), ddg_mean, yerr=res_ddg['ddg_std'].fillna(0),
             fmt='none', color='black', capsize=3, linewidth=1)
ax1.axhline(HOTSPOT_CUTOFF, color='red', linestyle='--', linewidth=1.5)
ax1.set_xticks(range(len(labels_r)))
ax1.set_xticklabels(labels_r, rotation=90, fontsize=9)
ax1.set_xlabel('Residue', fontsize=10)
ax1.set_ylabel('Mean ΔΔG (kcal/mol)', fontsize=10)
ax1.set_title('ΔΔG per Residue with HADDOCK AIR Classification', fontsize=11)
for color, label in [('red','Active (hotspot)'), ('orange','Passive'), ('steelblue','Not selected')]:
    ax1.bar([], [], color=color, label=label)
ax1.legend(fontsize=8, loc='upper left')

# Ax2: Pie chart of mutation categories
all_ddg = df_brs['ddg'].dropna()
n_stab    = (all_ddg < 0).sum()
n_neutral = ((all_ddg >= 0) & (all_ddg < HOTSPOT_CUTOFF)).sum()
n_hot     = (all_ddg >= HOTSPOT_CUTOFF).sum()
ax2.pie([n_stab, n_neutral, n_hot],
        labels=[f'Stabilising\n(ΔΔG < 0)\nn={n_stab}',
                f'Moderate\n(0 ≤ ΔΔG < {HOTSPOT_CUTOFF})\nn={n_neutral}',
                f'Hotspot\n(ΔΔG ≥ {HOTSPOT_CUTOFF})\nn={n_hot}'],
        colors=['#2ecc71','steelblue','red'],
        autopct='%1.0f%%', startangle=90, textprops={'fontsize':9})
ax2.set_title('Mutation Categories\n(1BRS_A_D, all mutations)', fontsize=11)

# Ax3: Lollipop chart — barnase active residues
act_A_sorted = df_active_A.sort_values('ddg_mean', ascending=False)
act_D_sorted = df_active_D.sort_values('ddg_mean', ascending=False) if len(df_active_D) else pd.DataFrame()

all_active = pd.concat([
    act_A_sorted.assign(protein='Barnase'),
    act_D_sorted.assign(protein='Barstar') if len(act_D_sorted) else pd.DataFrame()
]).reset_index(drop=True)

c_map = {'Barnase': 'steelblue', 'Barstar': 'darkorange'}
for i, row in all_active.iterrows():
    c = c_map.get(row['protein'], 'gray')
    ax3.plot([0, row['ddg_mean']], [i, i], color=c, linewidth=2, alpha=0.7)
    ax3.scatter(row['ddg_mean'], i, color=c, s=80, zorder=3)
    ax3.text(row['ddg_mean']+0.1, i, f"{row['label']} ({row['rSASA']:.0f}% SA)",
             fontsize=9, va='center')

ax3.axvline(HOTSPOT_CUTOFF, color='red', linestyle='--', linewidth=1.5)
ax3.set_yticks(range(len(all_active)))
ax3.set_yticklabels(all_active['protein'].values, fontsize=9)
ax3.set_xlabel('Mean ΔΔG (kcal/mol)', fontsize=10)
ax3.set_title('HADDOCK Active Residues Selected from SKEMPI\n(annotated with relative SASA %)', fontsize=11)
for prot, col in c_map.items():
    ax3.plot([], [], color=col, linewidth=3, label=prot)
ax3.legend(fontsize=9)

fig.suptitle('HADDOCK AIR Design from Mutagenesis Data — Barnase-Barstar Complex',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(os.path.join(IMAGES, 'fig11_air_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_air_summary.png")

# ── save AIR summary ──────────────────────────────────────────────────────────
air_summary = {
    'active_barnase': df_active_A.to_dict('records'),
    'active_barstar': df_active_D.to_dict('records'),
    'passive_barnase': df_passive_A.to_dict('records') if len(df_passive_A) else [],
    'passive_barstar': df_passive_D.to_dict('records') if len(df_passive_D) else [],
    'n_air_pairs': int(len(df_active_A) * len(df_active_D)),
}
with open(os.path.join(OUTPUT, 'air_summary.json'), 'w') as f:
    json.dump(air_summary, f, indent=2, default=float)
print("\nSaved air_summary.json")
print(f"\nTotal HADDOCK AIR pairs: {air_summary['n_air_pairs']}")
print("\nDone — haddock_air_design.py")
