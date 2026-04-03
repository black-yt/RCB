"""
Structural analysis of the barnase-barstar complex (1brs_AD.pdb).
Computes interface residues, buried surface area, residue properties,
and secondary structure composition.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from Bio.PDB import PDBParser, DSSP, SASA
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_002_20260401_154822'
PDB_FILE  = os.path.join(WORKSPACE, 'data', '1brs_AD.pdb')
OUTPUT    = os.path.join(WORKSPACE, 'outputs')
IMAGES    = os.path.join(WORKSPACE, 'report', 'images')

# ── helpers ──────────────────────────────────────────────────────────────────
AA_PROPS = {
    'ALA': 'hydrophobic', 'VAL': 'hydrophobic', 'ILE': 'hydrophobic',
    'LEU': 'hydrophobic', 'MET': 'hydrophobic', 'PHE': 'hydrophobic',
    'TRP': 'hydrophobic', 'PRO': 'hydrophobic',
    'GLY': 'neutral',
    'SER': 'polar',      'THR': 'polar',      'CYS': 'polar',
    'TYR': 'polar',      'ASN': 'polar',      'GLN': 'polar',
    'LYS': 'positive',   'ARG': 'positive',   'HIS': 'positive',
    'ASP': 'negative',   'GLU': 'negative',
}

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V',
}

def atom_coords(residue):
    return np.array([a.get_vector().get_array() for a in residue.get_atoms()
                     if a.element != 'H'])

def min_distance(res_a, res_b):
    ca = atom_coords(res_a)
    cb = atom_coords(res_b)
    if len(ca) == 0 or len(cb) == 0:
        return np.inf
    diff = ca[:, None, :] - cb[None, :, :]
    return np.sqrt((diff**2).sum(axis=2)).min()

# ── parse structure ───────────────────────────────────────────────────────────
parser = PDBParser(QUIET=True)
structure = parser.get_structure('1BRS', PDB_FILE)
model = structure[0]

chain_A = model['A']  # barnase
chain_D = model['D']  # barstar

res_A = [r for r in chain_A if r.get_resname() in AA3TO1]
res_D = [r for r in chain_D if r.get_resname() in AA3TO1]
print(f"Chain A (barnase): {len(res_A)} residues")
print(f"Chain D (barstar): {len(res_D)} residues")

# ── interface detection (5 Å cutoff) ─────────────────────────────────────────
CUTOFF = 5.0
interface_A, interface_D = [], []
dist_matrix = np.zeros((len(res_A), len(res_D)))

for i, ra in enumerate(res_A):
    for j, rd in enumerate(res_D):
        d = min_distance(ra, rd)
        dist_matrix[i, j] = d
        if d <= CUTOFF:
            if ra not in interface_A:
                interface_A.append(ra)
            if rd not in interface_D:
                interface_D.append(rd)

print(f"\nInterface residues (≤{CUTOFF}Å):")
print(f"  Barnase (A): {len(interface_A)}")
print(f"  Barstar (D): {len(interface_D)}")

def res_label(r):
    return f"{AA3TO1.get(r.get_resname(),'?')}{r.get_id()[1]}"

print("\nBarnase interface residues:", [res_label(r) for r in interface_A])
print("Barstar interface residues:", [res_label(r) for r in interface_D])

# ── buried surface area (approximate using SASA) ─────────────────────────────
# Use Bio.PDB SASA (Shrake-Rupley)
from Bio.PDB.SASA import ShrakeRupley
sr = ShrakeRupley()

# Complex SASA
sr.compute(model, level='R')
sasa_complex = {}
for chain_id in ['A', 'D']:
    for res in model[chain_id]:
        if res.get_resname() in AA3TO1:
            sasa_complex[(chain_id, res.get_id()[1])] = res.sasa

# Individual chain SASA (for BSA)
model_A_only = structure[0].copy()
# We'll compute per-residue SASA using atoms directly
# Simple approach: compute SASA on sub-structures
from Bio.PDB import Structure, Model, Chain
from copy import deepcopy

struct_A = Structure.Structure('A')
model_A  = Model.Model(0)
chain_Ac = deepcopy(chain_A)
model_A.add(chain_Ac)
struct_A.add(model_A)

struct_D = Structure.Structure('D')
model_D  = Model.Model(0)
chain_Dc = deepcopy(chain_D)
model_D.add(chain_Dc)
struct_D.add(model_D)

sr.compute(struct_A[0], level='R')
sasa_A_alone = {}
for res in struct_A[0]['A']:
    if res.get_resname() in AA3TO1:
        sasa_A_alone[res.get_id()[1]] = res.sasa

sr.compute(struct_D[0], level='R')
sasa_D_alone = {}
for res in struct_D[0]['D']:
    if res.get_resname() in AA3TO1:
        sasa_D_alone[res.get_id()[1]] = res.sasa

# BSA per residue
bsa_A = {}
for res in res_A:
    rid = res.get_id()[1]
    s_alone   = sasa_A_alone.get(rid, 0)
    s_complex = sasa_complex.get(('A', rid), 0)
    bsa_A[rid] = max(0, s_alone - s_complex)

bsa_D = {}
for res in res_D:
    rid = res.get_id()[1]
    s_alone   = sasa_D_alone.get(rid, 0)
    s_complex = sasa_complex.get(('D', rid), 0)
    bsa_D[rid] = max(0, s_alone - s_complex)

total_bsa_A = sum(bsa_A.values())
total_bsa_D = sum(bsa_D.values())
total_bsa   = (total_bsa_A + total_bsa_D) / 2  # average
print(f"\nBuried Surface Area:")
print(f"  Barnase contribution: {total_bsa_A:.1f} Å²")
print(f"  Barstar contribution: {total_bsa_D:.1f} Å²")
print(f"  Average BSA: {total_bsa:.1f} Å²")

# ── residue property analysis ─────────────────────────────────────────────────
def count_props(res_list):
    props = [AA_PROPS.get(r.get_resname(), 'neutral') for r in res_list]
    from collections import Counter
    return Counter(props)

props_A_all  = count_props(res_A)
props_D_all  = count_props(res_D)
props_A_iface = count_props(interface_A)
props_D_iface = count_props(interface_D)

print("\nResidue properties - all vs interface:")
print("Barnase all:", dict(props_A_all))
print("Barnase iface:", dict(props_A_iface))
print("Barstar all:", dict(props_D_all))
print("Barstar iface:", dict(props_D_iface))

# ── save interface residue table ───────────────────────────────────────────────
records = []
for r in interface_A:
    rid = r.get_id()[1]
    records.append({
        'chain':'A','protein':'barnase',
        'res_num': rid,
        'res_name': r.get_resname(),
        'res_1letter': AA3TO1.get(r.get_resname(),'?'),
        'property': AA_PROPS.get(r.get_resname(),'neutral'),
        'bsa': bsa_A.get(rid, 0),
        'sasa_complex': sasa_complex.get(('A',rid),0),
        'sasa_alone': sasa_A_alone.get(rid,0),
    })
for r in interface_D:
    rid = r.get_id()[1]
    records.append({
        'chain':'D','protein':'barstar',
        'res_num': rid,
        'res_name': r.get_resname(),
        'res_1letter': AA3TO1.get(r.get_resname(),'?'),
        'property': AA_PROPS.get(r.get_resname(),'neutral'),
        'bsa': bsa_D.get(rid, 0),
        'sasa_complex': sasa_complex.get(('D',rid),0),
        'sasa_alone': sasa_D_alone.get(rid,0),
    })

df_iface = pd.DataFrame(records)
df_iface.to_csv(os.path.join(OUTPUT, 'interface_residues.csv'), index=False)
print(f"\nSaved interface residues table ({len(df_iface)} rows)")

# ── save BSA per residue ────────────────────────────────────────────────────
bsa_records = []
for r in res_A:
    rid = r.get_id()[1]
    bsa_records.append({'chain':'A','res_num':rid,'res_name':r.get_resname(),
                         'bsa':bsa_A.get(rid,0),'sasa_alone':sasa_A_alone.get(rid,0)})
for r in res_D:
    rid = r.get_id()[1]
    bsa_records.append({'chain':'D','res_num':rid,'res_name':r.get_resname(),
                         'bsa':bsa_D.get(rid,0),'sasa_alone':sasa_D_alone.get(rid,0)})
df_bsa = pd.DataFrame(bsa_records)
df_bsa.to_csv(os.path.join(OUTPUT, 'bsa_per_residue.csv'), index=False)

# ── secondary structure composition ──────────────────────────────────────────
# Use DSSP via Biopython
try:
    dssp = DSSP(model, PDB_FILE, dssp='mkdssp')
    ss_dict = {}
    for key in dssp.property_dict:
        chain_id, res_id = key
        ss = dssp[key][2]  # SS assignment
        ss_dict[(chain_id, res_id[1])] = ss
    print(f"\nDSSP computed for {len(ss_dict)} residues")
    dssp_available = True
except Exception as e:
    print(f"DSSP not available: {e}")
    dssp_available = False

# ── Figure 1: Interface overview - BSA per residue ────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for ax, (chain_id, prot_name, res_list, bsa_dict, color) in zip(
        axes, [('A','Barnase',res_A,bsa_A,'steelblue'),
               ('D','Barstar',res_D,bsa_D,'darkorange')]):
    res_nums = [r.get_id()[1] for r in res_list]
    bsa_vals = [bsa_dict.get(n, 0) for n in res_nums]
    colors_bar = ['red' if bsa_dict.get(r.get_id()[1],0) > 0 else 'lightgray'
                  for r in res_list]
    ax.bar(range(len(res_nums)), bsa_vals, color=colors_bar, width=1.0, linewidth=0)
    ax.set_xlabel('Residue Index', fontsize=11)
    ax.set_ylabel('Buried Surface Area (Å²)', fontsize=11)
    ax.set_title(f'{prot_name} (Chain {chain_id}) — Per-Residue Buried Surface Area\n'
                 f'Interface residues highlighted in red (≤5Å cutoff)',
                 fontsize=12)
    ax.axhline(0, color='black', linewidth=0.5)
    # Label interface residues with large BSA
    for i, r in enumerate(res_list):
        bsa_val = bsa_dict.get(r.get_id()[1], 0)
        if bsa_val > 30:
            ax.text(i, bsa_val + 1, res_label(r), fontsize=7, ha='center',
                    rotation=90, va='bottom', color='darkred')

red_patch   = mpatches.Patch(color='red',      label='Interface residue (BSA > 0)')
gray_patch  = mpatches.Patch(color='lightgray', label='Non-interface residue')
ax.legend(handles=[red_patch, gray_patch], fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig1_bsa_per_residue.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_bsa_per_residue.png")

# ── Figure 2: Residue property composition ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
prop_order = ['hydrophobic', 'polar', 'positive', 'negative', 'neutral']
colors_prop = ['#e67e22','#2ecc71','#3498db','#e74c3c','#95a5a6']

for ax, (prot_name, props_all, props_if) in zip(
        axes,
        [('Barnase (Chain A)', props_A_all, props_A_iface),
         ('Barstar (Chain D)', props_D_all, props_D_iface)]):
    x = np.arange(len(prop_order))
    w = 0.35
    all_vals = [props_all.get(p, 0) for p in prop_order]
    if_vals  = [props_if.get(p, 0) for p in prop_order]
    all_pct  = [v/sum(all_vals)*100 for v in all_vals]
    if_pct   = [v/sum(if_vals)*100  for v in if_vals] if sum(if_vals)>0 else [0]*len(prop_order)

    bars1 = ax.bar(x - w/2, all_pct, w, label='All residues', color=colors_prop, alpha=0.6)
    bars2 = ax.bar(x + w/2, if_pct,  w, label='Interface',    color=colors_prop, alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in prop_order], rotation=30, ha='right')
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'{prot_name}\nResidue Type Composition', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 60)
    # Add value labels
    for bar in bars2:
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.0f}%', ha='center',
                    va='bottom', fontsize=8)

plt.suptitle('Residue Physicochemical Properties: Full Protein vs Interface',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig2_residue_properties.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_residue_properties.png")

# ── Figure 3: Distance heatmap (interface) ─────────────────────────────────
# Select only interfacing residues
iface_idx_A = [i for i, r in enumerate(res_A) if r in interface_A]
iface_idx_D = [j for j, r in enumerate(res_D) if r in interface_D]
sub_dist = dist_matrix[np.ix_(iface_idx_A, iface_idx_D)]
labels_A = [res_label(res_A[i]) for i in iface_idx_A]
labels_D = [res_label(res_D[j]) for j in iface_idx_D]

fig, ax = plt.subplots(figsize=(max(8, len(labels_D)*0.4),
                                max(6, len(labels_A)*0.4)))
im = ax.imshow(sub_dist, cmap='RdYlGn_r', vmin=0, vmax=10, aspect='auto')
ax.set_xticks(range(len(labels_D)))
ax.set_xticklabels(labels_D, rotation=90, fontsize=8)
ax.set_yticks(range(len(labels_A)))
ax.set_yticklabels(labels_A, fontsize=8)
ax.set_xlabel('Barstar (Chain D) residues', fontsize=11)
ax.set_ylabel('Barnase (Chain A) residues', fontsize=11)
ax.set_title('Minimum Inter-Residue Distances at the Barnase-Barstar Interface\n'
             '(Å; only residues within 5 Å of partner shown)', fontsize=12)
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Distance (Å)', fontsize=10)
# Mark contacts < 5Å
for i in range(len(labels_A)):
    for j in range(len(labels_D)):
        if sub_dist[i,j] <= 5.0:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                        edgecolor='black', linewidth=0.8))
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig3_interface_distance_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_interface_distance_heatmap.png")

# ── summary statistics ────────────────────────────────────────────────────────
summary = {
    'barnase_total_residues': len(res_A),
    'barstar_total_residues': len(res_D),
    'barnase_interface_residues': len(interface_A),
    'barstar_interface_residues': len(interface_D),
    'total_bsa_A': round(total_bsa_A, 1),
    'total_bsa_D': round(total_bsa_D, 1),
    'avg_bsa': round(total_bsa, 1),
    'interface_cutoff_A': CUTOFF,
    'barnase_iface_res_labels': [res_label(r) for r in interface_A],
    'barstar_iface_res_labels': [res_label(r) for r in interface_D],
}
import json
with open(os.path.join(OUTPUT, 'structural_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved structural_summary.json")
print("\nDone — structural_analysis.py")
