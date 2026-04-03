
import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PDB = os.path.join(BASE, 'data', '1brs_AD.pdb')
DATA_SKEMPI = os.path.join(BASE, 'data', 'skempi_v2.csv')
OUT_DIR = os.path.join(BASE, 'outputs')
FIG_DIR = os.path.join(BASE, 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set(style='whitegrid', context='talk')

# 1. Parse PDB and compute interface contacts between chains A and D
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

parser = PDBParser(QUIET=True)
structure = parser.get_structure('1brs', DATA_PDB)
model = structure[0]

chainA = model['A']
chainD = model['D']

atomsA = [atom for atom in chainA.get_atoms() if atom.element != 'H']
atomsD = [atom for atom in chainD.get_atoms() if atom.element != 'H']

ns = NeighborSearch(list(model.get_atoms()))
cutoff = 5.0
contacts = []
for atomA in atomsA:
    for atomB in ns.search(atomA.coord, cutoff, level='A'):
        if atomB.get_parent().get_parent().id != 'D':
            continue
        resA = atomA.get_parent()
        resB = atomB.get_parent()
        if not (is_aa(resA) and is_aa(resB)):
            continue
        dist = atomA - atomB
        contacts.append({
            'chainA_resid': resA.get_id()[1],
            'chainA_resname': resA.get_resname(),
            'atomA': atomA.get_name(),
            'chainD_resid': resB.get_id()[1],
            'chainD_resname': resB.get_resname(),
            'atomB': atomB.get_name(),
            'distance': dist
        })

contacts_df = pd.DataFrame(contacts).drop_duplicates()
contacts_path = os.path.join(OUT_DIR, 'barnase_barstar_interface_contacts.csv')
contacts_df.to_csv(contacts_path, index=False)

# Interface residue-level summary
interfaceA = contacts_df[['chainA_resid','chainA_resname']].drop_duplicates()
interfaceD = contacts_df[['chainD_resid','chainD_resname']].drop_duplicates()
interfaceA['chain'] = 'A'
interfaceD['chain'] = 'D'
interfaceA.rename(columns={'chainA_resid':'resid','chainA_resname':'resname'}, inplace=True)
interfaceD.rename(columns={'chainD_resid':'resid','chainD_resname':'resname'}, inplace=True)
interface_df = pd.concat([interfaceA, interfaceD], ignore_index=True)
interface_df.to_csv(os.path.join(OUT_DIR, 'interface_residues.csv'), index=False)

# Plot distance distribution
plt.figure(figsize=(7,5))
sns.histplot(contacts_df['distance'], bins=30, kde=True, color='tab:blue')
plt.xlabel('Interatomic distance (Å)')
plt.ylabel('Count')
plt.title('Interface Interatomic Distance Distribution')
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, 'distance_distribution.png')
plt.savefig(fig1_path, dpi=300)
plt.close()

# Contact map heatmap (residue-level)
contact_counts = contacts_df.groupby(['chainA_resid','chainD_resid']).size().unstack(fill_value=0)
plt.figure(figsize=(10,8))
sns.heatmap(contact_counts, cmap='viridis')
plt.xlabel('Barstar residue (chain D)')
plt.ylabel('Barnase residue (chain A)')
plt.title('Residue-level Contact Map (Atom Count within 5 Å)')
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'contact_map.png')
plt.savefig(fig2_path, dpi=300)
plt.close()

# 2. Analyze SKEMPI data focusing on barnase-barstar mutations where possible
skempi = pd.read_csv(DATA_SKEMPI, sep=';')

# Basic cleaning
skempi = skempi.rename(columns={c: c.strip() for c in skempi.columns})

# Infer complex names column
complex_col = None
for cand in ['complex','PDB','pdbid','#Pdb']:
    if cand in skempi.columns:
        complex_col = cand
        break

# Overview plots: overall dG changes
if 'ddG' in skempi.columns:
    ddg_col = 'ddG'
else:
    # guess a column containing ddG
    ddg_candidates = [c for c in skempi.columns if 'ddg' in c.lower()]
    ddg_col = ddg_candidates[0] if ddg_candidates else None

numeric_ddg = skempi[ddg_col].astype(float) if ddg_col is not None else None

if numeric_ddg is not None:
    plt.figure(figsize=(7,5))
    sns.histplot(numeric_ddg, bins=50, kde=True, color='tab:orange')
    plt.xlabel(r'$\Delta\Delta G$ (kcal/mol)')
    plt.ylabel('Count')
    plt.title('Distribution of Binding Free Energy Changes (All SKEMPI 2.0)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'skempi_ddg_distribution.png'), dpi=300)
    plt.close()

# Focus on barnase-barstar (if present)
if complex_col is not None:
    mask_barnase = skempi[complex_col].astype(str).str.contains('1BRS', case=False, na=False)
    skempi_brs = skempi[mask_barnase].copy()
else:
    skempi_brs = pd.DataFrame()

skempi_brs.to_csv(os.path.join(OUT_DIR, 'skempi_barnase_barstar_subset.csv'), index=False)

if not skempi_brs.empty and ddg_col is not None:
    numeric_ddg_brs = skempi_brs[ddg_col].astype(float)
    plt.figure(figsize=(7,5))
    sns.histplot(numeric_ddg_brs, bins=20, kde=True, color='tab:green')
    plt.xlabel(r'$\Delta\Delta G$ (kcal/mol)')
    plt.ylabel('Count')
    plt.title('Barnase-Barstar Binding Free Energy Changes (SKEMPI 2.0)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'barnase_barstar_ddg_distribution.png'), dpi=300)
    plt.close()

# Map mutations to interface vs non-interface (if residue info available)
# We have interface residues from the PDB; use them to classify mutations
interface_resids = interface_df.copy()
interface_resids['key'] = interface_resids['chain'] + interface_resids['resid'].astype(str)
interface_keys = set(interface_resids['key'])

mut_chain_col = None
mut_pos_col = None
for c in skempi.columns:
    lc = c.lower()
    if 'chain_mut' in lc or 'mut_chain' in lc:
        mut_chain_col = c
    if ('mutation' in lc or 'pos' in lc) and mut_pos_col is None:
        mut_pos_col = c

if mut_chain_col is not None and mut_pos_col is not None and ddg_col is not None:
    def classify_row(row):
        try:
            chain = str(row[mut_chain_col]).strip()
            pos = int(''.join(ch for ch in str(row[mut_pos_col]) if ch.isdigit()))
            key = f"{chain}{pos}"
            return 'interface' if key in interface_keys else 'non-interface'
        except Exception:
            return 'unknown'
    skempi['interface_class'] = skempi.apply(classify_row, axis=1)
    mask_valid = (skempi['interface_class'] != 'unknown') & skempi[ddg_col].notna()
    comp_df = skempi.loc[mask_valid, ['interface_class', ddg_col]].copy()
    comp_df[ddg_col] = comp_df[ddg_col].astype(float)
    plt.figure(figsize=(7,5))
    sns.boxplot(data=comp_df, x='interface_class', y=ddg_col, order=['interface','non-interface'])
    plt.xlabel('Mutation location')
    plt.ylabel(r'$\Delta\Delta G$ (kcal/mol)')
    plt.title('Effect of Interface vs Non-interface Mutations on Binding Affinity')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'interface_vs_noninterface_ddg.png'), dpi=300)
    plt.close()

print('Analysis complete.')
