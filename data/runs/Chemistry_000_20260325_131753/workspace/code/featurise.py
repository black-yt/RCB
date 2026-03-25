"""
Molecular Graph Featurisation + Dataset Pipeline
================================================
Converts SMILES strings into PyG Data objects with:
  - Atom features: atomic_num, degree, formal_charge, hybridisation,
                   aromaticity, H count, ring membership → 12-dim
  - Bond features: bond type, ring, conjugation, stereo → 6-dim
  - Non-covalent "virtual" edges: distance-based hydrogen-bond donor/acceptor pairs
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.rdchem import (
    BondType as BT, HybridizationType as HT, ChiralType as CT
)


# ─────────────────────────────────────────────────────────────────────────────
# Atom featuriser
# ─────────────────────────────────────────────────────────────────────────────

ATOM_TYPES = list(range(1, 120))  # atomic number 1-119
DEGREE_BINS = list(range(11))
FORMAL_CHARGE_BINS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
HYBRIDISATION = [HT.S, HT.SP, HT.SP2, HT.SP3, HT.SP3D, HT.SP3D2, HT.OTHER]


def one_hot(value, categories):
    """Return a one-hot list (with 'unknown' bucket at end)."""
    vec = [0] * (len(categories) + 1)
    if value in categories:
        vec[categories.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom) -> np.ndarray:
    feat = []
    feat += one_hot(atom.GetAtomicNum(), list(range(1, 53)))   # 53 dim (1-52 + unk)
    feat += one_hot(atom.GetDegree(), DEGREE_BINS)              # 11 dim
    feat += one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_BINS) # 12 dim
    feat += one_hot(atom.GetHybridization(), HYBRIDISATION)     # 8 dim
    feat += [
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetMass() / 100.0,
    ]  # 4 dim
    return np.array(feat, dtype=np.float32)   # total: 88 dim


# ─────────────────────────────────────────────────────────────────────────────
# Bond featuriser
# ─────────────────────────────────────────────────────────────────────────────

BOND_TYPES = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
STEREO_TYPES = list(range(6))


def bond_features(bond) -> np.ndarray:
    feat = []
    feat += one_hot(bond.GetBondType(), BOND_TYPES)         # 5 dim
    feat += one_hot(int(bond.GetStereo()), STEREO_TYPES)    # 7 dim
    feat += [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]  # 2 dim
    return np.array(feat, dtype=np.float32)   # total: 14 dim


# ─────────────────────────────────────────────────────────────────────────────
# Virtual non-covalent edge features (all zeros with extra flag=1)
# ─────────────────────────────────────────────────────────────────────────────

def virtual_edge_features(n_bond_feat: int) -> np.ndarray:
    feat = np.zeros(n_bond_feat, dtype=np.float32)
    feat[-1] = 1.0   # mark as non-covalent
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# SMILES → PyG Data
# ─────────────────────────────────────────────────────────────────────────────

# HBA / HBD SMARTS patterns (simple subset)
HBD_SMARTS = Chem.MolFromSmarts('[#7H,#8H]')
HBA_SMARTS = Chem.MolFromSmarts('[#7,#8;!$([#7,#8]~[#7,#8]);!$([#7,#8]~[#6]=O)]')


def smiles_to_graph(smi: str, y=None, add_virtual: bool = True) -> Data | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # ── Atom features ──────────────────────────────────────────────────────
    atom_feat_list = [atom_features(a) for a in mol.GetAtoms()]
    n_atoms = len(atom_feat_list)
    if n_atoms == 0:
        return None
    x = torch.tensor(np.stack(atom_feat_list), dtype=torch.float)   # (N, 88)

    # ── Covalent bond features ─────────────────────────────────────────────
    src_list, dst_list, edge_feat_list = [], [], []
    n_bond_feat = None
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        if n_bond_feat is None:
            n_bond_feat = len(bf) + 1  # +1 for non-covalent flag
        bf_ext = np.append(bf, 0.0)    # covalent flag = 0
        src_list += [i, j]
        dst_list += [j, i]
        edge_feat_list += [bf_ext, bf_ext]

    if n_bond_feat is None:
        n_bond_feat = 15   # fallback

    # Handle isolated atoms (no bonds) – self-loop with zero edge features
    if len(src_list) == 0:
        src_list = [0]; dst_list = [0]
        edge_feat_list = [np.zeros(n_bond_feat, dtype=np.float32)]

    # ── Non-covalent (virtual) edges: HBD→HBA pairs ────────────────────────
    if add_virtual:
        try:
            hbd_matches = mol.GetSubstructMatches(HBD_SMARTS)
            hba_matches = mol.GetSubstructMatches(HBA_SMARTS)
            hbd_idx = [m[0] for m in hbd_matches]
            hba_idx = [m[0] for m in hba_matches]
            for d in hbd_idx:
                for a in hba_idx:
                    if d != a:
                        vf = virtual_edge_features(n_bond_feat)
                        src_list += [d, a]
                        dst_list += [a, d]
                        edge_feat_list += [vf, vf]
        except Exception:
            pass

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(np.stack(edge_feat_list), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str, smiles_col: str, label_cols: list[str], max_mol: int | None = None) -> list[Data]:
    """Load a CSV → list of PyG Data objects. Drops invalid SMILES."""
    df = pd.read_csv(csv_path)
    if max_mol:
        df = df.head(max_mol)
    graphs = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        if not isinstance(smi, str):
            continue
        labels = row[label_cols].values.astype(np.float32)
        g = smiles_to_graph(smi, y=labels)
        if g is not None:
            graphs.append(g)
    return graphs


# Dataset specifications
DATASET_SPECS = {
    'bace': {
        'path': 'data/bace.csv',
        'smiles_col': 'smiles',
        'label_cols': ['label'],
        'n_tasks': 1,
        'task_type': 'binary',
    },
    'bbbp': {
        'path': 'data/bbbp.csv',
        'smiles_col': 'smiles',
        'label_cols': ['label'],
        'n_tasks': 1,
        'task_type': 'binary',
    },
    'clintox': {
        'path': 'data/clintox.csv',
        'smiles_col': 'smiles',
        'label_cols': ['FDA_APPROVED', 'CT_TOX'],
        'n_tasks': 2,
        'task_type': 'binary',
    },
    'hiv': {
        'path': 'data/hiv.csv',
        'smiles_col': 'smiles',
        'label_cols': ['label'],
        'n_tasks': 1,
        'task_type': 'binary',
        'max_mol': 5000,     # subset for speed
    },
    'muv': {
        'path': 'data/muv.csv',
        'smiles_col': 'smiles',
        'label_cols': [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644',
            'MUV-652', 'MUV-689', 'MUV-692', 'MUV-712',
            'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858',
            'MUV-859',
        ],
        'n_tasks': 17,
        'task_type': 'binary',
        'max_mol': 5000,
    },
}
