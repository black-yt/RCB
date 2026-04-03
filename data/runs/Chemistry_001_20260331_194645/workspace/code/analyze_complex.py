#!/usr/bin/env python
import os, math, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem

from Bio.PDB import PDBParser

BASE = Path('.').resolve()
DATA_PROT = BASE / 'data/sample/2l3r/2l3r_protein.pdb'
DATA_LIG = BASE / 'data/sample/2l3r/2l3r_ligand.sdf'
OUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report/images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


def load_protein_ca(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', str(pdb_path))
    cas = []
    res_ids = []
    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' in res:
                    ca = res['CA']
                    cas.append(ca.coord)
                    res_ids.append((chain.id, res.id[1], res.get_resname()))
    coords = np.array(cas, dtype=float)
    return coords, res_ids


def load_ligand_coords(sdf_path):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = suppl[0]
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    atomic_nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return mol, coords, atomic_nums


def protein_stats(ca_coords, res_ids):
    # pairwise distance matrix
    dists = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
    np.save(OUT_DIR / 'protein_ca_coords.npy', ca_coords)
    np.save(OUT_DIR / 'protein_ca_dists.npy', dists)

    # distance histogram
    plt.figure(figsize=(4,3))
    tri = dists[np.triu_indices_from(dists, k=1)]
    sns.histplot(tri, bins=40, kde=True)
    plt.xlabel('CA-CA distance (Å)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'protein_ca_distance_hist.png', dpi=200)
    plt.close()

    # contact map
    plt.figure(figsize=(4,4))
    sns.heatmap(dists, cmap='viridis', cbar_kws={'label': 'Å'})
    plt.title('Protein CA-CA distance matrix')
    plt.xlabel('Residue index')
    plt.ylabel('Residue index')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'protein_ca_distance_matrix.png', dpi=200)
    plt.close()


def ligand_stats(lig_coords, atomic_nums):
    np.save(OUT_DIR / 'ligand_coords.npy', lig_coords)
    np.save(OUT_DIR / 'ligand_atomic_nums.npy', atomic_nums)

    # BB radius of gyration
    center = lig_coords.mean(axis=0)
    rg = math.sqrt(((lig_coords - center)**2).sum(axis=1).mean())

    # element composition
    unique, counts = np.unique(atomic_nums, return_counts=True)
    comp = {int(z): int(c) for z, c in zip(unique, counts)}
    with open(OUT_DIR / 'ligand_composition.json', 'w') as f:
        json.dump({'radius_gyration': rg, 'composition': comp}, f, indent=2)

    # 2D scatter (first two PCA components via SVD)
    X = lig_coords - lig_coords.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = X @ Vt[:2].T
    plt.figure(figsize=(4,4))
    sc = plt.scatter(pcs[:,0], pcs[:,1], c=atomic_nums, cmap='tab20', s=10)
    plt.colorbar(sc, label='Atomic number')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Ligand atomic coordinates (PCA)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ligand_pca_scatter.png', dpi=200)
    plt.close()


def protein_ligand_contacts(ca_coords, lig_coords, cutoff=6.0):
    # simple nearest distances between ligand atoms and each CA
    diff = ca_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # [N_res, N_atoms]
    min_per_res = dists.min(axis=1)
    np.save(OUT_DIR / 'protein_ligand_min_dists.npy', min_per_res)

    plt.figure(figsize=(5,3))
    plt.plot(min_per_res)
    plt.axhline(cutoff, color='r', linestyle='--', label=f'{cutoff} Å')
    plt.xlabel('Residue index (with CA)')
    plt.ylabel('Min distance to ligand (Å)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'protein_ligand_min_distance_profile.png', dpi=200)
    plt.close()


# Minimal diffusion-like process on ligand conditioned on fixed protein

def diffusion_demo(lig_coords, n_steps=50, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    traj = [lig_coords.copy()]
    x = lig_coords.copy()
    for t in range(1, n_steps+1):
        noise = rng.normal(scale=sigma, size=x.shape)
        # simple drift back towards original coords to mimic denoising
        alpha = t / n_steps
        drift = (lig_coords - x) * alpha
        x = x + noise + drift
        traj.append(x.copy())
    traj = np.stack(traj, axis=0)
    np.save(OUT_DIR / 'ligand_diffusion_traj.npy', traj)

    # visualize RMSD to target over time
    diffs = traj - lig_coords[None, :, :]
    rmsd = np.sqrt((diffs**2).sum(axis=(1,2)) / lig_coords.shape[0])
    plt.figure(figsize=(4,3))
    plt.plot(rmsd)
    plt.xlabel('Step')
    plt.ylabel('RMSD to reference (Å)')
    plt.title('Toy diffusion denoising trajectory')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ligand_diffusion_rmsd.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    ca_coords, res_ids = load_protein_ca(DATA_PROT)
    mol, lig_coords, atomic_nums = load_ligand_coords(DATA_LIG)
    protein_stats(ca_coords, res_ids)
    ligand_stats(lig_coords, atomic_nums)
    protein_ligand_contacts(ca_coords, lig_coords)
    diffusion_demo(lig_coords)
