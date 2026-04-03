"""
Protein Complex Structural Alignment Analysis
==============================================
Implements TM-score-based structural alignment between protein complexes 7xg4 and 6n40,
following the methodology of Foldseek-Multimer / US-align for complex alignment.

References:
- van Kempen et al. (2024) Foldseek. Nature Biotechnology 42:243-246
- Zhang et al. (2022) US-align. Nature Methods 19:1109-1115
- Zhang & Skolnick (2005) TM-align. Nucleic Acids Research 33:2302-2309
"""

import numpy as np
from Bio.PDB import PDBParser, Selection, PPBuilder
from Bio.PDB.vectors import Vector
import warnings
import json
import os

warnings.filterwarnings('ignore')

DATA_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_002_20260401_200700/data"
OUT_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_002_20260401_200700/outputs"

# ─── TM-score core implementation ────────────────────────────────────────────

def get_ca_coords(chain):
    """Extract Cα coordinates from a chain (protein only)."""
    coords = []
    residue_ids = []
    for residue in chain.get_residues():
        if residue.id[0] == ' ' and 'CA' in residue:  # standard amino acid
            coords.append(residue['CA'].get_coord())
            residue_ids.append(residue.id[1])
    return np.array(coords), residue_ids


def d0(L):
    """TM-score normalization distance parameter."""
    if L < 15:
        return 0.5
    return 1.24 * (L - 15) ** (1/3) - 1.8


def tm_score_from_coords(coords1, coords2, L_target):
    """
    Compute TM-score given paired Cα coordinates.
    TM-score = (1/L_target) * sum_i [1 / (1 + (d_i/d0(L_target))^2)]
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0
    d0_val = d0(L_target)
    dists = np.linalg.norm(coords1 - coords2, axis=1)
    scores = 1.0 / (1.0 + (dists / d0_val) ** 2)
    return np.sum(scores) / L_target


def kabsch_rotation(P, Q):
    """
    Compute optimal rotation matrix R to superimpose P onto Q using Kabsch algorithm.
    Returns rotation matrix R and translation vectors.
    """
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t, centroid_P, centroid_Q


def apply_rotation(coords, R, t):
    """Apply rotation R and translation t to coordinates."""
    return (R @ coords.T).T + t


def tm_align_pair(coords1, coords2):
    """
    Simplified TM-align between two sets of Cα coordinates.
    Iterative alignment using TM-score maximization.
    Returns: TM-score (normalized by shorter seq), RMSD, rotation matrix, translation.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0, 999.0, np.eye(3), np.zeros(3)

    n1, n2 = len(coords1), len(coords2)
    L_min = min(n1, n2)
    L_max = max(n1, n2)
    d0_val = d0(L_min)

    # Initial alignment: use all paired residues (up to L_min)
    p1 = coords1[:L_min].copy()
    p2 = coords2[:L_min].copy()

    best_tm = 0.0
    best_R = np.eye(3)
    best_t = np.zeros(3)

    for iteration in range(20):
        if len(p1) < 3:
            break
        R, t, _, _ = kabsch_rotation(p1, p2)
        coords1_rot = apply_rotation(coords1, R, t)

        # Compute distances to all residues in coords2 (greedy assignment for shorter)
        if n1 <= n2:
            dists = np.linalg.norm(coords1_rot[:, None, :] - coords2[None, :, :], axis=2)
        else:
            dists = np.linalg.norm(coords2[:, None, :] - coords1_rot[None, :, :], axis=2)

        # For each residue in shorter, find best match
        min_idx = np.argmin(dists, axis=1)

        # TM-score with current rotation
        d_pairs = np.linalg.norm(coords1_rot[:L_min] - coords2[:L_min], axis=1)
        tm = np.sum(1.0 / (1.0 + (d_pairs / d0_val) ** 2)) / L_min

        if tm > best_tm:
            best_tm = tm
            best_R = R
            best_t = t

        # Select well-aligned residues for next iteration
        threshold = d0_val * 1.5 * (1 + iteration * 0.1)
        mask = d_pairs < threshold
        if mask.sum() < 3:
            mask = d_pairs < np.percentile(d_pairs, 50)
        if mask.sum() < 3:
            break

        p1 = coords1[:L_min][mask]
        p2 = coords2[:L_min][mask]

    # Final RMSD with best rotation
    coords1_rot = apply_rotation(coords1[:L_min], best_R, best_t)
    rmsd = np.sqrt(np.mean(np.sum((coords1_rot - coords2[:L_min])**2, axis=1)))

    return best_tm, rmsd, best_R, best_t


def pairwise_chain_tm_scores(chains1, chains2):
    """Compute all pairwise TM-scores between chains from two complexes."""
    results = {}
    for cid1, (coords1, _) in chains1.items():
        results[cid1] = {}
        for cid2, (coords2, _) in chains2.items():
            if len(coords1) < 5 or len(coords2) < 5:
                results[cid1][cid2] = (0.0, 999.0)
                continue
            tm, rmsd, _, _ = tm_align_pair(coords1, coords2)
            results[cid1][cid2] = (tm, rmsd)
    return results


def greedy_chain_assignment(tm_matrix, chain_ids1, chain_ids2):
    """
    Greedy assignment of chains from complex1 to complex2 based on TM-scores.
    Similar to the approach in US-align and QSalign.
    """
    # Build flat list of (tm, id1, id2)
    pairs = []
    for c1 in chain_ids1:
        for c2 in chain_ids2:
            tm = tm_matrix[c1][c2][0]
            pairs.append((tm, c1, c2))
    pairs.sort(reverse=True)

    used1, used2 = set(), set()
    assignment = []
    for tm, c1, c2 in pairs:
        if c1 not in used1 and c2 not in used2 and tm > 0.1:
            assignment.append((c1, c2, tm, tm_matrix[c1][c2][1]))
            used1.add(c1)
            used2.add(c2)

    return assignment


def complex_tm_score(assignment, chains1, chains2):
    """
    Compute complex-level TM-score as described in US-align/QSalign:
    TM_complex = sum over aligned chain pairs of [TM_chain_pair * L_chain] / L_total
    """
    total_L = sum(len(c) for c, _ in chains1.values())
    tm_complex = 0.0
    for c1, c2, tm, rmsd in assignment:
        L_chain = len(chains1[c1][0])
        tm_complex += tm * L_chain
    tm_complex /= max(total_L, 1)
    return tm_complex


# ─── Load structures ──────────────────────────────────────────────────────────

def load_protein_chains(pdb_path, pdb_id):
    """Load all protein chains from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]

    chains = {}
    chain_info = {}
    for chain in model.get_chains():
        coords, res_ids = get_ca_coords(chain)
        if len(coords) >= 10:  # at least 10 residues
            chains[chain.id] = (coords, res_ids)
            chain_info[chain.id] = {
                'n_residues': len(coords),
                'first_res': res_ids[0] if res_ids else None,
                'last_res': res_ids[-1] if res_ids else None,
            }

    return structure, chains, chain_info


def analyze_complex(structure, pdb_id):
    """Analyze a protein complex structure."""
    model = structure[0]
    info = {
        'pdb_id': pdb_id,
        'n_chains_total': 0,
        'n_protein_chains': 0,
        'n_nucleic_chains': 0,
        'chains': {},
        'total_residues': 0,
        'total_atoms': 0,
    }

    for chain in model.get_chains():
        info['n_chains_total'] += 1
        residues = list(chain.get_residues())
        n_res = len([r for r in residues if r.id[0] == ' '])
        n_atoms = sum(len(list(r.get_atoms())) for r in residues)

        # Classify chain type
        aa_residues = [r for r in residues if r.id[0] == ' ' and 'CA' in r]
        na_residues = [r for r in residues if r.id[0] == ' ' and ("P" in r or "O3'" in r)]

        chain_type = 'protein' if len(aa_residues) > len(na_residues) else 'nucleic'
        if len(aa_residues) >= 10:
            info['n_protein_chains'] += 1
        else:
            info['n_nucleic_chains'] += 1

        info['chains'][chain.id] = {
            'type': chain_type,
            'n_residues': n_res,
            'n_atoms': n_atoms,
            'n_aa_residues': len(aa_residues),
        }
        info['total_residues'] += n_res
        info['total_atoms'] += n_atoms

    return info


# ─── Main analysis ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Protein Complex Structural Alignment: 7xg4 vs 6n40")
    print("=" * 70)

    # Load structures
    print("\n[1] Loading PDB structures...")
    struct_7xg4, chains_7xg4, chain_info_7xg4 = load_protein_chains(
        os.path.join(DATA_DIR, "7xg4.pdb"), "7xg4")
    struct_6n40, chains_6n40, chain_info_6n40 = load_protein_chains(
        os.path.join(DATA_DIR, "6n40.pdb"), "6n40")

    # Analyze complexes
    print("\n[2] Analyzing complex compositions...")
    info_7xg4 = analyze_complex(struct_7xg4, "7xg4")
    info_6n40 = analyze_complex(struct_6n40, "6n40")

    print(f"\n7xg4 (Type IV-A CRISPR-Cas, Pseudomonas aeruginosa):")
    print(f"  Total chains: {info_7xg4['n_chains_total']}")
    print(f"  Protein chains: {info_7xg4['n_protein_chains']}")
    print(f"  Nucleic acid chains: {info_7xg4['n_nucleic_chains']}")
    print(f"  Total residues: {info_7xg4['total_residues']}")
    print(f"  Total atoms: {info_7xg4['total_atoms']}")
    for cid, ci in info_7xg4['chains'].items():
        print(f"    Chain {cid}: {ci['type']}, {ci['n_residues']} residues")

    print(f"\n6n40 (MmpL3 membrane transporter, M. smegmatis):")
    print(f"  Total chains: {info_6n40['n_chains_total']}")
    print(f"  Protein chains: {info_6n40['n_protein_chains']}")
    print(f"  Total residues: {info_6n40['total_residues']}")
    print(f"  Total atoms: {info_6n40['total_atoms']}")
    for cid, ci in info_6n40['chains'].items():
        print(f"    Chain {cid}: {ci['type']}, {ci['n_residues']} residues")

    # Compute all pairwise chain TM-scores
    print("\n[3] Computing pairwise chain TM-scores (7xg4 chains vs 6n40 chains)...")
    chain_ids1 = list(chains_7xg4.keys())
    chain_ids2 = list(chains_6n40.keys())
    print(f"  7xg4 protein chains: {chain_ids1}")
    print(f"  6n40 protein chains: {chain_ids2}")

    pairwise_results = pairwise_chain_tm_scores(chains_7xg4, chains_6n40)

    # Print TM-score matrix
    print(f"\n  TM-score matrix (7xg4 chains vs 6n40 chains):")
    header = "         " + "  ".join(f"{c2:>6}" for c2 in chain_ids2)
    print(f"  {header}")
    tm_matrix_data = {}
    for c1 in chain_ids1:
        row = f"  {c1:>6}:  "
        tm_matrix_data[c1] = {}
        for c2 in chain_ids2:
            tm, rmsd = pairwise_results[c1][c2]
            row += f"  {tm:6.3f}"
            tm_matrix_data[c1][c2] = tm
        print(row)

    # Chain assignment
    print("\n[4] Computing optimal chain assignment...")
    assignment = greedy_chain_assignment(pairwise_results, chain_ids1, chain_ids2)

    print("\n  Chain Correspondence Table:")
    print(f"  {'7xg4 Chain':>12} | {'6n40 Chain':>10} | {'TM-score':>10} | {'RMSD (Å)':>10}")
    print("  " + "-" * 52)
    for c1, c2, tm, rmsd in assignment:
        n1 = len(chains_7xg4[c1][0])
        n2 = len(chains_6n40[c2][0])
        print(f"  {c1:>12} | {c2:>10} | {tm:>10.4f} | {rmsd:>10.3f}  ({n1} vs {n2} res)")

    # Compute complex-level TM-score
    tm_complex = complex_tm_score(assignment, chains_7xg4, chains_6n40)
    print(f"\n  Complex-level TM-score (weighted): {tm_complex:.4f}")

    # Per-chain alignment details with superimposition vectors
    print("\n[5] Computing superimposition vectors for best-matched chain pairs...")
    superimposition_data = []
    for c1, c2, tm, rmsd in assignment:
        coords1 = chains_7xg4[c1][0]
        coords2 = chains_6n40[c2][0]
        tm_val, rmsd_val, R, t = tm_align_pair(coords1, coords2)

        superimposition_data.append({
            'chain_7xg4': c1,
            'chain_6n40': c2,
            'tm_score': float(tm_val),
            'rmsd': float(rmsd_val),
            'n_residues_7xg4': len(coords1),
            'n_residues_6n40': len(coords2),
            'rotation_matrix': R.tolist(),
            'translation_vector': t.tolist(),
            'centroid_7xg4': coords1.mean(axis=0).tolist(),
            'centroid_6n40': coords2.mean(axis=0).tolist(),
        })

        print(f"\n  Chains 7xg4:{c1} ↔ 6n40:{c2}")
        print(f"    TM-score: {tm_val:.4f}, RMSD: {rmsd_val:.3f} Å")
        print(f"    Rotation matrix:\n      {R}")
        print(f"    Translation: {t}")

    # Self-alignment (7xg4 vs itself) for calibration
    print("\n[6] Self-alignment calibration (chain A of 7xg4 vs itself)...")
    test_chain = list(chains_7xg4.keys())[0]
    coords_self = chains_7xg4[test_chain][0]
    tm_self, rmsd_self, _, _ = tm_align_pair(coords_self, coords_self)
    print(f"  Self TM-score (chain {test_chain}): {tm_self:.4f}, RMSD: {rmsd_self:.3f} Å  [expected: ~1.0, ~0.0]")

    # Chain length statistics
    print("\n[7] Chain length statistics...")
    lengths_7xg4 = [len(v[0]) for v in chains_7xg4.values()]
    lengths_6n40 = [len(v[0]) for v in chains_6n40.values()]

    chain_stats = {
        '7xg4': {
            'chain_lengths': {k: len(v[0]) for k, v in chains_7xg4.items()},
            'mean_length': float(np.mean(lengths_7xg4)),
            'total_residues': int(np.sum(lengths_7xg4)),
        },
        '6n40': {
            'chain_lengths': {k: len(v[0]) for k, v in chains_6n40.items()},
            'mean_length': float(np.mean(lengths_6n40)),
            'total_residues': int(np.sum(lengths_6n40)),
        }
    }

    # Save all results
    results = {
        'complex_info': {'7xg4': info_7xg4, '6n40': info_6n40},
        'chain_stats': chain_stats,
        'pairwise_tm_matrix': {c1: {c2: pairwise_results[c1][c2][0] for c2 in chain_ids2}
                                for c1 in chain_ids1},
        'pairwise_rmsd_matrix': {c1: {c2: pairwise_results[c1][c2][1] for c2 in chain_ids2}
                                  for c1 in chain_ids1},
        'chain_assignment': [
            {'chain_7xg4': c1, 'chain_6n40': c2, 'tm_score': tm, 'rmsd': rmsd}
            for c1, c2, tm, rmsd in assignment
        ],
        'complex_tm_score': float(tm_complex),
        'superimposition_data': superimposition_data,
    }

    out_path = os.path.join(OUT_DIR, "alignment_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    return results


if __name__ == "__main__":
    results = main()
