"""Pairwise structural alignment between protein complexes 7xg4 and 6n40.

This script:
- Parses the PDB structures using Biopython
- Extracts C-alpha coordinates for each chain
- Performs a simple chain-wise alignment using Kabsch on the first N residues
- Explores all query chains (7xg4) against target chain A (6n40) and reports TM-like scores
- Produces figures for chain lengths, RMSD distributions, and best superposition

Note: This is a simplified research implementation, not an optimized production tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Bio.PDB import PDBParser


@dataclass
class ChainCoords:
    chain_id: str
    res_ids: List[int]
    coords: np.ndarray  # (N, 3)


def load_structure_ca(path: Path) -> Dict[str, ChainCoords]:
    parser = PDBParser(QUIET=True)
    structure_id = path.stem
    structure = parser.get_structure(structure_id, str(path))

    chains: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for model in structure:
        for chain in model:
            cid = chain.id
            if cid not in chains:
                chains[cid] = []
            for res in chain:
                hetflag, resseq, icode = res.id
                if hetflag != " ":
                    continue
                if "CA" not in res:
                    continue
                ca = res["CA"].coord.astype(float)
                chains[cid].append((resseq, ca))

    result: Dict[str, ChainCoords] = {}
    for cid, lst in chains.items():
        if not lst:
            continue
        lst.sort(key=lambda x: x[0])
        res_ids = [r for r, _ in lst]
        coords = np.stack([c for _, c in lst], axis=0)
        result[cid] = ChainCoords(chain_id=cid, res_ids=res_ids, coords=coords)
    return result


@dataclass
class AlignmentResult:
    query_chain: str
    target_chain: str
    n_aligned: int
    rmsd: float
    tm_score: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    C = np.dot(P_centered.T, Q_centered)
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] *= -1
    R = np.dot(V, Wt)
    t = Q_centroid - R @ P_centroid

    P_aligned = (R @ P.T).T + t
    diff2 = np.sum((P_aligned - Q) ** 2, axis=1)
    rmsd = math.sqrt(diff2.mean())
    return R, t, rmsd


def tm_score(P: np.ndarray, Q: np.ndarray, L_target: int) -> float:
    assert P.shape == Q.shape
    L_aln = P.shape[0]
    if L_aln == 0:
        return 0.0
    d0 = 1.24 * (L_target - 15) ** (1 / 3) - 1.8
    d0 = max(d0, 0.5)
    diff2 = np.sum((P - Q) ** 2, axis=1)
    score = (1.0 / L_target) * np.sum(1.0 / (1.0 + diff2 / (d0**2)))
    return float(score)


def align_chain_pair(q: ChainCoords, t: ChainCoords, max_residues: int | None = None) -> AlignmentResult:
    n = min(len(q.coords), len(t.coords))
    if max_residues is not None:
        n = min(n, max_residues)
    if n < 5:
        raise ValueError("Too few residues for reliable alignment")

    P = q.coords[:n]
    Q = t.coords[:n]

    R, t_vec, rmsd = kabsch(P, Q)

    P_aligned = (R @ P.T).T + t_vec
    tm = tm_score(P_aligned, Q, L_target=len(t.coords))

    return AlignmentResult(
        query_chain=q.chain_id,
        target_chain=t.chain_id,
        n_aligned=n,
        rmsd=rmsd,
        tm_score=tm,
        R=R,
        t=t_vec,
    )


def run_pairwise_alignment(query_pdb: Path, target_pdb: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    q_chains = load_structure_ca(query_pdb)
    t_chains = load_structure_ca(target_pdb)

    target_chain_id = "A"
    if target_chain_id not in t_chains:
        target_chain_id = sorted(t_chains.keys())[0]
    t = t_chains[target_chain_id]

    results: List[AlignmentResult] = []
    for qid, q in q_chains.items():
        try:
            res = align_chain_pair(q, t)
            results.append(res)
        except Exception as e:
            print(f"Skipping chain {qid}: {e}")

    results.sort(key=lambda r: r.tm_score, reverse=True)

    import csv

    csv_path = outdir / "chain_alignment_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_chain", "target_chain", "n_aligned", "rmsd", "tm_score"])
        for r in results:
            w.writerow([r.query_chain, r.target_chain, r.n_aligned, f"{r.rmsd:.3f}", f"{r.tm_score:.4f}"])

    print("Top chain alignments (by TM-score):")
    for r in results[:5]:
        print(
            f"Q{r.query_chain}-T{r.target_chain}: N={r.n_aligned}, RMSD={r.rmsd:.2f} Å, TM={r.tm_score:.3f}"
        )

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    q_lengths = [len(q.coords) for q in q_chains.values()]
    sns.barplot(x=list(q_chains.keys()), y=q_lengths, ax=ax1, color="steelblue")
    ax1.set_xlabel("7xg4 chain ID")
    ax1.set_ylabel("Number of Cα residues")
    ax1.set_title("Chain lengths in query complex 7xg4")
    fig1.tight_layout()
    fig1.savefig(outdir / "chain_lengths_7xg4.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x=[r.n_aligned for r in results],
        y=[r.tm_score for r in results],
        size=[-r.rmsd for r in results],
        hue=[r.query_chain for r in results],
        ax=ax2,
    )
    ax2.set_xlabel("Number of aligned residues")
    ax2.set_ylabel("TM-score")
    ax2.set_title("Chain-wise alignment quality vs. coverage")
    fig2.tight_layout()
    fig2.savefig(outdir / "tm_vs_coverage.png", dpi=300)

    if results:
        best = results[0]
        q_best = q_chains[best.query_chain]
        n = best.n_aligned
        P = q_best.coords[:n]
        Q = t.coords[:n]
        P_aligned = (best.R @ P.T).T + best.t

        fig3 = plt.figure(figsize=(6, 6))
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.plot(P_aligned[:, 0], P_aligned[:, 1], P_aligned[:, 2], label="7xg4 (aligned)")
        ax3.plot(Q[:, 0], Q[:, 1], Q[:, 2], label="6n40", alpha=0.7)
        ax3.set_title(
            f"Best chain superposition Q{best.query_chain}/T{best.target_chain}\n"
            f"N={best.n_aligned}, RMSD={best.rmsd:.2f} Å, TM={best.tm_score:.3f}"
        )
        ax3.set_xlabel("X (Å)")
        ax3.set_ylabel("Y (Å)")
        ax3.set_zlabel("Z (Å)")
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(outdir / "best_chain_superposition.png", dpi=300)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    query = base / "data/7xg4.pdb"
    target = base / "data/6n40.pdb"
    outdir = base / "outputs"
    run_pairwise_alignment(query, target, outdir)
