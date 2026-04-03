#!/usr/bin/env python3
"""Reproducible structural alignment analysis for the provided PDB inputs.

The script builds a lightweight, TM-score-style structural alignment workflow
for protein chains and applies it to the supplied 7xg4 and 6n40 structures.
It writes tabular outputs, figures, and summary JSON files for the report.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy.optimize import linear_sum_assignment


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


@dataclass
class ResidueRecord:
    resseq: int
    icode: str
    resname: str
    one_letter: str
    ca_coord: np.ndarray


@dataclass
class ChainRecord:
    structure_id: str
    chain_id: str
    molecule_type: str
    molecule_name: str
    residues: List[ResidueRecord]

    @property
    def length(self) -> int:
        return len(self.residues)

    @property
    def coords(self) -> np.ndarray:
        return np.vstack([r.ca_coord for r in self.residues])

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter for r in self.residues)


@dataclass
class AlignmentResult:
    query_structure: str
    query_chain: str
    target_structure: str
    target_chain: str
    aligned_pairs: int
    rmsd: float
    tm_score_query_norm: float
    tm_score_target_norm: float
    tm_score_min_norm: float
    sequence_identity: float
    query_length: int
    target_length: int
    rotation: List[List[float]]
    translation: List[float]
    aligned_query_indices: List[int]
    aligned_target_indices: List[int]
    aligned_distances: List[float]
    seed_description: str


def tm_d0(length: int) -> float:
    if length <= 15:
        return 0.5
    return max(0.5, 1.24 * ((length - 15) ** (1.0 / 3.0)) - 1.8)


def parse_compnd_map(pdb_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    current_name = ""
    current_chains: List[str] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("COMPND"):
                continue
            payload = line[10:].strip()
            if "MOL_ID:" in payload and current_chains:
                for chain in current_chains:
                    mapping[chain] = current_name or "UNKNOWN"
                current_name = ""
                current_chains = []
            if payload.startswith("MOLECULE:"):
                current_name = payload.split(":", 1)[1].strip().rstrip(";")
            elif payload.startswith("CHAIN:"):
                chains = payload.split(":", 1)[1].strip().rstrip(";")
                current_chains = [c.strip() for c in chains.split(",")]
        if current_chains:
            for chain in current_chains:
                mapping[chain] = current_name or "UNKNOWN"
    return mapping


def residue_one_letter(resname: str) -> str:
    return protein_letters_3to1.get(resname.upper(), "X")


def infer_molecule_type(residues: Sequence[ResidueRecord]) -> str:
    if residues and all(r.one_letter in {"A", "C", "G", "U", "X"} for r in residues):
        return "nucleic_acid"
    return "protein"


def load_structure(pdb_path: Path) -> List[ChainRecord]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    compnd_map = parse_compnd_map(pdb_path)
    chain_records: List[ChainRecord] = []

    for chain in model:
        residues: List[ResidueRecord] = []
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                # The target alignment task is protein-only for these inputs.
                continue
            one = residue_one_letter(residue.get_resname())
            residues.append(
                ResidueRecord(
                    resseq=int(residue.id[1]),
                    icode=(residue.id[2] or "").strip(),
                    resname=residue.get_resname().strip(),
                    one_letter=one,
                    ca_coord=residue["CA"].coord.astype(float),
                )
            )
        if not residues:
            chain_records.append(
                ChainRecord(
                    structure_id=pdb_path.stem,
                    chain_id=chain.id,
                    molecule_type="nucleic_acid",
                    molecule_name=compnd_map.get(chain.id, "UNKNOWN"),
                    residues=[],
                )
            )
            continue
        mol_type = infer_molecule_type(residues)
        chain_records.append(
            ChainRecord(
                structure_id=pdb_path.stem,
                chain_id=chain.id,
                molecule_type=mol_type,
                molecule_name=compnd_map.get(chain.id, "UNKNOWN"),
                residues=residues,
            )
        )
    return chain_records


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(P) != len(Q) or len(P) == 0:
        raise ValueError("Kabsch requires equally sized non-empty point sets.")
    p_centroid = P.mean(axis=0)
    q_centroid = Q.mean(axis=0)
    P0 = P - p_centroid
    Q0 = Q - q_centroid
    cov = P0.T @ Q0
    V, _, Wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = q_centroid - p_centroid @ R
    return R, t


def transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords @ rotation + translation


def tm_score(distances: np.ndarray, norm_length: int) -> float:
    d0 = tm_d0(norm_length)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / norm_length)


def sequence_identity(query: ChainRecord, target: ChainRecord, q_idx: List[int], t_idx: List[int]) -> float:
    if not q_idx:
        return 0.0
    matches = sum(
        query.residues[i].one_letter == target.residues[j].one_letter
        for i, j in zip(q_idx, t_idx)
    )
    return matches / len(q_idx)


def ungapped_seed_pairs(n_query: int, n_target: int, min_overlap: int = 25) -> List[Tuple[List[int], List[int], str]]:
    seeds: List[Tuple[List[int], List[int], str]] = []
    for shift in range(-(n_query - min_overlap), n_target - min_overlap + 1):
        q_idx: List[int] = []
        t_idx: List[int] = []
        for i in range(n_query):
            j = i + shift
            if 0 <= j < n_target:
                q_idx.append(i)
                t_idx.append(j)
        if len(q_idx) >= min_overlap:
            seeds.append((q_idx, t_idx, f"ungapped_shift_{shift}"))
    return seeds


def fragment_seed_pairs(
    query_coords: np.ndarray,
    target_coords: np.ndarray,
    fragment_length: int = 20,
    step: int = 60,
) -> List[Tuple[List[int], List[int], str]]:
    n_query, n_target = len(query_coords), len(target_coords)
    seeds: List[Tuple[List[int], List[int], str]] = []
    if n_query < fragment_length or n_target < fragment_length:
        return seeds
    for qi in range(0, n_query - fragment_length + 1, step):
        q_idx = list(range(qi, qi + fragment_length))
        q_fragment = query_coords[q_idx]
        for tj in range(0, n_target - fragment_length + 1, step):
            t_idx = list(range(tj, tj + fragment_length))
            t_fragment = target_coords[t_idx]
            _, _ = kabsch(q_fragment, t_fragment)
            seeds.append((q_idx, t_idx, f"fragment_q{qi}_t{tj}"))
    return seeds


def local_tm_align(
    query_coords: np.ndarray,
    target_coords: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    gap_penalty: float = 0.2,
) -> Tuple[List[int], List[int]]:
    q_trans = transform(query_coords, rotation, translation)
    d0 = tm_d0(min(len(query_coords), len(target_coords)))
    dist_matrix = np.linalg.norm(q_trans[:, None, :] - target_coords[None, :, :], axis=2)
    score_matrix = 1.0 / (1.0 + (dist_matrix / d0) ** 2) - 0.5

    n, m = score_matrix.shape
    H = np.zeros((n + 1, m + 1), dtype=float)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    best = (0.0, 0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = H[i - 1, j - 1] + score_matrix[i - 1, j - 1]
            up = H[i - 1, j] - gap_penalty
            left = H[i, j - 1] - gap_penalty
            score = max(0.0, diag, up, left)
            H[i, j] = score
            if score == 0.0:
                trace[i, j] = 0
            elif score == diag:
                trace[i, j] = 1
            elif score == up:
                trace[i, j] = 2
            else:
                trace[i, j] = 3
            if score > best[0]:
                best = (score, i, j)

    q_idx: List[int] = []
    t_idx: List[int] = []
    _, i, j = best
    while i > 0 and j > 0 and H[i, j] > 0.0:
        move = trace[i, j]
        if move == 1:
            q_idx.append(i - 1)
            t_idx.append(j - 1)
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            break
    q_idx.reverse()
    t_idx.reverse()
    return q_idx, t_idx


def refine_alignment(query: ChainRecord, target: ChainRecord) -> AlignmentResult:
    query_coords = query.coords
    target_coords = target.coords
    seeds = ungapped_seed_pairs(len(query_coords), len(target_coords))
    seeds.extend(fragment_seed_pairs(query_coords, target_coords))

    evaluated: List[Tuple[float, List[int], List[int], np.ndarray, np.ndarray, str]] = []
    for q_idx, t_idx, label in seeds:
        R, t = kabsch(query_coords[q_idx], target_coords[t_idx])
        q_seed = transform(query_coords[q_idx], R, t)
        d = np.linalg.norm(q_seed - target_coords[t_idx], axis=1)
        seed_score = tm_score(d, min(len(query_coords), len(target_coords)))
        evaluated.append((seed_score, q_idx, t_idx, R, t, label))

    evaluated.sort(key=lambda x: x[0], reverse=True)
    best_result: AlignmentResult | None = None
    seen = 0

    for _, q_idx, t_idx, R, t, label in evaluated:
        if seen >= 12:
            break
        seen += 1
        prev_pairs: Tuple[Tuple[int, ...], Tuple[int, ...]] | None = None
        curr_q_idx, curr_t_idx = q_idx, t_idx
        curr_R, curr_t = R, t
        for _iteration in range(8):
            refined_q_idx, refined_t_idx = local_tm_align(query_coords, target_coords, curr_R, curr_t)
            if len(refined_q_idx) < 10:
                break
            pair_state = (tuple(refined_q_idx), tuple(refined_t_idx))
            if pair_state == prev_pairs:
                break
            prev_pairs = pair_state
            curr_q_idx, curr_t_idx = refined_q_idx, refined_t_idx
            curr_R, curr_t = kabsch(query_coords[curr_q_idx], target_coords[curr_t_idx])

        if len(curr_q_idx) < 10:
            continue
        q_aligned = transform(query_coords[curr_q_idx], curr_R, curr_t)
        t_aligned = target_coords[curr_t_idx]
        distances = np.linalg.norm(q_aligned - t_aligned, axis=1)
        rmsd = float(np.sqrt(np.mean(np.square(distances))))
        candidate = AlignmentResult(
            query_structure=query.structure_id,
            query_chain=query.chain_id,
            target_structure=target.structure_id,
            target_chain=target.chain_id,
            aligned_pairs=len(curr_q_idx),
            rmsd=rmsd,
            tm_score_query_norm=tm_score(distances, len(query_coords)),
            tm_score_target_norm=tm_score(distances, len(target_coords)),
            tm_score_min_norm=tm_score(distances, min(len(query_coords), len(target_coords))),
            sequence_identity=sequence_identity(query, target, curr_q_idx, curr_t_idx),
            query_length=len(query_coords),
            target_length=len(target_coords),
            rotation=np.round(curr_R, 6).tolist(),
            translation=np.round(curr_t, 6).tolist(),
            aligned_query_indices=curr_q_idx,
            aligned_target_indices=curr_t_idx,
            aligned_distances=np.round(distances, 3).tolist(),
            seed_description=label,
        )
        if best_result is None or candidate.tm_score_min_norm > best_result.tm_score_min_norm:
            best_result = candidate

    if best_result is None:
        raise RuntimeError(f"No viable alignment found for {query.chain_id} vs {target.chain_id}")
    return best_result


def build_chain_dataframe(chain_records: Iterable[ChainRecord]) -> pd.DataFrame:
    rows = []
    for chain in chain_records:
        rows.append(
            {
                "structure": chain.structure_id,
                "chain": chain.chain_id,
                "molecule_type": chain.molecule_type,
                "molecule_name": chain.molecule_name,
                "length": chain.length,
            }
        )
    return pd.DataFrame(rows)


def save_alignment_pdb(alignment: AlignmentResult, query: ChainRecord, target: ChainRecord, out_path: Path) -> None:
    R = np.array(alignment.rotation)
    t = np.array(alignment.translation)
    q_coords = transform(query.coords[alignment.aligned_query_indices], R, t)
    t_coords = target.coords[alignment.aligned_target_indices]

    lines: List[str] = []
    atom_serial = 1
    for idx, coord in zip(alignment.aligned_query_indices, q_coords):
        residue = query.residues[idx]
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  {residue.resname:>3s} Q{residue.resseq:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           C"
        )
        atom_serial += 1
    for idx, coord in zip(alignment.aligned_target_indices, t_coords):
        residue = target.residues[idx]
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  {residue.resname:>3s} T{residue.resseq:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           C"
        )
        atom_serial += 1
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


def bootstrap_tm_scores(
    alignment: AlignmentResult,
    query: ChainRecord,
    target: ChainRecord,
    n_boot: int = 200,
    sample_fraction: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q_idx = np.array(alignment.aligned_query_indices)
    t_idx = np.array(alignment.aligned_target_indices)
    rows = []
    sample_size = max(10, int(len(q_idx) * sample_fraction))
    for boot_id in range(n_boot):
        picked = np.sort(rng.choice(len(q_idx), size=sample_size, replace=False))
        qb = q_idx[picked]
        tb = t_idx[picked]
        R, t = kabsch(query.coords[qb], target.coords[tb])
        q_coords = transform(query.coords[qb], R, t)
        distances = np.linalg.norm(q_coords - target.coords[tb], axis=1)
        rows.append(
            {
                "bootstrap_id": boot_id,
                "tm_score_min_norm": tm_score(distances, min(query.length, target.length)),
                "rmsd": float(np.sqrt(np.mean(np.square(distances)))),
            }
        )
    return pd.DataFrame(rows)


def plot_data_overview(chain_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    protein_df = chain_df[chain_df["molecule_type"] == "protein"].copy()
    protein_df["chain_label"] = protein_df["structure"] + ":" + protein_df["chain"]

    sns.barplot(
        data=protein_df,
        x="chain_label",
        y="length",
        hue="structure",
        dodge=False,
        palette={"7xg4": "#1f77b4", "6n40": "#d62728"},
        ax=axes[0],
    )
    axes[0].set_title("Protein chain lengths")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Resolved C-alpha residues")
    axes[0].tick_params(axis="x", rotation=45)

    mol_counts = chain_df.groupby(["structure", "molecule_type"]).size().reset_index(name="count")
    sns.barplot(
        data=mol_counts,
        x="structure",
        y="count",
        hue="molecule_type",
        palette={"protein": "#2ca02c", "nucleic_acid": "#9467bd"},
        ax=axes[1],
    )
    axes[1].set_title("Molecule-type composition")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Chain count")

    fig.suptitle("Input structure overview", fontsize=14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_chain_scores(score_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    ranked = score_df.sort_values("tm_score_min_norm", ascending=False)
    sns.barplot(
        data=ranked,
        x="query_chain",
        y="tm_score_min_norm",
        color="#1f77b4",
        ax=axes[0],
    )
    axes[0].set_title("Best refined alignment score by query chain")
    axes[0].set_xlabel("7xg4 protein chain")
    axes[0].set_ylabel("TM-score (min-length normalization)")

    heatmap_df = score_df[["query_chain", "tm_score_query_norm", "tm_score_target_norm", "tm_score_min_norm"]].set_index("query_chain")
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "TM-score"},
        ax=axes[1],
    )
    axes[1].set_title("Normalization-dependent TM-scores")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("7xg4 protein chain")

    fig.suptitle("Chain-wise alignment landscape against 6n40:A", fontsize=14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def principal_axes(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh.T


def plot_superposition(
    alignment: AlignmentResult,
    query: ChainRecord,
    target: ChainRecord,
    out_path: Path,
) -> None:
    R = np.array(alignment.rotation)
    t = np.array(alignment.translation)
    q_coords = transform(query.coords[alignment.aligned_query_indices], R, t)
    t_coords = target.coords[alignment.aligned_target_indices]

    basis = principal_axes(np.vstack([q_coords, t_coords]))
    q_proj = (q_coords - t_coords.mean(axis=0)) @ basis[:, :2]
    t_proj = (t_coords - t_coords.mean(axis=0)) @ basis[:, :2]
    distances = np.array(alignment.aligned_distances)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(t_proj[:, 0], t_proj[:, 1], color="#d62728", linewidth=2, label=f"{target.structure_id}:{target.chain_id}")
    axes[0].plot(q_proj[:, 0], q_proj[:, 1], color="#1f77b4", linewidth=2, alpha=0.8, label=f"{query.structure_id}:{query.chain_id}")
    axes[0].scatter(t_proj[0, 0], t_proj[0, 1], color="#d62728", s=50)
    axes[0].scatter(q_proj[0, 0], q_proj[0, 1], color="#1f77b4", s=50)
    axes[0].set_title("Principal-plane superposition of aligned residues")
    axes[0].set_xlabel("PC1 (A)")
    axes[0].set_ylabel("PC2 (A)")
    axes[0].legend(frameon=False)

    axes[1].plot(np.arange(1, len(distances) + 1), distances, color="#2ca02c", linewidth=1.8)
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1, label="5 A reference")
    axes[1].set_title("Per-residue aligned distance profile")
    axes[1].set_xlabel("Aligned residue pair rank")
    axes[1].set_ylabel("Distance after superposition (A)")
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"Best structural correspondence: {query.structure_id}:{query.chain_id} vs {target.structure_id}:{target.chain_id}",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_validation(
    score_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    sns.scatterplot(
        data=score_df,
        x="aligned_pairs",
        y="tm_score_min_norm",
        hue="query_chain",
        palette="tab10",
        s=90,
        ax=axes[0],
    )
    axes[0].set_title("Coverage versus score across candidate query chains")
    axes[0].set_xlabel("Aligned residue pairs")
    axes[0].set_ylabel("TM-score (min-length normalization)")

    sns.histplot(
        data=bootstrap_df,
        x="tm_score_min_norm",
        bins=20,
        color="#1f77b4",
        kde=True,
        ax=axes[1],
    )
    axes[1].set_title("Bootstrap stability of best alignment")
    axes[1].set_xlabel("Bootstrap TM-score")
    axes[1].set_ylabel("Count")

    fig.suptitle("Alignment validation and robustness", fontsize=14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    query_chains = load_structure(DATA_DIR / "7xg4.pdb")
    target_chains = load_structure(DATA_DIR / "6n40.pdb")

    chain_df = build_chain_dataframe(query_chains + target_chains)
    chain_df.to_csv(OUTPUT_DIR / "chain_overview.csv", index=False)

    query_proteins = [c for c in query_chains if c.molecule_type == "protein" and c.length > 0]
    target_proteins = [c for c in target_chains if c.molecule_type == "protein" and c.length > 0]

    pair_results: List[AlignmentResult] = []
    for q in query_proteins:
        for t in target_proteins:
            pair_results.append(refine_alignment(q, t))

    pair_df = pd.DataFrame([asdict(r) for r in pair_results])
    pair_df.to_csv(OUTPUT_DIR / "chain_pair_alignment_scores.csv", index=False)

    score_matrix = np.array(
        [
            [r.tm_score_min_norm for r in pair_results if r.query_chain == q.chain_id and r.target_chain == t.chain_id][0]
            for q in query_proteins
            for t in target_proteins
        ]
    ).reshape(len(query_proteins), len(target_proteins))
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    assignments = [
        {
            "query_chain": query_proteins[i].chain_id,
            "target_chain": target_proteins[j].chain_id,
            "tm_score_min_norm": float(score_matrix[i, j]),
        }
        for i, j in zip(row_ind, col_ind)
    ]
    best_assignment = max(assignments, key=lambda x: x["tm_score_min_norm"])
    best_result = max(pair_results, key=lambda r: r.tm_score_min_norm)

    query_chain_map = {c.chain_id: c for c in query_proteins}
    target_chain_map = {c.chain_id: c for c in target_proteins}
    best_query = query_chain_map[best_result.query_chain]
    best_target = target_chain_map[best_result.target_chain]

    bootstrap_df = bootstrap_tm_scores(best_result, best_query, best_target)
    bootstrap_df.to_csv(OUTPUT_DIR / "bootstrap_best_alignment.csv", index=False)

    aligned_pairs_table = pd.DataFrame(
        {
            "query_chain": best_result.query_chain,
            "query_resseq": [best_query.residues[i].resseq for i in best_result.aligned_query_indices],
            "query_resname": [best_query.residues[i].resname for i in best_result.aligned_query_indices],
            "target_chain": best_result.target_chain,
            "target_resseq": [best_target.residues[j].resseq for j in best_result.aligned_target_indices],
            "target_resname": [best_target.residues[j].resname for j in best_result.aligned_target_indices],
            "distance_after_superposition": best_result.aligned_distances,
        }
    )
    aligned_pairs_table.to_csv(OUTPUT_DIR / "best_alignment_residue_pairs.csv", index=False)

    summary = {
        "query_structure": "7xg4",
        "target_structure": "6n40",
        "query_protein_chains": [c.chain_id for c in query_proteins],
        "target_protein_chains": [c.chain_id for c in target_proteins],
        "assignment_strategy": "maximum-weight bipartite matching on chain-pair TM-scores",
        "chain_assignments": assignments,
        "best_alignment": asdict(best_result),
        "bootstrap_tm_score_mean": float(bootstrap_df["tm_score_min_norm"].mean()),
        "bootstrap_tm_score_std": float(bootstrap_df["tm_score_min_norm"].std(ddof=1)),
        "caveat": (
            "7xg4 is annotated as a dodecameric protein-RNA-DNA assembly, whereas 6n40 is annotated "
            "as a monomeric membrane protein. Therefore the supplied pair supports only a partial "
            "protein-chain versus monomer alignment analysis, not a homologous full-complex comparison."
        ),
    }
    with (OUTPUT_DIR / "alignment_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    save_alignment_pdb(best_result, best_query, best_target, OUTPUT_DIR / "best_superposition_ca_only.pdb")

    plot_data_overview(chain_df, REPORT_IMG_DIR / "figure_data_overview.png")
    plot_chain_scores(pair_df, REPORT_IMG_DIR / "figure_chain_scores.png")
    plot_superposition(best_result, best_query, best_target, REPORT_IMG_DIR / "figure_superposition.png")
    plot_validation(pair_df, bootstrap_df, REPORT_IMG_DIR / "figure_validation.png")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    main()
