#!/usr/bin/env python3
"""Prototype multimodal diffusion-style complex refinement on the 2L3R sample.

The full research task asks for a unified deep learning framework spanning
proteins, nucleic acids, and small molecules. The workspace only provides a
single protein-ligand complex, so this script implements a reproducible
feasibility study:

1. Parse the experimental protein and ligand structures.
2. Build a unified graph with modality-aware edges.
3. Train a compact equivariant denoiser on synthetic rigid-body augmented
   noisy decoys of the known complex.
4. Compare a cross-modal model against a no-cross-edge ablation and an
   identity baseline.
5. Save metrics, figures, and a publication-style report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data" / "sample" / "2l3r"
OUTPUT_DIR = WORKSPACE / "outputs"
IMAGE_DIR = WORKSPACE / "report" / "images"


AA_TO_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}

ELEMENTS = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
]
ELEMENT_TO_INDEX = {name: i for i, name in enumerate(ELEMENTS)}

EDGE_CHAIN = 0
EDGE_PROTEIN_SPATIAL = 1
EDGE_LIGAND_BOND = 2
EDGE_LIGAND_SPATIAL = 3
EDGE_CROSS = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class ProteinResidue:
    resseq: int
    resname: str
    coord: np.ndarray


@dataclass
class LigandAtom:
    element: str
    coord: np.ndarray
    original_index: int


@dataclass
class ComplexSample:
    protein_residues: List[ProteinResidue]
    ligand_atoms: List[LigandAtom]
    ligand_bonds: List[Tuple[int, int]]
    ligand_signatures: List[str]
    native_coords: np.ndarray
    protein_count: int
    ligand_count: int
    node_types: np.ndarray
    token_indices: np.ndarray
    token_groups: np.ndarray
    sequence_positions: np.ndarray
    edges_full: np.ndarray
    edge_types_full: np.ndarray
    edges_no_cross: np.ndarray
    edge_types_no_cross: np.ndarray


def parse_protein_ca(path: Path) -> List[ProteinResidue]:
    residues: List[ProteinResidue] = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resseq = int(line[22:26])
            if resseq in seen:
                continue
            seen.add(resseq)
            resname = line[17:20].strip()
            coord = np.array(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=np.float32,
            )
            residues.append(ProteinResidue(resseq=resseq, resname=resname, coord=coord))
    return residues


def _normalize_element(symbol: str) -> str:
    symbol = symbol.strip()
    if not symbol:
        return "C"
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def parse_ligand_sdf(path: Path) -> Tuple[List[LigandAtom], List[Tuple[int, int]], List[str]]:
    lines = path.read_text().splitlines()
    atom_count = int(lines[3][:3])
    bond_count = int(lines[3][3:6])

    atoms_all: List[LigandAtom] = []
    for idx in range(atom_count):
        line = lines[4 + idx]
        x = float(line[:10])
        y = float(line[10:20])
        z = float(line[20:30])
        element = _normalize_element(line[31:34])
        atoms_all.append(
            LigandAtom(
                element=element,
                coord=np.array([x, y, z], dtype=np.float32),
                original_index=idx,
            )
        )

    bonds_all: List[Tuple[int, int]] = []
    for idx in range(bond_count):
        line = lines[4 + atom_count + idx]
        a = int(line[:3]) - 1
        b = int(line[3:6]) - 1
        bonds_all.append((a, b))

    heavy_indices = [i for i, atom in enumerate(atoms_all) if atom.element != "H"]
    remap = {old: new for new, old in enumerate(heavy_indices)}
    atoms = [atoms_all[i] for i in heavy_indices]
    bonds = [(remap[a], remap[b]) for a, b in bonds_all if a in remap and b in remap]

    degrees = {i: 0 for i in range(len(atoms))}
    for a, b in bonds:
        degrees[a] += 1
        degrees[b] += 1
    signatures = [f"{atom.element}:{degrees[i]}" for i, atom in enumerate(atoms)]
    return atoms, bonds, signatures


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def make_edges(
    protein_residues: Sequence[ProteinResidue],
    ligand_atoms: Sequence[LigandAtom],
    ligand_bonds: Sequence[Tuple[int, int]],
    protein_spatial_cutoff: float = 10.0,
    ligand_spatial_cutoff: float = 4.5,
    cross_cutoff: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    protein_count = len(protein_residues)
    ligand_count = len(ligand_atoms)
    edges: List[Tuple[int, int]] = []
    edge_types: List[int] = []

    def add_undirected(a: int, b: int, edge_type: int) -> None:
        edges.append((a, b))
        edge_types.append(edge_type)
        edges.append((b, a))
        edge_types.append(edge_type)

    for i in range(protein_count - 1):
        add_undirected(i, i + 1, EDGE_CHAIN)

    protein_coords = np.stack([r.coord for r in protein_residues], axis=0)
    protein_dist = pairwise_distances(protein_coords, protein_coords)
    for i in range(protein_count):
        for j in range(i + 2, protein_count):
            if protein_dist[i, j] <= protein_spatial_cutoff:
                add_undirected(i, j, EDGE_PROTEIN_SPATIAL)

    ligand_offset = protein_count
    for a, b in ligand_bonds:
        add_undirected(ligand_offset + a, ligand_offset + b, EDGE_LIGAND_BOND)

    ligand_coords = np.stack([a.coord for a in ligand_atoms], axis=0)
    ligand_dist = pairwise_distances(ligand_coords, ligand_coords)
    ligand_bond_set = {tuple(sorted(bond)) for bond in ligand_bonds}
    for i in range(ligand_count):
        for j in range(i + 1, ligand_count):
            if (i, j) in ligand_bond_set:
                continue
            if ligand_dist[i, j] <= ligand_spatial_cutoff:
                add_undirected(ligand_offset + i, ligand_offset + j, EDGE_LIGAND_SPATIAL)

    cross_dist = pairwise_distances(protein_coords, ligand_coords)
    for i in range(protein_count):
        for j in range(ligand_count):
            if cross_dist[i, j] <= cross_cutoff:
                add_undirected(i, ligand_offset + j, EDGE_CROSS)

    edges_arr = np.array(edges, dtype=np.int64)
    edge_types_arr = np.array(edge_types, dtype=np.int64)
    keep = edge_types_arr != EDGE_CROSS
    return edges_arr, edge_types_arr, edges_arr[keep], edge_types_arr[keep]


def build_complex_sample() -> ComplexSample:
    protein_residues = parse_protein_ca(DATA_DIR / "2l3r_protein.pdb")
    ligand_atoms, ligand_bonds, ligand_signatures = parse_ligand_sdf(DATA_DIR / "2l3r_ligand.sdf")

    protein_count = len(protein_residues)
    ligand_count = len(ligand_atoms)
    protein_coords = np.stack([r.coord for r in protein_residues], axis=0)
    ligand_coords = np.stack([a.coord for a in ligand_atoms], axis=0)
    native_coords = np.concatenate([protein_coords, ligand_coords], axis=0).astype(np.float32)

    node_types = np.concatenate(
        [
            np.zeros(protein_count, dtype=np.int64),
            np.full(ligand_count, 2, dtype=np.int64),
        ]
    )
    token_indices = np.concatenate(
        [
            np.array([AA_TO_INDEX.get(r.resname, 0) for r in protein_residues], dtype=np.int64),
            np.array([ELEMENT_TO_INDEX.get(a.element, 0) for a in ligand_atoms], dtype=np.int64),
        ]
    )
    token_groups = np.concatenate(
        [
            np.zeros(protein_count, dtype=np.int64),
            np.full(ligand_count, 2, dtype=np.int64),
        ]
    )
    sequence_positions = np.concatenate(
        [
            np.linspace(0.0, 1.0, protein_count, dtype=np.float32),
            np.linspace(0.0, 1.0, ligand_count, dtype=np.float32),
        ]
    )
    edges_full, edge_types_full, edges_no_cross, edge_types_no_cross = make_edges(
        protein_residues, ligand_atoms, ligand_bonds
    )

    return ComplexSample(
        protein_residues=protein_residues,
        ligand_atoms=ligand_atoms,
        ligand_bonds=ligand_bonds,
        ligand_signatures=ligand_signatures,
        native_coords=native_coords,
        protein_count=protein_count,
        ligand_count=ligand_count,
        node_types=node_types,
        token_indices=token_indices,
        token_groups=token_groups,
        sequence_positions=sequence_positions,
        edges_full=edges_full,
        edge_types_full=edge_types_full,
        edges_no_cross=edges_no_cross,
        edge_types_no_cross=edge_types_no_cross,
    )


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def generate_decoys(
    native_coords: np.ndarray,
    sample_count: int,
    seed: int,
    sigma_range: Tuple[float, float] = (0.25, 2.25),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean = np.zeros((sample_count, native_coords.shape[0], 3), dtype=np.float32)
    noisy = np.zeros_like(clean)
    sigmas = np.zeros(sample_count, dtype=np.float32)
    for i in range(sample_count):
        rotation = random_rotation_matrix(rng)
        translation = rng.normal(scale=2.0, size=(1, 3)).astype(np.float32)
        transformed = native_coords @ rotation.T + translation
        sigma = rng.uniform(*sigma_range)
        clean[i] = transformed
        noisy[i] = transformed + rng.normal(scale=sigma, size=transformed.shape).astype(np.float32)
        sigmas[i] = sigma
    return clean, noisy, sigmas


class ComplexDecoyDataset(torch.utils.data.Dataset):
    def __init__(self, clean: np.ndarray, noisy: np.ndarray, sigmas: np.ndarray):
        self.clean = torch.from_numpy(clean)
        self.noisy = torch.from_numpy(noisy)
        self.sigmas = torch.from_numpy(sigmas)

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "clean": self.clean[idx],
            "noisy": self.noisy[idx],
            "sigma": self.sigmas[idx],
        }


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, 1))
    return emb


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_type_count: int, time_dim: int, rbf_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rbf_dim = rbf_dim
        self.edge_emb = nn.Embedding(edge_type_count, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + time_dim + rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        centers = torch.linspace(0.0, 12.0, rbf_dim)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 0.5

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances[..., None] - self.rbf_centers[None, None, :]
        return torch.exp(-self.rbf_gamma * diff * diff)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        x_src = x[:, src, :]
        x_dst = x[:, dst, :]
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        rel = x_src - x_dst
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        rbf = self._rbf(dist)
        edge_feat = self.edge_emb(edge_types).unsqueeze(0).expand(x.shape[0], -1, -1)
        t_feat = t_emb.unsqueeze(1).expand(-1, edge_index.shape[0], -1)

        msg_input = torch.cat([h_src, h_dst, edge_feat, t_feat, rbf], dim=-1)
        messages = self.edge_mlp(msg_input)
        coord_scale = self.coord_mlp(messages) / (dist.unsqueeze(-1) + 1.0)

        new_x = x.clone()
        new_h = h.clone()
        for batch_idx in range(x.shape[0]):
            coord_agg = torch.zeros_like(x[batch_idx])
            feat_agg = torch.zeros_like(h[batch_idx])
            coord_agg.index_add_(0, src, coord_scale[batch_idx] * rel[batch_idx])
            feat_agg.index_add_(0, src, messages[batch_idx])
            new_x[batch_idx] = x[batch_idx] + coord_agg
            new_h[batch_idx] = h[batch_idx] + self.node_mlp(
                torch.cat([h[batch_idx], feat_agg], dim=-1)
            )

        return new_x, new_h


class UnifiedDiffusionRefiner(nn.Module):
    def __init__(
        self,
        protein_vocab: int,
        nucleotide_vocab: int,
        ligand_vocab: int,
        hidden_dim: int = 96,
        layers: int = 4,
        time_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.protein_emb = nn.Embedding(protein_vocab, hidden_dim)
        self.nucleotide_emb = nn.Embedding(max(1, nucleotide_vocab), hidden_dim)
        self.ligand_emb = nn.Embedding(ligand_vocab, hidden_dim)
        self.modality_emb = nn.Embedding(3, hidden_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_dim),
        )
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_type_count=5, time_dim=time_dim) for _ in range(layers)]
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def initial_features(
        self,
        token_groups: torch.Tensor,
        token_indices: torch.Tensor,
        node_types: torch.Tensor,
        sequence_positions: torch.Tensor,
    ) -> torch.Tensor:
        device = token_indices.device
        features = torch.zeros(token_indices.shape[0], self.hidden_dim, device=device)
        protein_mask = token_groups == 0
        nucleotide_mask = token_groups == 1
        ligand_mask = token_groups == 2
        if protein_mask.any():
            features[protein_mask] = self.protein_emb(token_indices[protein_mask])
        if nucleotide_mask.any():
            features[nucleotide_mask] = self.nucleotide_emb(token_indices[nucleotide_mask].clamp_min(0))
        if ligand_mask.any():
            features[ligand_mask] = self.ligand_emb(token_indices[ligand_mask])
        features = features + self.modality_emb(node_types)
        features = features + self.position_mlp(sequence_positions[:, None])
        return features

    def forward(
        self,
        noisy_coords: torch.Tensor,
        sigma: torch.Tensor,
        token_groups: torch.Tensor,
        token_indices: torch.Tensor,
        node_types: torch.Tensor,
        sequence_positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        h0 = self.initial_features(token_groups, token_indices, node_types, sequence_positions)
        h = h0.unsqueeze(0).expand(noisy_coords.shape[0], -1, -1).clone()
        x = noisy_coords
        t_emb = self.time_mlp(timestep_embedding(sigma, self.time_dim))
        for layer in self.layers:
            x, h = layer(x, h, edge_index, edge_types, t_emb)
        delta = self.out_mlp(h)
        return x + delta


def weighted_coordinate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    protein_count: int,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    protein_loss = F.mse_loss(pred[:, :protein_count, :], target[:, :protein_count, :])
    ligand_loss = F.mse_loss(pred[:, protein_count:, :], target[:, protein_count:, :])
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    pred_dist = torch.linalg.norm(pred[:, src, :] - pred[:, dst, :], dim=-1)
    true_dist = torch.linalg.norm(target[:, src, :] - target[:, dst, :], dim=-1)
    distance_loss = F.mse_loss(pred_dist, true_dist)
    return protein_loss + 1.5 * ligand_loss + 0.1 * distance_loss


def kabsch_align(
    pred: np.ndarray, ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_center = pred.mean(axis=0, keepdims=True)
    ref_center = ref.mean(axis=0, keepdims=True)
    pred0 = pred - pred_center
    ref0 = ref - ref_center
    cov = pred0.T @ ref0
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    rot = u @ correction @ vt
    aligned = pred0 @ rot + ref_center
    return aligned, rot, ref_center - pred_center @ rot


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def symmetry_aware_ligand_rmsd(
    pred: np.ndarray, ref: np.ndarray, signatures: Sequence[str]
) -> float:
    n = len(signatures)
    cost = np.full((n, n), 1e6, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if signatures[i] == signatures[j]:
                cost[i, j] = np.linalg.norm(pred[i] - ref[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    return rmsd(pred[row_ind], ref[col_ind])


def noise_bin_label(sigma: float) -> str:
    if sigma < 0.9:
        return "low"
    if sigma < 1.6:
        return "mid"
    return "high"


def train_model(
    model_name: str,
    edges: np.ndarray,
    edge_types: np.ndarray,
    sample: ComplexSample,
    train_set: ComplexDecoyDataset,
    val_set: ComplexDecoyDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[UnifiedDiffusionRefiner, List[Dict[str, float]]]:
    model = UnifiedDiffusionRefiner(
        protein_vocab=len(AA_TO_INDEX),
        nucleotide_vocab=4,
        ligand_vocab=len(ELEMENTS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    token_groups = torch.from_numpy(sample.token_groups).to(device)
    token_indices = torch.from_numpy(sample.token_indices).to(device)
    node_types = torch.from_numpy(sample.node_types).to(device)
    sequence_positions = torch.from_numpy(sample.sequence_positions).to(device)
    edge_index = torch.from_numpy(edges).to(device)
    edge_type_tensor = torch.from_numpy(edge_types).to(device)

    best_state = None
    best_val = float("inf")
    patience = 12
    patience_left = patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            sigma = batch["sigma"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(
                noisy,
                sigma,
                token_groups,
                token_indices,
                node_types,
                sequence_positions,
                edge_index,
                edge_type_tensor,
            )
            loss = weighted_coordinate_loss(pred, clean, sample.protein_count, edge_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                sigma = batch["sigma"].to(device)
                pred = model(
                    noisy,
                    sigma,
                    token_groups,
                    token_indices,
                    node_types,
                    sequence_positions,
                    edge_index,
                    edge_type_tensor,
                )
                loss = weighted_coordinate_loss(pred, clean, sample.protein_count, edge_index)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def evaluate_model(
    model_name: str,
    predictor,
    sample: ComplexSample,
    test_set: ComplexDecoyDataset,
    device: torch.device,
) -> List[Dict[str, float]]:
    protein_count = sample.protein_count
    ligand_signatures = sample.ligand_signatures
    results: List[Dict[str, float]] = []

    token_groups = torch.from_numpy(sample.token_groups).to(device)
    token_indices = torch.from_numpy(sample.token_indices).to(device)
    node_types = torch.from_numpy(sample.node_types).to(device)
    sequence_positions = torch.from_numpy(sample.sequence_positions).to(device)
    full_edges = torch.from_numpy(sample.edges_full).to(device)
    full_edge_types = torch.from_numpy(sample.edge_types_full).to(device)

    for idx in range(len(test_set)):
        item = test_set[idx]
        clean = item["clean"].numpy()
        noisy = item["noisy"].numpy()
        sigma = float(item["sigma"].item())

        pred = predictor(
            noisy=noisy,
            sigma=sigma,
            token_groups=token_groups,
            token_indices=token_indices,
            node_types=node_types,
            sequence_positions=sequence_positions,
            edge_index=full_edges,
            edge_types=full_edge_types,
        )

        pred_protein = pred[:protein_count]
        clean_protein = clean[:protein_count]
        pred_aligned_protein, rot, shift = kabsch_align(pred_protein, clean_protein)
        pred_aligned_ligand = pred[protein_count:] @ rot + shift
        clean_ligand = clean[protein_count:]

        protein_rmsd = rmsd(pred_aligned_protein, clean_protein)
        ligand_rmsd = symmetry_aware_ligand_rmsd(pred_aligned_ligand, clean_ligand, ligand_signatures)

        results.append(
            {
                "model": model_name,
                "sample_index": idx,
                "sigma": sigma,
                "noise_bin": noise_bin_label(sigma),
                "protein_rmsd": protein_rmsd,
                "ligand_rmsd": ligand_rmsd,
            }
        )
    return results


def summarize_results(records: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    models = sorted({record["model"] for record in records})
    for model in models:
        subset = [r for r in records if r["model"] == model]
        summary[model] = {
            "protein_rmsd_mean": float(np.mean([r["protein_rmsd"] for r in subset])),
            "protein_rmsd_median": float(np.median([r["protein_rmsd"] for r in subset])),
            "ligand_rmsd_mean": float(np.mean([r["ligand_rmsd"] for r in subset])),
            "ligand_rmsd_median": float(np.median([r["ligand_rmsd"] for r in subset])),
        }
    return summary


def write_csv(path: Path, records: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def plot_data_overview(sample: ComplexSample, summary_path: Path) -> None:
    protein_coords = np.stack([r.coord for r in sample.protein_residues], axis=0)
    ligand_coords = np.stack([a.coord for a in sample.ligand_atoms], axis=0)
    cross_dist = pairwise_distances(protein_coords, ligand_coords)
    min_dist = cross_dist.min(axis=1)
    contact_mask = min_dist <= 8.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(
        x=["Protein Cα residues", "Ligand heavy atoms", "Cross-modal edges"],
        y=[
            sample.protein_count,
            sample.ligand_count,
            int(np.sum(sample.edge_types_full == EDGE_CROSS) // 2),
        ],
        palette=["#1b9e77", "#d95f02", "#7570b3"],
        ax=axes[0],
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Complex Components")
    axes[0].tick_params(axis="x", rotation=15)

    residue_numbers = [r.resseq for r in sample.protein_residues]
    axes[1].plot(residue_numbers, min_dist, color="#1f78b4", linewidth=2)
    axes[1].axhline(8.0, color="#e31a1c", linestyle="--", linewidth=1.5, label="8 Å contact cutoff")
    axes[1].fill_between(
        residue_numbers,
        min_dist,
        8.0,
        where=contact_mask,
        alpha=0.3,
        color="#a6cee3",
        interpolate=True,
    )
    axes[1].set_title("Residue Proximity To Ligand")
    axes[1].set_xlabel("Protein residue index")
    axes[1].set_ylabel("Min CA-heavy atom distance (Å)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_complex_overlay(sample: ComplexSample, path: Path) -> None:
    protein_coords = np.stack([r.coord for r in sample.protein_residues], axis=0)
    ligand_coords = np.stack([a.coord for a in sample.ligand_atoms], axis=0)
    cross_dist = pairwise_distances(protein_coords, ligand_coords)
    min_dist = cross_dist.min(axis=1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        protein_coords[:, 0],
        protein_coords[:, 1],
        protein_coords[:, 2],
        color="#808080",
        linewidth=1.2,
        alpha=0.7,
    )
    scatter = ax.scatter(
        protein_coords[:, 0],
        protein_coords[:, 1],
        protein_coords[:, 2],
        c=min_dist,
        cmap="viridis_r",
        s=18,
        label="Protein Cα",
    )
    ax.scatter(
        ligand_coords[:, 0],
        ligand_coords[:, 1],
        ligand_coords[:, 2],
        color="#d95f02",
        s=28,
        alpha=0.85,
        label="Ligand heavy atoms",
    )
    ax.set_title("Native 2L3R Protein-Ligand Geometry")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.legend(frameon=False, loc="upper left")
    fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08, label="Residue-ligand distance (Å)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Sequence[Dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in [("identity-free", "#d95f02"), ("cross-modal", "#1b9e77")]:
        subset = [row for row in history if row["model"] == model_name]
        ax.plot([r["epoch"] for r in subset], [r["train_loss"] for r in subset], color=color, alpha=0.4)
        ax.plot(
            [r["epoch"] for r in subset],
            [r["val_loss"] for r in subset],
            color=color,
            linewidth=2.0,
            label=f"{model_name} val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training And Validation Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(records: Sequence[Dict[str, float]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = ["identity", "identity-free", "cross-modal"]
    palette = {"identity": "#bdbdbd", "identity-free": "#fc8d62", "cross-modal": "#66c2a5"}

    protein_data = [[r["protein_rmsd"] for r in records if r["model"] == model] for model in order]
    ligand_data = [[r["ligand_rmsd"] for r in records if r["model"] == model] for model in order]

    bp0 = axes[0].boxplot(protein_data, tick_labels=order, patch_artist=True)
    bp1 = axes[1].boxplot(ligand_data, tick_labels=order, patch_artist=True)
    for patch, model in zip(bp0["boxes"], order):
        patch.set(facecolor=palette[model], alpha=0.8)
    for patch, model in zip(bp1["boxes"], order):
        patch.set(facecolor=palette[model], alpha=0.8)

    axes[0].set_title("Protein Cα RMSD")
    axes[0].set_ylabel("Å")
    axes[1].set_title("Ligand Heavy-Atom RMSD")
    axes[1].set_ylabel("Å")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_noise_stratified(records: Sequence[Dict[str, float]], path: Path) -> None:
    order = ["identity", "identity-free", "cross-modal"]
    bins = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for metric, ax in [("protein_rmsd", axes[0]), ("ligand_rmsd", axes[1])]:
        for offset, model in enumerate(order):
            medians = []
            for noise_bin in bins:
                subset = [
                    row[metric]
                    for row in records
                    if row["model"] == model and row["noise_bin"] == noise_bin
                ]
                medians.append(float(np.median(subset)))
            x = np.arange(len(bins)) + (offset - 1) * 0.08
            ax.plot(x, medians, marker="o", linewidth=2, label=model)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(bins)
        ax.set_ylabel("Median RMSD (Å)")
    axes[0].set_title("Protein Recovery By Noise Level")
    axes[1].set_title("Ligand Recovery By Noise Level")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_overlay(
    sample: ComplexSample,
    test_set: ComplexDecoyDataset,
    predictors: Dict[str, callable],
    device: torch.device,
    path: Path,
) -> None:
    token_groups = torch.from_numpy(sample.token_groups).to(device)
    token_indices = torch.from_numpy(sample.token_indices).to(device)
    node_types = torch.from_numpy(sample.node_types).to(device)
    sequence_positions = torch.from_numpy(sample.sequence_positions).to(device)
    full_edges = torch.from_numpy(sample.edges_full).to(device)
    full_edge_types = torch.from_numpy(sample.edge_types_full).to(device)
    protein_count = sample.protein_count

    best_idx = None
    best_gain = -1.0
    predictions = {}
    for idx in range(len(test_set)):
        item = test_set[idx]
        clean = item["clean"].numpy()
        noisy = item["noisy"].numpy()
        sigma = float(item["sigma"].item())
        cross_pred = predictors["cross-modal"](
            noisy=noisy,
            sigma=sigma,
            token_groups=token_groups,
            token_indices=token_indices,
            node_types=node_types,
            sequence_positions=sequence_positions,
            edge_index=full_edges,
            edge_types=full_edge_types,
        )
        base_rmsd = symmetry_aware_ligand_rmsd(
            noisy[protein_count:], clean[protein_count:], sample.ligand_signatures
        )
        cross_rmsd = symmetry_aware_ligand_rmsd(
            cross_pred[protein_count:], clean[protein_count:], sample.ligand_signatures
        )
        gain = base_rmsd - cross_rmsd
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
            predictions = {
                "clean": clean,
                "noisy": noisy,
                "cross-modal": cross_pred,
            }

    assert best_idx is not None
    clean = predictions["clean"]
    noisy = predictions["noisy"]
    cross_pred = predictions["cross-modal"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dims, title in [
        (axes[0], (0, 1), "x-y projection"),
        (axes[1], (0, 2), "x-z projection"),
    ]:
        ax.plot(
            clean[:protein_count, dims[0]],
            clean[:protein_count, dims[1]],
            color="#808080",
            linewidth=1.0,
            alpha=0.5,
        )
        ax.scatter(
            noisy[protein_count:, dims[0]],
            noisy[protein_count:, dims[1]],
            color="#e31a1c",
            s=22,
            alpha=0.55,
            label="Noisy ligand",
        )
        ax.scatter(
            cross_pred[protein_count:, dims[0]],
            cross_pred[protein_count:, dims[1]],
            color="#1b9e77",
            s=20,
            alpha=0.8,
            label="Predicted ligand",
        )
        ax.scatter(
            clean[protein_count:, dims[0]],
            clean[protein_count:, dims[1]],
            color="#1f78b4",
            s=16,
            alpha=0.8,
            label="Native ligand",
        )
        ax.set_title(title)
        ax.set_xlabel(f"{['x', 'y', 'z'][dims[0]]} (Å)")
        ax.set_ylabel(f"{['x', 'y', 'z'][dims[1]]} (Å)")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(f"Example ligand recovery on held-out decoy #{best_idx}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-samples", type=int, default=1400)
    parser.add_argument("--val-samples", type=int, default=240)
    parser.add_argument("--test-samples", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    sample = build_complex_sample()
    device = select_device()

    train_clean, train_noisy, train_sigmas = generate_decoys(
        sample.native_coords, args.train_samples, seed=args.seed
    )
    val_clean, val_noisy, val_sigmas = generate_decoys(
        sample.native_coords, args.val_samples, seed=args.seed + 1
    )
    test_clean, test_noisy, test_sigmas = generate_decoys(
        sample.native_coords, args.test_samples, seed=args.seed + 2
    )

    train_set = ComplexDecoyDataset(train_clean, train_noisy, train_sigmas)
    val_set = ComplexDecoyDataset(val_clean, val_noisy, val_sigmas)
    test_set = ComplexDecoyDataset(test_clean, test_noisy, test_sigmas)

    identity_free_model, history_no_cross = train_model(
        model_name="identity-free",
        edges=sample.edges_no_cross,
        edge_types=sample.edge_types_no_cross,
        sample=sample,
        train_set=train_set,
        val_set=val_set,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    cross_modal_model, history_cross = train_model(
        model_name="cross-modal",
        edges=sample.edges_full,
        edge_types=sample.edge_types_full,
        sample=sample,
        train_set=train_set,
        val_set=val_set,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    history = history_no_cross + history_cross

    def make_predictor(model: UnifiedDiffusionRefiner):
        model.eval()

        def predictor(
            noisy: np.ndarray,
            sigma: float,
            token_groups: torch.Tensor,
            token_indices: torch.Tensor,
            node_types: torch.Tensor,
            sequence_positions: torch.Tensor,
            edge_index: torch.Tensor,
            edge_types: torch.Tensor,
        ) -> np.ndarray:
            with torch.no_grad():
                pred = model(
                    torch.from_numpy(noisy[None, ...]).to(device),
                    torch.tensor([sigma], dtype=torch.float32, device=device),
                    token_groups,
                    token_indices,
                    node_types,
                    sequence_positions,
                    edge_index,
                    edge_types,
                )
            return pred[0].detach().cpu().numpy()

        return predictor

    predictors = {
        "identity-free": make_predictor(identity_free_model),
        "cross-modal": make_predictor(cross_modal_model),
        "identity": lambda noisy, **_: noisy,
    }

    results = []
    for model_name in ["identity", "identity-free", "cross-modal"]:
        results.extend(
            evaluate_model(
                model_name=model_name,
                predictor=predictors[model_name],
                sample=sample,
                test_set=test_set,
                device=device,
            )
        )

    summary = summarize_results(results)
    contact_residues = sum(
        1
        for residue in sample.protein_residues
        if np.min(
            np.linalg.norm(
                residue.coord[None, :] - np.stack([a.coord for a in sample.ligand_atoms], axis=0),
                axis=-1,
            )
        )
        <= 8.0
    )
    metadata = {
        "device": str(device),
        "seed": args.seed,
        "protein_ca_residues": sample.protein_count,
        "ligand_heavy_atoms": sample.ligand_count,
        "ligand_bonds": len(sample.ligand_bonds),
        "cross_modal_edges": int(np.sum(sample.edge_types_full == EDGE_CROSS) // 2),
        "contact_residues_within_8A": contact_residues,
        "note": (
            "The task description states that the protein file contains 107 CA-only residues, "
            "but the actual file contains 161 CA atoms with full-atom records in the original chain segment. "
            "This study uses the on-disk structure as the source of truth."
        ),
        "summary_metrics": summary,
    }

    write_csv(
        OUTPUT_DIR / "training_history.csv",
        history,
        fieldnames=["model", "epoch", "train_loss", "val_loss"],
    )
    write_csv(
        OUTPUT_DIR / "evaluation_metrics.csv",
        results,
        fieldnames=["model", "sample_index", "sigma", "noise_bin", "protein_rmsd", "ligand_rmsd"],
    )
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(metadata, indent=2))
    torch.save(identity_free_model.state_dict(), OUTPUT_DIR / "identity_free_model.pt")
    torch.save(cross_modal_model.state_dict(), OUTPUT_DIR / "cross_modal_model.pt")

    plot_data_overview(sample, IMAGE_DIR / "data_overview.png")
    plot_complex_overlay(sample, IMAGE_DIR / "native_complex_geometry.png")
    plot_training_curves(history, IMAGE_DIR / "training_curves.png")
    plot_metric_comparison(results, IMAGE_DIR / "rmsd_comparison.png")
    plot_noise_stratified(results, IMAGE_DIR / "noise_stratified_performance.png")
    plot_example_overlay(sample, test_set, predictors, device, IMAGE_DIR / "example_ligand_recovery.png")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
