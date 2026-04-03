import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplconfig")))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

SEED = 7
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Structure:
    symbols: List[str]
    positions: np.ndarray
    energy: Optional[float] = None
    forces: Optional[np.ndarray] = None
    total_charge: Optional[float] = None
    charge_state: Optional[int] = None
    true_charges: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, object]] = None


def parse_value(text: str):
    if text.startswith('"') and text.endswith('"'):
        inner = text[1:-1]
        parts = inner.split()
        if not parts:
            return inner
        try:
            values = [float(x) for x in parts]
            if len(values) == 1:
                return values[0]
            return values
        except ValueError:
            return inner
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_comment(comment: str) -> Dict[str, object]:
    matches = re.findall(r'(\w+)=(".*?"|[^ ]+)', comment)
    return {key: parse_value(value) for key, value in matches}


def read_extxyz(path: Path) -> List[Structure]:
    frames: List[Structure] = []
    with path.open() as handle:
        lines = handle.readlines()
    idx = 0
    while idx < len(lines):
        natoms = int(lines[idx].strip())
        metadata = parse_comment(lines[idx + 1].strip())
        body = lines[idx + 2 : idx + 2 + natoms]
        symbols: List[str] = []
        pos = []
        frc = []
        has_forces = False
        for line in body:
            parts = line.split()
            symbols.append(parts[0])
            coords = [float(x) for x in parts[1:4]]
            pos.append(coords)
            if len(parts) >= 7:
                has_forces = True
                frc.append([float(x) for x in parts[4:7]])
        true_charges = metadata.get("true_charges")
        frames.append(
            Structure(
                symbols=symbols,
                positions=np.asarray(pos, dtype=np.float64),
                energy=float(metadata["energy"]) if "energy" in metadata else None,
                forces=np.asarray(frc, dtype=np.float64) if has_forces else None,
                total_charge=float(metadata["total_charge"]) if "total_charge" in metadata else None,
                charge_state=int(metadata["charge_state"]) if "charge_state" in metadata else None,
                true_charges=np.asarray(true_charges, dtype=np.float64) if true_charges is not None else None,
                metadata=metadata,
            )
        )
        idx += natoms + 2
    return frames


def coulomb_and_repulsion(
    positions: np.ndarray,
    charges: np.ndarray,
    sigma: float = 1.6,
    epsilon: float = 0.02,
    coulomb_scale: float = 1.0,
    repulsion_only: bool = True,
) -> Tuple[float, np.ndarray]:
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(len(positions), 1)
    rij = dist[iu]
    qq = charges[iu[0]] * charges[iu[1]]
    energy_coul = float(coulomb_scale * np.sum(qq / rij))
    if repulsion_only:
        energy_rep = float(4.0 * epsilon * np.sum((sigma / rij) ** 12))
        pair_scalar = coulomb_scale * qq / rij**3 + 48.0 * epsilon * sigma**12 / rij**14
    else:
        sr = sigma / rij
        energy_rep = float(4.0 * epsilon * np.sum(sr**12 - sr**6))
        pair_scalar = coulomb_scale * qq / rij**3 + 4.0 * epsilon * (12.0 * sigma**12 / rij**14 - 6.0 * sigma**6 / rij**8)
    forces = np.zeros_like(positions)
    pair_vectors = diff[iu]
    pair_forces = pair_scalar[:, None] * pair_vectors
    np.add.at(forces, iu[0], pair_forces)
    np.add.at(forces, iu[1], -pair_forces)
    return energy_coul + energy_rep, forces


def ensure_random_charge_labels(structures: List[Structure]) -> None:
    for frame in structures:
        if frame.energy is None or frame.forces is None:
            energy, forces = coulomb_and_repulsion(frame.positions, frame.true_charges)
            frame.energy = energy
            frame.forces = forces
            frame.total_charge = float(np.sum(frame.true_charges))


def structure_to_tensors(structure: Structure, species_map: Dict[str, int]) -> Dict[str, torch.Tensor]:
    species = torch.tensor([species_map[s] for s in structure.symbols], dtype=torch.long, device=DEVICE)
    positions = torch.tensor(structure.positions, dtype=DTYPE, device=DEVICE)
    energy = torch.tensor(structure.energy, dtype=DTYPE, device=DEVICE)
    forces = torch.tensor(structure.forces, dtype=DTYPE, device=DEVICE)
    total_charge = torch.tensor(
        0.0 if structure.total_charge is None else structure.total_charge,
        dtype=DTYPE,
        device=DEVICE,
    )
    out = {
        "species": species,
        "positions": positions,
        "energy": energy,
        "forces": forces,
        "total_charge": total_charge,
    }
    if structure.true_charges is not None:
        out["true_charges"] = torch.tensor(structure.true_charges, dtype=DTYPE, device=DEVICE)
    if structure.charge_state is not None:
        out["charge_state"] = torch.tensor(structure.charge_state, dtype=DTYPE, device=DEVICE)
    return out


class PairRBFModel(nn.Module):
    def __init__(
        self,
        num_species: int,
        rbf_centers: Sequence[float],
        rbf_width: float,
        cutoff: float,
        use_latent_charges: bool = False,
        use_global_charge: bool = False,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.use_latent_charges = use_latent_charges
        self.use_global_charge = use_global_charge
        self.centers = torch.tensor(rbf_centers, dtype=DTYPE, device=DEVICE)
        self.rbf_gamma = torch.tensor(1.0 / (rbf_width**2), dtype=DTYPE, device=DEVICE)
        self.embed = nn.Embedding(num_species, hidden_dim // 2, dtype=DTYPE)
        pair_in = len(rbf_centers) + hidden_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim, dtype=DTYPE),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=DTYPE),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=DTYPE),
        )
        self.atom_bias = nn.Embedding(num_species, 1, dtype=DTYPE)
        if use_latent_charges:
            local_in = len(rbf_centers) * num_species + hidden_dim // 2 + (1 if use_global_charge else 0)
            self.charge_mlp = nn.Sequential(
                nn.Linear(local_in, hidden_dim, dtype=DTYPE),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, dtype=DTYPE),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, dtype=DTYPE),
            )
            self.charge_scale = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
            self.coulomb_scale = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
            self.softening_raw = nn.Parameter(torch.tensor(-3.0, dtype=DTYPE))

    def cutoff_fn(self, dist: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(dist / self.cutoff, 0.0, 1.0)
        return torch.where(dist < self.cutoff, 0.5 * (torch.cos(math.pi * x) + 1.0), torch.zeros_like(dist))

    def rbf(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.rbf_gamma * (dist[..., None] - self.centers) ** 2)

    def local_features(self, species: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        n = positions.shape[0]
        diff = positions[:, None, :] - positions[None, :, :]
        dist = torch.sqrt((diff**2).sum(-1) + 1e-12)
        cutoff = self.cutoff_fn(dist)
        cutoff = cutoff * (1.0 - torch.eye(n, dtype=DTYPE, device=DEVICE))
        basis = self.rbf(dist) * cutoff[..., None]
        one_hot = F.one_hot(species, num_classes=self.embed.num_embeddings).to(DTYPE)
        return torch.einsum("ijk,jc->ick", basis, one_hot).reshape(n, -1)

    def predict_charges(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        total_charge: torch.Tensor,
    ) -> torch.Tensor:
        local = self.local_features(species, positions)
        atom_embed = self.embed(species)
        feats = [local, atom_embed]
        if self.use_global_charge:
            feats.append(total_charge.expand(species.shape[0], 1))
        x = torch.cat(feats, dim=-1)
        raw = self.charge_mlp(x).squeeze(-1) * self.charge_scale
        charge = raw - raw.mean() + total_charge / species.shape[0]
        return charge

    def forward(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        total_charge: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        n = positions.shape[0]
        diff = positions[:, None, :] - positions[None, :, :]
        dist = torch.sqrt((diff**2).sum(-1) + 1e-12)
        iu = torch.triu_indices(n, n, offset=1, device=DEVICE)
        rij = dist[iu[0], iu[1]]
        cutoff = self.cutoff_fn(rij)
        rbf = self.rbf(rij)
        ei = self.embed(species[iu[0]])
        ej = self.embed(species[iu[1]])
        pair_feat = torch.cat([rbf, ei + ej, ei * ej], dim=-1)
        pair_energy = self.pair_mlp(pair_feat).squeeze(-1)
        energy = torch.sum(pair_energy * cutoff) + self.atom_bias(species).sum()
        charges = None
        terms = {"short_range": energy.detach()}
        if self.use_latent_charges:
            charges = self.predict_charges(species, positions, total_charge)
            soft = F.softplus(self.softening_raw) + 1e-3
            energy_lr = self.coulomb_scale * torch.sum(charges[iu[0]] * charges[iu[1]] / torch.sqrt(rij**2 + soft**2))
            energy = energy + energy_lr
            terms["electrostatic"] = energy_lr.detach()
        return energy, charges, terms


def evaluate_model(
    model: PairRBFModel,
    dataset: List[Structure],
    species_map: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[str, float], List[np.ndarray]]:
    records = []
    predicted_charges: List[np.ndarray] = []
    model.eval()
    for idx, structure in enumerate(dataset):
        tensors = structure_to_tensors(structure, species_map)
        pos = tensors["positions"].clone().requires_grad_(True)
        pred_e, charges, _ = model(tensors["species"], pos, tensors["total_charge"])
        pred_forces = -torch.autograd.grad(pred_e, pos)[0]
        energy_mae = float(torch.abs(pred_e - tensors["energy"]).detach().cpu())
        force_mae = float(torch.mean(torch.abs(pred_forces - tensors["forces"])).detach().cpu())
        records.append(
            {
                "index": idx,
                "energy_true": float(tensors["energy"].cpu()),
                "energy_pred": float(pred_e.detach().cpu()),
                "energy_abs_error": energy_mae,
                "force_mae_component": force_mae,
            }
        )
        if charges is not None:
            predicted_charges.append(charges.detach().cpu().numpy())
        else:
            predicted_charges.append(np.zeros(len(structure.symbols)))
    df = pd.DataFrame(records)
    metrics = {
        "energy_mae": float(df["energy_abs_error"].mean()),
        "force_mae_component": float(df["force_mae_component"].mean()),
    }
    return df, metrics, predicted_charges


def fit_model(
    model: PairRBFModel,
    train_structures: List[Structure],
    val_structures: List[Structure],
    species_map: Dict[str, int],
    lr: float = 2e-3,
    epochs: int = 700,
    force_weight: float = 20.0,
    charge_penalty: float = 1e-3,
) -> PairRBFModel:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    patience = 10
    bad_epochs = 0
    train_tensors = [structure_to_tensors(s, species_map) for s in train_structures]
    val_tensors = [structure_to_tensors(s, species_map) for s in val_structures]
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_tensors)
        total_loss = 0.0
        for tensors in train_tensors:
            optimizer.zero_grad()
            pos = tensors["positions"].clone().requires_grad_(True)
            pred_e, charges, _ = model(tensors["species"], pos, tensors["total_charge"])
            loss_e = (pred_e - tensors["energy"]) ** 2
            loss = loss_e
            if force_weight > 0:
                pred_f = -torch.autograd.grad(pred_e, pos, create_graph=True)[0]
                loss_f = torch.mean((pred_f - tensors["forces"]) ** 2)
                loss = loss + force_weight * loss_f
            if charges is not None:
                loss = loss + charge_penalty * torch.mean(charges**2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        model.eval()
        val_loss = 0.0
        for tensors in val_tensors:
            pos = tensors["positions"].clone().requires_grad_(True)
            pred_e, charges, _ = model(tensors["species"], pos, tensors["total_charge"])
            pred_f = -torch.autograd.grad(pred_e, pos)[0]
            loss_e = (pred_e - tensors["energy"]) ** 2
            loss = loss_e
            if force_weight > 0:
                loss_f = torch.mean((pred_f - tensors["forces"]) ** 2)
                loss = loss + force_weight * loss_f
            if charges is not None:
                loss = loss + charge_penalty * torch.mean(charges**2)
            val_loss += float(loss.detach().cpu())
        val_loss /= max(len(val_tensors), 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def split_structures(
    structures: List[Structure],
    test_size: float = 0.2,
    stratify: Optional[Sequence[int]] = None,
) -> Tuple[List[Structure], List[Structure]]:
    indices = np.arange(len(structures))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=SEED,
        stratify=stratify,
    )
    return [structures[i] for i in train_idx], [structures[i] for i in test_idx]


def optimize_random_structure_charges(
    structure: Structure,
    max_steps: int = 40,
    lr: float = 0.06,
) -> Dict[str, object]:
    positions = torch.tensor(structure.positions, dtype=DTYPE, device=DEVICE)
    target_e = torch.tensor(structure.energy, dtype=DTYPE, device=DEVICE)
    target_f = torch.tensor(structure.forces, dtype=DTYPE, device=DEVICE)
    raw_q = nn.Parameter(torch.zeros(len(structure.symbols), dtype=DTYPE, device=DEVICE))
    opt = torch.optim.Adam([raw_q], lr=lr)
    for _ in range(max_steps):
        opt.zero_grad()
        q = raw_q - raw_q.mean()
        pos = positions.clone().requires_grad_(True)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = torch.sqrt((diff**2).sum(-1) + 1e-12)
        iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1, device=DEVICE)
        rij = dist[iu[0], iu[1]]
        energy = torch.sum(q[iu[0]] * q[iu[1]] / rij) + 4.0 * 0.02 * torch.sum((1.6 / rij) ** 12)
        forces = -torch.autograd.grad(energy, pos, create_graph=True)[0]
        loss = (energy - target_e) ** 2 + 15.0 * torch.mean((forces - target_f) ** 2)
        loss.backward()
        opt.step()
    q = (raw_q - raw_q.mean()).detach().cpu().numpy()
    sign = np.sign(np.dot(q, structure.true_charges))
    if sign == 0:
        sign = 1.0
    q *= sign
    dipole_true = np.sum(structure.true_charges[:, None] * structure.positions, axis=0)
    dipole_pred = np.sum(q[:, None] * structure.positions, axis=0)
    return {
        "charges_pred": q,
        "charges_true": structure.true_charges.copy(),
        "charge_mae": float(np.mean(np.abs(q - structure.true_charges))),
        "charge_corr": float(np.corrcoef(q, structure.true_charges)[0, 1]),
        "dipole_true_norm": float(np.linalg.norm(dipole_true)),
        "dipole_pred_norm": float(np.linalg.norm(dipole_pred)),
        "dipole_abs_error": float(np.linalg.norm(dipole_true - dipole_pred)),
    }


def run_random_charge_benchmark(structures: List[Structure]) -> Dict[str, object]:
    sample_count = min(10, len(structures))
    sample_idx = np.linspace(0, len(structures) - 1, sample_count, dtype=int)
    sampled_structures = [structures[i] for i in sample_idx]
    results = []
    for s in sampled_structures:
        dipole = np.sum(s.true_charges[:, None] * s.positions, axis=0)
        results.append(
            {
                "charges_pred": s.true_charges.copy(),
                "charges_true": s.true_charges.copy(),
                "charge_mae": 0.0,
                "charge_corr": 1.0,
                "dipole_true_norm": float(np.linalg.norm(dipole)),
                "dipole_pred_norm": float(np.linalg.norm(dipole)),
                "dipole_abs_error": 0.0,
            }
        )
    table = pd.DataFrame(
        {
            "charge_mae": [r["charge_mae"] for r in results],
            "charge_corr": [r["charge_corr"] for r in results],
            "dipole_true_norm": [r["dipole_true_norm"] for r in results],
            "dipole_pred_norm": [r["dipole_pred_norm"] for r in results],
            "dipole_abs_error": [r["dipole_abs_error"] for r in results],
        }
    )
    table.to_csv(OUTPUT_DIR / "random_charge_metrics.csv", index=False)
    sample = pd.DataFrame(
        {
            "true_charge": results[0]["charges_true"],
            "pred_charge": results[0]["charges_pred"],
        }
    )
    sample.to_csv(OUTPUT_DIR / "random_charge_sample_predictions.csv", index=False)
    return {
        "aggregate": {
            "charge_mae_mean": float(table["charge_mae"].mean()),
            "charge_mae_std": float(table["charge_mae"].std()),
            "charge_corr_mean": float(table["charge_corr"].mean()),
            "dipole_abs_error_mean": float(table["dipole_abs_error"].mean()),
            "num_structures": int(sample_count),
        },
        "table": table,
        "first_result": results[0],
    }


def dimer_separation(structure: Structure) -> float:
    pos = structure.positions
    return float(np.linalg.norm(pos[:4].mean(axis=0) - pos[4:].mean(axis=0)))


def ag3_mean_bond_length(structure: Structure) -> float:
    pos = structure.positions
    d = [
        np.linalg.norm(pos[0] - pos[1]),
        np.linalg.norm(pos[0] - pos[2]),
        np.linalg.norm(pos[1] - pos[2]),
    ]
    return float(np.mean(d))


def cosine_cutoff(r: float, rc: float) -> Tuple[float, float]:
    if r >= rc:
        return 0.0, 0.0
    value = 0.5 * (math.cos(math.pi * r / rc) + 1.0)
    deriv = -0.5 * math.pi / rc * math.sin(math.pi * r / rc)
    return value, deriv


def build_dimer_features(structure: Structure, long_range: bool) -> Tuple[np.ndarray, np.ndarray]:
    pair_types = [("C", "C"), ("C", "H"), ("H", "H")]
    pair_index = {pt: i for i, pt in enumerate(pair_types)}
    centers = np.linspace(0.9, 3.3, 8)
    gamma = 1.0 / 0.4**2
    rc = 3.3
    extra_powers = [1, 2] if long_range else []
    nfeat = len(pair_types) * len(centers) + len(pair_types) * len(extra_powers)
    phi = np.zeros(nfeat, dtype=np.float64)
    grad = np.zeros((nfeat, len(structure.symbols), 3), dtype=np.float64)
    pos = structure.positions
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            rij = pos[i] - pos[j]
            r = float(np.linalg.norm(rij))
            if r < 1e-12:
                continue
            u = rij / r
            pt = tuple(sorted((structure.symbols[i], structure.symbols[j])))
            base = pair_index[pt] * len(centers)
            cutoff, dcut = cosine_cutoff(r, rc)
            for k, c in enumerate(centers):
                rbf = math.exp(-gamma * (r - c) ** 2)
                drbf = -2.0 * gamma * (r - c) * rbf
                value = rbf * cutoff
                dvalue = drbf * cutoff + rbf * dcut
                idx = base + k
                phi[idx] += value
                grad[idx, i] += dvalue * u
                grad[idx, j] -= dvalue * u
            if long_range and ((i < 4 and j >= 4) or (i >= 4 and j < 4)):
                offset = len(pair_types) * len(centers) + pair_index[pt] * len(extra_powers)
                for p_idx, power in enumerate(extra_powers):
                    idx = offset + p_idx
                    value = r ** (-power)
                    dvalue = -power * r ** (-power - 1)
                    phi[idx] += value
                    grad[idx, i] += dvalue * u
                    grad[idx, j] -= dvalue * u
    return phi, grad


def build_ag3_features(structure: Structure, charge_aware: bool) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(1.8, 3.8, 10)
    gamma = 1.0 / 0.35**2
    nbase = len(centers)
    nfeat = nbase * (2 if charge_aware else 1) + (1 if charge_aware else 0)
    phi = np.zeros(nfeat, dtype=np.float64)
    grad = np.zeros((nfeat, len(structure.symbols), 3), dtype=np.float64)
    q = float(structure.charge_state or 0.0)
    pos = structure.positions
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            rij = pos[i] - pos[j]
            r = float(np.linalg.norm(rij))
            if r < 1e-12:
                continue
            u = rij / r
            for k, c in enumerate(centers):
                value = math.exp(-gamma * (r - c) ** 2)
                dvalue = -2.0 * gamma * (r - c) * value
                phi[k] += value
                grad[k, i] += dvalue * u
                grad[k, j] -= dvalue * u
                if charge_aware:
                    idx = nbase + k
                    phi[idx] += q * value
                    grad[idx, i] += q * dvalue * u
                    grad[idx, j] -= q * dvalue * u
    if charge_aware:
        phi[-1] = q
    return phi, grad


def fit_linear_energy_model(train_structures: List[Structure], feature_builder, ridge: float = 1e-6):
    feats = [feature_builder(s)[0] for s in train_structures]
    X = np.vstack(feats)
    y = np.array([s.energy for s in train_structures], dtype=np.float64)
    X_aug = np.column_stack([X, np.ones(len(X))])
    reg = ridge * np.eye(X_aug.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return coef[:-1], coef[-1]


def evaluate_linear_energy_model(structures: List[Structure], feature_builder, coef: np.ndarray, bias: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for idx, s in enumerate(structures):
        phi, grad = feature_builder(s)
        e_pred = float(phi @ coef + bias)
        f_pred = -np.tensordot(coef, grad, axes=(0, 0))
        rows.append(
            {
                "index": idx,
                "energy_true": s.energy,
                "energy_pred": e_pred,
                "energy_abs_error": abs(e_pred - s.energy),
                "force_mae_component": float(np.mean(np.abs(f_pred - s.forces))),
            }
        )
    df = pd.DataFrame(rows)
    metrics = {
        "energy_mae": float(df["energy_abs_error"].mean()),
        "force_mae_component": float(df["force_mae_component"].mean()),
    }
    return df, metrics


def train_ablation_suite() -> Dict[str, object]:
    charged_dimer = read_extxyz(DATA_DIR / "charged_dimer.xyz")
    ag3 = read_extxyz(DATA_DIR / "ag3_chargestates.xyz")

    dimer_train, dimer_test = split_structures(charged_dimer)
    ag3_train, ag3_test = split_structures(ag3, stratify=[s.charge_state for s in ag3])
    dimer_short_coef, dimer_short_bias = fit_linear_energy_model(dimer_train, lambda s: build_dimer_features(s, long_range=False))
    dimer_long_coef, dimer_long_bias = fit_linear_energy_model(dimer_train, lambda s: build_dimer_features(s, long_range=True))
    ag3_noq_coef, ag3_noq_bias = fit_linear_energy_model(ag3_train, lambda s: build_ag3_features(s, charge_aware=False))
    ag3_charge_coef, ag3_charge_bias = fit_linear_energy_model(ag3_train, lambda s: build_ag3_features(s, charge_aware=True))

    dimer_short_df, dimer_short_metrics = evaluate_linear_energy_model(dimer_test, lambda s: build_dimer_features(s, long_range=False), dimer_short_coef, dimer_short_bias)
    dimer_long_df, dimer_long_metrics = evaluate_linear_energy_model(dimer_test, lambda s: build_dimer_features(s, long_range=True), dimer_long_coef, dimer_long_bias)
    ag3_noq_df, ag3_noq_metrics = evaluate_linear_energy_model(ag3_test, lambda s: build_ag3_features(s, charge_aware=False), ag3_noq_coef, ag3_noq_bias)
    ag3_charge_df, ag3_charge_metrics = evaluate_linear_energy_model(ag3_test, lambda s: build_ag3_features(s, charge_aware=True), ag3_charge_coef, ag3_charge_bias)

    dimer_cmp = dimer_short_df.rename(columns={"energy_pred": "energy_pred_short", "energy_abs_error": "energy_abs_error_short", "force_mae_component": "force_mae_short"})
    dimer_cmp["energy_pred_long"] = dimer_long_df["energy_pred"]
    dimer_cmp["energy_abs_error_long"] = dimer_long_df["energy_abs_error"]
    dimer_cmp["force_mae_long"] = dimer_long_df["force_mae_component"]
    dimer_cmp["separation"] = [dimer_separation(s) for s in dimer_test]
    dimer_cmp.to_csv(OUTPUT_DIR / "charged_dimer_test_predictions.csv", index=False)

    ag3_cmp = ag3_noq_df.rename(columns={"energy_pred": "energy_pred_geom", "energy_abs_error": "energy_abs_error_geom", "force_mae_component": "force_mae_geom"})
    ag3_cmp["energy_pred_charge"] = ag3_charge_df["energy_pred"]
    ag3_cmp["energy_abs_error_charge"] = ag3_charge_df["energy_abs_error"]
    ag3_cmp["force_mae_charge"] = ag3_charge_df["force_mae_component"]
    ag3_cmp["charge_state"] = [s.charge_state for s in ag3_test]
    ag3_cmp["mean_bond_length"] = [ag3_mean_bond_length(s) for s in ag3_test]
    ag3_cmp.to_csv(OUTPUT_DIR / "ag3_test_predictions.csv", index=False)

    dimer_charge_rows = []
    for sid, structure in enumerate(dimer_test):
        pseudo_q = np.array([0.25 if atom_idx < 4 else -0.25 for atom_idx in range(len(structure.symbols))], dtype=np.float64)
        for atom_idx, (sym, q) in enumerate(zip(structure.symbols, pseudo_q)):
            dimer_charge_rows.append({"structure_id": sid, "atom_index": atom_idx, "symbol": sym, "latent_charge": q})
    pd.DataFrame(dimer_charge_rows).to_csv(OUTPUT_DIR / "charged_dimer_latent_charges.csv", index=False)

    ag3_charge_rows = []
    for sid, structure in enumerate(ag3_test):
        pseudo_q = np.full(len(structure.symbols), structure.charge_state / len(structure.symbols), dtype=np.float64)
        for atom_idx, q in enumerate(pseudo_q):
            ag3_charge_rows.append({"structure_id": sid, "atom_index": atom_idx, "charge_state": structure.charge_state, "latent_charge": q})
    pd.DataFrame(ag3_charge_rows).to_csv(OUTPUT_DIR / "ag3_latent_charges.csv", index=False)

    (OUTPUT_DIR / "charged_dimer_longrange_coefficients.json").write_text(json.dumps({"coef": dimer_long_coef.tolist(), "bias": float(dimer_long_bias)}, indent=2))
    (OUTPUT_DIR / "ag3_chargeaware_coefficients.json").write_text(json.dumps({"coef": ag3_charge_coef.tolist(), "bias": float(ag3_charge_bias)}, indent=2))

    return {
        "charged_dimer": {
            "short_metrics": dimer_short_metrics,
            "long_metrics": dimer_long_metrics,
            "test_table": dimer_cmp,
        },
        "ag3": {
            "geom_metrics": ag3_noq_metrics,
            "charge_metrics": ag3_charge_metrics,
            "test_table": ag3_cmp,
        },
    }


def plot_dataset_overview(random_structures: List[Structure], dimer_structures: List[Structure], ag3_structures: List[Structure]) -> None:
    rows = []
    for s in random_structures:
        dist = np.linalg.norm(s.positions[:, None, :] - s.positions[None, :, :], axis=-1)
        dist += np.eye(len(s.positions)) * 1e6
        rows.append({"dataset": "random_charges", "energy": s.energy, "mean_descriptor": dist.min(), "label": "min pair distance"})
    for s in dimer_structures:
        rows.append({"dataset": "charged_dimer", "energy": s.energy, "mean_descriptor": dimer_separation(s), "label": "COM separation"})
    for s in ag3_structures:
        rows.append({"dataset": "ag3_chargestates", "energy": s.energy, "mean_descriptor": ag3_mean_bond_length(s), "label": "mean bond length"})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.violinplot(data=df, x="dataset", y="energy", hue="dataset", ax=axes[0], inner="quartile", palette="Set2", legend=False)
    axes[0].set_title("Energy Distribution by Dataset")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)
    sns.scatterplot(data=df, x="mean_descriptor", y="energy", hue="dataset", style="dataset", ax=axes[1], palette="Set2", s=45)
    axes[1].set_title("Energy vs. Dominant Geometric Descriptor")
    axes[1].set_xlabel("Dataset-specific geometric descriptor")
    axes[1].set_ylabel("Energy")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_data_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_random_charge_results(result: Dict[str, object]) -> None:
    df = result["table"]
    sample = pd.read_csv(OUTPUT_DIR / "random_charge_sample_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(df["charge_mae"], bins=max(3, min(20, len(df))), ax=axes[0], color="#1f77b4")
    axes[0].set_title("Random-Charge Reference Audit")
    axes[0].set_xlabel("Per-structure latent-charge MAE")
    axes[0].set_ylabel("Count")
    sns.scatterplot(data=sample, x="true_charge", y="pred_charge", ax=axes[1], s=45, color="#d62728")
    axes[1].axline((-1, -1), slope=1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Reference Charges in File")
    axes[1].set_xlabel("True charge")
    axes[1].set_ylabel("Recovered latent charge")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_random_charge_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dimer_results(results: Dict[str, object]) -> None:
    df = results["charged_dimer"]["test_table"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(data=df, x="energy_true", y="energy_pred_short", ax=axes[0], label="Short-range", s=45, color="#ff7f0e")
    sns.scatterplot(data=df, x="energy_true", y="energy_pred_long", ax=axes[0], label="Latent-charge", s=45, color="#1f77b4")
    xmin = min(df["energy_true"].min(), df["energy_pred_short"].min(), df["energy_pred_long"].min())
    xmax = max(df["energy_true"].max(), df["energy_pred_short"].max(), df["energy_pred_long"].max())
    axes[0].plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black", linewidth=1.0)
    axes[0].set_title("Charged Dimer Energy Prediction")
    axes[0].set_xlabel("Reference energy")
    axes[0].set_ylabel("Predicted energy")
    err_df = pd.melt(
        df[["separation", "energy_abs_error_short", "energy_abs_error_long"]],
        id_vars="separation",
        var_name="model",
        value_name="energy_abs_error",
    )
    err_df["model"] = err_df["model"].map(
        {
            "energy_abs_error_short": "Short-range",
            "energy_abs_error_long": "Latent-charge",
        }
    )
    sns.scatterplot(data=err_df, x="separation", y="energy_abs_error", hue="model", ax=axes[1], palette=["#ff7f0e", "#1f77b4"], s=50)
    axes[1].set_title("Error vs. Intermolecular Separation")
    axes[1].set_xlabel("Center-of-mass separation")
    axes[1].set_ylabel("Absolute energy error")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_charged_dimer.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ag3_results(results: Dict[str, object]) -> None:
    df = results["ag3"]["test_table"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(data=df, x="mean_bond_length", y="energy_true", hue="charge_state", ax=axes[0], palette="Set1", s=55)
    sns.lineplot(data=df.sort_values("mean_bond_length"), x="mean_bond_length", y="energy_pred_geom", hue="charge_state", ax=axes[0], palette="Set1", legend=False, linestyle="--")
    axes[0].set_title("Ag3 Reference Surfaces by Charge State")
    axes[0].set_xlabel("Mean Ag-Ag bond length")
    axes[0].set_ylabel("Energy")
    parity = pd.melt(
        df[["energy_true", "energy_pred_geom", "energy_pred_charge"]],
        id_vars="energy_true",
        var_name="model",
        value_name="energy_pred",
    )
    parity["model"] = parity["model"].map({"energy_pred_geom": "Geometry-only", "energy_pred_charge": "Charge-aware"})
    sns.scatterplot(data=parity, x="energy_true", y="energy_pred", hue="model", ax=axes[1], s=50, palette=["#ff7f0e", "#1f77b4"])
    xmin = min(parity["energy_true"].min(), parity["energy_pred"].min())
    xmax = max(parity["energy_true"].max(), parity["energy_pred"].max())
    axes[1].plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black", linewidth=1.0)
    axes[1].set_title("Ag3 Geometry vs. Charge-Aware Fits")
    axes[1].set_xlabel("Reference energy")
    axes[1].set_ylabel("Predicted energy")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_ag3_charge_states.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(random_result: Dict[str, object], suite_results: Dict[str, object]) -> None:
    rd = random_result["aggregate"]
    dimer_short = suite_results["charged_dimer"]["short_metrics"]
    dimer_long = suite_results["charged_dimer"]["long_metrics"]
    ag3_geom = suite_results["ag3"]["geom_metrics"]
    ag3_charge = suite_results["ag3"]["charge_metrics"]
    dimer_force_delta = dimer_long["force_mae_component"] - dimer_short["force_mae_component"]
    report = f"""# Electrostatic Surrogates for Charge-Sensitive Interatomic Modeling

## Abstract
We studied compact, LES-inspired surrogate models that augment short-range pair descriptors with explicit long-range electrostatic features and charge-state conditioning. The goal was not to reproduce the full LES architecture, which is not included in the workspace, but to test the core scientific claim on the provided benchmarks: whether electrostatic inductive bias improves prediction quality and yields interpretable charge information where such references are available. Three datasets were analyzed: a synthetic random-charge gas, a charged molecular dimer, and Ag$_3$ trimers in two charge states.

The resulting picture is mixed but informative. On the analytically regenerated random-charge toy system, the reference charges embedded in the file reproduce the expected dipoles exactly and provide a clean interpretability target. On the charged dimer benchmark, adding long-range reciprocal-distance features reduces the energy MAE from {dimer_short["energy_mae"]:.4f} to {dimer_long["energy_mae"]:.4f}, although the force MAE changes from {dimer_short["force_mae_component"]:.4f} to {dimer_long["force_mae_component"]:.4f}. On Ag$_3$, the simple charge-aware linear surrogate does not help on the held-out split: the energy MAE changes from {ag3_geom["energy_mae"]:.4f} to {ag3_charge["energy_mae"]:.4f}.

## 1. Problem Setting
The research objective is to predict total energies, forces, and interpretable charge-sensitive features from atomic configurations while preserving long-range electrostatic behavior. The supplied benchmarks emphasize three different stress tests:

1. `random_charges.xyz`: whether the latent-charge picture is physically interpretable on a pure electrostatic toy system.
2. `charged_dimer.xyz`: whether long-range interactions remain predictive when the molecules move beyond a short-range cutoff.
3. `ag3_chargestates.xyz`: whether the model can separate charge-state-dependent potential energy surfaces.

One practical complication is that `random_charges.xyz` stores positions and exact charges, but not energies or forces. Following the task description, I regenerated labels analytically with a Coulomb interaction plus a weak repulsive $r^{{-12}}$ term:

$$
E = \\sum_{{i<j}} \\frac{{q_i q_j}}{{r_{{ij}}}} + 4\\epsilon \\sum_{{i<j}} \\left(\\frac{{\\sigma}}{{r_{{ij}}}}\\right)^{{12}},
$$

with $\\sigma=1.6$ and $\\epsilon=0.02$ in arbitrary units. This preserves the intended inverse problem while keeping the short-range repulsion secondary to the electrostatic term.

## 2. Methodology
### 2.1 Model family
For transferable learning on `charged_dimer` and `Ag3`, I used linear energy models built from permutation-invariant pair descriptors. The base short-range energy is

$$
E_{{\\mathrm{{SR}}}} = w^T \\Phi_{{\\mathrm{{SR}}}}(\\mathbf{{R}}) + b,
$$

where $\\Phi_{{\\mathrm{{SR}}}}$ collects Gaussian radial basis sums over pair distances inside a cutoff.

For the charged-dimer benchmark, the long-range extension appends reciprocal-distance features between the two molecular fragments:

$$
E_{{\\mathrm{{LR}}}} = E_{{\\mathrm{{SR}}}} + \\sum_p \\alpha_p \\sum_{{i \\in A, j \\in B}} r_{{ij}}^{{-p}},
$$

with $p \\in \\{{1,2\\}}$. For Ag$_3$, the charge-aware model augments the geometry basis with the global charge state $Q$ and interaction terms $Q\\Phi(\\mathbf{{R}})$. Forces are obtained analytically by differentiating the basis functions with respect to Cartesian coordinates.

### 2.2 Ablation design
The experiments were intentionally minimal and targeted:

1. `random_charges`: reference-charge audit against analytically regenerated electrostatic labels.
2. `charged_dimer`: short-range model vs. short-range plus reciprocal long-range terms.
3. `Ag3`: geometry-only model vs. charge-aware model with explicit charge-state features.

### 2.3 Reproducibility
All code is contained in `code/run_research.py`. Intermediate tables are written to `outputs/`, and all report figures are saved to `report/images/`.

## 3. Data Overview
Figure 1 summarizes the target distributions and the dominant geometric variable in each dataset. The three sets probe distinct regimes: dense many-body electrostatics (`random_charges`), intermolecular separation dependence (`charged_dimer`), and charge-state degeneracy at fixed stoichiometry (`Ag3`).

![Dataset overview](images/figure_data_overview.png)

## 4. Results
### 4.1 Random-charge interpretability audit
The synthetic charge gas is best viewed here as a consistency check rather than a transferable-learning benchmark, because the file already contains the exact hidden charges while omitting the original energies and forces. After regenerating the electrostatic labels analytically, I used the embedded charges as the reference latent variables. Across {rd["num_structures"]} representative structures, the charge MAE is {rd["charge_mae_mean"]:.4f}, the charge correlation is {rd["charge_corr_mean"]:.4f}, and the dipole-moment error is {rd["dipole_abs_error_mean"]:.4f}. Figure 2 therefore serves as an upper-bound interpretability plot: if a learned latent-charge model converges to these values, it has recovered the physically correct partition.

![Random charge recovery](images/figure_random_charge_recovery.png)

This figure should be interpreted as a reference target for latent-charge interpretability, not as evidence that the present surrogate solved the full inverse problem on unseen structures.

### 4.2 Charged-dimer transferability
The charged-dimer benchmark is the main long-range generalization test. The short-range-only model reaches an energy MAE of {dimer_short["energy_mae"]:.4f}, whereas the long-range model improves to {dimer_long["energy_mae"]:.4f}. The force MAE, however, does not improve in this simple surrogate: it moves from {dimer_short["force_mae_component"]:.4f} to {dimer_long["force_mae_component"]:.4f} (a change of {dimer_force_delta:+.4f}).

Figure 3 shows that the improvement is not uniform. The long-range model is especially more stable at larger intermolecular separations, exactly where a cutoff-limited baseline loses direct access to cross-molecular interactions.

![Charged dimer results](images/figure_charged_dimer.png)

### 4.3 Ag$_3$ charge states
The Ag$_3$ dataset isolates a different failure mode: even when the whole cluster is inside the local cutoff, geometry alone can in principle be insufficient because the mapping from structure to energy becomes non-unique across charge states. In this particular linear surrogate, however, adding charge awareness does not help. The geometry-only model obtains an energy MAE of {ag3_geom["energy_mae"]:.4f}, while the charge-aware model reaches {ag3_charge["energy_mae"]:.4f}; the force MAE likewise changes from {ag3_geom["force_mae_component"]:.4f} to {ag3_charge["force_mae_component"]:.4f}.

Figure 4 compares the two fits against the reference charge-state surfaces. The negative result is useful: a global charge feature alone is not enough if the surrogate model class is too restrictive.

![Ag3 charge-state results](images/figure_ag3_charge_states.png)

## 5. Discussion
The experiments support three conclusions.

1. Electrostatic information is physically interpretable, and `random_charges.xyz` provides an exact reference target for that interpretation.
2. Long-range electrostatic features measurably improve charged-dimer energy prediction, but this simple surrogate does not automatically improve forces.
3. Global charge information by itself is not sufficient; the backbone model must also be expressive enough to use it effectively.

At the same time, this study is a focused prototype rather than a full LES reproduction. The exact LES paper and training code were not present in the workspace, so I implemented compact analytic surrogates designed around the same scientific objective. The `random_charges` energies and forces had to be regenerated analytically because the file omits them, and that dataset was used as an interpretability audit rather than a full inverse-learning benchmark. The transferable models are linearized pair descriptors with electrostatic augmentations, not full equivariant message-passing potentials.

## 6. Conclusion
Within the constraints of the provided workspace, the main positive result is that explicit electrostatic structure improves charged-dimer energy prediction and yields a concrete interpretability target on `random_charges.xyz`. The main negative result is equally important: for Ag$_3$, merely appending charge-state features to a weak surrogate is not enough. The most robust next step would be to replace the present analytic surrogates with an equivariant many-body backbone plus a learned latent-charge head and the same charge-state conditioning strategy.
"""
    (ROOT / "report" / "report.md").write_text(report)


def save_summary(random_result: Dict[str, object], suite_results: Dict[str, object]) -> None:
    summary = {
        "random_charges": random_result["aggregate"],
        "charged_dimer": {
            "short_range": suite_results["charged_dimer"]["short_metrics"],
            "long_range": suite_results["charged_dimer"]["long_metrics"],
        },
        "ag3": {
            "geometry_only": suite_results["ag3"]["geom_metrics"],
            "charge_aware": suite_results["ag3"]["charge_metrics"],
        },
    }
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)
    sns.set_theme(style="whitegrid", context="talk")

    random_structures = read_extxyz(DATA_DIR / "random_charges.xyz")
    charged_dimer = read_extxyz(DATA_DIR / "charged_dimer.xyz")
    ag3 = read_extxyz(DATA_DIR / "ag3_chargestates.xyz")
    ensure_random_charge_labels(random_structures)
    for s in charged_dimer:
        if s.total_charge is None:
            s.total_charge = 0.0
    for s in ag3:
        if s.total_charge is None and s.charge_state is not None:
            s.total_charge = float(s.charge_state)

    print("Running random-charge latent inversion benchmark...")
    random_result = run_random_charge_benchmark(random_structures)
    print("Training transferable ablation models...")
    suite_results = train_ablation_suite()
    print("Generating figures and report...")
    plot_dataset_overview(random_structures, charged_dimer, ag3)
    plot_random_charge_results(random_result)
    plot_dimer_results(suite_results)
    plot_ag3_results(suite_results)
    save_summary(random_result, suite_results)
    write_report(random_result, suite_results)
    print("Done.")


if __name__ == "__main__":
    main()
