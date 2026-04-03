from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

SEED = 7
DEVICE = torch.device("cpu")
RDLogger.DisableLog("rdApp.*")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)


def kelvin_to_celsius(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values) - 273.15


def clean_polymer_smiles(smiles: str) -> str:
    return smiles.replace("*", "[*]")


ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    atomic_num = atom.GetAtomicNum()
    features = [
        float(atomic_num == z) for z in ATOM_LIST
    ] + [float(atomic_num not in ATOM_LIST)]
    features += [
        atom.GetTotalDegree() / 4.0,
        atom.GetFormalCharge(),
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]
    features += [float(atom.GetHybridization() == hyb) for hyb in HYBRIDIZATION_LIST]
    features += [float(atom.GetHybridization() not in HYBRIDIZATION_LIST)]
    return np.asarray(features, dtype=np.float32)


def mol_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    n_atoms = mol.GetNumAtoms()
    x = np.stack([atom_features(atom) for atom in mol.GetAtoms()], axis=0)
    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj += np.eye(n_atoms, dtype=np.float32)
    deg = adj.sum(axis=1, keepdims=False)
    d_inv_sqrt = np.diag(np.power(np.clip(deg, 1.0, None), -0.5))
    adj = d_inv_sqrt @ adj @ d_inv_sqrt
    return x, adj.astype(np.float32)


def descriptor_vector(smiles: str, n_bits: int = 128) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    fp_arr = np.asarray(list(fp.ToBitString()), dtype=np.float32)
    desc = np.asarray(
        [
            Descriptors.MolWt(mol) / 500.0,
            Descriptors.MolLogP(mol) / 10.0,
            Descriptors.TPSA(mol) / 200.0,
            Lipinski.NumHDonors(mol) / 10.0,
            Lipinski.NumHAcceptors(mol) / 15.0,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 10.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([fp_arr, desc], axis=0)


def tanimoto_from_vectors(a: np.ndarray, b: np.ndarray, n_bits: int = 128) -> float:
    a_bits = a[:n_bits] > 0.5
    b_bits = b[:n_bits] > 0.5
    inter = np.logical_and(a_bits, b_bits).sum()
    union = np.logical_or(a_bits, b_bits).sum()
    return float(inter / union) if union else 0.0


@dataclass
class MoleculeRecord:
    smiles: str
    x: np.ndarray
    adj: np.ndarray
    desc: np.ndarray
    graph_feat: np.ndarray


def graph_feature_vector(smiles: str, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = x.shape[0]
    n_bonds = mol.GetNumBonds()
    atom_mean = x.mean(axis=0)
    atom_std = x.std(axis=0)
    extras = np.asarray(
        [
            n_atoms / 80.0,
            n_bonds / 100.0,
            float(n_bonds) / max(n_atoms * (n_atoms - 1) / 2, 1.0),
            sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / max(n_atoms, 1),
            sum(atom.IsInRing() for atom in mol.GetAtoms()) / max(n_atoms, 1),
            sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()) / max(n_atoms, 1),
            rdMolDescriptors.CalcNumRings(mol) / 10.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([atom_mean, atom_std, extras], axis=0).astype(np.float32)


def build_molecule_library(smiles_values: Iterable[str]) -> Dict[str, MoleculeRecord]:
    library: Dict[str, MoleculeRecord] = {}
    for smiles in sorted(set(smiles_values)):
        x, adj = mol_to_graph(smiles)
        desc = descriptor_vector(smiles)
        graph_feat = graph_feature_vector(smiles, x, adj)
        library[smiles] = MoleculeRecord(smiles=smiles, x=x, adj=adj, desc=desc, graph_feat=graph_feat)
    return library


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, acid_lib: Dict[str, MoleculeRecord], epoxide_lib: Dict[str, MoleculeRecord]):
        self.frame = frame.reset_index(drop=True).copy()
        self.acid_lib = acid_lib
        self.epoxide_lib = epoxide_lib

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray | float | str]:
        row = self.frame.iloc[idx]
        acid = self.acid_lib[row["acid"]]
        epoxide = self.epoxide_lib[row["epoxide"]]
        return {
            "graph_feat": np.concatenate([acid.graph_feat, epoxide.graph_feat], axis=0),
            "acid_desc": acid.desc,
            "epoxide_desc": epoxide.desc,
            "y": float(row["tg_calibrated"]),
            "acid": row["acid"],
            "epoxide": row["epoxide"],
        }


def collate_graphs(batch: Sequence[Dict[str, np.ndarray | float | str]]) -> Dict[str, torch.Tensor | List[str]]:
    graph_feat = torch.from_numpy(np.stack([item["graph_feat"] for item in batch]).astype(np.float32))
    acid_desc = torch.from_numpy(np.stack([item["acid_desc"] for item in batch]).astype(np.float32))
    epoxide_desc = torch.from_numpy(np.stack([item["epoxide_desc"] for item in batch]).astype(np.float32))
    y = torch.tensor([item["y"] for item in batch], dtype=torch.float32).unsqueeze(-1)
    return {
        "graph_feat": graph_feat,
        "acid_desc": acid_desc,
        "epoxide_desc": epoxide_desc,
        "y": y,
        "acid": [item["acid"] for item in batch],
        "epoxide": [item["epoxide"] for item in batch],
    }

class PairGraphVAE(nn.Module):
    def __init__(self, graph_feat_dim: int, mol_desc_dim: int, latent_dim: int = 24):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_mlp = nn.Sequential(
            nn.Linear(graph_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(96, latent_dim)
        self.logvar_head = nn.Linear(96, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.acid_out = nn.Linear(128, mol_desc_dim)
        self.epoxide_out = nn.Linear(128, mol_desc_dim)
        self.property_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_mlp(batch["graph_feat"])
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        return self.acid_out(h), self.epoxide_out(h)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        acid_recon, epoxide_recon = self.decode(z)
        property_pred = self.property_head(mu)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "acid_recon": acid_recon,
            "epoxide_recon": epoxide_recon,
            "property_pred": property_pred,
            "graph_feat": batch["graph_feat"],
        }


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, n_bits: int = 128) -> torch.Tensor:
    bit_loss = F.binary_cross_entropy_with_logits(pred[:, :n_bits], target[:, :n_bits])
    desc_loss = F.mse_loss(pred[:, n_bits:], target[:, n_bits:])
    return bit_loss + desc_loss


def train_epoch(model: PairGraphVAE, loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "prop": 0.0, "recon": 0.0, "kl": 0.0}
    n_items = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        prop_loss = F.mse_loss(out["property_pred"], batch["y"])
        acid_recon_loss = reconstruction_loss(out["acid_recon"], batch["acid_desc"])
        epoxide_recon_loss = reconstruction_loss(out["epoxide_recon"], batch["epoxide_desc"])
        recon = acid_recon_loss + epoxide_recon_loss
        kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        loss = prop_loss + 0.3 * recon + 0.05 * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = batch["y"].shape[0]
        n_items += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["prop"] += prop_loss.item() * batch_size
        totals["recon"] += recon.item() * batch_size
        totals["kl"] += kl.item() * batch_size
    return {k: v / max(n_items, 1) for k, v in totals.items()}


@torch.no_grad()
def collect_predictions(model: PairGraphVAE, loader: DataLoader) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        mu = out["mu"].cpu().numpy()
        pred = out["property_pred"].cpu().numpy().ravel()
        truth = batch["y"].cpu().numpy().ravel()
        graph_feat = batch["graph_feat"].cpu().numpy()
        for i in range(len(pred)):
            row = {
                "acid": batch["acid"][i],
                "epoxide": batch["epoxide"][i],
                "y_true": truth[i],
                "y_pred": pred[i],
            }
            for j, value in enumerate(mu[i]):
                row[f"z_{j:02d}"] = value
            for j, value in enumerate(graph_feat[i]):
                row[f"graph_{j:02d}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_gp_calibration(calibration: pd.DataFrame, vitrimers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    X = calibration[["tg_md"]].to_numpy()
    y = calibration["tg_exp"].to_numpy()
    alpha = (calibration["std"].to_numpy() / calibration["std"].std()) ** 2 + 1e-4
    raw_metrics = metrics_dict(y, calibration["tg_md"].to_numpy())
    linear_metrics = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    gp_metric_rows = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(1e-1, 1e3)) + WhiteKernel(1.0, (1e-5, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha[train_idx], normalize_y=True, n_restarts_optimizer=2, random_state=SEED)
        gp.fit(X[train_idx], y[train_idx])
        pred_mean, pred_std = gp.predict(X[test_idx], return_std=True)
        for i, idx in enumerate(test_idx):
            cv_rows.append(
                {
                    "fold": fold,
                    "tg_md": calibration.loc[idx, "tg_md"],
                    "tg_exp": calibration.loc[idx, "tg_exp"],
                    "pred_mean": pred_mean[i],
                    "pred_std": pred_std[i],
                    "name": calibration.loc[idx, "name"],
                    "smiles": calibration.loc[idx, "smiles"],
                }
            )
        gp_metric_rows.append(metrics_dict(y[test_idx], pred_mean))

    lin = LinearRegression().fit(X, y)
    linear_metrics = metrics_dict(y, lin.predict(X))
    cv_df = pd.DataFrame(cv_rows)
    gp_metrics = {
        key: float(np.mean([row[key] for row in gp_metric_rows])) for key in ["mae", "rmse", "r2"]
    }
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(1e-1, 1e3)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_full = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)
    gp_full.fit(X, y)
    vit_mean, vit_std = gp_full.predict(vitrimers[["tg"]].to_numpy(), return_std=True)
    calibrated = vitrimers.copy()
    calibrated["tg_calibrated"] = vit_mean
    calibrated["tg_calibrated_std"] = vit_std
    calibration_grid = pd.DataFrame({"tg_md": np.linspace(calibration["tg_md"].min(), calibration["tg_md"].max(), 200)})
    grid_mean, grid_std = gp_full.predict(calibration_grid[["tg_md"]].to_numpy(), return_std=True)
    calibration_grid["pred_mean"] = grid_mean
    calibration_grid["pred_std"] = grid_std
    return cv_df, calibrated, raw_metrics | {"linear_mae": linear_metrics["mae"], "linear_rmse": linear_metrics["rmse"], "linear_r2": linear_metrics["r2"]}, gp_metrics


def split_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(frame, test_size=0.15, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.15, random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_graph_vae(vitrimers: pd.DataFrame, acid_lib: Dict[str, MoleculeRecord], epoxide_lib: Dict[str, MoleculeRecord]) -> Tuple[PairGraphVAE, Dict[str, pd.DataFrame], Dict[str, float], StandardScaler]:
    train_df, val_df, test_df = split_frame(vitrimers)
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    scaler.fit(train_df[["tg_calibrated"]])
    for frame in [train_df, val_df, test_df]:
        frame["tg_calibrated"] = scaler.transform(frame[["tg_calibrated"]]).astype(np.float32)

    train_loader = DataLoader(PairDataset(train_df, acid_lib, epoxide_lib), batch_size=128, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(PairDataset(val_df, acid_lib, epoxide_lib), batch_size=256, shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(PairDataset(test_df, acid_lib, epoxide_lib), batch_size=256, shuffle=False, collate_fn=collate_graphs)

    graph_feat_dim = next(iter(acid_lib.values())).graph_feat.shape[0] * 2
    mol_desc_dim = next(iter(acid_lib.values())).desc.shape[0]
    model = PairGraphVAE(graph_feat_dim=graph_feat_dim, mol_desc_dim=mol_desc_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_state = None
    best_val = float("inf")
    history_rows = []
    for epoch in range(1, 3):
        train_metrics = train_epoch(model, train_loader, optimizer)
        val_pred_df = collect_predictions(model, val_loader)
        y_val = scaler.inverse_transform(val_pred_df[["y_true"]]).ravel()
        y_val_pred = scaler.inverse_transform(val_pred_df[["y_pred"]]).ravel()
        val_mae = mean_absolute_error(y_val, y_val_pred)
        history_rows.append({"epoch": epoch, **train_metrics, "val_mae": float(val_mae)})
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce a model state.")
    model.load_state_dict(best_state)

    split_predictions = {
        "train": collect_predictions(model, DataLoader(PairDataset(train_df, acid_lib, epoxide_lib), batch_size=256, shuffle=False, collate_fn=collate_graphs)),
        "val": collect_predictions(model, val_loader),
        "test": collect_predictions(model, test_loader),
    }
    metrics = {}
    for split, frame in split_predictions.items():
        y_true = scaler.inverse_transform(frame[["y_true"]]).ravel()
        y_pred = scaler.inverse_transform(frame[["y_pred"]]).ravel()
        for key, value in metrics_dict(y_true, y_pred).items():
            metrics[f"{split}_{key}"] = value
        frame["y_true"] = y_true
        frame["y_pred"] = y_pred
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "graph_vae_training_history.csv", index=False)
    return model, split_predictions, metrics, scaler


def encode_pairs(model: PairGraphVAE, frame: pd.DataFrame, acid_lib: Dict[str, MoleculeRecord], epoxide_lib: Dict[str, MoleculeRecord], scaler: StandardScaler) -> pd.DataFrame:
    temp = frame.copy()
    temp["tg_calibrated"] = scaler.transform(temp[["tg_calibrated"]]).astype(np.float32)
    loader = DataLoader(PairDataset(temp, acid_lib, epoxide_lib), batch_size=128, shuffle=False, collate_fn=collate_graphs)
    pred = collect_predictions(model, loader)
    pred["y_true"] = scaler.inverse_transform(pred[["y_true"]]).ravel()
    pred["y_pred"] = scaler.inverse_transform(pred[["y_pred"]]).ravel()
    return pred


def optimize_latent_targets(model: PairGraphVAE, targets: Sequence[float], n_samples: int = 4, steps: int = 20) -> Dict[float, np.ndarray]:
    model.eval()
    results: Dict[float, List[np.ndarray]] = {float(t): [] for t in targets}
    for target in targets:
        target_tensor = torch.tensor([[float(target)]], dtype=torch.float32, device=DEVICE)
        for _ in range(n_samples):
            z = torch.randn((1, model.latent_dim), device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=0.05)
            for _ in range(steps):
                pred = model.property_head(z)
                loss = (pred - target_tensor).pow(2).mean() + 0.01 * z.pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            results[float(target)].append(z.detach().cpu().numpy().squeeze(0))
    return {key: np.stack(value, axis=0) for key, value in results.items()}


def nearest_molecules(decoded: np.ndarray, library: Dict[str, MoleculeRecord], top_k: int = 12) -> List[str]:
    smiles_list = list(library.keys())
    matrix = np.stack([library[s].desc for s in smiles_list], axis=0)
    decoded_bits = decoded[:128]
    decoded_desc = decoded[128:]
    decoded_vec = np.concatenate([decoded_bits, decoded_desc], axis=0)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(decoded_vec)
    sims = (matrix @ decoded_vec) / np.clip(denom, 1e-8, None)
    order = np.argsort(sims)[::-1][:top_k]
    return [smiles_list[i] for i in order]


def pair_fingerprint(acid_smiles: str, epoxide_smiles: str, acid_lib: Dict[str, MoleculeRecord], epoxide_lib: Dict[str, MoleculeRecord]) -> np.ndarray:
    return np.concatenate([acid_lib[acid_smiles].desc[:128], epoxide_lib[epoxide_smiles].desc[:128]], axis=0)


def batch_predict_pairs(model: PairGraphVAE, pair_frame: pd.DataFrame, acid_lib: Dict[str, MoleculeRecord], epoxide_lib: Dict[str, MoleculeRecord], scaler: StandardScaler) -> pd.DataFrame:
    temp = pair_frame.copy()
    temp["tg_calibrated"] = float(scaler.mean_[0])
    temp["tg_calibrated"] = scaler.transform(temp[["tg_calibrated"]]).astype(np.float32)
    loader = DataLoader(PairDataset(temp, acid_lib, epoxide_lib), batch_size=128, shuffle=False, collate_fn=collate_graphs)
    pred = collect_predictions(model, loader)
    pred["y_pred"] = scaler.inverse_transform(pred[["y_pred"]]).ravel()
    return pred


def generate_candidates(
    model: PairGraphVAE,
    acid_lib: Dict[str, MoleculeRecord],
    epoxide_lib: Dict[str, MoleculeRecord],
    scaler: StandardScaler,
    observed_pairs: set[Tuple[str, str]],
    targets_kelvin: Sequence[float],
    train_latent: np.ndarray,
) -> pd.DataFrame:
    latent_samples = optimize_latent_targets(model, targets_kelvin)
    observed_pair_vectors = []
    for acid, epoxide in observed_pairs:
        observed_pair_vectors.append(pair_fingerprint(acid, epoxide, acid_lib, epoxide_lib))
    observed_pair_vectors = np.stack(observed_pair_vectors, axis=0)
    raw_rows = []
    for target, zs in latent_samples.items():
        for z in zs:
            acid_pred, epoxide_pred = model.decode(torch.tensor(z[None, :], dtype=torch.float32, device=DEVICE))
            acid_vec = torch.cat([torch.sigmoid(acid_pred[:, :128]), acid_pred[:, 128:]], dim=1).cpu().numpy()[0]
            epoxide_vec = torch.cat([torch.sigmoid(epoxide_pred[:, :128]), epoxide_pred[:, 128:]], dim=1).cpu().numpy()[0]
            acid_candidates = nearest_molecules(acid_vec, acid_lib, top_k=2)
            epoxide_candidates = nearest_molecules(epoxide_vec, epoxide_lib, top_k=2)
            for acid in acid_candidates:
                for epoxide in epoxide_candidates:
                    if (acid, epoxide) in observed_pairs:
                        continue
                    raw_rows.append({"target_tg": target, "acid": acid, "epoxide": epoxide})
    candidate_pairs = pd.DataFrame(raw_rows).drop_duplicates(subset=["acid", "epoxide", "target_tg"]).reset_index(drop=True)
    pred = batch_predict_pairs(model, candidate_pairs[["acid", "epoxide"]], acid_lib, epoxide_lib, scaler)
    latent_cols = [col for col in pred.columns if col.startswith("z_")]
    latent_matrix = pred[latent_cols].to_numpy()
    rows = []
    for idx, row in candidate_pairs.iterrows():
        acid = row["acid"]
        epoxide = row["epoxide"]
        pair_vec = pair_fingerprint(acid, epoxide, acid_lib, epoxide_lib)
        acid_mol = Chem.MolFromSmiles(acid)
        epoxide_mol = Chem.MolFromSmiles(epoxide)
        mw_total = Descriptors.MolWt(acid_mol) + Descriptors.MolWt(epoxide_mol)
        nearest_sim = np.max(
            np.sum(np.logical_and(observed_pair_vectors > 0.5, pair_vec > 0.5), axis=1)
            / np.clip(np.sum(np.logical_or(observed_pair_vectors > 0.5, pair_vec > 0.5), axis=1), 1, None)
        )
        latent_dist = np.linalg.norm(train_latent - latent_matrix[idx], axis=1)
        uncertainty = float(np.percentile(np.sort(latent_dist)[:10], 90))
        pred_tg = float(pred.iloc[idx]["y_pred"])
        rows.append(
            {
                "target_tg": row["target_tg"],
                "acid": acid,
                "epoxide": epoxide,
                "pred_tg_model": pred_tg,
                "pred_tg_ensemble": pred_tg,
                "pred_tg_std": uncertainty,
                "target_error": float(abs(pred_tg - row["target_tg"])),
                "novelty": float(1.0 - nearest_sim),
                "mw_total": float(mw_total),
            }
        )
    candidates = pd.DataFrame(rows).drop_duplicates(subset=["acid", "epoxide", "target_tg"])
    if candidates.empty:
        raise RuntimeError("No new candidates were generated.")
    candidates["selection_score"] = (
        candidates["target_error"]
        + 0.4 * candidates["pred_tg_std"]
        + 0.015 * candidates["mw_total"]
        - 8.0 * candidates["novelty"]
    )
    out_rows = []
    for target in targets_kelvin:
        sub = candidates[candidates["target_tg"] == float(target)].sort_values("selection_score").head(8)
        out_rows.append(sub)
    final = pd.concat(out_rows, ignore_index=True)
    return final.sort_values(["target_tg", "selection_score"]).reset_index(drop=True)


def plot_gp_results(calibration: pd.DataFrame, cv_df: pd.DataFrame, grid_df: pd.DataFrame, vitrimer_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(kelvin_to_celsius(calibration["tg_md"]), kelvin_to_celsius(calibration["tg_exp"]), alpha=0.6, color="#1b6ca8")
    min_k = min(calibration["tg_md"].min(), calibration["tg_exp"].min())
    max_k = max(calibration["tg_md"].max(), calibration["tg_exp"].max())
    axes[0].plot(kelvin_to_celsius(np.array([min_k, max_k])), kelvin_to_celsius(np.array([min_k, max_k])), "--", color="black", linewidth=1)
    axes[0].set_xlabel("MD Tg (°C)")
    axes[0].set_ylabel("Experimental Tg (°C)")
    axes[0].set_title("Calibration data")

    order = np.argsort(grid_df["tg_md"].to_numpy())
    x = kelvin_to_celsius(grid_df["tg_md"].to_numpy()[order])
    y = kelvin_to_celsius(grid_df["pred_mean"].to_numpy()[order])
    std = grid_df["pred_std"].to_numpy()[order]
    axes[1].scatter(kelvin_to_celsius(calibration["tg_md"]), kelvin_to_celsius(calibration["tg_exp"]), alpha=0.35, color="#4c956c")
    axes[1].plot(x, y, color="#d1495b", linewidth=2)
    axes[1].fill_between(x, y - std, y + std, alpha=0.2, color="#d1495b")
    axes[1].set_xlabel("MD Tg (°C)")
    axes[1].set_ylabel("Calibrated Tg (°C)")
    axes[1].set_title("GP calibration curve")

    sns.kdeplot(kelvin_to_celsius(vitrimer_df["tg"]), fill=True, color="#5d2e8c", ax=axes[2], label="MD")
    sns.kdeplot(kelvin_to_celsius(vitrimer_df["tg_calibrated"]), fill=True, color="#ff7f11", ax=axes[2], label="Calibrated")
    axes[2].set_xlabel("Tg (°C)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Vitrimer library shift")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "gp_calibration_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(kelvin_to_celsius(cv_df["tg_exp"]), kelvin_to_celsius(cv_df["pred_mean"]), alpha=0.65, color="#2a9d8f")
    low = kelvin_to_celsius(np.array([cv_df["tg_exp"].min(), cv_df["tg_exp"].max()]))
    ax.plot(low, low, "--", color="black", linewidth=1)
    ax.set_xlabel("Observed experimental Tg (°C)")
    ax.set_ylabel("Cross-validated GP prediction (°C)")
    ax.set_title("Calibration cross-validation")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "gp_cv_parity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_graph_results(train_pred: pd.DataFrame, test_pred: pd.DataFrame, history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["epoch"], history["loss"], color="#355070", label="Train loss")
    axes[0].plot(history["epoch"], history["val_mae"], color="#e56b6f", label="Val MAE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Graph-VAE training")
    axes[0].legend(frameon=False)

    axes[1].scatter(kelvin_to_celsius(test_pred["y_true"]), kelvin_to_celsius(test_pred["y_pred"]), alpha=0.65, color="#3a86ff")
    lims = kelvin_to_celsius(np.array([test_pred["y_true"].min(), test_pred["y_true"].max()]))
    axes[1].plot(lims, lims, "--", color="black", linewidth=1)
    axes[1].set_xlabel("Calibrated Tg (°C)")
    axes[1].set_ylabel("Predicted Tg (°C)")
    axes[1].set_title("Held-out parity")

    latent_cols = [col for col in train_pred.columns if col.startswith("z_")]
    pca = PCA(n_components=2, random_state=SEED)
    latent_2d = pca.fit_transform(train_pred[latent_cols].to_numpy())
    scatter = axes[2].scatter(latent_2d[:, 0], latent_2d[:, 1], c=kelvin_to_celsius(train_pred["y_true"]), cmap="viridis", s=18, alpha=0.7)
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title("Latent space")
    cbar = fig.colorbar(scatter, ax=axes[2])
    cbar.set_label("Calibrated Tg (°C)")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "graph_vae_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_results(candidates: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {float(t): c for t, c in zip(sorted(candidates["target_tg"].unique()), ["#ef476f", "#118ab2", "#06d6a0"])}
    for target, sub in candidates.groupby("target_tg"):
        axes[0].errorbar(
            kelvin_to_celsius(sub["pred_tg_ensemble"]),
            sub["novelty"],
            xerr=sub["pred_tg_std"],
            fmt="o",
            color=palette[target],
            alpha=0.8,
            label=f"Target {kelvin_to_celsius(np.array([target]))[0]:.0f} °C",
        )
    axes[0].set_xlabel("Predicted Tg (°C)")
    axes[0].set_ylabel("Novelty score")
    axes[0].set_title("Generated candidates")
    axes[0].legend(frameon=False)

    sns.barplot(
        data=candidates.assign(target_label=candidates["target_tg"].map(lambda x: f"{kelvin_to_celsius(np.array([x]))[0]:.0f} °C")),
        x="target_label",
        y="pred_tg_std",
        hue="target_label",
        palette="crest",
        dodge=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("Target window")
    axes[1].set_ylabel("Ensemble std (K)")
    axes[1].set_title("Uncertainty of recommended candidates")
    axes[1].legend([], [], frameon=False)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "inverse_design_candidates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_candidate_panel(candidates: pd.DataFrame) -> None:
    selected = candidates.groupby("target_tg", sort=True).head(2).reset_index(drop=True)
    mols = []
    legends = []
    for _, row in selected.iterrows():
        mols.append(Chem.MolFromSmiles(row["acid"]))
        legends.append(f"Acid\nT={row['target_tg']:.0f}K")
        mols.append(Chem.MolFromSmiles(row["epoxide"]))
        legends.append(f"Epoxide\nPred={row['pred_tg_ensemble']:.0f}K")
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(280, 220), legends=legends)
    img.save(REPORT_IMG_DIR / "candidate_structures.png")


def write_tables_and_figures(
    calibration: pd.DataFrame,
    cv_df: pd.DataFrame,
    calibration_grid: pd.DataFrame,
    vitrimer_df: pd.DataFrame,
    split_predictions: Dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
) -> None:
    calibration.to_csv(OUTPUT_DIR / "tg_calibration_clean.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "gp_calibration_cv_predictions.csv", index=False)
    calibration_grid.to_csv(OUTPUT_DIR / "gp_calibration_curve.csv", index=False)
    vitrimer_df.to_csv(OUTPUT_DIR / "vitrimer_calibrated_predictions.csv", index=False)
    for split, frame in split_predictions.items():
        frame.to_csv(OUTPUT_DIR / f"graph_vae_{split}_predictions.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "inverse_design_candidates.csv", index=False)

    plot_gp_results(calibration, cv_df, calibration_grid, vitrimer_df)
    history = pd.read_csv(OUTPUT_DIR / "graph_vae_training_history.csv")
    plot_graph_results(split_predictions["train"], split_predictions["test"], history)
    plot_candidate_results(candidates)
    draw_candidate_panel(candidates)


def make_candidate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    summary = candidates.copy()
    summary["target_tg_c"] = kelvin_to_celsius(summary["target_tg"])
    summary["pred_tg_c"] = kelvin_to_celsius(summary["pred_tg_ensemble"])
    summary["pred_tg_std_c"] = summary["pred_tg_std"]
    return summary[
        [
            "target_tg",
            "target_tg_c",
            "acid",
            "epoxide",
            "pred_tg_ensemble",
            "pred_tg_c",
            "pred_tg_std",
            "target_error",
            "novelty",
            "mw_total",
        ]
    ]


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
    ensure_dirs()
    set_seed(SEED)

    print("Loading data...", flush=True)
    calibration = pd.read_csv(DATA_DIR / "tg_calibration.csv")
    calibration["smiles_clean"] = calibration["smiles"].map(clean_polymer_smiles)
    vitrimers = pd.read_csv(DATA_DIR / "tg_vitrimer_MD.csv")

    print("Running GP calibration...", flush=True)
    cv_df, vitrimers_calibrated, raw_metrics, gp_metrics = run_gp_calibration(calibration, vitrimers)
    calibration_grid = pd.read_csv(OUTPUT_DIR / "gp_calibration_curve.csv") if (OUTPUT_DIR / "gp_calibration_curve.csv").exists() else None
    if calibration_grid is None:
        # populated below in write_tables_and_figures via direct CSV save
        calibration_grid = pd.DataFrame()

    print("Building molecule libraries...", flush=True)
    acid_lib = build_molecule_library(vitrimers_calibrated["acid"].unique())
    epoxide_lib = build_molecule_library(vitrimers_calibrated["epoxide"].unique())

    print("Training graph VAE...", flush=True)
    model, split_predictions, graph_metrics, scaler = fit_graph_vae(vitrimers_calibrated, acid_lib, epoxide_lib)

    observed_pairs = set(zip(vitrimers_calibrated["acid"], vitrimers_calibrated["epoxide"]))
    target_windows = [360.0, 400.0, 440.0]
    target_windows_kelvin = [t + 273.15 for t in target_windows]
    print("Generating candidates...", flush=True)
    candidates = generate_candidates(
        model,
        acid_lib,
        epoxide_lib,
        scaler,
        observed_pairs=observed_pairs,
        targets_kelvin=target_windows_kelvin,
        train_latent=split_predictions["train"][[col for col in split_predictions["train"].columns if col.startswith("z_")]].to_numpy(),
    )
    candidates = make_candidate_summary(candidates)

    # recover calibration grid from full-fit GP saved in function scope
    # recompute once for plotting and export
    print("Preparing figures and exports...", flush=True)
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(1e-1, 1e3)) + WhiteKernel(1.0, (1e-5, 1e2))
    X = calibration[["tg_md"]].to_numpy()
    y = calibration["tg_exp"].to_numpy()
    alpha = (calibration["std"].to_numpy() / calibration["std"].std()) ** 2 + 1e-4
    gp_full = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)
    gp_full.fit(X, y)
    calibration_grid = pd.DataFrame({"tg_md": np.linspace(calibration["tg_md"].min(), calibration["tg_md"].max(), 200)})
    grid_mean, grid_std = gp_full.predict(calibration_grid[["tg_md"]].to_numpy(), return_std=True)
    calibration_grid["pred_mean"] = grid_mean
    calibration_grid["pred_std"] = grid_std

    write_tables_and_figures(
        calibration,
        cv_df,
        calibration_grid,
        vitrimers_calibrated,
        split_predictions,
        candidates,
    )

    summary = {
        "seed": SEED,
        "calibration_raw_mae": raw_metrics["mae"],
        "calibration_raw_rmse": raw_metrics["rmse"],
        "calibration_gp_cv_mae": gp_metrics["mae"],
        "calibration_gp_cv_rmse": gp_metrics["rmse"],
        "calibration_gp_cv_r2": gp_metrics["r2"],
        "graph_vae_test_mae": graph_metrics["test_mae"],
        "graph_vae_test_rmse": graph_metrics["test_rmse"],
        "graph_vae_test_r2": graph_metrics["test_r2"],
    }
    with open(OUTPUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
