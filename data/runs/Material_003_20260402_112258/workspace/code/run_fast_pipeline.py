from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from run_pipeline import (
    DATA_DIR,
    OUTPUT_DIR,
    REPORT_IMG_DIR,
    SEED,
    build_molecule_library,
    clean_polymer_smiles,
    ensure_dirs,
    kelvin_to_celsius,
    pair_fingerprint,
    set_seed,
)


DEVICE = torch.device("cpu")


class MatrixVAE(nn.Module):
    def __init__(self, in_dim: int, recon_dim: int, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, recon_dim))
        self.prop = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decoder(z)
        pred = self.prop(mu)
        return mu, logvar, recon, pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def gp_calibrate(calibration: pd.DataFrame, vitrimers: pd.DataFrame):
    X = calibration[["tg_md"]].to_numpy()
    y = calibration["tg_exp"].to_numpy()
    alpha = (calibration["std"].to_numpy() / calibration["std"].std()) ** 2 + 1e-4
    cv_rows = []
    fold_metrics = []
    for fold, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X), start=1):
        kernel = ConstantKernel(1.0) * RBF(40.0) + WhiteKernel(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha[tr], normalize_y=True, n_restarts_optimizer=1, random_state=SEED)
        gp.fit(X[tr], y[tr])
        mean, std = gp.predict(X[te], return_std=True)
        fold_metrics.append(metrics(y[te], mean))
        for i, idx in enumerate(te):
            cv_rows.append({"fold": fold, "tg_md": X[idx, 0], "tg_exp": y[idx], "pred_mean": mean[i], "pred_std": std[i]})
    gp_full = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(40.0) + WhiteKernel(1.0), alpha=alpha, normalize_y=True, n_restarts_optimizer=1, random_state=SEED)
    gp_full.fit(X, y)
    mean, std = gp_full.predict(vitrimers[["tg"]].to_numpy(), return_std=True)
    out = vitrimers.copy()
    out["tg_calibrated"] = mean
    out["tg_calibrated_std"] = std
    grid = pd.DataFrame({"tg_md": np.linspace(calibration["tg_md"].min(), calibration["tg_md"].max(), 200)})
    gmean, gstd = gp_full.predict(grid[["tg_md"]].to_numpy(), return_std=True)
    grid["pred_mean"] = gmean
    grid["pred_std"] = gstd
    return pd.DataFrame(cv_rows), out, grid, {k: float(np.mean([m[k] for m in fold_metrics])) for k in ["mae", "rmse", "r2"]}


def make_pair_matrices(frame: pd.DataFrame, acid_lib, epoxide_lib):
    graph_x = []
    recon_x = []
    for _, row in frame.iterrows():
        acid = acid_lib[row["acid"]]
        epoxide = epoxide_lib[row["epoxide"]]
        graph_x.append(np.concatenate([acid.graph_feat, epoxide.graph_feat]))
        recon_x.append(np.concatenate([acid.desc, epoxide.desc]))
    return np.asarray(graph_x, dtype=np.float32), np.asarray(recon_x, dtype=np.float32)


def fit_fast_vae(frame: pd.DataFrame, acid_lib, epoxide_lib):
    idx = np.arange(len(frame))
    tr_val, te = train_test_split(idx, test_size=0.15, random_state=SEED)
    tr, va = train_test_split(tr_val, test_size=0.15, random_state=SEED)
    graph_x, recon_x = make_pair_matrices(frame, acid_lib, epoxide_lib)
    y = frame["tg_calibrated"].to_numpy().reshape(-1, 1)
    y_scaler = StandardScaler().fit(y[tr])
    y_scaled = y_scaler.transform(y).astype(np.float32)

    model = MatrixVAE(graph_x.shape[1], recon_x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tr = torch.tensor(graph_x[tr], dtype=torch.float32, device=DEVICE)
    recon_tr = torch.tensor(recon_x[tr], dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_scaled[tr], dtype=torch.float32, device=DEVICE)
    x_va = torch.tensor(graph_x[va], dtype=torch.float32, device=DEVICE)
    y_va = y[va].ravel()
    history = []
    best = None
    best_mae = float("inf")
    for epoch in range(1, 31):
        model.train()
        mu, logvar, recon, pred = model(x_tr)
        bit_loss = F.binary_cross_entropy_with_logits(recon[:, :256], recon_tr[:, :256])
        desc_loss = F.mse_loss(recon[:, 256:], recon_tr[:, 256:])
        prop_loss = F.mse_loss(pred, y_tr)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = prop_loss + 0.2 * (bit_loss + desc_loss) + 0.05 * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            mu_va, _, _, pred_va = model(x_va)
            pred_val = y_scaler.inverse_transform(pred_va.cpu().numpy()).ravel()
        val_mae = mean_absolute_error(y_va, pred_val)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_mae": float(val_mae)})
        if val_mae < best_mae:
            best_mae = val_mae
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)

    def predict(split_idx):
        x = torch.tensor(graph_x[split_idx], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mu, _, _, pred = model(x)
        pred_real = y_scaler.inverse_transform(pred.cpu().numpy()).ravel()
        out = pd.DataFrame({"y_true": y[split_idx].ravel(), "y_pred": pred_real, "acid": frame.iloc[split_idx]["acid"].to_list(), "epoxide": frame.iloc[split_idx]["epoxide"].to_list()})
        z = mu.cpu().numpy()
        for j in range(z.shape[1]):
            out[f"z_{j:02d}"] = z[:, j]
        return out

    preds = {"train": predict(tr), "val": predict(va), "test": predict(te)}
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "graph_vae_training_history.csv", index=False)
    return model, preds, {f"{k}_{m}": v for k, df in preds.items() for m, v in metrics(df["y_true"], df["y_pred"]).items()}, y_scaler


def generate_candidates(model, preds_train, acid_lib, epoxide_lib, scaler, observed_pairs):
    train_latent = preds_train[[c for c in preds_train.columns if c.startswith("z_")]].to_numpy()
    targets = [373.15, 423.15, 473.15]
    rows = []
    for target in targets:
        for _ in range(8):
            z = torch.randn((1, model.latent_dim), dtype=torch.float32, device=DEVICE, requires_grad=True)
            opt = torch.optim.Adam([z], lr=0.08)
            t = torch.tensor([[scaler.transform([[target]])[0, 0]]], dtype=torch.float32, device=DEVICE)
            for _ in range(30):
                pred = model.prop(z)
                loss = (pred - t).pow(2).mean() + 0.01 * z.pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                recon = model.decoder(z).cpu().numpy()[0]
                pred_y = scaler.inverse_transform(model.prop(z).cpu().numpy()).ravel()[0]
                z_np = z.cpu().numpy()[0]
            acid_scores = []
            ep_scores = []
            for smiles, rec in acid_lib.items():
                acid_scores.append((np.dot(recon[:134], rec.desc) / (np.linalg.norm(recon[:134]) * np.linalg.norm(rec.desc) + 1e-8), smiles))
            for smiles, rec in epoxide_lib.items():
                ep_scores.append((np.dot(recon[134:], rec.desc) / (np.linalg.norm(recon[134:]) * np.linalg.norm(rec.desc) + 1e-8), smiles))
            for _, acid in sorted(acid_scores, reverse=True)[:2]:
                for _, epoxide in sorted(ep_scores, reverse=True)[:2]:
                    if (acid, epoxide) in observed_pairs:
                        continue
                    dist = np.linalg.norm(train_latent - z_np, axis=1)
                    novelty = 1.0 - np.max([
                        np.logical_and(pair_fingerprint(acid, epoxide, acid_lib, epoxide_lib) > 0.5, pair_fingerprint(a, e, acid_lib, epoxide_lib) > 0.5).sum() /
                        max(np.logical_or(pair_fingerprint(acid, epoxide, acid_lib, epoxide_lib) > 0.5, pair_fingerprint(a, e, acid_lib, epoxide_lib) > 0.5).sum(), 1)
                        for a, e in list(observed_pairs)[:400]
                    ])
                    rows.append({
                        "target_tg": target,
                        "acid": acid,
                        "epoxide": epoxide,
                        "pred_tg_ensemble": float(pred_y),
                        "pred_tg_std": float(np.percentile(np.sort(dist)[:10], 90)),
                        "target_error": float(abs(pred_y - target)),
                        "novelty": float(novelty),
                        "mw_total": float(Descriptors.MolWt(Chem.MolFromSmiles(acid)) + Descriptors.MolWt(Chem.MolFromSmiles(epoxide))),
                    })
    df = pd.DataFrame(rows).drop_duplicates(subset=["acid", "epoxide", "target_tg"])
    df["selection_score"] = df["target_error"] + 0.2 * df["pred_tg_std"] + 0.01 * df["mw_total"] - 6 * df["novelty"]
    return df.sort_values(["target_tg", "selection_score"]).groupby("target_tg").head(6).reset_index(drop=True)


def save_figures(calibration, cv_df, grid_df, vitrimers, preds, candidates):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(kelvin_to_celsius(calibration["tg_md"]), kelvin_to_celsius(calibration["tg_exp"]), alpha=0.6)
    x = kelvin_to_celsius(grid_df["tg_md"])
    y = kelvin_to_celsius(grid_df["pred_mean"])
    s = grid_df["pred_std"]
    axes[1].scatter(kelvin_to_celsius(calibration["tg_md"]), kelvin_to_celsius(calibration["tg_exp"]), alpha=0.3)
    axes[1].plot(x, y, color="crimson")
    axes[1].fill_between(x, y - s, y + s, alpha=0.2)
    sns.kdeplot(kelvin_to_celsius(vitrimers["tg"]), fill=True, ax=axes[2], label="MD")
    sns.kdeplot(kelvin_to_celsius(vitrimers["tg_calibrated"]), fill=True, ax=axes[2], label="Calibrated")
    axes[2].legend(frameon=False)
    axes[0].set_title("Calibration data")
    axes[1].set_title("GP fit")
    axes[2].set_title("Library shift")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "gp_calibration_overview.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hist = pd.read_csv(OUTPUT_DIR / "graph_vae_training_history.csv")
    axes[0].plot(hist["epoch"], hist["loss"], label="loss")
    axes[0].plot(hist["epoch"], hist["val_mae"], label="val_mae")
    axes[0].legend(frameon=False)
    axes[1].scatter(kelvin_to_celsius(preds["test"]["y_true"]), kelvin_to_celsius(preds["test"]["y_pred"]), alpha=0.6)
    lim = kelvin_to_celsius(np.array([preds["test"]["y_true"].min(), preds["test"]["y_true"].max()]))
    axes[1].plot(lim, lim, "--", color="black")
    zcols = [c for c in preds["train"].columns if c.startswith("z_")]
    z2 = PCA(n_components=2, random_state=SEED).fit_transform(preds["train"][zcols].to_numpy())
    sc = axes[2].scatter(z2[:, 0], z2[:, 1], c=kelvin_to_celsius(preds["train"]["y_true"]), cmap="viridis", s=16)
    fig.colorbar(sc, ax=axes[2])
    axes[0].set_title("VAE training")
    axes[1].set_title("Held-out parity")
    axes[2].set_title("Latent space")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "graph_vae_results.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for target, sub in candidates.groupby("target_tg"):
        ax.errorbar(kelvin_to_celsius(sub["pred_tg_ensemble"]), sub["novelty"], xerr=sub["pred_tg_std"], fmt="o", label=f"{kelvin_to_celsius(np.array([target]))[0]:.0f} °C")
    ax.legend(frameon=False)
    ax.set_xlabel("Predicted Tg (°C)")
    ax.set_ylabel("Novelty")
    ax.set_title("Inverse-design candidates")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "inverse_design_candidates.png", dpi=300)
    plt.close(fig)

    mols, legends = [], []
    for _, row in candidates.groupby("target_tg").head(1).iterrows():
        mols += [Chem.MolFromSmiles(row["acid"]), Chem.MolFromSmiles(row["epoxide"])]
        legends += [f"Acid\n{row['target_tg']:.0f} K", f"Epoxide\n{row['pred_tg_ensemble']:.0f} K"]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 220), legends=legends)
    img.save(REPORT_IMG_DIR / "candidate_structures.png")


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
    ensure_dirs()
    set_seed(SEED)
    print("Loading data...", flush=True)
    calibration = pd.read_csv(DATA_DIR / "tg_calibration.csv")
    calibration["smiles_clean"] = calibration["smiles"].map(clean_polymer_smiles)
    vitrimers = pd.read_csv(DATA_DIR / "tg_vitrimer_MD.csv")
    print("Calibrating...", flush=True)
    cv_df, vitrimers_cal, grid_df, gp_metrics = gp_calibrate(calibration, vitrimers)
    print("Featurizing molecules...", flush=True)
    acid_lib = build_molecule_library(vitrimers_cal["acid"].unique())
    epoxide_lib = build_molecule_library(vitrimers_cal["epoxide"].unique())
    print("Training fast VAE...", flush=True)
    model, preds, vae_metrics, scaler = fit_fast_vae(vitrimers_cal, acid_lib, epoxide_lib)
    print("Generating candidates...", flush=True)
    candidates = generate_candidates(model, preds["train"], acid_lib, epoxide_lib, scaler, set(zip(vitrimers_cal["acid"], vitrimers_cal["epoxide"])))
    print("Saving outputs...", flush=True)
    calibration.to_csv(OUTPUT_DIR / "tg_calibration_clean.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "gp_calibration_cv_predictions.csv", index=False)
    grid_df.to_csv(OUTPUT_DIR / "gp_calibration_curve.csv", index=False)
    vitrimers_cal.to_csv(OUTPUT_DIR / "vitrimer_calibrated_predictions.csv", index=False)
    for split, df in preds.items():
        df.to_csv(OUTPUT_DIR / f"graph_vae_{split}_predictions.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "inverse_design_candidates.csv", index=False)
    save_figures(calibration, cv_df, grid_df, vitrimers_cal, preds, candidates)
    summary = {
        "calibration_gp_cv_mae": gp_metrics["mae"],
        "calibration_gp_cv_rmse": gp_metrics["rmse"],
        "calibration_gp_cv_r2": gp_metrics["r2"],
        "graph_vae_test_mae": vae_metrics["test_mae"],
        "graph_vae_test_rmse": vae_metrics["test_rmse"],
        "graph_vae_test_r2": vae_metrics["test_r2"],
    }
    with open(OUTPUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
