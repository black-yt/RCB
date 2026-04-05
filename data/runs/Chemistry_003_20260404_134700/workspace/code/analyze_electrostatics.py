from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"
SEED = 7

np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="talk")


def parse_comment(line: str) -> Dict[str, str]:
    return dict(re.findall(r'(\S+?)=(".*?"|\S+)', line.strip()))


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def load_extxyz(path: Path) -> List[dict]:
    frames: List[dict] = []
    with path.open() as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            n_atoms = int(first)
            comment = handle.readline().strip()
            meta = {k: strip_quotes(v) for k, v in parse_comment(comment).items()}
            atoms = []
            positions = []
            forces = []
            has_forces = False
            for _ in range(n_atoms):
                toks = handle.readline().split()
                atoms.append(toks[0])
                positions.append([float(x) for x in toks[1:4]])
                if len(toks) >= 7:
                    forces.append([float(x) for x in toks[4:7]])
                    has_forces = True
            frame = {
                "species": np.array(atoms),
                "positions": np.array(positions, dtype=float),
                "forces": np.array(forces, dtype=float) if has_forces else None,
                "energy": float(meta["energy"]) if "energy" in meta else None,
                "meta": meta,
            }
            if "true_charges" in meta:
                frame["true_charges"] = np.array([float(x) for x in shlex.split(meta["true_charges"])], dtype=float)
            frames.append(frame)
    return frames


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    delta = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> List[float]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(values, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)]


def summarize_random_charges(frames: List[dict]):
    rows = []
    for frame_idx, frame in enumerate(frames):
        pos = frame["positions"]
        q = frame["true_charges"]
        distances = pairwise_distances(pos)
        same = []
        opposite = []
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                if q[i] == q[j]:
                    same.append(distances[i, j])
                else:
                    opposite.append(distances[i, j])
        rows.append(
            {
                "frame": frame_idx,
                "n_atoms": len(q),
                "same_sign_mean_distance": float(np.mean(same)),
                "opposite_sign_mean_distance": float(np.mean(opposite)),
                "distance_gap_same_minus_opposite": float(np.mean(same) - np.mean(opposite)),
                "forces_available": False,
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "n_frames": int(len(frames)),
        "forces_available": False,
        "sign_recovery_possible_from_forces": False,
        "reason": "random_charges.xyz stores positions and true_charges but no forces or energies, so force-based sign recovery cannot be performed using only local files.",
        "mean_same_sign_distance": float(df["same_sign_mean_distance"].mean()),
        "mean_opposite_sign_distance": float(df["opposite_sign_mean_distance"].mean()),
        "mean_distance_gap_same_minus_opposite": float(df["distance_gap_same_minus_opposite"].mean()),
        "distance_gap_ci95": bootstrap_mean_ci(df["distance_gap_same_minus_opposite"].to_numpy()),
    }
    return df, summary


def charged_dimer_features(frame: dict) -> Dict[str, float]:
    pos = frame["positions"]
    mol_ids = np.array([0] * 4 + [1] * 4)
    dmat = pairwise_distances(pos)
    c0 = pos[mol_ids == 0].mean(axis=0)
    c1 = pos[mol_ids == 1].mean(axis=0)
    intra_means = []
    intra_stds = []
    for mol in [0, 1]:
        sub = dmat[np.ix_(mol_ids == mol, mol_ids == mol)]
        tri = sub[np.triu_indices(4, k=1)]
        intra_means.append(tri.mean())
        intra_stds.append(tri.std())
    inter = dmat[np.ix_(mol_ids == 0, mol_ids == 1)].ravel()
    return {
        "energy": float(frame["energy"]),
        "com_distance": float(np.linalg.norm(c0 - c1)),
        "inter_inv_r_sum": float(np.sum(1.0 / inter)),
        "inter_inv_r_mean": float(np.mean(1.0 / inter)),
        "inter_short_sum": float(np.sum(np.exp(-inter))),
        "inter_short_sq_sum": float(np.sum(np.exp(-(inter ** 2)))),
        "mol0_bond_mean": float(intra_means[0]),
        "mol1_bond_mean": float(intra_means[1]),
        "mol0_bond_std": float(intra_stds[0]),
        "mol1_bond_std": float(intra_stds[1]),
    }


def fit_cv_predictions(X: np.ndarray, y: np.ndarray, model_factory, splitter):
    preds = np.zeros_like(y, dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[test_idx])
        preds[test_idx] = fold_pred
        row = regression_metrics(y[test_idx], fold_pred)
        row["fold"] = fold
        fold_rows.append(row)
    return preds, pd.DataFrame(fold_rows)


def run_charged_dimer(frames: List[dict]):
    df = pd.DataFrame([charged_dimer_features(frame) for frame in frames])
    short_cols = ["inter_short_sum", "inter_short_sq_sum", "mol0_bond_mean", "mol1_bond_mean", "mol0_bond_std", "mol1_bond_std"]
    coul_cols = short_cols + ["inter_inv_r_sum", "inter_inv_r_mean", "com_distance"]
    y = df["energy"].to_numpy()
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pred_short, cv_short = fit_cv_predictions(df[short_cols].to_numpy(), y, lambda: Ridge(alpha=1e-8), splitter)
    pred_coul, cv_coul = fit_cv_predictions(df[coul_cols].to_numpy(), y, lambda: Ridge(alpha=1e-8), splitter)
    out = df.copy()
    out["pred_short_range"] = pred_short
    out["pred_coulomb_aware"] = pred_coul
    fold_df = pd.concat(
        [cv_short.assign(model="short_range_only"), cv_coul.assign(model="coulomb_aware")],
        ignore_index=True,
    )
    mae_short = mean_absolute_error(y, pred_short)
    mae_coul = mean_absolute_error(y, pred_coul)
    summary = {
        "n_frames": int(len(df)),
        "short_range_only": regression_metrics(y, pred_short),
        "coulomb_aware": regression_metrics(y, pred_coul),
        "delta_mae": float(mae_short - mae_coul),
        "delta_mae_percent": float(100.0 * (mae_short - mae_coul) / mae_short),
    }
    return out, fold_df, summary


def ag3_features(frame: dict, include_charge: bool) -> Dict[str, float]:
    pos = frame["positions"]
    dmat = pairwise_distances(pos)
    tri = np.sort(dmat[np.triu_indices(3, k=1)])
    area = 0.5 * np.linalg.norm(np.cross(pos[1] - pos[0], pos[2] - pos[0]))
    feats = {
        "r1": float(tri[0]),
        "r2": float(tri[1]),
        "r3": float(tri[2]),
        "mean_r": float(tri.mean()),
        "std_r": float(tri.std()),
        "area": float(area),
        "energy": float(frame["energy"]),
        "total_charge": float(frame["meta"].get("total_charge", 0.0)),
    }
    if not include_charge:
        feats.pop("total_charge")
    return feats


def run_ag3(frames: List[dict]):
    geom_df = pd.DataFrame([ag3_features(frame, include_charge=False) for frame in frames])
    geom_charge_df = pd.DataFrame([ag3_features(frame, include_charge=True) for frame in frames])
    y = geom_charge_df["energy"].to_numpy()
    groups = geom_charge_df["total_charge"].to_numpy()
    splitter = GroupKFold(n_splits=2)
    pred_geom = np.zeros_like(y, dtype=float)
    pred_geom_charge = np.zeros_like(y, dtype=float)
    fold_rows = []
    X_geom = geom_df.drop(columns=["energy"]).to_numpy()
    X_geom_charge = geom_charge_df.drop(columns=["energy"]).to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_geom, y, groups=groups)):
        model_geom = LinearRegression().fit(X_geom[train_idx], y[train_idx])
        model_geom_charge = LinearRegression().fit(X_geom_charge[train_idx], y[train_idx])
        pg = model_geom.predict(X_geom[test_idx])
        pgc = model_geom_charge.predict(X_geom_charge[test_idx])
        pred_geom[test_idx] = pg
        pred_geom_charge[test_idx] = pgc
        fold_rows.append({"fold": fold, "model": "geometry_only", **regression_metrics(y[test_idx], pg)})
        fold_rows.append({"fold": fold, "model": "geometry_plus_charge", **regression_metrics(y[test_idx], pgc)})
    out = geom_charge_df.copy()
    out["pred_geometry_only"] = pred_geom
    out["pred_geometry_plus_charge"] = pred_geom_charge
    mae_geom = mean_absolute_error(y, pred_geom)
    mae_geom_charge = mean_absolute_error(y, pred_geom_charge)
    summary = {
        "n_frames": int(len(out)),
        "geometry_only": regression_metrics(y, pred_geom),
        "geometry_plus_charge": regression_metrics(y, pred_geom_charge),
        "delta_mae": float(mae_geom - mae_geom_charge),
        "delta_mae_percent": float(100.0 * (mae_geom - mae_geom_charge) / mae_geom),
        "note": "In this dataset, +1 and -1 entries are duplicated geometries/energies, so charge labels may not add new information beyond geometry.",
    }
    return out, pd.DataFrame(fold_rows), summary


def make_figures(random_df: pd.DataFrame, dimer_df: pd.DataFrame, ag_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(random_df["distance_gap_same_minus_opposite"], bins=20, ax=axes[0], color="#4c72b0")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("Random charges: geometry distance gap")
    axes[0].set_xlabel("mean(same-sign d) - mean(opposite-sign d)")

    order = np.argsort(dimer_df["com_distance"].to_numpy())
    ordered = dimer_df.iloc[order]
    axes[1].plot(ordered["com_distance"], ordered["energy"], "o-", label="reference")
    axes[1].plot(ordered["com_distance"], ordered["pred_short_range"], "s--", label="short-range")
    axes[1].plot(ordered["com_distance"], ordered["pred_coulomb_aware"], "d--", label="coulomb-aware")
    axes[1].set_title("Charged dimer regression")
    axes[1].set_xlabel("COM distance")
    axes[1].set_ylabel("Energy")
    axes[1].legend(fontsize=10)

    sns.scatterplot(data=ag_df, x="energy", y="pred_geometry_only", hue="total_charge", ax=axes[2])
    sns.scatterplot(data=ag_df, x="energy", y="pred_geometry_plus_charge", hue="total_charge", style="total_charge", ax=axes[2], legend=False, marker="X")
    lo = min(ag_df["energy"].min(), ag_df[["pred_geometry_only", "pred_geometry_plus_charge"]].min().min())
    hi = max(ag_df["energy"].max(), ag_df[["pred_geometry_only", "pred_geometry_plus_charge"]].max().max())
    axes[2].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[2].set_title("Ag3: geometry vs geometry+charge")
    axes[2].set_xlabel("Reference energy")
    axes[2].set_ylabel("Predicted energy")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "electrostatics_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    err_df = pd.DataFrame(
        {
            "short_range_only": np.abs(dimer_df["energy"] - dimer_df["pred_short_range"]),
            "coulomb_aware": np.abs(dimer_df["energy"] - dimer_df["pred_coulomb_aware"]),
        }
    )
    sns.boxplot(data=err_df, ax=ax)
    ax.set_ylabel("Absolute error")
    ax.set_title("Charged dimer absolute error")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "charged_dimer_error_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.scatterplot(data=ag_df, x="energy", y="pred_geometry_only", hue="total_charge", style="total_charge", ax=axes[0])
    sns.scatterplot(data=ag_df, x="energy", y="pred_geometry_plus_charge", hue="total_charge", style="total_charge", ax=axes[1])
    for ax, title in zip(axes, ["Geometry only", "Geometry + charge"]):
        lo = min(ag_df["energy"].min(), ag_df[["pred_geometry_only", "pred_geometry_plus_charge"]].min().min())
        hi = max(ag_df["energy"].max(), ag_df[["pred_geometry_only", "pred_geometry_plus_charge"]].max().max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Reference energy")
        ax.set_ylabel("Predicted energy")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "ag3_charge_conditioning.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    random_frames = load_extxyz(DATA_DIR / "random_charges.xyz")
    dimer_frames = load_extxyz(DATA_DIR / "charged_dimer.xyz")
    ag3_frames = load_extxyz(DATA_DIR / "ag3_chargestates.xyz")

    random_df, random_summary = summarize_random_charges(random_frames)
    dimer_df, dimer_cv_df, dimer_summary = run_charged_dimer(dimer_frames)
    ag3_df, ag3_cv_df, ag3_summary = run_ag3(ag3_frames)

    random_df.to_csv(OUTPUT_DIR / "random_charge_analysis.csv", index=False)
    dimer_df.to_csv(OUTPUT_DIR / "charged_dimer_predictions.csv", index=False)
    dimer_cv_df.to_csv(OUTPUT_DIR / "charged_dimer_cv_metrics.csv", index=False)
    ag3_df.to_csv(OUTPUT_DIR / "ag3_predictions.csv", index=False)
    ag3_cv_df.to_csv(OUTPUT_DIR / "ag3_cv_metrics.csv", index=False)

    summary = {
        "seed": SEED,
        "random_charges": random_summary,
        "charged_dimer": dimer_summary,
        "ag3": ag3_summary,
    }
    with (OUTPUT_DIR / "electrostatics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    make_figures(random_df, dimer_df, ag3_df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
