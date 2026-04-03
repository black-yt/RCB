#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"


@dataclass
class Frame:
    species: List[str]
    positions: np.ndarray
    forces: Optional[np.ndarray]
    info: Dict[str, object]


@dataclass
class Dataset:
    name: str
    frames: List[Frame]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_comment(comment: str) -> Dict[str, object]:
    info: Dict[str, object] = {}
    for token in shlex.split(comment):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "Properties":
            info[key] = value
            continue
        if key == "pbc":
            info[key] = value.split()
            continue
        if key == "true_charges":
            info[key] = np.array([float(x) for x in value.split()], dtype=float)
            continue
        try:
            if any(ch in value for ch in ".eE"):
                info[key] = float(value)
            else:
                info[key] = int(value)
        except ValueError:
            info[key] = value
    return info


def load_xyz(path: Path) -> Dataset:
    frames: List[Frame] = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            natoms = int(line.strip())
            comment = f.readline().strip()
            info = parse_comment(comment)
            has_forces = "forces:R:3" in str(info.get("Properties", ""))
            species: List[str] = []
            positions = []
            forces = []
            for _ in range(natoms):
                parts = f.readline().split()
                species.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
                if has_forces:
                    forces.append([float(x) for x in parts[4:7]])
            frames.append(
                Frame(
                    species=species,
                    positions=np.array(positions, dtype=float),
                    forces=np.array(forces, dtype=float) if has_forces else None,
                    info=info,
                )
            )
    return Dataset(name=path.stem, frames=frames)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def linear_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "coef": coef,
        "pred": pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_json(path: Path, obj: object) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_overview(datasets: Dict[str, Dataset]) -> None:
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, dataset) in zip(axes, datasets.items()):
        energies_all = np.array([frame.info.get("energy", np.nan) for frame in dataset.frames], dtype=float)
        energies = energies_all[np.isfinite(energies_all)]
        natoms = len(dataset.frames[0].species)
        species = sorted(set(dataset.frames[0].species))
        force_norms = []
        for frame in dataset.frames:
            if frame.forces is not None:
                force_norms.extend(np.linalg.norm(frame.forces, axis=1).tolist())
        rows.append(
            {
                "dataset": name,
                "n_frames": len(dataset.frames),
                "atoms_per_frame": natoms,
                "species": ",".join(species),
                "has_energy": bool(len(energies) > 0),
                "energy_min": float(np.min(energies)) if len(energies) else None,
                "energy_max": float(np.max(energies)) if len(energies) else None,
                "energy_mean": float(np.mean(energies)) if len(energies) else None,
                "force_norm_mean": float(np.mean(force_norms)) if force_norms else None,
            }
        )
        if len(energies):
            ax.hist(energies, bins=18, color="#4477aa", alpha=0.8, edgecolor="black")
            ax.set_xlabel("Energy")
        else:
            ax.text(0.5, 0.5, "No energy field\nin dataset", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
        ax.set_title(name)
        ax.set_ylabel("Count")
    save_csv(OUTPUT_DIR / "dataset_overview.csv", rows, list(rows[0].keys()))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_energy_overview.png", dpi=200)
    plt.close(fig)


def analyze_random_charges(dataset: Dataset) -> Dict[str, object]:
    energy_rows = []
    pair_like = []
    pair_unlike = []
    dipoles = []
    quadrupoles = []
    charge_balance = []

    for idx, frame in enumerate(dataset.frames):
        q = np.asarray(frame.info["true_charges"], dtype=float)
        pos = frame.positions
        dist = pairwise_distances(pos)
        iu = np.triu_indices(len(q), k=1)
        d = dist[iu]
        qq = q[iu[0]] * q[iu[1]]
        coulomb = np.sum(qq / d)
        repulsive = np.sum(1.0 / d**12)
        total_proxy_energy = coulomb + repulsive
        energy_rows.append(
            {
                "frame": idx,
                "coulomb_descriptor": float(coulomb),
                "repulsive_descriptor": float(repulsive),
                "total_proxy_energy": float(total_proxy_energy),
                "net_charge": float(np.sum(q)),
            }
        )
        pair_like.extend(d[qq > 0].tolist())
        pair_unlike.extend(d[qq < 0].tolist())
        charge_balance.append(float(np.sum(q)))

        dipole = np.sum(q[:, None] * pos, axis=0)
        dipoles.append(np.linalg.norm(dipole))
        r2 = np.sum(pos**2, axis=1)
        quad = np.zeros((3, 3))
        for qi, ri, r2i in zip(q, pos, r2):
            quad += qi * (3.0 * np.outer(ri, ri) - np.eye(3) * r2i)
        quadrupoles.append(np.linalg.norm(quad))

    save_csv(
        OUTPUT_DIR / "random_charges_frame_metrics.csv",
        energy_rows,
        ["frame", "coulomb_descriptor", "repulsive_descriptor", "total_proxy_energy", "net_charge"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(pair_like, bins=30, alpha=0.7, label="like-charge pairs", color="#cc6677")
    axes[0].hist(pair_unlike, bins=30, alpha=0.7, label="opposite-charge pairs", color="#4477aa")
    axes[0].set_xlabel("Pair distance")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Random charges pair-distance structure")
    axes[0].legend(frameon=False)

    axes[1].scatter(
        [row["coulomb_descriptor"] for row in energy_rows],
        [row["repulsive_descriptor"] for row in energy_rows],
        s=18,
        alpha=0.8,
        color="#228833",
    )
    axes[1].set_xlabel("Coulomb descriptor")
    axes[1].set_ylabel("Repulsive descriptor")
    axes[1].set_title("Electrostatic vs short-range contributions")

    axes[2].hist(dipoles, bins=20, color="#aa3377", alpha=0.85, edgecolor="black")
    axes[2].set_xlabel("Dipole norm")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Charge-derived dipole distribution")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "random_charges_analysis.png", dpi=200)
    plt.close(fig)

    observables = {
        "proxy_energy_mean": float(np.mean([row["total_proxy_energy"] for row in energy_rows])),
        "proxy_energy_std": float(np.std([row["total_proxy_energy"] for row in energy_rows])),
        "dipole_norm_mean": float(np.mean(dipoles)),
        "dipole_norm_std": float(np.std(dipoles)),
        "quadrupole_norm_mean": float(np.mean(quadrupoles)),
        "quadrupole_norm_std": float(np.std(quadrupoles)),
        "max_abs_net_charge": float(np.max(np.abs(charge_balance))),
    }
    save_json(OUTPUT_DIR / "random_charges_observables.json", observables)

    return observables


def charged_dimer_features(frame: Frame) -> Dict[str, float]:
    pos = frame.positions
    mol_a = pos[:4]
    mol_b = pos[4:]
    com_a = mol_a.mean(axis=0)
    com_b = mol_b.mean(axis=0)
    separation = float(np.linalg.norm(com_a - com_b))

    def intra_descriptor(mol: np.ndarray) -> float:
        d = upper_triangle_values(pairwise_distances(mol))
        return float(np.sum(1.0 / d))

    d_inter = np.linalg.norm(mol_a[:, None, :] - mol_b[None, :, :], axis=-1)
    return {
        "energy": float(frame.info.get("energy", np.nan)),
        "separation": separation,
        "inv_sep": 1.0 / separation,
        "intra_sum": intra_descriptor(mol_a) + intra_descriptor(mol_b),
        "inter_sum": float(np.sum(1.0 / d_inter)),
    }


def analyze_charged_dimer(dataset: Dataset) -> Dict[str, object]:
    rows = [charged_dimer_features(frame) | {"frame": i} for i, frame in enumerate(dataset.frames)]
    y = np.array([row["energy"] for row in rows])
    X_short = np.column_stack([np.ones(len(rows)), np.array([row["intra_sum"] for row in rows])])
    X_long = np.column_stack(
        [
            np.ones(len(rows)),
            np.array([row["intra_sum"] for row in rows]),
            np.array([row["inv_sep"] for row in rows]),
            np.array([row["inter_sum"] for row in rows]),
        ]
    )
    fit_short = linear_fit(X_short, y)
    fit_long = linear_fit(X_long, y)

    for row, ps, pl in zip(rows, fit_short["pred"], fit_long["pred"]):
        row["pred_short_range"] = float(ps)
        row["pred_long_range"] = float(pl)

    save_csv(
        OUTPUT_DIR / "charged_dimer_metrics.csv",
        rows,
        [
            "frame",
            "energy",
            "separation",
            "inv_sep",
            "intra_sum",
            "inter_sum",
            "pred_short_range",
            "pred_long_range",
        ],
    )

    order = np.argsort([row["separation"] for row in rows])
    sep = np.array([rows[i]["separation"] for i in order])
    e = y[order]
    ps = fit_short["pred"][order]
    pl = fit_long["pred"][order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(sep, e, color="black", s=22, label="reference")
    axes[0].plot(sep, ps, color="#cc6677", lw=2, label="short-range baseline")
    axes[0].plot(sep, pl, color="#4477aa", lw=2, label="long-range augmented")
    axes[0].set_xlabel("Center-of-mass separation")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Charged dimer binding-curve proxy")
    axes[0].legend(frameon=False)

    axes[1].scatter(y, fit_short["pred"], s=20, color="#cc6677", alpha=0.8, label=f"short $R^2$={fit_short['r2']:.3f}")
    axes[1].scatter(y, fit_long["pred"], s=20, color="#4477aa", alpha=0.8, label=f"long $R^2$={fit_long['r2']:.3f}")
    lims = [min(y.min(), fit_long["pred"].min(), fit_short["pred"].min()), max(y.max(), fit_long["pred"].max(), fit_short["pred"].max())]
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_xlabel("True energy")
    axes[1].set_ylabel("Predicted energy")
    axes[1].set_title("Effect of explicit long-range descriptors")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "charged_dimer_analysis.png", dpi=200)
    plt.close(fig)

    return {
        "short_range_fit": {"rmse": fit_short["rmse"], "mae": fit_short["mae"], "r2": fit_short["r2"]},
        "long_range_fit": {"rmse": fit_long["rmse"], "mae": fit_long["mae"], "r2": fit_long["r2"]},
    }


def ag3_features(frame: Frame) -> Dict[str, float]:
    dist = upper_triangle_values(pairwise_distances(frame.positions))
    dist_sorted = np.sort(dist)
    return {
        "energy": float(frame.info.get("energy", np.nan)),
        "charge_state": float(frame.info.get("charge_state", 0)),
        "d1": float(dist_sorted[0]),
        "d2": float(dist_sorted[1]),
        "d3": float(dist_sorted[2]),
        "mean_bond": float(np.mean(dist_sorted)),
    }


def analyze_ag3(dataset: Dataset) -> Dict[str, object]:
    rows = [ag3_features(frame) | {"frame": i} for i, frame in enumerate(dataset.frames)]
    y = np.array([row["energy"] for row in rows])
    geom = np.column_stack(
        [
            np.ones(len(rows)),
            np.array([row["d1"] for row in rows]),
            np.array([row["d2"] for row in rows]),
            np.array([row["d3"] for row in rows]),
            np.array([row["d1"] ** 2 for row in rows]),
            np.array([row["d2"] ** 2 for row in rows]),
            np.array([row["d3"] ** 2 for row in rows]),
        ]
    )
    geom_charge = np.column_stack([geom, np.array([row["charge_state"] for row in rows])])
    fit_geom = linear_fit(geom, y)
    fit_charge = linear_fit(geom_charge, y)

    for row, pg, pc in zip(rows, fit_geom["pred"], fit_charge["pred"]):
        row["pred_geometry_only"] = float(pg)
        row["pred_geometry_plus_charge"] = float(pc)

    save_csv(
        OUTPUT_DIR / "ag3_metrics.csv",
        rows,
        [
            "frame",
            "charge_state",
            "energy",
            "d1",
            "d2",
            "d3",
            "mean_bond",
            "pred_geometry_only",
            "pred_geometry_plus_charge",
        ],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    charge_state = np.array([row["charge_state"] for row in rows])
    mean_bond = np.array([row["mean_bond"] for row in rows])
    colors = np.where(charge_state > 0, "#4477aa", "#cc6677")
    labels_done = set()
    for xb, yy, cc, cs in zip(mean_bond, y, colors, charge_state):
        label = "+1" if cs > 0 else "-1"
        if label in labels_done:
            label = None
        else:
            labels_done.add(label)
        axes[0].scatter(xb, yy, color=cc, s=24, alpha=0.85, label=label)
    order = np.argsort(mean_bond)
    axes[0].plot(mean_bond[order], fit_geom["pred"][order], color="black", lw=1.8, label="geometry-only fit")
    axes[0].plot(mean_bond[order], fit_charge["pred"][order], color="#228833", lw=1.8, label="geometry + charge-state fit")
    axes[0].set_xlabel("Mean Ag-Ag bond length")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Ag3 charge-state dependence")
    axes[0].legend(frameon=False)

    axes[1].scatter(y, fit_geom["pred"], s=20, color="black", alpha=0.8, label=f"geom only $R^2$={fit_geom['r2']:.3f}")
    axes[1].scatter(y, fit_charge["pred"], s=20, color="#228833", alpha=0.8, label=f"+ charge $R^2$={fit_charge['r2']:.3f}")
    lims = [min(y.min(), fit_geom["pred"].min(), fit_charge["pred"].min()), max(y.max(), fit_geom["pred"].max(), fit_charge["pred"].max())]
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_xlabel("True energy")
    axes[1].set_ylabel("Predicted energy")
    axes[1].set_title("Need for global charge information")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ag3_analysis.png", dpi=200)
    plt.close(fig)

    return {
        "geometry_only_fit": {"rmse": fit_geom["rmse"], "mae": fit_geom["mae"], "r2": fit_geom["r2"]},
        "geometry_plus_charge_fit": {"rmse": fit_charge["rmse"], "mae": fit_charge["mae"], "r2": fit_charge["r2"]},
    }


def write_summary(results: Dict[str, object]) -> None:
    lines = [
        "Long-range electrostatics benchmark analysis summary",
        "",
        "This analysis creates simple, interpretable descriptor-based baselines tailored to the three provided datasets.",
        "It does not train a full LES-like neural potential, but it quantifies the electrostatic structure that such a model must capture.",
        "",
        "Key findings:",
        f"- random_charges: exact charge-derived proxy energies show mean {results['random_charges']['proxy_energy_mean']:.4f} with dipole norm mean {results['random_charges']['dipole_norm_mean']:.4f}.",
        f"- charged_dimer: adding explicit long-range descriptors improved R^2 from {results['charged_dimer']['short_range_fit']['r2']:.4f} to {results['charged_dimer']['long_range_fit']['r2']:.4f}.",
        f"- ag3_chargestates: adding charge-state information improved R^2 from {results['ag3_chargestates']['geometry_only_fit']['r2']:.4f} to {results['ag3_chargestates']['geometry_plus_charge_fit']['r2']:.4f}.",
        "",
        "Generated outputs:",
        "- outputs/dataset_overview.csv",
        "- outputs/random_charges_frame_metrics.csv",
        "- outputs/random_charges_observables.json",
        "- outputs/charged_dimer_metrics.csv",
        "- outputs/ag3_metrics.csv",
        "- outputs/analysis_summary.json",
        "- report/images/dataset_energy_overview.png",
        "- report/images/random_charges_analysis.png",
        "- report/images/charged_dimer_analysis.png",
        "- report/images/ag3_analysis.png",
    ]
    (OUTPUT_DIR / "analysis_notes.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    datasets = {
        "random_charges": load_xyz(DATA_DIR / "random_charges.xyz"),
        "charged_dimer": load_xyz(DATA_DIR / "charged_dimer.xyz"),
        "ag3_chargestates": load_xyz(DATA_DIR / "ag3_chargestates.xyz"),
    }
    dataset_overview(datasets)
    results = {
        "random_charges": analyze_random_charges(datasets["random_charges"]),
        "charged_dimer": analyze_charged_dimer(datasets["charged_dimer"]),
        "ag3_chargestates": analyze_ag3(datasets["ag3_chargestates"]),
    }
    save_json(OUTPUT_DIR / "analysis_summary.json", results)
    write_summary(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
