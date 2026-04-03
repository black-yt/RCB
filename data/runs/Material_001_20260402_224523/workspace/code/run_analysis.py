#!/usr/bin/env python3
"""Main analysis entry point for the synthetic multimodal materials AI benchmark.

This script parses the provided text dataset and performs three task-aligned analyses:
1. Property prediction from synthetic graph/node features.
2. Structure-generation style distribution modeling for lattice-like parameters.
3. Autonomous optimization over synthetic processing variables.

Outputs are written under outputs/ and report/images/.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "M-AI-Synth__Materials_AI_Dataset_.txt"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def parse_dataset(path: Path) -> Dict[str, List[list]]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    sections: Dict[str, List[list]] = {}
    current = None
    for line in lines:
        if line.startswith("#"):
            if "property_prediction.py" in line:
                current = "property_prediction"
            elif "structure_generation.py" in line:
                current = "structure_generation"
            elif "autonomous_optimization.py" in line:
                current = "autonomous_optimization"
            else:
                current = None
            if current is not None:
                sections[current] = []
            continue
        if current is None:
            continue
        if line.startswith("[") and line.endswith("]"):
            sections[current].append(ast.literal_eval(line))
    return sections


def build_property_dataset(section: List[list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(section[0], dtype=int)
    node_values = np.asarray(section[1], dtype=float)
    edge_index_flat = np.asarray(section[2], dtype=int)
    targets = np.asarray(section[3], dtype=float)

    # The synthetic text block is not perfectly rectangular. We therefore build
    # graph-like samples using a rolling window whose width is inferred from the
    # repeated node-count metadata (all values are 5 in this dataset).
    window = int(np.median(counts)) if len(counts) else 5
    edge_pairs = edge_index_flat.reshape(-1, 2)
    usable = min(len(node_values), len(targets))

    features = []
    graph_targets = []
    for start in range(0, usable - window + 1):
        vals = node_values[start : start + window]
        tvals = targets[start : start + window]

        feat = [
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.median(vals)),
            float(np.mean(np.square(vals))),
            float(np.max(vals) - np.min(vals)),
            float(window),
            float(len(edge_pairs)),
            float(2 * len(edge_pairs) / max(window, 1)),
            float(start / max(usable - window, 1)),
        ]
        features.append(feat)
        graph_targets.append(float(np.mean(tvals)))

    feature_names = np.array([
        "node_mean",
        "node_std",
        "node_min",
        "node_max",
        "node_median",
        "node_sq_mean",
        "node_range",
        "n_nodes",
        "n_edges",
        "avg_degree",
        "relative_position",
    ])
    return np.asarray(features, dtype=float), np.asarray(graph_targets, dtype=float), feature_names


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    Xd = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ coef
    return {"coef": coef, "pred": pred}


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> Dict[str, np.ndarray]:
    Xd = np.column_stack([np.ones(len(X)), X])
    reg = np.eye(Xd.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y)
    pred = Xd @ coef
    return {"coef": coef, "pred": pred, "alpha": np.array([alpha])}


def leave_one_out_predictions(X: np.ndarray, y: np.ndarray, model: str = "linear") -> np.ndarray:
    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        Xte = X[~mask]
        if model == "linear":
            coef = fit_linear_regression(Xtr, ytr)["coef"]
        else:
            coef = fit_ridge_closed_form(Xtr, ytr)["coef"]
        Xte_d = np.column_stack([np.ones(len(Xte)), Xte])
        preds.append(float((Xte_d @ coef)[0]))
    return np.asarray(preds)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / denom if denom > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def analyze_property_prediction(section: List[list]) -> Dict[str, object]:
    X, y, feature_names = build_property_dataset(section)

    linear_fit = fit_linear_regression(X, y)
    ridge_fit = fit_ridge_closed_form(X, y)
    loo_linear = leave_one_out_predictions(X, y, model="linear")
    loo_ridge = leave_one_out_predictions(X, y, model="ridge")

    metrics = {
        "train_linear": regression_metrics(y, linear_fit["pred"]),
        "train_ridge": regression_metrics(y, ridge_fit["pred"]),
        "loo_linear": regression_metrics(y, loo_linear),
        "loo_ridge": regression_metrics(y, loo_ridge),
    }

    np.savetxt(OUTPUT_DIR / "property_features.csv", X, delimiter=",", header=",".join(feature_names), comments="")
    np.savetxt(OUTPUT_DIR / "property_targets.csv", y, delimiter=",", header="target", comments="")

    coeff_table = {
        "intercept": float(ridge_fit["coef"][0]),
        **{name: float(val) for name, val in zip(feature_names, ridge_fit["coef"][1:])},
    }
    with open(OUTPUT_DIR / "property_prediction_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "ridge_coefficients": coeff_table}, f, indent=2)

    # Figure 1: feature/target overview
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs[0, 0].plot(X[:, 0], y, "o", alpha=0.8)
    axs[0, 0].set_xlabel("Mean node feature")
    axs[0, 0].set_ylabel("Mean target")
    axs[0, 0].set_title("Property target vs node-feature mean")

    axs[0, 1].hist(y, bins=12, color="#4C72B0", alpha=0.85, edgecolor="black")
    axs[0, 1].set_title("Distribution of graph-level targets")
    axs[0, 1].set_xlabel("Target")

    stds = np.std(X, axis=0)
    corr = np.corrcoef(X, rowvar=False)
    corr[~np.isfinite(corr)] = 0.0
    im = axs[1, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axs[1, 0].set_xticks(range(len(feature_names)))
    axs[1, 0].set_yticks(range(len(feature_names)))
    axs[1, 0].set_xticklabels(feature_names, rotation=90, fontsize=8)
    axs[1, 0].set_yticklabels(feature_names, fontsize=8)
    axs[1, 0].set_title("Feature correlation matrix")
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046)

    axs[1, 1].plot(y, label="Observed", marker="o")
    axs[1, 1].plot(loo_ridge, label="LOO ridge prediction", marker="s")
    axs[1, 1].set_title("Observed vs cross-validated predictions")
    axs[1, 1].set_xlabel("Graph index")
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "property_prediction_overview.png", dpi=200)
    plt.close(fig)

    # Figure 2: parity plot and coefficient importance
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    lo = min(np.min(y), np.min(loo_ridge))
    hi = max(np.max(y), np.max(loo_ridge))
    axs[0].scatter(y, loo_ridge, color="#55A868")
    axs[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axs[0].set_xlabel("Observed target")
    axs[0].set_ylabel("Predicted target")
    axs[0].set_title("LOO ridge parity plot")

    coef_vals = ridge_fit["coef"][1:]
    order = np.argsort(np.abs(coef_vals))[::-1]
    axs[1].bar(np.array(feature_names)[order], coef_vals[order], color="#C44E52")
    axs[1].tick_params(axis="x", rotation=90)
    axs[1].set_title("Ridge coefficient magnitudes")
    axs[1].set_ylabel("Coefficient")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "property_prediction_validation.png", dpi=200)
    plt.close(fig)

    return {
        "n_graphs": int(len(y)),
        "feature_names": feature_names.tolist(),
        "metrics": metrics,
    }


def analyze_structure_generation(section: List[list]) -> Dict[str, object]:
    seq_a = np.asarray(section[0], dtype=float)
    seq_b = np.asarray(section[1], dtype=float)
    stacked = np.column_stack([seq_a, seq_b])
    mean = np.mean(stacked, axis=0)
    cov = np.cov(stacked.T)
    rng = np.random.default_rng(42)
    generated = rng.multivariate_normal(mean, cov + 1e-6 * np.eye(2), size=200)

    summary = {
        "sequence_a": {
            "mean": float(np.mean(seq_a)),
            "std": float(np.std(seq_a)),
            "min": float(np.min(seq_a)),
            "max": float(np.max(seq_a)),
        },
        "sequence_b": {
            "mean": float(np.mean(seq_b)),
            "std": float(np.std(seq_b)),
            "min": float(np.min(seq_b)),
            "max": float(np.max(seq_b)),
        },
        "correlation": float(np.corrcoef(seq_a, seq_b)[0, 1]),
        "generated_count": int(len(generated)),
    }

    np.savetxt(OUTPUT_DIR / "structure_generated_samples.csv", generated, delimiter=",", header="a_like,b_like", comments="")
    with open(OUTPUT_DIR / "structure_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].hist(seq_a, bins=14, alpha=0.7, label="Sequence A", color="#4C72B0", edgecolor="black")
    axs[0].hist(seq_b, bins=14, alpha=0.7, label="Sequence B", color="#DD8452", edgecolor="black")
    axs[0].legend()
    axs[0].set_xlabel("Lattice-like parameter value")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Empirical structure-parameter distributions")

    axs[1].scatter(seq_a, seq_b, s=20, alpha=0.65, label="Observed", color="#55A868")
    axs[1].scatter(generated[:, 0], generated[:, 1], s=16, alpha=0.35, label="Generated", color="#C44E52")
    axs[1].set_xlabel("a-like parameter")
    axs[1].set_ylabel("b-like parameter")
    axs[1].set_title("Observed vs generated structure space")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "structure_generation_analysis.png", dpi=200)
    plt.close(fig)

    return summary


def synthetic_objective(temp: np.ndarray, time: np.ndarray) -> np.ndarray:
    # Smooth surrogate objective: maximize around a physically plausible center.
    return (
        1.8 * np.exp(-((temp - 360.0) / 55.0) ** 2 - ((time - 18.0) / 5.5) ** 2)
        + 0.18 * np.sin(temp / 36.0)
        - 0.015 * ((time - 20.0) ** 2) / 10.0
        - 0.0009 * ((temp - 350.0) ** 2) / 10.0
    )


def analyze_optimization(section: List[list]) -> Dict[str, object]:
    temp_bounds = np.asarray(section[0], dtype=float)
    time_bounds = np.asarray(section[1], dtype=float)
    initial_temp = float(section[2][0])
    initial_time = float(section[3][0])
    step_fraction = float(section[4][0])
    n_iterations = int(section[5][0])

    temps = np.linspace(temp_bounds[0], temp_bounds[1], 120)
    times = np.linspace(time_bounds[0], time_bounds[1], 120)
    TT, HH = np.meshgrid(temps, times)
    surface = synthetic_objective(TT, HH)

    flat_idx = int(np.argmax(surface))
    opt_row, opt_col = np.unravel_index(flat_idx, surface.shape)
    best_temp = float(TT[opt_row, opt_col])
    best_time = float(HH[opt_row, opt_col])
    best_score = float(surface[opt_row, opt_col])

    current_temp, current_time = initial_temp, initial_time
    history = []
    for i in range(n_iterations):
        current_score = float(synthetic_objective(np.array([current_temp]), np.array([current_time]))[0])
        history.append([i, current_temp, current_time, current_score])
        current_temp = current_temp + step_fraction * (best_temp - current_temp)
        current_time = current_time + step_fraction * (best_time - current_time)

    history = np.asarray(history, dtype=float)
    np.savetxt(
        OUTPUT_DIR / "optimization_trajectory.csv",
        history,
        delimiter=",",
        header="iteration,temperature,time,score",
        comments="",
    )

    summary = {
        "bounds": {
            "temperature": temp_bounds.tolist(),
            "time": time_bounds.tolist(),
        },
        "initial_point": {"temperature": initial_temp, "time": initial_time},
        "best_point": {"temperature": best_temp, "time": best_time, "score": best_score},
        "iterations": n_iterations,
        "step_fraction": step_fraction,
        "final_score": float(history[-1, 3]),
    }
    with open(OUTPUT_DIR / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    cs = axs[0].contourf(TT, HH, surface, levels=20, cmap="viridis")
    axs[0].plot(history[:, 1], history[:, 2], "w-o", markersize=3, linewidth=1.5, label="Search trajectory")
    axs[0].scatter([best_temp], [best_time], color="red", marker="*", s=180, label="Estimated optimum")
    axs[0].set_xlabel("Temperature")
    axs[0].set_ylabel("Time")
    axs[0].set_title("Synthetic process optimization surface")
    axs[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(cs, ax=axs[0], fraction=0.046)

    axs[1].plot(history[:, 0], history[:, 3], marker="o", color="#C44E52")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Objective score")
    axs[1].set_title("Optimization improvement trajectory")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "optimization_analysis.png", dpi=200)
    plt.close(fig)

    return summary


def write_master_summary(parsed: Dict[str, List[list]], prop: Dict[str, object], struct: Dict[str, object], opt: Dict[str, object]) -> None:
    overview = {
        "dataset_sections": {k: len(v) for k, v in parsed.items()},
        "property_prediction": prop,
        "structure_generation": struct,
        "optimization": opt,
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)


def main() -> None:
    ensure_dirs()
    parsed = parse_dataset(DATA_FILE)
    prop = analyze_property_prediction(parsed["property_prediction"])
    struct = analyze_structure_generation(parsed["structure_generation"])
    opt = analyze_optimization(parsed["autonomous_optimization"])
    write_master_summary(parsed, prop, struct, opt)
    print("Analysis complete. Outputs written to outputs/ and report/images/.")


if __name__ == "__main__":
    main()
