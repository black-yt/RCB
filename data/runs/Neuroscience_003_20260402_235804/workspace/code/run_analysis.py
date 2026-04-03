#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 42
sns.set_theme(style="whitegrid", context="talk")
np.random.seed(SEED)


@dataclass
class Dataset:
    X: np.ndarray
    X_raw: np.ndarray
    obs: pd.DataFrame
    var_names: List[str]


def ensure_dirs() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "root": root,
        "data": root / "data",
        "code": root / "code",
        "outputs": root / "outputs",
        "report": root / "report",
        "images": root / "report" / "images",
    }
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    paths["images"].mkdir(parents=True, exist_ok=True)
    return paths


def _decode_array(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.array(out, dtype=object)


def load_h5ad(path: Path) -> Dataset:
    with h5py.File(path, "r") as f:
        X = f["X"][:].astype(np.float64)
        X_raw = f["layers/raw"][:].astype(np.float64) if "layers" in f and "raw" in f["layers"] else X.copy()

        var_names = _decode_array(f["var"]["_index"][:]).tolist()

        obs_group = f["obs"]
        categories_group = obs_group.get("__categories")
        obs_dict = {}
        for key in obs_group.keys():
            if key == "__categories":
                continue
            arr = obs_group[key][:]
            if categories_group is not None and key in categories_group:
                cats = _decode_array(categories_group[key][:])
                mapped = [cats[int(i)] if int(i) < len(cats) else str(i) for i in arr]
                obs_dict[key] = mapped
            elif getattr(arr.dtype, "kind", None) in {"S", "O", "U"}:
                obs_dict[key] = _decode_array(arr)
            else:
                obs_dict[key] = arr

        obs = pd.DataFrame(obs_dict)
        if "_index" in obs.columns:
            obs.index = obs["_index"].astype(str)
        else:
            obs.index = [f"cell_{i}" for i in range(X.shape[0])]

    for col in ["batch", "phase", "state"]:
        if col in obs.columns:
            obs[col] = obs[col].astype(str)
    return Dataset(X=X, X_raw=X_raw, obs=obs, var_names=var_names)


def clean_feature_names(names: Iterable[str]) -> List[str]:
    cleaned = []
    for name in names:
        x = name.replace("Int_MeanEdge_", "").replace("_cell", "")
        cleaned.append(x)
    return cleaned


def eta_squared(values: np.ndarray, groups: Iterable[str]) -> float:
    values = np.asarray(values, dtype=float)
    groups = pd.Series(list(groups)).fillna("nan").astype(str)
    overall = values.mean()
    ss_total = np.sum((values - overall) ** 2)
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for level, idx in groups.groupby(groups).groups.items():
        subset = values[np.asarray(list(idx), dtype=int)]
        if len(subset) == 0:
            continue
        ss_between += len(subset) * (subset.mean() - overall) ** 2
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def build_graph(coords: np.ndarray, n_neighbors: int = 15) -> sparse.csr_matrix:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(coords)
    dists, inds = nbrs.kneighbors(coords)
    n = coords.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for dist, j in zip(dists[i, 1:], inds[i, 1:]):
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(dist) + 1e-6, float(dist) + 1e-6])
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return graph


def compute_pseudotime(coords: np.ndarray, ages: np.ndarray) -> np.ndarray:
    graph = build_graph(coords, n_neighbors=15)
    root_mask = ages == np.nanmin(ages)
    root_idx = np.where(root_mask)[0]
    if len(root_idx) == 0:
        root_idx = np.array([int(np.argmin(ages))])
    dist = shortest_path(graph, directed=False, indices=root_idx)
    if dist.ndim == 2:
        pt = np.min(dist, axis=0)
    else:
        pt = dist
    if np.isinf(pt).any():
        finite_max = np.nanmax(pt[np.isfinite(pt)])
        pt[~np.isfinite(pt)] = finite_max
    pt = (pt - pt.min()) / max(1e-12, pt.max() - pt.min())
    if safe_spearman(pt, ages) < 0:
        pt = 1.0 - pt
    return pt


def local_smoothness(values: np.ndarray, neighbor_idx: np.ndarray) -> float:
    diffs = []
    for i in range(len(values)):
        nbr_vals = values[neighbor_idx[i, 1:]]
        diffs.append(np.mean(np.abs(values[i] - nbr_vals)))
    diffs = np.asarray(diffs)
    denom = np.std(values) + 1e-8
    score = 1.0 - float(np.mean(diffs) / (denom + 1e-8))
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def knn_indices(X: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(X)
    return nbrs.kneighbors(return_distance=False)


def score_features(
    X: np.ndarray,
    pseudotime: np.ndarray,
    ages: np.ndarray,
    obs: pd.DataFrame,
    feature_names: List[str],
    neighbor_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(feature_names):
        values = X[:, j]
        pt_corr = abs(safe_spearman(values, pseudotime))
        age_corr = abs(safe_spearman(values, ages))
        smooth = local_smoothness(values, neighbor_idx)
        var = float(np.var(values))
        batch_eta = eta_squared(values, obs["batch"]) if "batch" in obs.columns else 0.0
        phase_eta = eta_squared(values, obs["phase"]) if "phase" in obs.columns else 0.0
        state_eta = eta_squared(values, obs["state"]) if "state" in obs.columns else 0.0
        confound = 0.50 * batch_eta + 0.30 * phase_eta + 0.20 * state_eta
        score = 0.45 * pt_corr + 0.20 * age_corr + 0.20 * smooth + 0.15 * min(var, 1.0) - 0.35 * confound
        rows.append(
            {
                "feature": name,
                "trajectory_corr": pt_corr,
                "age_corr": age_corr,
                "smoothness": smooth,
                "variance": var,
                "batch_eta2": batch_eta,
                "phase_eta2": phase_eta,
                "state_eta2": state_eta,
                "confound_score": confound,
                "final_score": score,
            }
        )
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df


def neighbor_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    scores = []
    for i in range(a.shape[0]):
        sa = set(map(int, a[i, 1:]))
        sb = set(map(int, b[i, 1:]))
        union = len(sa | sb)
        inter = len(sa & sb)
        scores.append(inter / union if union else 1.0)
    return float(np.mean(scores))


def trajectory_metrics(X_full: np.ndarray, X_sub: np.ndarray, ages: np.ndarray) -> Dict[str, float]:
    scaler_full = StandardScaler()
    scaler_sub = StandardScaler()
    Z_full = scaler_full.fit_transform(X_full)
    Z_sub = scaler_sub.fit_transform(X_sub)
    n_comp_full = min(15, Z_full.shape[1], Z_full.shape[0] - 1)
    n_comp_sub = min(15, Z_sub.shape[1], Z_sub.shape[0] - 1)
    full_coords = PCA(n_components=n_comp_full, random_state=SEED).fit_transform(Z_full)
    sub_coords = PCA(n_components=n_comp_sub, random_state=SEED).fit_transform(Z_sub)
    pt_full = compute_pseudotime(full_coords, ages)
    pt_sub = compute_pseudotime(sub_coords, ages)
    knn_full = knn_indices(full_coords, n_neighbors=15)
    knn_sub = knn_indices(sub_coords, n_neighbors=15)
    return {
        "pseudotime_spearman_vs_full": abs(safe_spearman(pt_full, pt_sub)),
        "age_spearman_full": abs(safe_spearman(pt_full, ages)),
        "age_spearman_subset": abs(safe_spearman(pt_sub, ages)),
        "knn_jaccard_vs_full": neighbor_jaccard(knn_full, knn_sub),
        "explained_variance_pc1_full": float(np.var(full_coords[:, 0]) / np.var(Z_full).sum()) if np.var(Z_full).sum() > 0 else 0.0,
        "explained_variance_pc1_subset": float(np.var(sub_coords[:, 0]) / np.var(Z_sub).sum()) if np.var(Z_sub).sum() > 0 else 0.0,
    }


def evaluate_subset_sizes(X: np.ndarray, ranking: pd.DataFrame, ages: np.ndarray, subset_sizes: List[int]) -> pd.DataFrame:
    results = []
    for k in subset_sizes:
        sel = ranking.head(k).index.to_numpy()
        metrics = trajectory_metrics(X, X[:, sel], ages)
        metrics["n_features"] = k
        results.append(metrics)
    return pd.DataFrame(results)


def plot_overview(obs: pd.DataFrame, coords: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    sns.histplot(obs["annotated_age"], bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Annotated age distribution")
    axes[0].set_xlabel("annotated_age")

    if "state" in obs.columns:
        state_counts = obs["state"].value_counts().rename_axis("state").reset_index(name="count")
        sns.barplot(data=state_counts, x="state", y="count", ax=axes[1], palette="viridis")
        axes[1].set_title("Cell state counts")
        axes[1].tick_params(axis="x", rotation=25)
    else:
        axes[1].axis("off")

    scatter = axes[2].scatter(coords[:, 0], coords[:, 1], c=obs["annotated_age"], s=10, cmap="viridis")
    axes[2].set_title("Full-data PCA colored by annotated age")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[2], label="annotated_age")

    if "batch" in obs.columns:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=obs["batch"], s=18, linewidth=0, ax=axes[3], palette="Set2")
        axes[3].set_title("Full-data PCA colored by batch")
        axes[3].set_xlabel("PC1")
        axes[3].set_ylabel("PC2")
        axes[3].legend(title="batch", loc="best", frameon=True)
    else:
        axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_scores(scores: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    top = scores.head(top_n).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["final_score"], color="#55A868")
    ax.set_title(f"Top {top_n} trajectory-preserving features")
    ax.set_xlabel("final_score")
    ax.set_ylabel("feature")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(X: np.ndarray, feature_names: List[str], pseudotime: np.ndarray, out: Path, max_cells: int = 600) -> None:
    order = np.argsort(pseudotime)
    if len(order) > max_cells:
        keep = np.linspace(0, len(order) - 1, max_cells).astype(int)
        order = order[keep]
    X_ord = X[order]
    X_ord = (X_ord - X_ord.mean(axis=0)) / (X_ord.std(axis=0) + 1e-8)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(X_ord.T, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "z-score"}, yticklabels=feature_names)
    ax.set_title("Selected features ordered by inferred pseudotime")
    ax.set_xlabel("Cells (ordered by pseudotime)")
    ax.set_ylabel("Selected features")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_comparison(full_coords: np.ndarray, sub_coords: np.ndarray, obs: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc1 = axes[0].scatter(full_coords[:, 0], full_coords[:, 1], c=obs["annotated_age"], s=10, cmap="viridis")
    axes[0].set_title("Full feature set PCA")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc1, ax=axes[0], label="annotated_age")

    sc2 = axes[1].scatter(sub_coords[:, 0], sub_coords[:, 1], c=obs["annotated_age"], s=10, cmap="viridis")
    axes[1].set_title("Selected feature subset PCA")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(sc2, ax=axes[1], label="annotated_age")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation(eval_df: pd.DataFrame, out: Path) -> None:
    long_df = eval_df.melt(id_vars="n_features", value_vars=["pseudotime_spearman_vs_full", "age_spearman_subset", "knn_jaccard_vs_full"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=long_df, x="n_features", y="value", hue="metric", marker="o", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Trajectory preservation across subset sizes")
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel("Preservation score")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def choose_best_subset(eval_df: pd.DataFrame) -> int:
    score = 0.45 * eval_df["pseudotime_spearman_vs_full"] + 0.35 * eval_df["knn_jaccard_vs_full"] + 0.20 * eval_df["age_spearman_subset"]
    return int(eval_df.loc[score.idxmax(), "n_features"])


def save_summary(paths: Dict[str, Path], ds: Dataset, scores: pd.DataFrame, eval_df: pd.DataFrame, best_k: int, metrics: Dict[str, float]) -> None:
    summary = {
        "n_cells": int(ds.X.shape[0]),
        "n_features": int(ds.X.shape[1]),
        "best_subset_size": int(best_k),
        "selected_features": scores.head(best_k)["feature"].tolist(),
        "best_subset_metrics": metrics,
        "metadata_columns": ds.obs.columns.tolist(),
    }
    (paths["outputs"] / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    paths = ensure_dirs()
    ds = load_h5ad(paths["data"] / "adata_RPE.h5ad")
    pretty_names = clean_feature_names(ds.var_names)
    ds.obs["annotated_age"] = pd.to_numeric(ds.obs["annotated_age"], errors="coerce")
    ds.obs["annotated_age"] = ds.obs["annotated_age"].fillna(ds.obs["annotated_age"].median())

    scaler = StandardScaler()
    Z = scaler.fit_transform(ds.X)
    n_comp = min(15, Z.shape[1], Z.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca_coords = pca.fit_transform(Z)

    pseudotime = compute_pseudotime(pca_coords, ds.obs["annotated_age"].to_numpy())
    ds.obs["pseudotime"] = pseudotime

    neighbor_idx = knn_indices(pca_coords, n_neighbors=15)
    scores = score_features(
        X=Z,
        pseudotime=pseudotime,
        ages=ds.obs["annotated_age"].to_numpy(),
        obs=ds.obs,
        feature_names=pretty_names,
        neighbor_idx=neighbor_idx,
    )

    subset_sizes = [5, 10, 15, 20, 25, 30, 40, 50]
    eval_df = evaluate_subset_sizes(Z, scores, ds.obs["annotated_age"].to_numpy(), subset_sizes)
    best_k = choose_best_subset(eval_df)
    best_idx = scores.head(best_k).index.to_numpy()

    subset_Z = Z[:, best_idx]
    full_2d = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    sub_2d = PCA(n_components=2, random_state=SEED).fit_transform(subset_Z)
    best_metrics = trajectory_metrics(Z, subset_Z, ds.obs["annotated_age"].to_numpy())

    ds.obs.to_csv(paths["outputs"] / "cell_metadata_with_pseudotime.csv", index=True)
    scores.to_csv(paths["outputs"] / "feature_ranking.csv", index=False)
    eval_df.to_csv(paths["outputs"] / "subset_evaluation.csv", index=False)
    pd.DataFrame({"selected_feature": scores.head(best_k)["feature"]}).to_csv(paths["outputs"] / "selected_features.csv", index=False)
    save_summary(paths, ds, scores, eval_df, best_k, best_metrics)

    plot_overview(ds.obs, full_2d, paths["images"] / "data_overview.png")
    plot_feature_scores(scores, paths["images"] / "top_feature_scores.png", top_n=min(20, len(scores)))
    plot_heatmap(subset_Z[:, : min(best_k, subset_Z.shape[1])], scores.head(best_k)["feature"].tolist(), pseudotime, paths["images"] / "selected_feature_heatmap.png")
    plot_embedding_comparison(full_2d, sub_2d, ds.obs, paths["images"] / "embedding_comparison.png")
    plot_validation(eval_df, paths["images"] / "subset_validation.png")

    ranked_preview = scores.head(15).copy()
    ranked_preview.insert(0, "rank", np.arange(1, len(ranked_preview) + 1))
    ranked_preview.to_csv(paths["outputs"] / "top15_feature_ranking_detailed.csv", index=False)

    with open(paths["outputs"] / "run_manifest.txt", "w") as fh:
        fh.write("Main analysis entry point: code/run_analysis.py\n")
        fh.write(f"Best subset size: {best_k}\n")
        fh.write("Generated figures:\n")
        for name in [
            "data_overview.png",
            "top_feature_scores.png",
            "selected_feature_heatmap.png",
            "embedding_comparison.png",
            "subset_validation.png",
        ]:
            fh.write(f"- report/images/{name}\n")


if __name__ == "__main__":
    main()
