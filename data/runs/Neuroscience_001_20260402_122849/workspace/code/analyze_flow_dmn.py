from __future__ import annotations

import json
import os
import pickle
import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


DATA_DIR = ROOT / "data" / "flow" / "0000"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def install_flyvis_stubs() -> None:
    modules = [
        "flyvis",
        "flyvis.analysis",
        "flyvis.analysis.clustering",
        "flyvis.analysis.visualization",
        "flyvis.analysis.visualization.embedding",
    ]
    for name in modules:
        sys.modules.setdefault(name, types.ModuleType(name))

    class GaussianMixtureClustering:  # pragma: no cover - pickle shim
        pass

    class Embedding:  # pragma: no cover - pickle shim
        pass

    for mod_name in [
        "flyvis.analysis.clustering",
        "flyvis.analysis.visualization.embedding",
    ]:
        module = sys.modules[mod_name]
        module.GaussianMixtureClustering = GaussianMixtureClustering
        module.Embedding = Embedding


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoints() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    model_rows: list[dict[str, float | str]] = []
    tensor_store: dict[str, list[np.ndarray]] = {}
    decoder_store: dict[str, list[np.ndarray]] = {}

    model_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir() and path.name.isdigit())
    for model_dir in model_dirs:
        model_id = model_dir.name
        meta = load_yaml(model_dir / "_meta.yaml")
        checkpoint = torch.load(model_dir / "best_chkpt", map_location="cpu")
        network_state = checkpoint["network"]
        decoder_state = checkpoint["decoder"]["flow"]

        with h5py.File(model_dir / "validation_loss.h5", "r") as handle:
            validation_loss = float(handle["data"][()])

        row: dict[str, float | str] = {
            "model_id": model_id,
            "validation_loss": validation_loss,
            "fold": int(meta["config"]["task"]["fold"]),
            "seed": int(meta["config"]["task"]["seed"]),
            "batch_size": int(meta["config"]["task"]["batch_size"]),
            "n_iters": int(meta["config"]["task"]["n_iters"]),
        }
        model_rows.append(row)

        for name, tensor in network_state.items():
            tensor_store.setdefault(name, []).append(tensor.detach().cpu().numpy().astype(np.float64))
        for name, tensor in decoder_state.items():
            decoder_store.setdefault(name, []).append(tensor.detach().cpu().numpy().astype(np.float64))

    model_df = pd.DataFrame(model_rows).sort_values("model_id").reset_index(drop=True)
    stacked_tensors = {name: np.stack(arrays, axis=0) for name, arrays in tensor_store.items()}
    stacked_decoder = {name: np.stack(arrays, axis=0) for name, arrays in decoder_store.items()}
    return model_df, stacked_tensors, stacked_decoder


def compute_pca_coordinates(model_df: pd.DataFrame, stacked_tensors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    learnable_names = ["nodes_bias", "nodes_time_const", "edges_syn_strength"]
    matrices = [stacked_tensors[name].reshape(stacked_tensors[name].shape[0], -1) for name in learnable_names]
    x = np.concatenate(matrices, axis=1)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    x = x / scale

    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(x)
    model_df = model_df.copy()
    model_df["pc1"] = coords[:, 0]
    model_df["pc2"] = coords[:, 1]
    model_df["pc3"] = coords[:, 2]
    model_df["centroid_distance"] = np.linalg.norm(x, axis=1)
    model_df["loss_rank"] = model_df["validation_loss"].rank(method="dense")
    explained = pca.explained_variance_ratio_
    return model_df, explained


def load_celltype_embeddings() -> tuple[pd.DataFrame, pd.DataFrame]:
    install_flyvis_stubs()
    warnings.filterwarnings("ignore", category=UserWarning)

    summary_rows: list[dict[str, float | int | str]] = []
    point_rows: list[dict[str, float | int | str]] = []
    pickles = sorted((DATA_DIR / "umap_and_clustering").glob("*.pickle"))
    for path in pickles:
        with path.open("rb") as handle:
            obj = pickle.load(handle)

        cell_type = path.stem
        embedding = np.asarray(obj.embedding.__dict__["_embedding"], dtype=np.float64)
        labels = np.asarray(obj.labels, dtype=int)
        cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
        n_clusters = int(getattr(obj.gm, "n_components", len(cluster_ids)))
        centroid = embedding.mean(axis=0)
        spread = float(np.sqrt(((embedding - centroid) ** 2).sum(axis=1).mean()))
        x_span = float(embedding[:, 0].max() - embedding[:, 0].min())
        y_span = float(embedding[:, 1].max() - embedding[:, 1].min())
        hull_area = 0.0
        if embedding.shape[0] >= 3:
            try:
                hull_area = float(ConvexHull(embedding).volume)
            except Exception:
                hull_area = 0.0
        silhouette = np.nan
        if len(cluster_ids) > 1 and cluster_counts.min() > 1:
            silhouette = float(silhouette_score(embedding, labels))

        for model_index, (point, label) in enumerate(zip(embedding, labels)):
            point_rows.append(
                {
                    "cell_type": cell_type,
                    "model_index": model_index,
                    "model_id": f"{model_index:03d}",
                    "embed_x": float(point[0]),
                    "embed_y": float(point[1]),
                    "cluster": int(label),
                }
            )

        summary_rows.append(
            {
                "cell_type": cell_type,
                "n_models": int(embedding.shape[0]),
                "n_clusters": n_clusters,
                "cluster_balance": float(cluster_counts.max() / cluster_counts.sum()),
                "spread": spread,
                "x_span": x_span,
                "y_span": y_span,
                "convex_hull_area": hull_area,
                "silhouette": silhouette,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["n_clusters", "spread"], ascending=[False, False]).reset_index(drop=True)
    points_df = pd.DataFrame(point_rows).sort_values(["cell_type", "model_id"]).reset_index(drop=True)
    return summary_df, points_df


def add_performance_association(celltype_summary_df: pd.DataFrame, celltype_points_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss_by_model = model_df.set_index("model_id")["validation_loss"]
    rows: list[dict[str, float | int | str]] = []
    assoc_rows: list[dict[str, float | int | str]] = []

    for cell_type, subdf in celltype_points_df.groupby("cell_type", sort=True):
        merged = subdf.join(loss_by_model, on="model_id")
        losses = merged["validation_loss"].to_numpy(dtype=np.float64)
        grand_mean = losses.mean()
        ss_total = float(((losses - grand_mean) ** 2).sum())
        ss_between = 0.0
        for cluster_id, cluster_df in merged.groupby("cluster", sort=True):
            group_losses = cluster_df["validation_loss"].to_numpy(dtype=np.float64)
            ss_between += float(len(group_losses) * (group_losses.mean() - grand_mean) ** 2)
            rows.append(
                {
                    "cell_type": cell_type,
                    "cluster": int(cluster_id),
                    "n_models": int(len(group_losses)),
                    "mean_validation_loss": float(group_losses.mean()),
                    "std_validation_loss": float(group_losses.std(ddof=0)),
                    "min_validation_loss": float(group_losses.min()),
                    "max_validation_loss": float(group_losses.max()),
                }
            )
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        assoc_rows.append({"cell_type": cell_type, "validation_eta_sq": eta_sq})

    cluster_perf_df = pd.DataFrame(rows)
    assoc_df = pd.DataFrame(assoc_rows)
    merged_summary = celltype_summary_df.merge(assoc_df, on="cell_type", how="left")
    merged_summary["diversity_score"] = merged_summary["spread"].rank(pct=True) * merged_summary["validation_eta_sq"].rank(pct=True)
    return merged_summary, cluster_perf_df


def compute_label_nmi_matrix(celltype_points_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    label_map: dict[str, np.ndarray] = {}
    for cell_type, subdf in celltype_points_df.groupby("cell_type", sort=True):
        ordered = subdf.sort_values("model_id")
        label_map[cell_type] = ordered["cluster"].to_numpy(dtype=int)

    cell_types = list(label_map)
    n = len(cell_types)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, cell_i in enumerate(cell_types):
        for j, cell_j in enumerate(cell_types):
            if i == j:
                matrix[i, j] = 1.0
            elif j > i:
                value = normalized_mutual_info_score(label_map[cell_i], label_map[cell_j])
                matrix[i, j] = value
                matrix[j, i] = value

    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ordered_names = [cell_types[i] for i in order]
    ordered_matrix = pd.DataFrame(matrix[np.ix_(order, order)], index=ordered_names, columns=ordered_names)
    return ordered_matrix, ordered_names


def summarize_tensors(stacked_tensors: dict[str, np.ndarray], stacked_decoder: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    learnable = {"nodes_bias", "nodes_time_const", "edges_syn_strength"}
    fixed = {"edges_sign", "edges_syn_count"}

    for name, tensor in stacked_tensors.items():
        per_feature_std = tensor.std(axis=0).reshape(-1)
        rows.append(
            {
                "tensor": name,
                "group": "learned" if name in learnable else "fixed" if name in fixed else "other",
                "n_features": int(np.prod(tensor.shape[1:])),
                "ensemble_mean": float(tensor.mean()),
                "ensemble_std": float(tensor.std()),
                "feature_std_mean": float(per_feature_std.mean()),
                "feature_std_median": float(np.median(per_feature_std)),
                "feature_std_max": float(per_feature_std.max()),
            }
        )

    for name, tensor in stacked_decoder.items():
        per_feature_std = tensor.std(axis=0).reshape(-1)
        rows.append(
            {
                "tensor": f"decoder.flow::{name}",
                "group": "decoder",
                "n_features": int(np.prod(tensor.shape[1:])),
                "ensemble_mean": float(tensor.mean()),
                "ensemble_std": float(tensor.std()),
                "feature_std_mean": float(per_feature_std.mean()),
                "feature_std_median": float(np.median(per_feature_std)),
                "feature_std_max": float(per_feature_std.max()),
            }
        )

    return pd.DataFrame(rows).sort_values(["group", "tensor"]).reset_index(drop=True)


def save_tables(model_df: pd.DataFrame, tensor_summary_df: pd.DataFrame, celltype_summary_df: pd.DataFrame, celltype_points_df: pd.DataFrame, cluster_perf_df: pd.DataFrame, nmi_df: pd.DataFrame, explained_variance: np.ndarray) -> None:
    model_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    tensor_summary_df.to_csv(OUTPUT_DIR / "tensor_summary.csv", index=False)
    celltype_summary_df.to_csv(OUTPUT_DIR / "celltype_summary.csv", index=False)
    celltype_points_df.to_csv(OUTPUT_DIR / "celltype_embedding_points.csv", index=False)
    cluster_perf_df.to_csv(OUTPUT_DIR / "celltype_cluster_performance.csv", index=False)
    nmi_df.to_csv(OUTPUT_DIR / "celltype_label_nmi.csv")

    summary_payload = {
        "n_models": int(len(model_df)),
        "validation_loss": {
            "mean": float(model_df["validation_loss"].mean()),
            "std": float(model_df["validation_loss"].std(ddof=0)),
            "min": float(model_df["validation_loss"].min()),
            "max": float(model_df["validation_loss"].max()),
        },
        "pca_explained_variance_ratio": [float(x) for x in explained_variance],
        "n_cell_types": int(celltype_summary_df.shape[0]),
        "cluster_count_distribution": celltype_summary_df["n_clusters"].value_counts().sort_index().to_dict(),
        "top_diverse_cell_types": celltype_summary_df.sort_values("spread", ascending=False).head(10)["cell_type"].tolist(),
        "top_performance_associated_cell_types": celltype_summary_df.sort_values("validation_eta_sq", ascending=False).head(10)["cell_type"].tolist(),
    }
    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_overview(model_df: pd.DataFrame, tensor_summary_df: pd.DataFrame, explained_variance: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.histplot(model_df["validation_loss"], bins=12, kde=True, color="#355C7D", ax=axes[0, 0])
    axes[0, 0].set_title("Validation Loss Across 50 DMNs")
    axes[0, 0].set_xlabel("Validation loss")

    tensor_plot_df = tensor_summary_df[tensor_summary_df["group"].isin(["learned", "fixed", "decoder"])].copy()
    sns.barplot(data=tensor_plot_df, y="tensor", x="n_features", hue="group", dodge=False, palette="deep", ax=axes[0, 1])
    axes[0, 1].set_title("Parameter Inventory")
    axes[0, 1].set_xlabel("Features per tensor")
    axes[0, 1].set_ylabel("")
    axes[0, 1].legend(title="")

    scatter = axes[1, 0].scatter(
        model_df["pc1"],
        model_df["pc2"],
        c=model_df["validation_loss"],
        cmap="viridis_r",
        s=90,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1, 0].set_title(
        f"Learned-Parameter PCA (PC1 {explained_variance[0]:.1%}, PC2 {explained_variance[1]:.1%})"
    )
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[1, 0], label="Validation loss")

    feature_std_df = tensor_summary_df[tensor_summary_df["group"].isin(["learned", "fixed", "decoder"])].copy()
    sns.barplot(
        data=feature_std_df,
        y="tensor",
        x="feature_std_mean",
        hue="group",
        dodge=False,
        palette="deep",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Mean Feature-Level Ensemble Variability")
    axes[1, 1].set_xlabel("Mean standard deviation across models")
    axes[1, 1].set_ylabel("")
    axes[1, 1].legend_.remove()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_01_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_celltype_diversity(celltype_summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    scatter = axes[0].scatter(
        celltype_summary_df["spread"],
        celltype_summary_df["validation_eta_sq"],
        c=celltype_summary_df["n_clusters"],
        cmap="magma",
        s=80,
        edgecolor="black",
        linewidth=0.2,
    )
    axes[0].set_title("Cell-Type Diversity vs. Performance Relevance")
    axes[0].set_xlabel("Embedding spread")
    axes[0].set_ylabel("Variance in validation loss explained by clusters")
    fig.colorbar(scatter, ax=axes[0], label="Cluster count")

    label_df = pd.concat(
        [
            celltype_summary_df.nlargest(8, "spread"),
            celltype_summary_df.nlargest(8, "validation_eta_sq"),
        ]
    ).drop_duplicates("cell_type")
    for _, row in label_df.iterrows():
        axes[0].annotate(row["cell_type"], (row["spread"], row["validation_eta_sq"]), fontsize=10, xytext=(4, 4), textcoords="offset points")

    top_spread = celltype_summary_df.nlargest(18, "spread").sort_values("spread", ascending=True)
    sns.barplot(
        data=top_spread,
        y="cell_type",
        x="spread",
        hue="n_clusters",
        dodge=False,
        palette="magma",
        ax=axes[1],
    )
    axes[1].set_title("Most Diverse Cell Types")
    axes[1].set_xlabel("Embedding spread")
    axes[1].set_ylabel("")
    axes[1].legend(title="Clusters")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_02_celltype_diversity.png", bbox_inches="tight")
    plt.close(fig)


def plot_selected_embeddings(celltype_summary_df: pd.DataFrame, celltype_points_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    selected = (
        celltype_summary_df.sort_values(["diversity_score", "spread"], ascending=False)
        .head(9)["cell_type"]
        .tolist()
    )
    merged = celltype_points_df.merge(model_df[["model_id", "validation_loss"]], on="model_id", how="left")

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()
    for ax, cell_type in zip(axes, selected):
        subdf = merged[merged["cell_type"] == cell_type].copy()
        sns.scatterplot(
            data=subdf,
            x="embed_x",
            y="embed_y",
            hue="cluster",
            size="validation_loss",
            palette="tab10",
            sizes=(50, 180),
            linewidth=0.3,
            edgecolor="black",
            legend=False,
            ax=ax,
        )
        meta = celltype_summary_df.set_index("cell_type").loc[cell_type]
        ax.set_title(
            f"{cell_type}\nclusters={int(meta['n_clusters'])}, spread={meta['spread']:.2f}, eta²={meta['validation_eta_sq']:.2f}"
        )
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_03_selected_embeddings.png", bbox_inches="tight")
    plt.close(fig)


def plot_cluster_performance(celltype_summary_df: pd.DataFrame, celltype_points_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    selected = celltype_summary_df.sort_values("validation_eta_sq", ascending=False).head(6)["cell_type"].tolist()
    merged = celltype_points_df.merge(model_df[["model_id", "validation_loss"]], on="model_id", how="left")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    for ax, cell_type in zip(axes, selected):
        subdf = merged[merged["cell_type"] == cell_type].copy()
        sns.boxplot(data=subdf, x="cluster", y="validation_loss", color="#A8DADC", fliersize=0, ax=ax)
        sns.stripplot(data=subdf, x="cluster", y="validation_loss", color="#1D3557", size=4, alpha=0.7, ax=ax)
        eta_sq = celltype_summary_df.set_index("cell_type").loc[cell_type, "validation_eta_sq"]
        ax.set_title(f"{cell_type} (eta²={eta_sq:.2f})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Validation loss")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_04_cluster_performance.png", bbox_inches="tight")
    plt.close(fig)


def plot_nmi_heatmap(nmi_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(nmi_df, cmap="rocket_r", vmin=0.0, vmax=1.0, square=True, cbar_kws={"label": "Normalized mutual information"}, ax=ax)
    ax.set_title("Shared Ensemble State Structure Across Cell Types")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_05_nmi_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    configure_style()

    model_df, stacked_tensors, stacked_decoder = load_checkpoints()
    model_df, explained_variance = compute_pca_coordinates(model_df, stacked_tensors)
    celltype_summary_df, celltype_points_df = load_celltype_embeddings()
    celltype_summary_df, cluster_perf_df = add_performance_association(celltype_summary_df, celltype_points_df, model_df)
    nmi_df, _ = compute_label_nmi_matrix(celltype_points_df)
    tensor_summary_df = summarize_tensors(stacked_tensors, stacked_decoder)

    centroid_corr = spearmanr(model_df["centroid_distance"], model_df["validation_loss"])
    model_df["centroid_loss_spearman_r"] = centroid_corr.correlation
    model_df["centroid_loss_spearman_p"] = centroid_corr.pvalue

    save_tables(
        model_df=model_df,
        tensor_summary_df=tensor_summary_df,
        celltype_summary_df=celltype_summary_df,
        celltype_points_df=celltype_points_df,
        cluster_perf_df=cluster_perf_df,
        nmi_df=nmi_df,
        explained_variance=explained_variance,
    )

    plot_overview(model_df, tensor_summary_df, explained_variance)
    plot_celltype_diversity(celltype_summary_df)
    plot_selected_embeddings(celltype_summary_df, celltype_points_df, model_df)
    plot_cluster_performance(celltype_summary_df, celltype_points_df, model_df)
    plot_nmi_heatmap(nmi_df)


if __name__ == "__main__":
    main()
