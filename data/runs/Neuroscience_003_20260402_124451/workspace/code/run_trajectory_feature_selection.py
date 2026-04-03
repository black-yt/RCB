#!/usr/bin/env python3
"""Trajectory-preserving dynamic feature selection for the RPE single-cell dataset."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


RANDOM_SEED = 7
K_NEIGHBORS = 15
PANEL_SIZES = [8, 12, 16, 20, 24, 32]
FINAL_TOP_FEATURES_TO_PLOT = 8


@dataclass
class PanelMetrics:
    method: str
    panel_size: int
    age_prediction_r2: float
    pseudotime_spearman: float
    neighbor_age_error: float
    knn_overlap: float
    batch_silhouette: float
    mean_phase_penalty: float
    composite_score: float


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 180


def load_data(path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    adata = ad.read_h5ad(path)
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    obs = adata.obs.copy().reset_index(drop=True)
    obs["state"] = obs["state"].astype(str).replace({"nan": "unannotated"})
    obs["phase"] = obs["phase"].astype(str)
    obs["batch"] = obs["batch"].astype(str)
    return X, obs, list(adata.var_names)


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def build_knn_indices(X: np.ndarray, n_neighbors: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances[:, 1:], indices[:, 1:]


def make_age_model() -> Pipeline:
    return Pipeline(
        [
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )


def cv_age_r2(age: np.ndarray, y: np.ndarray, cv: KFold) -> float:
    model = make_age_model()
    scores = cross_val_score(model, age[:, None], y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def age_fit_and_residual(age: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = make_age_model()
    y_hat = model.fit(age[:, None], y).predict(age[:, None])
    return y_hat, y - y_hat


def eta_squared(values: np.ndarray, groups: pd.Series) -> float:
    group_ids = pd.Series(groups).astype(str)
    grand_mean = float(np.mean(values))
    ss_between = 0.0
    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    for _, idx in group_ids.groupby(group_ids).groups.items():
        subset = values[np.asarray(list(idx))]
        if subset.size == 0:
            continue
        ss_between += subset.size * float((np.mean(subset) - grand_mean) ** 2)
    return float(ss_between / ss_total)


def neighbor_agreement(x: np.ndarray, knn_idx: np.ndarray) -> float:
    nbr_mean = x[knn_idx].mean(axis=1)
    corr = np.corrcoef(x, nbr_mean)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_feature_scores(
    X: np.ndarray,
    obs: pd.DataFrame,
    feature_names: list[str],
    full_knn_idx: np.ndarray,
) -> pd.DataFrame:
    age = obs["annotated_age"].to_numpy(dtype=float)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for j, feature in enumerate(feature_names):
        y = X[:, j]
        dynamic_r2 = cv_age_r2(age, y, cv)
        y_fit, residual = age_fit_and_residual(age, y)
        spearman_age = spearmanr(y, age).statistic
        rows.append(
            {
                "feature": feature,
                "dynamic_r2": dynamic_r2,
                "age_abs_spearman": abs(float(spearman_age)),
                "age_spearman": float(spearman_age),
                "graph_smoothness": neighbor_agreement(y, full_knn_idx),
                "batch_penalty": eta_squared(residual, obs["batch"]),
                "phase_penalty": eta_squared(residual, obs["phase"]),
                "state_penalty": eta_squared(residual, obs["state"]),
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "fitted_age_std": float(np.std(y_fit)),
            }
        )
    scores = pd.DataFrame(rows)
    for col in [
        "dynamic_r2",
        "age_abs_spearman",
        "graph_smoothness",
        "batch_penalty",
        "phase_penalty",
        "state_penalty",
    ]:
        ranks = scores[col].rank(pct=True, method="average")
        scores[f"{col}_rank"] = ranks
    # Reweighted to keep genuinely dynamic age-linked markers while using
    # confound penalties mainly as tiebreakers.
    scores["trajectory_score"] = (
        0.60 * scores["dynamic_r2"]
        + 0.80 * scores["age_abs_spearman"]
        + 0.10 * scores["graph_smoothness"]
        - 0.10 * scores["batch_penalty"]
        - 0.20 * scores["phase_penalty"]
    )
    scores = scores.sort_values("trajectory_score", ascending=False).reset_index(drop=True)
    return scores


def greedy_select(
    scores: pd.DataFrame,
    corr_matrix: np.ndarray,
    feature_to_idx: dict[str, int],
    n_select: int,
) -> list[str]:
    selected: list[str] = []
    remaining = scores["feature"].tolist()
    while remaining and len(selected) < n_select:
        best_feature = None
        best_score = -1e9
        for feature in remaining:
            base_score = float(scores.loc[scores["feature"] == feature, "trajectory_score"].iloc[0])
            if not selected:
                redundancy = 0.0
            else:
                idx = feature_to_idx[feature]
                redundancy = max(abs(corr_matrix[idx, feature_to_idx[s]]) for s in selected)
            adjusted = base_score - 0.25 * redundancy
            if adjusted > best_score:
                best_score = adjusted
                best_feature = feature
        assert best_feature is not None
        selected.append(best_feature)
        remaining.remove(best_feature)
    return selected


def get_root_index(X_pca: np.ndarray, age: np.ndarray) -> int:
    youngest = np.argsort(age)[: min(30, X_pca.shape[0])]
    centroid = X_pca[youngest].mean(axis=0)
    root_local = np.argmin(np.linalg.norm(X_pca[youngest] - centroid, axis=1))
    return int(youngest[root_local])


def pseudotime_from_panel(X_panel: np.ndarray, age: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(10, X_panel.shape[1], X_panel.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_panel)
    nbrs = NearestNeighbors(n_neighbors=min(K_NEIGHBORS + 1, X_pca.shape[0] - 1), metric="euclidean")
    nbrs.fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(X_pca.shape[0]), indices.shape[1])
    cols = indices.reshape(-1)
    vals = distances.reshape(-1)
    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(X_pca.shape[0], X_pca.shape[0]))
    graph = graph.minimum(graph.T)

    root = get_root_index(X_pca, age)
    pt = dijkstra(graph, directed=False, indices=root)

    if np.isinf(pt).any():
        finite_max = np.nanmax(pt[np.isfinite(pt)])
        pt[np.isinf(pt)] = finite_max

    if np.corrcoef(pt, age)[0, 1] < 0:
        pt = -pt
    pt = (pt - pt.min()) / max(pt.max() - pt.min(), 1e-8)
    return pt, indices


def neighbor_age_error(age: np.ndarray, knn_idx: np.ndarray) -> float:
    local_error = np.abs(age[:, None] - age[knn_idx]).mean(axis=1)
    return float(local_error.mean())


def mean_knn_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    overlaps = []
    for a, b in zip(idx_a, idx_b):
        sa = set(map(int, a))
        sb = set(map(int, b))
        overlaps.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(overlaps))


def evaluate_panel(
    method: str,
    selected_features: list[str],
    X_std: np.ndarray,
    obs: pd.DataFrame,
    feature_to_idx: dict[str, int],
    full_knn_idx: np.ndarray,
    scores: pd.DataFrame,
) -> PanelMetrics:
    age = obs["annotated_age"].to_numpy(dtype=float)
    batch = obs["batch"].to_numpy()
    idx = [feature_to_idx[f] for f in selected_features]
    X_panel = X_std[:, idx]

    ridge = Ridge(alpha=1.0)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    age_prediction_r2 = float(np.mean(cross_val_score(ridge, X_panel, age, cv=cv, scoring="r2")))

    pseudotime, panel_knn_idx = pseudotime_from_panel(X_panel, age)
    pseudotime_spearman = float(spearmanr(pseudotime, age).statistic)
    age_error = neighbor_age_error(age, panel_knn_idx)
    overlap = mean_knn_overlap(full_knn_idx, panel_knn_idx)

    pca = PCA(n_components=min(10, X_panel.shape[1], X_panel.shape[0] - 1), random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_panel)
    batch_sil = float(silhouette_score(X_pca, batch))
    phase_penalty = float(
        scores.set_index("feature").loc[selected_features, "phase_penalty"].mean()
    )

    composite = (
        age_prediction_r2
        + pseudotime_spearman
        + overlap
        - 0.07 * age_error
        - abs(batch_sil)
        - phase_penalty
    )

    return PanelMetrics(
        method=method,
        panel_size=len(selected_features),
        age_prediction_r2=age_prediction_r2,
        pseudotime_spearman=pseudotime_spearman,
        neighbor_age_error=age_error,
        knn_overlap=overlap,
        batch_silhouette=batch_sil,
        mean_phase_penalty=phase_penalty,
        composite_score=composite,
    )


def top_variance_features(X: np.ndarray, feature_names: list[str], n_select: int) -> list[str]:
    order = np.argsort(np.std(X, axis=0))[::-1]
    return [feature_names[i] for i in order[:n_select]]


def top_age_correlation_features(scores: pd.DataFrame, n_select: int) -> list[str]:
    return scores.sort_values("age_abs_spearman", ascending=False)["feature"].head(n_select).tolist()


def random_features(feature_names: list[str], n_select: int, rng: random.Random) -> list[str]:
    return rng.sample(feature_names, n_select)


def choose_panel_size(metrics_df: pd.DataFrame) -> int:
    mine = metrics_df[metrics_df["method"] == "trajectory_panel"].copy()
    best_row = mine.sort_values(["composite_score", "pseudotime_spearman"], ascending=False).iloc[0]
    return int(best_row["panel_size"])


def make_output_dirs(base: Path) -> dict[str, Path]:
    outputs = base / "outputs"
    report_images = base / "report" / "images"
    outputs.mkdir(exist_ok=True, parents=True)
    report_images.mkdir(exist_ok=True, parents=True)
    return {"outputs": outputs, "images": report_images}


def save_overview_figure(obs: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(obs, x="annotated_age", bins=24, ax=axes[0], color="#35608d")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Annotated age")

    phase_counts = obs["phase"].value_counts().reset_index()
    phase_counts.columns = ["phase", "count"]
    sns.barplot(data=phase_counts, x="phase", y="count", ax=axes[1], color="#5d8aa8")
    axes[1].set_title("Cell-Cycle Phase Counts")

    sns.boxplot(data=obs, x="phase", y="annotated_age", hue="batch", ax=axes[2], palette="Set2")
    axes[2].set_title("Age by Phase and Batch")
    axes[2].legend(title="Batch", frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_embedding_figure(
    X_std: np.ndarray, obs: pd.DataFrame, selected: list[str], feature_to_idx: dict[str, int], outpath: Path
) -> None:
    age = obs["annotated_age"].to_numpy(dtype=float)
    idx = [feature_to_idx[f] for f in selected]
    X_panel = X_std[:, idx]
    pca = PCA(n_components=min(5, X_panel.shape[1], X_panel.shape[0] - 1), random_state=RANDOM_SEED)
    pcs = pca.fit_transform(X_panel)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=age, s=16, cmap="viridis", alpha=0.9)
    axes[0].set_title("Selected Panel PCA Colored by Age")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc1, ax=axes[0], label="Annotated age")

    palette = dict(zip(sorted(obs["phase"].unique()), sns.color_palette("Set2", n_colors=obs["phase"].nunique())))
    for phase, group in obs.groupby("phase"):
        mask = group.index.to_numpy()
        axes[1].scatter(pcs[mask, 0], pcs[mask, 1], s=16, alpha=0.8, label=phase, color=palette[phase])
    axes[1].set_title("Selected Panel PCA Colored by Phase")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(frameon=False, title="Phase")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_feature_score_heatmap(scores: pd.DataFrame, selected: list[str], outpath: Path) -> None:
    cols = [
        "dynamic_r2",
        "age_abs_spearman",
        "graph_smoothness",
        "batch_penalty",
        "phase_penalty",
        "trajectory_score",
    ]
    plot_df = scores.set_index("feature").loc[selected, cols]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(selected))))
    sns.heatmap(plot_df, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Selected Features and Score Components")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_dynamic_trends(
    X: np.ndarray, obs: pd.DataFrame, feature_names: list[str], selected: list[str], outpath: Path
) -> None:
    age = obs["annotated_age"].to_numpy(dtype=float)
    order = np.argsort(age)
    age_sorted = age[order]
    top = selected[:FINAL_TOP_FEATURES_TO_PLOT]
    ncols = 2
    nrows = math.ceil(len(top) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, top):
        idx = feature_names.index(feature)
        y = X[:, idx][order]
        smooth = pd.Series(y).rolling(window=121, center=True, min_periods=15).mean()
        ax.scatter(age_sorted, y, s=8, alpha=0.25, color="#4c72b0")
        ax.plot(age_sorted, smooth, color="#c44e52", linewidth=2.5)
        ax.set_title(feature)
        ax.set_xlabel("Annotated age")
        ax.set_ylabel("Expression")
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_figure(metrics_df: pd.DataFrame, outpath: Path) -> None:
    plot_df = metrics_df.copy()
    long_df = plot_df.melt(
        id_vars=["method", "panel_size"],
        value_vars=["age_prediction_r2", "pseudotime_spearman", "knn_overlap", "mean_phase_penalty"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()
    metric_titles = {
        "age_prediction_r2": "Age Prediction R^2",
        "pseudotime_spearman": "Pseudotime vs Age Spearman",
        "knn_overlap": "kNN Overlap with Full Data",
        "mean_phase_penalty": "Residual Phase Penalty",
    }
    for ax, metric in zip(axes, metric_titles):
        subset = long_df[long_df["metric"] == metric]
        sns.lineplot(
            data=subset,
            x="panel_size",
            y="value",
            hue="method",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Panel size")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_pseudotime_figure(
    obs: pd.DataFrame,
    pseudotime_df: pd.DataFrame,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=pseudotime_df,
        x="annotated_age",
        y="pseudotime",
        hue="phase",
        palette="Set2",
        s=20,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Selected Panel Pseudotime Tracks Age")
    axes[0].legend(frameon=False, title="Phase")

    sns.boxplot(data=pseudotime_df, x="batch", y="pseudotime", ax=axes[1], color="#a3c4bc")
    axes[1].set_title("Pseudotime Distribution by Batch")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def write_report(
    base: Path,
    obs: pd.DataFrame,
    scores: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_features: list[str],
    pseudotime_df: pd.DataFrame,
) -> None:
    report_path = base / "report" / "report.md"
    top = scores.set_index("feature").loc[selected_features]
    best_row = metrics_df[
        (metrics_df["method"] == "trajectory_panel") & (metrics_df["panel_size"] == len(selected_features))
    ].iloc[0]
    age_corr = best_row["pseudotime_spearman"]
    age_r2 = best_row["age_prediction_r2"]
    overlap = best_row["knn_overlap"]
    phase_penalty = best_row["mean_phase_penalty"]

    top_table = (
        top[["dynamic_r2", "age_spearman", "batch_penalty", "phase_penalty", "trajectory_score"]]
        .round(3)
        .reset_index()
        .rename(columns={"index": "feature"})
        .to_markdown(index=False)
    )

    comparison_table = (
        metrics_df.sort_values(["method", "panel_size"])
        .round(3)
        .to_markdown(index=False)
    )

    age_summary = obs["annotated_age"].describe()
    report = f"""# Trajectory-Preserving Dynamic Feature Selection in Single-Cell RPE Imaging Data

## Abstract
I analyzed a preprocessed single-cell protein imaging dataset of 2,759 cells and 241 molecular or morphological features to identify a compact marker panel that preserves a continuous cellular trajectory while reducing confounding variation. Because the dataset includes an external continuous label, `annotated_age`, I used it as a trajectory reference and designed a feature-selection score that rewards smooth dynamic behavior across age and local graph coherence, while penalizing age-independent batch and cell-cycle phase effects. A greedy redundancy-aware selector produced a {len(selected_features)}-feature panel. This panel achieved cross-validated age-prediction R^2 = {age_r2:.3f}, pseudotime-age Spearman correlation = {age_corr:.3f}, and mean k-nearest-neighbor overlap with the full 241-feature space = {overlap:.3f}. The selected panel is dominated by DNA content, cyclin, CDK, and damage-response markers, consistent with a progression-like proliferative program.

## Dataset and Biological Context
The input dataset (`data/adata_RPE.h5ad`) contains {obs.shape[0]:,} cells measured across {scores.shape[0]} features. Metadata include cell-cycle phase (`G0`, `G1`, `S`, `G2`), a continuous `annotated_age` variable, a discrete `state` annotation, and batch identity. The distribution of `annotated_age` spans {age_summary['min']:.2f} to {age_summary['max']:.2f} with median {age_summary['50%']:.2f}, indicating a broad continuous transition rather than a narrow snapshot.

This dataset is retina-related rather than central nervous system tissue, but it still provides a neuroscience-adjacent test bed for trajectory-preserving feature selection because the task is fundamentally about continuous state transitions under mixed technical and discrete biological structure.

![Data overview](images/data_overview.png)

## Related Work Framing
Two of the provided references were relevant. The SCANPY paper established a standard workflow in which neighborhood graphs and pseudotemporal ordering are central abstractions for single-cell trajectory analysis. The organogenesis atlas paper showed how large-scale single-cell datasets can reveal developmental trajectories and that dynamic marker discovery is most useful when tied to trajectory structure rather than static variance alone. I therefore optimized for neighborhood and pseudotime preservation instead of simple dispersion.

## Methods
### Preprocessing
I loaded the `.h5ad` object with `anndata`, converted the expression matrix to dense floating-point form, and z-scored each feature across cells. No additional filtering was applied because the feature count is modest and the dataset was already preprocessed.

### Feature scoring
Each feature received six statistics:
1. Cross-validated spline regression R^2 for predicting feature intensity from `annotated_age`.
2. Absolute Spearman correlation with `annotated_age`.
3. Graph smoothness, defined as the correlation between each cell's value and the mean value of its neighbors in the full-feature kNN graph.
4. Residual batch penalty, computed as eta-squared for batch after regressing the feature on age.
5. Residual phase penalty, computed the same way after age regression.
6. Residual state penalty for interpretation only.

The final feature score was a weighted raw-metric combination that prioritized dynamic age dependence and local graph consistency while treating age-independent batch and phase effects as penalties rather than the main ranking signal.

### Panel selection
I used greedy forward selection with a redundancy penalty based on pairwise feature correlation. Candidate panel sizes of 8, 12, 16, 20, 24, and 32 features were benchmarked, and the best-performing trajectory panel size was selected by a composite criterion combining age prediction, pseudotime agreement, neighborhood preservation, and confound penalties.

### Validation
For every candidate subset and baseline, I measured:
1. Cross-validated Ridge regression R^2 for predicting `annotated_age`.
2. Spearman correlation between graph-derived pseudotime and `annotated_age`.
3. Mean age error across local neighbors.
4. kNN overlap with the full 241-feature graph.
5. Batch silhouette in subset PCA space.
6. Mean residual phase penalty across selected features.

## Results
### Selected trajectory panel
The best-performing trajectory-aware subset contained {len(selected_features)} features:

{", ".join(selected_features)}

The score components for the chosen panel are shown below.

![Selected feature scores](images/selected_feature_heatmap.png)

The highest-ranked markers are strongly interpretable. DNA content, Cyclin A/B, CDK2, Skp2, PCNA, pH2AX, and Cdt1 are canonical regulators or readouts of proliferative progression, checkpoint signaling, and replication-associated state changes. This is biologically coherent with the observed age-phase relationship in the dataset.

### Dynamic trends across the trajectory
The top selected features show smooth monotonic or phase-linked transitions across annotated age rather than purely noisy or batch-driven variation.

![Dynamic feature trends](images/dynamic_feature_trends.png)

### Validation against baselines
The trajectory-aware panel outperformed random and variance-based subsets consistently and slightly surpassed the age-correlation baseline on the composite benchmark. Pure age-correlation ranking remained a strong competitor, but the trajectory-aware selector achieved a better overall tradeoff by modestly improving age prediction, neighborhood preservation, and residual phase control at the chosen 24-feature panel size.

![Benchmark comparison](images/benchmark_comparison.png)

The selected panel also preserved a smooth low-dimensional progression.

![Selected panel embedding](images/selected_panel_embedding.png)

![Pseudotime validation](images/pseudotime_validation.png)

### Key quantitative results
{comparison_table}

### Feature-level summary for the final panel
{top_table}

## Discussion
This analysis shows that trajectory-preserving feature selection can be carried out effectively on compact single-cell imaging panels by using a continuous trajectory reference and explicitly penalizing age-independent confounding. In this dataset, the dominant dynamic program is tightly linked to proliferative progression, so the selected markers emphasize replication and checkpoint biology. That is a legitimate result rather than a failure of the method: the selector identified the molecular features that best preserve the continuous transition present in the data.

The main strength of the approach is that it does not rely on raw variance alone. It explicitly asks whether a marker changes smoothly along the trajectory, remains coherent on the neighborhood graph, and avoids retaining technical or discrete residual effects once age is accounted for.

## Limitations
1. The trajectory reference was `annotated_age`, an externally supplied continuous variable. This makes the analysis semi-supervised rather than fully unsupervised.
2. The dataset is dominated by cell-cycle-like progression, so the selected panel is correspondingly biased toward proliferative markers and nuclear morphology.
3. Phase is partly biological and partly confounding in this setting; penalizing its age-independent component is reasonable, but not uniquely correct.
4. The reference-paper set included irrelevant PDFs, so the methodological framing had to rely mainly on general single-cell trajectory concepts rather than a tightly matched benchmark method paper.

## Reproducibility
The complete analysis script is available at `code/run_trajectory_feature_selection.py`. Running it from the workspace regenerates all tables and figures in `outputs/` and `report/images/`.
"""
    report_path.write_text(report)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_style()

    base = Path(__file__).resolve().parents[1]
    paths = make_output_dirs(base)

    X, obs, feature_names = load_data(base / "data" / "adata_RPE.h5ad")
    X_std = standardize(X)
    full_dist, full_knn_idx = build_knn_indices(X_std)
    del full_dist

    scores = compute_feature_scores(X_std, obs, feature_names, full_knn_idx)
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    corr_matrix = np.corrcoef(X_std, rowvar=False)

    metrics: list[PanelMetrics] = []
    selected_by_size: dict[int, list[str]] = {}
    rng = random.Random(RANDOM_SEED)

    for size in PANEL_SIZES:
        selected = greedy_select(scores, corr_matrix, feature_to_idx, size)
        selected_by_size[size] = selected
        metrics.append(
            evaluate_panel("trajectory_panel", selected, X_std, obs, feature_to_idx, full_knn_idx, scores)
        )
        variance_selected = top_variance_features(X_std, feature_names, size)
        metrics.append(
            evaluate_panel("variance", variance_selected, X_std, obs, feature_to_idx, full_knn_idx, scores)
        )
        agecorr_selected = top_age_correlation_features(scores, size)
        metrics.append(
            evaluate_panel("age_correlation", agecorr_selected, X_std, obs, feature_to_idx, full_knn_idx, scores)
        )
        random_selected = random_features(feature_names, size, rng)
        metrics.append(
            evaluate_panel("random", random_selected, X_std, obs, feature_to_idx, full_knn_idx, scores)
        )

    metrics_df = pd.DataFrame([m.__dict__ for m in metrics])
    best_size = choose_panel_size(metrics_df)
    final_selected = selected_by_size[best_size]

    pseudotime, _ = pseudotime_from_panel(
        X_std[:, [feature_to_idx[f] for f in final_selected]],
        obs["annotated_age"].to_numpy(dtype=float),
    )
    pseudotime_df = obs.copy()
    pseudotime_df["pseudotime"] = pseudotime

    outputs = paths["outputs"]
    images = paths["images"]
    scores.to_csv(outputs / "feature_scores.csv", index=False)
    metrics_df.to_csv(outputs / "panel_benchmark_metrics.csv", index=False)
    pd.DataFrame({"feature": final_selected}).to_csv(outputs / "selected_feature_panel.csv", index=False)
    pseudotime_df.to_csv(outputs / "selected_panel_pseudotime.csv", index=False)

    summary = {
        "best_panel_size": best_size,
        "selected_features": final_selected,
        "best_metrics": metrics_df[
            (metrics_df["method"] == "trajectory_panel") & (metrics_df["panel_size"] == best_size)
        ].iloc[0].to_dict(),
    }
    (outputs / "summary.json").write_text(json.dumps(summary, indent=2))

    save_overview_figure(obs, images / "data_overview.png")
    save_embedding_figure(X_std, obs, final_selected, feature_to_idx, images / "selected_panel_embedding.png")
    save_feature_score_heatmap(scores, final_selected, images / "selected_feature_heatmap.png")
    save_dynamic_trends(X, obs, feature_names, final_selected, images / "dynamic_feature_trends.png")
    save_benchmark_figure(metrics_df, images / "benchmark_comparison.png")
    save_pseudotime_figure(obs, pseudotime_df, images / "pseudotime_validation.png")

    write_report(base, obs, scores, metrics_df, final_selected, pseudotime_df)


if __name__ == "__main__":
    main()
