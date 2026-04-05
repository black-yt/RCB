import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 0
sns.set_theme(style="whitegrid", context="talk")
np.random.seed(SEED)
sc.settings.verbosity = 0


def ensure_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def clean_feature_name(name: str) -> str:
    for prefix in ["Int_MeanEdge_", "Int_MeanIntensity_", "Int_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ["_cell", "_nuc", "_cyto"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def prepare_adata(adata):
    adata = adata.copy()
    adata.var_names_make_unique()
    adata.obs = adata.obs.copy()
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    adata.obs["phase"] = adata.obs["phase"].astype(str)
    adata.obs["state"] = adata.obs["state"].astype(str)
    adata.obs["state_clean"] = adata.obs["state"].replace({"nan": "unknown"})
    adata.obs["annotated_age"] = pd.to_numeric(adata.obs["annotated_age"], errors="coerce")
    mask = adata.obs["annotated_age"].notna().to_numpy()
    adata = adata[mask].copy()
    X = ensure_dense(adata.X).astype(float)
    scaler = StandardScaler()
    adata.X = scaler.fit_transform(X)
    adata.var["clean_name"] = [clean_feature_name(v) for v in adata.var_names]
    return adata


def compute_full_embedding(adata):
    work = adata.copy()
    sc.pp.pca(work, n_comps=min(30, work.n_vars - 1), random_state=SEED)
    sc.pp.neighbors(work, n_neighbors=20, n_pcs=min(20, work.obsm["X_pca"].shape[1]), random_state=SEED)
    sc.tl.umap(work, random_state=SEED)
    return work


def graph_distances(connectivities):
    G = nx.from_scipy_sparse_array(connectivities)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = connectivities.shape[0]
    dist = np.full((n, n), np.inf)
    for i, d in lengths.items():
        for j, v in d.items():
            dist[i, j] = v
    return dist


def derive_reference_pseudotime(work):
    ages = work.obs["annotated_age"].to_numpy()
    root = int(np.argmin(ages))
    dist = graph_distances(work.obsp["connectivities"])
    if np.isinf(dist[root]).all():
        pseudotime = stats.rankdata(ages)
    else:
        pseudo = dist[root].copy()
        finite = np.isfinite(pseudo)
        if finite.sum() < len(pseudo):
            pseudo[~finite] = np.nanmax(pseudo[finite]) + 1
        pseudotime = pseudo
    pseudotime = (pseudotime - np.min(pseudotime)) / (np.max(pseudotime) - np.min(pseudotime) + 1e-8)
    return pseudotime


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / max(n, 1)
    r, k = tab.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def score_features(adata, pseudotime):
    X = ensure_dense(adata.X)
    ages = adata.obs["annotated_age"].to_numpy()
    batch = adata.obs["batch"].to_numpy()
    phase = adata.obs["phase"].to_numpy()
    state = adata.obs["state_clean"].to_numpy()
    mi = mutual_info_regression(X, pseudotime, random_state=SEED)
    rows = []
    for i, gene in enumerate(adata.var_names):
        vec = X[:, i]
        sp_age = abs(stats.spearmanr(vec, ages).statistic)
        sp_pseudo = abs(stats.spearmanr(vec, pseudotime).statistic)
        f_batch = stats.f_oneway(*[vec[batch == b] for b in np.unique(batch)]).statistic if len(np.unique(batch)) > 1 else 0.0
        f_phase = stats.f_oneway(*[vec[phase == p] for p in np.unique(phase)]).statistic if len(np.unique(phase)) > 1 else 0.0
        f_state = stats.f_oneway(*[vec[state == s] for s in np.unique(state)]).statistic if len(np.unique(state)) > 1 else 0.0
        score_dynamic = 0.45 * sp_pseudo + 0.25 * sp_age + 0.15 * mi[i] - 0.10 * np.log1p(f_batch) - 0.05 * cramers_v(pd.qcut(vec, q=4, duplicates='drop'), state)
        rows.append({
            "feature": gene,
            "clean_name": adata.var.loc[gene, "clean_name"],
            "variance": float(np.var(vec)),
            "spearman_age": float(sp_age),
            "spearman_pseudotime": float(sp_pseudo),
            "mutual_info_pseudotime": float(mi[i]),
            "f_batch": float(f_batch),
            "f_phase": float(f_phase),
            "f_state": float(f_state),
            "dynamic_score": float(score_dynamic),
        })
    df = pd.DataFrame(rows)
    df["variance_rank"] = df["variance"].rank(ascending=False, method="min")
    df["dynamic_rank"] = df["dynamic_score"].rank(ascending=False, method="min")
    return df.sort_values("dynamic_score", ascending=False)


def get_subsets(scores, sizes=(10, 20, 30, 50)):
    subsets = {}
    for k in sizes:
        subsets[f"dynamic_top_{k}"] = scores.nlargest(k, "dynamic_score")["feature"].tolist()
        subsets[f"variance_top_{k}"] = scores.nlargest(k, "variance")["feature"].tolist()
    subsets["full"] = scores["feature"].tolist()
    return subsets


def batch_entropy(labels):
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def evaluate_subset(adata, subset_features, full_umap, ref_pseudotime):
    work = adata[:, subset_features].copy()
    n_comps = max(2, min(20, work.n_vars - 1)) if work.n_vars > 2 else 2
    sc.pp.pca(work, n_comps=n_comps, random_state=SEED)
    sc.pp.neighbors(work, n_neighbors=20, n_pcs=min(10, work.obsm["X_pca"].shape[1]), random_state=SEED)
    sc.tl.umap(work, random_state=SEED)

    ages = work.obs["annotated_age"].to_numpy()
    batches = work.obs["batch"].to_numpy()
    phases = work.obs["phase"].to_numpy()

    nbrs = NearestNeighbors(n_neighbors=16).fit(work.obsm["X_umap"])
    inds = nbrs.kneighbors(return_distance=False)
    neighbor_ages = np.array([ages[idx[1:]].mean() for idx in inds])
    age_mse = mean_squared_error(ages, neighbor_ages)
    age_spearman = stats.spearmanr(ages, neighbor_ages).statistic
    pt_spearman = stats.spearmanr(ref_pseudotime, work.obsm["X_umap"][:, 0]).statistic
    full_corr = np.corrcoef(full_umap[:, 0], work.obsm["X_umap"][:, 0])[0, 1]

    batch_entropies = [batch_entropy(batches[idx[1:]]) for idx in inds]
    mean_batch_entropy = float(np.mean(batch_entropies))
    batch_sil = silhouette_score(work.obsm["X_umap"], batches) if len(np.unique(batches)) > 1 else np.nan
    phase_sil = silhouette_score(work.obsm["X_umap"], phases) if len(np.unique(phases)) > 1 else np.nan

    return {
        "n_features": len(subset_features),
        "age_neighbor_mse": float(age_mse),
        "age_neighbor_spearman": float(age_spearman),
        "pseudotime_axis_spearman": float(pt_spearman),
        "umap_axis0_corr_full": float(full_corr),
        "mean_batch_entropy": mean_batch_entropy,
        "batch_silhouette": float(batch_sil),
        "phase_silhouette": float(phase_sil),
        "umap": work.obsm["X_umap"].copy(),
    }


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_plots(adata, full_work, scores, eval_df, subset_umaps, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    ages = adata.obs["annotated_age"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    sns.histplot(ages, bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Annotated age distribution")
    axes[0].set_xlabel("Annotated age")
    sns.countplot(data=adata.obs, x="phase", order=sorted(adata.obs['phase'].unique()), ax=axes[1], color="#55A868")
    axes[1].set_title("Cell-cycle phase composition")
    axes[1].tick_params(axis='x', rotation=45)
    sns.countplot(data=adata.obs, x="state_clean", order=adata.obs['state_clean'].value_counts().index, ax=axes[2], color="#C44E52")
    axes[2].set_title("Cell state composition")
    axes[2].tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(report_dir / "data_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sca = ax.scatter(full_work.obsm["X_umap"][:, 0], full_work.obsm["X_umap"][:, 1], c=ages, cmap="viridis", s=18)
    ax.set_title("Full-feature UMAP colored by annotated age")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.colorbar(sca, ax=ax, label="Annotated age")
    fig.tight_layout()
    fig.savefig(report_dir / "full_umap_age.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_scores = scores.head(20).copy()
    top_scores = top_scores.sort_values("dynamic_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top_scores["clean_name"], top_scores["dynamic_score"], color="#4C72B0")
    ax.set_title("Top dynamic trajectory-preserving features")
    ax.set_xlabel("Dynamic score")
    fig.tight_layout()
    fig.savefig(report_dir / "top_dynamic_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_df = eval_df.melt(id_vars=["method", "n_features"], value_vars=["age_neighbor_spearman", "pseudotime_axis_spearman", "mean_batch_entropy"], var_name="metric", value_name="value")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_titles = {
        "age_neighbor_spearman": "Neighbor age concordance",
        "pseudotime_axis_spearman": "Pseudotime alignment",
        "mean_batch_entropy": "Batch mixing entropy",
    }
    for ax, metric in zip(axes, metric_titles):
        sub = plot_df[plot_df["metric"] == metric]
        sns.lineplot(data=sub, x="n_features", y="value", hue="method", marker="o", ax=ax)
        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Selected features")
        ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(report_dir / "method_comparison_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    chosen = [k for k in ["dynamic_top_20", "variance_top_20"] if k in subset_umaps]
    fig, axes = plt.subplots(1, len(chosen), figsize=(7 * len(chosen), 6))
    if len(chosen) == 1:
        axes = [axes]
    for ax, key in zip(axes, chosen):
        um = subset_umaps[key]
        sca = ax.scatter(um[:, 0], um[:, 1], c=ages, cmap="viridis", s=18)
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.colorbar(sca, ax=axes, label="Annotated age")
    fig.tight_layout()
    fig.savefig(report_dir / "subset_umap_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=scores, x="spearman_pseudotime", y="variance", alpha=0.6, s=40, ax=ax)
    top = scores.head(10)
    for _, row in top.iterrows():
        ax.text(row["spearman_pseudotime"], row["variance"], row["clean_name"], fontsize=9)
    ax.set_title("Dynamic association versus variance")
    fig.tight_layout()
    fig.savefig(report_dir / "score_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reportdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    report_dir = Path(args.reportdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input)
    adata = prepare_adata(adata)
    full_work = compute_full_embedding(adata)
    ref_pseudotime = derive_reference_pseudotime(full_work)
    scores = score_features(adata, ref_pseudotime)
    subsets = get_subsets(scores)

    results = []
    subset_umaps = {}
    for name, feats in subsets.items():
        metrics = evaluate_subset(adata, feats, full_work.obsm["X_umap"], ref_pseudotime)
        subset_umaps[name] = metrics.pop("umap")
        method = "full" if name == "full" else name.split("_top_")[0]
        metrics["subset"] = name
        metrics["method"] = method
        results.append(metrics)

    eval_df = pd.DataFrame(results).sort_values(["n_features", "method"])
    selected_summary = scores.head(20)[["feature", "clean_name", "dynamic_score", "spearman_age", "spearman_pseudotime", "f_batch", "f_phase", "f_state"]]

    data_summary = {
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "annotated_age_min": float(adata.obs["annotated_age"].min()),
        "annotated_age_max": float(adata.obs["annotated_age"].max()),
        "phase_counts": adata.obs["phase"].value_counts().to_dict(),
        "state_counts": adata.obs["state_clean"].value_counts().to_dict(),
        "batch_counts": adata.obs["batch"].value_counts().to_dict(),
    }

    scores.to_csv(outdir / "feature_scores.csv", index=False)
    selected_summary.to_csv(outdir / "selected_feature_summary.csv", index=False)
    eval_df.to_csv(outdir / "evaluation_metrics.csv", index=False)
    save_json(outdir / "feature_sets.json", subsets)
    save_json(outdir / "data_summary.json", data_summary)
    np.save(outdir / "full_umap.npy", full_work.obsm["X_umap"])
    np.save(outdir / "reference_pseudotime.npy", ref_pseudotime)
    for key, val in subset_umaps.items():
        np.save(outdir / f"{key}_umap.npy", val)

    make_plots(adata, full_work, scores, eval_df, subset_umaps, report_dir)


if __name__ == "__main__":
    main()
