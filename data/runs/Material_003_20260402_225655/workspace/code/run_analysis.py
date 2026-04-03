#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"
CODE_DIR = ROOT / "code"


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    bias: float
    pearson_r: float


COMMON_ATOMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si"]
TARGETS = [300, 350, 400, 450]


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    calib = pd.read_csv(DATA_DIR / "tg_calibration.csv")
    vitrimer = pd.read_csv(DATA_DIR / "tg_vitrimer_MD.csv")
    return calib, vitrimer


def smiles_stats(smiles: str) -> Dict[str, float]:
    s = str(smiles)
    stats = {
        "length": len(s),
        "ring_digits": sum(ch.isdigit() for ch in s),
        "branches": s.count("(") + s.count(")"),
        "double_bonds": s.count("="),
        "triple_bonds": s.count("#"),
        "aromatic_lower": sum(ch in "cnosp" for ch in s),
        "halogens": s.count("Cl") + s.count("Br") + s.count("F") + s.count("I"),
        "hetero_total": 0,
    }
    hetero = 0
    for atom in COMMON_ATOMS:
        count = s.count(atom)
        stats[f"atom_{atom}"] = count
        if atom != "C":
            hetero += count
    stats["hetero_total"] = hetero
    return stats


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(math.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        bias=float(np.mean(y_pred - y_true)),
        pearson_r=float(np.corrcoef(y_true, y_pred)[0, 1]),
    )


def build_calibration_features(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.DataFrame([smiles_stats(s) for s in df["smiles"]])
    out = pd.concat([df[["tg_md", "std"]].reset_index(drop=True), stats_df], axis=1)
    return out


def fit_calibration_model(calib: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, Metrics]:
    features = build_calibration_features(calib)
    y = calib["tg_exp"].to_numpy()

    numeric_cols = list(features.columns)
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            )
        ]
    )
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(1e-6, 1e3)
    )
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED, n_restarts_optimizer=3)
    pipe = Pipeline([("pre", pre), ("gpr", model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_pred = cross_val_predict(pipe, features, y, cv=cv, method="predict", n_jobs=None)
    metrics = compute_metrics(y, cv_pred)

    pipe.fit(features, y)
    pred_mean, pred_std = pipe.predict(features, return_std=True)

    calib_results = calib.copy()
    calib_results["cv_pred_tg_exp"] = cv_pred
    calib_results["fit_pred_tg_exp"] = pred_mean
    calib_results["fit_pred_std"] = pred_std
    calib_results["cv_residual"] = calib_results["cv_pred_tg_exp"] - calib_results["tg_exp"]
    return pipe, calib_results, metrics


def plot_calibration_overview(calib_results: pd.DataFrame, metrics: Metrics) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    sns.scatterplot(
        data=calib_results,
        x="tg_exp",
        y="cv_pred_tg_exp",
        hue="std",
        palette="viridis",
        ax=ax,
        s=70,
        edgecolor="none",
    )
    lims = [
        min(calib_results["tg_exp"].min(), calib_results["cv_pred_tg_exp"].min()) - 10,
        max(calib_results["tg_exp"].max(), calib_results["cv_pred_tg_exp"].max()) + 10,
    ]
    ax.plot(lims, lims, "k--", lw=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Gaussian-process Tg calibration")
    ax.set_xlabel("Experimental Tg (K)")
    ax.set_ylabel("Cross-validated predicted Tg (K)")
    txt = f"MAE={metrics.mae:.1f} K\nRMSE={metrics.rmse:.1f} K\nR²={metrics.r2:.2f}\nr={metrics.pearson_r:.2f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", bbox=dict(fc="white", ec="0.8"))

    ax = axes[1]
    sns.scatterplot(data=calib_results, x="tg_md", y="cv_residual", color="#d95f02", ax=ax, s=65)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title("Calibration residuals vs MD Tg")
    ax.set_xlabel("MD-simulated Tg (K)")
    ax.set_ylabel("Predicted - experimental Tg (K)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_performance.png", bbox_inches="tight")
    plt.close(fig)


def summarize_data(calib: pd.DataFrame, vitrimer: pd.DataFrame) -> Dict[str, object]:
    acid_unique = vitrimer["acid"].nunique()
    epoxide_unique = vitrimer["epoxide"].nunique()
    out = {
        "calibration_rows": int(len(calib)),
        "vitrimer_rows": int(len(vitrimer)),
        "calibration_tg_exp_range": [float(calib["tg_exp"].min()), float(calib["tg_exp"].max())],
        "calibration_tg_md_range": [float(calib["tg_md"].min()), float(calib["tg_md"].max())],
        "vitrimer_tg_md_range": [float(vitrimer["tg"].min()), float(vitrimer["tg"].max())],
        "unique_acids": int(acid_unique),
        "unique_epoxides": int(epoxide_unique),
        "combinatorial_density": float(len(vitrimer) / max(acid_unique * epoxide_unique, 1)),
    }
    return out


def plot_data_overview(calib: pd.DataFrame, vitrimer: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.histplot(calib["tg_exp"], bins=25, kde=True, color="#1b9e77", ax=axes[0, 0])
    axes[0, 0].set_title("Experimental Tg distribution (calibration set)")
    axes[0, 0].set_xlabel("Experimental Tg (K)")

    sns.histplot(calib["tg_md"], bins=25, kde=True, color="#7570b3", ax=axes[0, 1])
    axes[0, 1].set_title("MD Tg distribution (calibration set)")
    axes[0, 1].set_xlabel("MD Tg (K)")

    sns.scatterplot(data=calib, x="tg_md", y="tg_exp", hue="std", palette="magma", ax=axes[1, 0], s=60, edgecolor="none")
    axes[1, 0].set_title("Calibration dataset: MD vs experimental Tg")
    axes[1, 0].set_xlabel("MD Tg (K)")
    axes[1, 0].set_ylabel("Experimental Tg (K)")

    sns.histplot(vitrimer["tg"], bins=30, kde=True, color="#e7298a", ax=axes[1, 1])
    axes[1, 1].set_title("MD Tg distribution (vitrimer design space)")
    axes[1, 1].set_xlabel("MD Tg (K)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_overview.png", bbox_inches="tight")
    plt.close(fig)


def apply_calibration(pipe: Pipeline, vitrimer: pd.DataFrame) -> pd.DataFrame:
    pseudo_polymer = (vitrimer["acid"].astype(str) + "." + vitrimer["epoxide"].astype(str)).rename("smiles")
    proxy = pd.DataFrame({
        "smiles": pseudo_polymer,
        "tg_md": vitrimer["tg"].astype(float),
        "std": vitrimer["std"].astype(float),
    })
    features = build_calibration_features(proxy)
    mean_pred, std_pred = pipe.predict(features, return_std=True)

    out = vitrimer.copy()
    out["calibrated_tg"] = mean_pred
    out["calibration_uncertainty"] = std_pred
    out["acidity_stats_length"] = out["acid"].astype(str).str.len()
    out["epoxide_stats_length"] = out["epoxide"].astype(str).str.len()
    return out


def build_latent_space(vitrimer_scored: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    combo_text = (vitrimer_scored["acid"].astype(str) + " | " + vitrimer_scored["epoxide"].astype(str)).tolist()
    vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    X = vect.fit_transform(combo_text)

    svd_dim = min(16, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=SEED)
    X_svd = svd.fit_transform(X)

    aux = vitrimer_scored[["tg", "std", "calibrated_tg", "calibration_uncertainty"]].to_numpy()
    aux = StandardScaler().fit_transform(aux)
    latent_input = np.hstack([X_svd, aux])

    pca = PCA(n_components=2, random_state=SEED)
    embedding_2d = pca.fit_transform(latent_input)

    gmm = GaussianMixture(n_components=8, covariance_type="full", random_state=SEED)
    clusters = gmm.fit_predict(latent_input)
    log_density = gmm.score_samples(latent_input)

    latent_df = vitrimer_scored.copy()
    latent_df["latent_1"] = embedding_2d[:, 0]
    latent_df["latent_2"] = embedding_2d[:, 1]
    latent_df["cluster"] = clusters
    latent_df["latent_density"] = log_density

    meta = {
        "tfidf_vocab_size": int(len(vect.vocabulary_)),
        "svd_dim": int(svd_dim),
        "explained_variance_ratio_svd": [float(x) for x in svd.explained_variance_ratio_[:10]],
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "gmm_components": 8,
    }
    return latent_df, meta


def plot_latent_space(latent_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(
        data=latent_df.sample(min(3000, len(latent_df)), random_state=SEED),
        x="latent_1",
        y="latent_2",
        hue="calibrated_tg",
        palette="viridis",
        s=45,
        edgecolor="none",
        ax=axes[0],
    )
    axes[0].set_title("Latent design space colored by calibrated Tg")

    sns.scatterplot(
        data=latent_df.sample(min(3000, len(latent_df)), random_state=SEED),
        x="latent_1",
        y="latent_2",
        hue="cluster",
        palette="tab10",
        s=45,
        edgecolor="none",
        ax=axes[1],
    )
    axes[1].set_title("Latent design space clusters")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "latent_design_space.png", bbox_inches="tight")
    plt.close(fig)


def recommend_candidates(latent_df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    latent_coords = scaler.fit_transform(latent_df[["latent_1", "latent_2"]])
    nn = NearestNeighbors(n_neighbors=min(25, len(latent_df)))
    nn.fit(latent_coords)

    rows: List[pd.DataFrame] = []
    for target in TARGETS:
        temp = latent_df.copy()
        temp["target_tg"] = target
        temp["tg_distance"] = (temp["calibrated_tg"] - target).abs()
        temp["uncertainty_penalty"] = temp["calibration_uncertainty"]
        temp["density_bonus"] = -temp["latent_density"]
        temp["score"] = temp["tg_distance"] + 0.30 * temp["uncertainty_penalty"] + 0.15 * temp["density_bonus"]
        temp = temp.sort_values(["score", "tg_distance", "calibration_uncertainty"]).copy()

        selected_idx: List[int] = []
        used_clusters = set()
        for idx, row in temp.iterrows():
            if row["cluster"] in used_clusters and len(selected_idx) < 3:
                continue
            selected_idx.append(idx)
            used_clusters.add(row["cluster"])
            if len(selected_idx) >= 5:
                break
        if len(selected_idx) < 5:
            for idx in temp.index:
                if idx not in selected_idx:
                    selected_idx.append(idx)
                if len(selected_idx) >= 5:
                    break

        block = temp.loc[selected_idx].copy()
        block["rank_within_target"] = range(1, len(block) + 1)
        rows.append(block)

    recommended = pd.concat(rows, ignore_index=True)
    recommended = recommended[
        [
            "target_tg",
            "rank_within_target",
            "acid",
            "epoxide",
            "tg",
            "std",
            "calibrated_tg",
            "calibration_uncertainty",
            "cluster",
            "latent_1",
            "latent_2",
            "score",
        ]
    ]
    return recommended.sort_values(["target_tg", "rank_within_target"]).reset_index(drop=True)


def plot_candidate_map(latent_df: pd.DataFrame, recommended: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    base = latent_df.sample(min(3500, len(latent_df)), random_state=SEED)
    sns.scatterplot(
        data=base,
        x="latent_1",
        y="latent_2",
        hue="calibrated_tg",
        palette="coolwarm",
        s=35,
        edgecolor="none",
        alpha=0.45,
        ax=ax,
        legend=False,
    )
    sns.scatterplot(
        data=recommended,
        x="latent_1",
        y="latent_2",
        style="target_tg",
        hue="target_tg",
        palette="Set1",
        s=220,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )
    for _, row in recommended.iterrows():
        ax.text(row["latent_1"] + 0.03, row["latent_2"] + 0.03, f"{int(row['target_tg'])}-{int(row['rank_within_target'])}", fontsize=9)
    ax.set_title("Inverse-design candidate selections in latent space")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "candidate_map.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibrated_distribution(vitrimer_scored: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(vitrimer_scored["tg"], label="MD Tg", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(vitrimer_scored["calibrated_tg"], label="Calibrated Tg", fill=True, alpha=0.35, ax=ax)
    for target in TARGETS:
        ax.axvline(target, color="k", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title("Vitrimer design-space Tg distribution before and after calibration")
    ax.set_xlabel("Tg (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibrated_distribution.png", bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    data_summary: Dict[str, object],
    calibration_metrics: Metrics,
    latent_meta: Dict[str, object],
    calib_results: pd.DataFrame,
    vitrimer_scored: pd.DataFrame,
    latent_df: pd.DataFrame,
    recommended: pd.DataFrame,
) -> None:
    (OUTPUT_DIR / "data_summary.json").write_text(json.dumps(data_summary, indent=2))
    (OUTPUT_DIR / "calibration_metrics.json").write_text(json.dumps(asdict(calibration_metrics), indent=2))
    (OUTPUT_DIR / "latent_model_summary.json").write_text(json.dumps(latent_meta, indent=2))

    calib_results.to_csv(OUTPUT_DIR / "calibration_predictions.csv", index=False)
    vitrimer_scored.sort_values("calibrated_tg", ascending=False).to_csv(OUTPUT_DIR / "vitrimer_calibrated_predictions.csv", index=False)
    latent_df.to_csv(OUTPUT_DIR / "vitrimer_latent_space.csv", index=False)
    recommended.to_csv(OUTPUT_DIR / "inverse_design_candidates.csv", index=False)

    top_summary = recommended.groupby("target_tg").first().reset_index()[
        ["target_tg", "acid", "epoxide", "calibrated_tg", "calibration_uncertainty"]
    ]
    top_summary.to_csv(OUTPUT_DIR / "candidate_summary_by_target.csv", index=False)


def build_research_summary(
    data_summary: Dict[str, object],
    metrics: Metrics,
    recommended: pd.DataFrame,
) -> None:
    lines = []
    lines.append("AI-guided inverse-design analysis summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Calibration dataset: {data_summary['calibration_rows']} polymers")
    lines.append(f"Vitrimer design space: {data_summary['vitrimer_rows']} acid/epoxide combinations")
    lines.append(f"Unique acids: {data_summary['unique_acids']}; unique epoxides: {data_summary['unique_epoxides']}")
    lines.append("")
    lines.append("Gaussian-process calibration performance (5-fold CV):")
    lines.append(f"- MAE: {metrics.mae:.2f} K")
    lines.append(f"- RMSE: {metrics.rmse:.2f} K")
    lines.append(f"- R^2: {metrics.r2:.3f}")
    lines.append(f"- Pearson r: {metrics.pearson_r:.3f}")
    lines.append("")
    lines.append("Top-ranked candidates by target calibrated Tg:")
    for target in TARGETS:
        subset = recommended[recommended["target_tg"] == target].sort_values("rank_within_target").head(1)
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- Target {target} K: predicted {row['calibrated_tg']:.1f} ± {row['calibration_uncertainty']:.1f} K | acid={row['acid']} | epoxide={row['epoxide']}"
        )
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    calib, vitrimer = load_data()
    data_summary = summarize_data(calib, vitrimer)
    plot_data_overview(calib, vitrimer)

    pipe, calib_results, calibration_metrics = fit_calibration_model(calib)
    plot_calibration_overview(calib_results, calibration_metrics)

    vitrimer_scored = apply_calibration(pipe, vitrimer)
    plot_calibrated_distribution(vitrimer_scored)

    latent_df, latent_meta = build_latent_space(vitrimer_scored)
    plot_latent_space(latent_df)

    recommended = recommend_candidates(latent_df)
    plot_candidate_map(latent_df, recommended)

    write_outputs(data_summary, calibration_metrics, latent_meta, calib_results, vitrimer_scored, latent_df, recommended)
    build_research_summary(data_summary, calibration_metrics, recommended)
    print("Analysis complete. Outputs written to outputs/ and report/images/.")


if __name__ == "__main__":
    main()
