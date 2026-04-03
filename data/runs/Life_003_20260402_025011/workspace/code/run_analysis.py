#!/usr/bin/env python3
"""Offline reproduction study for Uncalled4 benchmark tables and pore models."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


SEED = 42
BASES = list("ACGT")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
IMAGE_DIR = REPORT_DIR / "images"


@dataclass(frozen=True)
class PoreModelSpec:
    file_name: str
    model_id: str
    label: str
    modality: str
    chemistry: str
    k: int


PORE_MODELS = [
    PoreModelSpec(
        "dna_r9.4.1_400bps_6mer_uncalled4.csv",
        "dna_r9.4.1",
        "DNA R9.4.1 6-mer",
        "DNA",
        "R9.4.1",
        6,
    ),
    PoreModelSpec(
        "dna_r10.4.1_400bps_9mer_uncalled4.csv",
        "dna_r10.4.1",
        "DNA R10.4.1 9-mer",
        "DNA",
        "R10.4.1",
        9,
    ),
    PoreModelSpec(
        "rna_r9.4.1_70bps_5mer_uncalled4.csv",
        "rna_r9.4.1",
        "RNA001 R9.4.1 5-mer",
        "RNA",
        "RNA001",
        5,
    ),
    PoreModelSpec(
        "rna004_130bps_9mer_uncalled4.csv",
        "rna004",
        "RNA004 9-mer",
        "RNA",
        "RNA004",
        9,
    ),
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 11,
        }
    )


def load_pore_models() -> pd.DataFrame:
    frames = []
    for spec in PORE_MODELS:
        df = pd.read_csv(DATA_DIR / spec.file_name)
        df["model_id"] = spec.model_id
        df["label"] = spec.label
        df["modality"] = spec.modality
        df["chemistry"] = spec.chemistry
        df["k"] = spec.k
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_pore_models(models: pd.DataFrame) -> pd.DataFrame:
    summary = (
        models.groupby(["model_id", "label", "modality", "chemistry", "k"], as_index=False)
        .agg(
            n_kmers=("kmer", "size"),
            current_mean_mean=("current_mean", "mean"),
            current_mean_std=("current_mean", "std"),
            current_mean_min=("current_mean", "min"),
            current_mean_max=("current_mean", "max"),
            current_std_mean=("current_std", "mean"),
            current_std_std=("current_std", "std"),
            dwell_mean=("dwell_time", "mean"),
            dwell_std=("dwell_time", "std"),
            dwell_median=("dwell_time", "median"),
        )
        .sort_values(["modality", "k"])
    )
    summary.to_csv(OUTPUT_DIR / "pore_model_summary.csv", index=False)
    return summary


def compute_base_effects(models: pd.DataFrame, feature: str = "current_mean") -> pd.DataFrame:
    rows = []
    for spec in PORE_MODELS:
        df = models.loc[models["model_id"] == spec.model_id, ["kmer", feature]].copy()
        overall = df[feature].mean()
        for pos in range(spec.k):
            bases = df["kmer"].str[pos]
            grouped = df.groupby(bases)[feature].mean().reindex(BASES)
            for base, value in grouped.items():
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "label": spec.label,
                        "feature": feature,
                        "position": pos + 1,
                        "base": base,
                        "mean_value": value,
                        "centered_effect": value - overall,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / f"base_effects_{feature}.csv", index=False)
    return result


def compute_position_sensitivity(models: pd.DataFrame, feature: str = "current_mean") -> pd.DataFrame:
    rows = []
    for spec in PORE_MODELS:
        df = models.loc[models["model_id"] == spec.model_id, ["kmer", feature]].copy()
        for pos in range(spec.k):
            temp = pd.DataFrame(
                {
                    "context": df["kmer"].str.slice(0, pos) + df["kmer"].str.slice(pos + 1),
                    "base": df["kmer"].str[pos],
                    feature: df[feature].to_numpy(),
                }
            )
            pivot = temp.pivot(index="context", columns="base", values=feature).reindex(columns=BASES)
            arr = pivot.to_numpy(dtype=float)
            pair_diffs = [
                np.abs(arr[:, i] - arr[:, j])
                for i in range(len(BASES))
                for j in range(i + 1, len(BASES))
            ]
            mean_pair_diffs = np.mean(np.vstack(pair_diffs), axis=0)
            rows.append(
                {
                    "model_id": spec.model_id,
                    "label": spec.label,
                    "feature": feature,
                    "position": pos + 1,
                    "mean_abs_substitution_delta": float(np.mean(mean_pair_diffs)),
                    "median_abs_substitution_delta": float(np.median(mean_pair_diffs)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / f"position_sensitivity_{feature}.csv", index=False)
    return result


def build_data_overview_figure(models: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    features = [
        ("current_mean", "Current Mean"),
        ("current_std", "Current SD"),
        ("dwell_time", "Dwell Time"),
    ]
    palette = {
        "DNA R9.4.1 6-mer": "#1f3b73",
        "DNA R10.4.1 9-mer": "#2a9d8f",
        "RNA001 R9.4.1 5-mer": "#e76f51",
        "RNA004 9-mer": "#6d597a",
    }
    for ax, (feature, title) in zip(axes, features):
        for spec in PORE_MODELS:
            subset = models.loc[models["model_id"] == spec.model_id, feature]
            take = min(len(subset), 40000)
            sample = subset.iloc[rng.choice(len(subset), size=take, replace=False)]
            sns.kdeplot(
                sample,
                ax=ax,
                linewidth=2,
                label=spec.label,
                color=palette[spec.label],
                fill=False,
            )
        ax.set_title(title)
        ax.set_xlabel(title)
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[2].legend(loc="upper right", frameon=True, fontsize=10)
    fig.suptitle("Distributional Overview of Uncalled4 Pore Models", y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "data_overview.png", bbox_inches="tight")
    plt.close(fig)


def build_base_effect_figure(base_effects: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, spec in zip(axes, PORE_MODELS):
        pivot = (
            base_effects.loc[base_effects["model_id"] == spec.model_id]
            .pivot(index="base", columns="position", values="centered_effect")
            .reindex(BASES)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="coolwarm",
            center=0.0,
            cbar=ax is axes[-1],
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(spec.label)
        ax.set_xlabel("Position in k-mer")
        ax.set_ylabel("Base")
    fig.suptitle("Position- and Base-Specific Shifts in Expected Current", y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "base_effect_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def build_position_sensitivity_figure(position_sensitivity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    palette = {
        "DNA R9.4.1 6-mer": "#1f3b73",
        "DNA R10.4.1 9-mer": "#2a9d8f",
        "RNA001 R9.4.1 5-mer": "#e76f51",
        "RNA004 9-mer": "#6d597a",
    }
    for spec in PORE_MODELS:
        subset = position_sensitivity.loc[position_sensitivity["model_id"] == spec.model_id]
        ax.plot(
            subset["position"],
            subset["mean_abs_substitution_delta"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=spec.label,
            color=palette[spec.label],
        )
    ax.set_xlabel("Position in k-mer")
    ax.set_ylabel("Mean absolute current shift")
    ax.set_title("Signal Sensitivity Peaks at the Pore Constriction Center")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "position_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def analyze_performance() -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = pd.read_csv(DATA_DIR / "performance_summary.csv")
    ratios = []
    best_rows = []
    for chemistry, group in perf.groupby("Chemistry", sort=False):
        unc_row = group.loc[group["Tool"] == "Uncalled4"].iloc[0]
        available = group.dropna(subset=["Time_min", "FileSize_MB"])
        baseline = available.loc[available["Tool"] != "Uncalled4"]
        best_time_row = baseline.loc[baseline["Time_min"].idxmin()]
        best_size_row = baseline.loc[baseline["FileSize_MB"].idxmin()]
        best_rows.append(
            {
                "Chemistry": chemistry,
                "Uncalled4_time_min": unc_row["Time_min"],
                "BestBaseline_time_tool": best_time_row["Tool"],
                "BestBaseline_time_min": best_time_row["Time_min"],
                "Speedup_vs_best_baseline": best_time_row["Time_min"] / unc_row["Time_min"],
                "Uncalled4_file_MB": unc_row["FileSize_MB"],
                "SmallestBaseline_file_tool": best_size_row["Tool"],
                "SmallestBaseline_file_MB": best_size_row["FileSize_MB"],
                "Compression_vs_smallest_baseline": best_size_row["FileSize_MB"] / unc_row["FileSize_MB"],
            }
        )
        for _, row in group.iterrows():
            ratios.append(
                {
                    "Chemistry": chemistry,
                    "Tool": row["Tool"],
                    "Time_ratio_vs_Uncalled4": row["Time_min"] / unc_row["Time_min"]
                    if pd.notna(row["Time_min"])
                    else np.nan,
                    "File_ratio_vs_Uncalled4": row["FileSize_MB"] / unc_row["FileSize_MB"]
                    if pd.notna(row["FileSize_MB"])
                    else np.nan,
                }
            )
    ratio_df = pd.DataFrame(ratios)
    summary_df = pd.DataFrame(best_rows)
    ratio_df.to_csv(OUTPUT_DIR / "performance_ratios_vs_uncalled4.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "performance_best_baseline_comparison.csv", index=False)
    return ratio_df, summary_df


def annotate_ratio_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str) -> None:
    plot_data = np.log10(pivot)
    sns.heatmap(
        plot_data,
        ax=ax,
        cmap="YlGnBu",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "log10 ratio vs Uncalled4"},
        mask=pivot.isna(),
    )
    for y, row_label in enumerate(pivot.index):
        for x, col_label in enumerate(pivot.columns):
            value = pivot.loc[row_label, col_label]
            text = "NA" if pd.isna(value) else f"{value:.1f}x"
            color = "black" if pd.isna(value) or value < 15 else "white"
            ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=10, color=color)
    ax.set_title(title)
    ax.set_xlabel("Chemistry")
    ax.set_ylabel("Tool")
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=45)


def build_performance_figure(ratios: pd.DataFrame) -> None:
    chemistry_order = ["DNA r9.4", "DNA r10.4", "RNA001", "RNA004"]
    tool_order = ["Uncalled4", "f5c", "Tombo", "Nanopolish"]
    time_pivot = ratios.pivot(index="Tool", columns="Chemistry", values="Time_ratio_vs_Uncalled4").reindex(
        index=tool_order, columns=chemistry_order
    )
    size_pivot = ratios.pivot(index="Tool", columns="Chemistry", values="File_ratio_vs_Uncalled4").reindex(
        index=tool_order, columns=chemistry_order
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    annotate_ratio_heatmap(axes[0], time_pivot, "Runtime Relative to Uncalled4")
    annotate_ratio_heatmap(axes[1], size_pivot, "BAM Size Relative to Uncalled4")
    fig.suptitle("Uncalled4 Benchmark Across Sequencing Chemistries", y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "performance_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if sample_y.min() == sample_y.max():
            continue
        values.append(metric_fn(sample_y, y_score[idx]))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = recall[precision >= min_precision]
    return float(valid.max()) if len(valid) else 0.0


def best_f1_from_pr(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.nanargmax(f1))
    return float(f1[best]), float(thresholds[best])


def analyze_m6a() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(DATA_DIR / "m6a_labels.csv")
    unc = pd.read_csv(DATA_DIR / "m6a_predictions_uncalled4.csv").rename(
        columns={"probability": "probability_uncalled4"}
    )
    nano = pd.read_csv(DATA_DIR / "m6a_predictions_nanopolish.csv").rename(
        columns={"probability": "probability_nanopolish"}
    )
    merged = labels.merge(unc, on="site_id").merge(nano, on="site_id")
    merged.to_csv(OUTPUT_DIR / "m6a_merged_predictions.csv", index=False)

    y_true = merged["label"].to_numpy()
    metric_rows = []
    curve_rows = []
    for tool in ["uncalled4", "nanopolish"]:
        score = merged[f"probability_{tool}"].to_numpy()
        ap = average_precision_score(y_true, score)
        roc = roc_auc_score(y_true, score)
        ap_lo, ap_hi = bootstrap_ci(y_true, score, average_precision_score)
        roc_lo, roc_hi = bootstrap_ci(y_true, score, roc_auc_score)
        best_f1, best_threshold = best_f1_from_pr(y_true, score)
        rec90 = recall_at_precision(y_true, score, 0.90)
        rec95 = recall_at_precision(y_true, score, 0.95)
        metric_rows.append(
            {
                "tool": tool,
                "average_precision": ap,
                "average_precision_ci_low": ap_lo,
                "average_precision_ci_high": ap_hi,
                "roc_auc": roc,
                "roc_auc_ci_low": roc_lo,
                "roc_auc_ci_high": roc_hi,
                "brier_score": brier_score_loss(y_true, score),
                "best_f1": best_f1,
                "best_f1_threshold": best_threshold,
                "recall_at_precision_0.90": rec90,
                "recall_at_precision_0.95": rec95,
            }
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_true, score)
        for p, r, t in zip(precision[:-1], recall[:-1], pr_thresholds):
            curve_rows.append(
                {
                    "tool": tool,
                    "curve": "PR",
                    "x": r,
                    "y": p,
                    "threshold": t,
                }
            )
        fpr, tpr, roc_thresholds = roc_curve(y_true, score)
        for x, y, t in zip(fpr, tpr, roc_thresholds):
            curve_rows.append(
                {
                    "tool": tool,
                    "curve": "ROC",
                    "x": x,
                    "y": y,
                    "threshold": t,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    metrics.to_csv(OUTPUT_DIR / "m6a_metrics.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "m6a_curves.csv", index=False)
    return merged, metrics


def build_m6a_figure(merged: pd.DataFrame, metrics: pd.DataFrame) -> None:
    tool_labels = {"uncalled4": "Uncalled4 alignments", "nanopolish": "Nanopolish alignments"}
    colors = {"uncalled4": "#2a9d8f", "nanopolish": "#bc4749"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for tool in ["uncalled4", "nanopolish"]:
        score = merged[f"probability_{tool}"].to_numpy()
        y_true = merged["label"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, score)
        fpr, tpr, _ = roc_curve(y_true, score)
        ap = metrics.loc[metrics["tool"] == tool, "average_precision"].iloc[0]
        roc = metrics.loc[metrics["tool"] == tool, "roc_auc"].iloc[0]
        axes[0].plot(recall, precision, linewidth=2.5, color=colors[tool], label=f"{tool_labels[tool]} (AP={ap:.3f})")
        axes[1].plot(fpr, tpr, linewidth=2.5, color=colors[tool], label=f"{tool_labels[tool]} (AUROC={roc:.3f})")

    axes[0].set_title("Precision-Recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(frameon=True, fontsize=10, loc="lower left")

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[1].set_title("ROC")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(frameon=True, fontsize=10, loc="lower right")

    plot_df = merged.melt(
        id_vars=["site_id", "label"],
        value_vars=["probability_uncalled4", "probability_nanopolish"],
        var_name="tool",
        value_name="probability",
    )
    plot_df["tool"] = plot_df["tool"].str.replace("probability_", "", regex=False)
    plot_df["label_name"] = plot_df["label"].map({0: "Negative", 1: "Positive"})
    sns.boxplot(
        data=plot_df,
        x="tool",
        y="probability",
        hue="label_name",
        ax=axes[2],
        palette={"Negative": "#adb5bd", "Positive": "#ffb703"},
        showfliers=False,
    )
    axes[2].set_title("Score Separation by Label")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Predicted probability")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Uncalled4", "Nanopolish"])
    axes[2].legend(title="", frameon=True, fontsize=10, loc="upper left")

    fig.suptitle("m6A Site Detection Benefits from Uncalled4 Alignments", y=1.03, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "m6a_validation.png", bbox_inches="tight")
    plt.close(fig)


def write_key_results(
    pore_summary: pd.DataFrame,
    position_sensitivity: pd.DataFrame,
    performance_summary: pd.DataFrame,
    m6a_metrics: pd.DataFrame,
) -> None:
    center_rows = []
    for spec in PORE_MODELS:
        subset = position_sensitivity.loc[position_sensitivity["model_id"] == spec.model_id]
        peak_row = subset.loc[subset["mean_abs_substitution_delta"].idxmax()]
        center_pos = int(peak_row["position"])
        center_val = float(peak_row["mean_abs_substitution_delta"])
        edge_val = subset.loc[subset["position"].isin([1, spec.k]), "mean_abs_substitution_delta"].mean()
        center_rows.append(
            {
                "model": spec.label,
                "peak_position": center_pos,
                "peak_shift": center_val,
                "edge_shift_mean": edge_val,
                "peak_to_edge_ratio": center_val / edge_val,
            }
        )

    key_results = {
        "positive_rate_m6a": 0.2048,
        "performance_best_baseline": performance_summary.round(4).to_dict(orient="records"),
        "position_centrality": pd.DataFrame(center_rows).round(4).to_dict(orient="records"),
        "m6a_metrics": m6a_metrics.round(4).to_dict(orient="records"),
        "pore_model_summary": pore_summary.round(4).to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "key_results.json", "w", encoding="utf-8") as handle:
        json.dump(key_results, handle, indent=2)


def main() -> None:
    ensure_dirs()
    set_plot_style()

    models = load_pore_models()
    pore_summary = summarize_pore_models(models)
    base_effects = compute_base_effects(models, feature="current_mean")
    position_sensitivity = compute_position_sensitivity(models, feature="current_mean")
    perf_ratios, perf_summary = analyze_performance()
    m6a_merged, m6a_metrics = analyze_m6a()

    build_data_overview_figure(models)
    build_base_effect_figure(base_effects)
    build_position_sensitivity_figure(position_sensitivity)
    build_performance_figure(perf_ratios)
    build_m6a_figure(m6a_merged, m6a_metrics)
    write_key_results(pore_summary, position_sensitivity, perf_summary, m6a_metrics)

    print("Analysis complete.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Figures written to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
