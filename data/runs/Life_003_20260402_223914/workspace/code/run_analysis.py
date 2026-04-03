#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required to run this analysis script. Please install it in the environment."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"

PORE_FILES = {
    "DNA r9.4.1 6-mer": DATA_DIR / "dna_r9.4.1_400bps_6mer_uncalled4.csv",
    "DNA r10.4.1 9-mer": DATA_DIR / "dna_r10.4.1_400bps_9mer_uncalled4.csv",
    "RNA r9.4.1 5-mer": DATA_DIR / "rna_r9.4.1_70bps_5mer_uncalled4.csv",
    "RNA004 9-mer": DATA_DIR / "rna004_130bps_9mer_uncalled4.csv",
}

PERFORMANCE_FILE = DATA_DIR / "performance_summary.csv"
M6A_LABELS_FILE = DATA_DIR / "m6a_labels.csv"
M6A_UNCALLED4_FILE = DATA_DIR / "m6a_predictions_uncalled4.csv"
M6A_NANOPOLISH_FILE = DATA_DIR / "m6a_predictions_nanopolish.csv"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")


def load_pore_models() -> Dict[str, pd.DataFrame]:
    pore_models: Dict[str, pd.DataFrame] = {}
    for chemistry, path in PORE_FILES.items():
        df = pd.read_csv(path)
        df["chemistry"] = chemistry
        df["kmer_length"] = df["kmer"].str.len()
        df["gc_count"] = df["kmer"].str.count("G") + df["kmer"].str.count("C")
        df["gc_fraction"] = df["gc_count"] / df["kmer_length"]
        for base in "ACGT":
            df[f"count_{base}"] = df["kmer"].str.count(base)
            df[f"frac_{base}"] = df[f"count_{base}"] / df["kmer_length"]
        center_idx = (df["kmer_length"] // 2).astype(int)
        df["central_base"] = [kmer[idx] for kmer, idx in zip(df["kmer"], center_idx)]
        pore_models[chemistry] = df
    return pore_models


def summarize_pore_models(pore_models: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[dict] = []
    corr_rows: List[dict] = []
    central_rows: List[dict] = []

    for chemistry, df in pore_models.items():
        summaries.append(
            {
                "chemistry": chemistry,
                "n_kmers": int(len(df)),
                "kmer_length": int(df["kmer_length"].iloc[0]),
                "current_mean_mean": df["current_mean"].mean(),
                "current_mean_std": df["current_mean"].std(),
                "current_mean_min": df["current_mean"].min(),
                "current_mean_max": df["current_mean"].max(),
                "current_std_mean": df["current_std"].mean(),
                "dwell_time_mean": df["dwell_time"].mean(),
                "gc_fraction_mean": df["gc_fraction"].mean(),
                "missing_values": int(df.isna().sum().sum()),
            }
        )
        corr_rows.append(
            {
                "chemistry": chemistry,
                "corr_current_vs_gc_fraction": df["current_mean"].corr(df["gc_fraction"]),
                "corr_current_vs_dwell": df["current_mean"].corr(df["dwell_time"]),
                "corr_current_vs_current_std": df["current_mean"].corr(df["current_std"]),
            }
        )
        central = (
            df.groupby("central_base")[["current_mean", "current_std", "dwell_time"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        renamed_columns = []
        for c in central.columns:
            if isinstance(c, tuple):
                if c[0] == "central_base":
                    renamed_columns.append("central_base")
                else:
                    renamed_columns.append(f"{c[0]}_{c[1]}")
            else:
                renamed_columns.append(c)
        central.columns = renamed_columns
        central["chemistry"] = chemistry
        central_rows.append(central)

    summary_df = pd.DataFrame(summaries).sort_values("chemistry")
    corr_df = pd.DataFrame(corr_rows).sort_values("chemistry")
    central_df = pd.concat(central_rows, ignore_index=True)
    return summary_df, corr_df, central_df


def create_dataset_overview(pore_models: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    performance = pd.read_csv(PERFORMANCE_FILE)
    labels = pd.read_csv(M6A_LABELS_FILE)
    uncalled4 = pd.read_csv(M6A_UNCALLED4_FILE)
    nanopolish = pd.read_csv(M6A_NANOPOLISH_FILE)

    rows: List[dict] = []
    for chemistry, df in pore_models.items():
        rows.append(
            {
                "dataset": chemistry,
                "type": "pore_model",
                "rows": len(df),
                "columns": df.shape[1],
                "missing_values": int(df.isna().sum().sum()),
                "notes": f"k={int(df['kmer_length'].iloc[0])}",
            }
        )
    rows.extend(
        [
            {
                "dataset": "performance_summary",
                "type": "benchmark",
                "rows": len(performance),
                "columns": performance.shape[1],
                "missing_values": int(performance.isna().sum().sum()),
                "notes": f"tools={performance['Tool'].nunique()}, chemistries={performance['Chemistry'].nunique()}",
            },
            {
                "dataset": "m6a_labels",
                "type": "labels",
                "rows": len(labels),
                "columns": labels.shape[1],
                "missing_values": int(labels.isna().sum().sum()),
                "notes": f"positive_rate={labels['label'].mean():.4f}",
            },
            {
                "dataset": "m6a_predictions_uncalled4",
                "type": "predictions",
                "rows": len(uncalled4),
                "columns": uncalled4.shape[1],
                "missing_values": int(uncalled4.isna().sum().sum()),
                "notes": "probability scores",
            },
            {
                "dataset": "m6a_predictions_nanopolish",
                "type": "predictions",
                "rows": len(nanopolish),
                "columns": nanopolish.shape[1],
                "missing_values": int(nanopolish.isna().sum().sum()),
                "notes": "probability scores",
            },
        ]
    )
    return pd.DataFrame(rows)


def analyze_performance() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf = pd.read_csv(PERFORMANCE_FILE)
    perf["Time_hr"] = perf["Time_min"] / 60.0
    perf["Throughput_relative_to_Uncalled4"] = np.nan
    perf["Size_relative_to_Uncalled4"] = np.nan

    summary_rows: List[dict] = []
    for chemistry, chem_df in perf.groupby("Chemistry"):
        uncalled_row = chem_df[chem_df["Tool"] == "Uncalled4"]
        base_time = uncalled_row["Time_min"].iloc[0] if not uncalled_row.empty else np.nan
        base_size = uncalled_row["FileSize_MB"].iloc[0] if not uncalled_row.empty else np.nan
        mask = perf["Chemistry"] == chemistry
        perf.loc[mask, "Throughput_relative_to_Uncalled4"] = base_time / perf.loc[mask, "Time_min"]
        perf.loc[mask, "Size_relative_to_Uncalled4"] = perf.loc[mask, "FileSize_MB"] / base_size

        valid = chem_df.dropna(subset=["Time_min", "FileSize_MB"])
        fastest = valid.sort_values("Time_min").iloc[0]["Tool"] if not valid.empty else None
        smallest = valid.sort_values("FileSize_MB").iloc[0]["Tool"] if not valid.empty else None
        summary_rows.append(
            {
                "Chemistry": chemistry,
                "n_tools_reported": int(valid["Tool"].nunique()),
                "fastest_tool": fastest,
                "smallest_output_tool": smallest,
                "uncalled4_time_min": base_time,
                "uncalled4_file_size_mb": base_size,
            }
        )

    pairwise = []
    for chemistry, chem_df in perf.groupby("Chemistry"):
        u = chem_df[chem_df["Tool"] == "Uncalled4"]
        if u.empty:
            continue
        u_time = u["Time_min"].iloc[0]
        u_size = u["FileSize_MB"].iloc[0]
        for _, row in chem_df.iterrows():
            if row["Tool"] == "Uncalled4":
                continue
            pairwise.append(
                {
                    "Chemistry": chemistry,
                    "Comparator": row["Tool"],
                    "time_speedup_vs_comparator": row["Time_min"] / u_time if pd.notna(row["Time_min"]) else np.nan,
                    "size_reduction_vs_comparator": row["FileSize_MB"] / u_size if pd.notna(row["FileSize_MB"]) else np.nan,
                }
            )

    return perf, pd.DataFrame(summary_rows), pd.DataFrame(pairwise)


def load_m6a_data() -> pd.DataFrame:
    labels = pd.read_csv(M6A_LABELS_FILE)
    uncalled4 = pd.read_csv(M6A_UNCALLED4_FILE).rename(columns={"probability": "prob_uncalled4"})
    nanopolish = pd.read_csv(M6A_NANOPOLISH_FILE).rename(columns={"probability": "prob_nanopolish"})
    merged = labels.merge(uncalled4, on="site_id").merge(nanopolish, on="site_id")
    merged["prob_delta_uncalled4_minus_nanopolish"] = (
        merged["prob_uncalled4"] - merged["prob_nanopolish"]
    )
    return merged


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)

    eval_thresholds = np.unique(np.concatenate([
        np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
        pr_thresholds if len(pr_thresholds) else np.array([]),
    ]))
    threshold_rows = []
    for thr in eval_thresholds:
        preds = (y_score >= thr).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision_thr = tp / (tp + fp) if (tp + fp) else 0.0
        recall_thr = tp / (tp + fn) if (tp + fn) else 0.0
        f1_thr = 2 * precision_thr * recall_thr / (precision_thr + recall_thr) if (precision_thr + recall_thr) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        threshold_rows.append(
            {
                "threshold": float(thr),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision_thr,
                "recall": recall_thr,
                "f1": f1_thr,
                "specificity": specificity,
            }
        )
    threshold_df = pd.DataFrame(threshold_rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    best_row = threshold_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0].to_dict()

    curve_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
        }
    )
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "best_threshold_by_f1": float(best_row["threshold"]),
        "best_f1": float(best_row["f1"]),
        "threshold_table": threshold_df,
        "pr_curve": curve_df,
        "roc_curve": roc_df,
    }


def analyze_m6a() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    merged = load_m6a_data()
    y_true = merged["label"].to_numpy()

    metrics_uncalled4 = classification_metrics(y_true, merged["prob_uncalled4"].to_numpy())
    metrics_nanopolish = classification_metrics(y_true, merged["prob_nanopolish"].to_numpy())

    comparison = pd.DataFrame(
        [
            {
                "model": "Uncalled4-derived",
                "auprc": metrics_uncalled4["auprc"],
                "auroc": metrics_uncalled4["auroc"],
                "best_threshold_by_f1": metrics_uncalled4["best_threshold_by_f1"],
                "best_f1": metrics_uncalled4["best_f1"],
            },
            {
                "model": "Nanopolish-derived",
                "auprc": metrics_nanopolish["auprc"],
                "auroc": metrics_nanopolish["auroc"],
                "best_threshold_by_f1": metrics_nanopolish["best_threshold_by_f1"],
                "best_f1": metrics_nanopolish["best_f1"],
            },
        ]
    )

    quantile_rows = []
    for model_name, score_col in [
        ("Uncalled4-derived", "prob_uncalled4"),
        ("Nanopolish-derived", "prob_nanopolish"),
    ]:
        work = merged[["label", score_col]].copy()
        work["quantile_bin"] = pd.qcut(work[score_col], q=10, duplicates="drop")
        q = (
            work.groupby("quantile_bin")
            .agg(n_sites=("label", "size"), positive_rate=("label", "mean"), score_mean=(score_col, "mean"))
            .reset_index()
        )
        q["model"] = model_name
        quantile_rows.append(q)
    quantiles = pd.concat(quantile_rows, ignore_index=True)

    threshold_tables = {
        "uncalled4": metrics_uncalled4["threshold_table"],
        "nanopolish": metrics_nanopolish["threshold_table"],
    }
    curves = {
        "uncalled4_pr": metrics_uncalled4["pr_curve"],
        "nanopolish_pr": metrics_nanopolish["pr_curve"],
        "uncalled4_roc": metrics_uncalled4["roc_curve"],
        "nanopolish_roc": metrics_nanopolish["roc_curve"],
    }
    summary = {
        "n_sites": int(len(merged)),
        "positive_sites": int(merged["label"].sum()),
        "positive_rate": float(merged["label"].mean()),
        "uncalled4_minus_nanopolish_auprc": float(metrics_uncalled4["auprc"] - metrics_nanopolish["auprc"]),
        "uncalled4_minus_nanopolish_auroc": float(metrics_uncalled4["auroc"] - metrics_nanopolish["auroc"]),
    }
    return merged, comparison, quantiles, {**threshold_tables, **curves, "summary": summary}


def save_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUTPUT_DIR / filename, index=False)


def plot_dataset_overview(overview: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    left = overview.copy()
    left["label"] = left["dataset"] + "\n(" + left["type"] + ")"
    axes[0].barh(left["label"], left["rows"], color="#4C72B0")
    axes[0].set_title("Dataset row counts")
    axes[0].set_xlabel("Rows")

    axes[1].barh(left["label"], left["missing_values"], color="#DD8452")
    axes[1].set_title("Missing values by dataset")
    axes[1].set_xlabel("Count of missing cells")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "dataset_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pore_model_distributions(pore_models: Dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(pore_models.values(), ignore_index=True)
    metrics = ["current_mean", "current_std", "dwell_time"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, metric in zip(axes, metrics):
        if sns is not None:
            sns.violinplot(data=combined, x="chemistry", y=metric, ax=ax, inner="quartile", cut=0)
        else:
            chemistries = combined["chemistry"].unique()
            data = [combined.loc[combined["chemistry"] == c, metric].to_numpy() for c in chemistries]
            ax.violinplot(data, showmeans=True)
            ax.set_xticks(np.arange(1, len(chemistries) + 1))
            ax.set_xticklabels(chemistries, rotation=30, ha="right")
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pore_model_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gc_relationship(pore_models: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, (chemistry, df) in zip(axes, pore_models.items()):
        ax.scatter(df["gc_fraction"], df["current_mean"], s=8, alpha=0.15, color="#4C72B0")
        b1, b0 = np.polyfit(df["gc_fraction"], df["current_mean"], deg=1)
        xs = np.linspace(df["gc_fraction"].min(), df["gc_fraction"].max(), 100)
        ax.plot(xs, b1 * xs + b0, color="#C44E52", linewidth=2)
        corr = df["gc_fraction"].corr(df["current_mean"])
        ax.set_title(f"{chemistry}\nr = {corr:.3f}")
        ax.set_xlabel("GC fraction")
        ax.set_ylabel("Current mean")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gc_vs_current_relationship.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_central_base_effects(central_df: pd.DataFrame) -> None:
    plot_df = central_df[["chemistry", "central_base", "current_mean_mean", "current_mean_std"]].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    bases = ["A", "C", "G", "T"]
    chemistries = list(plot_df["chemistry"].unique())
    width = 0.18
    x = np.arange(len(chemistries))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, base in enumerate(bases):
        sub = plot_df[plot_df["central_base"] == base].set_index("chemistry").reindex(chemistries)
        ax.bar(
            x + (i - 1.5) * width,
            sub["current_mean_mean"],
            width=width,
            label=base,
            color=colors[i],
            yerr=sub["current_mean_std"],
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(chemistries, rotation=20, ha="right")
    ax.set_ylabel("Mean current by central base")
    ax.set_title("Central-base effects on pore-model current")
    ax.legend(title="Central base")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "central_base_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_performance(perf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    left = perf.dropna(subset=["Time_min"]).copy()
    pivot_time = left.pivot(index="Chemistry", columns="Tool", values="Time_min")
    pivot_time.plot(kind="bar", ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Alignment time (min, log scale)")
    axes[0].set_title("Runtime benchmark across chemistries")
    axes[0].tick_params(axis="x", rotation=20)

    right = perf.dropna(subset=["FileSize_MB"]).copy()
    pivot_size = right.pivot(index="Chemistry", columns="Tool", values="FileSize_MB")
    pivot_size.plot(kind="bar", ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Output file size (MB, log scale)")
    axes[1].set_title("Output size benchmark across chemistries")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "performance_benchmarks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_performance_relative(perf: pd.DataFrame) -> None:
    rel = perf[perf["Tool"] != "Uncalled4"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    rel_time = rel.dropna(subset=["time_label"]) if "time_label" in rel.columns else rel.copy()
    if sns is not None:
        sns.barplot(data=rel, x="Chemistry", y="Throughput_relative_to_Uncalled4", hue="Tool", ax=axes[0])
        sns.barplot(data=rel, x="Chemistry", y="Size_relative_to_Uncalled4", hue="Tool", ax=axes[1])
    else:
        for ax, value_col, title in [
            (axes[0], "Throughput_relative_to_Uncalled4", "Comparator runtime relative to Uncalled4"),
            (axes[1], "Size_relative_to_Uncalled4", "Comparator output size relative to Uncalled4"),
        ]:
            pivot = rel.pivot(index="Chemistry", columns="Tool", values=value_col)
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=20)
    axes[0].axhline(1.0, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylabel("Comparator time / Uncalled4 time")
    axes[0].set_title("Runtime inflation relative to Uncalled4")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].axhline(1.0, linestyle="--", color="black", linewidth=1)
    axes[1].set_ylabel("Comparator file size / Uncalled4 file size")
    axes[1].set_title("Output-size inflation relative to Uncalled4")
    axes[1].tick_params(axis="x", rotation=20)

    handles0, labels0 = axes[0].get_legend_handles_labels()
    if handles0:
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    handles1, labels1 = axes[1].get_legend_handles_labels()
    if handles1:
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "performance_relative_to_uncalled4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_m6a_curves(curve_data: dict, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(
        curve_data["uncalled4_pr"]["recall"],
        curve_data["uncalled4_pr"]["precision"],
        label=f"Uncalled4-derived (AUPRC={comparison.loc[comparison['model']=='Uncalled4-derived', 'auprc'].iloc[0]:.3f})",
        linewidth=2.5,
    )
    axes[0].plot(
        curve_data["nanopolish_pr"]["recall"],
        curve_data["nanopolish_pr"]["precision"],
        label=f"Nanopolish-derived (AUPRC={comparison.loc[comparison['model']=='Nanopolish-derived', 'auprc'].iloc[0]:.3f})",
        linewidth=2.5,
    )
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("m6A precision-recall comparison")
    axes[0].legend()

    axes[1].plot(
        curve_data["uncalled4_roc"]["fpr"],
        curve_data["uncalled4_roc"]["tpr"],
        label=f"Uncalled4-derived (AUROC={comparison.loc[comparison['model']=='Uncalled4-derived', 'auroc'].iloc[0]:.3f})",
        linewidth=2.5,
    )
    axes[1].plot(
        curve_data["nanopolish_roc"]["fpr"],
        curve_data["nanopolish_roc"]["tpr"],
        label=f"Nanopolish-derived (AUROC={comparison.loc[comparison['model']=='Nanopolish-derived', 'auroc'].iloc[0]:.3f})",
        linewidth=2.5,
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("m6A ROC comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "m6a_pr_roc_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_m6a_score_distributions(merged: pd.DataFrame) -> None:
    long_df = pd.concat(
        [
            merged[["label", "prob_uncalled4"]].rename(columns={"prob_uncalled4": "probability"}).assign(model="Uncalled4-derived"),
            merged[["label", "prob_nanopolish"]].rename(columns={"prob_nanopolish": "probability"}).assign(model="Nanopolish-derived"),
        ],
        ignore_index=True,
    )
    long_df["label_name"] = long_df["label"].map({0: "Unmodified", 1: "m6A positive"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, model in zip(axes, ["Uncalled4-derived", "Nanopolish-derived"]):
        sub = long_df[long_df["model"] == model]
        if sns is not None:
            sns.histplot(data=sub, x="probability", hue="label_name", bins=30, stat="density", common_norm=False, ax=ax, alpha=0.5)
        else:
            for label_name, color in [("Unmodified", "#4C72B0"), ("m6A positive", "#C44E52")]:
                arr = sub.loc[sub["label_name"] == label_name, "probability"].to_numpy()
                ax.hist(arr, bins=30, density=True, alpha=0.5, label=label_name, color=color)
            ax.legend()
        ax.set_title(model)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "m6a_score_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_calibration(quantiles: pd.DataFrame) -> None:
    plot_df = quantiles.copy()
    plot_df["bin_index"] = plot_df.groupby("model").cumcount() + 1
    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.lineplot(data=plot_df, x="bin_index", y="positive_rate", hue="model", marker="o", linewidth=2.5, ax=ax)
    else:
        for model, sub in plot_df.groupby("model"):
            ax.plot(sub["bin_index"], sub["positive_rate"], marker="o", linewidth=2.5, label=model)
        ax.legend()
    ax.set_xlabel("Prediction-score decile (low to high)")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Observed m6A enrichment across score deciles")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "m6a_quantile_enrichment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest() -> None:
    manifest = {
        "outputs": sorted([p.name for p in OUTPUT_DIR.glob("*.csv")]) + sorted([p.name for p in OUTPUT_DIR.glob("*.json")]),
        "figures": sorted([p.name for p in FIG_DIR.glob("*.png")]),
        "entry_point": "code/run_analysis.py",
    }
    with open(OUTPUT_DIR / "analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    ensure_dirs()
    set_style()

    pore_models = load_pore_models()
    dataset_overview = create_dataset_overview(pore_models)
    pore_summary, pore_correlations, central_base_summary = summarize_pore_models(pore_models)
    performance_detailed, performance_summary, performance_pairwise = analyze_performance()
    m6a_merged, m6a_comparison, m6a_quantiles, m6a_extra = analyze_m6a()

    # Save outputs
    save_table(dataset_overview, "dataset_overview.csv")
    save_table(pore_summary, "pore_model_summary.csv")
    save_table(pore_correlations, "pore_model_correlations.csv")
    save_table(central_base_summary, "central_base_summary.csv")
    save_table(performance_detailed, "performance_detailed_metrics.csv")
    save_table(performance_summary, "performance_summary_analysis.csv")
    save_table(performance_pairwise, "performance_pairwise_vs_uncalled4.csv")
    save_table(m6a_merged, "m6a_merged_predictions.csv")
    save_table(m6a_comparison, "m6a_model_comparison.csv")
    save_table(m6a_quantiles, "m6a_quantile_enrichment.csv")
    save_table(m6a_extra["uncalled4"], "m6a_thresholds_uncalled4.csv")
    save_table(m6a_extra["nanopolish"], "m6a_thresholds_nanopolish.csv")
    save_table(m6a_extra["uncalled4_pr"], "m6a_pr_curve_uncalled4.csv")
    save_table(m6a_extra["nanopolish_pr"], "m6a_pr_curve_nanopolish.csv")
    save_table(m6a_extra["uncalled4_roc"], "m6a_roc_curve_uncalled4.csv")
    save_table(m6a_extra["nanopolish_roc"], "m6a_roc_curve_nanopolish.csv")

    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_overview": {
                    "n_datasets": int(len(dataset_overview)),
                },
                "pore_models": {
                    "chemistries": list(PORE_FILES.keys()),
                },
                "m6a": m6a_extra["summary"],
            },
            f,
            indent=2,
        )

    # Create figures
    plot_dataset_overview(dataset_overview)
    plot_pore_model_distributions(pore_models)
    plot_gc_relationship(pore_models)
    plot_central_base_effects(central_base_summary)
    plot_performance(performance_detailed)
    plot_performance_relative(performance_detailed)
    plot_m6a_curves(m6a_extra, m6a_comparison)
    plot_m6a_score_distributions(m6a_merged)
    plot_quantile_calibration(m6a_quantiles)
    write_manifest()

    print("Analysis completed. Outputs written to outputs/ and report/images/.")


if __name__ == "__main__":
    main()
