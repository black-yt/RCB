#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"

SELECTED_VACCINE_PATH = DATA_DIR / "selected-vaccine-elements.budget-10.minsum.adaptive.csv"
VACCINE_COMPOSITION_PATH = DATA_DIR / "vaccine.budget-10.minsum.adaptive.csv"
CELL_POPULATIONS_PATH = DATA_DIR / "cell-populations.csv"
FINAL_RESPONSE_PATH = DATA_DIR / "final-response-likelihoods.csv"
SIM_RESPONSE_PATH = DATA_DIR / "sim-specific-response-likelihoods.csv"
RUNTIME_PATH = DATA_DIR / "optimization_runtime_data.csv"
SCORE_GLOB = "vaccine-elements.scores.100-cells.10x.rep-*.csv"

RESPONSE_THRESHOLDS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / name, index=False)


def save_json(obj: dict, name: str) -> None:
    with open(OUTPUT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def parse_rep_from_filename(path: Path) -> int:
    match = re.search(r"rep-(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse repetition from {path}")
    return int(match.group(1))


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


def load_inputs() -> Dict[str, pd.DataFrame]:
    data = {
        "selected": pd.read_csv(SELECTED_VACCINE_PATH),
        "vaccine": pd.read_csv(VACCINE_COMPOSITION_PATH),
        "cells": pd.read_csv(CELL_POPULATIONS_PATH),
        "final_response": pd.read_csv(FINAL_RESPONSE_PATH),
        "sim_response": pd.read_csv(SIM_RESPONSE_PATH),
        "runtime": pd.read_csv(RUNTIME_PATH),
    }
    return data


def summarize_inputs(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        row = {
            "dataset": name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": ";".join(df.columns),
        }
        for col in df.columns[:4]:
            row[f"unique_{col}"] = int(df[col].nunique(dropna=False))
        rows.append(row)
    for path in sorted(DATA_DIR.glob(SCORE_GLOB)):
        df = pd.read_csv(path)
        rows.append({
            "dataset": path.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": ";".join(df.columns),
            "unique_cell_id": int(df["cell_id"].nunique()),
            "unique_vaccine_element": int(df["vaccine_element"].nunique()),
        })
    return pd.DataFrame(rows)


def build_selection_summary(selected: pd.DataFrame, vaccine: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = selected.copy()
    selected["repetition"] = selected["repetition"].astype(int)

    selection_frequency = (
        selected.groupby("peptide")
        .agg(selection_count=("repetition", "nunique"), mean_run_time=("run_time", "mean"), weight=("weight", "first"))
        .reset_index()
        .sort_values(["selection_count", "peptide"], ascending=[False, True])
    )
    selection_frequency["selection_fraction"] = selection_frequency["selection_count"] / selected["repetition"].nunique()
    selection_frequency["in_simplified_vaccine_file"] = selection_frequency["peptide"].isin(vaccine["peptide"]).astype(int)

    rep_sets = (
        selected.groupby("repetition")["peptide"]
        .apply(lambda s: tuple(sorted(s.tolist())))
        .reset_index(name="selected_peptides")
    )
    rep_sets["selected_peptides_str"] = rep_sets["selected_peptides"].apply(lambda x: ";".join(x))
    rep_sets["selection_set_size"] = rep_sets["selected_peptides"].apply(len)

    pairwise_rows = []
    rep_to_set = {int(r.repetition): set(r.selected_peptides) for r in rep_sets.itertuples()}
    for rep_a, rep_b in combinations(sorted(rep_to_set), 2):
        set_a, set_b = rep_to_set[rep_a], rep_to_set[rep_b]
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        pairwise_rows.append({
            "repetition_a": rep_a,
            "repetition_b": rep_b,
            "intersection_size": inter,
            "union_size": union,
            "iou": inter / union if union else np.nan,
            "symmetric_difference_size": len(set_a ^ set_b),
        })
    pairwise_iou = pd.DataFrame(pairwise_rows)
    return selection_frequency, rep_sets, pairwise_iou


def load_score_tables(selected_peptides: Iterable[str]) -> pd.DataFrame:
    selected_peptides = set(selected_peptides)
    frames: List[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob(SCORE_GLOB)):
        rep = parse_rep_from_filename(path)
        df = pd.read_csv(path)
        df["repetition"] = rep
        df["is_selected"] = df["vaccine_element"].isin(selected_peptides)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def reconstruct_cell_response(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_scores = scores.loc[scores["is_selected"]].copy()
    by_cell = (
        selected_scores.groupby(["repetition", "cell_id"])
        .agg(
            p_no_response_product=("p_no_response", "prod"),
            mean_element_response=("p_response", "mean"),
            max_element_response=("p_response", "max"),
            selected_element_hits=("vaccine_element", "nunique"),
        )
        .reset_index()
    )
    by_cell["reconstructed_p_response"] = 1.0 - by_cell["p_no_response_product"]
    by_cell["population"] = by_cell["repetition"].map(lambda rep: f"100-cells.10x, {rep}")
    by_cell["name"] = by_cell["cell_id"].astype(int)

    contribution = (
        selected_scores.groupby(["repetition", "vaccine_element"])
        .agg(
            mean_p_response=("p_response", "mean"),
            median_p_response=("p_response", "median"),
            max_p_response=("p_response", "max"),
            min_p_response=("p_response", "min"),
            responding_cells_50pct=("p_response", lambda s: int((s >= 0.5).sum())),
        )
        .reset_index()
        .sort_values(["repetition", "mean_p_response"], ascending=[True, False])
    )
    return by_cell, contribution


def validate_reconstruction(reconstructed: pd.DataFrame, final_response: pd.DataFrame, sim_response: pd.DataFrame) -> pd.DataFrame:
    final_ref = final_response.copy()
    final_ref["name"] = final_ref["name"].astype(int)

    sim_ref = sim_response.copy()
    sim_ref["name"] = sim_ref["name"].astype(int)
    sim_ref["repetition"] = sim_ref["vaccine"].str.extract(r"rep-(\d+)").astype(int)

    merged_final = reconstructed.merge(
        final_ref[["population", "name", "p_response", "log_p_response", "num_presented_peptides"]],
        on=["population", "name"],
        how="left",
    )
    merged_sim = reconstructed.merge(
        sim_ref[["repetition", "name", "p_response", "log_p_response"]].rename(
            columns={"p_response": "sim_specific_p_response", "log_p_response": "sim_specific_log_p_response"}
        ),
        on=["repetition", "name"],
        how="left",
    )
    validation = merged_final.merge(
        merged_sim[["repetition", "name", "sim_specific_p_response", "sim_specific_log_p_response"]],
        on=["repetition", "name"],
        how="left",
    )
    validation["abs_diff_final"] = (validation["reconstructed_p_response"] - validation["p_response"]).abs()
    validation["abs_diff_sim_specific"] = (validation["reconstructed_p_response"] - validation["sim_specific_p_response"]).abs()
    validation["reconstructed_log_p_response"] = np.log(np.clip(validation["reconstructed_p_response"], 1e-300, 1.0))
    validation["abs_diff_log_final"] = (validation["reconstructed_log_p_response"] - validation["log_p_response"]).abs()
    return validation.sort_values(["repetition", "name"])


def build_coverage_metrics(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep, rep_df in validation.groupby("repetition"):
        for threshold in RESPONSE_THRESHOLDS:
            rows.append({
                "repetition": int(rep),
                "threshold": threshold,
                "coverage_ratio": float((rep_df["reconstructed_p_response"] >= threshold).mean()),
                "mean_response_above_threshold": float(rep_df.loc[rep_df["reconstructed_p_response"] >= threshold, "reconstructed_p_response"].mean()) if (rep_df["reconstructed_p_response"] >= threshold).any() else np.nan,
                "covered_cells": int((rep_df["reconstructed_p_response"] >= threshold).sum()),
                "total_cells": int(len(rep_df)),
            })
    coverage = pd.DataFrame(rows)
    summary = (
        coverage.groupby("threshold")
        .agg(
            mean_coverage_ratio=("coverage_ratio", "mean"),
            std_coverage_ratio=("coverage_ratio", "std"),
            min_coverage_ratio=("coverage_ratio", "min"),
            max_coverage_ratio=("coverage_ratio", "max"),
        )
        .reset_index()
    )
    save_df(summary, "coverage_summary_by_threshold.csv")
    return coverage


def build_cell_population_summary(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    cells["repetition"] = cells["repetition"].astype(int)
    summary = (
        cells.groupby(["repetition", "cell_ids"])
        .agg(
            presented_peptide_count=("presented_peptides", "count"),
            unique_presented_peptides=("presented_peptides", "nunique"),
            unique_mutations=("mutation", "nunique"),
            hla_count=("presented_hlas", "nunique"),
        )
        .reset_index()
        .rename(columns={"cell_ids": "cell_id"})
    )
    return summary


def build_runtime_summary(runtime: pd.DataFrame) -> pd.DataFrame:
    runtime = runtime.copy()
    runtime["PopulationSize"] = runtime["PopulationSize"].astype(int)
    runtime["RunTime"] = runtime["RunTime"].astype(float)
    runtime["log10_population_size"] = np.log10(runtime["PopulationSize"])
    runtime["log10_runtime"] = np.log10(runtime["RunTime"])

    pop_summary = (
        runtime.groupby("PopulationSize")
        .agg(
            mean_runtime=("RunTime", "mean"),
            median_runtime=("RunTime", "median"),
            std_runtime=("RunTime", "std"),
            min_runtime=("RunTime", "min"),
            max_runtime=("RunTime", "max"),
            sample_count=("SampleID", "nunique"),
        )
        .reset_index()
        .sort_values("PopulationSize")
    )
    coeffs = np.polyfit(runtime["log10_population_size"], runtime["log10_runtime"], 1)
    pop_summary["global_scaling_exponent"] = coeffs[0]
    save_json({
        "runtime_power_law_fit": {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
        }
    }, "runtime_model.json")
    return pop_summary


def build_overall_metrics(selection_frequency: pd.DataFrame, pairwise_iou: pd.DataFrame, validation: pd.DataFrame, coverage: pd.DataFrame, runtime_summary: pd.DataFrame) -> dict:
    threshold_05 = coverage.loc[coverage["threshold"] == 0.5, "coverage_ratio"]
    metrics = {
        "selected_vaccine_elements": selection_frequency["peptide"].tolist(),
        "num_selected_elements": int(selection_frequency["peptide"].nunique()),
        "mean_pairwise_iou": float(pairwise_iou["iou"].mean()) if not pairwise_iou.empty else math.nan,
        "min_pairwise_iou": float(pairwise_iou["iou"].min()) if not pairwise_iou.empty else math.nan,
        "max_pairwise_iou": float(pairwise_iou["iou"].max()) if not pairwise_iou.empty else math.nan,
        "mean_cell_response_probability": float(validation["reconstructed_p_response"].mean()),
        "median_cell_response_probability": float(validation["reconstructed_p_response"].median()),
        "std_cell_response_probability": float(validation["reconstructed_p_response"].std()),
        "coverage_ratio_threshold_0_5_mean": float(threshold_05.mean()) if len(threshold_05) else math.nan,
        "coverage_ratio_threshold_0_5_min": float(threshold_05.min()) if len(threshold_05) else math.nan,
        "reconstruction_max_abs_error": float(validation["abs_diff_final"].max()),
        "runtime_scaling_exponent": float(runtime_summary["global_scaling_exponent"].iloc[0]),
    }
    return metrics


def plot_data_overview(cell_summary: pd.DataFrame, image_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(cell_summary["presented_peptide_count"], bins=20, color="#4C78A8", edgecolor="white")
    axes[0].set_title("Presented peptides per cell")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Cells")

    rep_means = cell_summary.groupby("repetition")["unique_mutations"].mean().reset_index()
    axes[1].bar(rep_means["repetition"], rep_means["unique_mutations"], color="#F58518")
    axes[1].set_title("Mean unique mutations per cell by repetition")
    axes[1].set_xlabel("Repetition")
    axes[1].set_ylabel("Mean unique mutations")

    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def plot_response_distribution(validation: pd.DataFrame, image_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].hist(validation["reconstructed_p_response"], bins=25, color="#54A24B", edgecolor="white")
    axes[0].set_title("Per-cell vaccine response probabilities")
    axes[0].set_xlabel("Reconstructed p(response)")
    axes[0].set_ylabel("Cells")

    box_data = [validation.loc[validation["repetition"] == rep, "reconstructed_p_response"].values for rep in sorted(validation["repetition"].unique())]
    axes[1].boxplot(box_data, labels=sorted(validation["repetition"].unique()), patch_artist=True,
                    boxprops=dict(facecolor="#E45756", alpha=0.6), medianprops=dict(color="black"))
    axes[1].set_title("Response probability by repetition")
    axes[1].set_xlabel("Repetition")
    axes[1].set_ylabel("Reconstructed p(response)")

    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_curve(coverage: pd.DataFrame, image_path: Path) -> None:
    summary = (
        coverage.groupby("threshold")
        .agg(mean_coverage=("coverage_ratio", "mean"), std_coverage=("coverage_ratio", "std"))
        .reset_index()
        .sort_values("threshold")
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(summary["threshold"], summary["mean_coverage"], marker="o", color="#72B7B2")
    ax.fill_between(
        summary["threshold"],
        summary["mean_coverage"] - summary["std_coverage"].fillna(0),
        summary["mean_coverage"] + summary["std_coverage"].fillna(0),
        alpha=0.2,
        color="#72B7B2",
    )
    ax.set_title("Tumor-cell coverage vs response threshold")
    ax.set_xlabel("Response probability threshold")
    ax.set_ylabel("Coverage ratio")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def plot_vaccine_composition(selection_frequency: pd.DataFrame, image_path: Path) -> None:
    plot_df = selection_frequency.sort_values(["selection_count", "peptide"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(plot_df["peptide"], plot_df["selection_count"], color="#B279A2")
    ax.set_title("Optimized vaccine composition stability")
    ax.set_xlabel("Selections across 10 repetitions")
    ax.set_ylabel("Neoantigen element")
    ax.set_xlim(0, 10.5)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def plot_iou_heatmap(rep_sets: pd.DataFrame, image_path: Path) -> None:
    reps = sorted(rep_sets["repetition"].tolist())
    rep_to_set = {int(r.repetition): set(r.selected_peptides) for r in rep_sets.itertuples()}
    matrix = np.zeros((len(reps), len(reps)))
    for i, rep_a in enumerate(reps):
        for j, rep_b in enumerate(reps):
            set_a, set_b = rep_to_set[rep_a], rep_to_set[rep_b]
            matrix[i, j] = len(set_a & set_b) / len(set_a | set_b)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(reps)), reps)
    ax.set_yticks(range(len(reps)), reps)
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Repetition")
    ax.set_title("Pairwise IoU of optimized vaccine sets")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("IoU")
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(runtime: pd.DataFrame, runtime_summary: pd.DataFrame, image_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for sample_id, sample_df in runtime.groupby("SampleID"):
        sample_df = sample_df.sort_values("PopulationSize")
        axes[0].plot(sample_df["PopulationSize"], sample_df["RunTime"], marker="o", alpha=0.75, label=str(sample_id))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Runtime by sample and population size")
    axes[0].set_xlabel("Population size")
    axes[0].set_ylabel("Runtime (s)")

    axes[1].plot(runtime_summary["PopulationSize"], runtime_summary["mean_runtime"], marker="o", color="#FF9DA6")
    axes[1].fill_between(
        runtime_summary["PopulationSize"],
        runtime_summary["min_runtime"],
        runtime_summary["max_runtime"],
        color="#FF9DA6",
        alpha=0.2,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Mean runtime with across-sample range")
    axes[1].set_xlabel("Population size")
    axes[1].set_ylabel("Runtime (s)")

    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    configure_matplotlib()

    data = load_inputs()
    input_summary = summarize_inputs(data)
    save_df(input_summary, "input_data_summary.csv")

    selection_frequency, rep_sets, pairwise_iou = build_selection_summary(data["selected"], data["vaccine"])
    save_df(selection_frequency, "vaccine_selection_frequency.csv")
    save_df(rep_sets[["repetition", "selected_peptides_str", "selection_set_size"]], "vaccine_selection_by_repetition.csv")
    save_df(pairwise_iou, "vaccine_pairwise_iou.csv")

    scores = load_score_tables(selection_frequency["peptide"])
    save_df(scores, "combined_vaccine_element_scores.csv")

    reconstructed, contribution = reconstruct_cell_response(scores)
    save_df(reconstructed, "reconstructed_cell_response.csv")
    save_df(contribution, "selected_element_contribution_summary.csv")

    validation = validate_reconstruction(reconstructed, data["final_response"], data["sim_response"])
    save_df(validation, "response_reconstruction_validation.csv")

    coverage = build_coverage_metrics(validation)
    save_df(coverage, "coverage_by_threshold_and_repetition.csv")

    cell_summary = build_cell_population_summary(data["cells"])
    save_df(cell_summary, "cell_population_summary.csv")

    runtime_summary = build_runtime_summary(data["runtime"])
    save_df(runtime_summary, "runtime_summary.csv")

    overall_metrics = build_overall_metrics(selection_frequency, pairwise_iou, validation, coverage, runtime_summary)
    save_json(overall_metrics, "overall_metrics.json")

    plot_data_overview(cell_summary, IMAGE_DIR / "data_overview.png")
    plot_response_distribution(validation, IMAGE_DIR / "response_distribution.png")
    plot_coverage_curve(coverage, IMAGE_DIR / "coverage_curve.png")
    plot_vaccine_composition(selection_frequency, IMAGE_DIR / "vaccine_composition.png")
    plot_iou_heatmap(rep_sets, IMAGE_DIR / "vaccine_iou_heatmap.png")
    plot_runtime(data["runtime"], runtime_summary, IMAGE_DIR / "runtime_scaling.png")


if __name__ == "__main__":
    main()
