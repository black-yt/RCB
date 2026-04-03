import itertools
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
IMAGE_DIR = REPORT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200


def parse_rep_from_filename(path: Path) -> int:
    match = re.search(r"rep-(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not parse repetition from {path}")
    return int(match.group(1))


def parse_rep_from_population(value: str) -> int:
    match = re.search(r",\s*(\d+)$", value)
    if not match:
        raise ValueError(f"Could not parse repetition from population {value}")
    return int(match.group(1))


def load_inputs() -> dict:
    cell_pop = pd.read_csv(DATA_DIR / "cell-populations.csv")
    final_resp = pd.read_csv(DATA_DIR / "final-response-likelihoods.csv")
    runtime = pd.read_csv(DATA_DIR / "optimization_runtime_data.csv")
    selected = pd.read_csv(DATA_DIR / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
    sim_specific = pd.read_csv(DATA_DIR / "sim-specific-response-likelihoods.csv")
    vaccine = pd.read_csv(DATA_DIR / "vaccine.budget-10.minsum.adaptive.csv")
    score_paths = sorted(DATA_DIR.glob("vaccine-elements.scores.*.csv"))
    scores = {}
    for path in score_paths:
        rep = parse_rep_from_filename(path)
        scores[rep] = pd.read_csv(path)
    return {
        "cell_pop": cell_pop,
        "final_resp": final_resp,
        "runtime": runtime,
        "selected": selected,
        "sim_specific": sim_specific,
        "vaccine": vaccine,
        "scores": scores,
    }


def compute_validation(final_resp: pd.DataFrame, selected: pd.DataFrame, scores: dict) -> pd.DataFrame:
    records = []
    final_resp = final_resp.copy()
    final_resp["rep"] = final_resp["population"].map(parse_rep_from_population)
    for rep, score_df in scores.items():
        chosen = set(selected.loc[selected["repetition"] == rep, "peptide"])
        agg = (
            score_df.loc[score_df["vaccine_element"].isin(chosen)]
            .groupby("cell_id")["p_no_response"]
            .prod()
            .rename("recomputed_p_no_response")
            .reset_index()
        )
        agg["recomputed_p_response"] = 1.0 - agg["recomputed_p_no_response"]
        truth = (
            final_resp.loc[final_resp["rep"] == rep, ["name", "p_response", "num_presented_peptides"]]
            .rename(columns={"name": "cell_id"})
            .copy()
        )
        merged = truth.merge(agg, on="cell_id", how="left")
        merged["rep"] = rep
        merged["abs_diff"] = (merged["recomputed_p_response"] - merged["p_response"]).abs()
        records.append(merged)
    validation = pd.concat(records, ignore_index=True)
    validation.to_csv(OUTPUT_DIR / "validation_recomputed_vs_provided.csv", index=False)
    return validation


def build_candidate_matrices(scores: dict) -> tuple[list[str], dict[int, pd.DataFrame], np.ndarray]:
    candidate_order = sorted(next(iter(scores.values()))["vaccine_element"].unique().tolist())
    rep_matrices = {}
    stacked = []
    for rep, score_df in scores.items():
        pivot = score_df.pivot(index="cell_id", columns="vaccine_element", values="p_no_response")
        pivot = pivot[candidate_order].sort_index()
        rep_matrices[rep] = pivot
        stacked.append(pivot.to_numpy())
    pooled_matrix = np.vstack(stacked)
    return candidate_order, rep_matrices, pooled_matrix


def subset_metrics(matrix: np.ndarray, subset_idx: tuple[int, ...]) -> dict:
    p_response = 1.0 - np.prod(matrix[:, subset_idx], axis=1)
    return {
        "mean_p_response": float(np.mean(p_response)),
        "median_p_response": float(np.median(p_response)),
        "coverage_05": float(np.mean(p_response >= 0.5)),
        "coverage_09": float(np.mean(p_response >= 0.9)),
        "min_p_response": float(np.min(p_response)),
        "p_response_vector": p_response,
    }


def exact_budget_sweep(candidate_order: list[str], pooled_matrix: np.ndarray, rep_matrices: dict[int, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for budget in range(1, len(candidate_order) + 1):
        best = None
        for subset_idx in itertools.combinations(range(len(candidate_order)), budget):
            pooled = subset_metrics(pooled_matrix, subset_idx)
            record = {
                "budget": budget,
                "subset": tuple(candidate_order[i] for i in subset_idx),
                "subset_label": "|".join(candidate_order[i] for i in subset_idx),
                "mean_p_response": pooled["mean_p_response"],
                "median_p_response": pooled["median_p_response"],
                "coverage_05": pooled["coverage_05"],
                "coverage_09": pooled["coverage_09"],
                "min_p_response": pooled["min_p_response"],
            }
            per_rep_means = []
            per_rep_cov09 = []
            for rep, matrix_df in rep_matrices.items():
                rep_metrics = subset_metrics(matrix_df.to_numpy(), subset_idx)
                record[f"rep_{rep}_mean_p_response"] = rep_metrics["mean_p_response"]
                record[f"rep_{rep}_coverage_09"] = rep_metrics["coverage_09"]
                per_rep_means.append(rep_metrics["mean_p_response"])
                per_rep_cov09.append(rep_metrics["coverage_09"])
            record["mean_p_response_sd_across_reps"] = float(np.std(per_rep_means))
            record["coverage_09_sd_across_reps"] = float(np.std(per_rep_cov09))
            if best is None or record["mean_p_response"] > best["mean_p_response"] + 1e-12:
                best = record
        records.append(best)
    budget_df = pd.DataFrame(records)
    budget_df.to_csv(OUTPUT_DIR / "budget_optimization_summary.csv", index=False)
    return budget_df


def compute_element_summary(cell_pop: pd.DataFrame, scores: dict, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    presentation_counts = cell_pop["mutation"].value_counts().to_dict()
    selected_counts = selected["peptide"].value_counts().to_dict()
    for rep, score_df in scores.items():
        summary = score_df.groupby("vaccine_element")["p_response"].agg(["mean", "median", "max"]).reset_index()
        summary["rep"] = rep
        rows.append(summary)
    by_rep = pd.concat(rows, ignore_index=True)
    aggregated = (
        by_rep.groupby("vaccine_element")[["mean", "median", "max"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "vaccine_element",
        "mean_p_response_mean",
        "mean_p_response_sd",
        "median_p_response_mean",
        "median_p_response_sd",
        "max_p_response_mean",
        "max_p_response_sd",
    ]
    aggregated["selection_frequency"] = aggregated["vaccine_element"].map(selected_counts).fillna(0).astype(int)
    aggregated["selected"] = aggregated["selection_frequency"] > 0
    aggregated["presentation_count"] = aggregated["vaccine_element"].map(presentation_counts).fillna(0).astype(int)
    aggregated["presentation_fraction"] = aggregated["presentation_count"] / max(cell_pop.shape[0], 1)
    aggregated = aggregated.sort_values(["selected", "mean_p_response_mean", "presentation_count"], ascending=[False, False, False])
    aggregated.to_csv(OUTPUT_DIR / "element_summary.csv", index=False)
    return aggregated


def compute_selected_set_metrics(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = (
        selected.groupby("repetition")["peptide"]
        .agg(lambda x: tuple(sorted(x.tolist())))
        .reset_index()
        .rename(columns={"peptide": "selected_subset"})
    )
    reps = sets["repetition"].tolist()
    iou_records = []
    for rep_a in reps:
        set_a = set(sets.loc[sets["repetition"] == rep_a, "selected_subset"].iloc[0])
        for rep_b in reps:
            set_b = set(sets.loc[sets["repetition"] == rep_b, "selected_subset"].iloc[0])
            iou = len(set_a & set_b) / len(set_a | set_b)
            iou_records.append({"rep_a": rep_a, "rep_b": rep_b, "iou": iou})
    iou_df = pd.DataFrame(iou_records)
    iou_df.to_csv(OUTPUT_DIR / "iou_matrix.csv", index=False)
    sets.to_csv(OUTPUT_DIR / "selected_vaccine_subsets.csv", index=False)
    return sets, iou_df


def compute_response_summaries(final_resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_resp = final_resp.copy()
    final_resp["rep"] = final_resp["population"].map(parse_rep_from_population)
    summary = (
        final_resp.groupby("rep")["p_response"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_p_response", "median": "median_p_response", "min": "min_p_response", "max": "max_p_response"})
    )
    summary.to_csv(OUTPUT_DIR / "final_response_summary.csv", index=False)

    thresholds = np.linspace(0.0, 1.0, 101)
    records = []
    for rep, rep_df in final_resp.groupby("rep"):
        values = rep_df["p_response"].to_numpy()
        for threshold in thresholds:
            records.append(
                {
                    "rep": rep,
                    "threshold": threshold,
                    "coverage": float(np.mean(values >= threshold)),
                }
            )
    overall = final_resp["p_response"].to_numpy()
    for threshold in thresholds:
        records.append(
            {
                "rep": "overall",
                "threshold": threshold,
                "coverage": float(np.mean(overall >= threshold)),
            }
        )
    coverage = pd.DataFrame(records)
    coverage.to_csv(OUTPUT_DIR / "coverage_by_threshold.csv", index=False)
    return summary, coverage


def compute_runtime_summary(runtime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime = runtime.copy()
    sample_rows = []
    for sample_id, sample_df in runtime.groupby("SampleID"):
        x = np.log10(sample_df["PopulationSize"].to_numpy())
        y = np.log10(sample_df["RunTime"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        sample_rows.append(
            {
                "SampleID": sample_id,
                "log10_slope": float(slope),
                "log10_intercept": float(intercept),
                "min_runtime": float(sample_df["RunTime"].min()),
                "max_runtime": float(sample_df["RunTime"].max()),
            }
        )
    by_sample = pd.DataFrame(sample_rows).sort_values("SampleID")
    by_sample.to_csv(OUTPUT_DIR / "runtime_by_sample.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "median_log10_slope": float(by_sample["log10_slope"].median()),
                "mean_log10_slope": float(by_sample["log10_slope"].mean()),
                "min_runtime": float(runtime["RunTime"].min()),
                "max_runtime": float(runtime["RunTime"].max()),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    return by_sample, summary


def compute_ablation(candidate_order: list[str], pooled_matrix: np.ndarray, selected_subset: tuple[str, ...]) -> pd.DataFrame:
    chosen_idx = tuple(candidate_order.index(name) for name in selected_subset)
    baseline = subset_metrics(pooled_matrix, chosen_idx)["mean_p_response"]
    rows = []
    for element in selected_subset:
        reduced = tuple(name for name in selected_subset if name != element)
        reduced_idx = tuple(candidate_order.index(name) for name in reduced)
        reduced_score = subset_metrics(pooled_matrix, reduced_idx)["mean_p_response"]
        rows.append(
            {
                "analysis": "remove_selected_element",
                "element": element,
                "mean_p_response": reduced_score,
                "delta_vs_budget10_optimum": reduced_score - baseline,
            }
        )
    excluded = [name for name in candidate_order if name not in selected_subset]
    full_idx = tuple(candidate_order.index(name) for name in candidate_order)
    full_score = subset_metrics(pooled_matrix, full_idx)["mean_p_response"]
    for element in excluded:
        with_extra = tuple(sorted(selected_subset + (element,)))
        rows.append(
            {
                "analysis": "add_excluded_element",
                "element": element,
                "mean_p_response": full_score,
                "delta_vs_budget10_optimum": full_score - baseline,
            }
        )
    ablation = pd.DataFrame(rows)
    ablation.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    return ablation


def save_key_metrics(data: dict) -> None:
    with open(OUTPUT_DIR / "key_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def plot_mutation_prevalence(element_summary: pd.DataFrame) -> None:
    plot_df = element_summary.sort_values("presentation_count", ascending=False).copy()
    plot_df["status"] = np.where(plot_df["selected"], "Selected", "Excluded")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="vaccine_element", y="presentation_count", hue="status", palette={"Selected": "#1f77b4", "Excluded": "#d62728"})
    plt.xlabel("Neoantigen element")
    plt.ylabel("Presentation count in simulated cells")
    plt.title("Presentation prevalence of candidate neoantigen elements")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_1_mutation_prevalence.png")
    plt.close()


def plot_element_heatmap(scores: dict, selected_subset: tuple[str, ...]) -> None:
    rows = []
    for rep, score_df in scores.items():
        summary = score_df.groupby("vaccine_element")["p_response"].mean().reset_index()
        summary["rep"] = rep
        rows.append(summary)
    plot_df = pd.concat(rows, ignore_index=True)
    pivot = plot_df.pivot(index="vaccine_element", columns="rep", values="p_response")
    pivot = pivot.loc[sorted(pivot.index, key=lambda x: (x not in selected_subset, x))]
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, cmap="mako", annot=True, fmt=".2f", cbar_kws={"label": "Mean per-cell response probability"})
    plt.xlabel("Repetition")
    plt.ylabel("Neoantigen element")
    plt.title("Element-level response contribution across repetitions")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_2_element_heatmap.png")
    plt.close()


def plot_response_distribution(final_resp: pd.DataFrame) -> None:
    plot_df = final_resp.copy()
    plot_df["rep"] = plot_df["population"].map(parse_rep_from_population)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x="rep", y="p_response", inner="quartile", color="#4c72b0")
    plt.xlabel("Repetition")
    plt.ylabel("Final per-cell immune response probability")
    plt.title("Distribution of per-cell response probabilities for the optimized vaccine")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_3_response_distribution.png")
    plt.close()


def plot_coverage_curve(coverage: pd.DataFrame) -> None:
    plot_df = coverage.copy()
    overall = plot_df.loc[plot_df["rep"] == "overall"]
    by_rep = plot_df.loc[plot_df["rep"] != "overall"].copy()
    by_rep["rep"] = by_rep["rep"].astype(int)
    plt.figure(figsize=(10, 6))
    for rep, rep_df in by_rep.groupby("rep"):
        plt.plot(rep_df["threshold"], rep_df["coverage"], color="#b0b0b0", alpha=0.5, linewidth=1)
    plt.plot(overall["threshold"], overall["coverage"], color="#d62728", linewidth=3, label="Overall")
    plt.xlabel("Response-probability threshold")
    plt.ylabel("Coverage ratio of tumor cells")
    plt.title("Coverage curve induced by the optimized vaccine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_4_coverage_curve.png")
    plt.close()


def plot_budget_tradeoff(budget_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(budget_df["budget"], budget_df["mean_p_response"], marker="o", color="#1f77b4", label="Mean response")
    ax2.plot(budget_df["budget"], budget_df["coverage_09"], marker="s", color="#ff7f0e", label="Coverage at p>=0.9")
    ax1.set_xlabel("Budget (number of neoantigen elements)")
    ax1.set_ylabel("Mean per-cell response probability", color="#1f77b4")
    ax2.set_ylabel("Coverage ratio at p>=0.9", color="#ff7f0e")
    ax1.set_title("Exact budget sweep over all candidate subsets")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    fig.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_5_budget_tradeoff.png")
    plt.close()


def plot_iou_matrix(iou_df: pd.DataFrame) -> None:
    pivot = iou_df.pivot(index="rep_a", columns="rep_b", values="iou")
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot, vmin=0, vmax=1, cmap="crest", annot=True, fmt=".2f", cbar_kws={"label": "IoU"})
    plt.xlabel("Repetition")
    plt.ylabel("Repetition")
    plt.title("Pairwise IoU of optimized vaccine compositions")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_6_iou_matrix.png")
    plt.close()


def plot_runtime_scaling(runtime: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    for sample_id, sample_df in runtime.groupby("SampleID"):
        ordered = sample_df.sort_values("PopulationSize")
        plt.plot(ordered["PopulationSize"], ordered["RunTime"], marker="o", linewidth=2, alpha=0.7, label=str(sample_id))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Population size")
    plt.ylabel("Optimization runtime (seconds)")
    plt.title("Runtime scaling across patient samples")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_7_runtime_scaling.png")
    plt.close()


def plot_validation(validation: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(validation["p_response"], validation["recomputed_p_response"], s=18, alpha=0.6, color="#2ca02c")
    lims = [0.0, 1.0]
    plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Provided final p_response")
    plt.ylabel("Recomputed p_response")
    plt.title("Exact reconstruction of final response probabilities")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_8_validation.png")
    plt.close()


def main() -> None:
    data = load_inputs()
    validation = compute_validation(data["final_resp"], data["selected"], data["scores"])
    candidate_order, rep_matrices, pooled_matrix = build_candidate_matrices(data["scores"])
    budget_df = exact_budget_sweep(candidate_order, pooled_matrix, rep_matrices)
    element_summary = compute_element_summary(data["cell_pop"], data["scores"], data["selected"])
    selected_sets, iou_df = compute_selected_set_metrics(data["selected"])
    response_summary, coverage = compute_response_summaries(data["final_resp"])
    runtime_by_sample, runtime_summary = compute_runtime_summary(data["runtime"])

    selected_subset = tuple(sorted(data["vaccine"]["peptide"].tolist()))
    ablation = compute_ablation(candidate_order, pooled_matrix, selected_subset)

    best_budget10 = budget_df.loc[budget_df["budget"] == 10].iloc[0]
    key_metrics = {
        "n_repetitions": int(data["selected"]["repetition"].nunique()),
        "n_cells_total": int(sum(df["cell_id"].nunique() for df in data["scores"].values())),
        "n_cells_per_repetition": int(next(iter(data["scores"].values()))["cell_id"].nunique()),
        "n_candidates": len(candidate_order),
        "budget": 10,
        "selected_vaccine_elements": list(selected_subset),
        "excluded_elements": [name for name in candidate_order if name not in selected_subset],
        "overall_mean_p_response": float(data["final_resp"]["p_response"].mean()),
        "overall_median_p_response": float(data["final_resp"]["p_response"].median()),
        "coverage_at_05": float(np.mean(data["final_resp"]["p_response"] >= 0.5)),
        "coverage_at_09": float(np.mean(data["final_resp"]["p_response"] >= 0.9)),
        "coverage_at_095": float(np.mean(data["final_resp"]["p_response"] >= 0.95)),
        "pairwise_iou_mean": float(iou_df["iou"].mean()),
        "pairwise_iou_min": float(iou_df["iou"].min()),
        "validation_max_abs_diff": float(validation["abs_diff"].max()),
        "budget10_optimal_mean_p_response": float(best_budget10["mean_p_response"]),
        "budget10_optimal_subset": best_budget10["subset_label"].split("|"),
        "budget12_mean_p_response": float(budget_df.loc[budget_df["budget"] == 12, "mean_p_response"].iloc[0]),
        "budget10_to_12_gain": float(
            budget_df.loc[budget_df["budget"] == 12, "mean_p_response"].iloc[0]
            - budget_df.loc[budget_df["budget"] == 10, "mean_p_response"].iloc[0]
        ),
        "runtime_median_log10_slope": float(runtime_summary["median_log10_slope"].iloc[0]),
        "runtime_min_seconds": float(data["runtime"]["RunTime"].min()),
        "runtime_max_seconds": float(data["runtime"]["RunTime"].max()),
    }
    save_key_metrics(key_metrics)

    plot_mutation_prevalence(element_summary)
    plot_element_heatmap(data["scores"], selected_subset)
    plot_response_distribution(data["final_resp"])
    plot_coverage_curve(coverage)
    plot_budget_tradeoff(budget_df)
    plot_iou_matrix(iou_df)
    plot_runtime_scaling(data["runtime"])
    plot_validation(validation)

    overview = pd.DataFrame(
        [
            {"metric": "Candidate neoantigen elements", "value": len(candidate_order)},
            {"metric": "Selected vaccine elements", "value": len(selected_subset)},
            {"metric": "Repetitions", "value": data["selected"]["repetition"].nunique()},
            {"metric": "Cells per repetition", "value": next(iter(data["scores"].values()))["cell_id"].nunique()},
            {"metric": "Total evaluated cells", "value": sum(df["cell_id"].nunique() for df in data["scores"].values())},
            {"metric": "Overall mean p_response", "value": round(data["final_resp"]["p_response"].mean(), 6)},
            {"metric": "Coverage at p>=0.9", "value": round(float(np.mean(data["final_resp"]["p_response"] >= 0.9)), 6)},
            {"metric": "Mean pairwise IoU", "value": round(float(iou_df["iou"].mean()), 6)},
            {"metric": "Validation max abs diff", "value": validation["abs_diff"].max()},
        ]
    )
    overview.to_csv(OUTPUT_DIR / "analysis_overview.csv", index=False)

    print("Analysis completed.")
    print(json.dumps(key_metrics, indent=2, sort_keys=True))
    print("Outputs written to", OUTPUT_DIR)
    print("Figures written to", IMAGE_DIR)
    print("Per-repetition response summary:")
    print(response_summary.to_string(index=False))
    print("Runtime slopes:")
    print(runtime_by_sample.to_string(index=False))
    print("Ablation summary:")
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
