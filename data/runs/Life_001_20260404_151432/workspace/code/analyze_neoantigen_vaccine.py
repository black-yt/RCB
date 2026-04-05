import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def ensure_dirs():
    OUTPUTS.mkdir(exist_ok=True, parents=True)
    REPORT_IMAGES.mkdir(exist_ok=True, parents=True)


def load_inputs():
    files = {
        "cell_populations": DATA / "cell-populations.csv",
        "final_response": DATA / "final-response-likelihoods.csv",
        "runtime": DATA / "optimization_runtime_data.csv",
        "selected_elements": DATA / "selected-vaccine-elements.budget-10.minsum.adaptive.csv",
        "sim_response": DATA / "sim-specific-response-likelihoods.csv",
        "vaccine_summary": DATA / "vaccine.budget-10.minsum.adaptive.csv",
    }
    score_files = sorted(DATA.glob("vaccine-elements.scores.100-cells.10x.rep-*.csv"))
    loaded = {key: pd.read_csv(path) for key, path in files.items()}
    loaded["score_files"] = score_files
    loaded["score_tables"] = {path.stem: pd.read_csv(path) for path in score_files}
    return loaded


def parse_repetition_from_vaccine(vaccine_name: str):
    if pd.isna(vaccine_name):
        return np.nan
    token = str(vaccine_name).split("rep-")[-1]
    try:
        return int(token)
    except ValueError:
        return np.nan


def build_input_audit(data):
    audit = {}
    for key, df in data.items():
        if key in {"score_files", "score_tables"}:
            continue
        audit[key] = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "n_duplicates": int(df.duplicated().sum()),
        }
    audit["score_tables"] = {
        name: {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "n_duplicates": int(df.duplicated().sum()),
        }
        for name, df in data["score_tables"].items()
    }
    return audit


def prepare_response_metrics(sim_response, final_response):
    sim_response = sim_response.copy()
    final_response = final_response.copy()
    sim_response["repetition"] = sim_response["vaccine"].apply(parse_repetition_from_vaccine)
    sim_response[["simulation_name", "population_rep"]] = sim_response["population"].str.split(",", n=1, expand=True
    )
    sim_response["population_rep"] = sim_response["population_rep"].str.strip()

    response_summary = sim_response.groupby("repetition").agg(
        n_cells=("name", "count"),
        mean_p_response=("p_response", "mean"),
        median_p_response=("p_response", "median"),
        std_p_response=("p_response", "std"),
        min_p_response=("p_response", "min"),
        max_p_response=("p_response", "max"),
        mean_presented_peptides=("num_presented_peptides", "mean"),
    ).reset_index()
    response_summary["coverage_p_ge_0_50"] = sim_response.groupby("repetition")["p_response"].apply(lambda s: (s >= 0.50).mean()).values
    response_summary["coverage_p_ge_0_90"] = sim_response.groupby("repetition")["p_response"].apply(lambda s: (s >= 0.90).mean()).values
    response_summary["coverage_p_ge_0_95"] = sim_response.groupby("repetition")["p_response"].apply(lambda s: (s >= 0.95).mean()).values

    overall = {
        "n_cells": int(len(sim_response)),
        "mean_p_response": float(sim_response["p_response"].mean()),
        "median_p_response": float(sim_response["p_response"].median()),
        "std_p_response": float(sim_response["p_response"].std()),
        "coverage_p_ge_0_50": float((sim_response["p_response"] >= 0.50).mean()),
        "coverage_p_ge_0_90": float((sim_response["p_response"] >= 0.90).mean()),
        "coverage_p_ge_0_95": float((sim_response["p_response"] >= 0.95).mean()),
        "aggregate_mean_p_response_final_file": float(final_response["p_response"].mean()),
    }
    return sim_response, response_summary, overall


def prepare_selection_metrics(selected_elements, vaccine_summary):
    selected = selected_elements.copy()
    selected = selected.rename(columns={"peptide": "mutation_element"})
    selected_sets = selected.groupby("repetition")["mutation_element"].apply(lambda s: sorted(set(s))).to_dict()

    iou_rows = []
    for rep_i, rep_j in combinations(sorted(selected_sets), 2):
        set_i, set_j = set(selected_sets[rep_i]), set(selected_sets[rep_j])
        inter = len(set_i & set_j)
        union = len(set_i | set_j)
        iou_rows.append({"rep_i": rep_i, "rep_j": rep_j, "intersection": inter, "union": union, "iou": inter / union if union else np.nan})
    iou_df = pd.DataFrame(iou_rows)

    reps = sorted(selected_sets)
    iou_matrix = pd.DataFrame(np.eye(len(reps)), index=reps, columns=reps)
    for row in iou_rows:
        iou_matrix.loc[row["rep_i"], row["rep_j"]] = row["iou"]
        iou_matrix.loc[row["rep_j"], row["rep_i"]] = row["iou"]

    mutation_frequency = selected.groupby("mutation_element").agg(
        count_selected=("repetition", "count"),
        n_repetitions=("repetition", lambda s: s.nunique()),
        mean_runtime=("run_time", "mean"),
    ).reset_index().sort_values(["n_repetitions", "mutation_element"], ascending=[False, True])
    mutation_frequency["selection_frequency"] = mutation_frequency["n_repetitions"] / selected["repetition"].nunique()

    consensus = vaccine_summary.rename(columns={"peptide": "mutation_element"}).copy()
    return selected, selected_sets, iou_df, iou_matrix, mutation_frequency, consensus


def prepare_cell_coverage(cell_populations, selected_sets):
    cell_pop = cell_populations.copy()
    per_cell = cell_pop.groupby(["repetition", "cell_ids"]).agg(
        unique_presented_mutations=("mutation", lambda s: sorted(set(s))),
        n_unique_mutations=("mutation", lambda s: s.nunique()),
        n_presentations=("mutation", "count"),
    ).reset_index()

    def count_hits(row):
        selected = set(selected_sets.get(row["repetition"], []))
        presented = set(row["unique_presented_mutations"])
        return len(selected & presented)

    per_cell["selected_mutation_hits"] = per_cell.apply(count_hits, axis=1)
    for k in [1, 2, 3]:
        per_cell[f"covered_ge_{k}"] = per_cell["selected_mutation_hits"] >= k

    coverage_summary = per_cell.groupby("repetition").agg(
        n_cells=("cell_ids", "count"),
        mean_unique_mutations=("n_unique_mutations", "mean"),
        mean_selected_hits=("selected_mutation_hits", "mean"),
        median_selected_hits=("selected_mutation_hits", "median"),
        max_selected_hits=("selected_mutation_hits", "max"),
        coverage_ge_1=("covered_ge_1", "mean"),
        coverage_ge_2=("covered_ge_2", "mean"),
        coverage_ge_3=("covered_ge_3", "mean"),
    ).reset_index()

    overall = {
        "coverage_ge_1": float(per_cell["covered_ge_1"].mean()),
        "coverage_ge_2": float(per_cell["covered_ge_2"].mean()),
        "coverage_ge_3": float(per_cell["covered_ge_3"].mean()),
        "mean_selected_hits": float(per_cell["selected_mutation_hits"].mean()),
        "median_selected_hits": float(per_cell["selected_mutation_hits"].median()),
    }
    return per_cell, coverage_summary, overall


def summarize_score_tables(score_tables, selected_sets):
    rows = []
    for name, df in score_tables.items():
        rep = int(name.split("rep-")[-1])
        selected = set(selected_sets.get(rep, []))
        df = df.copy()
        df["is_selected"] = df["vaccine_element"].isin(selected)
        selected_df = df[df["is_selected"]]
        rows.append({
            "repetition": rep,
            "n_rows": int(len(df)),
            "n_selected_rows": int(len(selected_df)),
            "mean_selected_element_p_response": float(selected_df["p_response"].mean()),
            "max_selected_element_p_response": float(selected_df["p_response"].max()),
            "n_unique_selected_elements_in_scores": int(selected_df["vaccine_element"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("repetition")


def analyze_runtime(runtime_df, selected):
    runtime = runtime_df.copy().sort_values(["SampleID", "PopulationSize"])
    summary = runtime.groupby("PopulationSize").agg(
        mean_runtime=("RunTime", "mean"),
        median_runtime=("RunTime", "median"),
        std_runtime=("RunTime", "std"),
        min_runtime=("RunTime", "min"),
        max_runtime=("RunTime", "max"),
    ).reset_index()

    x = np.log10(runtime["PopulationSize"].astype(float).values)
    y = np.log10(runtime["RunTime"].astype(float).values)
    slope, intercept = np.polyfit(x, y, 1)
    runtime_scaling = {
        "log10_slope": float(slope),
        "log10_intercept": float(intercept),
    }

    selection_runtime_summary = selected.groupby("repetition").agg(
        selection_runtime=("run_time", "first"),
        n_elements=("mutation_element", "count"),
    ).reset_index()
    return runtime, summary, runtime_scaling, selection_runtime_summary


def merge_main_table(per_cell, sim_response):
    sim_small = sim_response[["repetition", "name", "p_response", "num_presented_peptides"]].copy()
    sim_small = sim_small.rename(columns={"name": "cell_ids", "p_response": "cell_p_response"})
    sim_small["cell_ids"] = sim_small["cell_ids"].astype(int)
    merged = per_cell.merge(sim_small, on=["repetition", "cell_ids"], how="left")
    return merged


def save_tables(**tables):
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(OUTPUTS / f"{name}.csv", index=False)


def plot_data_overview(cell_populations, sim_response):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mutation_counts = cell_populations.groupby("repetition")["mutation"].nunique().reset_index(name="n_unique_mutations")
    sns.barplot(data=mutation_counts, x="repetition", y="n_unique_mutations", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Unique presented mutations by repetition")
    axes[0].set_xlabel("Repetition")
    axes[0].set_ylabel("Unique mutations")

    sns.histplot(sim_response["num_presented_peptides"], bins=20, kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("Distribution of presented peptides per cell")
    axes[1].set_xlabel("Number of presented peptides")
    axes[1].set_ylabel("Cell count")
    fig.savefig(REPORT_IMAGES / "data_overview.png")
    plt.close(fig)


def plot_response_distribution(sim_response, response_summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(data=sim_response, x="repetition", y="p_response", ax=axes[0], inner="quartile", color="#8172B2")
    axes[0].set_title("Per-cell immune response probability by repetition")
    axes[0].set_xlabel("Repetition")
    axes[0].set_ylabel("p(response)")
    axes[0].set_ylim(0, 1.02)

    melted = response_summary.melt(
        id_vars="repetition",
        value_vars=["coverage_p_ge_0_50", "coverage_p_ge_0_90", "coverage_p_ge_0_95"],
        var_name="threshold",
        value_name="coverage",
    )
    threshold_map = {
        "coverage_p_ge_0_50": ">= 0.50",
        "coverage_p_ge_0_90": ">= 0.90",
        "coverage_p_ge_0_95": ">= 0.95",
    }
    melted["threshold"] = melted["threshold"].map(threshold_map)
    sns.lineplot(data=melted, x="repetition", y="coverage", hue="threshold", marker="o", ax=axes[1])
    axes[1].set_title("High-response coverage by probability threshold")
    axes[1].set_xlabel("Repetition")
    axes[1].set_ylabel("Fraction of cells")
    axes[1].set_ylim(0, 1.02)
    fig.savefig(REPORT_IMAGES / "response_probability_distribution.png")
    plt.close(fig)


def plot_coverage(coverage_summary):
    melted = coverage_summary.melt(
        id_vars="repetition",
        value_vars=["coverage_ge_1", "coverage_ge_2", "coverage_ge_3"],
        var_name="criterion",
        value_name="coverage",
    )
    label_map = {"coverage_ge_1": ">=1 hit", "coverage_ge_2": ">=2 hits", "coverage_ge_3": ">=3 hits"}
    melted["criterion"] = melted["criterion"].map(label_map)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=melted, x="repetition", y="coverage", hue="criterion", marker="o", ax=ax)
    ax.set_title("Tumor-cell coverage sensitivity to hit threshold")
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Coverage ratio")
    ax.set_ylim(0, 1.02)
    fig.savefig(REPORT_IMAGES / "coverage_thresholds.png")
    plt.close(fig)


def plot_iou(iou_matrix, mutation_frequency):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(iou_matrix.astype(float), annot=True, cmap="viridis", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Pairwise IoU of optimized vaccine compositions")
    axes[0].set_xlabel("Repetition")
    axes[0].set_ylabel("Repetition")

    freq = mutation_frequency.sort_values(["n_repetitions", "mutation_element"], ascending=[False, True])
    sns.barplot(data=freq, x="mutation_element", y="selection_frequency", color="#C44E52", ax=axes[1])
    axes[1].set_title("Mutation-element selection frequency")
    axes[1].set_xlabel("Mutation element")
    axes[1].set_ylabel("Fraction of repetitions selected")
    axes[1].tick_params(axis="x", rotation=45)
    fig.savefig(REPORT_IMAGES / "iou_heatmap.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sns.barplot(data=freq, x="mutation_element", y="n_repetitions", color="#64B5CD", ax=ax2)
    ax2.set_title("Mutation selection counts across optimized vaccines")
    ax2.set_xlabel("Mutation element")
    ax2.set_ylabel("Selection count")
    ax2.tick_params(axis="x", rotation=45)
    fig2.savefig(REPORT_IMAGES / "mutation_frequency.png")
    plt.close(fig2)


def plot_runtime(runtime, summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=runtime, x="PopulationSize", y="RunTime", hue="SampleID", marker="o", ax=axes[0])
    axes[0].set_title("Optimization runtime scaling by patient sample")
    axes[0].set_xlabel("Population size")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    sns.lineplot(data=summary, x="PopulationSize", y="mean_runtime", marker="o", ax=axes[1], color="#4C72B0", label="Mean")
    axes[1].fill_between(summary["PopulationSize"], summary["min_runtime"], summary["max_runtime"], alpha=0.2, color="#4C72B0", label="Min-max range")
    axes[1].set_title("Aggregate runtime trend across samples")
    axes[1].set_xlabel("Population size")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend()
    fig.savefig(REPORT_IMAGES / "runtime_scaling.png")
    plt.close(fig)


def plot_validation(merged):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=merged, x="selected_mutation_hits", y="cell_p_response", hue="repetition", palette="tab10", ax=axes[0], s=45)
    axes[0].set_title("Selected mutation hits vs cell response")
    axes[0].set_xlabel("Selected mutation hits in cell")
    axes[0].set_ylabel("p(response)")

    grouped = merged.groupby("selected_mutation_hits").agg(mean_response=("cell_p_response", "mean"), n_cells=("cell_ids", "count")).reset_index()
    sns.barplot(data=grouped, x="selected_mutation_hits", y="mean_response", color="#55A868", ax=axes[1])
    axes[1].set_title("Mean response by number of matched vaccine mutations")
    axes[1].set_xlabel("Selected mutation hits")
    axes[1].set_ylabel("Mean p(response)")
    fig.savefig(REPORT_IMAGES / "validation_hits_vs_response.png")
    plt.close(fig)


def main():
    ensure_dirs()
    data = load_inputs()

    audit = build_input_audit(data)
    with open(OUTPUTS / "input_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    sim_response, response_summary, overall_response = prepare_response_metrics(
        data["sim_response"], data["final_response"]
    )
    selected, selected_sets, iou_df, iou_matrix, mutation_frequency, consensus = prepare_selection_metrics(
        data["selected_elements"], data["vaccine_summary"]
    )
    per_cell, coverage_summary, overall_coverage = prepare_cell_coverage(data["cell_populations"], selected_sets)
    score_summary = summarize_score_tables(data["score_tables"], selected_sets)
    runtime, runtime_summary, runtime_scaling, selection_runtime_summary = analyze_runtime(data["runtime"], selected)
    merged = merge_main_table(per_cell, sim_response)

    headline = {
        "overall_mean_p_response": overall_response["mean_p_response"],
        "overall_median_p_response": overall_response["median_p_response"],
        "coverage_ge_1": overall_coverage["coverage_ge_1"],
        "coverage_ge_2": overall_coverage["coverage_ge_2"],
        "coverage_ge_3": overall_coverage["coverage_ge_3"],
        "mean_pairwise_iou": float(iou_df["iou"].mean()),
        "min_pairwise_iou": float(iou_df["iou"].min()),
        "max_pairwise_iou": float(iou_df["iou"].max()),
        "runtime_log10_slope": runtime_scaling["log10_slope"],
        "consensus_size": int(len(consensus)),
        "n_unique_selected_elements_across_reps": int(mutation_frequency["mutation_element"].nunique()),
    }
    with open(OUTPUTS / "headline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)

    summary_rows = []
    for metric, value in {**overall_response, **overall_coverage, **runtime_scaling}.items():
        summary_rows.append({"metric": metric, "value": value})
    summary_rows.extend([
        {"metric": "mean_pairwise_iou", "value": float(iou_df["iou"].mean())},
        {"metric": "median_pairwise_iou", "value": float(iou_df["iou"].median())},
        {"metric": "n_unique_selected_elements_across_reps", "value": int(mutation_frequency["mutation_element"].nunique())},
    ])
    summary_df = pd.DataFrame(summary_rows)

    save_tables(
        response_summary_by_rep=response_summary,
        coverage_summary_by_rep=coverage_summary,
        pairwise_iou=iou_df,
        iou_matrix=iou_matrix.reset_index().rename(columns={"index": "repetition"}),
        mutation_frequency=mutation_frequency,
        consensus_vaccine_summary=consensus,
        per_cell_mutation_coverage=per_cell,
        merged_cell_metrics=merged,
        score_summary_by_rep=score_summary,
        runtime_summary=runtime_summary,
        runtime_raw=runtime,
        selection_runtime_by_rep=selection_runtime_summary,
        summary_metrics=summary_df,
    )

    plot_data_overview(data["cell_populations"], sim_response)
    plot_response_distribution(sim_response, response_summary)
    plot_coverage(coverage_summary)
    plot_iou(iou_matrix, mutation_frequency)
    plot_runtime(runtime, runtime_summary)
    plot_validation(merged)


if __name__ == "__main__":
    main()
