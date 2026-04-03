#!/usr/bin/env python3
"""Main analysis entry point for the SXS-style waveform accuracy study.

This script analyzes three synthetic numerical-relativity quality-control datasets:
- fig6_data.csv: highest-resolution waveform mismatch distribution
- fig7_data.csv: mode-resolved mismatch distributions for ell=2..8
- fig8_data.csv: extrapolation-order discrepancy distributions

Outputs:
- CSV summaries in outputs/
- JSON metadata/interpretation aids in outputs/
- PNG figures in report/images/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover - fallback if seaborn unavailable
    sns = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def read_data() -> Dict[str, pd.DataFrame]:
    fig6 = pd.read_csv(DATA_DIR / "fig6_data.csv")
    fig7 = pd.read_csv(DATA_DIR / "fig7_data.csv")
    fig8 = pd.read_csv(DATA_DIR / "fig8_data.csv")
    return {"fig6": fig6, "fig7": fig7, "fig8": fig8}


def validate_data(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    expected = {
        "fig6": {"rows": 1500, "cols": 1, "columns": ["waveform_difference"]},
        "fig7": {"rows": 1500, "cols": 7, "columns": [f"ell{i}" for i in range(2, 9)]},
        "fig8": {"rows": 1200, "cols": 2, "columns": ["N2vsN3", "N2vsN4"]},
    }

    for name, df in frames.items():
        exp = expected[name]
        records.append(
            {
                "dataset": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "expected_rows": exp["rows"],
                "expected_cols": exp["cols"],
                "shape_matches_expected": bool(df.shape == (exp["rows"], exp["cols"])),
                "columns_match_expected": list(df.columns) == exp["columns"],
                "min_value": float(df.min().min()),
                "max_value": float(df.max().max()),
                "nonpositive_entries": int((df <= 0).sum().sum()),
                "missing_entries": int(df.isna().sum().sum()),
            }
        )
    validation = pd.DataFrame.from_records(records)
    validation.to_csv(OUTPUT_DIR / "data_validation_summary.csv", index=False)
    return validation


def summarize_series(name: str, values: pd.Series) -> Dict[str, float]:
    arr = values.to_numpy(dtype=float)
    log10_arr = np.log10(arr)
    q = np.quantile(arr, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "name": name,
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "geometric_mean": float(10 ** np.mean(log10_arr)),
        "log10_mean": float(np.mean(log10_arr)),
        "log10_std": float(np.std(log10_arr, ddof=1)),
        "fraction_below_1e-3": float(np.mean(arr < 1e-3)),
        "fraction_below_1e-4": float(np.mean(arr < 1e-4)),
        "fraction_above_1e-2": float(np.mean(arr > 1e-2)),
        "fraction_above_1e-1": float(np.mean(arr > 1e-1)),
    }


def create_overall_summaries(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    fig6_summary = pd.DataFrame([summarize_series("resolution_mismatch", frames["fig6"].iloc[:, 0])])
    fig6_summary.to_csv(OUTPUT_DIR / "fig6_summary_statistics.csv", index=False)

    fig7_records = [summarize_series(col, frames["fig7"][col]) for col in frames["fig7"].columns]
    fig7_summary = pd.DataFrame(fig7_records)
    fig7_summary["ell"] = fig7_summary["name"].str.replace("ell", "", regex=False).astype(int)
    fig7_summary = fig7_summary.sort_values("ell")
    fig7_summary.to_csv(OUTPUT_DIR / "fig7_mode_summary_statistics.csv", index=False)

    fig8_records = [summarize_series(col, frames["fig8"][col]) for col in frames["fig8"].columns]
    fig8_summary = pd.DataFrame(fig8_records)
    fig8_summary.to_csv(OUTPUT_DIR / "fig8_extrapolation_summary_statistics.csv", index=False)

    return {"fig6": fig6_summary, "fig7": fig7_summary, "fig8": fig8_summary}


def build_mode_trend_table(fig7_summary: pd.DataFrame) -> pd.DataFrame:
    trend = fig7_summary[["ell", "median", "q95", "q99", "log10_std", "fraction_above_1e-2"]].copy()
    trend["median_ratio_to_ell2"] = trend["median"] / trend.loc[trend["ell"] == 2, "median"].iloc[0]
    trend["q95_ratio_to_ell2"] = trend["q95"] / trend.loc[trend["ell"] == 2, "q95"].iloc[0]
    trend.to_csv(OUTPUT_DIR / "mode_trend_analysis.csv", index=False)
    return trend


def build_extrapolation_comparison(fig8: pd.DataFrame) -> pd.DataFrame:
    ratio = fig8["N2vsN4"] / fig8["N2vsN3"]
    log_diff = np.log10(fig8["N2vsN4"]) - np.log10(fig8["N2vsN3"])
    comparison = pd.DataFrame(
        {
            "metric": [
                "median_ratio_N2vsN4_to_N2vsN3",
                "mean_ratio_N2vsN4_to_N2vsN3",
                "fraction_N2vsN4_gt_N2vsN3",
                "fraction_N2vsN4_gt_2x_N2vsN3",
                "median_log10_difference",
                "mean_log10_difference",
                "spearman_like_rank_corr",
            ],
            "value": [
                float(np.median(ratio)),
                float(np.mean(ratio)),
                float(np.mean(fig8["N2vsN4"] > fig8["N2vsN3"])),
                float(np.mean(fig8["N2vsN4"] > 2.0 * fig8["N2vsN3"])),
                float(np.median(log_diff)),
                float(np.mean(log_diff)),
                float(pd.Series(fig8["N2vsN3"]).rank().corr(pd.Series(fig8["N2vsN4"]).rank())),
            ],
        }
    )
    comparison.to_csv(OUTPUT_DIR / "extrapolation_order_comparison.csv", index=False)
    return comparison


def save_key_findings(fig6_summary: pd.DataFrame, fig7_summary: pd.DataFrame, fig8_summary: pd.DataFrame) -> None:
    fig6_row = fig6_summary.iloc[0]
    ell2 = fig7_summary.loc[fig7_summary["ell"] == 2].iloc[0]
    ell8 = fig7_summary.loc[fig7_summary["ell"] == 8].iloc[0]
    n23 = fig8_summary.loc[fig8_summary["name"] == "N2vsN3"].iloc[0]
    n24 = fig8_summary.loc[fig8_summary["name"] == "N2vsN4"].iloc[0]

    findings = {
        "resolution_distribution": {
            "median": float(fig6_row["median"]),
            "q95": float(fig6_row["q95"]),
            "fraction_below_1e-3": float(fig6_row["fraction_below_1e-3"]),
            "fraction_above_1e-2": float(fig6_row["fraction_above_1e-2"]),
        },
        "mode_growth": {
            "ell2_median": float(ell2["median"]),
            "ell8_median": float(ell8["median"]),
            "ell8_to_ell2_median_ratio": float(ell8["median"] / ell2["median"]),
            "ell2_q95": float(ell2["q95"]),
            "ell8_q95": float(ell8["q95"]),
        },
        "extrapolation_behavior": {
            "N2vsN3_median": float(n23["median"]),
            "N2vsN4_median": float(n24["median"]),
            "median_ratio": float(n24["median"] / n23["median"]),
            "N2vsN3_fraction_below_1e-4": float(n23["fraction_below_1e-4"]),
            "N2vsN4_fraction_below_1e-4": float(n24["fraction_below_1e-4"]),
        },
        "limitations": [
            "The provided inputs are synthetic summary datasets rather than full binary-parameter metadata or raw waveforms.",
            "Interpretation is therefore limited to statistical error characterization, not direct astrophysical dependence on mass ratio, spin, or eccentricity.",
        ],
    }
    with open(OUTPUT_DIR / "key_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)


def _histplot(ax, data: Iterable[float], bins: int, color: str, label: str | None = None) -> None:
    if sns is not None:
        sns.histplot(data, bins=bins, stat="density", kde=True, ax=ax, color=color, label=label, alpha=0.35)
    else:
        ax.hist(list(data), bins=bins, density=True, color=color, alpha=0.4, label=label)


def plot_resolution_distribution(fig6: pd.DataFrame) -> None:
    values = fig6["waveform_difference"].to_numpy()
    log_values = np.log10(values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    _histplot(axes[0], values, bins=40, color="#4C72B0")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Waveform difference")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of highest-resolution waveform differences")
    for thr in [1e-4, 1e-3, 1e-2]:
        axes[0].axvline(thr, color="black", linestyle="--", linewidth=1)

    _histplot(axes[1], log_values, bins=40, color="#55A868")
    axes[1].set_xlabel(r"$\log_{10}$(waveform difference)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log-scaled view of resolution-mismatch distribution")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_resolution_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_mode_comparison(fig7: pd.DataFrame, fig7_summary: pd.DataFrame) -> None:
    long_df = fig7.melt(var_name="mode", value_name="waveform_difference")
    long_df["ell"] = long_df["mode"].str.replace("ell", "", regex=False).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    if sns is not None:
        sns.boxplot(data=long_df, x="ell", y="waveform_difference", ax=axes[0], color="#8172B3", showfliers=False)
        sns.stripplot(
            data=long_df.sample(min(1200, len(long_df)), random_state=7),
            x="ell",
            y="waveform_difference",
            ax=axes[0],
            color="black",
            alpha=0.15,
            size=2,
        )
    else:
        grouped = [long_df.loc[long_df["ell"] == ell, "waveform_difference"].values for ell in sorted(long_df["ell"].unique())]
        axes[0].boxplot(grouped, showfliers=False)
        axes[0].set_xticklabels(sorted(long_df["ell"].unique()))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Spherical-harmonic mode ell")
    axes[0].set_ylabel("Waveform difference")
    axes[0].set_title("Mode-resolved waveform-difference distributions")

    axes[1].plot(fig7_summary["ell"], fig7_summary["median"], marker="o", linewidth=2, label="Median")
    axes[1].plot(fig7_summary["ell"], fig7_summary["q95"], marker="s", linewidth=2, label="95th percentile")
    axes[1].fill_between(fig7_summary["ell"], fig7_summary["q25"], fig7_summary["q75"], alpha=0.25, label="IQR")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Spherical-harmonic mode ell")
    axes[1].set_ylabel("Waveform difference")
    axes[1].set_title("Growth of typical and tail errors with mode index")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_mode_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_extrapolation_comparison(fig8: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    _histplot(axes[0], fig8["N2vsN3"], bins=35, color="#64B5CD", label="N=2 vs N=3")
    _histplot(axes[0], fig8["N2vsN4"], bins=35, color="#C44E52", label="N=2 vs N=4")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Extrapolation-order waveform difference")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Extrapolation-order discrepancy distributions")
    axes[0].legend()

    axes[1].scatter(fig8["N2vsN3"], fig8["N2vsN4"], s=18, alpha=0.45, color="#4C72B0", edgecolor="none")
    min_val = float(min(fig8.min()))
    max_val = float(max(fig8.max()))
    line = np.logspace(np.log10(min_val), np.log10(max_val), 200)
    axes[1].plot(line, line, linestyle="--", color="black", linewidth=1, label="Equality")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N=2 vs N=3")
    axes[1].set_ylabel("N=2 vs N=4")
    axes[1].set_title("Pairwise comparison of extrapolation discrepancies")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_extrapolation_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_catalog_overview(fig6: pd.DataFrame, fig7_summary: pd.DataFrame, fig8_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    sorted_values = np.sort(fig6["waveform_difference"].to_numpy())
    frac = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    axes[0].plot(sorted_values, frac, color="#4C72B0", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Resolution mismatch")
    axes[0].set_ylabel("Cumulative fraction of simulations")
    axes[0].set_title("Catalog-level cumulative accuracy")

    axes[1].bar(fig7_summary["ell"], fig7_summary["fraction_above_1e-2"], color="#DD8452")
    axes[1].set_xlabel("Mode ell")
    axes[1].set_ylabel("Fraction with mismatch > 1e-2")
    axes[1].set_title("High-error tail by harmonic mode")

    axes[2].bar(fig8_summary["name"], fig8_summary["median"], color=["#55A868", "#C44E52"])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Median extrapolation difference")
    axes[2].set_title("Typical extrapolation discrepancies")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_catalog_overview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    setup_style()

    frames = read_data()
    validate_data(frames)
    summaries = create_overall_summaries(frames)
    build_mode_trend_table(summaries["fig7"])
    build_extrapolation_comparison(frames["fig8"])
    save_key_findings(summaries["fig6"], summaries["fig7"], summaries["fig8"])

    plot_resolution_distribution(frames["fig6"])
    plot_mode_comparison(frames["fig7"], summaries["fig7"])
    plot_extrapolation_comparison(frames["fig8"])
    plot_catalog_overview(frames["fig6"], summaries["fig7"], summaries["fig8"])

    run_manifest = {
        "inputs": {
            "fig6": "data/fig6_data.csv",
            "fig7": "data/fig7_data.csv",
            "fig8": "data/fig8_data.csv",
        },
        "outputs": [
            "outputs/data_validation_summary.csv",
            "outputs/fig6_summary_statistics.csv",
            "outputs/fig7_mode_summary_statistics.csv",
            "outputs/fig8_extrapolation_summary_statistics.csv",
            "outputs/mode_trend_analysis.csv",
            "outputs/extrapolation_order_comparison.csv",
            "outputs/key_findings.json",
        ],
        "figures": [
            "report/images/figure_resolution_distribution.png",
            "report/images/figure_mode_comparison.png",
            "report/images/figure_extrapolation_comparison.png",
            "report/images/figure_catalog_overview.png",
        ],
    }
    with open(OUTPUT_DIR / "analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
