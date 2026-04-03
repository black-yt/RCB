#!/usr/bin/env python3
"""Reproducible analysis of synthetic SXS catalog uncertainty datasets."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"


@dataclass
class LognormalFit:
    shape_sigma: float
    scale: float
    mu_log: float
    sigma_log: float
    ks_statistic: float
    ks_pvalue: float


def ensure_dirs() -> None:
    (OUTPUT_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        draws[i] = statistic(sample)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize_series(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy()
    median_ci = bootstrap_ci(values, np.median)
    mean_ci = bootstrap_ci(values, np.mean)
    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "median_ci_low": median_ci[0],
        "median_ci_high": median_ci[1],
        "mean_ci_low": mean_ci[0],
        "mean_ci_high": mean_ci[1],
        "geometric_mean": float(np.exp(np.mean(np.log(values)))),
    }


def fit_lognormal(series: pd.Series) -> LognormalFit:
    values = series.to_numpy()
    shape, loc, scale = stats.lognorm.fit(values, floc=0)
    if loc != 0:
        raise ValueError("Expected zero location for lognormal fit")
    mu_log = float(np.log(scale))
    sigma_log = float(shape)
    ks_stat, ks_p = stats.kstest(values, "lognorm", args=(shape, 0, scale))
    return LognormalFit(
        shape_sigma=float(shape),
        scale=float(scale),
        mu_log=mu_log,
        sigma_log=sigma_log,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
    )


def add_ecdf(ax: plt.Axes, values: np.ndarray, label: str, color: str) -> None:
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    ax.step(values, y, where="post", label=label, color=color, linewidth=2)


def plot_overview(fig6: pd.DataFrame, fig7: pd.DataFrame, fig8: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.hist(fig6["waveform_difference"], bins=np.logspace(-5.2, -1.2, 35), color="#1f77b4", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Resolution mismatch")
    ax.set_ylabel("Simulations")
    ax.set_title("Overall resolution uncertainty")

    ax = axes[1]
    mode_long = fig7.melt(var_name="mode", value_name="difference")
    sns.boxplot(
        data=mode_long,
        x="mode",
        y="difference",
        ax=ax,
        color="#e8c97a",
        showfliers=False,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Spherical-harmonic mode")
    ax.set_ylabel("Mode mismatch")
    ax.set_title("Mode-resolved uncertainty")

    ax = axes[2]
    ax.scatter(fig8["N2vsN3"], fig8["N2vsN4"], s=18, alpha=0.45, color="#8c564b", edgecolors="none")
    lims = [
        min(fig8.min().min(), 1e-6),
        max(fig8.max().max(), 5e-3),
    ]
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.5, label="Parity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("N=2 vs N=3")
    ax.set_ylabel("N=2 vs N=4")
    ax.set_title("Extrapolation-order comparison")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "data_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fig6(fig6: pd.DataFrame, fit: LognormalFit) -> None:
    values = fig6["waveform_difference"].to_numpy()
    xs = np.logspace(np.log10(values.min() * 0.8), np.log10(values.max() * 1.2), 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.hist(values, bins=np.logspace(-5.2, -1.2, 35), density=True, color="#4c72b0", alpha=0.75, label="Empirical")
    ax.plot(xs, stats.lognorm.pdf(xs, fit.shape_sigma, loc=0, scale=fit.scale), color="#dd8452", linewidth=2.5, label="Fitted log-normal")
    ax.axvline(np.median(values), color="#55a868", linestyle="--", linewidth=2, label="Median")
    ax.set_xscale("log")
    ax.set_xlabel("Waveform difference")
    ax.set_ylabel("Density")
    ax.set_title("Resolution-error distribution")
    ax.legend(frameon=True)

    ax = axes[1]
    add_ecdf(ax, values, "Empirical ECDF", "#4c72b0")
    ax.plot(xs, stats.lognorm.cdf(xs, fit.shape_sigma, loc=0, scale=fit.scale), color="#dd8452", linewidth=2.5, label="Fitted CDF")
    for thr, color in [(1e-3, "#c44e52"), (1e-2, "#8172b3")]:
        ax.axvline(thr, color=color, linestyle="--", linewidth=1.8, label=f"{thr:.0e} threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Waveform difference")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Accuracy thresholds")
    ax.legend(frameon=True, loc="lower right")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "resolution_error_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fig7(fig7: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ell = summary["ell"].to_numpy()
    ax.plot(ell, summary["median"], marker="o", color="#4c72b0", linewidth=2.5, label="Median")
    ax.fill_between(ell, summary["p25"], summary["p75"], color="#4c72b0", alpha=0.18, label="IQR")
    ax.plot(ell, summary["p95"], marker="s", color="#dd8452", linewidth=2, label="95th percentile")
    ax.set_yscale("log")
    ax.set_xlabel("Mode ell")
    ax.set_ylabel("Waveform difference")
    ax.set_title("Uncertainty increases with mode order")
    ax.legend(frameon=True)

    ax = axes[1]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", square=True, cbar_kws={"label": "Spearman rho"}, ax=ax)
    ax.set_title("Cross-mode correlation")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "mode_dependence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    mode_long = fig7.melt(var_name="mode", value_name="difference")
    sns.violinplot(
        data=mode_long,
        x="mode",
        y="difference",
        inner="quartile",
        cut=0,
        density_norm="width",
        ax=ax,
        color="#ccb974",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Spherical-harmonic mode")
    ax.set_ylabel("Waveform difference")
    ax.set_title("Mode-resolved scatter broadens at high ell")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "mode_violin.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fig8(fig8: pd.DataFrame, ratio_summary: dict[str, float]) -> None:
    ratio = fig8["N2vsN4"] / fig8["N2vsN3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(fig8["N2vsN3"], fig8["N2vsN4"], s=18, alpha=0.45, color="#8c564b", edgecolors="none")
    lims = [min(fig8.min().min(), 1e-6), max(fig8.max().max(), 5e-3)]
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("N=2 vs N=3")
    ax.set_ylabel("N=2 vs N=4")
    ax.set_title("Higher-order extrapolation differences are larger")

    ax = axes[1]
    ax.hist(ratio, bins=np.logspace(np.log10(ratio.min()), np.log10(ratio.max()), 40), color="#55a868", alpha=0.8)
    ax.axvline(ratio_summary["median"], color="black", linestyle="--", linewidth=2, label="Median ratio")
    ax.set_xscale("log")
    ax.set_xlabel("(N=2 vs N=4) / (N=2 vs N=3)")
    ax.set_ylabel("Simulations")
    ax.set_title("Paired amplification factor")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "extrapolation_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation(fig6: pd.DataFrame, fig8: pd.DataFrame, fit6: LognormalFit) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    values = np.sort(fig6["waveform_difference"].to_numpy())
    probs = (np.arange(1, len(values) + 1) - 0.5) / len(values)
    theoretical = stats.lognorm.ppf(probs, fit6.shape_sigma, loc=0, scale=fit6.scale)
    ax.scatter(theoretical, values, s=15, alpha=0.5, color="#4c72b0")
    bounds = [min(theoretical.min(), values.min()), max(theoretical.max(), values.max())]
    ax.plot(bounds, bounds, linestyle="--", color="black", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fitted log-normal quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("QQ check: overall resolution errors")

    ax = axes[1]
    log_n3 = np.log10(fig8["N2vsN3"])
    log_n4 = np.log10(fig8["N2vsN4"])
    delta = log_n4 - log_n3
    mean_log = 0.5 * (log_n4 + log_n3)
    ax.scatter(mean_log, delta, s=18, alpha=0.45, color="#c44e52", edgecolors="none")
    mean_delta = float(delta.mean())
    std_delta = float(delta.std(ddof=1))
    for y, style, label in [
        (mean_delta, "-", "Mean"),
        (mean_delta + 1.96 * std_delta, "--", "95% limits"),
        (mean_delta - 1.96 * std_delta, "--", None),
    ]:
        ax.axhline(y, color="black", linestyle=style, linewidth=1.5, label=label)
    ax.set_xlabel("Mean log10 difference")
    ax.set_ylabel("log10(N2vsN4) - log10(N2vsN3)")
    ax.set_title("Bland-Altman style paired validation")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "validation_checks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    fig6 = pd.read_csv(DATA_DIR / "fig6_data.csv")
    fig7 = pd.read_csv(DATA_DIR / "fig7_data.csv")
    fig8 = pd.read_csv(DATA_DIR / "fig8_data.csv")

    summary = {
        "fig6": summarize_series(fig6["waveform_difference"]),
        "fig8_N2vsN3": summarize_series(fig8["N2vsN3"]),
        "fig8_N2vsN4": summarize_series(fig8["N2vsN4"]),
    }

    fit6 = fit_lognormal(fig6["waveform_difference"])
    fit8_n3 = fit_lognormal(fig8["N2vsN3"])
    fit8_n4 = fit_lognormal(fig8["N2vsN4"])

    fig6_thresholds = pd.DataFrame(
        {
            "threshold": [1e-4, 3e-4, 1e-3, 1e-2],
            "fraction_below": [
                float((fig6["waveform_difference"] < thr).mean())
                for thr in [1e-4, 3e-4, 1e-3, 1e-2]
            ],
        }
    )

    fig7_mode_summary = []
    for col in fig7.columns:
        ell = int(col.replace("ell", ""))
        s = summarize_series(fig7[col])
        s["ell"] = ell
        s["fraction_below_1e3"] = float((fig7[col] < 1e-3).mean())
        s["fraction_below_1e2"] = float((fig7[col] < 1e-2).mean())
        fig7_mode_summary.append(s)
    fig7_mode_summary = pd.DataFrame(fig7_mode_summary).sort_values("ell").reset_index(drop=True)

    log_median_slope, log_median_intercept, r_value, p_value, stderr = stats.linregress(
        fig7_mode_summary["ell"].to_numpy(),
        np.log10(fig7_mode_summary["median"].to_numpy()),
    )
    spearman_mode = stats.spearmanr(fig7_mode_summary["ell"], fig7_mode_summary["median"])
    fig7_corr = fig7.corr(method="spearman")

    ratio = fig8["N2vsN4"] / fig8["N2vsN3"]
    log_ratio = np.log10(fig8["N2vsN4"]) - np.log10(fig8["N2vsN3"])
    wilcoxon = stats.wilcoxon(log_ratio, alternative="greater", zero_method="wilcox")
    ratio_summary = summarize_series(ratio.rename("ratio"))
    ratio_summary["fraction_gt_1"] = float((ratio > 1).mean())
    ratio_summary["spearman_n3_n4"] = float(stats.spearmanr(fig8["N2vsN3"], fig8["N2vsN4"]).statistic)
    ratio_summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
    ratio_summary["wilcoxon_pvalue"] = float(wilcoxon.pvalue)

    model_fit_table = pd.DataFrame(
        [
            {"dataset": "fig6_resolution", **asdict(fit6)},
            {"dataset": "fig8_N2vsN3", **asdict(fit8_n3)},
            {"dataset": "fig8_N2vsN4", **asdict(fit8_n4)},
        ]
    )

    trend_summary = {
        "log10_median_vs_ell_slope": float(log_median_slope),
        "log10_median_vs_ell_intercept": float(log_median_intercept),
        "linear_r_value": float(r_value),
        "linear_p_value": float(p_value),
        "linear_stderr": float(stderr),
        "spearman_rho": float(spearman_mode.statistic),
        "spearman_p_value": float(spearman_mode.pvalue),
    }

    summary["fig6_thresholds"] = fig6_thresholds.set_index("threshold")["fraction_below"].to_dict()
    summary["fig7_trend"] = trend_summary
    summary["fig8_ratio"] = ratio_summary
    summary["lognormal_fits"] = {
        "fig6": asdict(fit6),
        "fig8_N2vsN3": asdict(fit8_n3),
        "fig8_N2vsN4": asdict(fit8_n4),
    }

    plot_overview(fig6, fig7, fig8)
    plot_fig6(fig6, fit6)
    plot_fig7(fig7, fig7_mode_summary, fig7_corr)
    plot_fig8(fig8, ratio_summary)
    plot_validation(fig6, fig8, fit6)

    fig6_thresholds.to_csv(OUTPUT_DIR / "fig6_thresholds.csv", index=False)
    fig7_mode_summary.to_csv(OUTPUT_DIR / "fig7_mode_summary.csv", index=False)
    fig7_corr.to_csv(OUTPUT_DIR / "fig7_spearman_correlation.csv")
    model_fit_table.to_csv(OUTPUT_DIR / "lognormal_fit_summary.csv", index=False)

    with open(OUTPUT_DIR / "summary_statistics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    report_table = pd.DataFrame(
        [
            ["Resolution median mismatch", summary["fig6"]["median"]],
            ["Resolution 95th percentile", summary["fig6"]["p95"]],
            ["Fraction below 1e-3", summary["fig6_thresholds"][1e-3]],
            ["Mode-ell=2 median", float(fig7_mode_summary.loc[fig7_mode_summary["ell"] == 2, "median"].iloc[0])],
            ["Mode-ell=8 median", float(fig7_mode_summary.loc[fig7_mode_summary["ell"] == 8, "median"].iloc[0])],
            ["Median extrapolation ratio N4/N3", ratio_summary["median"]],
            ["Fraction N4 > N3", ratio_summary["fraction_gt_1"]],
        ],
        columns=["metric", "value"],
    )
    report_table.to_csv(OUTPUT_DIR / "report_key_metrics.csv", index=False)

    print("Analysis complete.")
    print(f"Saved figures to {IMAGE_DIR}")
    print(f"Saved tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
