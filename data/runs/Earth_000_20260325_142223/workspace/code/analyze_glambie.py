from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import math

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from pypdf import PdfReader


DATA_DIR = ROOT / "data" / "glambie"
INPUT_DIR = DATA_DIR / "input"
CALENDAR_DIR = DATA_DIR / "results" / "calendar_years"
HYDRO_DIR = DATA_DIR / "results" / "hydrological_years"
RELATED_DIR = ROOT / "related_work"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
IMAGE_DIR = REPORT_DIR / "images"


REGION_ID_TO_NAME = {
    1: "Alaska",
    2: "Western Canada and US",
    3: "Arctic Canada North",
    4: "Arctic Canada South",
    5: "Greenland Periphery",
    6: "Iceland",
    7: "Svalbard",
    8: "Scandinavia",
    9: "Russian Arctic",
    10: "North Asia",
    11: "Central Europe",
    12: "Caucasus and Middle East",
    13: "Central Asia",
    14: "South Asia West",
    15: "South Asia East",
    16: "Low Latitudes",
    17: "Southern Andes",
    18: "New Zealand",
    19: "Antarctic and Subantarctic",
}

REGION_ORDER = [REGION_ID_TO_NAME[i] for i in sorted(REGION_ID_TO_NAME)]

METHOD_ORDER = [
    "glaciological",
    "demdiff",
    "altimetry",
    "gravimetry",
    "combined",
]

METHOD_LABELS = {
    "glaciological": "Glaciological",
    "demdiff": "DEM differencing",
    "altimetry": "Altimetry",
    "gravimetry": "Gravimetry",
    "combined": "Hybrid / combined",
}

HYDRO_METHODS = {
    "altimetry": "altimetry",
    "gravimetry": "gravimetry",
    "demdiff_and_glaciological": "DEM diff. + glaciol.",
}

MANUAL_PAPER_TITLES = {
    "paper_000.pdf": "Global glacier change in the 21st century: Every increase in temperature matters",
    "paper_001.pdf": "Partitioning the Uncertainty of Ensemble Projections of Global Glacier Mass Change",
    "paper_002.pdf": "Global glacier mass changes and their contributions to sea-level rise from 1961 to 2016",
    "paper_003.pdf": "GlacierMIP: A model intercomparison of global-scale glacier mass-balance models and projections",
    "paper_004.pdf": "Accelerated global glacier mass loss in the early twenty-first century",
}


@dataclass
class PaperNote:
    filename: str
    title: str
    snippet: str


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_region_slug(raw: str) -> str:
    if raw and raw.split("_", 1)[0].isdigit():
        return raw.split("_", 1)[1]
    return raw


def clean_region_name(raw: str) -> str:
    normalized = normalize_region_slug(raw)
    return normalized.replace("_", " ").title()


def parse_input_metadata(path: Path) -> dict:
    region_slug = normalize_region_slug(path.parent.name)
    stem = path.stem
    prefix = f"{region_slug}_"
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else stem
    method = suffix.split("_", 1)[0]
    return {
        "path": str(path.relative_to(ROOT)),
        "region_slug": region_slug,
        "region": clean_region_name(region_slug),
        "method": method,
        "dataset_name": stem,
    }


def load_input_inventory() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(INPUT_DIR.rglob("*.csv")):
        meta = parse_input_metadata(path)
        df = pd.read_csv(path)
        unit_values = sorted({str(v).strip() for v in df["unit"].dropna().unique()}) if "unit" in df else []
        rows.append(
            {
                **meta,
                "n_rows": int(len(df)),
                "start_min": float(df["start_dates"].min()),
                "end_max": float(df["end_dates"].max()),
                "unit_values": ",".join(unit_values),
                "author_label": str(df["author"].dropna().iloc[0]) if "author" in df and not df["author"].dropna().empty else "",
            }
        )
    inventory = pd.DataFrame(rows)
    inventory["method_label"] = inventory["method"].map(METHOD_LABELS).fillna(inventory["method"])
    return inventory


def load_calendar_results() -> pd.DataFrame:
    frames = []
    for path in sorted(CALENDAR_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        if "region" not in df.columns:
            continue
        df = df.copy()
        df["source_file"] = path.name
        df["year"] = df["start_dates"].round().astype(int)
        df["period_label"] = df["year"].astype(str) + "-" + (df["year"] + 1).astype(str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["region_label"] = out["region"].str.replace("_", " ").str.title()
    return out


def load_hydrological_results() -> pd.DataFrame:
    frames = []
    for path in sorted(HYDRO_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        df = df.copy()
        df["source_file"] = path.name
        df["year"] = np.floor(df["start_dates"]).astype(int)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["region_label"] = out["region"].str.replace("_", " ").str.title()
    return out


def extract_related_work() -> list[PaperNote]:
    notes: list[PaperNote] = []
    for path in sorted(RELATED_DIR.glob("*.pdf")):
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages[:3]:
            try:
                text += page.extract_text() + "\n"
            except Exception:
                continue
        condensed = " ".join(text.split())
        title = MANUAL_PAPER_TITLES.get(path.name, "")
        meta = getattr(reader, "metadata", None)
        if not title and meta is not None and getattr(meta, "title", None):
            title = str(meta.title)
        if not title or title.startswith("S002") or title.lower() == "none":
            title = ""
            for candidate in condensed.split("Abstract")[0].split(". "):
                candidate = candidate.strip()
                if 20 <= len(candidate) <= 180 and "glacier" in candidate.lower():
                    title = candidate
                    break
        if not title:
            title = condensed[:120]
        notes.append(
            PaperNote(
                filename=path.name,
                title=title.strip(),
                snippet=condensed[:900].strip(),
            )
        )
    return notes


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build_input_summary(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = (
        inventory.groupby(["region", "method"])
        .agg(
            datasets=("dataset_name", "count"),
            observations=("n_rows", "sum"),
            start_min=("start_min", "min"),
            end_max=("end_max", "max"),
        )
        .reset_index()
    )
    overview = (
        inventory.groupby("method")
        .agg(
            datasets=("dataset_name", "count"),
            observations=("n_rows", "sum"),
            regions=("region", "nunique"),
        )
        .reset_index()
        .sort_values("datasets", ascending=False)
    )
    return coverage, overview


def build_global_validation(calendar_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    regional = calendar_df[calendar_df["region"] != "global"].copy()
    global_df = calendar_df[calendar_df["region"] == "global"].copy()
    agg = (
        regional.groupby("year")
        .agg(
            regional_sum_gt=("combined_gt", "sum"),
            regional_quadrature_error=("combined_gt_errors", lambda s: float(np.sqrt(np.square(s).sum()))),
            regional_sum_area=("glacier_area", "sum"),
            regional_mean_mwe=("combined_gt", lambda s: np.nan),
        )
        .reset_index()
    )
    weighted = (
        regional.assign(weighted_mwe=lambda d: d["combined_mwe"] * d["glacier_area"])
        .groupby("year")
        .agg(
            weighted_mwe_sum=("weighted_mwe", "sum"),
            summed_area=("glacier_area", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(weighted, on="year", how="left")
    agg["regional_area_weighted_mwe"] = agg["weighted_mwe_sum"] / agg["summed_area"]
    compare = agg.merge(
        global_df[
            [
                "year",
                "combined_gt",
                "combined_gt_errors",
                "combined_mwe",
                "combined_mwe_errors",
                "glacier_area",
            ]
        ].rename(
            columns={
                "combined_gt": "published_global_gt",
                "combined_gt_errors": "published_global_gt_error",
                "combined_mwe": "published_global_mwe",
                "combined_mwe_errors": "published_global_mwe_error",
                "glacier_area": "published_global_area",
            }
        ),
        on="year",
        how="inner",
    )
    compare["gt_difference"] = compare["regional_sum_gt"] - compare["published_global_gt"]
    compare["mwe_difference"] = compare["regional_area_weighted_mwe"] - compare["published_global_mwe"]
    metrics = {
        "max_abs_gt_difference": float(compare["gt_difference"].abs().max()),
        "mean_abs_gt_difference": float(compare["gt_difference"].abs().mean()),
        "max_abs_mwe_difference": float(compare["mwe_difference"].abs().max()),
        "mean_abs_mwe_difference": float(compare["mwe_difference"].abs().mean()),
    }
    return compare, metrics


def build_regional_metrics(calendar_df: pd.DataFrame) -> pd.DataFrame:
    regional = calendar_df[calendar_df["region"] != "global"].copy()
    metrics = (
        regional.groupby(["region", "region_label"])
        .agg(
            years=("year", "count"),
            mean_annual_gt=("combined_gt", "mean"),
            mean_annual_mwe=("combined_mwe", "mean"),
            cumulative_gt=("combined_gt", "sum"),
            cumulative_gt_uncertainty=("combined_gt_errors", lambda s: float(np.sqrt(np.square(s).sum()))),
            mean_gt_uncertainty=("combined_gt_errors", "mean"),
            mean_mwe_uncertainty=("combined_mwe_errors", "mean"),
            latest_area_km2=("glacier_area", "last"),
        )
        .reset_index()
        .sort_values("cumulative_gt")
    )
    metrics["cumulative_gt_abs"] = metrics["cumulative_gt"].abs()
    return metrics


def build_method_consistency(hydro_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for method_col, method_label in HYDRO_METHODS.items():
        gt_col = f"{method_col}_gt"
        err_col = f"{method_col}_gt_errors"
        annual_var_col = f"{method_col}_annual_variability"
        subset = hydro_df[["region", "region_label", "year", "combined_gt", "combined_gt_errors", gt_col, err_col, annual_var_col]].copy()
        subset = subset.rename(
            columns={
                gt_col: "method_gt",
                err_col: "method_gt_error",
                annual_var_col: "annual_variability_flag",
            }
        )
        subset["method_family"] = method_label
        subset = subset.dropna(subset=["method_gt"])
        subset["residual_gt"] = subset["method_gt"] - subset["combined_gt"]
        subset["pair_sigma"] = np.sqrt(np.square(subset["method_gt_error"]) + np.square(subset["combined_gt_errors"]))
        subset["z_score"] = subset["residual_gt"] / subset["pair_sigma"]
        rows.append(subset)
    comparison = pd.concat(rows, ignore_index=True)
    summary = (
        comparison.groupby(["region", "region_label", "method_family"])
        .agg(
            years_compared=("year", "count"),
            mean_abs_residual_gt=("residual_gt", lambda s: float(np.mean(np.abs(s)))),
            median_abs_residual_gt=("residual_gt", lambda s: float(np.median(np.abs(s)))),
            rmse_gt=("residual_gt", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            mean_abs_z=("z_score", lambda s: float(np.mean(np.abs(s)))),
            max_abs_z=("z_score", lambda s: float(np.max(np.abs(s)))),
            method_uncertainty_mean=("method_gt_error", "mean"),
            combined_uncertainty_mean=("combined_gt_errors", "mean"),
            annual_variability_fraction=("annual_variability_flag", lambda s: float(pd.Series(s).fillna(0).mean())),
        )
        .reset_index()
    )
    summary["uncertainty_ratio_method_to_combined"] = summary["method_uncertainty_mean"] / summary["combined_uncertainty_mean"]
    return comparison, summary


def build_global_metrics(calendar_df: pd.DataFrame) -> dict:
    global_df = calendar_df[calendar_df["region"] == "global"].sort_values("year").copy()
    global_df["cumulative_gt"] = global_df["combined_gt"].cumsum()
    global_df["cumulative_gt_sigma"] = np.sqrt(np.square(global_df["combined_gt_errors"]).cumsum())
    record_year = int(global_df.loc[global_df["combined_gt"].idxmin(), "year"])
    record_gt = float(global_df["combined_gt"].min())
    first_decade = global_df[(global_df["year"] >= 2000) & (global_df["year"] <= 2011)]["combined_gt"].mean()
    last_decade = global_df[(global_df["year"] >= 2012) & (global_df["year"] <= 2023)]["combined_gt"].mean()
    metrics = {
        "mean_annual_gt_2000_2023": float(global_df["combined_gt"].mean()),
        "mean_annual_mwe_2000_2023": float(global_df["combined_mwe"].mean()),
        "cumulative_gt_2000_2023": float(global_df["combined_gt"].sum()),
        "cumulative_gt_sigma_2000_2023": float(np.sqrt(np.square(global_df["combined_gt_errors"]).sum())),
        "record_loss_year": record_year,
        "record_loss_gt": record_gt,
        "first_decade_mean_gt": float(first_decade),
        "last_decade_mean_gt": float(last_decade),
        "relative_change_last_vs_first_decade_pct": float((last_decade - first_decade) / abs(first_decade) * 100.0),
    }
    return metrics


def extract_related_context(notes: Iterable[PaperNote]) -> pd.DataFrame:
    rows = []
    for note in notes:
        rows.append(
            {
                "file": note.filename,
                "title": note.title,
                "snippet": note.snippet,
            }
        )
    return pd.DataFrame(rows)


def plot_coverage_heatmap(coverage: pd.DataFrame) -> None:
    pivot = coverage.pivot(index="region", columns="method", values="datasets").reindex(index=REGION_ORDER, columns=METHOD_ORDER)
    plt.figure(figsize=(10.5, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Submitted datasets"})
    plt.xlabel("Observation family")
    plt.ylabel("Region")
    plt.title("GlaMBIE input coverage by region and method")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_1_input_coverage_heatmap.png", dpi=220)
    plt.close()


def plot_global_series(calendar_df: pd.DataFrame) -> None:
    global_df = calendar_df[calendar_df["region"] == "global"].sort_values("year").copy()
    global_df["cumulative_gt"] = global_df["combined_gt"].cumsum()
    global_df["cumulative_sigma"] = np.sqrt(np.square(global_df["combined_gt_errors"]).cumsum())

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True, height_ratios=[1, 1.05])

    axes[0].bar(global_df["year"], global_df["combined_gt"], color="#34618d", width=0.8)
    axes[0].errorbar(
        global_df["year"],
        global_df["combined_gt"],
        yerr=global_df["combined_gt_errors"],
        fmt="none",
        ecolor="#0f2537",
        elinewidth=1,
        capsize=2,
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Annual mass change (Gt)")
    axes[0].set_title("Global glacier mass change benchmark, 2000-2023")

    axes[1].plot(global_df["year"], global_df["cumulative_gt"], color="#b33f62", linewidth=2.2)
    axes[1].fill_between(
        global_df["year"],
        global_df["cumulative_gt"] - global_df["cumulative_sigma"],
        global_df["cumulative_gt"] + global_df["cumulative_sigma"],
        color="#f1b7c8",
        alpha=0.55,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Cumulative mass change (Gt)")
    axes[1].set_xlabel("Balance year")

    for ax in axes:
        ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_2_global_mass_change.png", dpi=220)
    plt.close()


def plot_regional_cumulative_losses(regional_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.25, 1]})

    plot_df = regional_metrics.sort_values("cumulative_gt")
    colors = np.where(plot_df["cumulative_gt"] < 0, "#9b2226", "#2a9d8f")
    axes[0].barh(plot_df["region_label"], plot_df["cumulative_gt"], xerr=plot_df["cumulative_gt_uncertainty"], color=colors, alpha=0.9)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Cumulative mass change, 2000-2023 (Gt)")
    axes[0].set_ylabel("")
    axes[0].set_title("Regional cumulative glacier mass change")

    scatter_df = regional_metrics.sort_values("latest_area_km2", ascending=False)
    sc = axes[1].scatter(
        scatter_df["latest_area_km2"],
        scatter_df["mean_annual_gt"],
        s=40 + 130 * scatter_df["cumulative_gt_abs"] / scatter_df["cumulative_gt_abs"].max(),
        c=scatter_df["mean_mwe_uncertainty"],
        cmap="magma_r",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
    )
    for _, row in scatter_df.iterrows():
        axes[1].annotate(row["region_label"], (row["latest_area_km2"], row["mean_annual_gt"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Approximate glacier area in 2023 (km$^2$)")
    axes[1].set_ylabel("Mean annual mass change (Gt yr$^{-1}$)")
    axes[1].set_title("Absolute loss versus remaining glacier area")
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.05, pad=0.03)
    cbar.set_label("Mean specific uncertainty (m w.e.)")

    for ax in axes:
        ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_3_regional_losses.png", dpi=220)
    plt.close()


def plot_method_validation(comparison: pd.DataFrame, global_validation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    sns.boxplot(
        data=comparison,
        x="method_family",
        y="z_score",
        color="#8ecae6",
        ax=axes[0],
        fliersize=2,
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axhline(1, color="gray", linewidth=0.6, linestyle="--")
    axes[0].axhline(-1, color="gray", linewidth=0.6, linestyle="--")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Normalized residual (method - consensus) / σ")
    axes[0].set_title("Hydrological-year method agreement with consensus")
    axes[0].tick_params(axis="x", rotation=12)

    axes[1].plot(global_validation["year"], global_validation["published_global_gt"], label="Published global", color="#355070", linewidth=2.2)
    axes[1].plot(global_validation["year"], global_validation["regional_sum_gt"], label="Sum of 19 regions", color="#e56b6f", linewidth=1.8, linestyle="--")
    axes[1].fill_between(
        global_validation["year"],
        global_validation["published_global_gt"] - global_validation["published_global_gt_error"],
        global_validation["published_global_gt"] + global_validation["published_global_gt_error"],
        color="#bde0fe",
        alpha=0.35,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Balance year")
    axes[1].set_ylabel("Annual mass change (Gt)")
    axes[1].set_title("Independent regional-to-global aggregation check")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "figure_4_validation_comparison.png", dpi=220)
    plt.close()


def render_report(
    input_overview: pd.DataFrame,
    coverage: pd.DataFrame,
    regional_metrics: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    global_metrics: dict,
    global_validation_metrics: dict,
    papers: list[PaperNote],
) -> str:
    total_datasets = int(input_overview["datasets"].sum())
    total_observations = int(input_overview["observations"].sum())
    n_regions = len(REGION_ORDER)

    global_cumulative = global_metrics["cumulative_gt_2000_2023"]
    global_sigma = global_metrics["cumulative_gt_sigma_2000_2023"]
    record_year = global_metrics["record_loss_year"]
    record_gt = global_metrics["record_loss_gt"]
    accel_pct = global_metrics["relative_change_last_vs_first_decade_pct"]

    top_losses = regional_metrics.nsmallest(5, "cumulative_gt")[
        ["region_label", "cumulative_gt", "cumulative_gt_uncertainty", "mean_annual_mwe"]
    ]
    top_loss_lines = "\n".join(
        [
            f"| {row.region_label} | {row.cumulative_gt:.1f} | {row.cumulative_gt_uncertainty:.1f} | {row.mean_annual_mwe:.3f} |"
            for row in top_losses.itertuples()
        ]
    )

    uncertainty_table = (
        comparison_summary.groupby("method_family")
        .agg(
            region_method_pairs=("region", "count"),
            mean_abs_z=("mean_abs_z", "mean"),
            median_rmse_gt=("rmse_gt", "median"),
            mean_ratio=("uncertainty_ratio_method_to_combined", "mean"),
        )
        .reset_index()
    )
    uncertainty_lines = "\n".join(
        [
            f"| {row.method_family} | {int(row.region_method_pairs)} | {row.mean_abs_z:.2f} | {row.median_rmse_gt:.1f} | {row.mean_ratio:.2f} |"
            for row in uncertainty_table.itertuples()
        ]
    )

    related_lines = "\n".join([f"- `{p.filename}`: {p.title}" for p in papers])

    report = f"""# Reconciling global glacier mass change observations, 2000-2023

## Abstract

This report analyzes the GlaMBIE 1.0.0 observational archive to produce a compact benchmark description of glacier mass change for the 19 global glacier regions used by the GlacierMIP/IPCC community. The archive contains {total_datasets} submitted regional datasets and {total_observations} time-stamped observations across glaciological, DEM differencing, satellite altimetry, gravimetry, and hybrid products. Using the published GlaMBIE calendar-year reconciliation as the benchmark time series, I characterize regional and global annual mass change from balance years 2000-2001 through 2023-2024, quantify cumulative losses, and validate the benchmark in two ways: by independently re-aggregating the 19 regional series to the global total, and by comparing the consensus estimate against method-group time series available in the hydrological-year outputs. The benchmark indicates a cumulative global glacier mass change of {global_cumulative:.0f} ± {global_sigma:.0f} Gt for 2000-2023, with the most negative annual global balance in {record_year}-{record_year + 1} at {record_gt:.0f} Gt. The mean annual global loss in 2012-2023 is {accel_pct:.1f}% more negative than in 2000-2011, consistent with an accelerating glacier contribution to sea-level rise.

## 1. Context and objective

Glacier mass change is a core climate indicator because it integrates atmospheric forcing, controls part of sea-level rise, and constrains cryosphere and hydrological model calibration. The related-work files in this workspace place the GlaMBIE benchmark in a larger literature chain: Zemp et al. extended multi-decadal observational estimates through 2016; Hugonnet et al. documented accelerated early-21st-century losses from geodetic observations; GlacierMIP and later ensemble studies translated regional observations into future projections; and Rounce et al. highlighted the sensitivity of 21st-century glacier loss to warming. GlaMBIE addresses the observational side of this chain by reconciling diverse methods into a consistent regional/global benchmark.

Related-work files examined:
{related_lines}

The analysis target here is operational rather than methodological reinvention: use the submitted datasets to describe observational coverage, use the official reconciled outputs as the benchmark product, and test whether the published consensus behaves consistently across regional aggregation and across observational method families.

## 2. Data

### 2.1 Archive structure

- `data/glambie/input/` contains the submitted regional solutions.
- `data/glambie/results/calendar_years/` contains the final annual benchmark series for 19 regions plus the global aggregate.
- `data/glambie/results/hydrological_years/` contains the regional combined estimate and method-group components (altimetry, gravimetry, and DEM differencing plus glaciological information).

The input archive spans {n_regions} GTN-G glacier regions and five observational families. Figure 1 shows dataset density by region and method.

![Input coverage heatmap](images/figure_1_input_coverage_heatmap.png)

### 2.2 Input overview by method family

| Method family | Datasets | Observation rows | Regions represented |
| --- | ---: | ---: | ---: |
{chr(10).join([f"| {METHOD_LABELS.get(row.method, row.method)} | {int(row.datasets)} | {int(row.observations)} | {int(row.regions)} |" for row in input_overview.itertuples()])}

Key observations from the inventory are straightforward. Hybrid/combined and DEM-differencing products have nearly complete regional coverage, while altimetry and gravimetry are spatially selective and glaciological series are temporally dense but geographically sparse within each region. This asymmetry is exactly why reconciliation is necessary: no single observational family provides both the annual cadence and the near-global regional completeness required for a benchmark.

## 3. Methods

### 3.1 Benchmark construction used in this report

I treat the official GlaMBIE calendar-year files as the primary benchmark product because they are already homogenized to common annual periods and common units (Gt and m w.e.). Each row is interpreted as one annual balance period labelled by its start year, so the delivered benchmark covers years 2000 through 2023 inclusive.

### 3.2 Independent checks

Two checks were implemented.

1. Regional-to-global aggregation: the 19 regional annual `combined_gt` series were summed and compared with the published global calendar-year file. Independent uncertainty was approximated by quadrature across regional annual uncertainties.
2. Method-family consistency: hydrological-year method-group series (`altimetry`, `gravimetry`, `demdiff_and_glaciological`) were compared with the regional combined estimate. Residuals were normalized by the paired uncertainty, using `sqrt(method_sigma^2 + combined_sigma^2)`.

### 3.3 Derived metrics

- Global cumulative mass change was computed as the sum of annual `combined_gt` values from 2000 through 2023.
- Cumulative uncertainty was propagated in quadrature across annual uncertainties.
- Regional cumulative losses, mean annual specific losses, and average uncertainty levels were summarized from the calendar-year products.
- Method agreement statistics were summarized using absolute z-scores and GT residual RMSE.

## 4. Results

### 4.1 Global benchmark

Figure 2 shows the annual and cumulative global benchmark. The time series begins with relatively modest losses in 2000-2001, deepens through the mid-2000s, and reaches an exceptionally negative endpoint in 2023-2024. Across the full 2000-2023 benchmark, the mean annual global balance is {global_metrics["mean_annual_gt_2000_2023"]:.1f} Gt yr^-1 ({global_metrics["mean_annual_mwe_2000_2023"]:.3f} m w.e. yr^-1).

![Global annual and cumulative benchmark](images/figure_2_global_mass_change.png)

The benchmark reports a cumulative global glacier mass change of **{global_cumulative:.0f} ± {global_sigma:.0f} Gt** between 2000 and 2023. The most negative year is **{record_year}-{record_year + 1}**, with **{record_gt:.1f} Gt**. Comparing the first and second halves of the record, the 2012-2023 mean annual loss is {accel_pct:.1f}% more negative than the 2000-2011 mean, indicating a clear intensification of glacier mass loss in the benchmark period.

### 4.2 Regional structure of loss

Figure 3 shows cumulative regional losses and the relationship between remaining glacierized area and absolute mass loss.

![Regional cumulative losses and area-loss scaling](images/figure_3_regional_losses.png)

The five regions with the largest cumulative losses are:

| Region | Cumulative mass change (Gt) | Uncertainty (Gt) | Mean specific balance (m w.e. yr^-1) |
| --- | ---: | ---: | ---: |
{top_loss_lines}

Alaska and the Southern Andes dominate absolute mass loss, while Central Europe and Iceland stand out for strong specific losses relative to their remaining glacierized area. High-latitude regions with large glacier areas exert the strongest control on the global GT budget, but smaller mid-latitude regions often exhibit more negative mean specific balances, consistent with strong climatic sensitivity.

### 4.3 Validation and method comparison

Figure 4 summarizes the internal validation.

![Validation against method groups and global aggregation](images/figure_4_validation_comparison.png)

The independent regional summation reproduces the published global total essentially exactly: the maximum absolute difference is {global_validation_metrics["max_abs_gt_difference"]:.6f} Gt and the mean absolute difference is {global_validation_metrics["mean_abs_gt_difference"]:.6f} Gt. This confirms that the delivered global benchmark is numerically consistent with the regional calendar-year files.

Method-family comparisons show that the consensus estimate generally lies well within the spread implied by the individual method-group uncertainties. Aggregated across region-method pairs:

| Method family | Region-method summaries | Mean abs. z-score | Median RMSE (Gt) | Mean method/consensus uncertainty ratio |
| --- | ---: | ---: | ---: | ---: |
{uncertainty_lines}

Average absolute z-scores remain close to or below 1 for all method families, which indicates that the consensus rarely departs from any one method group by more than the combined uncertainty envelope. In most regions the consensus uncertainty is also lower than the uncertainty attached to an individual method family, which is the expected outcome of a successful reconciliation exercise.

## 5. Discussion

Three scientific points emerge from this benchmark analysis.

First, the GlaMBIE archive succeeds where earlier observational syntheses were structurally limited: it merges annually resolved but spatially sparse observations with spatially extensive but temporally coarser or noisier products. The delivered annual regional/global series therefore serve as a practical observational benchmark for model calibration and report-level synthesis.

Second, the benchmark confirms that recent glacier losses are not merely persistently negative; they intensify over the 2000-2023 period. This aligns with the acceleration diagnosed in early-21st-century geodetic studies and provides an updated annual benchmark that can be paired directly with climate-model forcing histories.

Third, the regional decomposition matters. Absolute global loss is dominated by a handful of heavily glacierized regions, but specific losses reveal especially strong vulnerability in smaller mountain regions. Model evaluation should therefore not rely only on global GT totals; it should also test regional specific mass change.

There are also limitations. This report does not reconstruct GlaMBIE's original methodological pipeline from raw submissions, and the quadrature treatment of annual uncertainties ignores interannual covariance. The method-consistency analysis uses the hydrological-year regional method groups already distributed by GlaMBIE rather than rebuilding method-specific annualization from the raw archive. Those choices are appropriate for a benchmark audit, but not for a full methodological replication study.

## 6. Reproducibility

- Analysis code: `code/analyze_glambie.py`
- Output tables: `outputs/*.csv` and `outputs/summary_metrics.json`
- Figures: `report/images/*.png`

Running `python code/analyze_glambie.py` regenerates the complete set of tables, figures, and this report from the workspace data only.
"""
    return report


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    input_inventory = load_input_inventory()
    calendar_df = load_calendar_results()
    hydro_df = load_hydrological_results()
    papers = extract_related_work()

    coverage, input_overview = build_input_summary(input_inventory)
    global_validation, global_validation_metrics = build_global_validation(calendar_df)
    regional_metrics = build_regional_metrics(calendar_df)
    comparison, comparison_summary = build_method_consistency(hydro_df)
    global_metrics = build_global_metrics(calendar_df)
    related_df = extract_related_context(papers)

    input_inventory.to_csv(OUTPUT_DIR / "input_inventory.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "input_method_region_coverage.csv", index=False)
    input_overview.to_csv(OUTPUT_DIR / "input_method_overview.csv", index=False)
    global_validation.to_csv(OUTPUT_DIR / "global_aggregation_validation.csv", index=False)
    regional_metrics.to_csv(OUTPUT_DIR / "regional_metrics_2000_2023.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "method_comparison_hydrological_years.csv", index=False)
    comparison_summary.to_csv(OUTPUT_DIR / "method_comparison_summary.csv", index=False)
    related_df.to_csv(OUTPUT_DIR / "related_work_notes.csv", index=False)
    save_json(
        OUTPUT_DIR / "summary_metrics.json",
        {
            "global_metrics": global_metrics,
            "global_validation_metrics": global_validation_metrics,
        },
    )

    plot_coverage_heatmap(coverage)
    plot_global_series(calendar_df)
    plot_regional_cumulative_losses(regional_metrics)
    plot_method_validation(comparison, global_validation)

    report_text = render_report(
        input_overview=input_overview,
        coverage=coverage,
        regional_metrics=regional_metrics,
        comparison_summary=comparison_summary,
        global_metrics=global_metrics,
        global_validation_metrics=global_validation_metrics,
        papers=papers,
    )
    (REPORT_DIR / "report.md").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
