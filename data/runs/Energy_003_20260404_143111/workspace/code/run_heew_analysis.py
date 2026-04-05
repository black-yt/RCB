from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "HEEW_Mini-Dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_IMG_DIR = BASE_DIR / "report" / "images"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

ENERGY_COLS = [
    "Electricity [kW]",
    "Heat [mmBTU]",
    "Cooling Energy [Ton]",
    "PV Power Generation [kW]",
    "Greenhouse Gas Emission [Ton]",
]
WEATHER_COLS = [
    "Temperature [°F]",
    "Dew Point [°F]",
    "Humidity [%]",
    "Wind Speed [mph]",
    "Wind Gust [mph]",
    "Pressure [in]",
    "Precipitation [in]",
]
ALL_COLS = ENERGY_COLS + WEATHER_COLS
EXPECTED_INDEX = pd.date_range("2014-01-01 00:00:00", "2014-12-31 23:00:00", freq="h")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    cleaned = name.replace(" [", "_").replace("]", "")
    for src, dst in [(" ", "_"), ("/", "_per_"), ("°", "deg"), ("%", "pct")]:
        cleaned = cleaned.replace(src, dst)
    return cleaned.replace("__", "_")


def make_timestamp(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    return pd.to_datetime(df[["year", "month", "day", "hour"]])


def load_dataset() -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.stem.endswith("_energy"):
            key = csv_path.stem.replace("_energy", "")
        elif csv_path.stem.endswith("_weather"):
            key = csv_path.stem
        else:
            key = csv_path.stem
        df = pd.read_csv(csv_path)
        df["timestamp"] = make_timestamp(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        datasets[key] = df
    return datasets


def audit_datasets(datasets: Dict[str, pd.DataFrame]) -> None:
    audit_rows: List[dict] = []
    schema_rows: List[dict] = []
    notes = {}

    for name, df in datasets.items():
        value_cols = [c for c in df.columns if c not in {"year", "month", "day", "hour", "datetime", "timestamp"}]
        duplicated = int(df["timestamp"].duplicated().sum())
        missing_ts = int(len(EXPECTED_INDEX.difference(pd.DatetimeIndex(df["timestamp"]))))
        inferred_freq = pd.infer_freq(df["timestamp"][:10]) if len(df) >= 10 else None
        negative_counts = {col: int((df[col] < 0).sum()) for col in value_cols if pd.api.types.is_numeric_dtype(df[col])}
        na_counts = {col: int(df[col].isna().sum()) for col in value_cols}
        audit_rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "timestamp_min": df["timestamp"].min(),
                "timestamp_max": df["timestamp"].max(),
                "inferred_freq": inferred_freq,
                "missing_timestamps": missing_ts,
                "duplicate_timestamps": duplicated,
                **{f"na__{k}": v for k, v in na_counts.items()},
                **{f"negative__{k}": v for k, v in negative_counts.items()},
            }
        )
        notes[name] = {
            "columns": value_cols,
            "na_counts": na_counts,
            "negative_counts": negative_counts,
        }
        for col in df.columns:
            schema_rows.append({"dataset": name, "column": col, "dtype": str(df[col].dtype)})

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_DIR / "data_audit_summary.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(OUTPUT_DIR / "per_file_schema_summary.csv", index=False)
    (OUTPUT_DIR / "data_audit_notes.json").write_text(json.dumps(notes, indent=2, default=str))

    coverage_matrix = pd.DataFrame(
        {
            f"M{month:02d}": [
                int(
                    df.assign(month=df["timestamp"].dt.month)
                    .groupby("month")
                    .size()
                    .reindex(range(1, 13), fill_value=0)
                    .loc[month]
                    > 0
                )
                for df in datasets.values()
            ]
            for month in range(1, 13)
        },
        index=list(datasets.keys()),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        coverage_matrix,
        cmap="Greens",
        cbar=False,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Dataset coverage by month (all files fully cover 2014)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Dataset")
    fig.savefig(REPORT_IMG_DIR / "data_coverage_heatmap.png")
    plt.close(fig)


def build_analysis_panel(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    building_frames = []
    cleaning_records = []

    for name, df in datasets.items():
        working = df.copy()
        before_rows = len(working)
        before_na = int(working.isna().sum().sum())
        working = working.drop_duplicates(subset=["timestamp"], keep="first").set_index("timestamp").sort_index()
        working = working.reindex(EXPECTED_INDEX)
        numeric_cols = [c for c in working.columns if c not in {"year", "month", "day", "hour", "datetime"}]
        working[numeric_cols] = working[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if set(ENERGY_COLS).intersection(working.columns):
            for col in ENERGY_COLS:
                if col in working.columns:
                    working.loc[working[col] < 0, col] = np.nan
        after_na = int(working.isna().sum().sum())
        cleaning_records.append(
            {
                "dataset": name,
                "rows_before": before_rows,
                "rows_after": len(working),
                "na_before": before_na,
                "na_after": after_na,
                "rows_added_by_reindex": len(working) - before_rows,
            }
        )
        if name.startswith("BN"):
            renamed = working[ENERGY_COLS].rename(columns={c: f"{name}_{sanitize(c)}" for c in ENERGY_COLS})
            building_frames.append(renamed)
        elif name in {"CN01", "Total"}:
            renamed = working[ENERGY_COLS].rename(columns={c: f"{name}_{sanitize(c)}" for c in ENERGY_COLS})
            building_frames.append(renamed)
        elif name == "Total_weather":
            renamed = working[WEATHER_COLS].rename(columns={c: f"Weather_{sanitize(c)}" for c in WEATHER_COLS})
            building_frames.append(renamed)

    panel = pd.concat(building_frames, axis=1)
    panel.index.name = "timestamp"
    panel["month"] = panel.index.month
    panel["dayofweek"] = panel.index.dayofweek
    panel["hour"] = panel.index.hour
    panel.to_csv(OUTPUT_DIR / "aligned_hourly_panel.csv")
    pd.DataFrame(cleaning_records).to_csv(OUTPUT_DIR / "cleaning_impact_summary.csv", index=False)

    decisions = {
        "duplicate_rule": "Drop duplicated timestamps, keep first occurrence.",
        "alignment_rule": "Reindex all datasets to complete hourly range for 2014.",
        "negative_value_rule": "Replace negative energy values with NaN; no negatives were observed in this mini-dataset.",
        "imputation_rule": "No forward/backward filling for primary analysis; correlation uses pairwise complete observations.",
    }
    (OUTPUT_DIR / "cleaning_decisions.json").write_text(json.dumps(decisions, indent=2))

    missing_before_after = pd.DataFrame(cleaning_records)[["dataset", "na_before", "na_after"]].set_index("dataset")
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_before_after.plot(kind="bar", ax=ax)
    ax.set_title("Missing values before and after cleaning")
    ax.set_ylabel("Count")
    fig.savefig(REPORT_IMG_DIR / "missingness_before_after.png")
    plt.close(fig)

    example_cols = [
        "BN001_Electricity_kW",
        "CN01_Electricity_kW",
        "Total_Electricity_kW",
        "Weather_Temperature_degF",
    ]
    subset = panel.loc["2014-07-01":"2014-07-14", example_cols].copy()
    subset["Weather_Temperature_degF"] = (subset["Weather_Temperature_degF"] - subset["Weather_Temperature_degF"].mean()) / subset["Weather_Temperature_degF"].std()
    subset[["BN001_Electricity_kW", "CN01_Electricity_kW", "Total_Electricity_kW"]] = subset[["BN001_Electricity_kW", "CN01_Electricity_kW", "Total_Electricity_kW"]].apply(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(14, 6))
    subset.plot(ax=ax)
    ax.set_title("Example standardized series overlay (July 1–14, 2014)")
    ax.set_ylabel("Standardized value")
    fig.savefig(REPORT_IMG_DIR / "example_series_overlay.png")
    plt.close(fig)

    return panel


def daily_aggregate(panel: pd.DataFrame) -> pd.DataFrame:
    daily = panel.drop(columns=["month", "dayofweek", "hour"]).resample("D").mean(numeric_only=True)
    daily.index.name = "date"
    return daily


def deseasonalize(series: pd.Series, hours: pd.Series) -> pd.Series:
    hour_means = series.groupby(hours).transform("mean")
    return series - hour_means


def correlation_analysis(panel: pd.DataFrame) -> None:
    building_cols = [f"BN{i:03d}_Electricity_kW" for i in range(1, 11)]
    pearson = panel[building_cols].corr(method="pearson")
    spearman = panel[building_cols].corr(method="spearman")
    pearson.to_csv(OUTPUT_DIR / "correlation_buildings_pearson.csv")
    spearman.to_csv(OUTPUT_DIR / "correlation_buildings_spearman.csv")

    hierarchy_weather_cols = [
        "CN01_Electricity_kW",
        "Total_Electricity_kW",
        "CN01_Heat_mmBTU",
        "Total_Heat_mmBTU",
        "CN01_Cooling_Energy_Ton",
        "Total_Cooling_Energy_Ton",
        "CN01_PV_Power_Generation_kW",
        "Total_PV_Power_Generation_kW",
        "Weather_Temperature_degF",
        "Weather_Humidity_pct",
        "Weather_Wind_Speed_mph",
        "Weather_Pressure_in",
        "Weather_Precipitation_in",
    ]
    hierarchy_weather = panel[hierarchy_weather_cols].corr(method="pearson")
    hierarchy_weather.to_csv(OUTPUT_DIR / "correlation_hierarchy_weather.csv")

    monthly_rows = []
    for month, group in panel.groupby(panel.index.month):
        daily = group.drop(columns=["month", "dayofweek", "hour"]).resample("D").mean(numeric_only=True)
        monthly_rows.append(
            {
                "month": month,
                "hourly_total_electricity_vs_temp": group["Total_Electricity_kW"].corr(group["Weather_Temperature_degF"]),
                "hourly_total_cooling_vs_temp": group["Total_Cooling_Energy_Ton"].corr(group["Weather_Temperature_degF"]),
                "hourly_total_heat_vs_temp": group["Total_Heat_mmBTU"].corr(group["Weather_Temperature_degF"]),
                "daily_total_electricity_vs_temp": daily["Total_Electricity_kW"].corr(daily["Weather_Temperature_degF"]),
                "daily_total_cooling_vs_temp": daily["Total_Cooling_Energy_Ton"].corr(daily["Weather_Temperature_degF"]),
                "daily_total_heat_vs_temp": daily["Total_Heat_mmBTU"].corr(daily["Weather_Temperature_degF"]),
            }
        )
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(OUTPUT_DIR / "monthly_correlation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pearson, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Building electricity Pearson correlation")
    fig.savefig(REPORT_IMG_DIR / "building_correlation_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(hierarchy_weather, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Hierarchy and weather correlation matrix")
    fig.savefig(REPORT_IMG_DIR / "hierarchy_weather_correlation_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_melt = monthly_df.melt(id_vars="month", var_name="metric", value_name="correlation")
    sns.lineplot(data=monthly_melt, x="month", y="correlation", hue="metric", marker="o", ax=ax)
    ax.set_title("Monthly stability of Total load-temperature correlations")
    ax.set_ylim(-1, 1)
    fig.savefig(REPORT_IMG_DIR / "monthly_correlation_stability.png")
    plt.close(fig)

    # de-seasonalized robustness table
    residual_rows = []
    for target in ["Total_Electricity_kW", "Total_Cooling_Energy_Ton", "Total_Heat_mmBTU"]:
        resid = deseasonalize(panel[target], panel["hour"])
        resid_temp = deseasonalize(panel["Weather_Temperature_degF"], panel["hour"])
        residual_rows.append({"variable": target, "deseasonalized_corr_with_temp": resid.corr(resid_temp)})
    pd.DataFrame(residual_rows).to_csv(OUTPUT_DIR / "deseasonalized_temperature_correlations.csv", index=False)


def error_metrics(actual: pd.Series, reference: pd.Series, tol: float = 1e-6) -> dict:
    residual = actual - reference
    mae = float(np.nanmean(np.abs(residual)))
    rmse = float(np.sqrt(np.nanmean(np.square(residual))))
    mean_signed = float(np.nanmean(residual))
    denom = np.where(np.abs(reference) < tol, np.nan, np.abs(reference))
    mape = float(np.nanmean(np.abs(residual) / denom) * 100)
    smape = float(np.nanmean(2 * np.abs(residual) / (np.abs(actual) + np.abs(reference) + tol)) * 100)
    within_1pct = float(np.nanmean(np.abs(residual) <= 0.01 * np.abs(reference)) * 100)
    exact = float(np.nanmean(np.abs(residual) <= tol) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_signed_residual": mean_signed,
        "mape_pct": mape,
        "smape_pct": smape,
        "within_1pct_pct": within_1pct,
        "exact_match_pct": exact,
    }


def hierarchy_analysis(panel: pd.DataFrame) -> None:
    variable_map = {
        "Electricity": "Electricity_kW",
        "Heat": "Heat_mmBTU",
        "Cooling": "Cooling_Energy_Ton",
        "PV": "PV_Power_Generation_kW",
        "GHG": "Greenhouse_Gas_Emission_Ton",
    }
    residual_frame = pd.DataFrame(index=panel.index)
    metrics_rows = []
    daily_rows = []

    for label, suffix in variable_map.items():
        bn_cols = [f"BN{i:03d}_{suffix}" for i in range(1, 11)]
        bn_sum = panel[bn_cols].sum(axis=1)
        cn = panel[f"CN01_{suffix}"]
        total = panel[f"Total_{suffix}"]
        residual_frame[f"sum_bn_minus_cn01_{label}"] = bn_sum - cn
        residual_frame[f"sum_bn_minus_total_{label}"] = bn_sum - total
        residual_frame[f"cn01_minus_total_{label}"] = cn - total

        metrics_rows.append({"comparison": f"sum_bn_vs_cn01_{label}", **error_metrics(bn_sum, cn)})
        metrics_rows.append({"comparison": f"sum_bn_vs_total_{label}", **error_metrics(bn_sum, total)})
        metrics_rows.append({"comparison": f"cn01_vs_total_{label}", **error_metrics(cn, total)})

        daily_df = pd.DataFrame({"bn_sum": bn_sum, "cn01": cn, "total": total}).resample("D").mean()
        daily_rows.append({"comparison": f"sum_bn_vs_cn01_{label}", **error_metrics(daily_df["bn_sum"], daily_df["cn01"])})
        daily_rows.append({"comparison": f"sum_bn_vs_total_{label}", **error_metrics(daily_df["bn_sum"], daily_df["total"])})
        daily_rows.append({"comparison": f"cn01_vs_total_{label}", **error_metrics(daily_df["cn01"], daily_df["total"])})

    residual_frame.to_csv(OUTPUT_DIR / "hierarchy_residual_timeseries.csv")
    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "hierarchy_consistency_summary.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUTPUT_DIR / "hierarchy_daily_error_summary.csv", index=False)

    bn_sum_elec = panel[[f"BN{i:03d}_Electricity_kW" for i in range(1, 11)]].sum(axis=1)
    compare = pd.DataFrame(
        {
            "Sum of BN001-BN010": bn_sum_elec.resample("D").mean(),
            "CN01": panel["CN01_Electricity_kW"].resample("D").mean(),
            "Total": panel["Total_Electricity_kW"].resample("D").mean(),
        }
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    compare.loc["2014-01-01":"2014-03-31"].plot(ax=ax)
    ax.set_title("Daily electricity aggregation consistency (Q1 2014)")
    ax.set_ylabel("Electricity [kW]")
    fig.savefig(REPORT_IMG_DIR / "hierarchy_overlay_timeseries.png")
    plt.close(fig)

    residual_long = residual_frame[[c for c in residual_frame.columns if "Electricity" in c]].melt(var_name="comparison", value_name="residual")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=residual_long, x="comparison", y="residual", ax=ax)
    ax.set_title("Electricity hierarchy residual distributions")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(REPORT_IMG_DIR / "hierarchy_residual_distribution.png")
    plt.close(fig)

    monthly_error = residual_frame[["sum_bn_minus_total_Electricity", "sum_bn_minus_cn01_Electricity"]].resample("M").apply(lambda x: np.mean(np.abs(x)))
    monthly_error.index = monthly_error.index.month
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_error.plot(marker="o", ax=ax)
    ax.set_title("Monthly mean absolute aggregation error for electricity")
    ax.set_xlabel("Month")
    ax.set_ylabel("Absolute residual [kW]")
    fig.savefig(REPORT_IMG_DIR / "monthly_hierarchy_error.png")
    plt.close(fig)


def generate_summary_tables(panel: pd.DataFrame) -> None:
    summary_cols = [
        "Total_Electricity_kW",
        "Total_Heat_mmBTU",
        "Total_Cooling_Energy_Ton",
        "Total_PV_Power_Generation_kW",
        "Total_Greenhouse_Gas_Emission_Ton",
        "Weather_Temperature_degF",
        "Weather_Humidity_pct",
        "Weather_Wind_Speed_mph",
        "Weather_Pressure_in",
        "Weather_Precipitation_in",
    ]
    desc = panel[summary_cols].describe().T
    desc["missing_rate_pct"] = panel[summary_cols].isna().mean() * 100
    desc.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    outliers = []
    for col in summary_cols:
        series = panel[col].dropna()
        z = np.abs(zscore(series, nan_policy="omit"))
        outliers.append({"variable": col, "zscore_gt_3_count": int((z > 3).sum())})
    pd.DataFrame(outliers).to_csv(OUTPUT_DIR / "outlier_screening.csv", index=False)


def main() -> None:
    ensure_dirs()
    datasets = load_dataset()
    audit_datasets(datasets)
    panel = build_analysis_panel(datasets)
    generate_summary_tables(panel)
    correlation_analysis(panel)
    hierarchy_analysis(panel)
    daily_aggregate(panel).to_csv(OUTPUT_DIR / "daily_aligned_panel.csv")
    print("Analysis complete. Outputs written to outputs/ and report/images/.")


if __name__ == "__main__":
    main()
