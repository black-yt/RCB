from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "HEEW_Mini-Dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "report" / "images"

ENERGY_METRICS = [
    "Electricity [kW]",
    "Heat [mmBTU]",
    "Cooling Energy [Ton]",
    "PV Power Generation [kW]",
    "Greenhouse Gas Emission [Ton]",
]

WEATHER_METRICS = [
    "Temperature [°F]",
    "Dew Point [°F]",
    "Humidity [%]",
    "Wind Speed [mph]",
    "Wind Gust [mph]",
    "Pressure [in]",
    "Precipitation [in]",
]

RENAME_MAP = {
    "Electricity [kW]": "electricity_kw",
    "Heat [mmBTU]": "heat_mmbtu",
    "Cooling Energy [Ton]": "cooling_ton",
    "PV Power Generation [kW]": "pv_kw",
    "Greenhouse Gas Emission [Ton]": "ghg_ton",
    "Temperature [°F]": "temperature_f",
    "Dew Point [°F]": "dew_point_f",
    "Humidity [%]": "humidity_pct",
    "Wind Speed [mph]": "wind_speed_mph",
    "Wind Gust [mph]": "wind_gust_mph",
    "Pressure [in]": "pressure_in",
    "Precipitation [in]": "precipitation_in",
}


sns.set_theme(style="whitegrid", context="talk")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_energy_data() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("*_energy.csv")):
        entity = path.stem.replace("_energy", "")
        level = "building" if entity.startswith("BN") else "aggregate"
        df = pd.read_csv(path)
        df["entity"] = entity
        df["level"] = level
        df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        frames.append(df)

    energy = pd.concat(frames, ignore_index=True)
    energy = energy[["timestamp", "entity", "level", "year", "month", "day", "hour", *ENERGY_METRICS]]
    return energy.sort_values(["entity", "timestamp"]).reset_index(drop=True)


def load_weather_data() -> pd.DataFrame:
    weather = pd.read_csv(DATA_DIR / "Total_weather.csv")
    weather["timestamp"] = pd.to_datetime(weather["datetime"])
    weather = weather[["timestamp", *WEATHER_METRICS]].sort_values("timestamp").reset_index(drop=True)
    return weather


def build_unified_dataset(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    unified = energy.merge(weather, on="timestamp", how="left", validate="many_to_one")
    rename_targets = {c: RENAME_MAP[c] for c in RENAME_MAP if c in unified.columns}
    unified = unified.rename(columns=rename_targets)
    return unified


def profile_entities(energy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity, group in energy.groupby("entity"):
        duplicate_timestamps = int(group["timestamp"].duplicated().sum())
        timestamp_diff = group.sort_values("timestamp")["timestamp"].diff().dropna()
        irregular_steps = int((timestamp_diff != pd.Timedelta(hours=1)).sum())
        row = {
            "entity": entity,
            "level": group["level"].iloc[0],
            "rows": len(group),
            "start": group["timestamp"].min(),
            "end": group["timestamp"].max(),
            "duplicate_timestamps": duplicate_timestamps,
            "irregular_hour_steps": irregular_steps,
        }
        for metric in ENERGY_METRICS:
            row[f"missing__{RENAME_MAP[metric]}"] = int(group[metric].isna().sum())
            row[f"min__{RENAME_MAP[metric]}"] = float(group[metric].min())
            row[f"max__{RENAME_MAP[metric]}"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entity")


def profile_weather(weather: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in WEATHER_METRICS:
        series = weather[metric]
        rows.append(
            {
                "variable": RENAME_MAP[metric],
                "missing": int(series.isna().sum()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }
        )
    return pd.DataFrame(rows)


def overall_descriptive_statistics(unified: pd.DataFrame) -> pd.DataFrame:
    analysis_cols = [
        "electricity_kw",
        "heat_mmbtu",
        "cooling_ton",
        "pv_kw",
        "ghg_ton",
        "temperature_f",
        "dew_point_f",
        "humidity_pct",
        "wind_speed_mph",
        "wind_gust_mph",
        "pressure_in",
        "precipitation_in",
    ]
    stats = unified[analysis_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    stats["missing"] = unified[analysis_cols].isna().sum()
    return stats.reset_index().rename(columns={"index": "variable"})


def detect_outliers_by_entity(energy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity, group in energy.groupby("entity"):
        for metric in ENERGY_METRICS:
            series = group[metric]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
            negative_count = int((series < 0).sum())
            zero_count = int((series == 0).sum())
            rows.append(
                {
                    "entity": entity,
                    "variable": RENAME_MAP[metric],
                    "lower_iqr_bound": float(lower),
                    "upper_iqr_bound": float(upper),
                    "outlier_count": int(outlier_mask.sum()),
                    "outlier_fraction": float(outlier_mask.mean()),
                    "negative_count": negative_count,
                    "zero_count": zero_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["entity", "variable"]).reset_index(drop=True)


def summarize_cleaning_rules(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ENERGY_METRICS:
        series = energy[metric]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        rows.append(
            {
                "domain": "energy",
                "variable": RENAME_MAP[metric],
                "suggested_rule": "retain non-negative values; flag IQR outliers for review; preserve true zeros where physically plausible",
                "global_min": float(series.min()),
                "global_max": float(series.max()),
                "iqr_lower": float(q1 - 1.5 * iqr),
                "iqr_upper": float(q3 + 1.5 * iqr),
                "missing_count": int(series.isna().sum()),
                "negative_count": int((series < 0).sum()),
            }
        )
    for metric in WEATHER_METRICS:
        series = weather[metric]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        rows.append(
            {
                "domain": "weather",
                "variable": RENAME_MAP[metric],
                "suggested_rule": "retain observed values; flag extreme IQR outliers and impossible physical values for review",
                "global_min": float(series.min()),
                "global_max": float(series.max()),
                "iqr_lower": float(q1 - 1.5 * iqr),
                "iqr_upper": float(q3 + 1.5 * iqr),
                "missing_count": int(series.isna().sum()),
                "negative_count": int((series < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_hierarchical_validation(energy: pd.DataFrame) -> pd.DataFrame:
    building = energy[energy["entity"].str.startswith("BN")].copy()
    building_sum = (
        building.groupby("timestamp", as_index=False)[ENERGY_METRICS]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    rows = []
    for target in ["CN01", "Total"]:
        target_df = energy[energy["entity"] == target].sort_values("timestamp").reset_index(drop=True)
        for metric in ENERGY_METRICS:
            diff = building_sum[metric] - target_df[metric]
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(np.square(diff))))
            max_abs = float(np.max(np.abs(diff)))
            rows.append(
                {
                    "target_entity": target,
                    "variable": RENAME_MAP[metric],
                    "mae": mae,
                    "rmse": rmse,
                    "max_abs_error": max_abs,
                    "mean_signed_error": float(diff.mean()),
                    "allclose_atol_1e-9": bool(np.allclose(building_sum[metric], target_df[metric], atol=1e-9)),
                }
            )
    return pd.DataFrame(rows)


def compute_correlations(unified: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total = unified[unified["entity"] == "Total"].copy()
    corr_cols = [
        "electricity_kw",
        "heat_mmbtu",
        "cooling_ton",
        "pv_kw",
        "ghg_ton",
        "temperature_f",
        "dew_point_f",
        "humidity_pct",
        "wind_speed_mph",
        "wind_gust_mph",
        "pressure_in",
        "precipitation_in",
    ]
    corr_matrix = total[corr_cols].corr(method="pearson")
    corr_long = (
        corr_matrix.reset_index()
        .melt(id_vars="index", var_name="variable_2", value_name="pearson_r")
        .rename(columns={"index": "variable_1"})
    )
    energy_weather = corr_long[
        corr_long["variable_1"].isin(["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw", "ghg_ton"])
        & corr_long["variable_2"].isin(
            [
                "temperature_f",
                "dew_point_f",
                "humidity_pct",
                "wind_speed_mph",
                "wind_gust_mph",
                "pressure_in",
                "precipitation_in",
            ]
        )
    ].sort_values("pearson_r", ascending=False)
    return corr_matrix, energy_weather.reset_index(drop=True)


def monthly_entity_summary(unified: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        unified.groupby(["entity", "month"], as_index=False)[
            ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw", "ghg_ton"]
        ]
        .mean()
        .sort_values(["entity", "month"])
    )
    return monthly


def seasonal_total_summary(unified: pd.DataFrame) -> pd.DataFrame:
    total = unified[unified["entity"] == "Total"].copy()
    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    total["season"] = total["month"].map(season_map)
    return (
        total.groupby("season", as_index=False)[
            ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw", "ghg_ton", "temperature_f"]
        ]
        .mean()
        .sort_values("season")
    )


def save_plot(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_total_time_series(unified: pd.DataFrame) -> None:
    total = unified[unified["entity"] == "Total"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

    axes[0].plot(total["timestamp"], total["electricity_kw"], label="Electricity [kW]", linewidth=1.0)
    axes[0].plot(total["timestamp"], total["pv_kw"], label="PV [kW]", linewidth=1.0, alpha=0.85)
    axes[0].set_title("Total entity hourly electricity demand and PV generation (2014)")
    axes[0].legend(loc="upper right")

    axes[1].plot(total["timestamp"], total["heat_mmbtu"], label="Heat [mmBTU]", linewidth=1.0)
    axes[1].plot(total["timestamp"], total["cooling_ton"], label="Cooling [Ton]", linewidth=1.0, alpha=0.9)
    axes[1].set_title("Total entity hourly thermal loads (2014)")
    axes[1].legend(loc="upper right")

    axes[2].plot(total["timestamp"], total["temperature_f"], label="Temperature [°F]", linewidth=1.0)
    axes[2].plot(total["timestamp"], total["ghg_ton"], label="GHG emissions [Ton]", linewidth=1.0, alpha=0.85)
    axes[2].set_title("Total entity weather-emissions context (2014)")
    axes[2].legend(loc="upper right")

    save_plot(fig, "total_hourly_overview.png")


def plot_monthly_profiles(unified: pd.DataFrame) -> None:
    total = unified[unified["entity"] == "Total"].copy()
    monthly = (
        total.groupby("month", as_index=False)[
            ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw", "ghg_ton", "temperature_f"]
        ]
        .mean()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    sns.lineplot(data=monthly, x="month", y="electricity_kw", marker="o", ax=axes[0])
    sns.lineplot(data=monthly, x="month", y="pv_kw", marker="o", ax=axes[0])
    axes[0].set_title("Monthly mean electricity and PV")

    sns.lineplot(data=monthly, x="month", y="heat_mmbtu", marker="o", ax=axes[1])
    sns.lineplot(data=monthly, x="month", y="cooling_ton", marker="o", ax=axes[1])
    axes[1].set_title("Monthly mean thermal loads")

    sns.lineplot(data=monthly, x="month", y="ghg_ton", marker="o", ax=axes[2])
    axes[2].set_title("Monthly mean GHG emissions")

    sns.lineplot(data=monthly, x="month", y="temperature_f", marker="o", ax=axes[3], color="tab:red")
    axes[3].set_title("Monthly mean temperature")

    for ax in axes:
        ax.set_xlabel("Month")

    save_plot(fig, "monthly_profiles_total.png")


def plot_building_heatmap(unified: pd.DataFrame) -> None:
    building = unified[unified["entity"].str.startswith("BN")].copy()
    pivot = building.groupby(["entity", "month"], as_index=False)["electricity_kw"].mean().pivot(
        index="entity", columns="month", values="electricity_kw"
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title("Building-level monthly mean electricity demand")
    ax.set_xlabel("Month")
    ax.set_ylabel("Building")
    save_plot(fig, "building_monthly_electricity_heatmap.png")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Pearson correlations for Total energy-weather variables")
    save_plot(fig, "correlation_heatmap_total.png")


def plot_temperature_relationships(unified: pd.DataFrame) -> None:
    total = unified[unified["entity"] == "Total"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    sns.regplot(data=total.sample(min(len(total), 2000), random_state=42), x="temperature_f", y="electricity_kw", scatter_kws={"s": 8, "alpha": 0.3}, line_kws={"color": "black"}, ax=axes[0])
    axes[0].set_title("Electricity vs temperature")

    sns.regplot(data=total.sample(min(len(total), 2000), random_state=42), x="temperature_f", y="cooling_ton", scatter_kws={"s": 8, "alpha": 0.3}, line_kws={"color": "black"}, ax=axes[1])
    axes[1].set_title("Cooling vs temperature")

    sns.regplot(data=total.sample(min(len(total), 2000), random_state=42), x="temperature_f", y="heat_mmbtu", scatter_kws={"s": 8, "alpha": 0.3}, line_kws={"color": "black"}, ax=axes[2])
    axes[2].set_title("Heat vs temperature")

    save_plot(fig, "temperature_relationships.png")


def plot_hierarchical_consistency(energy: pd.DataFrame) -> None:
    building = energy[energy["entity"].str.startswith("BN")].copy()
    building_sum = building.groupby("timestamp", as_index=False)[ENERGY_METRICS].sum()
    total = energy[energy["entity"] == "Total"].sort_values("timestamp")
    compare = building_sum.merge(total[["timestamp", *ENERGY_METRICS]], on="timestamp", suffixes=("_bn_sum", "_total"))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    pairs = [
        ("Electricity [kW]", axes[0, 0]),
        ("Heat [mmBTU]", axes[0, 1]),
        ("Cooling Energy [Ton]", axes[1, 0]),
        ("PV Power Generation [kW]", axes[1, 1]),
    ]
    for metric, ax in pairs:
        ax.scatter(compare[f"{metric}_bn_sum"], compare[f"{metric}_total"], s=8, alpha=0.35)
        low = min(compare[f"{metric}_bn_sum"].min(), compare[f"{metric}_total"].min())
        high = max(compare[f"{metric}_bn_sum"].max(), compare[f"{metric}_total"].max())
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Sum of BN001-BN010")
        ax.set_ylabel("Total")
        ax.set_title(metric)
    save_plot(fig, "hierarchical_consistency_scatter.png")


def plot_daily_load_shapes(unified: pd.DataFrame) -> None:
    total = unified[unified["entity"] == "Total"].copy()
    diurnal = total.groupby("hour", as_index=False)[["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw"]].mean()
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw"]:
        ax.plot(diurnal["hour"], diurnal[col], marker="o", label=col)
    ax.set_title("Average daily load shape for Total entity")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average value")
    ax.legend()
    save_plot(fig, "daily_load_shape_total.png")


def write_summary_json(entity_profile: pd.DataFrame, hierarchy: pd.DataFrame, corr_table: pd.DataFrame) -> None:
    summary = {
        "dataset": {
            "energy_entities": entity_profile["entity"].tolist(),
            "n_entities": int(entity_profile["entity"].nunique()),
            "hours_per_entity": int(entity_profile["rows"].iloc[0]),
        },
        "hierarchy_validation": hierarchy.to_dict(orient="records"),
        "top_energy_weather_correlations": corr_table.head(10).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ensure_dirs()

    energy = load_energy_data()
    weather = load_weather_data()
    unified = build_unified_dataset(energy, weather)

    entity_profile = profile_entities(energy)
    weather_profile = profile_weather(weather)
    descriptive_stats = overall_descriptive_statistics(unified)
    outlier_summary = detect_outliers_by_entity(energy)
    cleaning_rules = summarize_cleaning_rules(energy, weather)
    hierarchy_validation = compute_hierarchical_validation(energy)
    corr_matrix, energy_weather_corr = compute_correlations(unified)
    monthly_summary = monthly_entity_summary(unified)
    seasonal_summary = seasonal_total_summary(unified)

    unified.to_csv(OUTPUT_DIR / "heew_unified_2014.csv", index=False)
    entity_profile.to_csv(OUTPUT_DIR / "entity_profile.csv", index=False)
    weather_profile.to_csv(OUTPUT_DIR / "weather_profile.csv", index=False)
    descriptive_stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False)
    outlier_summary.to_csv(OUTPUT_DIR / "outlier_summary_by_entity.csv", index=False)
    cleaning_rules.to_csv(OUTPUT_DIR / "suggested_cleaning_rules.csv", index=False)
    hierarchy_validation.to_csv(OUTPUT_DIR / "hierarchical_validation.csv", index=False)
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix_total.csv")
    energy_weather_corr.to_csv(OUTPUT_DIR / "energy_weather_correlations_total.csv", index=False)
    monthly_summary.to_csv(OUTPUT_DIR / "monthly_entity_summary.csv", index=False)
    seasonal_summary.to_csv(OUTPUT_DIR / "seasonal_total_summary.csv", index=False)
    write_summary_json(entity_profile, hierarchy_validation, energy_weather_corr)

    plot_total_time_series(unified)
    plot_monthly_profiles(unified)
    plot_building_heatmap(unified)
    plot_correlation_heatmap(corr_matrix)
    plot_temperature_relationships(unified)
    plot_hierarchical_consistency(energy)
    plot_daily_load_shapes(unified)

    print("Analysis script created for HEEW mini-dataset.")
    print(f"Data outputs directory: {OUTPUT_DIR}")
    print(f"Figure outputs directory: {FIG_DIR}")


if __name__ == "__main__":
    main()
