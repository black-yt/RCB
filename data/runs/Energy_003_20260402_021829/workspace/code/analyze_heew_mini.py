from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "HEEW_Mini-Dataset"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"

ENERGY_VARS = {
    "Electricity [kW]": "electricity_kw",
    "Heat [mmBTU]": "heat_mmbtu",
    "Cooling Energy [Ton]": "cooling_ton",
    "PV Power Generation [kW]": "pv_kw",
    "Greenhouse Gas Emission [Ton]": "ghg_ton",
}

WEATHER_VARS = {
    "Temperature [°F]": "temperature_f",
    "Dew Point [°F]": "dew_point_f",
    "Humidity [%]": "humidity_pct",
    "Wind Speed [mph]": "wind_speed_mph",
    "Wind Gust [mph]": "wind_gust_mph",
    "Pressure [in]": "pressure_in",
    "Precipitation [in]": "precipitation_in",
}

NONNEGATIVE_VARS = list(ENERGY_VARS.values()) + list(WEATHER_VARS.values())
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}


@dataclass(frozen=True)
class EntityMeta:
    entity_id: str
    level: str
    parent: str


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def entity_meta(entity_id: str) -> EntityMeta:
    if entity_id.startswith("BN"):
        return EntityMeta(entity_id=entity_id, level="building", parent="CN01")
    if entity_id == "CN01":
        return EntityMeta(entity_id=entity_id, level="community", parent="Total")
    if entity_id == "Total":
        return EntityMeta(entity_id=entity_id, level="district", parent="ROOT")
    raise ValueError(f"Unknown entity_id: {entity_id}")


def load_weather() -> pd.DataFrame:
    weather = pd.read_csv(DATA_DIR / "Total_weather.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather = weather.rename(columns=WEATHER_VARS)
    return weather.sort_values("datetime").reset_index(drop=True)


def load_energy() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("*_energy.csv")):
        entity_id = path.stem.replace("_energy", "")
        meta = entity_meta(entity_id)
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.rename(columns=ENERGY_VARS)
        df["entity_id"] = meta.entity_id
        df["level"] = meta.level
        df["parent"] = meta.parent
        frames.append(df)
    energy = pd.concat(frames, ignore_index=True)
    return energy.sort_values(["entity_id", "datetime"]).reset_index(drop=True)


def build_master_panel() -> pd.DataFrame:
    energy = load_energy()
    weather = load_weather()
    panel = energy.merge(weather, on="datetime", how="left", validate="many_to_one")
    panel["month_name"] = panel["datetime"].dt.month_name().str.slice(stop=3)
    panel["season"] = panel["datetime"].dt.month.map(SEASON_MAP)
    panel["day_of_week"] = panel["datetime"].dt.day_name().str.slice(stop=3)
    panel["hour_of_day"] = panel["datetime"].dt.hour
    panel["day_of_year"] = panel["datetime"].dt.dayofyear
    panel["hour_of_week"] = panel["datetime"].dt.dayofweek * 24 + panel["datetime"].dt.hour
    cols = [
        "datetime",
        "entity_id",
        "level",
        "parent",
        "year",
        "month",
        "day",
        "hour",
        "season",
        "month_name",
        "day_of_week",
        "hour_of_day",
        "day_of_year",
        "hour_of_week",
        *ENERGY_VARS.values(),
        *WEATHER_VARS.values(),
    ]
    return panel[cols].sort_values(["entity_id", "datetime"]).reset_index(drop=True)


def temporal_quality_checks(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for entity_id, grp in panel.groupby("entity_id", sort=True):
        grp = grp.sort_values("datetime")
        diffs = grp["datetime"].diff().dropna()
        records.append(
            {
                "entity_id": entity_id,
                "records": int(len(grp)),
                "duplicates": int(grp["datetime"].duplicated().sum()),
                "missing_timestamps": int((diffs != pd.Timedelta(hours=1)).sum()),
                "start": grp["datetime"].min().isoformat(),
                "end": grp["datetime"].max().isoformat(),
            }
        )
    return pd.DataFrame(records)


def missingness_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = list(ENERGY_VARS.values()) + list(WEATHER_VARS.values())
    for entity_id, grp in panel.groupby("entity_id", sort=True):
        total = len(grp)
        for col in variables:
            rows.append(
                {
                    "entity_id": entity_id,
                    "variable": col,
                    "missing_count": int(grp[col].isna().sum()),
                    "missing_rate": float(grp[col].isna().mean()),
                    "non_missing_count": int(total - grp[col].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def negative_value_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity_id, grp in panel.groupby("entity_id", sort=True):
        for col in NONNEGATIVE_VARS:
            rows.append(
                {
                    "entity_id": entity_id,
                    "variable": col,
                    "negative_count": int((grp[col] < 0).sum()),
                    "minimum": float(grp[col].min()),
                }
            )
    return pd.DataFrame(rows)


def seasonal_outlier_flags(panel: pd.DataFrame) -> pd.DataFrame:
    flags: list[pd.DataFrame] = []
    variables = list(ENERGY_VARS.values()) + list(WEATHER_VARS.values())
    for entity_id, grp in panel.groupby("entity_id", sort=True):
        base = grp[["datetime", "entity_id", "hour_of_week"]].copy()
        for col in variables:
            stats = grp.groupby("hour_of_week")[col].agg(
                median="median",
                mad=lambda s: np.median(np.abs(s - np.median(s))),
            )
            merged = grp[["datetime", "hour_of_week", col]].merge(
                stats, on="hour_of_week", how="left"
            )
            scale = 1.4826 * merged["mad"].replace(0, np.nan)
            robust_z = (merged[col] - merged["median"]).abs() / scale
            flagged = robust_z > 6
            tmp = base.copy()
            tmp["variable"] = col
            tmp["value"] = grp[col].to_numpy()
            tmp["seasonal_median"] = merged["median"].to_numpy()
            tmp["seasonal_mad"] = merged["mad"].to_numpy()
            tmp["robust_z"] = robust_z.to_numpy()
            tmp["flagged"] = flagged.fillna(False).to_numpy()
            flags.append(tmp)
    outliers = pd.concat(flags, ignore_index=True)
    return outliers[outliers["flagged"]].sort_values(["entity_id", "variable", "datetime"])


def clean_panel(panel: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    clean = panel.copy()
    for col in list(ENERGY_VARS.values()) + list(WEATHER_VARS.values()):
        flagged_pairs = outliers.loc[outliers["variable"] == col, ["entity_id", "datetime"]]
        if flagged_pairs.empty:
            continue
        flagged_pairs = flagged_pairs.assign(flagged=True).drop_duplicates()
        mask = (
            clean[["entity_id", "datetime"]]
            .merge(flagged_pairs, on=["entity_id", "datetime"], how="left")["flagged"]
            .eq(True)
            .to_numpy()
        )
        clean.loc[mask, col] = np.nan
    clean = clean.sort_values(["entity_id", "datetime"]).reset_index(drop=True)
    for entity_id, grp_idx in clean.groupby("entity_id").groups.items():
        grp = clean.loc[grp_idx].copy()
        vars_to_fill = list(ENERGY_VARS.values()) + list(WEATHER_VARS.values())
        grp[vars_to_fill] = grp[vars_to_fill].interpolate(
            method="linear", limit=3, limit_direction="both"
        )
        clean.loc[grp_idx, vars_to_fill] = grp[vars_to_fill].to_numpy()
    for col in NONNEGATIVE_VARS:
        clean.loc[clean[col] < 0, col] = np.nan
    return clean


def hierarchy_validation(panel: pd.DataFrame) -> pd.DataFrame:
    building_sum = (
        panel.loc[panel["level"] == "building", ["datetime", *ENERGY_VARS.values()]]
        .groupby("datetime", as_index=False)
        .sum()
    )
    rows = []
    for target in ["CN01", "Total"]:
        target_df = panel.loc[panel["entity_id"] == target, ["datetime", *ENERGY_VARS.values()]]
        merged = building_sum.merge(
            target_df, on="datetime", suffixes=("_buildings", f"_{target.lower()}")
        )
        for col in ENERGY_VARS.values():
            diff = merged[f"{col}_buildings"] - merged[f"{col}_{target.lower()}"]
            denom = merged[f"{col}_{target.lower()}"].abs().replace(0, np.nan)
            rows.append(
                {
                    "target_entity": target,
                    "variable": col,
                    "mae": float(diff.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "max_abs_error": float(diff.abs().max()),
                    "mape_pct": float((diff.abs() / denom).dropna().mean() * 100 if denom.notna().any() else 0.0),
                    "is_exact_match": bool(np.allclose(diff.to_numpy(), 0.0)),
                }
            )
    return pd.DataFrame(rows)


def entity_summary(panel: pd.DataFrame) -> pd.DataFrame:
    agg = (
        panel.groupby(["entity_id", "level"], as_index=False)[list(ENERGY_VARS.values())]
        .agg(["mean", "min", "max", "sum"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(part) for part in col if part).strip("_") for col in agg.columns.to_flat_index()
    ]
    return agg


def total_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    total = panel.loc[panel["entity_id"] == "Total", list(ENERGY_VARS.values()) + list(WEATHER_VARS.values())]
    return total.corr(method="spearman")


def weather_relationships(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw", "ghg_ton"]:
        for weather_col in ["temperature_f", "humidity_pct", "wind_speed_mph", "pressure_in"]:
            sub = panel.loc[panel["entity_id"] == "Total", [col, weather_col]].dropna()
            rows.append(
                {
                    "response": col,
                    "predictor": weather_col,
                    "spearman_r": float(sub[col].corr(sub[weather_col], method="spearman")),
                    "pearson_r": float(sub[col].corr(sub[weather_col], method="pearson")),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)


def style_plots() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )


def save_overview_figure(panel: pd.DataFrame) -> None:
    total = panel.loc[panel["entity_id"] == "Total"].copy()
    cols = [
        ("electricity_kw", "Electricity [kW]", "#264653"),
        ("heat_mmbtu", "Heat [mmBTU]", "#8ab17d"),
        ("cooling_ton", "Cooling [Ton]", "#e76f51"),
        ("pv_kw", "PV [kW]", "#e9c46a"),
        ("ghg_ton", "GHG [Ton]", "#6d597a"),
        ("temperature_f", "Temperature [F]", "#1d3557"),
    ]
    fig, axes = plt.subplots(len(cols), 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    for ax, (col, label, color) in zip(axes, cols):
        ax.plot(total["datetime"], total[col], color=color, linewidth=0.8)
        ax.set_ylabel(label)
    axes[0].set_title("HEEW Mini-Dataset 2014: district-level hourly trajectories")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("2014")
    fig.savefig(IMAGE_DIR / "figure_01_overview_time_series.png", bbox_inches="tight")
    plt.close(fig)


def save_monthly_profiles(panel: pd.DataFrame) -> None:
    total = panel.loc[panel["entity_id"] == "Total"].copy()
    monthly = total.groupby("month", as_index=False)[list(ENERGY_VARS.values()) + ["temperature_f"]].mean()
    monthly["month_label"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    mappings = [
        ("electricity_kw", "Monthly mean electricity", "#264653"),
        ("heat_mmbtu", "Monthly mean heat", "#8ab17d"),
        ("cooling_ton", "Monthly mean cooling", "#e76f51"),
        ("pv_kw", "Monthly mean PV generation", "#e9c46a"),
        ("ghg_ton", "Monthly mean GHG emissions", "#6d597a"),
        ("temperature_f", "Monthly mean air temperature", "#1d3557"),
    ]
    for ax, (col, title, color) in zip(axes.flat, mappings):
        sns.barplot(data=monthly, x="month_label", y=col, ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.savefig(IMAGE_DIR / "figure_02_monthly_profiles.png", bbox_inches="tight")
    plt.close(fig)


def save_diurnal_profiles(panel: pd.DataFrame) -> None:
    total = panel.loc[panel["entity_id"] == "Total"].copy()
    diurnal = (
        total.groupby(["season", "hour_of_day"], as_index=False)[
            ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw"]
        ]
        .mean()
    )
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    diurnal["season"] = pd.Categorical(diurnal["season"], categories=season_order, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, constrained_layout=True)
    mappings = [
        ("electricity_kw", "Electricity", "#264653"),
        ("heat_mmbtu", "Heat", "#8ab17d"),
        ("cooling_ton", "Cooling", "#e76f51"),
        ("pv_kw", "PV generation", "#e9c46a"),
    ]
    palette = {"Winter": "#355070", "Spring": "#6d597a", "Summer": "#e56b6f", "Fall": "#b56576"}
    for ax, (col, title, _) in zip(axes.flat, mappings):
        sns.lineplot(
            data=diurnal,
            x="hour_of_day",
            y=col,
            hue="season",
            hue_order=season_order,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("")
        ax.legend(title="")
    fig.savefig(IMAGE_DIR / "figure_03_seasonal_diurnal_profiles.png", bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Spearman correlations at district level")
    fig.savefig(IMAGE_DIR / "figure_04_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_temperature_relationships(panel: pd.DataFrame) -> None:
    total = panel.loc[panel["entity_id"] == "Total"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    mappings = [
        ("heat_mmbtu", "Heat vs temperature", "#8ab17d"),
        ("cooling_ton", "Cooling vs temperature", "#e76f51"),
        ("pv_kw", "PV vs temperature", "#e9c46a"),
    ]
    for ax, (col, title, color) in zip(axes, mappings):
        sns.regplot(
            data=total.sample(min(len(total), 2500), random_state=42),
            x="temperature_f",
            y=col,
            lowess=True,
            scatter_kws={"s": 10, "alpha": 0.25, "color": color},
            line_kws={"color": "black", "linewidth": 2},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Temperature [F]")
        ax.set_ylabel("")
    fig.savefig(IMAGE_DIR / "figure_05_temperature_relationships.png", bbox_inches="tight")
    plt.close(fig)


def save_hierarchy_validation_plot(panel: pd.DataFrame) -> None:
    building_sum = (
        panel.loc[panel["level"] == "building", ["datetime", *ENERGY_VARS.values()]]
        .groupby("datetime", as_index=False)
        .sum()
    )
    cn = panel.loc[panel["entity_id"] == "CN01", ["datetime", *ENERGY_VARS.values()]]
    merged = building_sum.merge(cn, on="datetime", suffixes=("_buildings", "_cn"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    mappings = [
        ("electricity_kw", "Electricity", "#264653"),
        ("heat_mmbtu", "Heat", "#8ab17d"),
        ("cooling_ton", "Cooling", "#e76f51"),
        ("pv_kw", "PV generation", "#e9c46a"),
    ]
    for ax, (col, title, color) in zip(axes.flat, mappings):
        x = merged[f"{col}_buildings"]
        y = merged[f"{col}_cn"]
        ax.scatter(x, y, s=10, alpha=0.35, color=color)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Sum of BN001-BN010")
        ax.set_ylabel("CN01")
    fig.savefig(IMAGE_DIR / "figure_06_hierarchy_validation.png", bbox_inches="tight")
    plt.close(fig)


def save_entity_comparison(panel: pd.DataFrame) -> None:
    buildings = panel.loc[panel["level"] == "building"].copy()
    annual = (
        buildings.groupby("entity_id", as_index=False)[
            ["electricity_kw", "heat_mmbtu", "cooling_ton", "pv_kw", "ghg_ton"]
        ]
        .sum()
        .melt(id_vars="entity_id", var_name="variable", value_name="annual_total")
    )
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    sns.barplot(data=annual, x="entity_id", y="annual_total", hue="variable", ax=ax)
    ax.set_title("Annual totals by building")
    ax.set_xlabel("")
    ax.set_ylabel("Annual total")
    ax.legend(title="")
    fig.savefig(IMAGE_DIR / "figure_07_building_annual_totals.png", bbox_inches="tight")
    plt.close(fig)


def save_cleaning_summary(outliers: pd.DataFrame, temporal_qc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    outlier_counts = outliers.groupby(["entity_id", "variable"], as_index=False).size()
    if outlier_counts.empty:
        axes[0].text(0.5, 0.5, "No outliers flagged", ha="center", va="center", fontsize=16)
        axes[0].set_axis_off()
    else:
        pivot = outlier_counts.pivot(index="entity_id", columns="variable", values="size").fillna(0)
        sns.heatmap(pivot, cmap="YlOrRd", ax=axes[0], cbar=False)
        axes[0].set_title("Seasonal outlier flags")
    qc_plot = temporal_qc.melt(id_vars="entity_id", value_vars=["duplicates", "missing_timestamps"])
    sns.barplot(data=qc_plot, x="entity_id", y="value", hue="variable", ax=axes[1])
    axes[1].set_title("Temporal quality checks")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="")
    fig.savefig(IMAGE_DIR / "figure_08_cleaning_audit.png", bbox_inches="tight")
    plt.close(fig)


def export_outputs(panel: pd.DataFrame, clean: pd.DataFrame, outliers: pd.DataFrame) -> dict[str, float | int | str]:
    temporal_qc = temporal_quality_checks(panel)
    missing_qc = missingness_summary(panel)
    negative_qc = negative_value_summary(panel)
    hierarchy_qc = hierarchy_validation(panel)
    summary = entity_summary(panel)
    corr = total_correlation(panel)
    relationships = weather_relationships(panel)

    panel.to_csv(OUTPUT_DIR / "heew_mini_integrated_panel.csv", index=False)
    clean.to_csv(OUTPUT_DIR / "heew_mini_cleaned_panel.csv", index=False)
    temporal_qc.to_csv(OUTPUT_DIR / "temporal_quality_checks.csv", index=False)
    missing_qc.to_csv(OUTPUT_DIR / "missingness_summary.csv", index=False)
    negative_qc.to_csv(OUTPUT_DIR / "negative_value_summary.csv", index=False)
    hierarchy_qc.to_csv(OUTPUT_DIR / "hierarchy_validation.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "entity_summary.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "total_correlation_matrix.csv")
    relationships.to_csv(OUTPUT_DIR / "weather_relationships.csv", index=False)
    outliers.to_csv(OUTPUT_DIR / "seasonal_outlier_flags.csv", index=False)

    style_plots()
    save_overview_figure(panel)
    save_monthly_profiles(panel)
    save_diurnal_profiles(panel)
    save_correlation_heatmap(corr)
    save_temperature_relationships(panel)
    save_hierarchy_validation_plot(panel)
    save_entity_comparison(panel)
    save_cleaning_summary(outliers, temporal_qc)

    metrics = {
        "records": int(len(panel)),
        "entities": int(panel["entity_id"].nunique()),
        "buildings": int((panel["level"] == "building").sum() / 8760),
        "time_steps_per_entity": int(panel.groupby("entity_id").size().iloc[0]),
        "outlier_flags": int(len(outliers)),
        "duplicate_timestamps": int(temporal_qc["duplicates"].sum()),
        "temporal_gap_count": int(temporal_qc["missing_timestamps"].sum()),
        "missing_values": int(missing_qc["missing_count"].sum()),
        "negative_values": int(negative_qc["negative_count"].sum()),
        "cn_equals_total": bool(
            panel.loc[panel["entity_id"] == "CN01", list(ENERGY_VARS.values())].reset_index(drop=True).equals(
                panel.loc[panel["entity_id"] == "Total", list(ENERGY_VARS.values())].reset_index(drop=True)
            )
        ),
    }
    with open(OUTPUT_DIR / "analysis_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    ensure_dirs()
    panel = build_master_panel()
    outliers = seasonal_outlier_flags(panel)
    clean = clean_panel(panel, outliers)
    metrics = export_outputs(panel, clean, outliers)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
