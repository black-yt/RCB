import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "HEEW_Mini-Dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "report" / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", context="talk")


def load_energy_level(level: str) -> pd.DataFrame:
    """Load energy data for a given level (BN001..BN010, CN01, Total)."""
    fname = f"{level}_energy.csv"
    df = pd.read_csv(DATA_DIR / fname)
    # build datetime index
    dt = pd.to_datetime(df[["year", "month", "day", "hour"]].rename(
        columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
    ))
    df["datetime"] = dt
    df = df.set_index("datetime").sort_index()
    return df


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "Total_weather.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def basic_quality_checks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Compute basic data quality metrics and return as one-row DataFrame."""
    metrics = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_missing": df.isna().sum().sum(),
        "frac_missing": df.isna().sum().sum() / (df.size),
    }
    # simple range checks for energy variables if present
    for col in df.columns:
        if df[col].dtype != "float64" and df[col].dtype != "int64":
            continue
        metrics[f"{col}_min"] = df[col].min()
        metrics[f"{col}_max"] = df[col].max()
    out = pd.DataFrame(metrics, index=[name])
    return out


def check_hierarchical_consistency(components, aggregate, vars_cols):
    """Compare sum of components with aggregate for given numeric columns.

    Returns DataFrame with MAE and MAPE for each variable.
    """
    comp_sum = sum(components.values())
    comp_sum = comp_sum[vars_cols].copy()
    agg = aggregate[vars_cols].loc[comp_sum.index]

    results = {}
    for col in vars_cols:
        diff = comp_sum[col] - agg[col]
        mae = diff.abs().mean()
        mape = (diff.abs() / (agg[col].replace(0, np.nan))).mean()
        results[col] = {"MAE": mae, "MAPE": mape}
    return pd.DataFrame(results).T


def plot_time_series(df: pd.DataFrame, cols, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    df[cols].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=300)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, cols, title: str, fname: str):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=300)
    plt.close(fig)


def main():
    # Load building-level energy data
    levels = [f"BN{i:03d}" for i in range(1, 11)] + ["CN01", "Total"]
    energy_data = {lvl: load_energy_level(lvl) for lvl in levels}
    weather = load_weather()

    # Align weather to Total energy (full-year hourly, 8760 rows expected)
    total = energy_data["Total"]
    weather_aligned = weather.reindex(total.index)

    # Basic quality summaries
    quality_rows = []
    for lvl, df in energy_data.items():
        quality_rows.append(basic_quality_checks(df, lvl))
    quality_rows.append(basic_quality_checks(weather_aligned, "Weather"))
    quality = pd.concat(quality_rows)
    quality.to_csv(OUTPUT_DIR / "basic_quality_summary.csv")

    # Hierarchical consistency: sum of 10 BN vs CN01 vs Total
    bns = {lvl: energy_data[lvl][["Electricity [kW]", "Heat [mmBTU]", "Cooling Energy [Ton]", "PV Power Generation [kW]", "Greenhouse Gas Emission [Ton]"]]
           for lvl in levels if lvl.startswith("BN")}
    cn01 = energy_data["CN01"][list(bns[next(iter(bns))].columns)]
    total_energy = total[list(bns[next(iter(bns))].columns)]

    cons_bn_vs_cn01 = check_hierarchical_consistency(bns, cn01, cn01.columns)
    cons_bn_vs_total = check_hierarchical_consistency(bns, total_energy, total_energy.columns)

    cons_bn_vs_cn01.to_csv(OUTPUT_DIR / "hierarchy_consistency_bn_vs_cn01.csv")
    cons_bn_vs_total.to_csv(OUTPUT_DIR / "hierarchy_consistency_bn_vs_total.csv")

    # Example cleaning: forward-fill any rare missing values after inspecting
    cleaned_total = total.copy()
    cleaned_total = cleaned_total.ffill().bfill()
    cleaned_total.to_csv(OUTPUT_DIR / "Total_energy_cleaned.csv")

    # Merge Total energy with weather for correlation analysis
    merged = cleaned_total.join(weather_aligned, how="left")
    merged.to_csv(OUTPUT_DIR / "Total_energy_weather_merged.csv")

    # Plot: annual load profiles (electricity, heat, cooling, PV)
    plot_time_series(
        cleaned_total,
        ["Electricity [kW]", "Heat [mmBTU]", "Cooling Energy [Ton]", "PV Power Generation [kW]"],
        "Total energy profiles in 2014",
        "fig_total_energy_profiles.png",
    )

    # Plot: daily average electricity
    daily_elec = cleaned_total["Electricity [kW]"].resample("D").mean()
    plot_time_series(
        daily_elec.to_frame(),
        ["Electricity [kW]"],
        "Daily mean electricity load (Total, 2014)",
        "fig_daily_electricity_total.png",
    )

    # Plot: PV vs irradiance proxy (temperature as simple proxy here) and other correlations
    corr_cols = [
        "Electricity [kW]",
        "Heat [mmBTU]",
        "Cooling Energy [Ton]",
        "PV Power Generation [kW]",
        "Greenhouse Gas Emission [Ton]",
        "Temperature [°F]",
        "Humidity [%]",
        "Wind Speed [mph]",
        "Pressure [in]",
        "Precipitation [in]",
    ]
    corr_cols = [c for c in corr_cols if c in merged.columns]
    plot_correlation(
        merged.dropna()[corr_cols],
        corr_cols,
        "Correlation between multi-energy loads and weather (Total, 2014)",
        "fig_corr_energy_weather_total.png",
    )

    # Plot: scatter PV vs temperature and electricity vs temperature
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(
        data=merged,
        x="Temperature [°F]",
        y="PV Power Generation [kW]",
        alpha=0.3,
        ax=axes[0],
    )
    axes[0].set_title("PV generation vs temperature")

    sns.scatterplot(
        data=merged,
        x="Temperature [°F]",
        y="Electricity [kW]",
        alpha=0.3,
        ax=axes[1],
    )
    axes[1].set_title("Electricity load vs temperature")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scatter_pv_elec_temperature.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
