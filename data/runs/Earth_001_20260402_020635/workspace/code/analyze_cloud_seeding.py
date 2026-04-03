#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = ROOT / "data" / "dataset1_cloud_seeding_records" / "cloud_seeding_us_2000_2025.csv"
GEOJSON_PATH = ROOT / "data" / "dataset1_cloud_seeding_records" / "us_states.geojson"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"


STATE_NAME_OVERRIDES = {
    "district of columbia": "District of Columbia",
    "north dakota": "North Dakota",
    "south dakota": "South Dakota",
    "new mexico": "New Mexico",
    "new york": "New York",
    "new jersey": "New Jersey",
    "new hampshire": "New Hampshire",
    "north carolina": "North Carolina",
    "south carolina": "South Carolina",
    "rhode island": "Rhode Island",
    "west virginia": "West Virginia",
}

EXCLUDED_MAP_STATES = {"Alaska", "Hawaii", "Puerto Rico"}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def split_unique_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    tokens: list[str] = []
    for item in str(value).split(","):
        token = clean_text(item)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def title_case_state(name: str) -> str:
    if name in STATE_NAME_OVERRIDES:
        return STATE_NAME_OVERRIDES[name]
    return name.title()


def infer_date(date_value: object, project_year: int) -> pd.Timestamp | pd.NaT:
    if pd.isna(date_value) or str(date_value).strip() == "":
        return pd.NaT

    match = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{2})\s*", str(date_value))
    if not match:
        return pd.NaT

    month, day, short_year = map(int, match.groups())
    candidates = [1900 + short_year, 2000 + short_year, 2100 + short_year]
    chosen_year = min(candidates, key=lambda y: abs(y - project_year))
    try:
        return pd.Timestamp(year=chosen_year, month=month, day=day)
    except ValueError:
        return pd.NaT


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["state"] = clean["state"].map(clean_text)
    clean["state_title"] = clean["state"].map(title_case_state)
    clean["season_normalized"] = clean["season"].map(lambda x: ", ".join(split_unique_tokens(x)))
    clean["purpose_tokens"] = clean["purpose"].map(split_unique_tokens)
    clean["purpose_combo_normalized"] = clean["purpose_tokens"].map(lambda x: " | ".join(sorted(x)))
    clean["agent_tokens"] = clean["agent"].map(split_unique_tokens)
    clean["agent_combo_normalized"] = clean["agent_tokens"].map(lambda x: " | ".join(x))
    clean["apparatus_tokens"] = clean["apparatus"].map(split_unique_tokens)
    clean["apparatus_combo_normalized"] = clean["apparatus_tokens"].map(lambda x: " | ".join(x))
    clean["start_date_inferred"] = clean.apply(
        lambda row: infer_date(row["start_date"], int(row["year"])), axis=1
    )
    clean["end_date_inferred"] = clean.apply(
        lambda row: infer_date(row["end_date"], int(row["year"])), axis=1
    )
    clean["duration_days_inferred"] = (
        clean["end_date_inferred"] - clean["start_date_inferred"]
    ).dt.days + 1
    clean["date_anomaly_flag"] = (
        clean["duration_days_inferred"].notna()
        & ((clean["duration_days_inferred"] < 1) | (clean["duration_days_inferred"] > 370))
    )
    return clean


def explode_token_column(df: pd.DataFrame, column: str, token_name: str) -> pd.DataFrame:
    exploded = (
        df[["filename", "project", "year", "state", "state_title", "operator_affiliation", column]]
        .explode(column)
        .rename(columns={column: token_name})
    )
    exploded = exploded[exploded[token_name].notna() & (exploded[token_name] != "")]
    return exploded.reset_index(drop=True)


def compute_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "records": len(df),
                "years_covered": f"{int(df['year'].min())}-{int(df['year'].max())}",
                "states_with_activity": df["state"].nunique(),
                "unique_projects": df["project"].nunique(),
                "unique_operators": df["operator_affiliation"].nunique(),
                "unique_target_areas": df["target_area"].nunique(dropna=True),
                "records_missing_apparatus": int(df["apparatus"].isna().sum()),
                "records_missing_control_area": int(df["control_area"].isna().sum()),
                "records_with_date_anomalies": int(df["date_anomaly_flag"].sum()),
            }
        ]
    )


def compute_state_table(df: pd.DataFrame) -> pd.DataFrame:
    state_counts = (
        df.groupby(["state", "state_title"])
        .size()
        .reset_index(name="records")
        .sort_values(["records", "state_title"], ascending=[False, True])
    )
    state_counts["share_of_records"] = state_counts["records"] / len(df)
    state_counts["cumulative_share"] = state_counts["share_of_records"].cumsum()
    return state_counts.reset_index(drop=True)


def compute_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    annual = (
        df.groupby("year")
        .agg(
            records=("project", "size"),
            active_states=("state", "nunique"),
            unique_projects=("project", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )
    annual["records_3yr_centered_ma"] = (
        annual["records"].rolling(3, center=True, min_periods=1).mean()
    )
    return annual


def compute_purpose_tables(purpose_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    purpose_counts = (
        purpose_long.groupby("purpose")
        .size()
        .reset_index(name="token_count")
        .sort_values(["token_count", "purpose"], ascending=[False, True])
    )
    period_labels = pd.cut(
        purpose_long["year"],
        bins=[1999, 2005, 2010, 2015, 2020, 2025],
        labels=["2000-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"],
    )
    purpose_long = purpose_long.assign(period=period_labels)
    purpose_period = (
        purpose_long.groupby(["period", "purpose"], observed=False)
        .size()
        .reset_index(name="token_count")
    )
    period_totals = purpose_period.groupby("period", observed=False)["token_count"].transform("sum")
    purpose_period["period_share"] = purpose_period["token_count"] / period_totals
    return purpose_counts.reset_index(drop=True), purpose_period.reset_index(drop=True)


def compute_agent_apparatus_table(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in clean.itertuples(index=False):
        for agent in row.agent_tokens:
            for apparatus in row.apparatus_tokens:
                rows.append(
                    {
                        "agent": agent,
                        "apparatus": apparatus,
                        "state": row.state,
                        "year": row.year,
                    }
                )
    pair_df = pd.DataFrame(rows)
    pair_counts = (
        pair_df.groupby(["agent", "apparatus"])
        .size()
        .reset_index(name="pair_count")
        .sort_values(["pair_count", "agent", "apparatus"], ascending=[False, True, True])
    )
    return pair_counts.reset_index(drop=True)


def compute_state_apparatus_table(clean: pd.DataFrame) -> pd.DataFrame:
    apparatus_long = explode_token_column(clean, "apparatus_tokens", "apparatus")
    state_apparatus = (
        apparatus_long.groupby(["state_title", "apparatus"])
        .size()
        .reset_index(name="token_count")
    )
    totals = state_apparatus.groupby("state_title")["token_count"].transform("sum")
    state_apparatus["state_share"] = state_apparatus["token_count"] / totals
    return state_apparatus.sort_values(
        ["state_title", "token_count"], ascending=[True, False]
    ).reset_index(drop=True)


def compute_validation_table(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    normalized_season = raw["season"].map(lambda x: ", ".join(split_unique_tokens(x)))
    normalized_purpose = raw["purpose"].map(
        lambda x: ", ".join(split_unique_tokens(x))
    )
    checks = [
        ("missing_apparatus", int(raw["apparatus"].isna().sum())),
        ("missing_target_area", int(raw["target_area"].isna().sum())),
        ("missing_start_date", int(raw["start_date"].isna().sum())),
        ("missing_end_date", int(raw["end_date"].isna().sum())),
        ("season_strings_changed_by_normalization", int((raw["season"] != normalized_season).sum())),
        ("purpose_strings_changed_by_normalization", int((raw["purpose"] != normalized_purpose).sum())),
        ("records_with_date_anomalies", int(clean["date_anomaly_flag"].sum())),
    ]
    return pd.DataFrame(checks, columns=["check", "count"])


def write_tables(
    clean: pd.DataFrame,
    overview: pd.DataFrame,
    states: pd.DataFrame,
    annual: pd.DataFrame,
    purpose_counts: pd.DataFrame,
    purpose_period: pd.DataFrame,
    agent_apparatus: pd.DataFrame,
    state_apparatus: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    clean.to_csv(OUTPUT_DIR / "cleaned_cloud_seeding_records.csv", index=False)
    overview.to_csv(OUTPUT_DIR / "table_01_dataset_overview.csv", index=False)
    states.to_csv(OUTPUT_DIR / "table_02_state_counts.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "table_03_annual_activity.csv", index=False)
    purpose_counts.to_csv(OUTPUT_DIR / "table_04_purpose_token_counts.csv", index=False)
    purpose_period.to_csv(OUTPUT_DIR / "table_05_purpose_by_period.csv", index=False)
    agent_apparatus.to_csv(OUTPUT_DIR / "table_06_agent_apparatus_pairs.csv", index=False)
    state_apparatus.to_csv(OUTPUT_DIR / "table_07_state_apparatus_tokens.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "table_08_validation_summary.csv", index=False)


def save_key_findings(overview: pd.DataFrame, states: pd.DataFrame, annual: pd.DataFrame, purpose_counts: pd.DataFrame) -> None:
    peak_year = annual.loc[annual["records"].idxmax()]
    trough_year = annual.loc[annual["records"].idxmin()]
    findings = {
        "records": int(overview.loc[0, "records"]),
        "years_covered": overview.loc[0, "years_covered"],
        "top_5_state_share": round(float(states.head(5)["share_of_records"].sum()), 4),
        "top_8_state_share": round(float(states.head(8)["share_of_records"].sum()), 4),
        "peak_year": {"year": int(peak_year["year"]), "records": int(peak_year["records"])},
        "trough_year": {"year": int(trough_year["year"]), "records": int(trough_year["records"])},
        "leading_purpose_tokens": purpose_counts.head(6).to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "key_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)


def load_geojson() -> dict:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def draw_geojson_state_map(ax: plt.Axes, counts: pd.Series) -> None:
    geojson = load_geojson()
    patches = []
    values = []

    for feature in geojson["features"]:
        state_name = feature["properties"]["name"]
        if state_name in EXCLUDED_MAP_STATES:
            continue
        value = float(counts.get(state_name, 0))
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue

        for polygon_coords in polygons:
            outer_ring = polygon_coords[0]
            patches.append(Polygon(outer_ring, closed=True))
            values.append(value)

    norm = mcolors.Normalize(vmin=0, vmax=max(values) if values else 1)
    collection = PatchCollection(
        patches,
        cmap=mpl.colormaps["YlOrRd"],
        norm=norm,
        linewidth=0.35,
        edgecolor="#4f4f4f",
    )
    collection.set_array(np.array(values))
    ax.add_collection(collection)
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = plt.colorbar(collection, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Project records", fontsize=9)


def plot_spatial_concentration(states: pd.DataFrame) -> None:
    counts = states.set_index("state_title")["records"]
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 1], figure=fig)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_cum = fig.add_subplot(gs[1, 1])

    draw_geojson_state_map(ax_map, counts)
    ax_map.set_title("Reported cloud-seeding activity is concentrated in western states", fontsize=14, pad=12)

    top_states = states.head(8).iloc[::-1]
    ax_bar.barh(top_states["state_title"], top_states["records"], color="#c65d2e")
    ax_bar.set_xlabel("Records")
    ax_bar.set_ylabel("")
    ax_bar.set_title("Top states by reported projects")
    for idx, value in enumerate(top_states["records"]):
        ax_bar.text(value + 2, idx, str(int(value)), va="center", fontsize=9)

    ordered = states["records"].to_numpy()
    cumulative = np.cumsum(ordered) / ordered.sum()
    x = np.arange(1, len(ordered) + 1)
    ax_cum.plot(x, cumulative, color="#184e77", marker="o", linewidth=2)
    ax_cum.axhline(0.8, color="#9e2a2b", linestyle="--", linewidth=1)
    ax_cum.axvline(5, color="#9e2a2b", linestyle="--", linewidth=1)
    ax_cum.set_xlim(1, len(ordered))
    ax_cum.set_ylim(0, 1.02)
    ax_cum.set_xlabel("Top N states")
    ax_cum.set_ylabel("Cumulative share of records")
    ax_cum.set_title("Concentration curve")
    ax_cum.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_01_spatial_concentration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_annual_dynamics(annual: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].bar(annual["year"], annual["records"], color="#7db7d8", edgecolor="#1d3557", linewidth=0.4)
    axes[0].plot(
        annual["year"],
        annual["records_3yr_centered_ma"],
        color="#d1495b",
        linewidth=2.5,
        label="3-year centered mean",
    )
    axes[0].axvspan(2020 - 0.5, 2021 + 0.5, color="#e9ecef", alpha=0.9)
    axes[0].set_title("Annual record counts declined after the mid-2000s peak")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Project records")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(annual["year"], annual["active_states"], color="#2a9d8f", marker="o", linewidth=2)
    axes[1].plot(annual["year"], annual["unique_projects"], color="#264653", marker="s", linewidth=2)
    axes[1].set_title("Geographic breadth narrowed after 2020")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Active states", "Unique projects"], frameon=False)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_02_annual_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_purpose_composition(purpose_long: pd.DataFrame, purpose_counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top = purpose_counts.copy().sort_values("token_count", ascending=True)
    axes[0].barh(top["purpose"], top["token_count"], color="#457b9d")
    axes[0].set_title("Snowpack augmentation dominates purpose tokens")
    axes[0].set_xlabel("Token count")
    axes[0].set_ylabel("")

    period_labels = ["2000-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]
    purpose_period = purpose_long.copy()
    purpose_period["period"] = pd.cut(
        purpose_period["year"],
        bins=[1999, 2005, 2010, 2015, 2020, 2025],
        labels=period_labels,
    )
    table = pd.crosstab(purpose_period["period"], purpose_period["purpose"], normalize="index")
    table = table[
        [col for col in ["augment snowpack", "increase precipitation", "suppress hail", "increase runoff", "suppress fog", "research"] if col in table.columns]
    ]
    bottom = np.zeros(len(table))
    colors = ["#1d3557", "#e76f51", "#f4a261", "#2a9d8f", "#8ecae6", "#6d597a"]
    for color, column in zip(colors, table.columns):
        axes[1].bar(table.index.astype(str), table[column], bottom=bottom, color=color, label=column)
        bottom += table[column].to_numpy()
    axes[1].set_title("Purpose mix shifted toward snowpack-oriented projects")
    axes[1].set_ylabel("Share of purpose tokens")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_03_purpose_composition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_agent_apparatus(clean: pd.DataFrame, state_apparatus: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for row in clean.itertuples(index=False):
        for agent in row.agent_tokens:
            for apparatus in row.apparatus_tokens:
                rows.append({"agent": agent, "apparatus": apparatus})
    pair_df = pd.DataFrame(rows)
    top_agents = pair_df["agent"].value_counts().head(10).index
    heatmap = pd.crosstab(pair_df["agent"], pair_df["apparatus"]).reindex(top_agents)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Agent-apparatus token pairs"},
        ax=axes[0],
    )
    axes[0].set_title("Silver iodide dominates both ground and airborne deployment")
    axes[0].set_xlabel("Apparatus")
    axes[0].set_ylabel("Agent")

    top_states = (
        state_apparatus.groupby("state_title")["token_count"].sum().sort_values(ascending=False).head(8).index
    )
    plot_df = state_apparatus[state_apparatus["state_title"].isin(top_states)].pivot(
        index="state_title", columns="apparatus", values="state_share"
    ).fillna(0)
    plot_df = plot_df.reindex(top_states)
    plot_df = plot_df[[col for col in ["ground", "airborne"] if col in plot_df.columns]]
    plot_df.plot(
        kind="barh",
        stacked=True,
        color=["#4d908e", "#f9844a"][: len(plot_df.columns)],
        ax=axes[1],
    )
    axes[1].set_title("State deployment styles differ sharply")
    axes[1].set_xlabel("Share of apparatus tokens")
    axes[1].set_ylabel("")
    axes[1].legend(frameon=False, title="")
    axes[1].grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_04_agent_apparatus.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_validation_audit(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    missingness = raw.isna().sum().sort_values(ascending=True)
    missingness = missingness[missingness > 0]
    axes[0].barh(missingness.index, missingness.values, color="#adb5bd")
    axes[0].set_title("Missingness is concentrated in control areas")
    axes[0].set_xlabel("Missing values")

    normalization_checks = pd.Series(
        {
            "season strings changed": int((raw["season"] != clean["season_normalized"]).sum()),
            "purpose strings changed": int((raw["purpose"] != clean["purpose"].map(lambda x: ", ".join(split_unique_tokens(x)))).sum()),
            "agent strings changed": int((raw["agent"] != raw["agent"].map(lambda x: ", ".join(split_unique_tokens(x)))).sum()),
        }
    )
    axes[1].bar(normalization_checks.index, normalization_checks.values, color="#90be6d")
    axes[1].set_title("Normalization affected only a few category strings")
    axes[1].set_ylabel("Records")
    axes[1].tick_params(axis="x", rotation=18)

    duration_status = pd.Series(
        {
            "valid inferred dates": int((clean["duration_days_inferred"].between(1, 370)).sum()),
            "date anomalies": int(clean["date_anomaly_flag"].sum()),
            "missing dates": int(
                clean["start_date_inferred"].isna().sum() + clean["end_date_inferred"].isna().sum()
            ),
        }
    )
    axes[2].bar(duration_status.index, duration_status.values, color=["#277da1", "#f94144", "#f9c74f"])
    axes[2].set_title("Date issues are limited but non-zero")
    axes[2].set_ylabel("Counts")
    axes[2].tick_params(axis="x", rotation=18)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_05_validation_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )

    raw = pd.read_csv(DATA_PATH)
    clean = normalize_dataset(raw)

    purpose_long = explode_token_column(clean, "purpose_tokens", "purpose")
    overview = compute_overview_table(clean)
    states = compute_state_table(clean)
    annual = compute_annual_table(clean)
    purpose_counts, purpose_period = compute_purpose_tables(purpose_long)
    agent_apparatus = compute_agent_apparatus_table(clean)
    state_apparatus = compute_state_apparatus_table(clean)
    validation = compute_validation_table(raw, clean)

    write_tables(
        clean,
        overview,
        states,
        annual,
        purpose_counts,
        purpose_period,
        agent_apparatus,
        state_apparatus,
        validation,
    )
    save_key_findings(overview, states, annual, purpose_counts)

    plot_spatial_concentration(states)
    plot_annual_dynamics(annual)
    plot_purpose_composition(purpose_long, purpose_counts)
    plot_agent_apparatus(clean, state_apparatus)
    plot_validation_audit(raw, clean)

    print("Analysis complete.")
    print(f"Tables written to: {OUTPUT_DIR}")
    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
