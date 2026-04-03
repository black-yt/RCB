#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset1_cloud_seeding_records"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"
CSV_PATH = DATA_DIR / "cloud_seeding_us_2000_2025.csv"
GEOJSON_PATH = DATA_DIR / "us_states.geojson"


REQUIRED_COLUMNS = [
    "filename",
    "project",
    "year",
    "season",
    "state",
    "operator_affiliation",
    "agent",
    "apparatus",
    "purpose",
    "target_area",
    "control_area",
    "start_date",
    "end_date",
]


STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.titleweight"] = "bold"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Not reported")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": "Not reported"})
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [c for c in df.columns if df[c].dtype == object]:
        df[col] = normalize_text(df[col]).str.lower()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    df["state_abbrev"] = df["state"].map(STATE_ABBREV).fillna(df["state"].str.upper().str[:2])
    return df


def split_multi_value(series: pd.Series) -> pd.Series:
    return (
        series.dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .replace({"": np.nan, "not reported": np.nan})
        .dropna()
    )


def write_summary_tables(df: pd.DataFrame) -> dict:
    yearly = df.groupby("year").size().rename("records").reset_index()
    yearly["yoy_change"] = yearly["records"].diff()
    yearly["yoy_pct_change"] = yearly["records"].pct_change() * 100
    yearly.to_csv(OUTPUT_DIR / "annual_activity.csv", index=False)

    state_counts = df.groupby(["state", "state_abbrev"]).size().rename("records").reset_index()
    state_counts = state_counts.sort_values("records", ascending=False)
    state_counts["share_pct"] = state_counts["records"] / len(df) * 100
    state_counts.to_csv(OUTPUT_DIR / "state_counts.csv", index=False)

    operator_counts = df.groupby("operator_affiliation").size().rename("records").reset_index()
    operator_counts = operator_counts.sort_values("records", ascending=False)
    operator_counts["share_pct"] = operator_counts["records"] / len(df) * 100
    operator_counts.to_csv(OUTPUT_DIR / "operator_counts.csv", index=False)

    season_counts = df.groupby("season").size().rename("records").reset_index()
    season_counts = season_counts.sort_values("records", ascending=False)
    season_counts["share_pct"] = season_counts["records"] / len(df) * 100
    season_counts.to_csv(OUTPUT_DIR / "season_counts.csv", index=False)

    purpose_long = split_multi_value(df["purpose"]).rename("purpose")
    purpose_counts = purpose_long.value_counts().rename_axis("purpose").reset_index(name="records")
    purpose_counts["share_pct_of_mentions"] = purpose_counts["records"] / purpose_counts["records"].sum() * 100
    purpose_counts.to_csv(OUTPUT_DIR / "purpose_counts_mentions.csv", index=False)

    agent_long = split_multi_value(df["agent"]).rename("agent")
    agent_counts = agent_long.value_counts().rename_axis("agent").reset_index(name="records")
    agent_counts["share_pct_of_mentions"] = agent_counts["records"] / agent_counts["records"].sum() * 100
    agent_counts.to_csv(OUTPUT_DIR / "agent_counts_mentions.csv", index=False)

    apparatus_long = split_multi_value(df["apparatus"]).rename("apparatus")
    apparatus_counts = apparatus_long.value_counts().rename_axis("apparatus").reset_index(name="records")
    apparatus_counts["share_pct_of_mentions"] = apparatus_counts["records"] / apparatus_counts["records"].sum() * 100
    apparatus_counts.to_csv(OUTPUT_DIR / "apparatus_counts_mentions.csv", index=False)

    purpose_year = (
        pd.crosstab(df["year"], split_multi_value(df.set_index("year")["purpose"]).reset_index()["purpose"])
        if False else None
    )
    purpose_year = (
        df[["year", "purpose"]]
        .assign(purpose=df["purpose"].str.split(","))
        .explode("purpose")
        .assign(purpose=lambda x: x["purpose"].str.strip())
        .query("purpose.notna() and purpose != '' and purpose != 'not reported'", engine="python")
        .groupby(["year", "purpose"])
        .size()
        .rename("records")
        .reset_index()
        .pivot(index="year", columns="purpose", values="records")
        .fillna(0)
        .astype(int)
    )
    purpose_year.to_csv(OUTPUT_DIR / "purpose_by_year.csv")

    agent_apparatus = (
        df[["agent", "apparatus"]]
        .assign(agent=df["agent"].str.split(","), apparatus=df["apparatus"].str.split(","))
        .explode("agent")
        .explode("apparatus")
        .assign(
            agent=lambda x: x["agent"].str.strip(),
            apparatus=lambda x: x["apparatus"].str.strip(),
        )
        .query(
            "agent.notna() and apparatus.notna() and agent != '' and apparatus != '' and agent != 'not reported' and apparatus != 'not reported'",
            engine="python",
        )
        .groupby(["agent", "apparatus"])
        .size()
        .rename("records")
        .reset_index()
        .sort_values("records", ascending=False)
    )
    agent_apparatus.to_csv(OUTPUT_DIR / "agent_apparatus_pairs.csv", index=False)

    overview = {
        "n_records": int(len(df)),
        "n_projects": int(df["project"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "n_years": int(df["year"].nunique()),
        "n_states": int(df["state"].nunique()),
        "top_state": state_counts.iloc[0].to_dict(),
        "top_operator": operator_counts.iloc[0].to_dict(),
        "top_purpose_mention": purpose_counts.iloc[0].to_dict(),
        "top_agent_mention": agent_counts.iloc[0].to_dict(),
        "top_apparatus_mention": apparatus_counts.iloc[0].to_dict(),
        "records_missing_control_area": int((df["control_area"] == "not reported").sum()),
        "records_missing_duration": int(df["duration_days"].isna().sum()),
        "median_duration_days": float(df["duration_days"].dropna().median()),
    }
    (OUTPUT_DIR / "analysis_overview.json").write_text(json.dumps(overview, indent=2))

    top_lines = []
    top_lines.append(f"Records: {overview['n_records']}")
    top_lines.append(f"Years covered: {overview['year_min']}-{overview['year_max']} ({overview['n_years']} years)")
    top_lines.append(f"States represented: {overview['n_states']}")
    top_lines.append(f"Top state: {overview['top_state']['state']} ({overview['top_state']['records']} records, {overview['top_state']['share_pct']:.1f}%)")
    top_lines.append(f"Top operator: {overview['top_operator']['operator_affiliation']} ({overview['top_operator']['records']} records)")
    top_lines.append(f"Top purpose mention: {overview['top_purpose_mention']['purpose']} ({overview['top_purpose_mention']['records']} mentions)")
    top_lines.append(f"Top agent mention: {overview['top_agent_mention']['agent']} ({overview['top_agent_mention']['records']} mentions)")
    top_lines.append(f"Top apparatus mention: {overview['top_apparatus_mention']['apparatus']} ({overview['top_apparatus_mention']['records']} mentions)")
    (OUTPUT_DIR / "key_findings_snapshot.txt").write_text("\n".join(top_lines) + "\n")

    return {
        "yearly": yearly,
        "state_counts": state_counts,
        "operator_counts": operator_counts,
        "season_counts": season_counts,
        "purpose_counts": purpose_counts,
        "agent_counts": agent_counts,
        "apparatus_counts": apparatus_counts,
        "purpose_year": purpose_year,
        "agent_apparatus": agent_apparatus,
    }


def _extract_polygons(geometry: dict) -> Iterable[list[list[float]]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        for ring in coords[:1]:
            yield ring
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon[:1]:
                yield ring


def plot_state_choropleth(state_counts: pd.DataFrame) -> None:
    geo = json.loads(GEOJSON_PATH.read_text())
    count_map = dict(zip(state_counts["state"], state_counts["records"]))
    patches = []
    values = []
    labels = []

    for feature in geo["features"]:
        name = str(feature["properties"].get("name", "")).strip().lower()
        value = float(count_map.get(name, 0))
        for ring in _extract_polygons(feature["geometry"]):
            arr = np.asarray(ring)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            patches.append(Polygon(arr[:, :2], closed=True))
            values.append(value)
            labels.append(name)

    fig, ax = plt.subplots(figsize=(16, 10))
    norm = mcolors.Normalize(vmin=0, vmax=max(values) if values else 1)
    collection = PatchCollection(patches, cmap="Blues", norm=norm, edgecolor="white", linewidth=0.5)
    collection.set_array(np.array(values))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(collection, ax=ax, fraction=0.028, pad=0.01)
    cbar.set_label("Number of reported projects")
    ax.set_title("Spatial concentration of reported U.S. cloud-seeding records, 2000–2025")

    for _, row in state_counts.head(10).iterrows():
        abbrev = row["state_abbrev"]
        target_name = row["state"]
        coords = []
        for feature in geo["features"]:
            name = str(feature["properties"].get("name", "")).strip().lower()
            if name == target_name:
                for ring in _extract_polygons(feature["geometry"]):
                    arr = np.asarray(ring)
                    if arr.ndim == 2 and arr.shape[0] >= 3:
                        coords.append(arr[:, :2])
        if coords:
            stacked = np.vstack(coords)
            x, y = stacked[:, 0].mean(), stacked[:, 1].mean()
            ax.text(x, y, abbrev, ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_spatial_concentration_map.png", bbox_inches="tight")
    plt.close(fig)


def plot_annual_activity(yearly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=yearly, x="year", y="records", marker="o", linewidth=2.5, ax=ax, color="#1f77b4")
    ax.fill_between(yearly["year"], yearly["records"], alpha=0.15, color="#1f77b4")
    ax.set_title("Annual reported cloud-seeding activity in the United States")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of records")
    ax.set_xticks(yearly["year"])
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_annual_activity.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_states(state_counts: pd.DataFrame) -> None:
    top = state_counts.head(12).sort_values("records")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="records", y="state", palette="Blues_r", ax=ax)
    ax.set_title("Top states by number of reported cloud-seeding records")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("State")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_top_states.png", bbox_inches="tight")
    plt.close(fig)


def plot_purpose_composition(purpose_counts: pd.DataFrame) -> None:
    top = purpose_counts.head(8).copy()
    other = purpose_counts.iloc[8:]["records"].sum()
    if other > 0:
        top = pd.concat([top, pd.DataFrame([{"purpose": "other", "records": other}])], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("Set2", n_colors=len(top))
    ax.pie(top["records"], labels=top["purpose"], autopct=lambda p: f"{p:.1f}%", startangle=90, colors=colors, textprops={"fontsize": 10})
    ax.set_title("Composition of stated cloud-seeding purposes\n(based on purpose mentions)")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_purpose_composition.png", bbox_inches="tight")
    plt.close(fig)


def plot_purpose_by_year(purpose_year: pd.DataFrame) -> None:
    top_cols = purpose_year.sum(axis=0).sort_values(ascending=False).head(5).index.tolist()
    subset = purpose_year[top_cols]
    fig, ax = plt.subplots(figsize=(12, 7))
    subset.plot(kind="area", stacked=True, ax=ax, alpha=0.85, cmap="tab20c")
    ax.set_title("Annual dynamics of the main stated purposes")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of records mentioning each purpose")
    ax.legend(title="Purpose", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_purpose_by_year.png", bbox_inches="tight")
    plt.close(fig)


def plot_agent_apparatus_heatmap(agent_apparatus: pd.DataFrame) -> None:
    top_agents = agent_apparatus.groupby("agent")["records"].sum().sort_values(ascending=False).head(8).index
    top_apparatus = agent_apparatus.groupby("apparatus")["records"].sum().sort_values(ascending=False).head(6).index
    mat = (
        agent_apparatus[agent_apparatus["agent"].isin(top_agents) & agent_apparatus["apparatus"].isin(top_apparatus)]
        .pivot(index="agent", columns="apparatus", values="records")
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Record count"}, ax=ax)
    ax.set_title("Agent–apparatus deployment patterns")
    ax.set_xlabel("Deployment apparatus")
    ax.set_ylabel("Seeding agent")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_agent_apparatus_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_apparatus_mix(apparatus_counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=apparatus_counts, x="apparatus", y="records", palette="crest", ax=ax)
    ax.set_title("Deployment apparatus mentions across reported records")
    ax.set_xlabel("Apparatus")
    ax.set_ylabel("Number of mentions")
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "figure_apparatus_mix.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = load_data()
    tables = write_summary_tables(df)
    plot_state_choropleth(tables["state_counts"])
    plot_top_states(tables["state_counts"])
    plot_annual_activity(tables["yearly"])
    plot_purpose_composition(tables["purpose_counts"])
    plot_purpose_by_year(tables["purpose_year"])
    plot_agent_apparatus_heatmap(tables["agent_apparatus"])
    plot_apparatus_mix(tables["apparatus_counts"])
    print("Analysis complete. Outputs written to outputs/ and report/images/.")


if __name__ == "__main__":
    main()
