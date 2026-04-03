"""Reproducible analysis of NOAA cloud-seeding records, 2000–2025.

This script reads the project-level dataset provided in
`data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv`
and produces the tables and figures used in the accompanying report.

Usage (from workspace root):

    python code/analysis_cloud_seeding.py

Outputs:
- Tables in `outputs/`
- Figures in `report/images/`
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data():
    base = Path("data/dataset1_cloud_seeding_records")
    df = pd.read_csv(base / "cloud_seeding_us_2000_2025.csv")
    states_gdf = gpd.read_file(base / "us_states.geojson")
    return df, states_gdf


def ensure_dirs():
    Path("outputs").mkdir(exist_ok=True)
    Path("report/images").mkdir(parents=True, exist_ok=True)


def data_overview(df: pd.DataFrame):
    """Basic descriptive statistics and narrative stats."""
    summary = df.describe(include="all")
    summary.to_csv("outputs/data_overview_summary.csv")

    annual = df.groupby("year").size().reset_index(name="projects")
    by_state = (
        df.groupby("state").size().reset_index(name="projects").sort_values("projects", ascending=False)
    )
    purpose = (
        df.groupby("purpose").size().reset_index(name="projects").sort_values("projects", ascending=False)
    )
    agent = (
        df.groupby("agent").size().reset_index(name="projects").sort_values("projects", ascending=False)
    )
    apparatus = (
        df.groupby("apparatus").size().reset_index(name="projects").sort_values("projects", ascending=False)
    )

    lines = []
    lines.append(f"Total records: {len(df)}")
    lines.append(f"Time span: {df['year'].min()}–{df['year'].max()}")
    lines.append(f"Number of states: {df['state'].nunique()}")
    lines.append(f"Median projects per year: {annual['projects'].median():.1f}")
    max_row = annual.loc[annual["projects"].idxmax()]
    lines.append(f"Max projects in a year: {int(max_row['projects'])} (year {int(max_row['year'])})")

    if not by_state.empty:
        lines.append(
            f"Top state: {by_state.iloc[0]['state']} ({int(by_state.iloc[0]['projects'])} projects)"
        )

    def top_n_str(df_, label, n=3):
        parts = [f"{row[0]} ({int(row[1])})" for row in df_.head(n).itertuples(index=False)]
        return f"Top {n} {label}: " + ", ".join(parts)

    lines.append(top_n_str(purpose, "purposes"))
    lines.append(top_n_str(agent, "agents"))
    lines.append(top_n_str(apparatus, "apparatus types"))

    Path("outputs/narrative_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    # Save key tables for transparency
    annual.to_csv("outputs/annual_activity.csv", index=False)
    by_state.to_csv("outputs/projects_by_state.csv", index=False)
    purpose.to_csv("outputs/purpose_composition.csv", index=False)
    agent.to_csv("outputs/agent_counts.csv", index=False)
    apparatus.to_csv("outputs/apparatus_counts.csv", index=False)


def plot_annual_activity(df: pd.DataFrame):
    annual = df.groupby("year").size().reset_index(name="projects")
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=annual, x="year", y="projects", marker="o")
    plt.title("Annual Cloud-Seeding Projects in the US (2000–2025)")
    plt.xlabel("Year")
    plt.ylabel("Number of projects")
    plt.tight_layout()
    plt.savefig("report/images/annual_activity.png", dpi=300)
    plt.close()


def plot_spatial_concentration(df: pd.DataFrame, states_gdf: gpd.GeoDataFrame):
    by_state = df.groupby("state").size().reset_index(name="projects")

    # Try to match state identifier column
    key = None
    for cand in ["state", "STUSPS", "NAME", "name"]:
        if cand in states_gdf.columns:
            key = cand
            break

    if key is None:
        print("Warning: could not find a matching state column in GeoJSON; skipping map.")
        return

    merged = states_gdf.merge(by_state, left_on=key, right_on="state", how="left")
    merged["projects"] = merged["projects"].fillna(0)

    plt.figure(figsize=(10, 6))
    merged.plot(
        column="projects",
        cmap="viridis",
        linewidth=0.8,
        edgecolor="0.8",
        legend=True,
    )
    plt.title("Cloud-Seeding Projects by State, 2000–2025")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("report/images/spatial_concentration_states.png", dpi=300)
    plt.close()

    # Top states barplot
    top_states = (
        by_state.sort_values("projects", ascending=False).head(15).reset_index(drop=True)
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=top_states, x="state", y="projects")
    plt.title("Top 15 States by Number of Cloud-Seeding Projects")
    plt.xlabel("State")
    plt.ylabel("Number of projects")
    plt.tight_layout()
    plt.savefig("report/images/top_states.png", dpi=300)
    plt.close()


def plot_purpose_composition(df: pd.DataFrame):
    purpose_counts = df["purpose"].value_counts().reset_index()
    purpose_counts.columns = ["purpose", "n"]
    plt.figure(figsize=(8, 4))
    sns.barplot(data=purpose_counts, x="purpose", y="n")
    plt.xticks(rotation=45, ha="right")
    plt.title("Purpose Composition of Cloud-Seeding Projects")
    plt.xlabel("Purpose")
    plt.ylabel("Number of projects")
    plt.tight_layout()
    plt.savefig("report/images/purpose_composition.png", dpi=300)
    plt.close()


def plot_agent_apparatus_patterns(df: pd.DataFrame):
    combo = df.groupby(["agent", "apparatus"]).size().reset_index(name="n")
    combo.to_csv("outputs/agent_apparatus_patterns.csv", index=False)

    pivot = combo.pivot(index="agent", columns="apparatus", values="n").fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=False, cmap="mako")
    plt.title("Agent–Apparatus Deployment Patterns")
    plt.xlabel("Apparatus")
    plt.ylabel("Seeding agent")
    plt.tight_layout()
    plt.savefig("report/images/agent_apparatus_heatmap.png", dpi=300)
    plt.close()


def plot_seasonality(df: pd.DataFrame):
    season_counts = df["season"].value_counts().reset_index()
    season_counts.columns = ["season", "n"]
    plt.figure(figsize=(6, 4))
    sns.barplot(data=season_counts, x="season", y="n")
    plt.title("Projects by Season")
    plt.xlabel("Season")
    plt.ylabel("Number of projects")
    plt.tight_layout()
    plt.savefig("report/images/seasonality.png", dpi=300)
    plt.close()


def main():
    ensure_dirs()
    df, states_gdf = load_data()
    data_overview(df)
    plot_annual_activity(df)
    plot_spatial_concentration(df, states_gdf)
    plot_purpose_composition(df)
    plot_agent_apparatus_patterns(df)
    plot_seasonality(df)


if __name__ == "__main__":
    main()
