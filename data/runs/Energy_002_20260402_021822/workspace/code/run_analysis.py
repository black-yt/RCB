import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"
ASSUMPTIONS_PATH = ROOT / "code" / "assumptions.json"

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", context="talk")


def crf(wacc: float, years: int) -> float:
    if wacc == 0:
        return 1 / years
    return wacc * (1 + wacc) ** years / ((1 + wacc) ** years - 1)


def annualized_cost(capex: float, wacc: float, lifetime: int, fom_share: float = 0.0) -> float:
    return capex * (crf(wacc, lifetime) + fom_share)


def load_inputs():
    with open(ASSUMPTIONS_PATH, "r", encoding="utf-8") as f:
        assumptions = json.load(f)

    sites = pd.read_csv(DATA_DIR / "hex_final_NA_min.csv")
    world = gpd.read_file(DATA_DIR / "africa_map" / "ne_10m_admin_0_countries.shp")
    geometry = [Point(xy) for xy in zip(sites["lon"], sites["lat"])]
    sites_gdf = gpd.GeoDataFrame(sites.copy(), geometry=geometry, crs="EPSG:4326")
    countries = world[["ADMIN", "CONTINENT", "geometry"]].copy()
    sites_gdf = sites_gdf.sjoin(countries, how="left", predicate="within").drop(columns=["index_right"])
    return assumptions, sites_gdf, world


def map_resource_scores_to_cf(df: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    pv = assumptions["technology"]["pv"]
    wind = assumptions["technology"]["wind"]
    out = df.copy()
    out["pv_cf"] = pv["cf_intercept"] + pv["cf_slope"] * out["theo_pv"]
    out["wind_cf"] = wind["cf_intercept"] + wind["cf_slope"] * out["theo_wind"]
    return out


def renewable_lcoe(capex_per_kw: float, cf_value: float, wacc: float, lifetime: int, fom_share: float) -> float:
    annual_cost = annualized_cost(capex_per_kw, wacc, lifetime, fom_share)
    annual_mwh = 8760 * cf_value / 1000
    return annual_cost / annual_mwh


def choose_hybrid_mix(pv_cf: float, wind_cf: float, region: str, scenario: dict, assumptions: dict) -> dict:
    tech = assumptions["technology"]
    general = assumptions["general"]
    if region == "africa":
        wacc = scenario["africa_wacc"]
        pv_capex = tech["pv"]["capex_eur_per_kw_africa"]
        wind_capex = tech["wind"]["capex_eur_per_kw_africa"]
        electrolyzer_capex = tech["electrolyzer"]["capex_eur_per_kw_africa"] * scenario["electrolyzer_capex_multiplier_africa"]
    else:
        wacc = scenario["europe_wacc"]
        pv_capex = tech["pv"]["capex_eur_per_kw_europe"]
        wind_capex = tech["wind"]["capex_eur_per_kw_europe"]
        electrolyzer_capex = tech["electrolyzer"]["capex_eur_per_kw_europe"] * scenario["electrolyzer_capex_multiplier_europe"]

    pv_lcoe = renewable_lcoe(
        pv_capex, pv_cf, wacc, tech["pv"]["lifetime_years"], tech["pv"]["fixed_opex_share"]
    )
    wind_lcoe = renewable_lcoe(
        wind_capex, wind_cf, wacc, tech["wind"]["lifetime_years"], tech["wind"]["fixed_opex_share"]
    )

    best = None
    for pv_share in np.linspace(0, 1, 21):
        wind_share = 1 - pv_share
        avg_cf = pv_share * pv_cf + wind_share * wind_cf
        complementarity = min(pv_share * pv_cf, wind_share * wind_cf)
        electrolyzer_cf = min(0.82, 1.25 * avg_cf + 0.18 * complementarity)
        if electrolyzer_cf <= 0:
            continue
        oversizing_factor = max(1.0, electrolyzer_cf / max(avg_cf, 1e-6))
        electricity_cost_eur_per_kg = (
            (pv_share * pv_lcoe + wind_share * wind_lcoe)
            * oversizing_factor
            * general["electrolyzer_specific_energy_kwh_per_kg"]
            / 1000
        )
        annual_kg_per_kw = electrolyzer_cf * 8760 / general["electrolyzer_specific_energy_kwh_per_kg"]
        electrolyzer_cost_eur_per_kg = annualized_cost(
            electrolyzer_capex,
            wacc,
            tech["electrolyzer"]["lifetime_years"],
            tech["electrolyzer"]["fixed_opex_share"],
        ) / annual_kg_per_kw
        total = electricity_cost_eur_per_kg + electrolyzer_cost_eur_per_kg
        candidate = {
            "pv_share": pv_share,
            "wind_share": wind_share,
            "pv_lcoe_eur_per_mwh": pv_lcoe,
            "wind_lcoe_eur_per_mwh": wind_lcoe,
            "avg_cf": avg_cf,
            "electrolyzer_cf": electrolyzer_cf,
            "oversizing_factor": oversizing_factor,
            "electricity_cost_eur_per_kg": electricity_cost_eur_per_kg,
            "electrolyzer_cost_eur_per_kg": electrolyzer_cost_eur_per_kg,
            "production_core_eur_per_kg": total,
        }
        if best is None or candidate["production_core_eur_per_kg"] < best["production_core_eur_per_kg"]:
            best = candidate

    return best


def water_cost_per_kg(wacc: float, water_distance_km: float, ocean_distance_km: float, assumptions: dict, scenario: dict) -> float:
    infra = assumptions["infrastructure"]
    general = assumptions["general"]
    annual_kg = general["export_scale_kt_h2_per_year"] * 1_000_000
    water_m3_per_kg = infra["water_use_l_per_kg_h2"] / 1000

    freshwater_pipeline = annualized_cost(
        infra["water_pipeline_capex_eur_per_km"] * water_distance_km * scenario["infra_capex_multiplier"],
        wacc,
        infra["water_pipeline_lifetime_years"],
        infra["water_pipeline_fom_share"],
    ) / annual_kg
    freshwater_total = freshwater_pipeline + water_m3_per_kg * infra["freshwater_treatment_eur_per_m3"]

    desal_pipeline = annualized_cost(
        infra["water_pipeline_capex_eur_per_km"] * ocean_distance_km * scenario["infra_capex_multiplier"],
        wacc,
        infra["water_pipeline_lifetime_years"],
        infra["water_pipeline_fom_share"],
    ) / annual_kg
    desal_total = desal_pipeline + water_m3_per_kg * infra["desalination_eur_per_m3"]

    return min(freshwater_total, desal_total)


def ammonia_export_chain_costs(row: pd.Series, scenario: dict, assumptions: dict) -> dict:
    general = assumptions["general"]
    infra = assumptions["infrastructure"]
    conversion = assumptions["conversion_chain"]
    annual_kg = general["export_scale_kt_h2_per_year"] * 1_000_000
    wacc = scenario["africa_wacc"]

    road_spur_cost = annualized_cost(
        infra["road_capex_eur_per_km"] * row["road_dist_km"] * scenario["infra_capex_multiplier"],
        wacc,
        infra["road_lifetime_years"],
        infra["road_fom_share"],
    ) / annual_kg
    inland_transport = road_spur_cost + infra["road_trucking_eur_per_kg_per_km_nh3"] * row["ocean_dist_km"]

    shipping = conversion["shipping_eur_per_kg_per_km"] * general["africa_to_rotterdam_km"]
    return {
        "road_spur_eur_per_kg": road_spur_cost,
        "inland_ammonia_transport_eur_per_kg": inland_transport,
        "ammonia_synthesis_eur_per_kg": conversion["ammonia_synthesis_eur_per_kg_h2"] * scenario["conversion_capex_multiplier"],
        "export_terminal_eur_per_kg": conversion["export_terminal_eur_per_kg_h2"] * scenario["conversion_capex_multiplier"],
        "shipping_eur_per_kg": shipping,
        "ammonia_cracking_eur_per_kg": conversion["ammonia_cracking_eur_per_kg_h2"] * scenario["conversion_capex_multiplier"],
        "eu_distribution_eur_per_kg": conversion["eu_distribution_eur_per_kg_h2"],
    }


def model_african_sites(sites: gpd.GeoDataFrame, assumptions: dict) -> pd.DataFrame:
    records = []
    for scenario_name, scenario in assumptions["scenarios"].items():
        for _, row in sites.iterrows():
            hybrid = choose_hybrid_mix(row["pv_cf"], row["wind_cf"], "africa", scenario, assumptions)
            water = water_cost_per_kg(
                scenario["africa_wacc"], row["waterbody_dist_km"], row["ocean_dist_km"], assumptions, scenario
            )
            export_chain = ammonia_export_chain_costs(row, scenario, assumptions)

            rec = row.drop(labels="geometry").to_dict()
            rec.update(hybrid)
            rec.update(export_chain)
            rec["scenario"] = scenario_name
            rec["scenario_label"] = scenario["label"]
            rec["water_cost_eur_per_kg"] = water
            rec["buffer_storage_eur_per_kg"] = assumptions["technology"]["buffer_storage"]["eur_per_kg_h2"]
            rec["site_production_eur_per_kg"] = (
                rec["electricity_cost_eur_per_kg"]
                + rec["electrolyzer_cost_eur_per_kg"]
                + rec["water_cost_eur_per_kg"]
                + rec["buffer_storage_eur_per_kg"]
            )
            rec["delivered_to_europe_eur_per_kg"] = (
                rec["site_production_eur_per_kg"]
                + rec["ammonia_synthesis_eur_per_kg"]
                + rec["inland_ammonia_transport_eur_per_kg"]
                + rec["export_terminal_eur_per_kg"]
                + rec["shipping_eur_per_kg"]
                + rec["ammonia_cracking_eur_per_kg"]
                + rec["eu_distribution_eur_per_kg"]
            )
            records.append(rec)
    return pd.DataFrame(records)


def model_europe_benchmarks(assumptions: dict) -> pd.DataFrame:
    records = []
    for scenario_name, scenario in assumptions["scenarios"].items():
        for benchmark in assumptions["europe_benchmarks"]:
            hybrid = choose_hybrid_mix(benchmark["pv_cf"], benchmark["wind_cf"], "europe", scenario, assumptions)
            water = water_cost_per_kg(
                scenario["europe_wacc"],
                benchmark["water_distance_km"],
                benchmark["water_distance_km"],
                assumptions,
                scenario,
            )
            annual_kg = assumptions["general"]["export_scale_kt_h2_per_year"] * 1_000_000
            road_spur = annualized_cost(
                assumptions["infrastructure"]["road_capex_eur_per_km"]
                * benchmark["road_distance_km"]
                * scenario["infra_capex_multiplier"],
                scenario["europe_wacc"],
                assumptions["infrastructure"]["road_lifetime_years"],
                assumptions["infrastructure"]["road_fom_share"],
            ) / annual_kg
            rec = {
                "scenario": scenario_name,
                "scenario_label": scenario["label"],
                "benchmark": benchmark["name"],
                "pv_cf": benchmark["pv_cf"],
                "wind_cf": benchmark["wind_cf"],
                "water_cost_eur_per_kg": water,
                "buffer_storage_eur_per_kg": assumptions["technology"]["buffer_storage"]["eur_per_kg_h2"] + 0.08,
                "local_delivery_eur_per_kg": 0.08 + road_spur,
            }
            rec.update(hybrid)
            rec["produced_in_europe_eur_per_kg"] = (
                rec["electricity_cost_eur_per_kg"]
                + rec["electrolyzer_cost_eur_per_kg"]
                + rec["water_cost_eur_per_kg"]
                + rec["buffer_storage_eur_per_kg"]
                + rec["local_delivery_eur_per_kg"]
            )
            records.append(rec)
    return pd.DataFrame(records)


def summarize_results(site_results: pd.DataFrame, eu_results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for scenario in site_results["scenario"].unique():
        subset = site_results[site_results["scenario"] == scenario].copy()
        eu_subset = eu_results[eu_results["scenario"] == scenario].copy()
        best_site = subset.nsmallest(1, "delivered_to_europe_eur_per_kg").iloc[0]
        row = {
            "scenario": scenario,
            "scenario_label": best_site["scenario_label"],
            "best_hex_id": best_site["hex_id"],
            "best_country": best_site.get("ADMIN"),
            "best_delivered_cost_eur_per_kg": best_site["delivered_to_europe_eur_per_kg"],
            "median_delivered_cost_eur_per_kg": subset["delivered_to_europe_eur_per_kg"].median(),
            "p10_delivered_cost_eur_per_kg": subset["delivered_to_europe_eur_per_kg"].quantile(0.10),
            "p90_delivered_cost_eur_per_kg": subset["delivered_to_europe_eur_per_kg"].quantile(0.90),
        }
        for _, eu_row in eu_subset.iterrows():
            metric = eu_row["benchmark"].lower().replace(" ", "_")
            row[f"{metric}_eur_per_kg"] = eu_row["produced_in_europe_eur_per_kg"]
            row[f"share_beating_{metric}"] = (
                subset["delivered_to_europe_eur_per_kg"] <= eu_row["produced_in_europe_eur_per_kg"]
            ).mean()
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def validation_table(site_results: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    best_base = site_results[site_results["scenario"] == "base"]["delivered_to_europe_eur_per_kg"].min()
    best_corridor = site_results[site_results["scenario"] == "corridor_policy"]["delivered_to_europe_eur_per_kg"].min()
    return pd.DataFrame(
        [
            {
                "reference_case": "Kenya export to Rotterdam in Müller et al. (2023)",
                "reference_cost_eur_per_kg": 7.0,
                "model_case": "Best available African sample, base 2030",
                "model_cost_eur_per_kg": best_base,
                "comment": "Lower than 2020s Kenya case because this study assumes 2030 CAPEX learning and a high-quality southern African sample."
            },
            {
                "reference_case": "Kenya export to Rotterdam in Müller et al. (2023)",
                "reference_cost_eur_per_kg": 7.0,
                "model_case": "Best available African sample, corridor policy 2030",
                "model_cost_eur_per_kg": best_corridor,
                "comment": "Illustrates the combined effect of technology learning, lower WACC, and corridor support on delivered cost."
            },
        ]
    )


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(exist_ok=True)


def save_outputs(site_results: pd.DataFrame, eu_results: pd.DataFrame, summary: pd.DataFrame, validation: pd.DataFrame):
    site_results.to_csv(OUTPUT_DIR / "site_results_by_scenario.csv", index=False)
    eu_results.to_csv(OUTPUT_DIR / "europe_benchmarks_by_scenario.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)


def plot_data_overview(sites: gpd.GeoDataFrame, world: gpd.GeoDataFrame):
    africa = world[world["CONTINENT"] == "Africa"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    africa.plot(ax=axes[0], color="#f2efe8", edgecolor="#9c927f", linewidth=0.5)
    scatter = axes[0].scatter(
        sites["lon"],
        sites["lat"],
        c=sites["theo_pv"],
        s=70 + 80 * sites["theo_wind"],
        cmap="YlOrBr",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_title("Candidate African hydrogen sites")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cbar = fig.colorbar(scatter, ax=axes[0], fraction=0.035, pad=0.02)
    cbar.set_label("Solar resource score")

    dist_cols = ["grid_dist_km", "road_dist_km", "ocean_dist_km", "waterbody_dist_km"]
    dist_df = sites[dist_cols].melt(var_name="metric", value_name="km")
    sns.boxplot(data=dist_df, x="metric", y="km", hue="metric", ax=axes[1], palette="crest", legend=False)
    axes[1].set_title("Infrastructure-distance distribution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Distance (km)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_1_data_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_map(site_results: pd.DataFrame, world: gpd.GeoDataFrame):
    africa = world[world["CONTINENT"] == "Africa"]
    base = site_results[site_results["scenario"] == "base"].copy()
    fig, ax = plt.subplots(figsize=(10, 9))
    africa.plot(ax=ax, color="#faf7f0", edgecolor="#ad9f8c", linewidth=0.4)
    sc = ax.scatter(
        base["lon"],
        base["lat"],
        c=base["delivered_to_europe_eur_per_kg"],
        cmap="viridis_r",
        s=95,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Base-scenario delivered green hydrogen cost to Europe")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("EUR/kg H2")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_2_baseline_cost_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_competitiveness(site_results: pd.DataFrame, eu_results: pd.DataFrame):
    summary = []
    for scenario in site_results["scenario"].unique():
        africa_cost = site_results[site_results["scenario"] == scenario]["delivered_to_europe_eur_per_kg"]
        for _, row in eu_results[eu_results["scenario"] == scenario].iterrows():
            summary.append(
                {
                    "scenario_label": row["scenario_label"],
                    "benchmark": row["benchmark"],
                    "europe_cost": row["produced_in_europe_eur_per_kg"],
                    "africa_p10": africa_cost.quantile(0.10),
                    "africa_median": africa_cost.median(),
                    "africa_min": africa_cost.min(),
                }
            )
    plot_df = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(plot_df))
    ax.bar(x - 0.25, plot_df["africa_min"], width=0.25, label="Africa best site", color="#1b9e77")
    ax.bar(x, plot_df["africa_p10"], width=0.25, label="Africa p10 site", color="#66a61e")
    ax.bar(x + 0.25, plot_df["europe_cost"], width=0.25, label="Europe benchmark", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s}\nvs {b}" for s, b in zip(plot_df["scenario_label"], plot_df["benchmark"])],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("EUR/kg H2")
    ax.set_title("African imports versus European green hydrogen benchmarks")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_3_competitiveness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_sensitivity(summary: pd.DataFrame):
    long_df = summary[
        ["scenario_label", "best_delivered_cost_eur_per_kg", "median_delivered_cost_eur_per_kg", "iberia_eur_per_kg", "north_sea_eur_per_kg", "central_europe_eur_per_kg"]
    ].melt("scenario_label", var_name="metric", value_name="eur_per_kg")
    label_map = {
        "best_delivered_cost_eur_per_kg": "Africa best",
        "median_delivered_cost_eur_per_kg": "Africa median",
        "iberia_eur_per_kg": "Europe Iberia",
        "north_sea_eur_per_kg": "Europe North Sea",
        "central_europe_eur_per_kg": "Europe Central",
    }
    long_df["metric"] = long_df["metric"].map(label_map)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=long_df, x="scenario_label", y="eur_per_kg", hue="metric", marker="o", linewidth=2.5, ax=ax)
    ax.set_title("Financing and policy scenarios shift import competitiveness")
    ax.set_xlabel("")
    ax.set_ylabel("EUR/kg H2")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_4_scenario_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(site_results: pd.DataFrame):
    top = (
        site_results.sort_values(["scenario", "delivered_to_europe_eur_per_kg"])
        .groupby("scenario", as_index=False)
        .first()
    )
    breakdown_cols = [
        "electricity_cost_eur_per_kg",
        "electrolyzer_cost_eur_per_kg",
        "water_cost_eur_per_kg",
        "buffer_storage_eur_per_kg",
        "ammonia_synthesis_eur_per_kg",
        "inland_ammonia_transport_eur_per_kg",
        "export_terminal_eur_per_kg",
        "shipping_eur_per_kg",
        "ammonia_cracking_eur_per_kg",
        "eu_distribution_eur_per_kg",
    ]
    plot_df = top[["scenario_label"] + breakdown_cols].melt("scenario_label", var_name="component", value_name="eur_per_kg")
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot = plot_df.pivot(index="scenario_label", columns="component", values="eur_per_kg")
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", width=0.72)
    ax.set_title("Delivered-cost breakdown for the least-cost site in each scenario")
    ax.set_xlabel("")
    ax.set_ylabel("EUR/kg H2")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_5_cost_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_validation(validation: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(validation))
    ax.bar(x - width / 2, validation["reference_cost_eur_per_kg"], width=width, label="Reference literature", color="#7570b3")
    ax.bar(x + width / 2, validation["model_cost_eur_per_kg"], width=width, label="This model", color="#e7298a")
    ax.set_xticks(x)
    ax.set_xticklabels(validation["model_case"], rotation=12, ha="right")
    ax.set_ylabel("EUR/kg H2")
    ax.set_title("Validation against literature-scale export costs")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_6_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report_inputs(site_results: pd.DataFrame, eu_results: pd.DataFrame, summary: pd.DataFrame):
    top_sites = (
        site_results.sort_values(["scenario", "delivered_to_europe_eur_per_kg"])
        .groupby("scenario")
        .head(5)
        .copy()
    )
    top_sites.to_csv(OUTPUT_DIR / "top_5_sites_each_scenario.csv", index=False)

    competitiveness_records = []
    for scenario in summary["scenario"]:
        africa_subset = site_results[site_results["scenario"] == scenario]
        eu_subset = eu_results[eu_results["scenario"] == scenario]
        for _, eu_row in eu_subset.iterrows():
            competitiveness_records.append(
                {
                    "scenario": scenario,
                    "benchmark": eu_row["benchmark"],
                    "europe_cost": eu_row["produced_in_europe_eur_per_kg"],
                    "africa_best": africa_subset["delivered_to_europe_eur_per_kg"].min(),
                    "africa_p10": africa_subset["delivered_to_europe_eur_per_kg"].quantile(0.10),
                    "africa_median": africa_subset["delivered_to_europe_eur_per_kg"].median(),
                    "share_competitive": (
                        africa_subset["delivered_to_europe_eur_per_kg"] <= eu_row["produced_in_europe_eur_per_kg"]
                    ).mean(),
                }
            )
    pd.DataFrame(competitiveness_records).to_csv(OUTPUT_DIR / "competitiveness_table.csv", index=False)


def main():
    ensure_dirs()
    assumptions, sites, world = load_inputs()
    sites = map_resource_scores_to_cf(sites, assumptions)
    site_results = model_african_sites(sites, assumptions)
    eu_results = model_europe_benchmarks(assumptions)
    summary = summarize_results(site_results, eu_results)
    validation = validation_table(site_results, assumptions)
    save_outputs(site_results, eu_results, summary, validation)
    write_report_inputs(site_results, eu_results, summary)
    plot_data_overview(sites, world)
    plot_baseline_map(site_results, world)
    plot_competitiveness(site_results, eu_results)
    plot_scenario_sensitivity(summary)
    plot_cost_breakdown(site_results)
    plot_validation(validation)
    print("Analysis complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
