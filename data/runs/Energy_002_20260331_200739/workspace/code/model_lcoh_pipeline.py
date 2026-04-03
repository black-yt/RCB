import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_IMG_DIR = BASE_DIR / "report" / "images"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)

HEX_CSV = DATA_DIR / "hex_final_NA_min.csv"
SHP_PATH = DATA_DIR / "africa_map" / "ne_10m_admin_0_countries.shp"

YEARS = 20
CAPACITY_FACTOR_PV = 0.22
CAPACITY_FACTOR_WIND = 0.40
ELECTROLYZER_CAPEX_2030 = 400  # USD/kW
ELECTROLYZER_OPEX_FRAC = 0.03
PV_CAPEX_2030 = 450  # USD/kW
WIND_CAPEX_2030 = 1100  # USD/kW
BALANCE_OF_PLANT_FRAC = 0.25

AMMONIA_PLANT_CAPEX = 700  # USD/kW_H2eq
AMMONIA_OPEX_FRAC = 0.04
SHIPPING_COST_BASE = 1.2  # USD/kg NH3, approx ~ USD/kg H2 eq
RECONVERSION_COST = 0.8  # USD/kg H2

ELECTRICITY_SHARE = 0.7
CAPEX_SHARE = 0.3

DISCOUNT_SCENARIOS = {
    "high_risk": 0.10,
    "baseline": 0.07,
    "derisked": 0.04,
}

EU_LCOH_REFERENCE = 3.0


def annuity_factor(r: float, n: int) -> float:
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def compute_lcoh_local(row, w_pv: float = 0.6, r: float = 0.07, max_pv: float = 1.0, max_wind: float = 1.0) -> float:
    cf_pv = CAPACITY_FACTOR_PV * (row["theo_pv"] / max_pv)
    cf_wind = CAPACITY_FACTOR_WIND * (row["theo_wind"] / max_wind)
    cf_hybrid = w_pv * cf_pv + (1 - w_pv) * cf_wind
    cf_hybrid = np.clip(cf_hybrid, 0.1, 0.7)

    distance_km = row[["grid_dist_km", "road_dist_km", "ocean_dist_km", "waterbody_dist_km"]].fillna(0).mean()
    infra_markup = 1 + 0.002 * distance_km

    pv_capex = PV_CAPEX_2030 * infra_markup
    wind_capex = WIND_CAPEX_2030 * infra_markup
    el_capex = ELECTROLYZER_CAPEX_2030 * infra_markup * (1 + BALANCE_OF_PLANT_FRAC)

    capex_system = w_pv * pv_capex + (1 - w_pv) * wind_capex + el_capex

    crf = annuity_factor(r, YEARS)

    annualized_capex = capex_system * crf
    annual_opex = ELECTROLYZER_OPEX_FRAC * el_capex + 0.02 * (w_pv * pv_capex + (1 - w_pv) * wind_capex)

    full_load_hours = cf_hybrid * 8760
    h2_output_per_kw = full_load_hours * 0.02

    lcoh = (annualized_capex + annual_opex) / (h2_output_per_kw + 1e-9)

    electricity_cost = ELECTRICITY_SHARE * lcoh
    capex_cost = CAPEX_SHARE * lcoh

    lcoh = electricity_cost + capex_cost

    return lcoh


def add_lcoh_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_pv = df["theo_pv"].max()
    max_wind = df["theo_wind"].max()
    for scen, r in DISCOUNT_SCENARIOS.items():
        df[f"lcoh_{scen}"] = df.apply(lambda row: compute_lcoh_local(row, r=r, max_pv=max_pv, max_wind=max_wind), axis=1)
    return df


def add_delivered_cost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for scen in DISCOUNT_SCENARIOS.keys():
        lcoh = df[f"lcoh_{scen}"]
        crf_eu = annuity_factor(0.05, YEARS)
        ammonia_chain_capex = AMMONIA_PLANT_CAPEX * crf_eu
        ammonia_chain_opex = AMMONIA_OPEX_FRAC * AMMONIA_PLANT_CAPEX
        ammonia_chain_cost = (ammonia_chain_capex + ammonia_chain_opex) / (8760 * 0.9 * 0.02)

        delivered = lcoh + ammonia_chain_cost + SHIPPING_COST_BASE + RECONVERSION_COST
        df[f"lcoh_delivered_{scen}"] = delivered
        df[f"premium_vs_eu_{scen}"] = delivered - EU_LCOH_REFERENCE
    return df


def join_with_countries(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf_c = gpd.read_file(SHP_PATH)
    africa = gdf_c[gdf_c["CONTINENT"] == "Africa"].copy()
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, africa[["ADMIN", "ISO_A3", "geometry"]], how="left", predicate="within")
    return joined


def plot_data_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.histplot(df["theo_pv"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("PV potential distribution")

    sns.histplot(df["theo_wind"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Wind potential distribution")

    sns.scatterplot(x="theo_pv", y="theo_wind", data=df, ax=axes[1, 0])
    axes[1, 0].set_title("PV vs wind potential")

    sns.histplot(df[["grid_dist_km", "road_dist_km", "ocean_dist_km", "waterbody_dist_km"]].melt()["value"],
                 kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Distance to infrastructure (all types)")

    plt.tight_layout()
    out_path = REPORT_IMG_DIR / "data_overview.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_lcoh_maps(gdf: gpd.GeoDataFrame):
    africa = gpd.read_file(SHP_PATH)
    africa = africa[africa["CONTINENT"] == "Africa"]

    for scen in DISCOUNT_SCENARIOS.keys():
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        africa.boundary.plot(ax=ax, color="black", linewidth=0.5)
        gdf.plot(column=f"lcoh_delivered_{scen}", ax=ax, cmap="viridis", legend=True,
                 legend_kwds={"label": "Delivered LCOH [USD/kg H2]"}, markersize=40)
        ax.set_title(f"Delivered green hydrogen cost to Europe ({scen})")
        ax.set_axis_off()
        out_path = REPORT_IMG_DIR / f"map_lcoh_delivered_{scen}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_scenario_comparison(df: pd.DataFrame):
    cols = [f"lcoh_delivered_{k}" for k in DISCOUNT_SCENARIOS.keys()]
    melted = df.melt(id_vars=["hex_id"], value_vars=cols, var_name="scenario", value_name="lcoh_delivered")
    melted["scenario"] = melted["scenario"].str.replace("lcoh_delivered_", "")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=melted, x="scenario", y="lcoh_delivered", ax=ax)
    ax.axhline(EU_LCOH_REFERENCE, color="red", linestyle="--", label="EU reference LCOH")
    ax.set_ylabel("Delivered cost [USD/kg H2]")
    ax.set_title("Impact of financing risk on delivered green H2 cost")
    ax.legend()
    out_path = REPORT_IMG_DIR / "scenario_comparison_boxplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(HEX_CSV)
    plot_data_overview(df)

    df_lcoh = add_lcoh_columns(df)
    df_deliv = add_delivered_cost(df_lcoh)

    df_deliv.to_csv(OUTPUT_DIR / "hex_lcoh_results.csv", index=False)

    gdf = join_with_countries(df_deliv)
    gdf.to_file(OUTPUT_DIR / "hex_lcoh_results.geojson", driver="GeoJSON")

    plot_lcoh_maps(gdf)
    plot_scenario_comparison(df_deliv)


if __name__ == "__main__":
    main()
