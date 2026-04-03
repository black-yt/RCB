#!/usr/bin/env python3
"""Main analysis entry point for the mangrove composite climate-risk task.

This script is intentionally self-contained and avoids non-standard geospatial
packages such as geopandas/shapely because they are not guaranteed to be
available in the benchmark runtime. It works directly with:
- GeoPackage files via sqlite3 + lightweight GeoPackage/WKB parsing
- NetCDF4/HDF5 files via h5py
- Numerical analysis via numpy/pandas/scipy
- Figures via matplotlib/seaborn

Outputs written by this script
------------------------------
outputs/data_inventory_summary.json
outputs/slr_scenario_summary.csv
outputs/tc_baseline_summary.json
outputs/mangrove_point_risk_sample.csv
outputs/country_service_risk.csv
outputs/top_country_rankings.csv
outputs/method_notes.txt
report/images/data_overview.png
report/images/slr_scenarios.png
report/images/composite_risk_map.png
report/images/country_risk_services.png

Methodological note
-------------------
The benchmark workspace provides historical tropical cyclone tracks but not an
explicit future TC projection file. To still produce a task-relevant composite
index, the script constructs a transparent "regime-shift proxy" by combining:
1. Historical baseline cyclone exposure from the provided track set.
2. Basin-specific scenario multipliers informed by the supplied related work,
   especially the evidence that future cyclone-risk changes are modest globally
   but divergent regionally.
3. Relative sea-level rise rates from IPCC AR6 gridded coastal locations.

This yields a reproducible scenario-based comparative risk index rather than a
full dynamical forecast.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "report" / "images"

MANGROVE_GPKG = DATA / "mangroves" / "gmw_v4_ref_smpls_qad_v12.gpkg"
COUNTRY_GPKG = DATA / "ecosystem" / "UCSC_CWON_countrybounds.gpkg"
TC_NETCDF = DATA / "tc" / "tracks_mit_mpi-esm1-2-hr_historical_reduced.nc"
SLR_FILES = {
    "ssp245": DATA / "slr" / "total_ssp245_medium_confidence_rates.nc",
    "ssp370": DATA / "slr" / "total_ssp370_medium_confidence_rates.nc",
    "ssp585": DATA / "slr" / "total_ssp585_medium_confidence_rates.nc",
}


# Scenario-specific cyclone-regime multipliers by broad basin.
# These are transparent proxies grounded in the related-work narrative:
# modest global changes, increases for the Americas, decreases in Oceania,
# and intermediate responses elsewhere.
TC_BASIN_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "ssp245": {
        "atlantic_east_pacific": 1.10,
        "west_pacific": 0.98,
        "indian": 1.03,
        "other": 1.00,
    },
    "ssp370": {
        "atlantic_east_pacific": 1.18,
        "west_pacific": 0.95,
        "indian": 1.07,
        "other": 1.02,
    },
    "ssp585": {
        "atlantic_east_pacific": 1.25,
        "west_pacific": 0.92,
        "indian": 1.12,
        "other": 1.03,
    },
}

# Category-1 threshold in m/s, used for normalizing wind intensity.
TC_WIND_THRESHOLD = 33.0
# Gaussian kernel scale in degrees for coastal-track exposure.
TC_KERNEL_SIGMA_DEG = 1.5
# Query radius for track influence around mangrove points / country centroids.
TC_QUERY_RADIUS_DEG = 3.0
# Approximate historical period covered by the provided TC file.
TC_YEARS = 2014 - 1850 + 1
# Mangrove sample is documented as 10% of the original sampling frame.
MANGROVE_SAMPLE_EXPANSION = 10.0


@dataclass
class SimpleGeometry:
    geom_type: str
    coords: object


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)


def gpkg_geom_to_wkb(blob: bytes) -> bytes:
    """Extract standard WKB payload from a GeoPackage geometry blob."""
    if blob[:2] != b"GP":
        raise ValueError("Not a GeoPackage geometry blob")
    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0b111
    envelope_sizes = {
        0: 0,
        1: 32,
        2: 48,
        3: 48,
        4: 64,
    }
    envelope_bytes = envelope_sizes.get(envelope_indicator)
    if envelope_bytes is None:
        raise ValueError(f"Unsupported GeoPackage envelope indicator: {envelope_indicator}")
    return blob[8 + envelope_bytes :]


class WKBReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))


# WKB geometry type codes
POINT = 1
LINESTRING = 2
POLYGON = 3
MULTIPOINT = 4
MULTILINESTRING = 5
MULTIPOLYGON = 6
GEOMETRYCOLLECTION = 7


def parse_wkb(data: bytes) -> SimpleGeometry:
    rdr = WKBReader(data)
    return _parse_wkb_with_reader(rdr)


def _parse_wkb_with_reader(rdr: WKBReader) -> SimpleGeometry:
    endian_flag = rdr.unpack("B")[0]
    endian = "<" if endian_flag == 1 else ">"
    geom_type = rdr.unpack(endian + "I")[0]

    if geom_type == POINT:
        x, y = rdr.unpack(endian + "dd")
        return SimpleGeometry("Point", (x, y))

    if geom_type == LINESTRING:
        n = rdr.unpack(endian + "I")[0]
        coords = [rdr.unpack(endian + "dd") for _ in range(n)]
        return SimpleGeometry("LineString", coords)

    if geom_type == POLYGON:
        n_rings = rdr.unpack(endian + "I")[0]
        rings = []
        for _ in range(n_rings):
            n = rdr.unpack(endian + "I")[0]
            rings.append([rdr.unpack(endian + "dd") for _ in range(n)])
        return SimpleGeometry("Polygon", rings)

    if geom_type == MULTIPOINT:
        n = rdr.unpack(endian + "I")[0]
        geoms = [_parse_wkb_with_reader(rdr).coords for _ in range(n)]
        return SimpleGeometry("MultiPoint", geoms)

    if geom_type == MULTILINESTRING:
        n = rdr.unpack(endian + "I")[0]
        geoms = [_parse_wkb_with_reader(rdr).coords for _ in range(n)]
        return SimpleGeometry("MultiLineString", geoms)

    if geom_type == MULTIPOLYGON:
        n = rdr.unpack(endian + "I")[0]
        geoms = [_parse_wkb_with_reader(rdr).coords for _ in range(n)]
        return SimpleGeometry("MultiPolygon", geoms)

    if geom_type == GEOMETRYCOLLECTION:
        n = rdr.unpack(endian + "I")[0]
        geoms = [_parse_wkb_with_reader(rdr) for _ in range(n)]
        return SimpleGeometry("GeometryCollection", geoms)

    raise ValueError(f"Unsupported WKB geometry type: {geom_type}")


def polygon_all_points(geom: SimpleGeometry) -> List[Tuple[float, float]]:
    if geom.geom_type == "Polygon":
        return [pt for ring in geom.coords for pt in ring]
    if geom.geom_type == "MultiPolygon":
        return [pt for poly in geom.coords for ring in poly for pt in ring]
    raise ValueError(f"Expected polygonal geometry, got {geom.geom_type}")


def approx_polygon_centroid(geom: SimpleGeometry) -> Tuple[float, float]:
    pts = polygon_all_points(geom)
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    lon = float((xs.min() + xs.max()) / 2.0)
    lat = float((ys.min() + ys.max()) / 2.0)
    return lon, lat


def basin_from_lon_lat(lon: float, lat: float) -> str:
    """Very coarse basin classification for tropical cyclone regimes."""
    if lat > 35 or lat < -35:
        return "other"
    if -120 <= lon <= -15:
        return "atlantic_east_pacific"
    if 100 <= lon <= 180 or -180 <= lon < -120:
        return "west_pacific"
    if 20 <= lon < 100:
        return "indian"
    return "other"


def percentile_rank(series: pd.Series) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    out = np.full(arr.shape, np.nan)
    if valid.any():
        vals = arr[valid]
        order = np.argsort(vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0, 1, len(vals), endpoint=True)
        out[np.where(valid)[0]] = ranks
    return pd.Series(out, index=series.index)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    if valid.any():
        v = values[valid]
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi - lo < 1e-12:
            out[valid] = 0.5
        else:
            out[valid] = (v - lo) / (hi - lo)
    return out


def read_mangrove_points(limit: int | None = None) -> pd.DataFrame:
    con = sqlite3.connect(MANGROVE_GPKG)
    query = "SELECT fid, uid, ref_cls, gmw_v3_xtras, coastTrain, gmw_v4_qa, geom FROM gmw_v4_ref_smpls_qad_v12_sample"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, con)
    con.close()

    xs, ys = [], []
    for blob in df["geom"]:
        geom = parse_wkb(gpkg_geom_to_wkb(blob))
        x, y = geom.coords
        xs.append(x)
        ys.append(y)
    df["lon"] = xs
    df["lat"] = ys
    df = df.drop(columns=["geom"])
    return df


def read_country_service_table() -> pd.DataFrame:
    con = sqlite3.connect(COUNTRY_GPKG)
    df = pd.read_sql_query("SELECT * FROM CWON_RESULTS_COUNTRY_ALLYEARS", con)
    geom_df = pd.read_sql_query("SELECT fid, geom FROM CWON_RESULTS_COUNTRY_ALLYEARS", con)
    con.close()

    centroids = []
    for blob in geom_df["geom"]:
        geom = parse_wkb(gpkg_geom_to_wkb(blob))
        lon, lat = approx_polygon_centroid(geom)
        centroids.append((lon, lat))
    df["centroid_lon"] = [p[0] for p in centroids]
    df["centroid_lat"] = [p[1] for p in centroids]
    df["tc_basin"] = [basin_from_lon_lat(x, y) for x, y in centroids]
    return df


def load_slr_scenario(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        lats = f["lat"][:].astype(float)
        lons = f["lon"][:].astype(float)
        years = f["years"][:].astype(int)
        quantiles = f["quantiles"][:].astype(float)
        q_idx = int(np.argmin(np.abs(quantiles - 0.5)))
        y_idx_2020 = int(np.argmin(np.abs(years - 2020)))
        y_idx_2100 = int(np.argmin(np.abs(years - 2100)))
        scale_factor = float(np.array(f["sea_level_change_rate"].attrs.get("scale_factor", [1.0]))[0])
        slr = f["sea_level_change_rate"]
        rates_2100 = slr[q_idx, y_idx_2100, :].astype(float) * scale_factor
        rates_2020 = slr[q_idx, y_idx_2020, :].astype(float) * scale_factor
        mean_2020_2100 = slr[q_idx, y_idx_2020 : y_idx_2100 + 1, :].astype(float).mean(axis=0) * scale_factor
    return {
        "lat": lats,
        "lon": lons,
        "years": years,
        "quantiles": quantiles,
        "median_idx": np.array([q_idx]),
        "rate_2020": rates_2020,
        "rate_2100": rates_2100,
        "rate_mean_2020_2100": mean_2020_2100,
    }


def build_slr_tree(slr_data: Dict[str, np.ndarray]) -> cKDTree:
    coords = np.column_stack([slr_data["lon"], slr_data["lat"]])
    return cKDTree(coords)


def assign_slr_to_points(points_df: pd.DataFrame, slr_data: Dict[str, np.ndarray], tree: cKDTree, prefix: str) -> pd.DataFrame:
    distances, idx = tree.query(points_df[["lon", "lat"]].to_numpy(), k=1)
    out = points_df.copy()
    out[f"{prefix}_slr_mean_mm_yr"] = slr_data["rate_mean_2020_2100"][idx]
    out[f"{prefix}_slr_2100_mm_yr"] = slr_data["rate_2100"][idx]
    out[f"{prefix}_slr_distance_deg"] = distances
    return out


def load_tc_points() -> pd.DataFrame:
    with h5py.File(TC_NETCDF, "r") as f:
        lat = f["lat"][:].astype(float)
        lon = f["lon"][:].astype(float)
        wind = f["wind"][:].astype(float)
    df = pd.DataFrame({"lon": lon, "lat": lat, "wind_ms": wind})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["wind_ms"] > 0) & (df["lat"].between(-60, 60))]
    return df.reset_index(drop=True)


def tc_exposure_kernel(target_xy: np.ndarray, tc_xy: np.ndarray, tc_weight: np.ndarray, radius: float, sigma: float) -> np.ndarray:
    tree = cKDTree(tc_xy)
    exposures = np.zeros(target_xy.shape[0], dtype=float)
    neighborhoods = tree.query_ball_point(target_xy, r=radius)
    for i, nbrs in enumerate(neighborhoods):
        if not nbrs:
            continue
        diffs = tc_xy[nbrs] - target_xy[i]
        d2 = np.sum(diffs * diffs, axis=1)
        kern = np.exp(-0.5 * d2 / (sigma ** 2))
        exposures[i] = float(np.sum(tc_weight[nbrs] * kern))
    return exposures


def prepare_tc_baseline(mangroves: pd.DataFrame, countries: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    tc = load_tc_points()
    tc_xy = tc[["lon", "lat"]].to_numpy()
    tc_weight = np.clip(tc["wind_ms"].to_numpy() / TC_WIND_THRESHOLD, 0, None) ** 2 / TC_YEARS

    mangrove_xy = mangroves[["lon", "lat"]].to_numpy()
    country_xy = countries[["centroid_lon", "centroid_lat"]].to_numpy()

    mangroves = mangroves.copy()
    countries = countries.copy()
    mangroves["tc_baseline_exposure"] = tc_exposure_kernel(
        mangrove_xy, tc_xy, tc_weight, radius=TC_QUERY_RADIUS_DEG, sigma=TC_KERNEL_SIGMA_DEG
    )
    countries["tc_baseline_exposure"] = tc_exposure_kernel(
        country_xy, tc_xy, tc_weight, radius=TC_QUERY_RADIUS_DEG, sigma=TC_KERNEL_SIGMA_DEG
    )

    summary = {
        "n_track_points": int(len(tc)),
        "historical_years_assumed": int(TC_YEARS),
        "mean_wind_ms": float(tc["wind_ms"].mean()),
        "p90_wind_ms": float(tc["wind_ms"].quantile(0.9)),
        "max_wind_ms": float(tc["wind_ms"].max()),
        "mean_mangrove_baseline_exposure": float(mangroves["tc_baseline_exposure"].mean()),
        "max_mangrove_baseline_exposure": float(mangroves["tc_baseline_exposure"].max()),
    }
    return mangroves, countries, summary


def add_scenario_risk_columns(mangroves: pd.DataFrame, countries: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mangroves = mangroves.copy()
    countries = countries.copy()

    for scenario in SLR_FILES.keys():
        mangroves[f"{scenario}_tc_regime_multiplier"] = mangroves.apply(
            lambda row: TC_BASIN_MULTIPLIERS[scenario][basin_from_lon_lat(row["lon"], row["lat"])], axis=1
        )
        countries[f"{scenario}_tc_regime_multiplier"] = countries["tc_basin"].map(TC_BASIN_MULTIPLIERS[scenario])

        mangroves[f"{scenario}_tc_shifted_exposure"] = (
            mangroves["tc_baseline_exposure"] * mangroves[f"{scenario}_tc_regime_multiplier"]
        )
        countries[f"{scenario}_tc_shifted_exposure"] = (
            countries["tc_baseline_exposure"] * countries[f"{scenario}_tc_regime_multiplier"]
        )

        mangroves[f"{scenario}_tc_norm"] = minmax_scale(mangroves[f"{scenario}_tc_shifted_exposure"].to_numpy())
        mangroves[f"{scenario}_slr_norm"] = minmax_scale(mangroves[f"{scenario}_slr_mean_mm_yr"].to_numpy())
        mangroves[f"{scenario}_risk_index"] = 0.5 * mangroves[f"{scenario}_tc_norm"] + 0.5 * mangroves[f"{scenario}_slr_norm"]

        countries[f"{scenario}_tc_norm"] = minmax_scale(countries[f"{scenario}_tc_shifted_exposure"].to_numpy())
        countries[f"{scenario}_slr_norm"] = minmax_scale(countries[f"{scenario}_slr_mean_mm_yr"].to_numpy())
        countries[f"{scenario}_risk_index"] = 0.5 * countries[f"{scenario}_tc_norm"] + 0.5 * countries[f"{scenario}_slr_norm"]

    return mangroves, countries


def summarize_slr(mangroves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SLR_FILES.keys():
        s = mangroves[f"{scenario}_slr_mean_mm_yr"]
        rows.append(
            {
                "scenario": scenario,
                "mean_mangrove_slr_mm_yr": float(s.mean()),
                "median_mangrove_slr_mm_yr": float(s.median()),
                "p90_mangrove_slr_mm_yr": float(s.quantile(0.9)),
                "max_mangrove_slr_mm_yr": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def enrich_country_service_metrics(countries: pd.DataFrame) -> pd.DataFrame:
    countries = countries.copy()

    countries["mangrove_area_ha_2020"] = pd.to_numeric(countries["Mang_Ha_2020"], errors="coerce")
    countries["service_population_benefit_2020"] = pd.to_numeric(countries["Ben_Pop_2020"], errors="coerce")
    countries["service_property_benefit_2020"] = pd.to_numeric(countries["Ben_Stock_2020"], errors="coerce")
    countries["population_at_risk_2020"] = pd.to_numeric(countries["Risk_Pop_2020"], errors="coerce")
    countries["property_at_risk_2020"] = pd.to_numeric(countries["Risk_Stock_2020"], errors="coerce")

    for scenario in SLR_FILES.keys():
        countries[f"{scenario}_mangrove_area_risk_ha_equiv"] = (
            countries["mangrove_area_ha_2020"] * countries[f"{scenario}_risk_index"]
        )
        countries[f"{scenario}_service_population_exposure"] = (
            countries["service_population_benefit_2020"] * countries[f"{scenario}_risk_index"]
        )
        countries[f"{scenario}_service_property_exposure"] = (
            countries["service_property_benefit_2020"] * countries[f"{scenario}_risk_index"]
        )
        countries[f"{scenario}_priority_score"] = (
            0.4 * countries[f"{scenario}_risk_index"].fillna(0)
            + 0.2 * minmax_scale(countries["mangrove_area_ha_2020"].to_numpy())
            + 0.2 * minmax_scale(countries["service_population_benefit_2020"].to_numpy())
            + 0.2 * minmax_scale(countries["service_property_benefit_2020"].to_numpy())
        )

    return countries


def make_data_overview_figure(mangroves: pd.DataFrame, tc_summary: Dict[str, float], country_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(mangroves["lon"], mangroves["lat"], s=3, alpha=0.25, color="#1b9e77")
    axes[0, 0].set_title("Sampled mangrove reference points")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")

    counts = mangroves["ref_cls"].value_counts().sort_index()
    axes[0, 1].bar(counts.index.astype(str), counts.values, color="#7570b3")
    axes[0, 1].set_title("Mangrove sample class counts")
    axes[0, 1].set_xlabel("Reference class")
    axes[0, 1].set_ylabel("Count")

    sns.histplot(mangroves["tc_baseline_exposure"], bins=40, ax=axes[1, 0], color="#d95f02")
    axes[1, 0].set_title("Baseline TC exposure across mangrove points")
    axes[1, 0].set_xlabel("Exposure index")

    country_area = country_df[["Country", "Mang_Ha_2020"]].copy()
    country_area["Mang_Ha_2020"] = pd.to_numeric(country_area["Mang_Ha_2020"], errors="coerce")
    country_area = country_area.nlargest(12, "Mang_Ha_2020").sort_values("Mang_Ha_2020")
    axes[1, 1].barh(country_area["Country"], country_area["Mang_Ha_2020"], color="#66a61e")
    axes[1, 1].set_title("Largest national mangrove areas in 2020")
    axes[1, 1].set_xlabel("Mangrove area (ha)")

    fig.suptitle(
        f"Input overview | TC points={tc_summary['n_track_points']:,} | historical years≈{tc_summary['historical_years_assumed']}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(IMAGES / "data_overview.png", dpi=200)
    plt.close(fig)


def make_slr_figure(slr_summary: pd.DataFrame, mangroves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.barplot(data=slr_summary, x="scenario", y="mean_mangrove_slr_mm_yr", ax=axes[0], color="#1f78b4")
    axes[0].set_title("Mean end-century coastal SLR rate near mangroves")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Mean SLR rate (mm/yr)")

    box_df = pd.concat(
        [
            pd.DataFrame({"scenario": s, "slr_mm_yr": mangroves[f"{s}_slr_mean_mm_yr"]})
            for s in SLR_FILES.keys()
        ],
        ignore_index=True,
    )
    sns.boxplot(data=box_df, x="scenario", y="slr_mm_yr", ax=axes[1], color="#a6cee3")
    axes[1].set_title("Distribution of local SLR rates across mangrove points")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("SLR rate (mm/yr)")

    fig.tight_layout()
    fig.savefig(IMAGES / "slr_scenarios.png", dpi=200)
    plt.close(fig)


def make_composite_risk_map(mangroves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    scenarios = list(SLR_FILES.keys())
    for ax, scenario in zip(axes, scenarios):
        sc = ax.scatter(
            mangroves["lon"],
            mangroves["lat"],
            c=mangroves[f"{scenario}_risk_index"],
            s=4,
            cmap="magma_r",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Composite mangrove risk: {scenario}")
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Composite risk index")
    fig.tight_layout()
    fig.savefig(IMAGES / "composite_risk_map.png", dpi=220)
    plt.close(fig)


def make_country_risk_services_figure(countries: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_df = countries[["Country", "ssp585_priority_score"]].nlargest(12, "ssp585_priority_score").sort_values("ssp585_priority_score")
    axes[0].barh(plot_df["Country"], plot_df["ssp585_priority_score"], color="#e7298a")
    axes[0].set_title("Top national adaptation priorities (SSP5-8.5)")
    axes[0].set_xlabel("Priority score")

    axes[1].scatter(
        countries["ssp585_risk_index"],
        countries["service_population_benefit_2020"],
        s=np.clip(countries["mangrove_area_ha_2020"] / 1500.0, 10, 400),
        alpha=0.7,
        color="#7570b3",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_title("Risk vs people benefiting from mangroves (2020)")
    axes[1].set_xlabel("Composite risk index (SSP5-8.5)")
    axes[1].set_ylabel("Population benefit metric")

    for _, row in countries.nlargest(8, "ssp585_priority_score").iterrows():
        axes[1].annotate(row["ISO3"], (row["ssp585_risk_index"], row["service_population_benefit_2020"]), fontsize=8)

    fig.tight_layout()
    fig.savefig(IMAGES / "country_risk_services.png", dpi=220)
    plt.close(fig)


def write_method_notes() -> None:
    text = """Composite mangrove climate-risk workflow used in this workspace
=========================================================

1. Mangrove sample:
   - Read the provided 10% Global Mangrove Watch sample directly from the GeoPackage.
   - Decode GeoPackage point geometries into longitude/latitude.

2. Relative sea-level rise (SLR):
   - Read AR6 regional relative SLR products for SSP2-4.5, SSP3-7.0, and SSP5-8.5.
   - Use the median quantile (0.5) and summarize 2020-2100 mean rates.
   - Assign the nearest AR6 coastal SLR location to each mangrove sample point and country centroid.

3. Tropical cyclone (TC) baseline hazard:
   - Read the provided historical TC track-point dataset.
   - Convert each track point into a local exposure contribution using a wind-squared Gaussian kernel.
   - Aggregate nearby track contributions around mangrove points and country centroids.

4. TC regime-shift proxy:
   - Because no future TC projection file is provided in the workspace, apply a scenario-specific,
     basin-level multiplier informed by the supplied related work: stronger increases around the
     Atlantic / East Pacific / Americas, modest increases in the Indian basin, and relative decreases
     around the western Pacific / Oceania.

5. Composite index:
   - Normalize shifted TC exposure and SLR exposure to [0,1].
   - Combine them with equal weights: Risk = 0.5 * TC_norm + 0.5 * SLR_norm.

6. Ecosystem-service linkage:
   - Use the provided country dataset to combine risk with mangrove area, population benefit,
     property benefit, and risk metrics.
   - Create a scenario-specific national priority score that blends hazard and service magnitude.

Important limitation:
---------------------
This is a transparent comparative-risk implementation constrained to the supplied inputs. It should be
interpreted as a task-specific screening framework rather than a full process-based forecast.
"""
    (OUTPUTS / "method_notes.txt").write_text(text)


def save_inventory(mangroves: pd.DataFrame, countries: pd.DataFrame) -> None:
    inventory = {
        "workspace": str(ROOT),
        "input_files": {
            "mangroves": str(MANGROVE_GPKG.relative_to(ROOT)),
            "countries": str(COUNTRY_GPKG.relative_to(ROOT)),
            "tc": str(TC_NETCDF.relative_to(ROOT)),
            "slr": {k: str(v.relative_to(ROOT)) for k, v in SLR_FILES.items()},
        },
        "mangrove_sample_points": int(len(mangroves)),
        "implied_full_sample_points": int(len(mangroves) * MANGROVE_SAMPLE_EXPANSION),
        "country_rows": int(len(countries)),
        "scenarios": list(SLR_FILES.keys()),
    }
    with open(OUTPUTS / "data_inventory_summary.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    print("Loading mangrove points...")
    mangroves = read_mangrove_points()
    print(f"Loaded {len(mangroves):,} mangrove sample points")

    print("Loading country ecosystem-service table...")
    countries = read_country_service_table()
    print(f"Loaded {len(countries):,} country rows")

    save_inventory(mangroves, countries)

    print("Loading SLR scenarios and assigning nearest coastal rates...")
    slr_cache = {}
    for scenario, path in SLR_FILES.items():
        slr_cache[scenario] = load_slr_scenario(path)
        slr_tree = build_slr_tree(slr_cache[scenario])
        mangroves = assign_slr_to_points(mangroves, slr_cache[scenario], slr_tree, scenario)
        countries = assign_slr_to_points(
            countries.rename(columns={"centroid_lon": "lon", "centroid_lat": "lat"}),
            slr_cache[scenario],
            slr_tree,
            scenario,
        ).rename(columns={"lon": "centroid_lon", "lat": "centroid_lat"})

    print("Computing baseline tropical cyclone exposure...")
    mangroves, countries, tc_summary = prepare_tc_baseline(mangroves, countries)

    print("Combining SLR and TC regime-shift proxies into composite risk...")
    mangroves, countries = add_scenario_risk_columns(mangroves, countries)
    countries = enrich_country_service_metrics(countries)

    print("Writing tabular outputs...")
    slr_summary = summarize_slr(mangroves)
    slr_summary.to_csv(OUTPUTS / "slr_scenario_summary.csv", index=False)

    with open(OUTPUTS / "tc_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(tc_summary, f, indent=2)

    keep_cols_points = [
        "fid",
        "uid",
        "ref_cls",
        "lon",
        "lat",
        "tc_baseline_exposure",
    ]
    for s in SLR_FILES.keys():
        keep_cols_points += [
            f"{s}_slr_mean_mm_yr",
            f"{s}_tc_regime_multiplier",
            f"{s}_tc_shifted_exposure",
            f"{s}_risk_index",
        ]
    mangroves[keep_cols_points].to_csv(OUTPUTS / "mangrove_point_risk_sample.csv", index=False)

    keep_cols_country = [
        "Country",
        "ISO3",
        "centroid_lon",
        "centroid_lat",
        "tc_basin",
        "Mang_Ha_2020",
        "Risk_Pop_2020",
        "Risk_Stock_2020",
        "Ben_Pop_2020",
        "Ben_Stock_2020",
        "mangrove_area_ha_2020",
        "service_population_benefit_2020",
        "service_property_benefit_2020",
        "tc_baseline_exposure",
    ]
    for s in SLR_FILES.keys():
        keep_cols_country += [
            f"{s}_slr_mean_mm_yr",
            f"{s}_tc_regime_multiplier",
            f"{s}_tc_shifted_exposure",
            f"{s}_risk_index",
            f"{s}_mangrove_area_risk_ha_equiv",
            f"{s}_service_population_exposure",
            f"{s}_service_property_exposure",
            f"{s}_priority_score",
        ]
    countries[keep_cols_country].to_csv(OUTPUTS / "country_service_risk.csv", index=False)

    ranking_rows = []
    for s in SLR_FILES.keys():
        top = countries.nlargest(15, f"{s}_priority_score")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            ranking_rows.append(
                {
                    "scenario": s,
                    "rank": rank,
                    "Country": row["Country"],
                    "ISO3": row["ISO3"],
                    "priority_score": row[f"{s}_priority_score"],
                    "risk_index": row[f"{s}_risk_index"],
                    "mangrove_area_ha_2020": row["mangrove_area_ha_2020"],
                    "population_benefit_2020": row["service_population_benefit_2020"],
                    "property_benefit_2020": row["service_property_benefit_2020"],
                }
            )
    pd.DataFrame(ranking_rows).to_csv(OUTPUTS / "top_country_rankings.csv", index=False)

    write_method_notes()

    print("Creating figures...")
    make_data_overview_figure(mangroves, tc_summary, countries)
    make_slr_figure(slr_summary, mangroves)
    make_composite_risk_map(mangroves)
    make_country_risk_services_figure(countries)

    print("Done. Outputs written to outputs/ and report/images/")


if __name__ == "__main__":
    main()
