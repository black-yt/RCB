from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

INPUT_FILE = DATA_DIR / "20231012-06_input_netcdf.nc"
FORECAST_FILE = DATA_DIR / "006.nc"

SELECTED_CHANNELS = ["Z500", "T2M", "U850", "TP"]
SURFACE_CHANNELS = {"T2M", "U10", "V10", "MSL", "TP"}
FAMILY_NAMES = {
    "Z": "Geopotential",
    "T": "Temperature",
    "U": "U-wind",
    "V": "V-wind",
    "R": "Relative humidity",
    "T2M": "2 m temperature",
    "U10": "10 m U-wind",
    "V10": "10 m V-wind",
    "MSL": "Mean sea level pressure",
    "TP": "Total precipitation",
}


@dataclass
class LoadedData:
    lat: np.ndarray
    lon: np.ndarray
    level_names: list[str]
    input_data: np.ndarray
    forecast_data: np.ndarray
    input_times: list[str]
    forecast_init_times: list[str]
    forecast_steps: list[int]
    input_description: str
    forecast_description: str


def parse_channel(channel: str) -> tuple[str, str]:
    if channel in SURFACE_CHANNELS:
        return channel, "surface"
    family = "Z" if channel.startswith("Z") else channel[0]
    return family, channel[1:]


def decode_fixed_width_strings(values: np.ndarray) -> list[str]:
    names: list[str] = []
    for row in values:
        text = "".join(str(item) for item in row).strip()
        names.append(text)
    return names


def to_time_strings(raw_times: np.ndarray, units: str, calendar: str) -> list[str]:
    return [str(value) for value in num2date(raw_times, units, calendar)]


def load_data() -> LoadedData:
    with Dataset(INPUT_FILE) as src:
        lat = np.array(src.variables["lat"][:], dtype=np.float64)
        lon = np.array(src.variables["lon"][:], dtype=np.float64)
        level_names = decode_fixed_width_strings(src.variables["level"][:])
        input_data = np.array(src.variables["data"][:], dtype=np.float64)
        input_times = to_time_strings(
            src.variables["time"][:],
            src.variables["time"].units,
            getattr(src.variables["time"], "calendar", "standard"),
        )
        input_description = getattr(src.variables["data"], "description", "")

    with Dataset(FORECAST_FILE) as src:
        forecast_data = np.array(src.variables["data"][0], dtype=np.float64)
        forecast_init_times = to_time_strings(
            src.variables["time"][:],
            src.variables["time"].units,
            getattr(src.variables["time"], "calendar", "standard"),
        )
        forecast_steps = [int(x) for x in src.variables["step"][:]]
        forecast_description = getattr(src.variables["data"], "description", "")

    return LoadedData(
        lat=lat,
        lon=lon,
        level_names=level_names,
        input_data=input_data,
        forecast_data=forecast_data,
        input_times=input_times,
        forecast_init_times=forecast_init_times,
        forecast_steps=forecast_steps,
        input_description=input_description,
        forecast_description=forecast_description,
    )


def area_weights(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    return weights


def weighted_mean(field: np.ndarray, weights: np.ndarray) -> float:
    return float(np.nansum(field * weights) / np.nansum(weights))


def weighted_std(field: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(field, weights)
    variance = np.nansum(((field - mean) ** 2) * weights) / np.nansum(weights)
    return float(np.sqrt(variance))


def weighted_pattern_corr(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    a_anom = a - weighted_mean(a, weights)
    b_anom = b - weighted_mean(b, weights)
    denom = np.sqrt(
        np.nansum((a_anom ** 2) * weights) * np.nansum((b_anom ** 2) * weights)
    )
    if denom == 0:
        return float("nan")
    return float(np.nansum(a_anom * b_anom * weights) / denom)


def rms(field: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.nansum((field ** 2) * weights) / np.nansum(weights)))


def neighbor_corr(field: np.ndarray, axis: int) -> float:
    if axis == 0:
        left = field[:-1, :].reshape(-1)
        right = field[1:, :].reshape(-1)
    else:
        left = field[:, :-1].reshape(-1)
        right = field[:, 1:].reshape(-1)
    return float(np.corrcoef(left, right)[0, 1])


def zonal_power_spectrum(field: np.ndarray) -> np.ndarray:
    centered = field - field.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, axis=1)
    power = np.abs(fft) ** 2
    return power.mean(axis=0)


def spectral_slope(power: np.ndarray) -> float:
    k = np.arange(power.size)
    mask = (k >= 1) & np.isfinite(power) & (power > 0)
    x = np.log(k[mask])
    y = np.log(power[mask])
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def summarize_data(data: LoadedData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    weights = area_weights(data.lat, data.lon)
    rows: list[dict] = []
    spectra: list[dict] = []

    for idx, channel in enumerate(data.level_names):
        family, level = parse_channel(channel)
        x0 = data.input_data[0, idx]
        x1 = data.input_data[1, idx]
        y = data.forecast_data[0, idx]

        obs_increment = x1 - x0
        forecast_increment = y - x1
        spectrum = zonal_power_spectrum(x1)
        k = np.arange(spectrum.size)
        high_freq_fraction = float(spectrum[k >= 20].sum() / spectrum.sum())

        rows.append(
            {
                "channel": channel,
                "family": family,
                "family_name": FAMILY_NAMES[family],
                "level": level,
                "t0_mean": weighted_mean(x0, weights),
                "t1_mean": weighted_mean(x1, weights),
                "f6_mean": weighted_mean(y, weights),
                "t1_std": weighted_std(x1, weights),
                "obs_increment_rms": rms(obs_increment, weights),
                "forecast_increment_rms": rms(forecast_increment, weights),
                "normalized_obs_tendency": rms(obs_increment, weights)
                / weighted_std(x1, weights),
                "normalized_forecast_tendency": rms(forecast_increment, weights)
                / weighted_std(x1, weights),
                "temporal_corr_t0_t1": weighted_pattern_corr(x0, x1, weights),
                "temporal_corr_t1_f6": weighted_pattern_corr(x1, y, weights),
                "increment_corr": weighted_pattern_corr(
                    obs_increment, forecast_increment, weights
                ),
                "spatial_corr_lon_t1": neighbor_corr(x1, axis=1),
                "spatial_corr_lat_t1": neighbor_corr(x1, axis=0),
                "spectral_slope_t1": spectral_slope(spectrum),
                "high_freq_power_fraction": high_freq_fraction,
            }
        )

        for wavenumber, power in enumerate(spectrum):
            spectra.append(
                {
                    "channel": channel,
                    "family": family,
                    "wavenumber": wavenumber,
                    "power": float(power),
                }
            )

    channel_df = pd.DataFrame(rows)
    spectra_df = pd.DataFrame(spectra)

    family_df = (
        channel_df.groupby(["family", "family_name"], as_index=False)
        .agg(
            channels=("channel", "count"),
            mean_t1_std=("t1_std", "mean"),
            mean_temporal_corr_t0_t1=("temporal_corr_t0_t1", "mean"),
            mean_temporal_corr_t1_f6=("temporal_corr_t1_f6", "mean"),
            mean_spatial_corr_lon=("spatial_corr_lon_t1", "mean"),
            mean_spatial_corr_lat=("spatial_corr_lat_t1", "mean"),
            mean_spectral_slope=("spectral_slope_t1", "mean"),
            mean_high_freq_power_fraction=("high_freq_power_fraction", "mean"),
        )
        .sort_values("family")
    )

    metadata_summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "forecast_file": str(FORECAST_FILE.relative_to(ROOT)),
        "input_shape": list(data.input_data.shape),
        "forecast_shape": list(data.forecast_data.shape),
        "grid_shape": [int(data.lat.size), int(data.lon.size)],
        "input_times": data.input_times,
        "forecast_init_times": data.forecast_init_times,
        "forecast_steps_hours": data.forecast_steps,
        "input_description": data.input_description,
        "forecast_description": data.forecast_description,
        "resolution_from_grid_degrees": {
            "lat": float(np.abs(np.diff(data.lat)).mean()),
            "lon": float(np.abs(np.diff(data.lon)).mean()),
        },
        "headline_findings": {
            "channel_mean_abs_mean": float(channel_df["t1_mean"].abs().mean()),
            "channel_std_mean": float(channel_df["t1_std"].mean()),
            "channel_std_std": float(channel_df["t1_std"].std()),
            "mean_temporal_corr_t0_t1": float(channel_df["temporal_corr_t0_t1"].mean()),
            "mean_temporal_corr_t1_f6": float(channel_df["temporal_corr_t1_f6"].mean()),
            "mean_spatial_corr_lon_t1": float(channel_df["spatial_corr_lon_t1"].mean()),
            "mean_spatial_corr_lat_t1": float(channel_df["spatial_corr_lat_t1"].mean()),
            "mean_spectral_slope_t1": float(channel_df["spectral_slope_t1"].mean()),
        },
    }

    return channel_df, family_df, spectra_df, metadata_summary


def plot_maps(data: LoadedData) -> None:
    sns.set_theme(style="white")
    lon2d, lat2d = np.meshgrid(data.lon, data.lat)
    fig, axes = plt.subplots(
        2, len(SELECTED_CHANNELS), figsize=(16, 6.5), constrained_layout=True
    )

    for col, channel in enumerate(SELECTED_CHANNELS):
        idx = data.level_names.index(channel)
        fields = [data.input_data[1, idx], data.forecast_data[0, idx]]
        label_titles = [
            f"{channel} at {data.input_times[1]}",
            f"{channel} FuXi +{data.forecast_steps[0]}h",
        ]
        vmin = np.percentile(np.concatenate([fields[0].ravel(), fields[1].ravel()]), 2)
        vmax = np.percentile(np.concatenate([fields[0].ravel(), fields[1].ravel()]), 98)
        cmap = "coolwarm" if channel != "TP" else "viridis"

        for row in range(2):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                lon2d,
                lat2d,
                fields[row],
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(label_titles[row], fontsize=10)
            ax.set_xlabel("Longitude")
            if col == 0:
                ax.set_ylabel("Latitude")
            fig.colorbar(mesh, ax=ax, shrink=0.72)

    fig.suptitle("Selected normalized weather channels show salt-and-pepper texture", fontsize=14)
    fig.savefig(REPORT_IMG_DIR / "overview_maps.png", dpi=200)
    plt.close(fig)


def plot_correlation_audit(channel_df: pd.DataFrame) -> None:
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    sns.boxplot(
        data=channel_df,
        x="family_name",
        y="temporal_corr_t0_t1",
        ax=axes[0],
        color="#7da0c8",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Temporal correlation: t0 vs t1")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Area-weighted pattern correlation")
    axes[0].tick_params(axis="x", rotation=50)

    sns.boxplot(
        data=channel_df,
        x="family_name",
        y="temporal_corr_t1_f6",
        ax=axes[1],
        color="#9bc995",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Temporal correlation: t1 vs FuXi +6h")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Area-weighted pattern correlation")
    axes[1].tick_params(axis="x", rotation=50)

    spatial_long = channel_df.melt(
        id_vars=["channel", "family_name"],
        value_vars=["spatial_corr_lon_t1", "spatial_corr_lat_t1"],
        var_name="direction",
        value_name="neighbor_corr",
    )
    spatial_long["direction"] = spatial_long["direction"].map(
        {"spatial_corr_lon_t1": "Longitude neighbor", "spatial_corr_lat_t1": "Latitude neighbor"}
    )
    sns.boxplot(
        data=spatial_long,
        x="direction",
        y="neighbor_corr",
        ax=axes[2],
        color="#e3aa62",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Spatial nearest-neighbor correlation at t1")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Pearson correlation")

    fig.savefig(REPORT_IMG_DIR / "correlation_audit.png", dpi=200)
    plt.close(fig)


def plot_distribution_audit(channel_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    sns.scatterplot(
        data=channel_df,
        x="t1_mean",
        y="t1_std",
        hue="family_name",
        ax=axes[0],
        s=55,
    )
    axes[0].set_title("Channel means and standard deviations collapse tightly")
    axes[0].set_xlabel("Area-weighted mean at t1")
    axes[0].set_ylabel("Area-weighted standard deviation at t1")

    sns.scatterplot(
        data=channel_df,
        x="high_freq_power_fraction",
        y="spectral_slope_t1",
        hue="family_name",
        ax=axes[1],
        s=55,
        legend=False,
    )
    axes[1].set_title("Spectral diagnostics cluster near white-noise values")
    axes[1].set_xlabel("High-frequency power fraction (k >= 20)")
    axes[1].set_ylabel("Log-log spectral slope")

    fig.savefig(REPORT_IMG_DIR / "distribution_audit.png", dpi=200)
    plt.close(fig)


def plot_spectra(spectra_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    for channel in SELECTED_CHANNELS:
        curve = spectra_df[spectra_df["channel"] == channel]
        ax.plot(
            curve["wavenumber"].iloc[1:],
            curve["power"].iloc[1:] / curve["power"].iloc[1:].max(),
            label=channel,
            linewidth=1.6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Zonal wavenumber")
    ax.set_ylabel("Normalized power")
    ax.set_title("Selected channels have flat spectra instead of atmospheric red spectra")
    ax.legend(frameon=True)
    fig.savefig(REPORT_IMG_DIR / "spectra.png", dpi=200)
    plt.close(fig)


def write_outputs(
    channel_df: pd.DataFrame, family_df: pd.DataFrame, spectra_df: pd.DataFrame, metadata: dict
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    channel_df.to_csv(OUTPUT_DIR / "channel_metrics.csv", index=False)
    family_df.to_csv(OUTPUT_DIR / "family_summary.csv", index=False)
    spectra_df.to_csv(OUTPUT_DIR / "zonal_spectra.csv", index=False)
    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> None:
    data = load_data()
    channel_df, family_df, spectra_df, metadata = summarize_data(data)
    write_outputs(channel_df, family_df, spectra_df, metadata)

    plot_maps(data)
    plot_correlation_audit(channel_df)
    plot_distribution_audit(channel_df)
    plot_spectra(spectra_df)

    summary_lines = [
        "Dataset audit complete.",
        f"Input grid: {data.lat.size} x {data.lon.size} with nominal spacing "
        f"{abs(np.diff(data.lat).mean()):.2f}° x {abs(np.diff(data.lon).mean()):.2f}°.",
        f"Input tensor shape: {tuple(data.input_data.shape)}.",
        f"Forecast tensor shape: {tuple(data.forecast_data.shape)}.",
        f"Mean temporal correlation t0->t1 across channels: {channel_df['temporal_corr_t0_t1'].mean():.4f}.",
        f"Mean temporal correlation t1->FuXi+6h across channels: {channel_df['temporal_corr_t1_f6'].mean():.4f}.",
        f"Mean longitude-neighbor correlation: {channel_df['spatial_corr_lon_t1'].mean():.4f}.",
        f"Mean latitude-neighbor correlation: {channel_df['spatial_corr_lat_t1'].mean():.4f}.",
        f"Mean spectral slope: {channel_df['spectral_slope_t1'].mean():.4f}.",
    ]
    with open(OUTPUT_DIR / "key_findings.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
