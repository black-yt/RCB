#!/usr/bin/env python3
"""Main entry point for workspace-specific ERA5/FuXi sample analysis.

This script performs a reproducible exploratory analysis of the provided
weather data files, writes numerical summaries under outputs/, and saves
report-ready figures under report/images/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import netcdf_file


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"

INPUT_FILE = DATA_DIR / "20231012-06_input_netcdf.nc"
FORECAST_FILE = DATA_DIR / "006.nc"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def read_netcdf(path: Path) -> Dict[str, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as ds:
        lat = np.array(ds.variables["lat"].data.copy())
        lon = np.array(ds.variables["lon"].data.copy())
        level_raw = np.array(ds.variables["level"].data.copy())
        level_names = [
            "".join(ch.decode() if isinstance(ch, bytes) else str(ch) for ch in row).strip()
            for row in level_raw
        ]
        time = np.array(ds.variables["time"].data.copy()) if "time" in ds.variables else None
        step = np.array(ds.variables["step"].data.copy()) if "step" in ds.variables else None
        data = np.array(ds.variables["data"].data.copy())
    return {
        "lat": lat,
        "lon": lon,
        "level_names": np.array(level_names, dtype=object),
        "time": time,
        "step": step,
        "data": data,
    }


def parse_channel(channel: str) -> Tuple[str, str]:
    surface = {"T2M", "U10", "V10", "MSL", "TP"}
    if channel in surface:
        return channel, "surface"
    prefix = channel[0]
    suffix = channel[1:]
    return prefix, suffix


def build_channel_table(level_names: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, channel in enumerate(level_names.tolist()):
        variable_group, pressure = parse_channel(channel)
        rows.append(
            {
                "channel_index": idx,
                "channel_name": channel,
                "variable_group": variable_group,
                "pressure_or_type": pressure,
                "is_surface": pressure == "surface",
            }
        )
    return pd.DataFrame(rows)


def summarize_tensor(data: np.ndarray, channel_names: List[str], kind: str) -> pd.DataFrame:
    if data.ndim == 4:
        # (time, channel, lat, lon)
        axis_prefix = "time"
        n_slices = data.shape[0]
    elif data.ndim == 5:
        # (time, step, channel, lat, lon)
        axis_prefix = "forecast_slice"
        n_slices = data.shape[0] * data.shape[1]
        data = data.reshape(n_slices, data.shape[2], data.shape[3], data.shape[4])
    else:
        raise ValueError(f"Unexpected data rank for {kind}: {data.shape}")

    rows = []
    for slice_idx in range(n_slices):
        block = data[slice_idx]
        for ch_idx, ch_name in enumerate(channel_names):
            arr = block[ch_idx]
            rows.append(
                {
                    "dataset": kind,
                    axis_prefix: int(slice_idx),
                    "channel_index": int(ch_idx),
                    "channel_name": ch_name,
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "p01": float(np.percentile(arr, 1)),
                    "p50": float(np.percentile(arr, 50)),
                    "p99": float(np.percentile(arr, 99)),
                }
            )
    return pd.DataFrame(rows)


def area_weights(latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))
    weights = np.clip(weights, 0.0, None)
    return weights / weights.mean()


def weighted_rmse(a: np.ndarray, b: np.ndarray, latitudes: np.ndarray) -> float:
    diff2 = (a - b) ** 2
    w = area_weights(latitudes)[:, None]
    return float(np.sqrt(np.sum(diff2 * w) / np.sum(np.ones_like(diff2) * w)))


def weighted_bias(a: np.ndarray, b: np.ndarray, latitudes: np.ndarray) -> float:
    diff = a - b
    w = area_weights(latitudes)[:, None]
    return float(np.sum(diff * w) / np.sum(np.ones_like(diff) * w))


def weighted_corr(a: np.ndarray, b: np.ndarray, latitudes: np.ndarray) -> float:
    w = area_weights(latitudes)[:, None]
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    w_flat = np.broadcast_to(w, a.shape).reshape(-1)
    wa = np.sum(w_flat * a_flat) / np.sum(w_flat)
    wb = np.sum(w_flat * b_flat) / np.sum(w_flat)
    av = a_flat - wa
    bv = b_flat - wb
    denom = np.sqrt(np.sum(w_flat * av * av) * np.sum(w_flat * bv * bv))
    if denom == 0:
        return float("nan")
    return float(np.sum(w_flat * av * bv) / denom)


def compare_forecast_to_latest_input(
    input_data: np.ndarray,
    forecast_data: np.ndarray,
    channel_names: List[str],
    latitudes: np.ndarray,
) -> pd.DataFrame:
    latest_input = input_data[-1]  # second 6-hour state
    forecast_6h = forecast_data[0, 0]
    rows = []
    for ch_idx, ch_name in enumerate(channel_names):
        ref = latest_input[ch_idx]
        pred = forecast_6h[ch_idx]
        rows.append(
            {
                "channel_index": ch_idx,
                "channel_name": ch_name,
                "variable_group": parse_channel(ch_name)[0],
                "rmse_vs_latest_input": weighted_rmse(pred, ref, latitudes),
                "bias_vs_latest_input": weighted_bias(pred, ref, latitudes),
                "corr_vs_latest_input": weighted_corr(pred, ref, latitudes),
                "forecast_mean": float(pred.mean()),
                "latest_input_mean": float(ref.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_metadata(input_ds: Dict[str, np.ndarray], forecast_ds: Dict[str, np.ndarray]) -> None:
    channel_names = input_ds["level_names"].tolist()
    groups = defaultdict(int)
    for ch in channel_names:
        groups[parse_channel(ch)[0]] += 1

    metadata = {
        "workspace": str(ROOT),
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "forecast_file": str(FORECAST_FILE.relative_to(ROOT)),
        "input_shape": list(map(int, input_ds["data"].shape)),
        "forecast_shape": list(map(int, forecast_ds["data"].shape)),
        "latitude_count": int(input_ds["lat"].shape[0]),
        "longitude_count": int(input_ds["lon"].shape[0]),
        "latitude_range": [float(input_ds["lat"].min()), float(input_ds["lat"].max())],
        "longitude_range": [float(input_ds["lon"].min()), float(input_ds["lon"].max())],
        "time_values_input": input_ds["time"].tolist() if input_ds["time"] is not None else None,
        "time_values_forecast": forecast_ds["time"].tolist() if forecast_ds["time"] is not None else None,
        "step_values_forecast": forecast_ds["step"].tolist() if forecast_ds["step"] is not None else None,
        "channel_count": len(channel_names),
        "channels": channel_names,
        "channel_group_counts": dict(groups),
    }
    (OUTPUT_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_domain(lat: np.ndarray, lon: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lon, np.zeros_like(lon), "o", markersize=2, alpha=0.5, label="Longitudes")
    ax.plot(np.zeros_like(lat), lat, "o", markersize=2, alpha=0.5, label="Latitudes")
    ax.set_title("Coordinate coverage of provided sample grid")
    ax.set_xlabel("Degrees")
    ax.set_ylabel("Degrees")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "grid_coverage.png", dpi=180)
    plt.close(fig)


def plot_channel_group_means(summary_df: pd.DataFrame) -> None:
    tmp = summary_df.copy()
    tmp["variable_group"] = tmp["channel_name"].apply(lambda x: parse_channel(x)[0])
    group_means = (
        tmp.groupby(["dataset", "variable_group"], as_index=False)[["mean", "std"]]
        .mean()
        .sort_values(["dataset", "variable_group"])
    )

    datasets = group_means["dataset"].unique().tolist()
    groups = group_means["variable_group"].unique().tolist()
    x = np.arange(len(groups))
    width = 0.35 if len(datasets) > 1 else 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, dataset in enumerate(datasets):
        vals = [
            float(group_means[(group_means["dataset"] == dataset) & (group_means["variable_group"] == g)]["mean"].iloc[0])
            for g in groups
        ]
        ax.bar(x + (i - (len(datasets) - 1) / 2) * width, vals, width=width, label=dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title("Mean standardized value by variable group")
    ax.set_ylabel("Spatial mean averaged over channels")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "channel_group_means.png", dpi=180)
    plt.close(fig)


def plot_channel_rmse(compare_df: pd.DataFrame) -> None:
    top = compare_df.sort_values("rmse_vs_latest_input", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["channel_name"], top["rmse_vs_latest_input"], color="tab:red", alpha=0.8)
    ax.set_title("Top 20 channels by 6-hour forecast RMSE vs latest input state")
    ax.set_xlabel("Latitude-weighted RMSE")
    ax.set_ylabel("Channel")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "top_channel_rmse.png", dpi=180)
    plt.close(fig)


def plot_spatial_maps(input_ds: Dict[str, np.ndarray], forecast_ds: Dict[str, np.ndarray]) -> None:
    lat = input_ds["lat"]
    lon = input_ds["lon"]
    channels = input_ds["level_names"].tolist()
    latest_input = input_ds["data"][-1]
    forecast = forecast_ds["data"][0, 0]
    targets = ["Z500", "T850", "U500", "T2M", "MSL", "TP"]
    available = [ch for ch in targets if ch in channels]

    fig, axes = plt.subplots(len(available), 3, figsize=(14, 3.3 * len(available)), constrained_layout=True)
    if len(available) == 1:
        axes = np.array([axes])

    lon_extent = [float(lon.min()), float(lon.max())]
    lat_extent = [float(lat.min()), float(lat.max())]
    extent = [lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]]

    for row, ch in enumerate(available):
        idx = channels.index(ch)
        inp = latest_input[idx]
        pred = forecast[idx]
        delta = pred - inp

        im0 = axes[row, 0].imshow(inp, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        axes[row, 0].set_title(f"Latest input: {ch}")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.03, pad=0.02)

        im1 = axes[row, 1].imshow(pred, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        axes[row, 1].set_title(f"Forecast +6h: {ch}")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.03, pad=0.02)

        vmax = np.nanmax(np.abs(delta))
        im2 = axes[row, 2].imshow(delta, origin="lower", aspect="auto", extent=extent, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[row, 2].set_title(f"Forecast - latest input: {ch}")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.03, pad=0.02)

        for col in range(3):
            axes[row, col].set_xlabel("Longitude")
            axes[row, col].set_ylabel("Latitude")

    fig.savefig(IMAGE_DIR / "sample_spatial_maps.png", dpi=180)
    plt.close(fig)


def plot_zonal_profiles(input_ds: Dict[str, np.ndarray], forecast_ds: Dict[str, np.ndarray]) -> None:
    lat = input_ds["lat"]
    channels = input_ds["level_names"].tolist()
    latest_input = input_ds["data"][-1]
    forecast = forecast_ds["data"][0, 0]
    targets = ["Z500", "T850", "U500", "T2M", "MSL", "TP"]
    available = [ch for ch in targets if ch in channels]

    fig, axes = plt.subplots(len(available), 1, figsize=(9, 2.8 * len(available)), sharex=True, constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    for ax, ch in zip(axes, available):
        idx = channels.index(ch)
        inp_profile = latest_input[idx].mean(axis=1)
        pred_profile = forecast[idx].mean(axis=1)
        ax.plot(lat, inp_profile, label="Latest input", linewidth=2)
        ax.plot(lat, pred_profile, label="Forecast +6h", linewidth=2)
        ax.set_title(f"Zonal mean profile: {ch}")
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Latitude")
    fig.savefig(IMAGE_DIR / "zonal_profiles.png", dpi=180)
    plt.close(fig)


def write_method_note() -> None:
    text = """This workspace only provides one two-step input sample and one +6h forecast sample, not a full multi-year training corpus. Accordingly, the analysis is designed as a rigorous exploratory and validation-oriented study rather than an unsupported end-to-end model training claim.

Planned scientific framing for the later report:
- Treat the benchmark objective as a cascade forecasting design problem inspired by FourCastNet and FengWu.
- Use three specialized U-Transformer stages conceptually partitioned by lead time (short-range, medium-range, extended-range) to limit autoregressive error accumulation.
- Use the provided files to document variable layout, sample forecast behavior, and practical constraints on evaluation in this workspace.
- Report quantitative diagnostics that are feasible from the available sample, especially channel-wise weighted RMSE/bias/correlation between the +6h forecast and the latest available input state.
"""
    (OUTPUT_DIR / "analysis_scope_note.txt").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    input_ds = read_netcdf(INPUT_FILE)
    forecast_ds = read_netcdf(FORECAST_FILE)

    channel_names = input_ds["level_names"].tolist()
    if channel_names != forecast_ds["level_names"].tolist():
        raise RuntimeError("Input and forecast channel layouts do not match.")

    channel_table = build_channel_table(input_ds["level_names"])
    input_summary = summarize_tensor(input_ds["data"], channel_names, kind="input")
    forecast_summary = summarize_tensor(forecast_ds["data"], channel_names, kind="forecast")
    compare_df = compare_forecast_to_latest_input(
        input_ds["data"], forecast_ds["data"], channel_names, input_ds["lat"]
    )

    channel_table.to_csv(OUTPUT_DIR / "channel_layout.csv", index=False)
    input_summary.to_csv(OUTPUT_DIR / "input_channel_summary.csv", index=False)
    forecast_summary.to_csv(OUTPUT_DIR / "forecast_channel_summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "forecast_vs_latest_input_metrics.csv", index=False)

    group_metrics = (
        compare_df.groupby("variable_group", as_index=False)[
            ["rmse_vs_latest_input", "bias_vs_latest_input", "corr_vs_latest_input"]
        ]
        .mean()
        .sort_values("rmse_vs_latest_input", ascending=False)
    )
    group_metrics.to_csv(OUTPUT_DIR / "group_level_metrics.csv", index=False)

    overall_summary = {
        "global_weighted_mean_rmse": float(compare_df["rmse_vs_latest_input"].mean()),
        "global_weighted_mean_abs_bias": float(compare_df["bias_vs_latest_input"].abs().mean()),
        "global_weighted_mean_corr": float(compare_df["corr_vs_latest_input"].mean()),
        "highest_rmse_channel": compare_df.sort_values("rmse_vs_latest_input", ascending=False).iloc[0]["channel_name"],
        "highest_corr_channel": compare_df.sort_values("corr_vs_latest_input", ascending=False).iloc[0]["channel_name"],
        "lowest_corr_channel": compare_df.sort_values("corr_vs_latest_input", ascending=True).iloc[0]["channel_name"],
    }
    (OUTPUT_DIR / "overall_metrics.json").write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    save_metadata(input_ds, forecast_ds)
    write_method_note()

    combined_summary = pd.concat([input_summary, forecast_summary], ignore_index=True)
    plot_domain(input_ds["lat"], input_ds["lon"])
    plot_channel_group_means(combined_summary)
    plot_channel_rmse(compare_df)
    plot_spatial_maps(input_ds, forecast_ds)
    plot_zonal_profiles(input_ds, forecast_ds)

    top_text = compare_df.sort_values("rmse_vs_latest_input", ascending=False).head(10).to_string(index=False)
    (OUTPUT_DIR / "top_rmse_channels.txt").write_text(top_text + "\n", encoding="utf-8")

    print("Analysis complete.")
    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print(f"Wrote figures to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
