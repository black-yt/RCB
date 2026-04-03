#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> Dict[str, np.ndarray]:
    raw_path = DATA_DIR / "raw_trARPES_data.h5"
    with h5py.File(raw_path, "r") as h5f:
        data = {key: h5f[key][...] for key in h5f.keys()}
    return data


def load_processed_data() -> Dict:
    with open(DATA_DIR / "processed_band_data.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_polarization_table() -> List[Dict[str, float]]:
    rows = []
    with open(DATA_DIR / "polarization_dependence_data.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "angle_degrees": float(row["angle_degrees"]),
                    "angle_radians": float(row["angle_radians"]),
                    "intensity": float(row["intensity"]),
                    "target_energy": float(row["target_energy"]),
                    "target_kx": float(row["target_kx"]),
                }
            )
    return rows


def nearest_idx(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def fit_cos2_model(angles_rad: np.ndarray, intensity: np.ndarray) -> Dict[str, float]:
    design = np.column_stack(
        [
            np.ones_like(angles_rad),
            np.cos(2.0 * angles_rad),
            np.sin(2.0 * angles_rad),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, intensity, rcond=None)
    pred = design @ coeffs
    ss_res = float(np.sum((intensity - pred) ** 2))
    ss_tot = float(np.sum((intensity - np.mean(intensity)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    amp = float(np.hypot(coeffs[1], coeffs[2]))
    phase = 0.5 * np.arctan2(coeffs[2], coeffs[1])
    return {
        "baseline": float(coeffs[0]),
        "cos2_coeff": float(coeffs[1]),
        "sin2_coeff": float(coeffs[2]),
        "amplitude": amp,
        "phase_radians": float(phase),
        "phase_degrees": float(np.degrees(phase)),
        "r_squared": float(r2),
        "predicted": pred.tolist(),
        "modulation_depth_fraction": float(amp / coeffs[0]) if coeffs[0] != 0 else float("nan"),
    }


def summarize_raw_data(raw: Dict[str, np.ndarray]) -> Dict:
    energy = raw["energy_axis"]
    kx = raw["kx_axis"]
    time_delays = raw["time_delays"]
    angles = raw["polarization_angles"]
    pump_off = raw["pump_off_spectrum"]

    pump_on_keys = sorted([k for k in raw if k.startswith("pump_on_angle_")], key=lambda x: int(x.split("_")[-1]))
    pump_on_stats = {}
    for key in pump_on_keys:
        arr = raw[key]
        diff = arr - pump_off
        pump_on_stats[key] = {
            "shape": list(arr.shape),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "mean_difference_vs_pump_off": float(np.mean(diff)),
            "max_difference_vs_pump_off": float(np.max(diff)),
        }

    return {
        "energy_points": int(energy.size),
        "kx_points": int(kx.size),
        "time_delay_points": int(time_delays.size),
        "pump_polarization_angles_deg": [int(a) for a in angles.tolist()],
        "energy_range_eV": [float(np.min(energy)), float(np.max(energy))],
        "kx_range_Ainv": [float(np.min(kx)), float(np.max(kx))],
        "time_delay_range_ps": [float(np.min(time_delays)), float(np.max(time_delays))],
        "pump_off_summary": {
            "shape": list(pump_off.shape),
            "min": float(np.min(pump_off)),
            "max": float(np.max(pump_off)),
            "mean": float(np.mean(pump_off)),
            "std": float(np.std(pump_off)),
        },
        "pump_on_summary": pump_on_stats,
    }


def compute_band_metrics(raw: Dict[str, np.ndarray], processed: Dict, pol_rows: List[Dict[str, float]]) -> Dict:
    energy = raw["energy_axis"]
    kx = raw["kx_axis"]
    pump_off = raw["pump_off_spectrum"]

    dirac_kx, dirac_energy = processed["dirac_point"]
    dirac_kx_idx = nearest_idx(kx, dirac_kx)
    dirac_energy_idx = nearest_idx(energy, dirac_energy)
    dirac_intensity = float(pump_off[dirac_energy_idx, dirac_kx_idx])

    replica_metrics = []
    for replica in processed["replica_bands"]:
        ridx_e = nearest_idx(energy, replica["energy"])
        ridx_k = nearest_idx(kx, replica["kx"])
        off_intensity = float(pump_off[ridx_e, ridx_k])
        pump_on_key = f"pump_on_angle_{processed['polarization_angle']}"
        on_intensity = float(raw[pump_on_key][ridx_e, ridx_k]) if pump_on_key in raw else float("nan")
        replica_metrics.append(
            {
                "order": int(replica["order"]),
                "kx_Ainv": float(replica["kx"]),
                "energy_eV": float(replica["energy"]),
                "raw_pump_off_intensity": off_intensity,
                "raw_pump_on_reference_angle_intensity": on_intensity,
                "processed_intensity": float(replica["intensity"]),
                "energy_offset_from_dirac_eV": float(replica["energy"] - dirac_energy),
                "momentum_offset_from_dirac_Ainv": float(replica["kx"] - dirac_kx),
                "pump_on_to_pump_off_ratio": float(on_intensity / off_intensity) if off_intensity != 0 else float("nan"),
                "pump_off_to_dirac_ratio": float(off_intensity / dirac_intensity) if dirac_intensity != 0 else float("nan"),
            }
        )

    dispersion = processed["band_dispersion"]
    energies = np.array([pt["energy"] for pt in dispersion], dtype=float)
    kxs = np.array([pt["kx"] for pt in dispersion], dtype=float)
    intensities = np.array([pt["intensity"] for pt in dispersion], dtype=float)
    mask = np.abs(energies - dirac_energy) > 0.03
    if np.any(mask):
        slope, intercept = np.polyfit(np.abs(kxs[mask]), np.abs(energies[mask] - dirac_energy), 1)
    else:
        slope, intercept = np.nan, np.nan

    angles_deg = np.array([row["angle_degrees"] for row in pol_rows], dtype=float)
    angles_rad = np.array([row["angle_radians"] for row in pol_rows], dtype=float)
    pol_intensity = np.array([row["intensity"] for row in pol_rows], dtype=float)
    pol_fit = fit_cos2_model(angles_rad, pol_intensity)

    return {
        "dirac_point": {
            "kx_Ainv": float(dirac_kx),
            "energy_eV": float(dirac_energy),
            "nearest_raw_kx_Ainv": float(kx[dirac_kx_idx]),
            "nearest_raw_energy_eV": float(energy[dirac_energy_idx]),
            "pump_off_intensity": dirac_intensity,
        },
        "replica_band_metrics": replica_metrics,
        "replica_energy_offsets_eV": [float(rep["energy"] - dirac_energy) for rep in processed["replica_bands"]],
        "pump_photon_energy_eV": float(processed["pump_energy"]),
        "mean_absolute_replica_offset_error_vs_pump_energy_eV": float(
            np.mean([
                abs(abs(rep["energy"] - dirac_energy) - processed["pump_energy"])
                for rep in processed["replica_bands"]
            ])
        ),
        "dispersion_fit": {
            "fit_type": "|E-E_D| vs |kx-kx_D| linear fit away from Dirac point",
            "slope_eV_per_Ainv": float(slope),
            "intercept_eV": float(intercept),
            "points_used": int(np.sum(mask)),
            "mean_processed_intensity": float(np.mean(intensities)),
        },
        "polarization_dependence": {
            "angles_deg": angles_deg.tolist(),
            "intensities": pol_intensity.tolist(),
            "max_intensity": float(np.max(pol_intensity)),
            "min_intensity": float(np.min(pol_intensity)),
            "anisotropy_ratio_max_over_min": float(np.max(pol_intensity) / np.min(pol_intensity)),
            "peak_to_peak_fraction_of_mean": float((np.max(pol_intensity) - np.min(pol_intensity)) / np.mean(pol_intensity)),
            "cos2_fit": pol_fit,
        },
    }


def write_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_text_summary(raw_summary: Dict, metrics: Dict) -> None:
    lines = []
    lines.append("Floquet-Bloch graphene tr-ARPES analysis summary")
    lines.append("=" * 48)
    lines.append("")
    lines.append("Data overview")
    lines.append(f"- Energy grid: {raw_summary['energy_points']} points spanning {raw_summary['energy_range_eV'][0]:.3f} to {raw_summary['energy_range_eV'][1]:.3f} eV")
    lines.append(f"- Momentum grid: {raw_summary['kx_points']} points spanning {raw_summary['kx_range_Ainv'][0]:.3f} to {raw_summary['kx_range_Ainv'][1]:.3f} A^-1")
    lines.append(f"- Time delays: {raw_summary['time_delay_points']} values spanning {raw_summary['time_delay_range_ps'][0]:.3f} to {raw_summary['time_delay_range_ps'][1]:.3f} ps")
    lines.append(f"- Pump polarization angles: {raw_summary['pump_polarization_angles_deg']}")
    lines.append("")
    d = metrics["dirac_point"]
    lines.append("Key physical observables")
    lines.append(f"- Dirac point near kx = {d['kx_Ainv']:.4f} A^-1, E = {d['energy_eV']:.4f} eV")
    lines.append(f"- Pump photon energy in processed features: {metrics['pump_photon_energy_eV']:.4f} eV")
    lines.append(f"- Mean absolute mismatch between |replica offset| and pump energy: {metrics['mean_absolute_replica_offset_error_vs_pump_energy_eV']:.5f} eV")
    lines.append(f"- Approximate |E-E_D| vs |k-k_D| slope: {metrics['dispersion_fit']['slope_eV_per_Ainv']:.4f} eV/A^-1")
    lines.append("")
    lines.append("Replica bands")
    for rep in metrics["replica_band_metrics"]:
        lines.append(
            f"- Order {rep['order']:+d}: kx={rep['kx_Ainv']:.4f} A^-1, E={rep['energy_eV']:.4f} eV, "
            f"offset={rep['energy_offset_from_dirac_eV']:+.4f} eV, "
            f"pump-on/pump-off={rep['pump_on_to_pump_off_ratio']:.3f}"
        )
    lines.append("")
    pol = metrics["polarization_dependence"]
    fit = pol["cos2_fit"]
    lines.append("Polarization dependence")
    lines.append(f"- Intensity anisotropy max/min = {pol['anisotropy_ratio_max_over_min']:.4f}")
    lines.append(f"- Peak-to-peak modulation / mean = {pol['peak_to_peak_fraction_of_mean']:.4f}")
    lines.append(f"- cos(2θ) fit R^2 = {fit['r_squared']:.4f}, modulation depth = {fit['modulation_depth_fraction']:.4f}")
    lines.append(f"- Fitted phase = {fit['phase_degrees']:.2f} degrees")
    lines.append("")
    lines.append("Interpretive note")
    lines.append(
        "- Replica energies sit approximately one pump-photon energy above and below the Dirac point, while the polarization response is weak but twofold-symmetric, "
        "consistent with Floquet-Bloch sidebands mixed with polarization-sensitive Volkov-type final-state dressing."
    )
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_spectra(raw: Dict[str, np.ndarray], processed: Dict) -> None:
    energy = raw["energy_axis"]
    kx = raw["kx_axis"]
    pump_off = raw["pump_off_spectrum"]
    reference_key = f"pump_on_angle_{processed['polarization_angle']}"
    pump_on = raw[reference_key]
    diff = pump_on - pump_off

    extent = [float(kx.min()), float(kx.max()), float(energy.min()), float(energy.max())]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    images = []
    for ax, arr, title, cmap in [
        (axes[0], pump_off, "Pump-off spectrum", "magma"),
        (axes[1], pump_on, f"Pump-on spectrum ({processed['polarization_angle']}°)", "magma"),
        (axes[2], diff, "Pump-on minus pump-off", "coolwarm"),
    ]:
        vmax = np.percentile(arr, 99)
        vmin = np.percentile(arr, 1) if cmap != "coolwarm" else -max(abs(np.percentile(arr, 1)), abs(np.percentile(arr, 99)))
        if cmap == "coolwarm":
            vmax = max(abs(np.percentile(arr, 1)), abs(np.percentile(arr, 99)))
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        images.append(im)
        ax.set_title(title)
        ax.set_xlabel(r"$k_x$ (A$^{-1}$)")
        ax.set_ylabel("Energy (eV)")
        ax.scatter(processed["dirac_point"][0], processed["dirac_point"][1], c="cyan", s=50, marker="x", label="Dirac point")
        for rep in processed["replica_bands"]:
            color = "lime" if rep["order"] > 0 else "yellow"
            ax.scatter(rep["kx"], rep["energy"], c=color, s=28)
        ax.legend(loc="upper right", fontsize=8)
    for ax, im in zip(axes, images):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(IMAGE_DIR / "spectral_overview.png", dpi=200)
    plt.close(fig)


def plot_replica_analysis(raw: Dict[str, np.ndarray], processed: Dict) -> None:
    energy = np.array(processed["energy_axis"], dtype=float)
    kx = np.array(processed["kx_axis"], dtype=float)
    pump_off = raw["pump_off_spectrum"]

    dirac_kx, dirac_energy = processed["dirac_point"]
    mask_e = (energy >= dirac_energy - 0.35) & (energy <= dirac_energy + 0.35)
    mask_k = (kx >= -0.12) & (kx <= 0.12)
    crop = pump_off[np.ix_(mask_e, mask_k)]
    extent = [float(kx[mask_k].min()), float(kx[mask_k].max()), float(energy[mask_e].min()), float(energy[mask_e].max())]

    dispersion = processed["band_dispersion"]
    disp_e = np.array([pt["energy"] for pt in dispersion], dtype=float)
    disp_k = np.array([pt["kx"] for pt in dispersion], dtype=float)
    disp_i = np.array([pt["intensity"] for pt in dispersion], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    im = axes[0].imshow(crop, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=np.percentile(crop, 2), vmax=np.percentile(crop, 99))
    axes[0].scatter(disp_k, disp_e, c="white", s=10, alpha=0.7, label="Extracted dispersion")
    axes[0].scatter(dirac_kx, dirac_energy, c="cyan", marker="x", s=70, label="Dirac point")
    for rep in processed["replica_bands"]:
        axes[0].scatter(rep["kx"], rep["energy"], c=("lime" if rep["order"] > 0 else "yellow"), s=40)
        axes[0].axhline(rep["energy"], color=("lime" if rep["order"] > 0 else "yellow"), lw=0.8, ls="--", alpha=0.6)
    axes[0].axhline(dirac_energy, color="cyan", lw=1.0, ls=":", alpha=0.8)
    axes[0].set_title("Dirac cone and Floquet replica locations")
    axes[0].set_xlabel(r"$k_x$ (A$^{-1}$)")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)

    abs_delta_e = np.abs(disp_e - dirac_energy)
    abs_delta_k = np.abs(disp_k - dirac_kx)
    mask = abs_delta_e > 0.03
    slope, intercept = np.polyfit(abs_delta_k[mask], abs_delta_e[mask], 1)
    fit_x = np.linspace(0, abs_delta_k.max(), 200)
    fit_y = slope * fit_x + intercept
    sc = axes[1].scatter(abs_delta_k, abs_delta_e, c=disp_i, cmap="viridis", s=18)
    axes[1].plot(fit_x, fit_y, color="red", lw=2, label=f"Linear fit: slope={slope:.3f} eV/A$^{{-1}}$")
    axes[1].axhline(processed["pump_energy"], color="black", lw=1.2, ls="--", label=f"Pump photon energy = {processed['pump_energy']:.3f} eV")
    for rep in processed["replica_bands"]:
        axes[1].scatter(abs(rep["kx"] - dirac_kx), abs(rep["energy"] - dirac_energy), c="orange", edgecolor="black", s=70)
    axes[1].set_title("Dispersion scale and replica energy offsets")
    axes[1].set_xlabel(r"$|k_x-k_D|$ (A$^{-1}$)")
    axes[1].set_ylabel(r"$|E-E_D|$ (eV)")
    axes[1].legend(loc="upper left", fontsize=8)
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03, label="Processed band intensity")

    fig.savefig(IMAGE_DIR / "replica_band_analysis.png", dpi=200)
    plt.close(fig)


def plot_polarization_dependence(pol_rows: List[Dict[str, float]]) -> Dict[str, float]:
    angles_deg = np.array([row["angle_degrees"] for row in pol_rows], dtype=float)
    angles_rad = np.array([row["angle_radians"] for row in pol_rows], dtype=float)
    intensity = np.array([row["intensity"] for row in pol_rows], dtype=float)
    fit = fit_cos2_model(angles_rad, intensity)

    theta_dense = np.linspace(0, np.pi, 360)
    fit_dense = (
        fit["baseline"]
        + fit["cos2_coeff"] * np.cos(2.0 * theta_dense)
        + fit["sin2_coeff"] * np.sin(2.0 * theta_dense)
    )

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(angles_deg, intensity, "o-", color="tab:blue", label="Measured replica intensity")
    ax1.plot(np.degrees(theta_dense), fit_dense, "--", color="tab:red", label=r"$a+b\cos 2\theta+c\sin 2\theta$")
    ax1.set_xlabel("Pump polarization angle (degrees)")
    ax1.set_ylabel("Replica-band intensity (arb. units)")
    ax1.set_title("Polarization-dependent replica intensity")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2.plot(angles_rad, intensity, "o", color="tab:blue")
    ax2.plot(theta_dense, fit_dense, color="tab:red")
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(-1)
    ax2.set_title("Weak twofold anisotropy in replica signal")

    fig.savefig(IMAGE_DIR / "polarization_dependence_analysis.png", dpi=200)
    plt.close(fig)
    return fit


def plot_angle_grid(raw: Dict[str, np.ndarray]) -> None:
    energy = raw["energy_axis"]
    kx = raw["kx_axis"]
    pump_off = raw["pump_off_spectrum"]
    angles = [int(a) for a in raw["polarization_angles"].tolist()]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.ravel()
    extent = [float(kx.min()), float(kx.max()), float(energy.min()), float(energy.max())]

    all_maps = [("off", pump_off)] + [(str(angle), raw[f"pump_on_angle_{angle}"]) for angle in angles]
    for ax, (label, arr) in zip(axes, all_maps):
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=np.percentile(arr, 2), vmax=np.percentile(arr, 99))
        ax.set_title("Pump off" if label == "off" else f"Pump on {label}°")
        ax.set_xlabel(r"$k_x$ (A$^{-1}$)")
        ax.set_ylabel("Energy (eV)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes[len(all_maps):]:
        ax.axis("off")

    fig.savefig(IMAGE_DIR / "pump_angle_grid.png", dpi=180)
    plt.close(fig)


def save_replica_table(metrics: Dict) -> None:
    rows = metrics["replica_band_metrics"]
    path = OUTPUT_DIR / "replica_band_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    raw = load_raw_data()
    processed = load_processed_data()
    pol_rows = load_polarization_table()

    raw_summary = summarize_raw_data(raw)
    metrics = compute_band_metrics(raw, processed, pol_rows)
    metrics["polarization_dependence"]["cos2_fit"] = plot_polarization_dependence(pol_rows)

    plot_spectra(raw, processed)
    plot_replica_analysis(raw, processed)
    plot_angle_grid(raw)

    write_json(OUTPUT_DIR / "raw_data_summary.json", raw_summary)
    write_json(OUTPUT_DIR / "analysis_metrics.json", metrics)
    save_replica_table(metrics)
    write_text_summary(raw_summary, metrics)

    print("Analysis complete.")
    print(f"Wrote summaries to: {OUTPUT_DIR}")
    print(f"Wrote figures to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
