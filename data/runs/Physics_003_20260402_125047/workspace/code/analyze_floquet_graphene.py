from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


@dataclass
class ConeFit:
    slope_abs_k_per_e: float
    intercept_abs_k: float
    v_fermi_eva: float
    rmse_abs_k: float
    dirac_energy_ev: float
    dirac_kx_a_inv: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> dict:
    with h5py.File(DATA_DIR / "raw_trARPES_data.h5", "r") as h5:
        data = {
            "energy": h5["energy_axis"][:],
            "kx": h5["kx_axis"][:],
            "time_delays": h5["time_delays"][:],
            "polarization_angles": h5["polarization_angles"][:],
            "pump_off": h5["pump_off_spectrum"][:],
            "pump_on": {
                int(angle): h5[f"pump_on_angle_{int(angle)}"][:]
                for angle in h5["polarization_angles"][:]
            },
        }

    with open(DATA_DIR / "processed_band_data.json") as fh:
        data["processed"] = json.load(fh)

    data["polarization_scan"] = pd.read_csv(DATA_DIR / "polarization_dependence_data.csv")
    return data


def extract_equilibrium_cone(energy: np.ndarray, kx: np.ndarray, pump_off: np.ndarray) -> tuple[ConeFit, pd.DataFrame]:
    smoothed = gaussian_filter1d(pump_off, sigma=1.2, axis=1)
    mask = (np.abs(energy) >= 0.03) & (np.abs(energy) <= 0.30)

    rows = []
    for e_val, row in zip(energy[mask], smoothed[mask]):
        pos_mask = kx > 0
        neg_mask = kx < 0
        k_pos = kx[pos_mask][np.argmax(row[pos_mask])]
        k_neg = kx[neg_mask][np.argmax(row[neg_mask])]
        rows.append(
            {
                "energy_ev": float(e_val),
                "kx_pos_a_inv": float(k_pos),
                "kx_neg_a_inv": float(k_neg),
                "abs_energy_ev": float(abs(e_val)),
                "abs_kx_a_inv": float((abs(k_pos) + abs(k_neg)) / 2.0),
            }
        )

    branch_df = pd.DataFrame(rows)
    coeffs = np.polyfit(branch_df["abs_energy_ev"], branch_df["abs_kx_a_inv"], 1)
    pred = np.polyval(coeffs, branch_df["abs_energy_ev"])
    rmse = float(np.sqrt(np.mean((branch_df["abs_kx_a_inv"] - pred) ** 2)))

    apex_idx = np.unravel_index(np.argmax(pump_off), pump_off.shape)
    cone_fit = ConeFit(
        slope_abs_k_per_e=float(coeffs[0]),
        intercept_abs_k=float(coeffs[1]),
        v_fermi_eva=float(1.0 / coeffs[0]),
        rmse_abs_k=rmse,
        dirac_energy_ev=float(energy[apex_idx[0]]),
        dirac_kx_a_inv=float(kx[apex_idx[1]]),
    )
    return cone_fit, branch_df


def harmonic4(theta: np.ndarray, c0: float, c4: float, phi: float) -> np.ndarray:
    return c0 + c4 * np.cos(4.0 * (theta - phi))


def analyze_polarization_scan(scan_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    popt, pcov = curve_fit(
        harmonic4,
        scan_df["angle_radians"].to_numpy(),
        scan_df["intensity"].to_numpy(),
        p0=[scan_df["intensity"].mean(), 0.005, 0.0],
    )
    pred = harmonic4(scan_df["angle_radians"].to_numpy(), *popt)
    residual = scan_df["intensity"].to_numpy() - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((scan_df["intensity"] - scan_df["intensity"].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    fit_summary = {
        "baseline_intensity": float(popt[0]),
        "fourfold_amplitude": float(popt[1]),
        "phase_radians": float(popt[2]),
        "phase_degrees": float(np.degrees(popt[2])),
        "r2": float(r2),
        "amplitude_fraction_of_baseline": float(popt[1] / popt[0]),
        "parameter_std": [float(x) for x in np.sqrt(np.diag(pcov))],
    }

    fitted_df = scan_df.copy()
    fitted_df["fit_intensity"] = pred
    fitted_df["residual"] = residual
    return fit_summary, fitted_df


def extract_replica_metrics(
    energy: np.ndarray,
    kx: np.ndarray,
    pump_off: np.ndarray,
    pump_on: dict[int, np.ndarray],
    cone_fit: ConeFit,
) -> tuple[pd.DataFrame, dict]:
    photon_energy_ev = 1.239841984 / 5.0
    target_k = cone_fit.slope_abs_k_per_e * photon_energy_ev + cone_fit.intercept_abs_k

    points = [
        ("crossing_E0_plusk", 0.0, target_k),
        ("crossing_E0_minusk", 0.0, -target_k),
        ("replica_plus", photon_energy_ev, 0.0),
        ("replica_minus", -photon_energy_ev, 0.0),
    ]

    rows = []
    for angle, arr in pump_on.items():
        diff = arr - pump_off
        for label, e_target, k_target in points:
            e_idx = int(np.argmin(np.abs(energy - e_target)))
            k_idx = int(np.argmin(np.abs(kx - k_target)))
            rows.append(
                {
                    "angle_deg": angle,
                    "feature": label,
                    "target_energy_ev": float(energy[e_idx]),
                    "target_kx_a_inv": float(kx[k_idx]),
                    "pump_on_intensity": float(arr[e_idx, k_idx]),
                    "pump_off_intensity": float(pump_off[e_idx, k_idx]),
                    "difference_intensity": float(diff[e_idx, k_idx]),
                }
            )

    metrics_df = pd.DataFrame(rows)

    peak_summary = {}
    diff0 = pump_on[0] - pump_off
    for label, e_target, k_target in points:
        e_mask = np.abs(energy - e_target) < 0.03
        k_mask = np.abs(kx - k_target) < 0.03
        sub = diff0[np.ix_(e_mask, k_mask)]
        idx = np.unravel_index(np.argmax(sub), sub.shape)
        e_vals = energy[e_mask]
        k_vals = kx[k_mask]
        peak_summary[label] = {
            "peak_difference_intensity": float(sub[idx]),
            "peak_energy_ev": float(e_vals[idx[0]]),
            "peak_kx_a_inv": float(k_vals[idx[1]]),
        }

    return metrics_df, {
        "pump_photon_energy_ev": float(photon_energy_ev),
        "predicted_sideband_crossing_kx_a_inv": float(target_k),
        "angle0_peak_locations": peak_summary,
    }


def compare_processed_metadata(data: dict, cone_fit: ConeFit) -> dict:
    processed = data["processed"]
    scan = data["polarization_scan"]

    comparison = {
        "json_pump_energy_ev": float(processed.get("pump_energy", np.nan)),
        "wavelength_inferred_photon_energy_ev": float(1.239841984 / 5.0),
        "csv_target_energy_ev": float(scan["target_energy"].iloc[0]),
        "csv_target_kx_a_inv": float(scan["target_kx"].iloc[0]),
        "raw_dirac_energy_ev": float(cone_fit.dirac_energy_ev),
        "raw_dirac_kx_a_inv": float(cone_fit.dirac_kx_a_inv),
        "json_dirac_point_raw": processed.get("dirac_point"),
        "note": (
            "Replica energies and target momentum in the helper files are internally consistent with the raw spectra, "
            "but the absolute dirac_point coordinates in processed_band_data.json do not match the raw HDF5 apex and "
            "were not used as the primary quantitative reference."
        ),
    }
    return comparison


def save_tables(branch_df: pd.DataFrame, polarization_df: pd.DataFrame, replica_df: pd.DataFrame, summary: dict) -> None:
    branch_df.to_csv(OUTPUT_DIR / "equilibrium_cone_fit_points.csv", index=False)
    polarization_df.to_csv(OUTPUT_DIR / "polarization_fit.csv", index=False)
    replica_df.to_csv(OUTPUT_DIR / "replica_feature_metrics.csv", index=False)
    with open(OUTPUT_DIR / "analysis_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def plot_overview(energy: np.ndarray, kx: np.ndarray, pump_off: np.ndarray, pump_on0: np.ndarray, summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    extent = [kx.min(), kx.max(), energy.min(), energy.max()]

    im0 = axes[0].imshow(pump_off, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[0].set_title("Pump Off")
    axes[0].set_xlabel(r"$k_x$ (1/Angstrom)")
    axes[0].set_ylabel("Energy (eV)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Intensity")

    im1 = axes[1].imshow(pump_on0, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[1].set_title("Pump On, 0 deg")
    axes[1].set_xlabel(r"$k_x$ (1/Angstrom)")
    axes[1].set_ylabel("Energy (eV)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Intensity")

    diff0 = pump_on0 - pump_off
    vmax = np.percentile(np.abs(diff0), 99.5)
    im2 = axes[2].imshow(
        diff0,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[2].set_title("Pump-Induced Difference")
    axes[2].set_xlabel(r"$k_x$ (1/Angstrom)")
    axes[2].set_ylabel("Energy (eV)")

    for point in summary["replica_geometry"]["angle0_peak_locations"].values():
        axes[2].plot(point["peak_kx_a_inv"], point["peak_energy_ev"], "ko", ms=4)

    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="Delta intensity")
    fig.savefig(REPORT_IMG_DIR / "figure_1_data_overview.png", dpi=300)
    plt.close(fig)


def plot_cone_and_sideband_geometry(
    branch_df: pd.DataFrame,
    cone_fit: ConeFit,
    summary: dict,
) -> None:
    photon_energy_ev = summary["replica_geometry"]["pump_photon_energy_ev"]
    e_line = np.linspace(0, branch_df["abs_energy_ev"].max(), 300)
    k_line = cone_fit.slope_abs_k_per_e * e_line + cone_fit.intercept_abs_k

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].scatter(branch_df["abs_energy_ev"], branch_df["abs_kx_a_inv"], s=18, color="#0d6efd", alpha=0.8)
    axes[0].plot(e_line, k_line, color="#d9480f", lw=2)
    axes[0].axvline(photon_energy_ev, color="black", ls="--", lw=1)
    axes[0].axhline(summary["replica_geometry"]["predicted_sideband_crossing_kx_a_inv"], color="black", ls="--", lw=1)
    axes[0].set_xlabel(r"$|E-E_D|$ (eV)")
    axes[0].set_ylabel(r"$|k_x-k_D|$ (1/Angstrom)")
    axes[0].set_title("Equilibrium Dirac-Cone Fit")
    axes[0].text(
        0.03,
        0.95,
        f"$v_F$ = {cone_fit.v_fermi_eva:.2f} eV A\nRMSE = {cone_fit.rmse_abs_k:.4f} 1/A",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
    )

    features = summary["replica_geometry"]["angle0_peak_locations"]
    x_vals = [-1, 0, 0, 1]
    e_vals = [
        features["replica_minus"]["peak_energy_ev"],
        features["crossing_E0_minusk"]["peak_energy_ev"],
        features["crossing_E0_plusk"]["peak_energy_ev"],
        features["replica_plus"]["peak_energy_ev"],
    ]
    k_vals = [
        features["replica_minus"]["peak_kx_a_inv"],
        features["crossing_E0_minusk"]["peak_kx_a_inv"],
        features["crossing_E0_plusk"]["peak_kx_a_inv"],
        features["replica_plus"]["peak_kx_a_inv"],
    ]
    labels = ["n=-1 apex", "E=0 crossing", "E=0 crossing", "n=+1 apex"]

    axes[1].scatter(k_vals, e_vals, s=60, color="#2f9e44")
    for x, y, label in zip(k_vals, e_vals, labels):
        axes[1].annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[1].axhline(photon_energy_ev, color="black", ls="--", lw=1)
    axes[1].axhline(-photon_energy_ev, color="black", ls="--", lw=1)
    axes[1].axhline(0.0, color="0.6", lw=1)
    axes[1].axvline(summary["replica_geometry"]["predicted_sideband_crossing_kx_a_inv"], color="black", ls="--", lw=1)
    axes[1].axvline(-summary["replica_geometry"]["predicted_sideband_crossing_kx_a_inv"], color="black", ls="--", lw=1)
    axes[1].set_xlabel(r"$k_x$ (1/Angstrom)")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].set_title("Observed Sideband Geometry, 0 deg")

    fig.savefig(REPORT_IMG_DIR / "figure_2_dirac_and_sidebands.png", dpi=300)
    plt.close(fig)


def plot_polarization_dependence(polarization_df: pd.DataFrame, summary: dict) -> None:
    theta_grid = np.linspace(0.0, np.pi, 400)
    fit = summary["polarization_fit"]
    fit_grid = harmonic4(theta_grid, fit["baseline_intensity"], fit["fourfold_amplitude"], fit["phase_radians"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].scatter(polarization_df["angle_degrees"], polarization_df["intensity"], s=55, color="#c2255c")
    axes[0].plot(np.degrees(theta_grid), fit_grid, color="#1c7ed6", lw=2)
    axes[0].set_xlabel("Pump polarization angle (deg)")
    axes[0].set_ylabel("Replica intensity (arb. units)")
    axes[0].set_title("Polarization Dependence of Replica Weight")
    axes[0].text(
        0.04,
        0.95,
        f"$I(\\theta)$ = $I_0 + A \\cos 4(\\theta-\\phi)$\n$R^2$ = {fit['r2']:.5f}\nA/$I_0$ = {100*fit['amplitude_fraction_of_baseline']:.2f}%",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
    )

    axes[1].bar(polarization_df["angle_degrees"], polarization_df["residual"], width=18, color="#495057")
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set_xlabel("Pump polarization angle (deg)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Fit Residuals")

    fig.savefig(REPORT_IMG_DIR / "figure_3_polarization_dependence.png", dpi=300)
    plt.close(fig)


def plot_angle_comparison(energy: np.ndarray, kx: np.ndarray, pump_off: np.ndarray, pump_on: dict[int, np.ndarray]) -> None:
    selected = [0, 60, 90]
    fig, axes = plt.subplots(1, len(selected), figsize=(14, 4.5), constrained_layout=True)
    extent = [kx.min(), kx.max(), energy.min(), energy.max()]

    vmax = max(np.percentile(np.abs(pump_on[a] - pump_off), 99.5) for a in selected)
    for ax, angle in zip(axes, selected):
        diff = pump_on[angle] - pump_off
        im = ax.imshow(
            diff,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Delta I, {angle} deg")
        ax.set_xlabel(r"$k_x$ (1/Angstrom)")
        ax.set_ylabel("Energy (eV)")

    fig.colorbar(im, ax=axes, shrink=0.85, label="Delta intensity")
    fig.savefig(REPORT_IMG_DIR / "figure_4_angle_difference_maps.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    data = load_data()
    cone_fit, branch_df = extract_equilibrium_cone(data["energy"], data["kx"], data["pump_off"])
    polarization_fit, polarization_df = analyze_polarization_scan(data["polarization_scan"])
    replica_df, replica_geometry = extract_replica_metrics(
        data["energy"], data["kx"], data["pump_off"], data["pump_on"], cone_fit
    )
    processed_comparison = compare_processed_metadata(data, cone_fit)

    angle_target_df = replica_df[replica_df["feature"] == "replica_plus"].copy()
    angle_target_df["scan_intensity"] = data["polarization_scan"]["intensity"].to_numpy()
    angle_target_df["scan_intensity_norm"] = angle_target_df["scan_intensity"] / angle_target_df["scan_intensity"].mean()
    angle_target_df["raw_diff_norm"] = angle_target_df["difference_intensity"] / angle_target_df["difference_intensity"].mean()
    raw_scan_correlation = float(np.corrcoef(angle_target_df["scan_intensity"], angle_target_df["difference_intensity"])[0, 1])

    summary = {
        "cone_fit": asdict(cone_fit),
        "polarization_fit": polarization_fit,
        "replica_geometry": replica_geometry,
        "processed_metadata_comparison": processed_comparison,
        "raw_vs_tabulated_polarization_correlation": raw_scan_correlation,
        "available_time_delays_fs": [float(x) for x in data["time_delays"]],
        "analysis_note": (
            "The HDF5 file provides equilibrium and polarization-resolved pump-on spectra plus a delay axis metadata vector, "
            "but not a full spectrum at every delay. The present analysis is therefore energy-momentum resolved and polarization resolved, "
            "with timing information limited to the available pump-on/off snapshots."
        ),
    }

    save_tables(branch_df, polarization_df, replica_df, summary)
    angle_target_df.to_csv(OUTPUT_DIR / "raw_vs_tabulated_polarization.csv", index=False)

    plot_overview(data["energy"], data["kx"], data["pump_off"], data["pump_on"][0], summary)
    plot_cone_and_sideband_geometry(branch_df, cone_fit, summary)
    plot_polarization_dependence(polarization_df, summary)
    plot_angle_comparison(data["energy"], data["kx"], data["pump_off"], data["pump_on"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
