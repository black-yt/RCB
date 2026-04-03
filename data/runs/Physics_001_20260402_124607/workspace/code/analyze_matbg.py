#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_matbg")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "MATBG Superfluid Stiffness Core Dataset.txt"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "report" / "images"


@dataclass
class FitResult:
    exponent: float
    scale: float
    rmse: float
    r2: float


def parse_dataset(path: Path) -> Dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    arrays: Dict[str, np.ndarray] = {}
    current_name = None
    collecting = False
    buffer = []

    for line in lines:
        if line.startswith("**") and line.endswith(":**"):
            current_name = line.strip("*:")
            collecting = False
            buffer = []
            continue

        if current_name and line.strip().startswith("["):
            collecting = True
            buffer = [line]
            if line.strip().endswith("]"):
                arrays[current_name] = np.fromstring(" ".join(buffer).strip("[]"), sep=" ")
                collecting = False
            continue

        if collecting:
            buffer.append(line)
            if line.strip().endswith("]"):
                arrays[current_name] = np.fromstring(" ".join(buffer).strip("[]"), sep=" ")
                collecting = False

    return arrays


def get_array(arrays: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    for key, value in arrays.items():
        if key.startswith(prefix):
            return value
    raise KeyError(prefix)


def expected_lengths() -> Dict[str, int]:
    return {
        "Carrier Density Data": 50,
        "Conventional Superfluid Stiffness": 50,
        "Quantum Geometric Superfluid Stiffness": 50,
        "Experimental Superfluid Stiffness Hole": 50,
        "Experimental Superfluid Stiffness Electron": 50,
        "Temperature Array": 100,
        "BCS Model Data": 100,
        "Nodal Superconductor Data": 100,
        "Power Law n=2.0 Data": 100,
        "Power Law n=2.5 Data": 100,
        "Power Law n=3.0 Data": 100,
        "Experimental Data with Noise": 100,
        "DC Current Array": 50,
        "Ginzburg-Landau Model": 50,
        "Linear Meissner Model": 50,
        "Experimental DC Data": 50,
        "Microwave Power Array": 50,
        "Microwave Current Amplitude": 50,
        "Experimental Microwave Data": 50,
    }


def integrity_report(arrays: Dict[str, np.ndarray]) -> Dict[str, dict]:
    report = {}
    for prefix, expected in expected_lengths().items():
        arr = get_array(arrays, prefix)
        report[prefix] = {
            "actual_length": int(len(arr)),
            "expected_length": int(expected),
            "status": "ok" if len(arr) == expected else "mismatch",
        }
    return report


def normalized_frequency(ds: np.ndarray) -> np.ndarray:
    return np.sqrt(ds / np.max(ds))


def normalized_resistance_proxy(ds: np.ndarray) -> np.ndarray:
    return 1.0 - ds / np.max(ds)


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return coeffs, r2


def fit_power_drop(x: np.ndarray, y: np.ndarray, exponents: Iterable[float]) -> Tuple[FitResult, Dict[str, FitResult]]:
    d0 = float(y[0])
    candidates: Dict[str, FitResult] = {}
    best = None
    best_key = None

    for n in exponents:
        xn = x**n
        scale = float(np.sum((d0 - y) * xn) / np.sum(xn**2))
        pred = d0 - scale * xn
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        fit = FitResult(exponent=float(n), scale=scale, rmse=rmse, r2=r2)
        key = f"{n:g}"
        candidates[key] = fit
        if best is None or rmse < best.rmse:
            best = fit
            best_key = key

    assert best is not None and best_key is not None
    return best, candidates


def fit_generalized_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    d0 = float(y[0])
    n_grid = np.linspace(0.5, 4.0, 351)
    tc_grid = np.linspace(1.0, 4.0, 301)
    xx = x[None, None, :]
    nn = n_grid[:, None, None]
    tt = tc_grid[None, :, None]
    pred = d0 * np.maximum(0.0, 1.0 - (xx / tt) ** nn)
    err = np.mean((pred - y[None, None, :]) ** 2, axis=2)
    idx = np.unravel_index(np.argmin(err), err.shape)
    n_best = float(n_grid[idx[0]])
    tc_best = float(tc_grid[idx[1]])
    rmse = float(math.sqrt(err[idx]))
    pred_best = d0 * np.maximum(0.0, 1.0 - (x / tc_best) ** n_best)
    ss_res = float(np.sum((y - pred_best) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "d0": d0,
        "n_best": n_best,
        "t_star_best": tc_best,
        "rmse": rmse,
        "r2": r2,
    }


def regenerate_temperature_models(t: np.ndarray, d0: float = 100.0, tc: float = 1.0) -> Dict[str, np.ndarray]:
    ratio = np.clip(t / tc, 0.0, None)
    return {
        "BCS": d0 * np.maximum(0.0, 1.0 - ratio**2.0),
        "Nodal": d0 * np.maximum(0.0, 1.0 - ratio),
        "n=2.5": d0 * np.maximum(0.0, 1.0 - ratio**2.5),
        "n=3": d0 * np.maximum(0.0, 1.0 - ratio**3.0),
    }


def regenerate_current_models(i_dc: np.ndarray, d0: float = 100.0, ic: float = 50.0) -> Dict[str, np.ndarray]:
    ratio = i_dc / ic
    return {
        "GL": d0 * np.maximum(0.0, 1.0 - ratio**2.0),
        "Linear": d0 * np.maximum(0.0, 1.0 - ratio),
    }


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.frameon": True,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES / name, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def make_figures(metrics: dict, arrays: Dict[str, np.ndarray]) -> None:
    n_eff = get_array(arrays, "Carrier Density Data")
    d_conv = get_array(arrays, "Conventional Superfluid Stiffness")
    d_geom = get_array(arrays, "Quantum Geometric Superfluid Stiffness")
    d_hole = get_array(arrays, "Experimental Superfluid Stiffness Hole")
    d_elec = get_array(arrays, "Experimental Superfluid Stiffness Electron")
    d_mean = 0.5 * (d_hole + d_elec)

    t_model = get_array(arrays, "Temperature Array")
    temp_models = regenerate_temperature_models(t_model)
    d_temp_exp = get_array(arrays, "Experimental Data with Noise")
    t_exp = np.linspace(float(t_model.min()), float(t_model.max()), len(d_temp_exp))
    best_temp = metrics["temperature"]["best_drop_fit"]
    pred_temp = d_temp_exp[0] - best_temp["scale"] * t_exp ** best_temp["exponent"]

    i_dc = get_array(arrays, "DC Current Array")
    current_models = regenerate_current_models(i_dc)
    i_mw = get_array(arrays, "Microwave Current Amplitude")
    d_mw = get_array(arrays, "Experimental Microwave Data")
    alpha = metrics["current"]["quadratic_fit"]["alpha_zero_intercept"]
    pred_mw = d_mw[0] - alpha * i_mw**2

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = list(metrics["integrity"].keys())
    actual = [metrics["integrity"][label]["actual_length"] for label in labels]
    expected = [metrics["integrity"][label]["expected_length"] for label in labels]
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, expected, width=0.4, label="Expected", color="#c9d5e8")
    ax.bar(xpos + 0.2, actual, width=0.4, label="Observed", color="#315c7c")
    ax.set_xticks(xpos)
    ax.set_xticklabels([label.replace(" Data", "").replace("Experimental ", "Exp ") for label in labels], rotation=80, ha="right")
    ax.set_ylabel("Array length")
    ax.set_title("Dataset Integrity Check")
    ax.legend()
    save_figure(fig, "figure_data_integrity.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(n_eff / 1e15, d_conv / 1e9, label="Conventional FL theory", lw=2, color="#9a6324")
    ax.plot(n_eff / 1e15, d_geom / 1e9, label="Quantum geometric theory", lw=2, color="#3b82f6")
    ax.plot(n_eff / 1e15, d_hole / 1e9, label="Experimental hole branch", lw=2, color="#d97706")
    ax.plot(n_eff / 1e15, d_elec / 1e9, label="Experimental electron branch", lw=2, color="#15803d")
    ax.set_xlabel(r"Carrier density $n_{\mathrm{eff}}$ ($10^{15}$ m$^{-2}$)")
    ax.set_ylabel(r"Superfluid stiffness $D_s$ ($10^9$ arb. units)")
    ax.set_title("Carrier-Density Dependence of Superfluid Stiffness")
    ax.legend(ncol=2)
    save_figure(fig, "figure_density_stiffness.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].plot(n_eff / 1e15, normalized_frequency(d_mean), color="#7c3aed", lw=2.4)
    axes[0].set_xlabel(r"Carrier density $n_{\mathrm{eff}}$ ($10^{15}$ m$^{-2}$)")
    axes[0].set_ylabel(r"Normalized $f_\mathrm{res}$")
    axes[0].set_title("Derived Resonance Frequency Proxy")
    axes[1].plot(n_eff / 1e15, normalized_resistance_proxy(d_mean), color="#b91c1c", lw=2.4)
    axes[1].set_xlabel(r"Carrier density $n_{\mathrm{eff}}$ ($10^{15}$ m$^{-2}$)")
    axes[1].set_ylabel(r"Normalized $R_\mathrm{dc}$ proxy")
    axes[1].set_title("Derived DC Resistance Proxy")
    save_figure(fig, "figure_density_derived_observables.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for label, curve in temp_models.items():
        ax.plot(t_model, curve, lw=1.8, label=label)
    ax.scatter(t_exp, d_temp_exp, s=12, color="#111827", alpha=0.75, label="Experimental noisy series")
    ax.plot(t_exp, pred_temp, color="#dc2626", lw=2.2, label=f"Best fit: D0 - A T^{best_temp['exponent']:.2f}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Superfluid stiffness (arb. units)")
    ax.set_title("Temperature Dependence and Power-Law Fits")
    ax.legend(ncol=2)
    save_figure(fig, "figure_temperature_dependence.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(i_dc, current_models["GL"], lw=2.0, label="GL depairing model", color="#1d4ed8")
    ax.plot(i_dc, current_models["Linear"], lw=2.0, label="Linear Meissner model", color="#94a3b8")
    ax.scatter(i_mw, d_mw, s=16, color="#047857", label="Experimental microwave response")
    ax.plot(i_mw, pred_mw, lw=2.2, color="#ef4444", label=r"Quadratic fit $D_s=D_0-\alpha I^2$")
    ax.set_xlabel("Current amplitude (nA)")
    ax.set_ylabel("Superfluid stiffness (arb. units)")
    ax.set_title("Current-Induced Suppression of Superfluid Stiffness")
    ax.legend()
    save_figure(fig, "figure_current_dependence.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].plot(i_mw, normalized_frequency(d_mw), color="#0f766e", lw=2.4)
    axes[0].set_xlabel("Microwave current amplitude (nA)")
    axes[0].set_ylabel(r"Normalized $f_\mathrm{res}$")
    axes[0].set_title("Current-Tuned Resonance Proxy")
    axes[1].plot(i_mw, normalized_resistance_proxy(d_mw), color="#991b1b", lw=2.4)
    axes[1].set_xlabel("Microwave current amplitude (nA)")
    axes[1].set_ylabel(r"Normalized $R_\mathrm{dc}$ proxy")
    axes[1].set_title("Current-Tuned Dissipation Proxy")
    save_figure(fig, "figure_current_derived_observables.png")


def main() -> None:
    ensure_dirs()
    set_plot_style()

    arrays = parse_dataset(DATA_PATH)
    integrity = integrity_report(arrays)

    n_eff = get_array(arrays, "Carrier Density Data")
    d_conv = get_array(arrays, "Conventional Superfluid Stiffness")
    d_geom = get_array(arrays, "Quantum Geometric Superfluid Stiffness")
    d_hole = get_array(arrays, "Experimental Superfluid Stiffness Hole")
    d_elec = get_array(arrays, "Experimental Superfluid Stiffness Electron")
    d_mean = 0.5 * (d_hole + d_elec)

    density_line, density_r2 = fit_linear(n_eff, d_mean)
    hole_electron_asym = np.abs(d_hole - d_elec) / d_mean

    t_model = get_array(arrays, "Temperature Array")
    d_temp_exp = get_array(arrays, "Experimental Data with Noise")
    t_exp = np.linspace(float(t_model.min()), float(t_model.max()), len(d_temp_exp))
    best_drop_fit, candidate_fits = fit_power_drop(t_exp, d_temp_exp, [1.0, 2.0, 2.5, 3.0])
    generalized_temp_fit = fit_generalized_power_law(t_exp, d_temp_exp)

    i_mw = get_array(arrays, "Microwave Current Amplitude")
    d_mw = get_array(arrays, "Experimental Microwave Data")
    delta = d_mw[0] - d_mw
    alpha_zero = float(np.sum(delta * i_mw**2) / np.sum(i_mw**4))
    pred_zero = d_mw[0] - alpha_zero * i_mw**2
    ss_res = float(np.sum((d_mw - pred_zero) ** 2))
    ss_tot = float(np.sum((d_mw - np.mean(d_mw)) ** 2))
    r2_zero = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    i_quad = math.sqrt(float(d_mw[0]) / alpha_zero)

    metrics = {
        "integrity": integrity,
        "density": {
            "exp_over_conventional_mean": float(np.mean(d_mean / d_conv)),
            "exp_over_conventional_min": float(np.min(d_mean / d_conv)),
            "exp_over_conventional_max": float(np.max(d_mean / d_conv)),
            "exp_over_geometric_mean": float(np.mean(d_mean / d_geom)),
            "exp_over_geometric_min": float(np.min(d_mean / d_geom)),
            "exp_over_geometric_max": float(np.max(d_mean / d_geom)),
            "hole_electron_asymmetry_mean": float(np.mean(hole_electron_asym)),
            "hole_electron_asymmetry_max": float(np.max(hole_electron_asym)),
            "linear_slope": float(density_line[0]),
            "linear_intercept": float(density_line[1]),
            "linear_r2": float(density_r2),
        },
        "temperature": {
            "best_drop_fit": {
                "exponent": float(best_drop_fit.exponent),
                "scale": float(best_drop_fit.scale),
                "rmse": float(best_drop_fit.rmse),
                "r2": float(best_drop_fit.r2),
            },
            "candidate_fits": {
                key: {
                    "exponent": value.exponent,
                    "scale": value.scale,
                    "rmse": value.rmse,
                    "r2": value.r2,
                }
                for key, value in candidate_fits.items()
            },
            "generalized_power_law": generalized_temp_fit,
        },
        "current": {
            "quadratic_fit": {
                "alpha_zero_intercept": alpha_zero,
                "quadratic_current_scale_nA": i_quad,
                "r2": r2_zero,
            }
        },
        "derived_observables": {
            "density_frequency_proxy_range": [
                float(np.min(normalized_frequency(d_mean))),
                float(np.max(normalized_frequency(d_mean))),
            ],
            "density_resistance_proxy_range": [
                float(np.min(normalized_resistance_proxy(d_mean))),
                float(np.max(normalized_resistance_proxy(d_mean))),
            ],
            "current_frequency_proxy_range": [
                float(np.min(normalized_frequency(d_mw))),
                float(np.max(normalized_frequency(d_mw))),
            ],
            "current_resistance_proxy_range": [
                float(np.min(normalized_resistance_proxy(d_mw))),
                float(np.max(normalized_resistance_proxy(d_mw))),
            ],
        },
    }

    write_json(OUTPUTS / "analysis_metrics.json", metrics)
    write_json(OUTPUTS / "data_integrity.json", integrity)

    make_figures(metrics, arrays)


if __name__ == "__main__":
    main()
