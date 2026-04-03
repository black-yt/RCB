#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "DESI_EDE_Repro_Data.txt"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"


MODEL_LABELS = {
    "lcdm": r"$\Lambda$CDM",
    "ede": "EDE",
    "w0wa": r"$w_0w_a$",
}

MODEL_COLORS = {
    "lcdm": "#1f3b73",
    "ede": "#c13f32",
    "w0wa": "#2c8c5a",
}

MODEL_ORDER = ["lcdm", "ede", "w0wa"]

PARAM_LABELS = {
    "omega_m": r"$\Omega_m$",
    "H0": r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]",
    "sigma8": r"$\sigma_8$",
    "ns": r"$n_s$",
    "ombh2": r"$\omega_b$",
    "ln10As": r"$\ln(10^{10} A_s)$",
    "tau": r"$\tau$",
    "f_EDE": r"$f_{\rm EDE}$",
    "log10_ac": r"$\log_{10} a_c$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
    "S8": r"$S_8$",
    "rd_ratio_proxy": r"$r_d/r_{d,\Lambda{\rm CDM}}$",
}


def load_summary_data(path: Path) -> dict:
    namespace: dict = {}
    exec(path.read_text(), {}, namespace)
    return namespace


def build_parameter_table(data: dict) -> pd.DataFrame:
    rows = []
    for model in ("lcdm", "ede", "w0wa"):
        params = data[f"{model}_params"]
        for parameter, (mean, sigma) in params.items():
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "parameter": parameter,
                    "parameter_label": PARAM_LABELS.get(parameter, parameter),
                    "mean": float(mean),
                    "sigma": float(sigma),
                }
            )
    df = pd.DataFrame(rows)

    derived = []
    for model, group in df.groupby("model"):
        lookup = group.set_index("parameter")
        omega_m = lookup.loc["omega_m", "mean"]
        sigma8 = lookup.loc["sigma8", "mean"]
        omega_m_sigma = lookup.loc["omega_m", "sigma"]
        sigma8_sigma = lookup.loc["sigma8", "sigma"]
        s8 = sigma8 * math.sqrt(omega_m / 0.3)
        rel_sigma = math.sqrt((sigma8_sigma / sigma8) ** 2 + (0.5 * omega_m_sigma / omega_m) ** 2)
        derived.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "parameter": "S8",
                "parameter_label": PARAM_LABELS["S8"],
                "mean": s8,
                "sigma": s8 * rel_sigma,
            }
        )
    return pd.concat([df, pd.DataFrame(derived)], ignore_index=True)


def build_distance_tables(data: dict) -> pd.DataFrame:
    mapping = {
        "desi_dvrd_points": ("DESI", "Delta(D_V/r_d)"),
        "desi_fap_points": ("DESI", "Delta(F_AP)"),
        "sne_mu_points": ("Union3", "Delta(mu)"),
    }
    rows = []
    for key, (survey, observable) in mapping.items():
        for z, value, error in data[key]:
            rows.append(
                {
                    "series": key,
                    "survey": survey,
                    "observable": observable,
                    "z": float(z),
                    "value": float(value),
                    "error": float(error),
                }
            )
    return pd.DataFrame(rows)


def weighted_line_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict:
    w = 1.0 / sigma**2
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    sxx = np.sum(w * (x - xbar) ** 2)
    sxy = np.sum(w * (x - xbar) * (y - ybar))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    slope_sigma = math.sqrt(1.0 / sxx)
    intercept_sigma = math.sqrt(np.sum(w * x**2) / (np.sum(w) * sxx))
    chi2_zero = np.sum((y / sigma) ** 2)
    chi2_line = np.sum(((y - (intercept + slope * x)) / sigma) ** 2)
    return {
        "weighted_mean": float(ybar),
        "weighted_mean_sigma": float(math.sqrt(1.0 / np.sum(w))),
        "slope_per_redshift": float(slope),
        "slope_sigma": float(slope_sigma),
        "intercept": float(intercept),
        "intercept_sigma": float(intercept_sigma),
        "chi2_zero_model": float(chi2_zero),
        "chi2_linear_model": float(chi2_line),
        "n_points": int(len(x)),
    }


def compute_model_shifts(param_df: pd.DataFrame) -> pd.DataFrame:
    lcdm = (
        param_df[param_df["model"] == "lcdm"][["parameter", "mean", "sigma"]]
        .rename(columns={"mean": "mean_lcdm", "sigma": "sigma_lcdm"})
        .set_index("parameter")
    )
    rows = []
    for model in ("ede", "w0wa"):
        compare = param_df[param_df["model"] == model][["parameter", "mean", "sigma"]].set_index("parameter")
        common = lcdm.index.intersection(compare.index)
        for parameter in common:
            delta = compare.loc[parameter, "mean"] - lcdm.loc[parameter, "mean_lcdm"]
            sigma_combined = math.sqrt(compare.loc[parameter, "sigma"] ** 2 + lcdm.loc[parameter, "sigma_lcdm"] ** 2)
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "parameter": parameter,
                    "parameter_label": PARAM_LABELS.get(parameter, parameter),
                    "delta": float(delta),
                    "sigma_combined": float(sigma_combined),
                    "shift_sigma": float(delta / sigma_combined),
                }
            )
    return pd.DataFrame(rows)


def compute_summary_metrics(param_df: pd.DataFrame, distance_df: pd.DataFrame) -> dict:
    metrics: dict[str, object] = {}
    pivot = param_df.pivot(index="parameter", columns="model", values="mean")
    pivot_sigma = param_df.pivot(index="parameter", columns="model", values="sigma")

    h0_lcdm = pivot.loc["H0", "lcdm"]
    h0_lcdm_sigma = pivot_sigma.loc["H0", "lcdm"]
    rd_proxy = []
    for model in ("lcdm", "ede", "w0wa"):
        h0 = pivot.loc["H0", model]
        h0_sigma = pivot_sigma.loc["H0", model]
        ratio = h0_lcdm / h0
        ratio_sigma = ratio * math.sqrt((h0_lcdm_sigma / h0_lcdm) ** 2 + (h0_sigma / h0) ** 2)
        rd_proxy.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "rd_ratio_proxy": ratio,
                "rd_ratio_proxy_sigma": ratio_sigma,
            }
        )
    metrics["rd_ratio_proxy"] = rd_proxy

    metrics["ede_detection_sigma"] = float(
        pivot.loc["f_EDE", "ede"] / pivot_sigma.loc["f_EDE", "ede"]
    )
    metrics["w0wa_deviation_from_lcdm_limit_sigma"] = {
        "w0_vs_minus1": float((pivot.loc["w0", "w0wa"] - (-1.0)) / pivot_sigma.loc["w0", "w0wa"]),
        "wa_vs_0": float(pivot.loc["wa", "w0wa"] / pivot_sigma.loc["wa", "w0wa"]),
    }

    metrics["distance_series"] = {}
    for series, group in distance_df.groupby("series"):
        metrics["distance_series"][series] = weighted_line_fit(
            group["z"].to_numpy(),
            group["value"].to_numpy(),
            group["error"].to_numpy(),
        )
    return metrics


def save_tables(param_df: pd.DataFrame, shift_df: pd.DataFrame, distance_df: pd.DataFrame, metrics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    param_df.to_csv(OUTPUT_DIR / "parameter_constraints.csv", index=False)
    shift_df.to_csv(OUTPUT_DIR / "parameter_shifts_vs_lcdm.csv", index=False)
    distance_df.to_csv(OUTPUT_DIR / "distance_residual_points.csv", index=False)
    with (OUTPUT_DIR / "summary_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)


def plot_data_overview(distance_df: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=False)
    order = [
        ("desi_dvrd_points", r"DESI $\Delta(D_V/r_d)$"),
        ("desi_fap_points", r"DESI $\Delta F_{\rm AP}$"),
        ("sne_mu_points", r"Union3 $\Delta \mu$"),
    ]

    for ax, (series, title) in zip(axes, order):
        group = distance_df[distance_df["series"] == series].sort_values("z")
        ax.errorbar(
            group["z"],
            group["value"],
            yerr=group["error"],
            fmt="o",
            color="#202c59",
            ecolor="#4c5a8a",
            capsize=3,
            linewidth=1.2,
        )
        ax.axhline(0.0, color="#b23a48", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Redshift z")
        ax.set_ylabel("Residual")
    fig.suptitle("Reconstructed Distance Residual Data from Figure 6", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "data_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_constraints(param_df: pd.DataFrame) -> None:
    params = ["omega_m", "H0", "sigma8", "ns", "S8"]
    fig, axes = plt.subplots(len(params), 1, figsize=(8.5, 11), sharex=False)
    for ax, param in zip(axes, params):
        subset = param_df[param_df["parameter"] == param].copy()
        subset["model"] = pd.Categorical(subset["model"], categories=MODEL_ORDER, ordered=True)
        subset = subset.sort_values("model")
        subset["ypos"] = np.arange(len(subset))[::-1]
        for _, row in subset.iterrows():
            ax.errorbar(
                row["mean"],
                row["ypos"],
                xerr=row["sigma"],
                fmt="o",
                color=MODEL_COLORS[row["model"]],
                capsize=4,
                markersize=6,
                linewidth=1.5,
            )
        ax.set_yticks(subset["ypos"], subset["model_label"])
        ax.set_xlabel(PARAM_LABELS[param])
        ax.set_title(PARAM_LABELS[param], fontsize=12)
    fig.suptitle("Marginalized Parameter Constraints from the Reproduction File", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.985))
    fig.savefig(IMAGE_DIR / "parameter_constraints.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_specific_constraints(param_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ede_params = param_df[(param_df["model"] == "ede") & (param_df["parameter"].isin(["f_EDE", "log10_ac"]))].copy()
    ede_params["ypos"] = np.arange(len(ede_params))[::-1]
    for _, row in ede_params.iterrows():
        axes[0].errorbar(
            row["mean"],
            row["ypos"],
            xerr=row["sigma"],
            fmt="o",
            color=MODEL_COLORS["ede"],
            capsize=4,
            linewidth=1.5,
        )
    axes[0].set_yticks(ede_params["ypos"], ede_params["parameter_label"])
    axes[0].axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].set_title("EDE-Specific Parameters")

    w_params = param_df[(param_df["model"] == "w0wa") & (param_df["parameter"].isin(["w0", "wa"]))].copy()
    w_params["ypos"] = np.arange(len(w_params))[::-1]
    refs = {"w0": -1.0, "wa": 0.0}
    for _, row in w_params.iterrows():
        axes[1].errorbar(
            row["mean"],
            row["ypos"],
            xerr=row["sigma"],
            fmt="o",
            color=MODEL_COLORS["w0wa"],
            capsize=4,
            linewidth=1.5,
        )
        axes[1].axvline(refs[row["parameter"]], color="#666666", linestyle=":", linewidth=1.0)
    axes[1].set_yticks(w_params["ypos"], w_params["parameter_label"])
    axes[1].set_title(r"$w_0w_a$ Parameters Relative to $\Lambda$CDM Limits")

    fig.suptitle("Model-Specific Degrees of Freedom", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.95))
    fig.savefig(IMAGE_DIR / "model_specific_constraints.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shift_significance(shift_df: pd.DataFrame) -> None:
    params = ["H0", "omega_m", "sigma8", "ns", "ombh2", "ln10As", "tau", "S8"]
    subset = shift_df[shift_df["parameter"].isin(params)].copy()
    subset["parameter"] = pd.Categorical(subset["parameter"], categories=params[::-1], ordered=True)
    subset = subset.sort_values("parameter")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    offset = {"ede": 0.18, "w0wa": -0.18}
    ypos_base = np.arange(len(params))
    for model in ("ede", "w0wa"):
        part = subset[subset["model"] == model]
        ypos = ypos_base + offset[model]
        ax.barh(
            ypos,
            part["shift_sigma"],
            height=0.32,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.9,
        )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(2.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.axvline(-2.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_yticks(ypos_base, [PARAM_LABELS[p] for p in params[::-1]])
    ax.set_xlabel(r"Shift relative to $\Lambda$CDM [combined $\sigma$]")
    ax.set_title("How EDE and $w_0w_a$ Move Core Cosmological Parameters")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "standardized_shifts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_acoustic_proxy(metrics: dict, param_df: pd.DataFrame) -> None:
    rd_df = pd.DataFrame(metrics["rd_ratio_proxy"])
    h0_df = param_df[param_df["parameter"] == "H0"].copy()
    h0_df = h0_df.set_index("model").loc[["lcdm", "ede", "w0wa"]].reset_index()
    rd_df = rd_df.set_index("model").loc[["lcdm", "ede", "w0wa"]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xpos = np.arange(len(h0_df))

    axes[0].bar(xpos, h0_df["mean"], yerr=h0_df["sigma"], color=[MODEL_COLORS[m] for m in h0_df["model"]], capsize=4)
    axes[0].set_xticks(xpos, [MODEL_LABELS[m] for m in h0_df["model"]])
    axes[0].set_ylabel(PARAM_LABELS["H0"])
    axes[0].set_title(r"$H_0$ Constraint")

    axes[1].bar(
        xpos,
        rd_df["rd_ratio_proxy"],
        yerr=rd_df["rd_ratio_proxy_sigma"],
        color=[MODEL_COLORS[m] for m in rd_df["model"]],
        capsize=4,
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(xpos, [MODEL_LABELS[m] for m in rd_df["model"]])
    axes[1].set_ylabel(PARAM_LABELS["rd_ratio_proxy"])
    axes[1].set_title(r"Inferred $r_d$ Shift if BAO Mainly Fixes $H_0 r_d$")

    fig.suptitle("Acoustic-Scale Interpretation of the Summary Constraints", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "acoustic_scale_proxy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    raw_data = load_summary_data(DATA_FILE)
    param_df = build_parameter_table(raw_data)
    distance_df = build_distance_tables(raw_data)
    shift_df = compute_model_shifts(param_df)
    metrics = compute_summary_metrics(param_df, distance_df)

    save_tables(param_df, shift_df, distance_df, metrics)
    plot_data_overview(distance_df)
    plot_parameter_constraints(param_df)
    plot_model_specific_constraints(param_df)
    plot_shift_significance(shift_df)
    plot_acoustic_proxy(metrics, param_df)


if __name__ == "__main__":
    main()
