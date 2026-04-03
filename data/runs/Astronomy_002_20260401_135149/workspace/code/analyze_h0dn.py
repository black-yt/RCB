from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "H0DN_MinimalDataset.txt"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"
MPL_DIR = OUTPUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(exist_ok=True)
    MPL_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> dict:
    namespace: dict = {}
    exec(path.read_text(), {}, namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


@dataclass
class PrimaryFit:
    measurements: pd.DataFrame
    hosts: List[str]
    host_mean: pd.Series
    covariance: pd.DataFrame
    correlation: pd.DataFrame


@dataclass
class LadderFit:
    name: str
    primary_name: str
    h0: float
    h0_err_low: float
    h0_err_high: float
    m_abs: float
    m_abs_err: float
    sigma_cal: float
    sigma_flow: float
    n_calibrators: int
    n_flow: int
    n_host_constraints: int
    nll: float
    calibrators: pd.DataFrame
    flow: pd.DataFrame


def build_measurement_frame(dataset: dict) -> pd.DataFrame:
    rows = []
    for host, method, anchor, mu_meas, err_meas in dataset["host_measurements"]:
        rows.append(
            {
                "host": host,
                "method": method,
                "anchor": anchor,
                "mu_meas": mu_meas,
                "err_meas": err_meas,
                "anchor_err": dataset["anchors"][anchor]["err"],
                "method_anchor_err": dataset["method_anchor_err"].get((method, anchor), 0.0),
            }
        )
    return pd.DataFrame(rows)


def primary_gls(
    dataset: dict,
    allowed_methods: Optional[Sequence[str]] = None,
    allowed_anchors: Optional[Sequence[str]] = None,
) -> PrimaryFit:
    frame = build_measurement_frame(dataset)
    if allowed_methods is not None:
        frame = frame[frame["method"].isin(allowed_methods)].copy()
    if allowed_anchors is not None:
        frame = frame[frame["anchor"].isin(allowed_anchors)].copy()
    if frame.empty:
        raise ValueError("No primary-indicator measurements remain after filtering.")

    hosts = sorted(frame["host"].unique())
    host_index = {host: i for i, host in enumerate(hosts)}

    n_obs = len(frame)
    x = np.zeros((n_obs, len(hosts)))
    y = frame["mu_meas"].to_numpy()
    cov = np.zeros((n_obs, n_obs))

    for i, row in frame.iterrows():
        x[frame.index.get_loc(i), host_index[row["host"]]] = 1.0
        cov[frame.index.get_loc(i), frame.index.get_loc(i)] += row["err_meas"] ** 2

    rows = frame.reset_index(drop=True)
    for i, row_i in rows.iterrows():
        for j, row_j in rows.iterrows():
            if row_i["anchor"] == row_j["anchor"]:
                cov[i, j] += row_i["anchor_err"] ** 2
            if (row_i["method"] == row_j["method"]) and (row_i["anchor"] == row_j["anchor"]):
                cov[i, j] += row_i["method_anchor_err"] ** 2

    inv_cov = np.linalg.inv(cov)
    host_cov = np.linalg.inv(x.T @ inv_cov @ x)
    host_mean = host_cov @ (x.T @ inv_cov @ y)

    host_mean_s = pd.Series(host_mean, index=hosts, name="mu_host")
    host_cov_df = pd.DataFrame(host_cov, index=hosts, columns=hosts)
    host_corr_df = host_cov_df.divide(np.sqrt(np.diag(host_cov)), axis=0).divide(np.sqrt(np.diag(host_cov)), axis=1)

    frame = frame.copy()
    frame["posterior_mu_host"] = frame["host"].map(host_mean_s)
    frame["posterior_host_err"] = frame["host"].map(pd.Series(np.sqrt(np.diag(host_cov)), index=hosts))

    return PrimaryFit(
        measurements=frame,
        hosts=hosts,
        host_mean=host_mean_s,
        covariance=host_cov_df,
        correlation=host_corr_df,
    )


def mu_cosmographic(z: np.ndarray, h0: float, c_km: float, q0: float = -0.55) -> np.ndarray:
    d_l_mpc = (c_km / h0) * z * (1.0 + 0.5 * (1.0 - q0) * z)
    return 5.0 * np.log10(d_l_mpc) + 25.0


def profile_h0_interval(
    objective,
    log_sigma_cal: float,
    log_sigma_flow: float,
    grid_min: float = 40.0,
    grid_max: float = 140.0,
    grid_n: int = 2001,
) -> Tuple[float, float, float]:
    grid = np.linspace(grid_min, grid_max, grid_n)
    vals = np.array([objective(np.log(h0), log_sigma_cal, log_sigma_flow) for h0 in grid])
    best_idx = int(np.argmin(vals))
    best_h0 = float(grid[best_idx])
    mask = vals <= vals.min() + 0.5
    return best_h0, best_h0 - float(grid[mask][0]), float(grid[mask][-1]) - best_h0


def fit_sn_ladder(
    dataset: dict,
    primary_fit: PrimaryFit,
    name: str,
    exclude_hosts: Optional[Iterable[str]] = None,
) -> LadderFit:
    exclude_hosts = set(exclude_hosts or [])
    c_km = dataset["c_km"]

    calibrators = []
    for host, m_b, err_m_b in dataset["sneia_calibrators"]:
        if host in exclude_hosts or host not in primary_fit.host_mean.index:
            continue
        calibrators.append((host, m_b, err_m_b))
    if len(calibrators) < 2:
        raise ValueError(f"Variant {name} has too few SN calibrators to fit.")

    cal_df = pd.DataFrame(calibrators, columns=["host", "m_b", "err_m_b"])
    cal_df["mu_host"] = cal_df["host"].map(primary_fit.host_mean)
    host_index = [primary_fit.hosts.index(host) for host in cal_df["host"]]
    host_cov = primary_fit.covariance.to_numpy()[np.ix_(host_index, host_index)]
    cov_cal = host_cov + np.diag(cal_df["err_m_b"].to_numpy() ** 2)

    flow_df = pd.DataFrame(
        dataset["hubble_flow_sneia"],
        columns=["z", "m_b", "err_m_b", "peculiar_velocity_kms"],
    )
    pv_mag = (5.0 / np.log(10.0)) * flow_df["peculiar_velocity_kms"].to_numpy() / (c_km * flow_df["z"].to_numpy())
    flow_base_var = flow_df["err_m_b"].to_numpy() ** 2 + pv_mag**2

    y_cal = cal_df["m_b"].to_numpy() - cal_df["mu_host"].to_numpy()
    z_flow = flow_df["z"].to_numpy()
    y_flow_obs = flow_df["m_b"].to_numpy()

    def objective(log_h0: float, log_sigma_cal: float, log_sigma_flow: float) -> float:
        h0 = math.exp(log_h0)
        sigma_cal = math.exp(log_sigma_cal)
        sigma_flow = math.exp(log_sigma_flow)

        flow_cov = np.diag(flow_base_var + sigma_flow**2)
        total_cov = np.block(
            [
                [cov_cal + np.eye(len(cal_df)) * sigma_cal**2, np.zeros((len(cal_df), len(flow_df)))],
                [np.zeros((len(flow_df), len(cal_df))), flow_cov],
            ]
        )
        y_flow = y_flow_obs - mu_cosmographic(z_flow, h0=h0, c_km=c_km)
        y_all = np.concatenate([y_cal, y_flow])

        inv_cov = np.linalg.inv(total_cov)
        one = np.ones(len(y_all))
        m_abs = float((one @ inv_cov @ y_all) / (one @ inv_cov @ one))
        resid = y_all - m_abs * one
        _, logdet = np.linalg.slogdet(total_cov)
        return 0.5 * (resid @ inv_cov @ resid + logdet + np.log(one @ inv_cov @ one))

    result = minimize(
        lambda theta: objective(*theta),
        x0=np.array([np.log(100.0), np.log(0.15), np.log(0.15)]),
        bounds=[
            (np.log(40.0), np.log(140.0)),
            (np.log(1e-4), np.log(1.0)),
            (np.log(1e-4), np.log(1.0)),
        ],
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed for variant {name}: {result.message}")

    log_h0, log_sigma_cal, log_sigma_flow = result.x
    h0 = math.exp(log_h0)
    sigma_cal = math.exp(log_sigma_cal)
    sigma_flow = math.exp(log_sigma_flow)

    flow_cov = np.diag(flow_base_var + sigma_flow**2)
    total_cov = np.block(
        [
            [cov_cal + np.eye(len(cal_df)) * sigma_cal**2, np.zeros((len(cal_df), len(flow_df)))],
            [np.zeros((len(flow_df), len(cal_df))), flow_cov],
        ]
    )
    y_flow = y_flow_obs - mu_cosmographic(z_flow, h0=h0, c_km=c_km)
    y_all = np.concatenate([y_cal, y_flow])
    inv_cov = np.linalg.inv(total_cov)
    one = np.ones(len(y_all))
    m_abs = float((one @ inv_cov @ y_all) / (one @ inv_cov @ one))
    m_abs_err = float(math.sqrt(1.0 / (one @ inv_cov @ one)))
    resid_all = y_all - m_abs * one

    best_h0, h0_err_low, h0_err_high = profile_h0_interval(objective, log_sigma_cal, log_sigma_flow)

    cal_df = cal_df.copy()
    cal_df["M_b"] = cal_df["m_b"] - cal_df["mu_host"]
    cal_df["baseline_M_b"] = m_abs
    cal_df["residual_mag"] = cal_df["M_b"] - m_abs
    cal_df["total_sigma_mag"] = np.sqrt(np.diag(cov_cal + np.eye(len(cal_df)) * sigma_cal**2))

    flow_df = flow_df.copy()
    flow_df["mu_model"] = mu_cosmographic(z_flow, h0=h0, c_km=c_km)
    flow_df["mu_inferred"] = flow_df["m_b"] - m_abs
    flow_df["residual_mag"] = flow_df["mu_inferred"] - flow_df["mu_model"]
    flow_df["peculiar_velocity_mag"] = pv_mag
    flow_df["total_sigma_mag"] = np.sqrt(flow_base_var + sigma_flow**2)

    return LadderFit(
        name=name,
        primary_name=primary_fit.measurements["method"].sort_values().unique()[0]
        if primary_fit.measurements["method"].nunique() == 1
        else "mixed",
        h0=best_h0,
        h0_err_low=h0_err_low,
        h0_err_high=h0_err_high,
        m_abs=m_abs,
        m_abs_err=m_abs_err,
        sigma_cal=sigma_cal,
        sigma_flow=sigma_flow,
        n_calibrators=len(cal_df),
        n_flow=len(flow_df),
        n_host_constraints=len(primary_fit.hosts),
        nll=float(objective(log_h0, log_sigma_cal, log_sigma_flow)),
        calibrators=cal_df,
        flow=flow_df,
    )


def fit_sbf_rank_diagnostic(dataset: dict) -> Dict[str, float]:
    hosts = [row[0] for row in dataset["sbf_calibrators"]]
    groups = sorted(set(dataset["host_group"][host] for host in hosts))
    params = ["M_SBF"] + [f"mu_{group}" for group in groups]
    design = np.zeros((len(hosts), len(params)))
    for i, host in enumerate(hosts):
        design[i, 0] = 1.0
        design[i, 1 + groups.index(dataset["host_group"][host])] = 1.0
    singular_values = np.linalg.svd(design, compute_uv=False)
    return {
        "n_sbf_calibrators": len(hosts),
        "n_group_parameters": len(groups),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "n_parameters": design.shape[1],
        "min_singular_value": float(singular_values.min()),
        "max_singular_value": float(singular_values.max()),
    }


def build_variant_suite(dataset: dict) -> Tuple[Dict[str, PrimaryFit], List[LadderFit]]:
    primary_variants = {
        "all_primary": primary_gls(dataset),
        "cepheid_only": primary_gls(dataset, allowed_methods=["Cepheid"]),
        "trgb_only": primary_gls(dataset, allowed_methods=["TRGB"]),
        "n4258_only": primary_gls(dataset, allowed_anchors=["N4258"]),
        "lmc_only": primary_gls(dataset, allowed_anchors=["LMC"]),
    }

    fits = [
        fit_sn_ladder(dataset, primary_variants["all_primary"], "baseline_all"),
        fit_sn_ladder(dataset, primary_variants["cepheid_only"], "cepheid_only"),
        fit_sn_ladder(dataset, primary_variants["trgb_only"], "trgb_only"),
        fit_sn_ladder(dataset, primary_variants["n4258_only"], "n4258_only"),
        fit_sn_ladder(dataset, primary_variants["lmc_only"], "lmc_only"),
        fit_sn_ladder(dataset, primary_variants["all_primary"], "exclude_ngc1309", exclude_hosts=["NGC1309"]),
    ]

    baseline_hosts = [host for host, _, _ in dataset["sneia_calibrators"]]
    for host in baseline_hosts:
        fits.append(
            fit_sn_ladder(
                dataset,
                primary_variants["all_primary"],
                f"jackknife_{host}",
                exclude_hosts=[host],
            )
        )
    return primary_variants, fits


def make_data_overview_figure(dataset: dict, primary_fits: Dict[str, PrimaryFit]) -> None:
    counts = pd.DataFrame(
        [
            ("Anchors", "Geometric anchors", len(dataset["anchors"])),
            ("Primary", "Host measurements", len(dataset["host_measurements"])),
            ("Secondary", "SN calibrators", len(dataset["sneia_calibrators"])),
            ("Secondary", "SBF calibrators", len(dataset["sbf_calibrators"])),
            ("Hubble flow", "SN flow", len(dataset["hubble_flow_sneia"])),
            ("Hubble flow", "SBF flow", len(dataset["hubble_flow_sbf"])),
        ],
        columns=["rung", "component", "count"],
    )
    method_counts = primary_fits["all_primary"].measurements.groupby(["method", "anchor"]).size().reset_index(name="count")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=counts, x="count", y="component", hue="rung", dodge=False, ax=axes[0], palette="crest")
    axes[0].set_title("Minimal Dataset Inventory")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")
    axes[0].legend(loc="lower right", frameon=True)

    method_counts["label"] = method_counts["method"] + " / " + method_counts["anchor"]
    sns.barplot(data=method_counts, x="count", y="label", ax=axes[1], hue="label", palette="mako", legend=False)
    axes[1].set_title("Primary Host Constraints")
    axes[1].set_xlabel("Measurements")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_primary_host_figure(primary_fit: PrimaryFit) -> None:
    hosts = primary_fit.hosts
    y_positions = {host: i for i, host in enumerate(hosts)}
    fig, ax = plt.subplots(figsize=(10, 5.5))

    palette = {"Cepheid": "#1f77b4", "TRGB": "#d62728"}
    markers = {"N4258": "o", "LMC": "s", "MW": "^"}
    for _, row in primary_fit.measurements.iterrows():
        ax.errorbar(
            row["mu_meas"],
            y_positions[row["host"]],
            xerr=row["err_meas"],
            fmt=markers.get(row["anchor"], "o"),
            color=palette.get(row["method"], "#444444"),
            alpha=0.85,
            ms=6,
            capsize=3,
        )

    for host in hosts:
        mu = primary_fit.host_mean[host]
        err = math.sqrt(primary_fit.covariance.loc[host, host])
        ypos = y_positions[host]
        ax.axvline(mu, ymin=(ypos - 0.35) / max(len(hosts) - 1, 1), ymax=(ypos + 0.35) / max(len(hosts) - 1, 1), color="#2ca02c", lw=2)
        ax.fill_betweenx([ypos - 0.2, ypos + 0.2], mu - err, mu + err, color="#2ca02c", alpha=0.18)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#1f77b4", label="Cepheid", markersize=7),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#d62728", label="TRGB", markersize=7),
        plt.Line2D([0], [0], color="#2ca02c", lw=2, label="GLS host posterior"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    ax.set_yticks(range(len(hosts)))
    ax.set_yticklabels(hosts)
    ax.set_xlabel(r"Distance modulus $\mu$ (mag)")
    ax.set_title("Primary-Indicator Host Distances")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "primary_host_distances.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_covariance_heatmap(primary_fit: PrimaryFit) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(primary_fit.correlation, annot=True, fmt=".2f", cmap="vlag", center=0.0, square=True, cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_title("Posterior Correlation of Host Distances")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "primary_covariance_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_calibrator_figure(fit: LadderFit) -> None:
    df = fit.calibrators.sort_values("M_b").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(
        df["M_b"],
        np.arange(len(df)),
        xerr=df["total_sigma_mag"],
        fmt="o",
        color="#0f766e",
        capsize=3,
    )
    ax.axvline(fit.m_abs, color="#b91c1c", lw=2, label=rf"Shared $M_B={fit.m_abs:.3f}$")
    ax.fill_betweenx(
        [-0.5, len(df) - 0.5],
        fit.m_abs - fit.m_abs_err,
        fit.m_abs + fit.m_abs_err,
        color="#b91c1c",
        alpha=0.18,
    )
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["host"])
    ax.set_xlabel(r"Calibrated SN absolute magnitude proxy $M_B$ (mag)")
    ax.set_title("SN Ia Calibrator Absolute-Magnitude Scatter")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "calibrator_absolute_magnitude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_hubble_diagram(fit: LadderFit) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    z_grid = np.linspace(0.02, 0.09, 300)
    ax.errorbar(
        fit.flow["z"],
        fit.flow["mu_inferred"],
        yerr=fit.flow["total_sigma_mag"],
        fmt="o",
        color="#1d4ed8",
        capsize=3,
        label="Hubble-flow SNe Ia",
    )
    ax.plot(z_grid, mu_cosmographic(z_grid, fit.h0, c_km=299792.458), color="#ea580c", lw=2.5, label=rf"Best fit $H_0={fit.h0:.1f}$")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"Inferred distance modulus $\mu$ (mag)")
    ax.set_title("Low-z Hubble Diagram from the Minimal SN Ladder")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "hubble_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_variant_forest(fits: List[LadderFit]) -> None:
    plot_rows = []
    for fit in fits:
        if fit.name.startswith("jackknife_"):
            continue
        plot_rows.append((fit.name, fit.h0, fit.h0_err_low, fit.h0_err_high, "Variant"))
    plot_rows.append(("Published target", 73.50, 0.81, 0.81, "Reference"))
    plot_rows.append(("Early-Universe reference", 67.4, 0.5, 0.5, "Reference"))
    df = pd.DataFrame(plot_rows, columns=["label", "h0", "err_low", "err_high", "kind"]).sort_values("h0")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(df))
    colors = df["kind"].map({"Variant": "#111827", "Reference": "#7c3aed"}).tolist()
    for yi, row, color in zip(y, df.itertuples(index=False), colors):
        ax.errorbar(
            row.h0,
            yi,
            xerr=np.array([[row.err_low], [row.err_high]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
            markersize=5,
        )
    ax.axvline(73.50, color="#7c3aed", ls="--", lw=1.5, alpha=0.8)
    ax.axvline(67.4, color="#7c3aed", ls=":", lw=1.5, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
    ax.set_title("Minimal-Dataset Ladder Variants")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "variant_h0_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tables(primary_fits: Dict[str, PrimaryFit], fits: List[LadderFit], sbf_diag: Dict[str, float]) -> None:
    host_table = primary_fits["all_primary"].host_mean.rename("mu_host").to_frame()
    host_table["mu_host_err"] = np.sqrt(np.diag(primary_fits["all_primary"].covariance))
    host_table.to_csv(OUTPUT_DIR / "primary_host_distances.csv", index_label="host")
    primary_fits["all_primary"].covariance.to_csv(OUTPUT_DIR / "primary_host_covariance.csv", index_label="host")

    variant_rows = []
    for fit in fits:
        variant_rows.append(
            {
                "variant": fit.name,
                "h0": fit.h0,
                "h0_err_low": fit.h0_err_low,
                "h0_err_high": fit.h0_err_high,
                "m_abs": fit.m_abs,
                "m_abs_err": fit.m_abs_err,
                "sigma_cal": fit.sigma_cal,
                "sigma_flow": fit.sigma_flow,
                "n_calibrators": fit.n_calibrators,
                "n_flow": fit.n_flow,
                "n_host_constraints": fit.n_host_constraints,
                "nll": fit.nll,
            }
        )
    variant_df = pd.DataFrame(variant_rows).sort_values("h0")
    variant_df.to_csv(OUTPUT_DIR / "variant_results.csv", index=False)

    baseline = next(fit for fit in fits if fit.name == "baseline_all")
    baseline.calibrators.to_csv(OUTPUT_DIR / "baseline_calibrators.csv", index=False)
    baseline.flow.to_csv(OUTPUT_DIR / "baseline_hubble_flow.csv", index=False)

    summary = {
        "published_target_h0": 73.50,
        "published_target_err": 0.81,
        "early_universe_reference_h0": 67.4,
        "early_universe_reference_err": 0.5,
        "baseline_variant": {
            "h0": baseline.h0,
            "h0_err_low": baseline.h0_err_low,
            "h0_err_high": baseline.h0_err_high,
            "m_abs": baseline.m_abs,
            "m_abs_err": baseline.m_abs_err,
            "sigma_cal": baseline.sigma_cal,
            "sigma_flow": baseline.sigma_flow,
        },
        "sbf_rank_diagnostic": sbf_diag,
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


def write_report(primary_fits: Dict[str, PrimaryFit], fits: List[LadderFit], sbf_diag: Dict[str, float]) -> None:
    baseline = next(fit for fit in fits if fit.name == "baseline_all")
    variants = pd.DataFrame(
        {
            "variant": [fit.name for fit in fits],
            "h0": [fit.h0 for fit in fits],
            "err_sym": [0.5 * (fit.h0_err_low + fit.h0_err_high) for fit in fits],
        }
    )
    jackknife = variants[variants["variant"].str.startswith("jackknife_")].copy()
    jackknife["host"] = jackknife["variant"].str.replace("jackknife_", "", regex=False)
    jackknife = jackknife.sort_values("h0")

    published_h0 = 73.50
    published_err = 0.81
    cmb_h0 = 67.4
    cmb_err = 0.5
    baseline_err = 0.5 * (baseline.h0_err_low + baseline.h0_err_high)
    delta_published = baseline.h0 - published_h0
    delta_cmb = baseline.h0 - cmb_h0
    tension_published = delta_published / math.sqrt(baseline_err**2 + published_err**2)
    tension_cmb = delta_cmb / math.sqrt(baseline_err**2 + cmb_err**2)

    lowest_jackknife = jackknife[["host", "h0"]].head(3).to_markdown(index=False)
    highest_jackknife = jackknife[["host", "h0"]].tail(3).to_markdown(index=False)

    report = f"""# Covariance-Weighted Reconstruction of a Minimal Local Distance Network

## Abstract
I implemented a reproducible generalized least-squares reconstruction of the minimal `H0DN` dataset supplied in `data/H0DN_MinimalDataset.txt`, propagated the covariance of primary-indicator host distances into a Type Ia supernova (SN Ia) ladder fit, and examined anchor and indicator variants. The solvable branch of the supplied network is the SN Ia ladder: geometric anchors feed host distance moduli through Cepheid/TRGB measurements, these calibrate SN Ia absolute magnitudes, and the calibrated SNe are compared with a low-redshift Hubble-flow sample. Within this minimal dataset, the baseline covariance-aware SN-only reconstruction gives `H0 = {baseline.h0:.2f} +{baseline.h0_err_high:.2f}/-{baseline.h0_err_low:.2f} km s^-1 Mpc^-1`, substantially above the task-stated Local Distance Network consensus (`73.50 ± 0.81 km s^-1 Mpc^-1`) and also above the early-universe reference (`67.4 ± 0.5 km s^-1 Mpc^-1`). The dominant reason is strong internal inconsistency in the supplied SN calibrator branch, which requires large fitted intrinsic scatter (`sigma_cal = {baseline.sigma_cal:.3f} mag`). I therefore interpret this analysis as a rigorous reconstruction of the *minimal provided data product*, not as a complete reproduction of the full paper result.

## 1. Scientific Context
The research goal described in the task is a Local Distance Network (LDN) measurement of the Hubble constant that combines multiple geometric anchors, primary stellar indicators, secondary indicators, and Hubble-flow tracers through covariance-weighted consensus. The `related_work/` directory contains surrounding SH0ES and Pantheon-era papers, especially the 2022 SH0ES Cepheid+SN analysis in `paper_000.pdf`, but it does not appear to contain the exact Local Distance Network paper referenced in the task description. Consequently, I used the prompt-stated consensus value (`73.50 ± 0.81 km s^-1 Mpc^-1`) as the published benchmark and treated the supplied dataset as the authoritative numerical input for the reconstruction.

## 2. Data and Model
### 2.1 Dataset structure
The minimal dataset contains:

- Three geometric anchors: MW, LMC, and NGC4258.
- Eleven primary-indicator host measurements, split between Cepheid and TRGB distances.
- Seven SN Ia calibrators linked to hosts with primary distances.
- Five Hubble-flow SN Ia data points.
- Three SBF calibrators and three Hubble-flow SBF points.

Figure ![Dataset overview](images/data_overview.png) summarizes the data inventory.

### 2.2 Primary-indicator GLS layer
For each host-distance measurement `y_i`, I modeled

`y_i = mu_host(i) + noise_i`

with covariance

`C_ij = delta_ij * sigma_meas,i^2 + I[anchor_i=anchor_j] * sigma_anchor^2 + I[(method_i,anchor_i)=(method_j,anchor_j)] * sigma_method-anchor^2`.

This construction follows the covariance logic implied by the dataset: measurements sharing an anchor inherit common anchor-distance uncertainty, and measurements sharing the same method-anchor calibration inherit an additional systematic term. Solving the generalized least-squares system yields posterior host moduli and their covariance matrix.

Figure ![Primary host distances](images/primary_host_distances.png) shows the raw measurements together with the GLS host posteriors, and Figure ![Primary covariance](images/primary_covariance_heatmap.png) shows the resulting host-distance correlation matrix.

### 2.3 SN ladder likelihood
I used the posterior host-distance covariance as input to a second-stage SN ladder model:

- Calibrators: `m_B^cal = mu_host + M_B + epsilon_cal`
- Hubble flow: `m_B^flow = mu(z, H0) + M_B + epsilon_flow`

where `mu(z, H0)` is a low-redshift cosmographic luminosity-distance modulus with fixed `q0 = -0.55`, and `epsilon_cal`, `epsilon_flow` are Gaussian intrinsic-scatter terms fitted simultaneously with `H0` and the shared SN absolute-magnitude proxy `M_B`. The primary host-distance covariance is propagated directly into the calibrator covariance matrix.

### 2.4 Why SBF is not part of the baseline fit
The supplied SBF block is not independently anchored in this minimal dataset. The SBF calibrator equations contain a shared SBF absolute-magnitude zero point plus group distance moduli (Fornax and Virgo), but no external absolute constraint on those group distances is included here. A simple rank diagnostic of the SBF calibrator design matrix gives rank `{sbf_diag["design_rank"]}` for `{sbf_diag["n_parameters"]}` linear parameters, demonstrating a one-dimensional zero-point degeneracy even before including the listed depth scatter. I therefore report the SBF branch as underconstrained rather than imposing hidden priors.

## 3. Results
### 3.1 Baseline covariance-aware SN result
The baseline fit using all available primary measurements yields:

- `H0 = {baseline.h0:.2f} +{baseline.h0_err_high:.2f}/-{baseline.h0_err_low:.2f} km s^-1 Mpc^-1`
- `M_B = {baseline.m_abs:.3f} ± {baseline.m_abs_err:.3f} mag`
- Calibrator intrinsic scatter `sigma_cal = {baseline.sigma_cal:.3f} mag`
- Hubble-flow intrinsic scatter `sigma_flow = {baseline.sigma_flow:.3f} mag`

Figure ![Calibrator scatter](images/calibrator_absolute_magnitude.png) shows the calibrator absolute-magnitude proxy distribution. The calibrator host `NGC1309` is conspicuously brighter than the rest of the set, while most other calibrators cluster around much fainter values. This drives the large inferred calibrator scatter and the broad sensitivity of `H0` to variant choices.

Figure ![Hubble diagram](images/hubble_diagram.png) shows the Hubble-flow SN distance moduli implied by the baseline calibration. The flow points are described by the fitted cosmographic curve only after allowing non-negligible intrinsic scatter.

### 3.2 Variant analysis
Figure ![Variant forest](images/variant_h0_forest.png) compares the main minimal-dataset variants with the task-stated published consensus and the early-universe reference. The most informative variants are:

- Baseline all-primary fit: `H0 = {next(fit for fit in fits if fit.name == "baseline_all").h0:.2f} km s^-1 Mpc^-1`
- Cepheid-only primary layer: `H0 = {next(fit for fit in fits if fit.name == "cepheid_only").h0:.2f} km s^-1 Mpc^-1`
- TRGB-only primary layer: `H0 = {next(fit for fit in fits if fit.name == "trgb_only").h0:.2f} km s^-1 Mpc^-1`
- NGC4258-only anchor: `H0 = {next(fit for fit in fits if fit.name == "n4258_only").h0:.2f} km s^-1 Mpc^-1`
- LMC-only anchor: `H0 = {next(fit for fit in fits if fit.name == "lmc_only").h0:.2f} km s^-1 Mpc^-1`
- Excluding `NGC1309`: `H0 = {next(fit for fit in fits if fit.name == "exclude_ngc1309").h0:.2f} km s^-1 Mpc^-1`

The jackknife fits show that removing `NGC1309` shifts `H0` upward most strongly, while removing several of the fainter calibrators shifts it downward. The ladder is therefore not close to the high-stability regime expected for a 1% consensus result.

### 3.3 Comparison with the target and the early-universe reference
Relative to the prompt-stated consensus:

- Difference from published target: `Delta H0 = {delta_published:.2f} km s^-1 Mpc^-1`
- Effective discrepancy using the baseline profile width: `{tension_published:.1f} sigma`

Relative to the early-universe reference:

- Difference from `67.4 ± 0.5`: `Delta H0 = {delta_cmb:.2f} km s^-1 Mpc^-1`
- Effective discrepancy using the baseline profile width: `{tension_cmb:.1f} sigma`

These numbers should not be overinterpreted as physical tension estimates. They mostly quantify that the minimal supplied dataset, as reconstructed here, does not numerically reproduce the paper-level consensus target.

## 4. Interpretation
The analysis supports three conclusions.

First, the covariance-aware GLS machinery itself is straightforward and reproducible with the provided data. The host-distance posteriors are well behaved, and the propagation of their covariance into the SN ladder is technically stable.

Second, the absolute-scale information carried by the SN calibrator branch is internally inconsistent. The calibrator absolute-magnitude proxy spans more than a magnitude, far larger than expected for a well-standardized SN Ia ladder. This is why the fit prefers a large intrinsic scatter term and why the recovered `H0` depends strongly on variant choice.

Third, the supplied SBF subset is not sufficient for an independent anchored `H0` estimate without additional priors or extra data products. A full Local Distance Network consensus measurement requires precisely those missing cross-links: more primary indicators, more secondary calibrators, explicit covariance structure across methods, and enough anchored branches to dilute any single outlier.

## 5. Limitations
- The exact Local Distance Network paper is not present in `related_work/`, so the published benchmark is taken from the task description rather than extracted from a source file in the workspace.
- The cosmographic Hubble-flow model is low-redshift and intentionally simple.
- I treated the listed anchor and method-anchor uncertainties as Gaussian shared covariance terms, which is consistent with the dataset layout but still an assumption.
- The SBF branch is only diagnosed, not forced into the baseline `H0` result, because the minimal dataset does not independently anchor it.

## 6. Reproducibility
The full analysis is implemented in `code/analyze_h0dn.py`. Running

```bash
python code/analyze_h0dn.py
```

regenerates all tables in `outputs/` and all figures in `report/images/`.

## Appendix: Notable jackknife shifts
The five lowest and highest jackknife `H0` values are:

Lowest:
{lowest_jackknife}

Highest:
{highest_jackknife}
"""
    (ROOT / "report" / "report.md").write_text(report)


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid", context="talk")
    dataset = load_dataset(DATA_PATH)
    primary_fits, fits = build_variant_suite(dataset)
    sbf_diag = fit_sbf_rank_diagnostic(dataset)

    make_data_overview_figure(dataset, primary_fits)
    make_primary_host_figure(primary_fits["all_primary"])
    make_covariance_heatmap(primary_fits["all_primary"])
    baseline = next(fit for fit in fits if fit.name == "baseline_all")
    make_calibrator_figure(baseline)
    make_hubble_diagram(baseline)
    make_variant_forest(fits)

    save_tables(primary_fits, fits, sbf_diag)
    write_report(primary_fits, fits, sbf_diag)


if __name__ == "__main__":
    main()
