#!/usr/bin/env python3
"""Reproducible analysis for 40-qubit random circuit sampling verification data."""

from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib_rcs_analysis"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
DATA_RESULTS = ROOT / "data" / "results" / "N40_verification"
DATA_AMPLITUDES = ROOT / "data" / "amplitudes" / "N40_verification"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"
SEED = 20260402


@dataclass(frozen=True)
class InstanceKey:
    n: int
    depth: int
    replica: int


def parse_instance_key(path: Path) -> InstanceKey:
    match = re.search(r"N(\d+)_d(\d+)_r(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse instance key from {path}")
    return InstanceKey(n=int(match.group(1)), depth=int(match.group(2)), replica=int(match.group(3)))


def normalize_bitstring(raw: str | Sequence[int]) -> str:
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("("):
            values = ast.literal_eval(raw)
            return "".join(str(int(bit)) for bit in values)
        if set(raw).issubset({"0", "1"}):
            return raw
        raise ValueError(f"Unsupported bitstring format: {raw[:80]}")
    return "".join(str(int(bit)) for bit in raw)


def amplitude_to_probability(raw: object) -> float:
    if isinstance(raw, (float, int)):
        return float(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if "j" in value:
            amp = complex(value.strip("()"))
            return float(amp.real * amp.real + amp.imag * amp.imag)
        return float(value)
    raise TypeError(f"Unsupported amplitude/probability payload: {type(raw)!r}")


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin


def bootstrap_mean_interval(values: Sequence[float], rng: np.random.Generator, num_boot: int = 4000) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, array.size, size=(num_boot, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def repeated_values_from_counts(metric_by_bitstring: Dict[str, float], counts: Dict[str, int]) -> List[float]:
    values: List[float] = []
    for bitstring, count in counts.items():
        values.extend([metric_by_bitstring[bitstring]] * int(count))
    return values


def load_xeb_records() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for counts_path in sorted(DATA_RESULTS.glob("N40_d*_XEB/*_counts.json")):
        key = parse_instance_key(counts_path)
        amp_path = DATA_AMPLITUDES / counts_path.parent.name / counts_path.name.replace("_counts.json", "_amplitudes.json")
        counts_raw = json.loads(counts_path.read_text())
        amps_raw = json.loads(amp_path.read_text())

        counts = {normalize_bitstring(k): int(v) for k, v in counts_raw.items()}
        ideal_probs = {normalize_bitstring(k): amplitude_to_probability(v) for k, v in amps_raw.items()}
        missing = sorted(set(counts) - set(ideal_probs))
        if missing:
            raise KeyError(f"Missing ideal probabilities for {counts_path}: {missing[:3]}")

        shot_metrics = repeated_values_from_counts(
            {bitstring: (2**key.n) * prob - 1 for bitstring, prob in ideal_probs.items()},
            counts,
        )
        mean_fidelity = float(np.mean(shot_metrics))
        std_fidelity = float(np.std(shot_metrics, ddof=1)) if len(shot_metrics) > 1 else 0.0
        se_fidelity = std_fidelity / math.sqrt(len(shot_metrics)) if len(shot_metrics) > 0 else float("nan")
        tcrit = stats.t.ppf(0.975, df=max(len(shot_metrics) - 1, 1))
        ci_low = mean_fidelity - tcrit * se_fidelity
        ci_high = mean_fidelity + tcrit * se_fidelity
        boot_low, boot_high = bootstrap_mean_interval(shot_metrics, rng)

        weighted_prob = float(np.average([ideal_probs[b] for b in counts], weights=[counts[b] for b in counts]))
        rows.append(
            {
                "n": key.n,
                "depth": key.depth,
                "replica": key.replica,
                "shots": int(sum(counts.values())),
                "unique_measured_bitstrings": int(len(counts)),
                "subset_size": int(len(ideal_probs)),
                "weighted_ideal_probability": weighted_prob,
                "xeb_fidelity": mean_fidelity,
                "xeb_std": std_fidelity,
                "xeb_se": se_fidelity,
                "xeb_ci95_low_t": ci_low,
                "xeb_ci95_high_t": ci_high,
                "xeb_ci95_low_boot": boot_low,
                "xeb_ci95_high_boot": boot_high,
            }
        )
    return pd.DataFrame(rows).sort_values(["depth", "replica"]).reset_index(drop=True)


def load_mb_records() -> pd.DataFrame:
    rows = []
    for counts_path in sorted(DATA_RESULTS.glob("N40_d*_MB/*_counts.json")):
        key = parse_instance_key(counts_path)
        ideal_path = counts_path.with_name(counts_path.name.replace("_counts.json", "_ideal_bitstring.json"))
        counts_raw = json.loads(counts_path.read_text())
        ideal_raw = json.loads(ideal_path.read_text())

        counts = {normalize_bitstring(k): int(v) for k, v in counts_raw.items()}
        ideal_bitstring = normalize_bitstring(ideal_raw)
        total_shots = int(sum(counts.values()))
        ideal_hits = int(counts.get(ideal_bitstring, 0))
        success_prob = ideal_hits / total_shots
        ci_low, ci_high = wilson_interval(ideal_hits, total_shots)
        baseline = 2 ** (-key.n)
        polarization = (success_prob - baseline) / (1 - baseline)

        rows.append(
            {
                "n": key.n,
                "depth": key.depth,
                "replica": key.replica,
                "shots": total_shots,
                "unique_measured_bitstrings": int(len(counts)),
                "ideal_hits": ideal_hits,
                "mb_success_probability": success_prob,
                "mb_polarization_proxy": polarization,
                "mb_ci95_low": ci_low,
                "mb_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values(["depth", "replica"]).reset_index(drop=True)


def summarise_by_depth(instances: pd.DataFrame, metric: str, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    grouped_rows = []
    for depth, frame in instances.groupby("depth", sort=True):
        values = frame[metric].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if len(values) > 0 else float("nan")
        ci_low, ci_high = bootstrap_mean_interval(values, rng)
        grouped_rows.append(
            {
                "depth": int(depth),
                f"{prefix}_instances": int(len(frame)),
                f"{prefix}_mean": mean,
                f"{prefix}_median": float(np.median(values)),
                f"{prefix}_std": std,
                f"{prefix}_sem": sem,
                f"{prefix}_ci95_low_boot": ci_low,
                f"{prefix}_ci95_high_boot": ci_high,
            }
        )
    return pd.DataFrame(grouped_rows).sort_values("depth").reset_index(drop=True)


def fit_exponential_decay(depth_summary: pd.DataFrame, metric_mean: str, metric_sem: str) -> dict:
    fit_frame = depth_summary[depth_summary[metric_mean] > 0].copy()
    x = fit_frame["depth"].to_numpy(dtype=float)
    y = fit_frame[metric_mean].to_numpy(dtype=float)
    sigma = fit_frame[metric_sem].replace(0, np.nan).fillna(fit_frame[metric_sem].mean()).to_numpy(dtype=float)

    def model(depth: np.ndarray, amplitude: float, decay_rate: float) -> np.ndarray:
        return amplitude * np.exp(-decay_rate * depth)

    params, covariance = optimize.curve_fit(
        model,
        x,
        y,
        p0=(float(y.max()), 0.05),
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0.0, 0.0], [10.0, 5.0]),
        maxfev=20000,
    )
    amp, decay = params
    amp_se, decay_se = np.sqrt(np.diag(covariance))
    predictions = model(x, *params)
    residuals = y - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "amplitude": float(amp),
        "amplitude_se": float(amp_se),
        "decay_rate_per_cycle": float(decay),
        "decay_rate_per_cycle_se": float(decay_se),
        "fit_r2": r2,
    }


def plot_data_overview(xeb: pd.DataFrame, mb: pd.DataFrame) -> None:
    depth_table = (
        xeb.groupby("depth", as_index=False)
        .agg(xeb_instances=("replica", "size"), xeb_shots=("shots", "mean"), xeb_subset=("subset_size", "mean"))
        .merge(
            mb.groupby("depth", as_index=False).agg(mb_instances=("replica", "size"), mb_shots=("shots", "mean")),
            on="depth",
            how="outer",
        )
        .sort_values("depth")
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    width = 0.38
    x = np.arange(len(depth_table))
    axes[0].bar(x - width / 2, depth_table["xeb_instances"], width=width, label="XEB instances", color="#1f77b4")
    axes[0].bar(x + width / 2, depth_table["mb_instances"], width=width, label="MB instances", color="#ff7f0e")
    axes[0].set_xticks(x, depth_table["depth"])
    axes[0].set_xlabel("Circuit depth")
    axes[0].set_ylabel("Instance count")
    axes[0].set_title("Dataset coverage by depth")
    axes[0].legend(frameon=False)

    axes[1].plot(depth_table["depth"], depth_table["xeb_shots"], marker="o", label="Mean XEB shots", color="#1f77b4")
    axes[1].plot(depth_table["depth"], depth_table["mb_shots"], marker="s", label="Mean MB shots", color="#ff7f0e")
    axes[1].plot(depth_table["depth"], depth_table["xeb_subset"], marker="^", label="XEB ideal subset size", color="#2ca02c")
    axes[1].set_xlabel("Circuit depth")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-instance sample budget")
    axes[1].legend(frameon=False)

    fig.savefig(REPORT_IMAGES / "data_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_xeb_fidelity(xeb: pd.DataFrame, xeb_depth: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    sns.stripplot(
        data=xeb,
        x="depth",
        y="xeb_fidelity",
        ax=axes[0],
        color="#4c72b0",
        alpha=0.65,
        size=4.0,
    )
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_title("Per-instance XEB fidelity estimates")
    axes[0].set_xlabel("Circuit depth")
    axes[0].set_ylabel(r"$F_{\mathrm{XEB}}$")

    axes[1].errorbar(
        xeb_depth["depth"],
        xeb_depth["xeb_mean"],
        yerr=1.96 * xeb_depth["xeb_sem"],
        fmt="o-",
        color="#4c72b0",
        capsize=4,
        linewidth=2,
    )
    axes[1].fill_between(
        xeb_depth["depth"],
        xeb_depth["xeb_ci95_low_boot"],
        xeb_depth["xeb_ci95_high_boot"],
        alpha=0.2,
        color="#4c72b0",
        label="Bootstrap 95% CI",
    )
    axes[1].set_title("Depth-averaged XEB fidelity")
    axes[1].set_xlabel("Circuit depth")
    axes[1].set_ylabel(r"Mean $F_{\mathrm{XEB}}$")
    axes[1].legend(frameon=False)

    fig.savefig(REPORT_IMAGES / "xeb_fidelity_by_depth.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_xeb_vs_mb(xeb_depth: pd.DataFrame, mb_depth: pd.DataFrame) -> None:
    merged = xeb_depth.merge(mb_depth, on="depth", how="inner")
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.errorbar(
        merged["depth"],
        merged["xeb_mean"],
        yerr=1.96 * merged["xeb_sem"],
        fmt="o-",
        linewidth=2,
        capsize=4,
        label="Mean XEB fidelity",
        color="#1f77b4",
    )
    ax.errorbar(
        merged["depth"],
        merged["mb_success_probability_mean"],
        yerr=1.96 * merged["mb_success_probability_sem"],
        fmt="s-",
        linewidth=2,
        capsize=4,
        label="MB target-hit probability",
        color="#d62728",
    )
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Metric value")
    ax.set_title("Consistent decay across XEB and MB-derived proxies")
    ax.legend(frameon=False)

    fig.savefig(REPORT_IMAGES / "xeb_mb_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classical_gap(xeb_depth: pd.DataFrame, n_qubits: int) -> None:
    porter_thomas_scale = 2 ** (-n_qubits / 2)
    ratio = xeb_depth["xeb_mean"] / porter_thomas_scale

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].errorbar(
        xeb_depth["depth"],
        xeb_depth["xeb_mean"],
        yerr=1.96 * xeb_depth["xeb_sem"],
        fmt="o-",
        linewidth=2,
        capsize=4,
        label="Experimental mean XEB fidelity",
        color="#1f77b4",
    )
    axes[0].axhline(
        porter_thomas_scale,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=rf"Porter-Thomas scale $2^{{-N/2}}$ (N={n_qubits})",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Circuit depth")
    axes[0].set_ylabel("Scale")
    axes[0].set_title("Experimental fidelity vs. Porter-Thomas scale")
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].plot(
        xeb_depth["depth"],
        ratio,
        color="#2ca02c",
        linestyle="-",
        marker="^",
        linewidth=2,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Circuit depth")
    axes[1].set_ylabel(r"$F_{\mathrm{XEB}} / 2^{-N/2}$")
    axes[1].set_title("Gap ratio")
    axes[1].axhspan(ratio.min(), ratio.max(), color="#2ca02c", alpha=0.08)
    axes[1].text(
        0.03,
        0.06,
        f"Range: {ratio.min():.2e} to {ratio.max():.2e}",
        transform=axes[1].transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.savefig(REPORT_IMAGES / "classical_gap_reference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(xeb: pd.DataFrame, mb: pd.DataFrame, xeb_depth: pd.DataFrame, mb_depth: pd.DataFrame, fit_stats: dict) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)

    combined = xeb.merge(
        mb[["n", "depth", "replica", "mb_success_probability", "mb_polarization_proxy", "mb_ci95_low", "mb_ci95_high"]],
        on=["n", "depth", "replica"],
        how="left",
    )
    combined.to_csv(OUTPUTS / "per_instance_fidelity_estimates.csv", index=False)
    xeb_depth.to_csv(OUTPUTS / "xeb_depth_summary.csv", index=False)
    mb_depth.to_csv(OUTPUTS / "mb_depth_summary.csv", index=False)
    (OUTPUTS / "fit_summary.json").write_text(json.dumps(fit_stats, indent=2))


def make_report_assets(xeb: pd.DataFrame, mb: pd.DataFrame, xeb_depth: pd.DataFrame, mb_depth: pd.DataFrame) -> None:
    plot_data_overview(xeb, mb)
    plot_xeb_fidelity(xeb, xeb_depth)
    plot_xeb_vs_mb(xeb_depth, mb_depth)
    plot_classical_gap(xeb_depth, int(xeb["n"].iloc[0]))


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    xeb = load_xeb_records()
    mb = load_mb_records()

    xeb_depth = summarise_by_depth(xeb, metric="xeb_fidelity", prefix="xeb")
    mb_depth = summarise_by_depth(mb, metric="mb_success_probability", prefix="mb_success_probability")
    fit_stats = fit_exponential_decay(xeb_depth, metric_mean="xeb_mean", metric_sem="xeb_sem")

    save_outputs(xeb, mb, xeb_depth, mb_depth, fit_stats)
    make_report_assets(xeb, mb, xeb_depth, mb_depth)

    print("Saved:")
    print(f"  {OUTPUTS / 'per_instance_fidelity_estimates.csv'}")
    print(f"  {OUTPUTS / 'xeb_depth_summary.csv'}")
    print(f"  {OUTPUTS / 'mb_depth_summary.csv'}")
    print(f"  {OUTPUTS / 'fit_summary.json'}")
    for image_name in [
        "data_overview.png",
        "xeb_fidelity_by_depth.png",
        "xeb_mb_comparison.png",
        "classical_gap_reference.png",
    ]:
        print(f"  {REPORT_IMAGES / image_name}")


if __name__ == "__main__":
    main()
