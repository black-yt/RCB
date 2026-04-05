import argparse
import ast
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
RNG = np.random.default_rng(12345)


def parse_bitstring_key(key):
    key = str(key).strip()
    if key.startswith("("):
        values = ast.literal_eval(key)
        if isinstance(values, int):
            return str(values)
        return "".join(str(int(v)) for v in values)
    if key.startswith("["):
        values = ast.literal_eval(key)
        return "".join(str(int(v)) for v in values)
    return key.replace(" ", "")


def amplitude_to_prob(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "real" in value and "imag" in value:
            return float(value["real"]) ** 2 + float(value["imag"]) ** 2
        if "prob" in value:
            return float(value["prob"])
    text = str(value).strip()
    try:
        obj = complex(text.replace(" ", ""))
        return float(abs(obj) ** 2)
    except Exception:
        return float(text)


def discover_xeb_pairs():
    results_root = Path("data/results")
    amplitude_root = Path("data/amplitudes")
    pairs = []
    for count_path in sorted(results_root.glob("**/*_XEB_counts.json")):
        match = re.search(r"N(\d+)_d(\d+)_r(\d+)_XEB_counts\.json$", count_path.name)
        if not match:
            continue
        n, d, r = map(int, match.groups())
        amp_name = count_path.name.replace("_counts.json", "_amplitudes.json")
        rel = count_path.relative_to(results_root)
        amp_path = amplitude_root / rel.parent / amp_name
        if not amp_path.exists():
            continue
        path_text = str(count_path)
        if "N40_verification" in path_text:
            scan_type = "depth_scan"
        elif "N_scan_depth12" in path_text:
            scan_type = "qubit_scan"
        else:
            scan_type = "other"
        pairs.append(
            {
                "N": n,
                "d": d,
                "r": r,
                "scan_type": scan_type,
                "counts_path": str(count_path),
                "amplitudes_path": str(amp_path),
            }
        )
    return pd.DataFrame(pairs)


def discover_mb_files():
    rows = []
    for count_path in sorted(Path("data/results").glob("**/*_MB_counts.json")):
        match = re.search(r"N(\d+)_d(\d+)_r(\d+)_MB_counts\.json$", count_path.name)
        if not match:
            continue
        n, d, r = map(int, match.groups())
        ideal_path = count_path.with_name(count_path.name.replace("_counts.json", "_ideal_bitstring.json"))
        if not ideal_path.exists():
            continue
        rows.append(
            {
                "N": n,
                "d": d,
                "r": r,
                "counts_path": str(count_path),
                "ideal_path": str(ideal_path),
            }
        )
    return pd.DataFrame(rows)


def weighted_bootstrap_fxeb(probs, counts, n, n_boot=2000):
    total = int(np.round(counts.sum()))
    if total <= 0:
        return math.nan, math.nan, math.nan, math.nan
    p = counts / counts.sum()
    idx = np.arange(len(probs))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        draw = RNG.choice(idx, size=total, replace=True, p=p)
        samples[b] = (2 ** n) * probs[draw].mean() - 1.0
    mean = float(samples.mean())
    se = float(samples.std(ddof=1))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return mean, se, float(lo), float(hi)


def analyze_xeb_pair(row):
    counts = json.loads(Path(row["counts_path"]).read_text())
    amplitudes = json.loads(Path(row["amplitudes_path"]).read_text())

    parsed_counts = {parse_bitstring_key(k): int(v) for k, v in counts.items()}
    parsed_probs = {parse_bitstring_key(k): amplitude_to_prob(v) for k, v in amplitudes.items()}
    matched = sorted(set(parsed_counts) & set(parsed_probs))
    counts_arr = np.array([parsed_counts[k] for k in matched], dtype=float)
    probs_arr = np.array([parsed_probs[k] for k in matched], dtype=float)
    total_counts = float(counts_arr.sum())
    weighted_mean_prob = float(np.average(probs_arr, weights=counts_arr)) if total_counts else math.nan
    fxeb = (2 ** int(row["N"])) * weighted_mean_prob - 1.0 if total_counts else math.nan
    expanded = np.repeat(probs_arr, counts_arr.astype(int))
    analytic_se = float((2 ** int(row["N"])) * np.std(expanded, ddof=1) / math.sqrt(len(expanded))) if len(expanded) > 1 else math.nan
    boot_mean, boot_se, ci_lo, ci_hi = weighted_bootstrap_fxeb(probs_arr, counts_arr, int(row["N"]))
    return {
        **row.to_dict(),
        "matched_keys": len(matched),
        "total_shots_subset": int(total_counts),
        "weighted_mean_prob": weighted_mean_prob,
        "mean_log_prob": float(np.average(np.log(np.clip(probs_arr, 1e-300, None)), weights=counts_arr)),
        "fxeb": float(fxeb),
        "analytic_se": analytic_se,
        "bootstrap_mean": boot_mean,
        "bootstrap_se": boot_se,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "uniform_baseline": 0.0,
        "ideal_subset_mean_fxeb": float((2 ** int(row["N"])) * probs_arr.mean() - 1.0),
        "max_prob": float(probs_arr.max()),
        "min_prob": float(probs_arr.min()),
    }


def summarize_groups(df, by_cols):
    rows = []
    for key, grp in df.groupby(by_cols):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: val for col, val in zip(by_cols, key)}
        vals = grp["fxeb"].to_numpy(dtype=float)
        rec.update(
            {
                "num_instances": int(len(grp)),
                "fxeb_mean": float(np.mean(vals)),
                "fxeb_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "fxeb_sem": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "fxeb_ci95_low": float(np.mean(vals) - 1.96 * (np.std(vals, ddof=1) / math.sqrt(len(vals)))) if len(vals) > 1 else float(vals[0]),
                "fxeb_ci95_high": float(np.mean(vals) + 1.96 * (np.std(vals, ddof=1) / math.sqrt(len(vals)))) if len(vals) > 1 else float(vals[0]),
                "bootstrap_se_mean": float(grp["bootstrap_se"].mean()),
                "matched_keys_mean": float(grp["matched_keys"].mean()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(by_cols).reset_index(drop=True)


def analyze_mb_row(row):
    counts = json.loads(Path(row["counts_path"]).read_text())
    ideal = json.loads(Path(row["ideal_path"]).read_text())
    target = "".join(str(int(x)) for x in ideal)
    parsed_counts = {parse_bitstring_key(k): int(v) for k, v in counts.items()}
    total = sum(parsed_counts.values())
    success = parsed_counts.get(target, 0)
    rate = success / total if total else math.nan
    se = math.sqrt(rate * (1 - rate) / total) if total else math.nan
    return {
        **row.to_dict(),
        "ideal_bitstring": target,
        "subset_shots": int(total),
        "success_count": int(success),
        "mb_success_rate": float(rate),
        "mb_success_se": float(se),
    }


def make_figures(xeb_instances, xeb_summary, mb_instances):
    img_dir = Path("report/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(xeb_instances["matched_keys"], bins=10, ax=axes[0])
    axes[0].set_title("Matched bitstrings per XEB instance")
    axes[0].set_xlabel("Matched keys")
    sns.histplot(xeb_instances["fxeb"], bins=30, ax=axes[1])
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Distribution of per-instance XEB fidelity estimates")
    axes[1].set_xlabel("Linear XEB fidelity estimate")
    fig.tight_layout()
    fig.savefig(img_dir / "data_overview_histograms.png", dpi=200)
    plt.close(fig)

    depth = xeb_summary[(xeb_summary["scan_type"] == "depth_scan") & (xeb_summary["N"] == 40)].sort_values("d")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(depth["d"], depth["fxeb_mean"], yerr=1.96 * depth["fxeb_sem"], marker="o", capsize=4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Circuit depth d")
    ax.set_ylabel("Mean XEB fidelity")
    ax.set_title("Depth scan for N=40")
    fig.tight_layout()
    fig.savefig(img_dir / "xeb_depth_scan_N40.png", dpi=200)
    plt.close(fig)

    nscan = xeb_summary[(xeb_summary["scan_type"] == "qubit_scan") & (xeb_summary["d"] == 12)].sort_values("N")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(nscan["N"], nscan["fxeb_mean"], yerr=1.96 * nscan["fxeb_sem"], marker="o", capsize=4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Qubit count N")
    ax.set_ylabel("Mean XEB fidelity")
    ax.set_title("Qubit-count scan at depth d=12")
    fig.tight_layout()
    fig.savefig(img_dir / "xeb_qubit_scan_d12.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.errorbar(depth["d"], depth["fxeb_mean"], yerr=1.96 * depth["fxeb_sem"], marker="o", color="tab:blue", capsize=4, label="XEB fidelity")
    ax1.set_xlabel("Circuit depth d")
    ax1.set_ylabel("Mean XEB fidelity", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax2 = ax1.twinx()
    classical_cost = 2.0 ** depth["d"]
    ax2.plot(depth["d"], classical_cost / classical_cost.iloc[0], marker="s", color="tab:red", label="Relative classical cost proxy")
    ax2.set_ylabel("Relative cost proxy (normalized $2^d$)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title("Experimental fidelity vs classical cost proxy")
    fig.tight_layout()
    fig.savefig(img_dir / "fidelity_vs_classical_gap_proxy.png", dpi=200)
    plt.close(fig)

    if not mb_instances.empty:
        mb_summary = mb_instances.groupby(["N", "d"]).agg(mean_success=("mb_success_rate", "mean"), sem_success=("mb_success_rate", lambda s: float(np.std(s, ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0)).reset_index()
        mb_depth = mb_summary[mb_summary["N"] == 40].sort_values("d")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(mb_depth["d"], mb_depth["mean_success"], yerr=1.96 * mb_depth["sem_success"], marker="o", capsize=4)
        ax.set_xlabel("Circuit depth d")
        ax.set_ylabel("Mean MB exact-bitstring rate")
        ax.set_title("Supplemental MB validation for N=40")
        fig.tight_layout()
        fig.savefig(img_dir / "mb_depth_scan_N40.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "full"], default="full")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    Path("report/images").mkdir(parents=True, exist_ok=True)

    xeb_inventory = discover_xeb_pairs()
    mb_inventory = discover_mb_files()
    xeb_inventory.to_csv("outputs/data_inventory.csv", index=False)

    schema_examples = {
        "xeb_example": json.loads(Path(xeb_inventory.iloc[0]["counts_path"]).read_text()) if not xeb_inventory.empty else {},
        "amplitude_example": json.loads(Path(xeb_inventory.iloc[0]["amplitudes_path"]).read_text()) if not xeb_inventory.empty else {},
        "mb_example": json.loads(Path(mb_inventory.iloc[0]["counts_path"]).read_text()) if not mb_inventory.empty else {},
    }
    Path("outputs/schema_examples.json").write_text(json.dumps(schema_examples, indent=2)[:20000])

    inspection = {
        "num_xeb_instances": int(len(xeb_inventory)),
        "num_mb_instances": int(len(mb_inventory)),
        "xeb_qubits": sorted(xeb_inventory["N"].unique().tolist()) if not xeb_inventory.empty else [],
        "xeb_depths": sorted(xeb_inventory["d"].unique().tolist()) if not xeb_inventory.empty else [],
        "mb_depths": sorted(mb_inventory["d"].unique().tolist()) if not mb_inventory.empty else [],
    }
    Path("outputs/inspection_summary.json").write_text(json.dumps(inspection, indent=2))

    if args.mode == "inspect":
        return

    xeb_results = pd.DataFrame([analyze_xeb_pair(row) for _, row in xeb_inventory.iterrows()])
    xeb_results = xeb_results.sort_values(["scan_type", "N", "d", "r"]).reset_index(drop=True)
    xeb_results.to_csv("outputs/xeb_instance_results.csv", index=False)

    xeb_summary = summarize_groups(xeb_results, ["scan_type", "N", "d"])
    xeb_summary.to_csv("outputs/xeb_group_summary.csv", index=False)

    mb_results = pd.DataFrame([analyze_mb_row(row) for _, row in mb_inventory.iterrows()]) if not mb_inventory.empty else pd.DataFrame()
    mb_results.to_csv("outputs/mb_instance_results.csv", index=False)

    overview = {
        "global_xeb_mean": float(xeb_results["fxeb"].mean()),
        "global_xeb_std": float(xeb_results["fxeb"].std(ddof=1)),
        "num_negative_instances": int((xeb_results["fxeb"] < 0).sum()),
        "num_positive_instances": int((xeb_results["fxeb"] > 0).sum()),
        "depth_scan_N40_mean_range": [float(xeb_summary[(xeb_summary['scan_type'] == 'depth_scan') & (xeb_summary['N'] == 40)]['fxeb_mean'].min()), float(xeb_summary[(xeb_summary['scan_type'] == 'depth_scan') & (xeb_summary['N'] == 40)]['fxeb_mean'].max())],
    }
    Path("outputs/analysis_overview.json").write_text(json.dumps(overview, indent=2))

    make_figures(xeb_results, xeb_summary, mb_results)


if __name__ == "__main__":
    main()
