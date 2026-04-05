from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "flow" / "0000"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200


class FlyvisUnpickler(pickle.Unpickler):
    """Fallback unpickler that replaces unknown flyvis classes with inert placeholders."""

    def find_class(self, module: str, name: str):
        if module.startswith("flyvis"):
            return type(name, (), {})
        return super().find_class(module, name)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def read_validation_loss(path: Path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["data"][()])


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def safe_repr(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def normal_ci_95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0])
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = 1.96 * sd / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "checkpoint_path": str(path.relative_to(ROOT)),
        "checkpoint_size_bytes": path.stat().st_size,
        "is_zip": zipfile.is_zipfile(path),
        "zip_members": np.nan,
        "zip_data_shards": np.nan,
        "parameter_keys_detected": np.nan,
        "parameter_key_preview": "",
        "network_parameter_shapes": "",
        "decoder_flow_parameter_shapes": "",
        "network_bias_mean": np.nan,
        "network_time_const_mean": np.nan,
        "network_syn_strength_mean": np.nan,
        "decoder_output_bias_mean": np.nan,
        "torch_load_success": False,
        "torch_load_error": "",
    }
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            info["zip_members"] = len(names)
            info["zip_data_shards"] = sum("/data/" in n for n in names)
    try:
        import torch

        obj = torch.load(path, map_location="cpu")
        info["torch_load_success"] = True
        if isinstance(obj, dict):
            keys = sorted(map(str, obj.keys()))
            info["parameter_keys_detected"] = len(keys)
            info["parameter_key_preview"] = json.dumps(keys[:12])
            if isinstance(obj.get("network"), dict):
                net = obj["network"]
                info["network_parameter_shapes"] = json.dumps({k: list(v.shape) for k, v in net.items() if hasattr(v, "shape")}, sort_keys=True)
                if "nodes_bias" in net:
                    info["network_bias_mean"] = float(net["nodes_bias"].float().mean())
                if "nodes_time_const" in net:
                    info["network_time_const_mean"] = float(net["nodes_time_const"].float().mean())
                if "edges_syn_strength" in net:
                    info["network_syn_strength_mean"] = float(net["edges_syn_strength"].float().mean())
            flow = obj.get("decoder", {}).get("flow") if isinstance(obj.get("decoder"), dict) else None
            if isinstance(flow, dict):
                info["decoder_flow_parameter_shapes"] = json.dumps({k: list(v.shape) for k, v in flow.items() if hasattr(v, "shape")}, sort_keys=True)
                if "decoder.0.bias" in flow:
                    info["decoder_output_bias_mean"] = float(flow["decoder.0.bias"].float().mean())
        else:
            info["parameter_keys_detected"] = 0
            info["parameter_key_preview"] = type(obj).__name__
    except Exception as e:  # noqa: BLE001
        info["torch_load_error"] = f"{type(e).__name__}: {e}"
    return info


def extract_cell_type_metadata(path: Path) -> dict[str, Any]:
    cell_type = path.stem
    match = re.match(r"([A-Za-z]+)(.*)", cell_type)
    family = match.group(1) if match else cell_type
    suffix = match.group(2) if match else ""
    return {
        "cell_type": cell_type,
        "family": family,
        "suffix": suffix,
        "pickle_path": str(path.relative_to(ROOT)),
    }


def try_pickle_summary(path: Path) -> dict[str, Any]:
    summary = {"pickle_load_success": False, "pickle_type": "", "pickle_error": ""}
    try:
        with path.open("rb") as f:
            obj = FlyvisUnpickler(f).load()
        summary["pickle_load_success"] = True
        summary["pickle_type"] = type(obj).__name__
    except Exception as e:  # noqa: BLE001
        summary["pickle_error"] = f"{type(e).__name__}: {e}"
    return summary


def analyze() -> None:
    ensure_dirs()

    run_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit()])
    rows: list[dict[str, Any]] = []
    config_flats: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        meta_path = run_dir / "_meta.yaml"
        meta = load_yaml(meta_path)
        config = meta["config"]
        flat = flatten_dict(config)
        config_flats.append(flat)

        val1 = read_validation_loss(run_dir / "validation_loss.h5")
        val2 = read_validation_loss(run_dir / "validation" / "loss.h5")
        best_chkpt = run_dir / "best_chkpt"
        chkpt_info = inspect_checkpoint(best_chkpt)
        checkpoint_rows.append({"run_id": run_id, **chkpt_info})

        row = {
            "run_id": run_id,
            "validation_loss": val1,
            "validation_loss_duplicate": val2,
            "duplicate_abs_diff": abs(val1 - val2),
            "status": meta.get("status", ""),
            "checkpoint_size_bytes": best_chkpt.stat().st_size,
            "network_bias_mean": chkpt_info["network_bias_mean"],
            "network_time_const_mean": chkpt_info["network_time_const_mean"],
            "network_syn_strength_mean": chkpt_info["network_syn_strength_mean"],
            "decoder_output_bias_mean": chkpt_info["decoder_output_bias_mean"],
            "dataset_type": config["task"]["dataset"]["type"],
            "tasks": ",".join(config["task"]["dataset"]["tasks"]),
            "n_frames": config["task"]["dataset"]["n_frames"],
            "dt": config["task"]["dataset"]["dt"],
            "batch_size": config["task"]["batch_size"],
            "n_iters": config["task"]["n_iters"],
            "fold": config["task"]["fold"],
            "seed": config["task"]["seed"],
            "decoder_type": config["task"]["decoder"]["flow"]["type"],
            "connectome_type": config["network"]["connectome"]["type"],
            "connectome_file": config["network"]["connectome"]["file"],
            "activation": config["network"]["dynamics"]["activation"]["type"],
            "syn_strength_scale": config["network"]["edge_config"]["syn_strength"]["scale"],
        }
        rows.append(row)

    ensemble_df = pd.DataFrame(rows).sort_values("validation_loss").reset_index(drop=True)
    ensemble_df["rank"] = np.arange(1, len(ensemble_df) + 1)
    ensemble_df.to_csv(OUTPUT_DIR / "ensemble_summary.csv", index=False)

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values("run_id").reset_index(drop=True)
    checkpoint_df.to_csv(OUTPUT_DIR / "checkpoint_inventory.csv", index=False)

    cfg_df = pd.DataFrame([{k: safe_repr(v) for k, v in flat.items()} for flat in config_flats])
    unique_counts = cfg_df.nunique(dropna=False).sort_values(ascending=False)
    varying = unique_counts[unique_counts > 1]
    constant = unique_counts[unique_counts == 1]
    config_consistency = {
        "n_runs": len(run_dirs),
        "n_varying_fields": int(varying.shape[0]),
        "varying_fields": {k: sorted(cfg_df[k].unique().tolist()) for k in varying.index.tolist()},
        "n_constant_fields": int(constant.shape[0]),
    }
    with (OUTPUT_DIR / "config_consistency.json").open("w") as f:
        json.dump(config_consistency, f, indent=2)

    exemplar_config = flatten_dict(load_yaml(run_dirs[0] / "_meta.yaml")["config"])
    with (OUTPUT_DIR / "model_hyperparameters.json").open("w") as f:
        json.dump(exemplar_config, f, indent=2)

    cluster_dir = DATA_ROOT / "umap_and_clustering"
    cluster_rows = []
    for p in sorted(cluster_dir.glob("*.pickle")):
        row = extract_cell_type_metadata(p)
        row.update(try_pickle_summary(p))
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(OUTPUT_DIR / "clustering_inventory.csv", index=False)

    losses = ensemble_df["validation_loss"].tolist()
    mean_loss = float(np.mean(losses))
    median_loss = float(np.median(losses))
    std_loss = float(np.std(losses, ddof=1))
    ci_low, ci_high = normal_ci_95(losses)
    best = ensemble_df.iloc[0]
    worst = ensemble_df.iloc[-1]
    q1, q3 = np.percentile(losses, [25, 75])
    cv = std_loss / mean_loss

    summary = {
        "n_models": int(len(ensemble_df)),
        "validation_loss_mean": mean_loss,
        "validation_loss_median": median_loss,
        "validation_loss_std": std_loss,
        "validation_loss_cv": float(cv),
        "validation_loss_min": float(best["validation_loss"]),
        "validation_loss_max": float(worst["validation_loss"]),
        "validation_loss_iqr": [float(q1), float(q3)],
        "validation_loss_mean_95ci": [float(ci_low), float(ci_high)],
        "best_run": best["run_id"],
        "worst_run": worst["run_id"],
        "relative_gap_best_to_worst_pct": float((worst["validation_loss"] - best["validation_loss"]) / best["validation_loss"] * 100),
        "duplicate_loss_max_abs_diff": float(ensemble_df["duplicate_abs_diff"].max()),
        "n_cluster_pickles": int(len(cluster_df)),
        "cluster_family_counts": Counter(cluster_df["family"]).most_common(),
        "torch_checkpoint_load_success_count": int(checkpoint_df["torch_load_success"].sum()),
        "network_bias_mean_mean": float(ensemble_df["network_bias_mean"].mean()),
        "network_bias_mean_std": float(ensemble_df["network_bias_mean"].std(ddof=1)),
        "network_time_const_mean_mean": float(ensemble_df["network_time_const_mean"].mean()),
        "network_time_const_mean_std": float(ensemble_df["network_time_const_mean"].std(ddof=1)),
        "network_syn_strength_mean_mean": float(ensemble_df["network_syn_strength_mean"].mean()),
        "network_syn_strength_mean_std": float(ensemble_df["network_syn_strength_mean"].std(ddof=1)),
        "loss_vs_syn_strength_corr": float(np.corrcoef(ensemble_df["validation_loss"], ensemble_df["network_syn_strength_mean"])[0, 1]),
        "loss_vs_time_const_corr": float(np.corrcoef(ensemble_df["validation_loss"], ensemble_df["network_time_const_mean"])[0, 1]),
    }
    with (OUTPUT_DIR / "analysis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    top10 = ensemble_df.head(10).copy()
    bottom10 = ensemble_df.tail(10).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ensemble_df["validation_loss"], bins=12, kde=True, ax=ax, color="#4C78A8")
    ax.axvline(mean_loss, color="#F58518", linestyle="--", label=f"Mean = {mean_loss:.3f}")
    ax.axvline(median_loss, color="#54A24B", linestyle=":", label=f"Median = {median_loss:.3f}")
    ax.set_title("Distribution of validation loss across 50 pretrained DMNs")
    ax.set_xlabel("Validation loss")
    ax.set_ylabel("Number of models")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "validation_loss_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = top10.sort_values("validation_loss", ascending=False)
    sns.barplot(data=plot_df, x="validation_loss", y="run_id", hue="run_id", palette="viridis", dodge=False, legend=False, ax=ax)
    ax.set_title("Top-10 runs ranked by lowest validation loss")
    ax.set_xlabel("Validation loss")
    ax.set_ylabel("Run ID")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "top_runs_barplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=ensemble_df, x="rank", y="validation_loss", marker="o", ax=ax, color="#E45756")
    ax.fill_between(ensemble_df["rank"], ensemble_df["validation_loss"].min(), ensemble_df["validation_loss"], alpha=0.15, color="#E45756")
    ax.set_title("Ordered validation loss profile across the model ensemble")
    ax.set_xlabel("Rank (1 = best)")
    ax.set_ylabel("Validation loss")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "ranked_validation_profile.png")
    plt.close(fig)

    family_counts = cluster_df["family"].value_counts().reset_index()
    family_counts.columns = ["family", "count"]
    fig, ax = plt.subplots(figsize=(12, 8))
    top_families = family_counts.head(15)
    sns.barplot(data=top_families, x="count", y="family", hue="family", palette="mako", dodge=False, legend=False, ax=ax)
    ax.set_title("Most frequent cell-type families represented in clustering artifacts")
    ax.set_xlabel("Number of clustering files")
    ax.set_ylabel("Cell-type family")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "clustering_celltype_coverage.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=ensemble_df,
        x="network_syn_strength_mean",
        y="validation_loss",
        hue="validation_loss",
        palette="viridis",
        s=80,
        ax=ax,
        legend=False,
    )
    ax.set_title("Mean learned synaptic strength versus validation loss")
    ax.set_xlabel("Mean synaptic-strength parameter")
    ax.set_ylabel("Validation loss")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "syn_strength_vs_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=ensemble_df,
        x="network_time_const_mean",
        y="validation_loss",
        hue="validation_loss",
        palette="magma",
        s=80,
        ax=ax,
        legend=False,
    )
    ax.set_title("Mean learned time constant versus validation loss")
    ax.set_xlabel("Mean time-constant parameter")
    ax.set_ylabel("Validation loss")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "time_const_vs_loss.png")
    plt.close(fig)

    comparison = pd.concat([
        top10.assign(group="top10"),
        bottom10.assign(group="bottom10"),
    ])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=comparison, x="group", y="validation_loss", hue="group", palette=["#72B7B2", "#F58518"], dodge=False, legend=False, ax=ax)
    sns.stripplot(data=comparison, x="group", y="validation_loss", color="black", alpha=0.7, size=5, ax=ax)
    ax.set_title("Validation loss contrast between top and bottom deciles")
    ax.set_xlabel("")
    ax.set_ylabel("Validation loss")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "top_bottom_deciles_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pretrained Drosophila flow DMN ensemble")
    _ = parser.parse_args()
    analyze()
