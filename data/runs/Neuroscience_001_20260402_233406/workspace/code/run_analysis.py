#!/usr/bin/env python3
"""Main analysis entry point for the Drosophila optic-lobe DMN workspace.

This script inspects the pretrained model ensemble, summarizes validation
performance, extracts task/config metadata, parses related-work PDF metadata,
and analyzes the provided UMAP/clustering pickles for neuron cell types.

Outputs are written under outputs/ and figures under report/images/.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run this analysis") from exc

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data" / "flow" / "0000"
RELATED_WORK_DIR = WORKSPACE / "related_work"
OUTPUT_DIR = WORKSPACE / "outputs"
FIG_DIR = WORKSPACE / "report" / "images"
MODEL_DIRS = sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()])
UMAP_DIR = DATA_DIR / "umap_and_clustering"


class GenericStateObject:
    """Fallback class for unpickling custom flyvis classes."""

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


class FlyvisCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("flyvis."):
            return GenericStateObject
        return super().find_class(module, name)


@dataclass
class UMAPSummary:
    cell_type: str
    n_models: int
    n_clusters: int
    embedding_x_mean: float
    embedding_y_mean: float
    embedding_x_std: float
    embedding_y_std: float
    silhouette_proxy: float
    cluster_entropy: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_validation_loss(path: Path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["data"][()])


def summarize_checkpoint_zip(path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        entries = []
        total_size = 0
        data_entries = 0
        tensor_bytes = 0
        for info in zf.infolist():
            total_size += info.file_size
            entries.append({"name": info.filename, "size": info.file_size})
            if "/data/" in info.filename:
                data_entries += 1
                tensor_bytes += info.file_size
        return {
            "zip_entries": len(entries),
            "total_uncompressed_bytes": total_size,
            "tensor_entry_count": data_entries,
            "tensor_bytes": tensor_bytes,
            "largest_entries": sorted(entries, key=lambda x: x["size"], reverse=True)[:10],
        }


def extract_core_config(meta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = meta["config"]
    dataset = cfg["task"]["dataset"]
    decoder = cfg["task"]["decoder"]["flow"]
    connectome = cfg["network"]["connectome"]
    node_cfg = cfg["network"]["node_config"]
    edge_cfg = cfg["network"]["edge_config"]
    return {
        "connectome_type": connectome.get("type"),
        "connectome_file": connectome.get("file"),
        "extent": connectome.get("extent"),
        "n_syn_fill": connectome.get("n_syn_fill"),
        "dynamics_type": cfg["network"]["dynamics"].get("type"),
        "activation": cfg["network"]["dynamics"].get("activation", {}).get("type"),
        "dataset_type": dataset.get("type"),
        "tasks": dataset.get("tasks"),
        "n_frames": dataset.get("n_frames"),
        "dt": dataset.get("dt"),
        "vertical_splits": dataset.get("vertical_splits"),
        "batch_size": cfg["task"].get("batch_size"),
        "n_iters": cfg["task"].get("n_iters"),
        "n_folds": cfg["task"].get("n_folds"),
        "fold": cfg["task"].get("fold"),
        "decoder_type": decoder.get("type"),
        "decoder_shape": decoder.get("shape"),
        "decoder_kernel_size": decoder.get("kernel_size"),
        "bias_trainable": node_cfg["bias"].get("requires_grad"),
        "time_const_trainable": node_cfg["time_const"].get("requires_grad"),
        "syn_strength_trainable": edge_cfg["syn_strength"].get("requires_grad"),
        "syn_count_trainable": edge_cfg["syn_count"].get("requires_grad"),
        "sign_trainable": edge_cfg["sign"].get("requires_grad"),
        "bias_groupby": node_cfg["bias"].get("groupby"),
        "time_const_groupby": node_cfg["time_const"].get("groupby"),
        "syn_strength_groupby": edge_cfg["syn_strength"].get("groupby"),
        "syn_count_groupby": edge_cfg["syn_count"].get("groupby"),
    }


def shannon_entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def silhouette_proxy(embedding: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if embedding.ndim != 2 or embedding.shape[0] < 3 or unique.size < 2:
        return float("nan")
    centers = {lab: embedding[labels == lab].mean(axis=0) for lab in unique}
    diffs = []
    for point, lab in zip(embedding, labels):
        own = np.linalg.norm(point - centers[lab])
        other = min(np.linalg.norm(point - centers[o]) for o in unique if o != lab)
        denom = max(own, other, 1e-8)
        diffs.append((other - own) / denom)
    return float(np.mean(diffs))


def load_umap_pickle(path: Path) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with path.open("rb") as f:
            obj = FlyvisCompatUnpickler(f).load()
    embedding = np.asarray(obj.embedding._embedding)
    labels = np.asarray(obj.labels)
    n_clusters = int(np.unique(labels).size)
    return {
        "cell_type": path.stem,
        "embedding": embedding,
        "labels": labels,
        "n_clusters": n_clusters,
        "scores": list(getattr(obj, "scores", [])),
        "selected_n_clusters": np.asarray(getattr(obj, "n_clusters", np.array([n_clusters]))).tolist(),
    }


def analyze_umap_pickles() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(UMAP_DIR.glob("*.pickle")):
        record = load_umap_pickle(path)
        emb = record["embedding"]
        labels = record["labels"]
        rows.append(UMAPSummary(
            cell_type=record["cell_type"],
            n_models=int(emb.shape[0]),
            n_clusters=record["n_clusters"],
            embedding_x_mean=float(np.mean(emb[:, 0])),
            embedding_y_mean=float(np.mean(emb[:, 1])),
            embedding_x_std=float(np.std(emb[:, 0])),
            embedding_y_std=float(np.std(emb[:, 1])),
            silhouette_proxy=silhouette_proxy(emb, labels),
            cluster_entropy=shannon_entropy(labels),
        ).__dict__)
    return pd.DataFrame(rows).sort_values(["n_clusters", "cluster_entropy", "cell_type"], ascending=[False, False, True])


def parse_related_work() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pdf_path in sorted(RELATED_WORK_DIR.glob("*.pdf")):
        row: Dict[str, Any] = {
            "file": pdf_path.name,
            "size_bytes": pdf_path.stat().st_size,
            "title": None,
            "author": None,
            "year": None,
            "topic_hint": None,
            "preview": None,
        }
        if PdfReader is not None:
            try:
                reader = PdfReader(str(pdf_path))
                meta = reader.metadata or {}
                title = str(meta.get("/Title") or "").strip() or None
                author = str(meta.get("/Author") or "").strip() or None
                preview = ""
                for page in reader.pages[:2]:
                    preview += (page.extract_text() or "") + " "
                preview = re.sub(r"\s+", " ", preview).strip()
                row["title"] = title
                row["author"] = author
                row["preview"] = preview[:500] if preview else None
                year_match = re.search(r"(19|20)\d{2}", (title or "") + " " + preview)
                if year_match:
                    row["year"] = int(year_match.group(0))
                lowered = preview.lower()
                if "motion" in lowered and "pathway" in lowered:
                    row["topic_hint"] = "motion pathways"
                elif "lobula plate" in lowered:
                    row["topic_hint"] = "lobula plate integration"
                elif "wiring diagram" in lowered or "parts list" in lowered:
                    row["topic_hint"] = "cell-type catalog / wiring diagram"
                elif "synaptic circuits" in lowered or "columns" in lowered:
                    row["topic_hint"] = "columnar circuit variation"
                elif "wiring economy" in lowered:
                    row["topic_hint"] = "spatial wiring constraints"
            except Exception as exc:
                row["preview"] = f"PDF parsing failed: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_models() -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    config_records: List[Dict[str, Any]] = []
    chkpt_stats = None
    for model_dir in MODEL_DIRS:
        meta = read_yaml(model_dir / "_meta.yaml")
        config_summary = extract_core_config(meta)
        config_records.append(config_summary)
        validation_loss = read_validation_loss(model_dir / "validation" / "loss.h5")
        top_level_loss = read_validation_loss(model_dir / "validation_loss.h5")
        chkpt_path = model_dir / "best_chkpt"
        if chkpt_stats is None:
            chkpt_stats = summarize_checkpoint_zip(chkpt_path)
        rows.append({
            "model_id": model_dir.name,
            "validation_loss": validation_loss,
            "top_level_validation_loss": top_level_loss,
            "loss_difference": top_level_loss - validation_loss,
            "checkpoint_size_bytes": chkpt_path.stat().st_size,
            **config_summary,
        })
    df = pd.DataFrame(rows).sort_values("validation_loss").reset_index(drop=True)
    cfg_df = pd.DataFrame(config_records)
    consistency = {
        col: int(cfg_df[col].astype(str).nunique()) for col in cfg_df.columns
    }
    overview = {
        "n_models": int(len(df)),
        "validation_loss_min": float(df["validation_loss"].min()),
        "validation_loss_median": float(df["validation_loss"].median()),
        "validation_loss_mean": float(df["validation_loss"].mean()),
        "validation_loss_max": float(df["validation_loss"].max()),
        "validation_loss_std": float(df["validation_loss"].std(ddof=1)),
        "best_model_id": str(df.iloc[0]["model_id"]),
        "worst_model_id": str(df.iloc[-1]["model_id"]),
        "checkpoint_template_summary": chkpt_stats,
        "config_field_unique_counts": consistency,
    }
    return df, overview


def make_validation_figures(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].hist(df["validation_loss"], bins=12, color="#4C78A8", edgecolor="black", alpha=0.85)
    axes[0].set_title("Ensemble validation-loss distribution")
    axes[0].set_xlabel("Validation loss")
    axes[0].set_ylabel("Model count")

    rank_df = df.sort_values("validation_loss").reset_index(drop=True)
    axes[1].plot(np.arange(1, len(rank_df) + 1), rank_df["validation_loss"], marker="o", linewidth=1.5, color="#F58518")
    axes[1].axhline(rank_df["validation_loss"].median(), linestyle="--", color="gray", linewidth=1, label="Median")
    axes[1].set_title("Validation-loss ranking across 50 pretrained DMNs")
    axes[1].set_xlabel("Rank (best to worst)")
    axes[1].set_ylabel("Validation loss")
    axes[1].legend(frameon=False)

    fig.savefig(FIG_DIR / "validation_loss_overview.png", dpi=200)
    plt.close(fig)


def make_umap_summary_figures(umap_df: pd.DataFrame) -> None:
    top = umap_df.head(15).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    order = umap_df.sort_values("n_clusters", ascending=False).head(20)
    axes[0].barh(order["cell_type"], order["n_clusters"], color="#54A24B")
    axes[0].invert_yaxis()
    axes[0].set_title("Cell types with the richest ensemble clustering")
    axes[0].set_xlabel("Number of clusters across 50 models")

    sc = axes[1].scatter(
        umap_df["cluster_entropy"],
        umap_df["silhouette_proxy"],
        c=umap_df["n_clusters"],
        cmap="viridis",
        s=50,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
    )
    for _, row in top.iterrows():
        axes[1].annotate(row["cell_type"], (row["cluster_entropy"], row["silhouette_proxy"]), fontsize=7, alpha=0.85)
    axes[1].set_title("Embedding diversity versus separability")
    axes[1].set_xlabel("Cluster entropy (bits)")
    axes[1].set_ylabel("Silhouette-like separation proxy")
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label("Cluster count")

    fig.savefig(FIG_DIR / "umap_cluster_summary.png", dpi=200)
    plt.close(fig)


def make_example_umap_panel() -> None:
    pickle_paths = sorted(UMAP_DIR.glob("*.pickle"))[:9]
    n = len(pickle_paths)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, path in zip(axes, pickle_paths):
        record = load_umap_pickle(path)
        emb = record["embedding"]
        labels = record["labels"]
        ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=24, alpha=0.9)
        ax.set_title(f"{record['cell_type']} (k={record['n_clusters']})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Representative ensemble embedding structure across cell types", fontsize=14)
    fig.savefig(FIG_DIR / "example_umap_panels.png", dpi=200)
    plt.close(fig)


def make_task_config_figure(overview: Dict[str, Any], model_df: pd.DataFrame) -> None:
    keys = [
        "n_models",
        "validation_loss_min",
        "validation_loss_median",
        "validation_loss_max",
    ]
    vals = [overview[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar([k.replace("validation_loss_", "loss_") for k in keys], vals, color=["#72B7B2", "#72B7B2", "#72B7B2", "#E45756"])
    axes[0].set_title("Ensemble size and loss summary")
    axes[0].tick_params(axis="x", rotation=20)

    trainable = pd.Series({
        "bias": bool(model_df["bias_trainable"].iloc[0]),
        "time_const": bool(model_df["time_const_trainable"].iloc[0]),
        "syn_strength": bool(model_df["syn_strength_trainable"].iloc[0]),
        "syn_count": bool(model_df["syn_count_trainable"].iloc[0]),
        "sign": bool(model_df["sign_trainable"].iloc[0]),
    })
    axes[1].bar(trainable.index, trainable.astype(int), color="#B279A2")
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title("Which connectome-derived parameters are optimized?")
    axes[1].set_ylabel("Trainable (1=yes)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.savefig(FIG_DIR / "task_config_overview.png", dpi=200)
    plt.close(fig)


def make_text_summary(model_df: pd.DataFrame, overview: Dict[str, Any], umap_df: pd.DataFrame, papers_df: pd.DataFrame) -> str:
    best = model_df.iloc[0]
    worst = model_df.iloc[-1]
    top_umap = umap_df.sort_values(["n_clusters", "cluster_entropy"], ascending=[False, False]).head(5)
    lines = [
        "Drosophila optic-lobe DMN ensemble analysis summary",
        "=" * 52,
        f"Models analyzed: {overview['n_models']}",
        f"Validation loss range: {overview['validation_loss_min']:.6f} to {overview['validation_loss_max']:.6f}",
        f"Validation loss mean ± SD: {overview['validation_loss_mean']:.6f} ± {overview['validation_loss_std']:.6f}",
        f"Best model: {best['model_id']} (loss={best['validation_loss']:.6f})",
        f"Worst model: {worst['model_id']} (loss={worst['validation_loss']:.6f})",
        "",
        "Architecture/task metadata inferred from configs:",
        f"- Connectome file: {best['connectome_file']}",
        f"- Connectome extent: {best['extent']}",
        f"- Dynamics: {best['dynamics_type']} with {best['activation']} activation",
        f"- Dataset: {best['dataset_type']} with {best['n_frames']} frames and dt={best['dt']}",
        f"- Task: {best['tasks']}",
        f"- Decoder: {best['decoder_type']} with shape={best['decoder_shape']} kernel={best['decoder_kernel_size']}",
        f"- Trainable parameter families: bias={best['bias_trainable']}, time_const={best['time_const_trainable']}, syn_strength={best['syn_strength_trainable']}, syn_count={best['syn_count_trainable']}, sign={best['sign_trainable']}",
        "",
        "Most heterogeneous cell-type embeddings (ensemble variability / multimodality):",
    ]
    for _, row in top_umap.iterrows():
        lines.append(
            f"- {row['cell_type']}: clusters={int(row['n_clusters'])}, entropy={row['cluster_entropy']:.3f}, separation_proxy={row['silhouette_proxy']:.3f}"
        )
    lines += [
        "",
        "Related-work papers parsed:",
    ]
    for _, row in papers_df.iterrows():
        lines.append(f"- {row['file']}: {row['title']} [{row['topic_hint']}]")
    lines += [
        "",
        "Generated figures:",
        "- report/images/validation_loss_overview.png",
        "- report/images/task_config_overview.png",
        "- report/images/umap_cluster_summary.png",
        "- report/images/example_umap_panels.png",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()

    model_df, overview = analyze_models()
    umap_df = analyze_umap_pickles()
    papers_df = parse_related_work()

    model_df.to_csv(OUTPUT_DIR / "model_ensemble_summary.csv", index=False)
    umap_df.to_csv(OUTPUT_DIR / "umap_cluster_summary.csv", index=False)
    papers_df.to_csv(OUTPUT_DIR / "related_work_summary.csv", index=False)
    save_json(OUTPUT_DIR / "ensemble_overview.json", overview)

    summary_text = make_text_summary(model_df, overview, umap_df, papers_df)
    (OUTPUT_DIR / "analysis_summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    make_validation_figures(model_df)
    make_task_config_figure(overview, model_df)
    make_umap_summary_figures(umap_df)
    make_example_umap_panel()

    manifest = {
        "outputs": [
            "outputs/model_ensemble_summary.csv",
            "outputs/umap_cluster_summary.csv",
            "outputs/related_work_summary.csv",
            "outputs/ensemble_overview.json",
            "outputs/analysis_summary.txt",
        ],
        "figures": [
            "report/images/validation_loss_overview.png",
            "report/images/task_config_overview.png",
            "report/images/umap_cluster_summary.png",
            "report/images/example_umap_panels.png",
        ],
    }
    save_json(OUTPUT_DIR / "analysis_manifest.json", manifest)

    print("Analysis script created. Run with: python code/run_analysis.py")


if __name__ == "__main__":
    main()
