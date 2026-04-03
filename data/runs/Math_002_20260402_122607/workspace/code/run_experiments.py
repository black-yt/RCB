from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mapf_core import (
    MAPFTask,
    collision_count,
    independent_shortest_paths,
    lns_solve,
    marl_initialize_paths,
    prioritized_planning,
    sample_task,
    train_shared_q,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
REPORT_IMG_DIR = WORKSPACE / "report" / "images"

GLOBAL_SEED = 7
ALL_FAMILIES = [
    "maps_60_10_10_0.175",
    "empty",
    "maze",
    "random_small",
    "random_medium",
    "random_large",
    "room",
    "warehouse",
]
EVAL_FAMILIES = [
    "empty",
    "maze",
    "random_small",
    "random_medium",
    "random_large",
    "warehouse",
]
FAMILY_AGENT_COUNTS = {
    "empty": [20, 32],
    "maze": [10, 16],
    "random_small": [6, 9],
    "random_medium": [16, 24],
    "random_large": [20],
    "warehouse": [14, 22],
}
FAMILY_MAP_LIMITS = {
    "empty": 4,
    "maze": 4,
    "random_small": 4,
    "random_medium": 4,
    "random_large": 2,
    "warehouse": 4,
}
TRAIN_EPISODES = 700
TRAIN_HORIZON = 40


def family_dirs() -> Dict[str, List[Path]]:
    mapping = {}
    for family in ALL_FAMILIES:
        root = DATA_DIR / family
        if any(root.glob("*.npy")):
            mapping[family] = [root]
        else:
            mapping[family] = sorted([p for p in root.iterdir() if p.is_dir()])
    return mapping


def load_maps() -> pd.DataFrame:
    rows = []
    for family, dirs in family_dirs().items():
        for folder in dirs:
            for path in sorted(folder.glob("*.npy")):
                grid = np.load(path)
                rows.append(
                    {
                        "family": family,
                        "folder": folder.name,
                        "map_name": path.stem,
                        "path": str(path),
                        "height": int(grid.shape[0]),
                        "width": int(grid.shape[1]),
                        "free_cells": int((grid == 0).sum()),
                        "obstacles": int((grid == -1).sum()),
                        "obstacle_density": float((grid == -1).mean()),
                    }
                )
    return pd.DataFrame(rows)


def choose_maps(meta: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(GLOBAL_SEED)
    chosen_rows = []
    for family in EVAL_FAMILIES:
        subset = meta[meta["family"] == family].copy()
        indices = sorted(subset.index.tolist())
        picks = rng.sample(indices, FAMILY_MAP_LIMITS[family])
        chosen_rows.append(subset.loc[picks])
    return pd.concat(chosen_rows, ignore_index=True)


def build_train_tasks(meta: pd.DataFrame) -> List[MAPFTask]:
    rng = random.Random(GLOBAL_SEED + 100)
    train_meta = pd.concat(
        [
            meta[meta["family"] == "random_small"].head(12),
            meta[meta["family"] == "random_medium"].head(12),
            meta[meta["family"] == "warehouse"].head(12),
        ],
        ignore_index=True,
    )
    tasks = []
    for idx, row in train_meta.iterrows():
        grid = np.load(row["path"])
        agent_count = min(FAMILY_AGENT_COUNTS[row["family"]][0], max(4, row["free_cells"] // 12))
        task = sample_task(
            grid=grid,
            agent_count=agent_count,
            rng=random.Random(rng.randint(0, 10**9)),
            map_name=row["map_name"],
            family=row["family"],
            seed=GLOBAL_SEED + idx,
        )
        tasks.append(task)
    return tasks


def build_eval_tasks(selected_meta: pd.DataFrame) -> List[MAPFTask]:
    tasks = []
    for idx, row in selected_meta.iterrows():
        grid = np.load(row["path"])
        for agent_count in FAMILY_AGENT_COUNTS[row["family"]]:
            feasible_agents = min(agent_count, max(4, row["free_cells"] // 3))
            task = sample_task(
                grid=grid,
                agent_count=feasible_agents,
                rng=random.Random(GLOBAL_SEED * 1000 + idx * 31 + feasible_agents),
                map_name=row["map_name"],
                family=row["family"],
                seed=GLOBAL_SEED * 1000 + idx * 31 + feasible_agents,
            )
            tasks.append(task)
    return tasks


def evaluate_tasks(tasks: Sequence[MAPFTask], learner) -> pd.DataFrame:
    rows = []
    for idx, task in enumerate(tasks, start=1):
        print(f"[eval] {idx}/{len(tasks)} {task.family} {task.map_name} agents={len(task.starts)}", flush=True)
        order = sorted(range(len(task.starts)), key=lambda a: abs(task.starts[a][0] - task.goals[a][0]) + abs(task.starts[a][1] - task.goals[a][1]), reverse=True)
        base_rng = random.Random(task.seed)

        pp = prioritized_planning(task, order=order, restart_budget=1, rng=random.Random(base_rng.randint(0, 10**9)))
        pp.method = "PP"
        pp_rr = prioritized_planning(task, order=order, restart_budget=4, rng=random.Random(base_rng.randint(0, 10**9)))
        pp_rr.method = "PP-RR"

        init_lns = independent_shortest_paths(task)
        lns = lns_solve(
            task,
            init_paths=init_lns,
            rng=random.Random(base_rng.randint(0, 10**9)),
            learner=None,
            max_iterations=24,
            neighborhood_size=min(8, max(4, len(task.starts) // 4)),
        )
        lns.method = "LNS-PP"

        init_hybrid = marl_initialize_paths(
            task,
            learner=learner,
            horizon=max(12, int(np.sqrt(task.grid.size))),
            rng=random.Random(base_rng.randint(0, 10**9)),
        )
        if (
            collision_count(init_hybrid) > collision_count(init_lns)
            or (
                collision_count(init_hybrid) == collision_count(init_lns)
                and sum(len(p) for p in init_hybrid) > sum(len(p) for p in init_lns)
            )
        ):
            init_hybrid = init_lns
        hybrid = lns_solve(
            task,
            init_paths=init_hybrid,
            rng=random.Random(base_rng.randint(0, 10**9)),
            learner=learner,
            max_iterations=24,
            neighborhood_size=min(8, max(4, len(task.starts) // 4)),
        )
        hybrid.method = "MARL-LNS-PP"

        for result in [pp, pp_rr, lns, hybrid]:
            rows.append(
                {
                    "family": task.family,
                    "map_name": task.map_name,
                    "agents": len(task.starts),
                    "seed": task.seed,
                    "method": result.method,
                    "success": int(result.success),
                    "runtime_sec": result.runtime,
                    "sum_of_costs": result.sum_of_costs,
                    "makespan": result.makespan,
                    "final_collisions": result.collision_count,
                    "iterations": result.iterations,
                }
            )
    return pd.DataFrame(rows)


def plot_dataset_overview(meta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    family_stats = meta.groupby("family").agg(
        maps=("map_name", "count"),
        mean_density=("obstacle_density", "mean"),
        mean_free=("free_cells", "mean"),
    ).reset_index()
    sns.barplot(data=family_stats, x="family", y="maps", ax=axes[0], color="#4c78a8")
    axes[0].set_title("Dataset Size by Family")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(data=family_stats, x="family", y="mean_density", ax=axes[1], color="#f58518")
    axes[1].set_title("Mean Obstacle Density")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "dataset_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results(results: pd.DataFrame) -> None:
    summary = (
        results.groupby(["family", "method", "agents"])
        .agg(
            success_rate=("success", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            sum_of_costs=("sum_of_costs", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.barplot(data=summary, x="family", y="success_rate", hue="method", ax=axes[0])
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(data=summary, x="family", y="runtime_sec", hue="method", ax=axes[1])
    axes[1].set_title("Mean Runtime (s)")
    axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(data=summary, x="family", y="sum_of_costs", hue="method", ax=axes[2])
    axes[2].set_title("Mean Sum of Costs")
    axes[2].tick_params(axis="x", rotation=35)

    handles, labels = axes[2].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(REPORT_IMG_DIR / "main_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    by_agents = (
        results.groupby(["agents", "method"])
        .agg(success_rate=("success", "mean"), runtime_sec=("runtime_sec", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.lineplot(data=by_agents, x="agents", y="success_rate", hue="method", marker="o", ax=axes[0])
    axes[0].set_title("Success vs Agent Count")
    axes[0].set_ylim(0, 1.05)

    sns.lineplot(data=by_agents, x="agents", y="runtime_sec", hue="method", marker="o", ax=axes[1])
    axes[1].set_title("Runtime vs Agent Count")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "scaling_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pairwise = results.pivot_table(
        index=["family", "map_name", "agents", "seed"],
        columns="method",
        values="success",
    ).reset_index()
    if {"LNS-PP", "MARL-LNS-PP"}.issubset(pairwise.columns):
        pairwise["delta_success"] = pairwise["MARL-LNS-PP"] - pairwise["LNS-PP"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(pairwise["delta_success"], bins=[-1.5, -0.5, 0.5, 1.5], ax=ax, color="#54a24b")
        ax.set_title("Hybrid vs LNS-PP Per-Instance Success Delta")
        ax.set_xlabel("MARL-LNS-PP success minus LNS-PP success")
        fig.tight_layout()
        fig.savefig(REPORT_IMG_DIR / "validation_delta.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_summary(meta: pd.DataFrame, selected_meta: pd.DataFrame, results: pd.DataFrame) -> None:
    dataset_summary = meta.groupby("family").agg(
        maps=("map_name", "count"),
        height=("height", "median"),
        width=("width", "median"),
        free_cells=("free_cells", "mean"),
        density=("obstacle_density", "mean"),
    )
    method_summary = results.groupby("method").agg(
        success_rate=("success", "mean"),
        runtime_sec=("runtime_sec", "mean"),
        sum_of_costs=("sum_of_costs", "mean"),
        makespan=("makespan", "mean"),
        final_collisions=("final_collisions", "mean"),
    )
    family_method_summary = results.groupby(["family", "method"]).agg(
        success_rate=("success", "mean"),
        runtime_sec=("runtime_sec", "mean"),
        sum_of_costs=("sum_of_costs", "mean"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_summary.round(4).to_csv(OUTPUT_DIR / "dataset_summary.csv")
    selected_meta.sort_values(["family", "map_name"]).to_csv(OUTPUT_DIR / "selected_maps.csv", index=False)
    results.sort_values(["family", "map_name", "agents", "method"]).to_csv(OUTPUT_DIR / "experiment_results.csv", index=False)
    method_summary.round(4).to_csv(OUTPUT_DIR / "method_summary.csv")
    family_method_summary.round(4).to_csv(OUTPUT_DIR / "family_method_summary.csv")

    payload = {
        "global_seed": GLOBAL_SEED,
        "train_episodes": TRAIN_EPISODES,
        "train_horizon": TRAIN_HORIZON,
        "family_map_limits": FAMILY_MAP_LIMITS,
        "families": EVAL_FAMILIES,
        "family_agent_counts": FAMILY_AGENT_COUNTS,
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_maps()
    print(f"[setup] loaded {len(meta)} maps across {meta['family'].nunique()} families", flush=True)
    selected_meta = choose_maps(meta)
    print(f"[setup] selected {len(selected_meta)} evaluation maps", flush=True)
    train_tasks = build_train_tasks(meta)
    print(f"[train] training shared Q on {len(train_tasks)} tasks", flush=True)
    learner = train_shared_q(
        train_tasks=train_tasks,
        episodes=TRAIN_EPISODES,
        horizon=TRAIN_HORIZON,
        seed=GLOBAL_SEED,
    )
    print("[train] training complete", flush=True)
    eval_tasks = build_eval_tasks(selected_meta)
    print(f"[eval] evaluating {len(eval_tasks)} generated MAPF tasks", flush=True)
    results = evaluate_tasks(eval_tasks, learner)

    write_summary(meta, selected_meta, results)
    plot_dataset_overview(meta)
    plot_results(results)

    method_summary = results.groupby("method").agg(
        success_rate=("success", "mean"),
        runtime_sec=("runtime_sec", "mean"),
        sum_of_costs=("sum_of_costs", "mean"),
    )
    print(method_summary.round(4).to_string())


if __name__ == "__main__":
    main()
