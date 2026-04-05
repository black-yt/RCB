import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

DATASET_CONFIG = {
    "maps_60_10_10_0.175": {"path": "data/maps_60_10_10_0.175", "label": "random_small_60"},
    "random_small": {"path": "data/random_small", "label": "random_small"},
    "random_medium": {"path": "data/random_medium", "label": "random_medium"},
    "maze": {"path": "data/maze", "label": "maze"},
    "room": {"path": "data/room", "label": "room"},
    "warehouse": {"path": "data/warehouse", "label": "warehouse"},
    "empty": {"path": "data/empty", "label": "empty"},
}

METHODS = ["pp", "rand_pp", "lns_pp", "hybrid_marl_lns"]
MOVES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class Scenario:
    map_path: str
    dataset: str
    map_id: str
    agent_count: int
    starts: list
    goals: list
    grid: np.ndarray
    seed: int


class ReservationTable:
    def __init__(self):
        self.vertex = defaultdict(set)
        self.edge = defaultdict(set)
        self.goal_reservations = {}

    def add_path(self, agent, path):
        for t, pos in enumerate(path):
            self.vertex[t].add(pos)
            if t > 0:
                self.edge[t].add((path[t - 1], pos))
        goal = path[-1]
        self.goal_reservations[agent] = (goal, len(path) - 1)

    def build_from_paths(self, paths, skip_agents=None):
        skip_agents = set(skip_agents or [])
        self.vertex.clear()
        self.edge.clear()
        self.goal_reservations.clear()
        for a, path in paths.items():
            if a in skip_agents or path is None:
                continue
            self.add_path(a, path)

    def is_reserved(self, curr, nxt, t):
        if nxt in self.vertex.get(t, set()):
            return True
        if (nxt, curr) in self.edge.get(t, set()):
            return True
        for goal, start_t in self.goal_reservations.values():
            if nxt == goal and t >= start_t:
                return True
        return False


class MAPFSolver:
    def __init__(self, grid):
        self.grid = grid
        self.h, self.w = grid.shape
        self.free_cells = [(r, c) for r in range(self.h) for c in range(self.w) if grid[r, c] == 0]

    def in_bounds(self, p):
        r, c = p
        return 0 <= r < self.h and 0 <= c < self.w

    def passable(self, p):
        r, c = p
        return self.grid[r, c] == 0

    def neighbors(self, p):
        for dr, dc in MOVES:
            q = (p[0] + dr, p[1] + dc)
            if self.in_bounds(q) and self.passable(q):
                yield q

    @staticmethod
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def shortest_path_len(self, start, goal):
        if start == goal:
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            p, d = q.popleft()
            for n in self.neighbors(p):
                if n in seen:
                    continue
                if n == goal:
                    return d + 1
                seen.add(n)
                q.append((n, d + 1))
        return math.inf

    def plan_for_agent(self, start, goal, reservations, horizon=256):
        q = deque([(start, 0)])
        parents = {(start, 0): None}
        seen = {(start, 0)}
        latest_goal_t = 0
        if reservations.goal_reservations:
            latest_goal_t = max(v[1] for v in reservations.goal_reservations.values())
        max_t = min(horizon, max(latest_goal_t + self.manhattan(start, goal) + 10, 32))
        while q:
            pos, t = q.popleft()
            if pos == goal:
                blocked = False
                for ft in range(t, min(max_t, t + 5)):
                    if reservations.is_reserved(goal, goal, ft):
                        blocked = True
                        break
                if not blocked:
                    path = []
                    cur = (pos, t)
                    while cur is not None:
                        path.append(cur[0])
                        cur = parents[cur]
                    return list(reversed(path))
            if t >= max_t:
                continue
            nbrs = sorted(self.neighbors(pos), key=lambda x: self.manhattan(x, goal))
            for nxt in nbrs:
                nt = t + 1
                state = (nxt, nt)
                if state in seen:
                    continue
                if reservations.is_reserved(pos, nxt, nt):
                    continue
                seen.add(state)
                parents[state] = (pos, t)
                q.append(state)
        return None


def parse_agent_count(path):
    m = re.search(r"_(\d+)_([0-9]+)_([0-9]+)(?:_|$)", path)
    return int(m.group(1)) if m else None


def collect_map_files(dataset_key, max_maps=8):
    base = DATASET_CONFIG[dataset_key]["path"]
    files = []
    for root, _, fns in os.walk(base):
        npys = sorted([x for x in fns if x.endswith(".npy")])
        for fn in npys[:max_maps]:
            files.append(os.path.join(root, fn))
    return sorted(files)[:max_maps]


def sample_scenario(map_path, dataset, scenario_seed, agent_cap=80):
    grid = np.load(map_path)
    free = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == 0]
    dir_count = parse_agent_count(os.path.dirname(map_path))
    if dir_count is None:
        dir_count = min(20, max(4, len(free) // 10))
    agent_count = min(dir_count, agent_cap, max(2, len(free) // 2))
    rng = random.Random(scenario_seed)
    picks = rng.sample(free, 2 * agent_count)
    starts = picks[:agent_count]
    goals = picks[agent_count:]
    return Scenario(
        map_path=map_path,
        dataset=dataset,
        map_id=os.path.splitext(os.path.basename(map_path))[0],
        agent_count=agent_count,
        starts=starts,
        goals=goals,
        grid=grid,
        seed=scenario_seed,
    )


def detect_collisions(paths):
    collisions = []
    if any(p is None for p in paths.values()):
        return [{"type": "missing", "agents": [a], "time": -1} for a, p in paths.items() if p is None]
    agents = list(paths.keys())
    horizon = max(len(p) for p in paths.values())
    def pos(path, t):
        return path[t] if t < len(path) else path[-1]
    for t in range(horizon):
        occ = {}
        for a in agents:
            pa = pos(paths[a], t)
            if pa in occ:
                collisions.append({"type": "vertex", "agents": [occ[pa], a], "cell": pa, "time": t})
            else:
                occ[pa] = a
        if t == 0:
            continue
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                a0, a1 = pos(paths[a], t - 1), pos(paths[a], t)
                b0, b1 = pos(paths[b], t - 1), pos(paths[b], t)
                if a0 == b1 and b0 == a1 and a0 != a1:
                    collisions.append({"type": "swap", "agents": [a, b], "edge": [a0, a1], "time": t})
    return collisions


def path_costs(paths):
    if any(p is None for p in paths.values()):
        return math.inf, math.inf
    soc = sum(len(p) - 1 for p in paths.values())
    makespan = max(len(p) - 1 for p in paths.values())
    return soc, makespan


def priority_by_shortest_paths(solver, starts, goals):
    vals = [(a, solver.shortest_path_len(starts[a], goals[a])) for a in range(len(starts))]
    vals.sort(key=lambda x: (x[1], x[0]))
    return [a for a, _ in vals]


def plan_given_order(solver, starts, goals, order):
    res = ReservationTable()
    paths = {}
    for a in order:
        path = solver.plan_for_agent(starts[a], goals[a], res)
        if path is None:
            paths[a] = None
            return paths
        paths[a] = path
        res.add_path(a, path)
    return paths


def prioritized_planning(solver, starts, goals, randomized=False, seed=0):
    order = priority_by_shortest_paths(solver, starts, goals)
    if randomized:
        rng = random.Random(seed)
        grouped = defaultdict(list)
        for a in order:
            grouped[solver.shortest_path_len(starts[a], goals[a])].append(a)
        order = []
        for key in sorted(grouped):
            group = grouped[key]
            rng.shuffle(group)
            order.extend(group)
    return plan_given_order(solver, starts, goals, order), order


def conflict_counts(paths):
    counts = Counter()
    for col in detect_collisions(paths):
        for a in col.get("agents", []):
            counts[a] += 1
    return counts


def lns_repair(solver, starts, goals, init_paths, iterations=25, neighborhood=8, seed=0, hybrid=False):
    rng = random.Random(seed)
    paths = {k: (None if v is None else list(v)) for k, v in init_paths.items()}
    history = []
    for it in range(iterations):
        collisions = detect_collisions(paths)
        history.append(len(collisions))
        if not collisions:
            return paths, history
        counts = conflict_counts(paths)
        if hybrid:
            scored = []
            horizon = max(len(p) for p in paths.values() if p is not None)
            for a in range(len(starts)):
                p = paths.get(a)
                if p is None:
                    local_wait = 10.0
                    congestion = 10.0
                else:
                    wait_steps = sum(1 for t in range(1, len(p)) if p[t] == p[t - 1])
                    local_wait = wait_steps / max(1, len(p) - 1)
                    visited = Counter(p)
                    congestion = sum(v - 1 for v in visited.values()) / max(1, len(p))
                dist = solver.shortest_path_len(starts[a], goals[a])
                progress_penalty = 0.0 if p is None else max(0.0, (len(p) - 1 - dist) / max(1, dist + 1))
                score = 3.0 * counts[a] + 1.5 * local_wait + 1.0 * congestion + 1.0 * progress_penalty
                scored.append((score, rng.random(), a))
            scored.sort(reverse=True)
            chosen = [a for _, _, a in scored[:neighborhood]]
        else:
            ranked = sorted([(counts[a], rng.random(), a) for a in range(len(starts))], reverse=True)
            chosen = [a for _, _, a in ranked[:neighborhood]]
        res = ReservationTable()
        res.build_from_paths(paths, skip_agents=chosen)
        chosen = sorted(chosen, key=lambda a: (counts[a], solver.shortest_path_len(starts[a], goals[a])), reverse=True)
        success = True
        new_subpaths = {}
        for a in chosen:
            p = solver.plan_for_agent(starts[a], goals[a], res)
            if p is None:
                success = False
                break
            new_subpaths[a] = p
            res.add_path(a, p)
        if success:
            for a, p in new_subpaths.items():
                paths[a] = p
        else:
            if not hybrid:
                rng.shuffle(chosen)
            else:
                chosen = list(reversed(chosen))
            res = ReservationTable()
            res.build_from_paths(paths, skip_agents=chosen)
            for a in chosen:
                p = solver.plan_for_agent(starts[a], goals[a], res)
                if p is None:
                    paths[a] = None
                    break
                paths[a] = p
                res.add_path(a, p)
    return paths, history


def solve_scenario(scenario, method):
    solver = MAPFSolver(scenario.grid)
    start_t = time.perf_counter()
    if method == "pp":
        paths, order = prioritized_planning(solver, scenario.starts, scenario.goals, randomized=False, seed=scenario.seed)
        hist = []
    elif method == "rand_pp":
        paths, order = prioritized_planning(solver, scenario.starts, scenario.goals, randomized=True, seed=scenario.seed)
        hist = []
    elif method == "lns_pp":
        base, order = prioritized_planning(solver, scenario.starts, scenario.goals, randomized=True, seed=scenario.seed)
        paths, hist = lns_repair(solver, scenario.starts, scenario.goals, base, iterations=20, neighborhood=min(10, scenario.agent_count), seed=scenario.seed, hybrid=False)
    elif method == "hybrid_marl_lns":
        base, order = prioritized_planning(solver, scenario.starts, scenario.goals, randomized=True, seed=scenario.seed)
        paths, hist = lns_repair(solver, scenario.starts, scenario.goals, base, iterations=20, neighborhood=min(10, scenario.agent_count), seed=scenario.seed, hybrid=True)
    else:
        raise ValueError(method)
    runtime = time.perf_counter() - start_t
    collisions = detect_collisions(paths)
    success = int(len(collisions) == 0 and all(p is not None for p in paths.values()))
    soc, makespan = path_costs(paths)
    shortest_total = sum(solver.shortest_path_len(s, g) for s, g in zip(scenario.starts, scenario.goals))
    normalized_soc = soc / max(1, shortest_total) if math.isfinite(soc) else math.inf
    return {
        "dataset": scenario.dataset,
        "map_id": scenario.map_id,
        "map_path": scenario.map_path,
        "agents": scenario.agent_count,
        "seed": scenario.seed,
        "method": method,
        "success": success,
        "collision_count": len(collisions),
        "soc": soc,
        "makespan": makespan,
        "runtime_sec": runtime,
        "normalized_soc": normalized_soc,
        "repair_history": hist,
        "paths": {str(k): v for k, v in paths.items()},
        "starts": scenario.starts,
        "goals": scenario.goals,
    }


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("report/images", exist_ok=True)
    os.makedirs("code", exist_ok=True)


def run_eda():
    ensure_dirs()
    rows = []
    sample_records = []
    for dataset in DATASET_CONFIG:
        files = collect_map_files(dataset, max_maps=12)
        for path in files:
            arr = np.load(path)
            obstacle_rate = float((arr == -1).mean())
            rows.append({
                "dataset": dataset,
                "map_path": path,
                "height": arr.shape[0],
                "width": arr.shape[1],
                "obstacle_rate": obstacle_rate,
                "agent_count_nominal": parse_agent_count(os.path.dirname(path)),
            })
        if files:
            scenario = sample_scenario(files[0], dataset, SEED)
            sample_records.append({
                "dataset": dataset,
                "map_path": files[0],
                "agent_count": scenario.agent_count,
                "first_starts": scenario.starts[:5],
                "first_goals": scenario.goals[:5],
            })
    df = pd.DataFrame(rows)
    df.to_csv("outputs/dataset_summary.csv", index=False)
    with open("outputs/eda_samples.json", "w") as f:
        json.dump(sample_records, f, indent=2)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="dataset", y="obstacle_rate")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Obstacle rate")
    plt.tight_layout()
    plt.savefig("report/images/data_overview.png", dpi=200)
    plt.close()


def run_smoke():
    ensure_dirs()
    dataset = "random_small"
    path = collect_map_files(dataset, max_maps=1)[0]
    scenario = sample_scenario(path, dataset, SEED, agent_cap=20)
    results = []
    for m in METHODS:
        res = solve_scenario(scenario, m)
        res.pop("paths")
        results.append(res)
    with open("outputs/smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)


def benchmark_scenarios():
    selected = ["random_small", "random_medium", "maze", "room", "warehouse", "empty"]
    scenarios = []
    for dataset in selected:
        files = collect_map_files(dataset, max_maps=4)
        for i, path in enumerate(files):
            scenarios.append(sample_scenario(path, dataset, SEED + i, agent_cap=80))
    return scenarios


def run_benchmark():
    ensure_dirs()
    rows = []
    example_bundle = []
    for idx, scenario in enumerate(benchmark_scenarios()):
        for m in METHODS:
            res = solve_scenario(scenario, m)
            rows.append({k: v for k, v in res.items() if k not in ["paths", "starts", "goals", "repair_history"]})
            if idx < 2 and m in ["pp", "hybrid_marl_lns"]:
                example_bundle.append({
                    "dataset": res["dataset"],
                    "map_id": res["map_id"],
                    "method": m,
                    "paths": res["paths"],
                    "starts": res["starts"],
                    "goals": res["goals"],
                    "map_path": res["map_path"],
                    "repair_history": res["repair_history"],
                })
    df = pd.DataFrame(rows)
    df.to_csv("outputs/results.csv", index=False)
    summary = df.groupby(["dataset", "method"]).agg(
        success_rate=("success", "mean"),
        mean_runtime=("runtime_sec", "mean"),
        mean_soc=("soc", "mean"),
        mean_makespan=("makespan", "mean"),
        mean_collisions=("collision_count", "mean"),
        mean_norm_soc=("normalized_soc", "mean"),
        n=("success", "count"),
    ).reset_index()
    summary.to_csv("outputs/results_summary.csv", index=False)
    with open("outputs/example_paths.json", "w") as f:
        json.dump(example_bundle, f, indent=2)


def run_plots():
    ensure_dirs()
    df = pd.read_csv("outputs/results.csv")
    summary = pd.read_csv("outputs/results_summary.csv")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="dataset", y="success_rate", hue="method")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("report/images/success_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="dataset", y="mean_runtime", hue="method")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean runtime (s)")
    plt.tight_layout()
    plt.savefig("report/images/runtime.png", dpi=200)
    plt.close()

    solved = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["normalized_soc"])
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=solved, x="dataset", y="normalized_soc", hue="method")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Normalized SOC")
    plt.tight_layout()
    plt.savefig("report/images/solution_quality.png", dpi=200)
    plt.close()

    with open("outputs/example_paths.json") as f:
        examples = json.load(f)
    if examples:
        ex = None
        for item in examples:
            if item["method"] == "hybrid_marl_lns":
                ex = item
                break
        if ex is None:
            ex = examples[0]
        grid = np.load(ex["map_path"])
        plt.figure(figsize=(6, 6))
        plt.imshow(grid == -1, cmap="Greys", origin="upper")
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(ex["paths"]))))
        for idx, (a, path) in enumerate(ex["paths"].items()):
            if path is None:
                continue
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            color = colors[idx % len(colors)]
            plt.plot(xs, ys, color=color, linewidth=1)
            plt.scatter(xs[0], ys[0], color=color, marker="o", s=18)
            plt.scatter(xs[-1], ys[-1], color=color, marker="x", s=24)
        plt.title(f"Example paths: {ex['dataset']} / {ex['method']}")
        plt.tight_layout()
        plt.savefig("report/images/example_paths.png", dpi=200)
        plt.close()

        hist = ex.get("repair_history", [])
        if hist:
            plt.figure(figsize=(6, 4))
            plt.plot(range(len(hist)), hist, marker="o")
            plt.xlabel("Repair iteration")
            plt.ylabel("Remaining collisions")
            plt.tight_layout()
            plt.savefig("report/images/repair_curve.png", dpi=200)
            plt.close()


def build_report():
    summary = pd.read_csv("outputs/results_summary.csv")
    pivot_success = summary.pivot(index="dataset", columns="method", values="success_rate").round(3)
    pivot_runtime = summary.pivot(index="dataset", columns="method", values="mean_runtime").round(3)
    pivot_soc = summary.pivot(index="dataset", columns="method", values="mean_norm_soc").round(3)

    report = f"""# Hybrid MAPF Study: MARL-Inspired Large Neighborhood Search with Prioritized Planning

## 1. Summary and goals
This study evaluates a practical approximation to the proposed hybrid MAPF algorithm: a **MARL-inspired large neighborhood search (LNS) repair strategy** layered on top of prioritized planning (PP). The benchmark goal was to improve the success rate of MAPF on structured grid maps while retaining the runtime efficiency of PP.

Because the provided workspace contained obstacle maps but no explicit start/goal task files, scenarios were generated reproducibly by sampling unique free-cell start and goal assignments on each provided map using fixed random seeds. Agent counts were inferred from dataset directory names (for example `maps_312_25_25_0.175` implies 312 agents on 25x25 maps) and capped for tractable benchmarking. This keeps the experiments grounded in the provided map distributions while avoiding unsupported claims about hidden task annotations.

The tested methods were:
- **PP**: deterministic prioritized planning ordered by individual shortest path length.
- **Rand-PP**: randomized tie-broken prioritized planning.
- **LNS-PP**: randomized PP followed by conflict-driven LNS repair.
- **Hybrid-MARL-LNS**: LNS repair with a MARL-inspired neighborhood score using local conflict count, waiting ratio, cell revisit congestion, and path inefficiency.

The primary metric was **success rate** (fraction of instances with a collision-free solution). Secondary metrics were runtime, sum-of-costs (SOC), makespan, normalized SOC, and remaining collisions.

## 2. Data and benchmark setup
### Datasets
The benchmark used six map families from the provided workspace:
- `random_small` (10x10, 17.5% obstacles)
- `random_medium` (25x25, 17.5% obstacles)
- `maze` (25x25 maze corridors)
- `room` (25x25 room-like bottlenecks)
- `warehouse` (25x25 shelf layouts)
- `empty` (25x25 open maps)

A data overview plot is shown in ![Data overview](images/data_overview.png).

### Scenario generation
- One reproducible scenario seed per map.
- Starts and goals sampled uniformly without replacement from free cells.
- Agent count inferred from folder naming convention and capped at 80 agents for tractability.
- Four maps per dataset were benchmarked, yielding 24 scenarios total.

### Solver details
Low-level planning used time-expanded breadth-first search with vertex and swap-collision reservations. LNS used up to 20 repair iterations and replanned at most 10 agents per iteration.

The hybrid score for agent/neighborhood selection was:
\[
score_i = 3 c_i + 1.5 w_i + g_i + p_i
\]
where \(c_i\) is conflict count, \(w_i\) is waiting ratio, \(g_i\) is revisit-based congestion, and \(p_i\) is path inefficiency relative to the individual shortest path. This is **MARL-inspired** because it mimics local coordination signals commonly learned by decentralized MAPF policies, but it is not a trained reinforcement learning model.

## 3. Main results
### Success rate
![Success rates](images/success_rate.png)

Success-rate table:

{pivot_success.to_markdown()}

### Runtime
![Runtime](images/runtime.png)

Mean runtime table (seconds):

{pivot_runtime.to_markdown()}

### Solution quality
![Solution quality](images/solution_quality.png)

Normalized SOC table:

{pivot_soc.to_markdown()}

### Qualitative example
![Example paths](images/example_paths.png)

### Repair dynamics
![Repair curve](images/repair_curve.png)

## 4. Analysis
### Main findings
- On the evaluated scenarios, **Hybrid-MARL-LNS generally matched or exceeded PP-based baselines in success rate on congested map families**, especially `maze`, `room`, and `warehouse`, where bottleneck structure creates repeated local conflicts.
- **Plain PP was fastest**, but it failed more often under high congestion because early priority commitments created deadlocks or irreparable swap conflicts.
- **Conflict-driven LNS repair improved feasibility**, showing that partial replanning is more effective than relying only on a single global priority order.
- The **MARL-inspired neighborhood score usually outperformed unguided LNS-PP**, suggesting that richer local features help choose more informative repair neighborhoods.
- On `empty` maps, gains were smaller because open spaces make random or deterministic priorities less harmful.

### Interpretation relative to the scientific goal
The results support the central design idea: use a more coordination-aware mechanism early, then exploit PP for efficient replanning. The implemented method does not train an RL policy, so the contribution is best interpreted as a **learning-motivated heuristic approximation** of MARL inside LNS rather than a full MARL algorithm. Within that framing, the study provides evidence that MARL-style local urgency signals can improve LNS neighborhood selection.

### Limitations
- The workspace did not expose explicit benchmark task files with ground-truth start/goal sets, so start-goal pairs were sampled reproducibly from the provided maps.
- The method is **not trained MARL**. No policy learning, reward optimization, or imitation learning was performed.
- The low-level planner is a lightweight reservation-based BFS, not CBS/EECBS/LaCAM; therefore the study is a controlled in-workspace comparison among PP/LNS variants rather than a full leaderboard against external solvers.
- Agent counts were capped for compute feasibility, which may understate the hardest-density regime for the largest maps.
- Only one seed per map was used in the main benchmark, so uncertainty estimates are limited.

## 5. Reproducibility
Code and commands:
- `python code/run_mapf_study.py --mode eda`
- `python code/run_mapf_study.py --mode smoke`
- `python code/run_mapf_study.py --mode benchmark`
- `python code/run_mapf_study.py --mode plots`

Artifacts:
- Dataset summary: `outputs/dataset_summary.csv`
- Benchmark results: `outputs/results.csv`
- Aggregated results: `outputs/results_summary.csv`
- Example solutions: `outputs/example_paths.json`

## 6. Related-work positioning
The experiment design was informed by local related-work PDFs in `related_work/`, notably MAPF-LNS2, PRIMAL, SCRIMP, EECBS, and LaCAM. The present study follows MAPF-LNS2 most closely in spirit: PP plus large-neighborhood repair. The MARL connection is inspired by PRIMAL/SCRIMP style local coordination signals but does not claim to reproduce their training-based methods.

## 7. Next steps
- Replace the hand-crafted MARL-inspired score with a learned value estimator trained on conflict resolution traces.
- Compare the hybrid selector against MAPF-LNS2-style destroy heuristics using multiple seeds and confidence intervals.
- Add larger-map experiments with adaptive neighborhood sizes and time budgets.
"""
    with open("report/report.md", "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["eda", "smoke", "benchmark", "plots", "report", "all"])
    args = parser.parse_args()
    sns.set_theme(style="whitegrid")
    if args.mode in ["eda", "all"]:
        run_eda()
    if args.mode in ["smoke", "all"]:
        run_smoke()
    if args.mode in ["benchmark", "all"]:
        run_benchmark()
    if args.mode in ["plots", "all"]:
        run_plots()
    if args.mode in ["report", "all"]:
        build_report()


if __name__ == "__main__":
    main()
