#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"

SEED = 20260402
MAX_TIME = 256
REPAIR_STEPS = 6
REPRESENTATIVE_MAPS_PER_FAMILY = 3

Coord = Tuple[int, int]
PathT = List[Coord]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MapRecord:
    family: str
    scenario: str
    file_path: Path
    nominal_agent_count: int
    height: int
    width: int
    obstacle_density: float
    free_cells: int
    obstacle_cells: int
    component_count: int
    largest_component_ratio: float
    mean_branching: float
    corridor_ratio: float


@dataclass
class PlannerResult:
    method: str
    family: str
    scenario: str
    map_file: str
    agents: int
    solved: bool
    runtime_sec: float
    sum_of_costs: Optional[int]
    makespan: Optional[int]
    vertex_collisions: int
    edge_collisions: int
    collision_events: int
    replans: int
    repair_rounds: int
    notes: str


DIR_FAMILY = {
    "maps_60_10_10_0.175": "random_small_top",
    "empty": "empty",
    "maze": "maze",
    "random_large": "random_large",
    "random_medium": "random_medium",
    "random_small": "random_small",
    "room": "room",
    "warehouse": "warehouse",
}


def infer_family(path: Path) -> str:
    rel = path.relative_to(DATA_DIR)
    return DIR_FAMILY.get(rel.parts[0], rel.parts[0])


AGENT_RE = re.compile(r"(\d+)")


def infer_nominal_agent_count(path: Path) -> int:
    nums = [int(x) for x in AGENT_RE.findall(path.name)]
    return nums[0] if nums else 0


def neighbors(grid: np.ndarray, cell: Coord) -> Iterable[Coord]:
    h, w = grid.shape
    r, c = cell
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            yield (nr, nc)


def component_stats(grid: np.ndarray) -> Tuple[int, float]:
    free = list(map(tuple, np.argwhere(grid == 0)))
    if not free:
        return 0, 0.0
    free_set = set(free)
    seen = set()
    sizes = []
    for cell in free:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in neighbors(grid, cur):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        sizes.append(size)
    total = len(free)
    return len(sizes), max(sizes) / total if total else 0.0


def branching_stats(grid: np.ndarray) -> Tuple[float, float]:
    free = np.argwhere(grid == 0)
    if len(free) == 0:
        return 0.0, 0.0
    degrees = []
    for r, c in free:
        deg = sum(1 for _ in neighbors(grid, (int(r), int(c))))
        degrees.append(deg)
    deg_arr = np.array(degrees, dtype=float)
    return float(deg_arr.mean()), float(np.mean(deg_arr <= 2))


def collect_map_records() -> List[MapRecord]:
    records: List[MapRecord] = []
    for file_path in sorted(DATA_DIR.rglob("*.npy")):
        grid = np.load(file_path)
        family = infer_family(file_path)
        scenario = file_path.parent.name
        obstacle_cells = int(np.sum(grid < 0))
        free_cells = int(np.sum(grid == 0))
        obstacle_density = obstacle_cells / grid.size
        component_count, largest_component_ratio = component_stats(grid)
        mean_branching, corridor_ratio = branching_stats(grid)
        records.append(
            MapRecord(
                family=family,
                scenario=scenario,
                file_path=file_path,
                nominal_agent_count=infer_nominal_agent_count(file_path.parent),
                height=int(grid.shape[0]),
                width=int(grid.shape[1]),
                obstacle_density=obstacle_density,
                free_cells=free_cells,
                obstacle_cells=obstacle_cells,
                component_count=component_count,
                largest_component_ratio=largest_component_ratio,
                mean_branching=mean_branching,
                corridor_ratio=corridor_ratio,
            )
        )
    return records


def write_dataset_summary(records: Sequence[MapRecord]) -> None:
    csv_path = OUTPUT_DIR / "dataset_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "family",
            "scenario",
            "map_file",
            "nominal_agent_count",
            "height",
            "width",
            "obstacle_density",
            "free_cells",
            "obstacle_cells",
            "component_count",
            "largest_component_ratio",
            "mean_branching",
            "corridor_ratio",
        ])
        for r in records:
            writer.writerow([
                r.family,
                r.scenario,
                r.file_path.name,
                r.nominal_agent_count,
                r.height,
                r.width,
                f"{r.obstacle_density:.6f}",
                r.free_cells,
                r.obstacle_cells,
                r.component_count,
                f"{r.largest_component_ratio:.6f}",
                f"{r.mean_branching:.6f}",
                f"{r.corridor_ratio:.6f}",
            ])

    by_family: Dict[str, Dict[str, float]] = {}
    fam_groups: Dict[str, List[MapRecord]] = defaultdict(list)
    for r in records:
        fam_groups[r.family].append(r)
    for family, items in fam_groups.items():
        by_family[family] = {
            "maps": len(items),
            "scenarios": len({x.scenario for x in items}),
            "grid_shapes": sorted({f"{x.height}x{x.width}" for x in items}),
            "mean_obstacle_density": round(float(np.mean([x.obstacle_density for x in items])), 4),
            "mean_component_count": round(float(np.mean([x.component_count for x in items])), 3),
            "mean_largest_component_ratio": round(float(np.mean([x.largest_component_ratio for x in items])), 4),
            "mean_corridor_ratio": round(float(np.mean([x.corridor_ratio for x in items])), 4),
        }
    (OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(by_family, indent=2))


@dataclass(frozen=True)
class AgentSpec:
    start: Coord
    goal: Coord


class ReservationTable:
    def __init__(self) -> None:
        self.vertex: Dict[Tuple[int, Coord], int] = {}
        self.edge: Dict[Tuple[int, Coord, Coord], int] = {}
        self.goal_block: Dict[Coord, int] = {}

    def reserve(self, path: PathT, agent_id: int) -> None:
        for t, pos in enumerate(path):
            self.vertex[(t, pos)] = agent_id
            if t > 0:
                prev = path[t - 1]
                self.edge[(t, prev, pos)] = agent_id
        if path:
            self.goal_block[path[-1]] = max(self.goal_block.get(path[-1], -1), len(path) - 1)

    def conflicts(self, prev: Coord, nxt: Coord, t: int) -> bool:
        if (t, nxt) in self.vertex:
            return True
        if (t, nxt, prev) in self.edge:
            return True
        if nxt in self.goal_block and t >= self.goal_block[nxt]:
            return True
        return False


def shortest_path_lengths(grid: np.ndarray, start: Coord) -> Dict[Coord, int]:
    dist = {start: 0}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nxt in neighbors(grid, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def choose_agent_count(record: MapRecord) -> int:
    base = max(4, min(18, int(0.03 * record.free_cells)))
    if record.family in {"empty", "random_large"}:
        base = min(base + 4, 20)
    if record.family in {"maze", "room", "warehouse"}:
        base = min(base + 2, 18)
    return min(base, max(2, record.free_cells // 8))


def sample_agents(grid: np.ndarray, count: int, rng: np.random.Generator) -> List[AgentSpec]:
    free = [tuple(map(int, x)) for x in np.argwhere(grid == 0)]
    if len(free) < count * 2:
        raise ValueError("not enough free cells")
    free_set = set(free)
    starts: List[Coord] = []
    goals: List[Coord] = []
    attempts = 0
    while len(starts) < count and attempts < 5000:
        attempts += 1
        s = free[rng.integers(len(free))]
        dmap = shortest_path_lengths(grid, s)
        viable = [c for c, d in dmap.items() if d >= max(4, (grid.shape[0] + grid.shape[1]) // 4) and c != s and c not in goals]
        if not viable or s in starts:
            continue
        g = viable[int(rng.integers(len(viable)))]
        starts.append(s)
        goals.append(g)
    if len(starts) < count:
        cells = free.copy()
        rng.shuffle(cells)
        starts = cells[:count]
        goals = cells[count : count * 2]
    agents = [AgentSpec(start=s, goal=g) for s, g in zip(starts, goals)]
    valid_agents = []
    used_starts = set()
    used_goals = set()
    for a in agents:
        if a.start in used_starts or a.goal in used_goals or a.start == a.goal:
            continue
        if a.start not in free_set or a.goal not in free_set:
            continue
        valid_agents.append(a)
        used_starts.add(a.start)
        used_goals.add(a.goal)
    if len(valid_agents) < max(2, count // 2):
        raise ValueError("agent sampling failed")
    return valid_agents


def heuristic(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def space_time_astar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    reservation: ReservationTable,
    congestion: Optional[Dict[Tuple[int, Coord], float]] = None,
    max_time: int = MAX_TIME,
) -> Optional[PathT]:
    import heapq

    stay_penalty = 0.25
    congestion = congestion or {}
    heap = []
    start_state = (heuristic(start, goal), 0.0, start, 0, None)
    heapq.heappush(heap, start_state)
    parents: Dict[Tuple[Coord, int], Tuple[Coord, int]] = {}
    g_cost = {(start, 0): 0.0}
    seen_goal_state: Optional[Tuple[Coord, int]] = None

    while heap:
        _, cost, pos, t, prev = heapq.heappop(heap)
        state = (pos, t)
        if cost > g_cost.get(state, float("inf")):
            continue
        if pos == goal:
            safe = True
            for future_t in range(t, min(max_time, t + 3)):
                if reservation.conflicts(pos, pos, future_t):
                    safe = False
                    break
            if safe:
                seen_goal_state = state
                break
        if t >= max_time:
            continue
        for nxt in list(neighbors(grid, pos)) + [pos]:
            nt = t + 1
            if reservation.conflicts(pos, nxt, nt):
                continue
            move_cost = 1.0 + (stay_penalty if nxt == pos else 0.0) + congestion.get((nt, nxt), 0.0)
            new_cost = cost + move_cost
            nstate = (nxt, nt)
            if new_cost < g_cost.get(nstate, float("inf")):
                g_cost[nstate] = new_cost
                parents[nstate] = state
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(heap, (priority, new_cost, nxt, nt, pos))
    if seen_goal_state is None:
        return None
    path = []
    cur = seen_goal_state
    while True:
        pos, t = cur
        path.append(pos)
        if t == 0:
            break
        cur = parents[cur]
    path.reverse()
    return path


def compute_collisions(paths: Sequence[PathT]) -> Tuple[int, int, int, Dict[int, int]]:
    max_len = max((len(p) for p in paths), default=0)
    vertex = 0
    edge = 0
    involvement = Counter()
    for t in range(max_len):
        occ = defaultdict(list)
        traversals = defaultdict(list)
        for i, path in enumerate(paths):
            pos = path[t] if t < len(path) else path[-1]
            occ[pos].append(i)
            prev = path[t - 1] if t - 1 >= 0 and t - 1 < len(path) else path[0]
            traversals[(prev, pos)].append(i)
        for agents in occ.values():
            if len(agents) > 1:
                vertex += math.comb(len(agents), 2)
                for a in agents:
                    involvement[a] += len(agents) - 1
        checked = set()
        for (u, v), agents in traversals.items():
            if (v, u) in traversals and (v, u) not in checked and u != v:
                others = traversals[(v, u)]
                edge += len(agents) * len(others)
                for a in agents:
                    involvement[a] += len(others)
                for b in others:
                    involvement[b] += len(agents)
                checked.add((u, v))
    return vertex, edge, vertex + edge, dict(involvement)


def prioritized_planning(
    grid: np.ndarray,
    agents: Sequence[AgentSpec],
    order: Sequence[int],
    congestion: Optional[Dict[Tuple[int, Coord], float]] = None,
) -> Tuple[Optional[List[PathT]], int]:
    reservation = ReservationTable()
    paths: List[Optional[PathT]] = [None] * len(agents)
    replans = 0
    for idx in order:
        agent = agents[idx]
        path = space_time_astar(grid, agent.start, agent.goal, reservation, congestion=congestion)
        replans += 1
        if path is None:
            return None, replans
        reservation.reserve(path, idx)
        paths[idx] = path
    return [p for p in paths if p is not None], replans


def independent_paths(grid: np.ndarray, agents: Sequence[AgentSpec]) -> List[PathT]:
    empty_res = ReservationTable()
    paths = []
    for a in agents:
        p = space_time_astar(grid, a.start, a.goal, empty_res, congestion=None)
        if p is None:
            p = [a.start, a.goal]
        paths.append(p)
    return paths


def congestion_from_paths(paths: Sequence[PathT]) -> Dict[Tuple[int, Coord], float]:
    counts = Counter()
    for p in paths:
        for t, pos in enumerate(p):
            counts[(t, pos)] += 1
    penalty = {}
    for key, v in counts.items():
        if v > 1:
            penalty[key] = 0.6 * (v - 1)
    return penalty


def marl_guided_hybrid(grid: np.ndarray, agents: Sequence[AgentSpec]) -> Tuple[Optional[List[PathT]], int, int]:
    rollout_paths = independent_paths(grid, agents)
    _, _, _, involvement = compute_collisions(rollout_paths)
    base_order = list(range(len(agents)))
    base_order.sort(key=lambda i: (-(involvement.get(i, 0)), heuristic(agents[i].start, agents[i].goal)))
    congestion = congestion_from_paths(rollout_paths)

    planned, replans = prioritized_planning(grid, agents, base_order, congestion=congestion)
    repair_rounds = 0
    if planned is None:
        return None, replans, repair_rounds

    for repair_rounds in range(1, REPAIR_STEPS + 1):
        v, e, total, involvement = compute_collisions(planned)
        if total == 0:
            return planned, replans, repair_rounds
        conflicted = sorted(involvement, key=lambda i: -involvement[i])[: max(2, len(agents) // 3)]
        static_agents = [i for i in range(len(agents)) if i not in conflicted]
        reservation = ReservationTable()
        new_paths: List[Optional[PathT]] = [None] * len(agents)
        for idx in static_agents:
            reservation.reserve(planned[idx], idx)
            new_paths[idx] = planned[idx]
        updated_rollout = planned.copy()
        local_congestion = congestion_from_paths([updated_rollout[i] for i in static_agents])
        local_order = sorted(conflicted, key=lambda i: (-(involvement.get(i, 0)), heuristic(agents[i].start, agents[i].goal)))
        feasible = True
        for idx in local_order:
            p = space_time_astar(grid, agents[idx].start, agents[idx].goal, reservation, congestion=local_congestion)
            replans += 1
            if p is None:
                feasible = False
                break
            reservation.reserve(p, idx)
            new_paths[idx] = p
        if feasible:
            planned = [p for p in new_paths if p is not None]
        else:
            break
    return planned, replans, repair_rounds


def evaluate_paths(paths: Sequence[PathT]) -> Tuple[int, int, int, int, int]:
    v, e, total, _ = compute_collisions(paths)
    makespan = max(len(p) - 1 for p in paths)
    sum_of_costs = sum(len(p) - 1 for p in paths)
    return sum_of_costs, makespan, v, e, total


def run_planner(method: str, grid: np.ndarray, agents: Sequence[AgentSpec], family: str, scenario: str, map_file: str) -> PlannerResult:
    start_time = time.perf_counter()
    if method == "prioritized_planning":
        order = list(range(len(agents)))
        order.sort(key=lambda i: heuristic(agents[i].start, agents[i].goal), reverse=True)
        planned, replans = prioritized_planning(grid, agents, order)
        repair_rounds = 0
    elif method == "hybrid_marl_lns":
        planned, replans, repair_rounds = marl_guided_hybrid(grid, agents)
    else:
        raise ValueError(method)
    runtime = time.perf_counter() - start_time

    if planned is None or len(planned) != len(agents):
        return PlannerResult(
            method=method,
            family=family,
            scenario=scenario,
            map_file=map_file,
            agents=len(agents),
            solved=False,
            runtime_sec=runtime,
            sum_of_costs=None,
            makespan=None,
            vertex_collisions=-1,
            edge_collisions=-1,
            collision_events=-1,
            replans=replans,
            repair_rounds=repair_rounds,
            notes="planning failed",
        )

    sum_of_costs, makespan, v, e, total = evaluate_paths(planned)
    return PlannerResult(
        method=method,
        family=family,
        scenario=scenario,
        map_file=map_file,
        agents=len(agents),
        solved=(total == 0),
        runtime_sec=runtime,
        sum_of_costs=sum_of_costs,
        makespan=makespan,
        vertex_collisions=v,
        edge_collisions=e,
        collision_events=total,
        replans=replans,
        repair_rounds=repair_rounds,
        notes="ok" if total == 0 else "residual conflicts",
    )


def choose_representatives(records: Sequence[MapRecord]) -> List[MapRecord]:
    grouped: Dict[str, List[MapRecord]] = defaultdict(list)
    for r in records:
        grouped[r.family].append(r)
    selected = []
    for family, items in grouped.items():
        items = sorted(items, key=lambda x: (x.scenario, x.file_path.name))
        idxs = np.linspace(0, len(items) - 1, num=min(REPRESENTATIVE_MAPS_PER_FAMILY, len(items)), dtype=int)
        for i in idxs:
            selected.append(items[int(i)])
    return selected


def write_experiment_results(results: Sequence[PlannerResult]) -> None:
    path = OUTPUT_DIR / "experiment_results.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "family",
            "scenario",
            "map_file",
            "agents",
            "solved",
            "runtime_sec",
            "sum_of_costs",
            "makespan",
            "vertex_collisions",
            "edge_collisions",
            "collision_events",
            "replans",
            "repair_rounds",
            "notes",
        ])
        for r in results:
            writer.writerow([
                r.method,
                r.family,
                r.scenario,
                r.map_file,
                r.agents,
                int(r.solved),
                f"{r.runtime_sec:.6f}",
                r.sum_of_costs if r.sum_of_costs is not None else "",
                r.makespan if r.makespan is not None else "",
                r.vertex_collisions,
                r.edge_collisions,
                r.collision_events,
                r.replans,
                r.repair_rounds,
                r.notes,
            ])

    summary = defaultdict(lambda: defaultdict(list))
    for r in results:
        summary[r.family][r.method].append(r)
    final = {}
    for family, methods in summary.items():
        final[family] = {}
        for method, rows in methods.items():
            solved = [int(x.solved) for x in rows]
            runtimes = [x.runtime_sec for x in rows]
            costs = [x.sum_of_costs for x in rows if x.sum_of_costs is not None]
            collisions = [x.collision_events for x in rows if x.collision_events >= 0]
            final[family][method] = {
                "instances": len(rows),
                "success_rate": round(float(np.mean(solved)) if solved else 0.0, 4),
                "mean_runtime_sec": round(float(np.mean(runtimes)) if runtimes else 0.0, 4),
                "mean_sum_of_costs": round(float(np.mean(costs)) if costs else 0.0, 3),
                "mean_collision_events": round(float(np.mean(collisions)) if collisions else 0.0, 3),
            }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(final, indent=2))


def plot_dataset_overview(records: Sequence[MapRecord]) -> None:
    families = sorted({r.family for r in records})
    counts = [sum(1 for r in records if r.family == fam) for fam in families]
    densities = [np.mean([r.obstacle_density for r in records if r.family == fam]) for fam in families]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(families, counts, color="#4C78A8")
    axes[0].set_title("Dataset size by map family")
    axes[0].set_ylabel("Number of maps")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(families, densities, color="#F58518")
    axes[1].set_title("Mean obstacle density by family")
    axes[1].set_ylabel("Obstacle density")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "dataset_overview.png", dpi=200)
    plt.close(fig)

    sample_records = []
    seen = set()
    for r in records:
        if r.family not in seen:
            sample_records.append(r)
            seen.add(r.family)
    cols = 4
    rows = math.ceil(len(sample_records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, rec in zip(axes, sample_records):
        grid = np.load(rec.file_path)
        ax.imshow(grid < 0, cmap="gray_r")
        ax.set_title(f"{rec.family}\n{rec.height}x{rec.width}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "map_family_examples.png", dpi=200)
    plt.close(fig)


def plot_results(results: Sequence[PlannerResult]) -> None:
    families = sorted({r.family for r in results})
    methods = ["prioritized_planning", "hybrid_marl_lns"]
    success = {m: [] for m in methods}
    runtime = {m: [] for m in methods}
    cost = {m: [] for m in methods}
    for family in families:
        fam_rows = [r for r in results if r.family == family]
        for method in methods:
            rows = [r for r in fam_rows if r.method == method]
            success[method].append(np.mean([int(r.solved) for r in rows]) if rows else 0.0)
            runtime[method].append(np.mean([r.runtime_sec for r in rows]) if rows else 0.0)
            costs = [r.sum_of_costs for r in rows if r.sum_of_costs is not None]
            cost[method].append(np.mean(costs) if costs else np.nan)

    x = np.arange(len(families))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = {"prioritized_planning": "#54A24B", "hybrid_marl_lns": "#E45756"}
    titles = ["Success rate", "Mean runtime (s)", "Mean sum of costs"]
    datasets = [success, runtime, cost]
    for ax, title, ds in zip(axes, titles, datasets):
        for i, method in enumerate(methods):
            ax.bar(x + (i - 0.5) * width, ds[method], width=width, label=method, color=palette[method])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=35, ha="right")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "planner_comparison.png", dpi=200)
    plt.close(fig)

    solved_rows = [r for r in results if r.sum_of_costs is not None]
    if solved_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, color in palette.items():
            rows = [r for r in solved_rows if r.method == method]
            ax.scatter(
                [r.runtime_sec for r in rows],
                [r.sum_of_costs for r in rows],
                label=method,
                color=color,
                alpha=0.8,
            )
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Sum of costs")
        ax.set_title("Efficiency-quality trade-off across evaluated instances")
        ax.legend()
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "runtime_vs_cost.png", dpi=200)
        plt.close(fig)


def write_narrative(records: Sequence[MapRecord], results: Sequence[PlannerResult]) -> None:
    family_counts = Counter(r.family for r in records)
    summary = defaultdict(dict)
    for family in sorted({r.family for r in results}):
        for method in sorted({r.method for r in results}):
            rows = [r for r in results if r.family == family and r.method == method]
            if not rows:
                continue
            summary[family][method] = {
                "success_rate": round(float(np.mean([int(r.solved) for r in rows])), 3),
                "mean_runtime": round(float(np.mean([r.runtime_sec for r in rows])), 4),
                "mean_cost": round(float(np.mean([r.sum_of_costs for r in rows if r.sum_of_costs is not None])), 3)
                if any(r.sum_of_costs is not None for r in rows)
                else None,
            }

    lines = []
    lines.append("Hybrid MAPF analysis summary")
    lines.append("=" * 28)
    lines.append("")
    lines.append("Dataset coverage:")
    for family, count in sorted(family_counts.items()):
        lines.append(f"- {family}: {count} maps")
    lines.append("")
    lines.append("Method summary by family:")
    for family in sorted(summary):
        lines.append(f"- {family}:")
        for method, vals in summary[family].items():
            lines.append(
                f"  - {method}: success={vals['success_rate']:.3f}, runtime={vals['mean_runtime']:.4f}s, mean_cost={vals['mean_cost']}"
            )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- The hybrid method emulates a MARL-guided early conflict forecast by prioritizing agents whose independently planned rollouts create the most congestion, then performs LNS-style repair rounds before falling back to reservation-based planning."
    )
    lines.append(
        "- The baseline relies only on prioritized planning, so differences in success or cost isolate the value of congestion-aware ordering and local repair."
    )
    lines.append(
        "- These experiments use synthetic start-goal assignments sampled from the provided occupancy grids because explicit agent-task files are not present in the workspace inputs."
    )
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    records = collect_map_records()
    write_dataset_summary(records)
    plot_dataset_overview(records)

    selected = choose_representatives(records)
    rng = np.random.default_rng(SEED)
    results: List[PlannerResult] = []
    instance_specs = []
    for rec in selected:
        grid = np.load(rec.file_path)
        agent_count = choose_agent_count(rec)
        try:
            agents = sample_agents(grid, agent_count, rng)
        except Exception as exc:
            continue
        instance_specs.append(
            {
                "family": rec.family,
                "scenario": rec.scenario,
                "map_file": rec.file_path.name,
                "agents": [{"start": list(a.start), "goal": list(a.goal)} for a in agents],
            }
        )
        for method in ["prioritized_planning", "hybrid_marl_lns"]:
            results.append(run_planner(method, grid, agents, rec.family, rec.scenario, rec.file_path.name))

    (OUTPUT_DIR / "sampled_instances.json").write_text(json.dumps(instance_specs, indent=2))
    write_experiment_results(results)
    plot_results(results)
    write_narrative(records, results)


if __name__ == "__main__":
    main()
