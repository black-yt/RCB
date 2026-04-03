from __future__ import annotations

import heapq
import math
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Position = Tuple[int, int]
Path = List[Position]

ACTIONS: Tuple[Position, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class MAPFTask:
    grid: np.ndarray
    starts: Tuple[Position, ...]
    goals: Tuple[Position, ...]
    map_name: str
    family: str
    seed: int


@dataclass
class SolverResult:
    success: bool
    paths: List[Path]
    runtime: float
    sum_of_costs: Optional[int]
    makespan: Optional[int]
    collision_count: int
    iterations: int
    method: str


def neighbors(grid: np.ndarray, pos: Position) -> Iterable[Position]:
    h, w = grid.shape
    r, c = pos
    for dr, dc in ACTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            yield nr, nc


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def normalize_paths(paths: Sequence[Path]) -> List[Path]:
    if not paths:
        return []
    max_len = max(len(path) for path in paths)
    normalized = []
    for path in paths:
        if not path:
            normalized.append(path)
            continue
        if len(path) < max_len:
            normalized.append(path + [path[-1]] * (max_len - len(path)))
        else:
            normalized.append(list(path))
    return normalized


def path_position(path: Path, t: int) -> Position:
    if t < len(path):
        return path[t]
    return path[-1]


def detect_conflicts(paths: Sequence[Path]) -> List[Dict[str, object]]:
    if not paths:
        return []
    conflicts: List[Dict[str, object]] = []
    max_t = max(len(path) for path in paths)
    for t in range(max_t):
        positions: Dict[Position, int] = {}
        for agent, path in enumerate(paths):
            pos = path_position(path, t)
            if pos in positions:
                conflicts.append(
                    {
                        "type": "vertex",
                        "time": t,
                        "a1": positions[pos],
                        "a2": agent,
                        "pos": pos,
                    }
                )
            else:
                positions[pos] = agent
        if t == max_t - 1:
            continue
        edges: Dict[Tuple[Position, Position], int] = {}
        for agent, path in enumerate(paths):
            edge = (path_position(path, t), path_position(path, t + 1))
            rev = (edge[1], edge[0])
            if rev in edges and edge[0] != edge[1]:
                conflicts.append(
                    {
                        "type": "swap",
                        "time": t,
                        "a1": edges[rev],
                        "a2": agent,
                        "edge": edge,
                    }
                )
            edges[edge] = agent
    return conflicts


def collision_count(paths: Sequence[Path]) -> int:
    return len(detect_conflicts(paths))


def sum_of_costs(paths: Sequence[Path]) -> int:
    return sum(max(len(path) - 1, 0) for path in paths)


def makespan(paths: Sequence[Path]) -> int:
    return max((len(path) - 1 for path in paths), default=0)


def connected_component(grid: np.ndarray, seed: Position) -> List[Position]:
    q = deque([seed])
    seen = {seed}
    out = []
    while q:
        cell = q.popleft()
        out.append(cell)
        for nxt in neighbors(grid, cell):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return out


def largest_component_cells(grid: np.ndarray) -> List[Position]:
    free_cells = list(map(tuple, np.argwhere(grid == 0)))
    seen = set()
    best: List[Position] = []
    for cell in free_cells:
        if cell in seen:
            continue
        comp = connected_component(grid, cell)
        seen.update(comp)
        if len(comp) > len(best):
            best = comp
    return best


def sample_task(
    grid: np.ndarray,
    agent_count: int,
    rng: random.Random,
    map_name: str,
    family: str,
    seed: int,
) -> MAPFTask:
    component = largest_component_cells(grid)
    if len(component) < 2 * agent_count:
        raise ValueError(f"Not enough connected free cells for {agent_count} agents")
    chosen = rng.sample(component, 2 * agent_count)
    starts = tuple(chosen[:agent_count])
    goals = tuple(chosen[agent_count:])
    return MAPFTask(
        grid=grid.copy(),
        starts=starts,
        goals=goals,
        map_name=map_name,
        family=family,
        seed=seed,
    )


def shortest_path_static(grid: np.ndarray, start: Position, goal: Position) -> Optional[Path]:
    frontier = [(manhattan(start, goal), 0, start)]
    parent = {start: None}
    g_cost = {start: 0}
    while frontier:
        _, g, node = heapq.heappop(frontier)
        if node == goal:
            path = [node]
            while parent[node] is not None:
                node = parent[node]
                path.append(node)
            return list(reversed(path))
        if g != g_cost[node]:
            continue
        for nxt in neighbors(grid, node):
            cand = g + 1
            if cand < g_cost.get(nxt, math.inf):
                g_cost[nxt] = cand
                parent[nxt] = node
                heapq.heappush(frontier, (cand + manhattan(nxt, goal), cand, nxt))
    return None


class ReservationTable:
    def __init__(self, paths: Sequence[Path]):
        self.vertex = defaultdict(set)
        self.edge = defaultdict(set)
        self.goal_times: Dict[Position, List[int]] = defaultdict(list)
        self.max_time = 0
        for path in paths:
            if not path:
                continue
            self.max_time = max(self.max_time, len(path) - 1)
            for t in range(len(path)):
                self.vertex[t].add(path[t])
                if t + 1 < len(path):
                    self.edge[t + 1].add((path[t], path[t + 1]))
            self.goal_times[path[-1]].append(len(path) - 1)

    def occupied(self, pos: Position, t: int) -> bool:
        if pos in self.vertex.get(t, set()):
            return True
        for goal_pos, times in self.goal_times.items():
            if goal_pos == pos and min(times) <= t:
                return True
        return False

    def edge_blocked(self, prev: Position, nxt: Position, t: int) -> bool:
        return (nxt, prev) in self.edge.get(t, set())

    def goal_safe(self, pos: Position, t: int) -> bool:
        for goal_pos, times in self.goal_times.items():
            if goal_pos == pos and min(times) <= t:
                return False
        future_vertex = self.vertex.get(t, set())
        if pos in future_vertex:
            return False
        for tau in range(t + 1, self.max_time + 2):
            if pos in self.vertex.get(tau, set()):
                return False
        return True


def spacetime_astar(
    grid: np.ndarray,
    start: Position,
    goal: Position,
    reserved_paths: Sequence[Path],
    max_time: Optional[int] = None,
    max_expansions: int = 15000,
) -> Optional[Path]:
    reservation = ReservationTable(reserved_paths)
    if max_time is None:
        max_time = max(
            reservation.max_time + manhattan(start, goal) + 10,
            manhattan(start, goal) + 10,
        )
    start_state = (start, 0)
    frontier = [(manhattan(start, goal), 0, start_state)]
    parent = {start_state: None}
    g_cost = {start_state: 0}
    expansions = 0
    while frontier:
        _, g, (node, t) = heapq.heappop(frontier)
        if g != g_cost[(node, t)]:
            continue
        expansions += 1
        if expansions > max_expansions:
            return None
        if node == goal and reservation.goal_safe(goal, t):
            state = (node, t)
            path = [node]
            while parent[state] is not None:
                state = parent[state]
                path.append(state[0])
            return list(reversed(path))
        if t >= max_time:
            continue
        for nxt in neighbors(grid, node):
            nt = t + 1
            if reservation.occupied(nxt, nt):
                continue
            if reservation.edge_blocked(node, nxt, nt):
                continue
            state = (nxt, nt)
            cand = g + 1
            if cand < g_cost.get(state, math.inf):
                g_cost[state] = cand
                parent[state] = (node, t)
                priority = cand + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, cand, state))
    return None


def prioritized_planning(
    task: MAPFTask,
    order: Sequence[int],
    restart_budget: int = 1,
    rng: Optional[random.Random] = None,
) -> SolverResult:
    start_time = time.perf_counter()
    rng = rng or random.Random(task.seed)
    best_paths: Optional[List[Path]] = None
    iterations = 0
    orders = [list(order)]
    for _ in range(restart_budget - 1):
        shuffled = list(order)
        rng.shuffle(shuffled)
        orders.append(shuffled)
    for current_order in orders:
        iterations += 1
        paths: List[Optional[Path]] = [None] * len(task.starts)
        success = True
        for agent in current_order:
            reserved = [p for idx, p in enumerate(paths) if idx != agent and p is not None]
            path = spacetime_astar(task.grid, task.starts[agent], task.goals[agent], reserved)
            if path is None:
                success = False
                break
            paths[agent] = path
        if success:
            best_paths = [p for p in paths if p is not None]
            break
    runtime = time.perf_counter() - start_time
    if best_paths is None:
        return SolverResult(
            success=False,
            paths=[],
            runtime=runtime,
            sum_of_costs=None,
            makespan=None,
            collision_count=-1,
            iterations=iterations,
            method="PP",
        )
    normalized = normalize_paths(best_paths)
    return SolverResult(
        success=True,
        paths=normalized,
        runtime=runtime,
        sum_of_costs=sum_of_costs(normalized),
        makespan=makespan(normalized),
        collision_count=collision_count(normalized),
        iterations=iterations,
        method="PP",
    )


class SharedQLearner:
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=float))

    def state_key(
        self,
        grid: np.ndarray,
        pos: Position,
        goal: Position,
        occupied_neighbors: Sequence[Position],
    ) -> Tuple[int, ...]:
        dr = max(-1, min(1, goal[0] - pos[0]))
        dc = max(-1, min(1, goal[1] - pos[1]))
        dist_bucket = min(manhattan(pos, goal), 4)
        occ = set(occupied_neighbors)
        obstacle_bits = []
        for move in ACTIONS[1:]:
            nr, nc = pos[0] + move[0], pos[1] + move[1]
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] != 0:
                obstacle_bits.append(1)
            else:
                obstacle_bits.append(0)
        occup_bits = [int((pos[0] + move[0], pos[1] + move[1]) in occ) for move in ACTIONS[1:]]
        return (dr, dc, dist_bucket, *obstacle_bits, *occup_bits)

    def act(self, key: Tuple[int, ...], rng: random.Random) -> int:
        if rng.random() < self.epsilon:
            return rng.randrange(len(ACTIONS))
        return int(np.argmax(self.q[key]))

    def update(
        self,
        key: Tuple[int, ...],
        action: int,
        reward: float,
        next_key: Tuple[int, ...],
    ) -> None:
        target = reward + self.gamma * np.max(self.q[next_key])
        self.q[key][action] += self.alpha * (target - self.q[key][action])


def step_joint_policy(
    grid: np.ndarray,
    positions: Sequence[Position],
    goals: Sequence[Position],
    policy: SharedQLearner,
    rng: random.Random,
    train: bool = False,
) -> Tuple[List[Position], List[float], List[int], List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    state_keys = []
    actions = []
    proposed = []
    pos_set = list(positions)
    for idx, pos in enumerate(positions):
        others = [p for j, p in enumerate(pos_set) if j != idx]
        key = policy.state_key(grid, pos, goals[idx], others)
        action = policy.act(key, rng)
        move = ACTIONS[action]
        nxt = (pos[0] + move[0], pos[1] + move[1])
        if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]) or grid[nxt] != 0:
            nxt = pos
        state_keys.append(key)
        actions.append(action)
        proposed.append(nxt)
    final_positions = list(proposed)
    rewards = [-0.05] * len(positions)
    counts = Counter(proposed)
    for i, nxt in enumerate(proposed):
        if counts[nxt] > 1:
            final_positions[i] = positions[i]
            rewards[i] -= 1.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if proposed[i] == positions[j] and proposed[j] == positions[i] and proposed[i] != positions[i]:
                final_positions[i] = positions[i]
                final_positions[j] = positions[j]
                rewards[i] -= 1.0
                rewards[j] -= 1.0
    next_keys = []
    for idx, pos in enumerate(final_positions):
        if pos == goals[idx]:
            rewards[idx] += 2.0
        rewards[idx] += 0.2 * (manhattan(positions[idx], goals[idx]) - manhattan(pos, goals[idx]))
        others = [p for j, p in enumerate(final_positions) if j != idx]
        next_keys.append(policy.state_key(grid, pos, goals[idx], others))
    return final_positions, rewards, actions, state_keys, next_keys


def train_shared_q(
    train_tasks: Sequence[MAPFTask],
    episodes: int,
    horizon: int,
    seed: int,
) -> SharedQLearner:
    rng = random.Random(seed)
    learner = SharedQLearner()
    for episode in range(episodes):
        task = train_tasks[episode % len(train_tasks)]
        positions = list(task.starts)
        learner.epsilon = max(0.02, 0.25 * (1.0 - episode / max(episodes, 1)))
        for _ in range(horizon):
            next_positions, rewards, actions, keys, next_keys = step_joint_policy(
                task.grid, positions, task.goals, learner, rng, train=True
            )
            for key, action, reward, next_key in zip(keys, actions, rewards, next_keys):
                learner.update(key, action, reward, next_key)
            positions = next_positions
            if all(pos == goal for pos, goal in zip(positions, task.goals)):
                break
    learner.epsilon = 0.0
    return learner


def marl_initialize_paths(
    task: MAPFTask,
    learner: SharedQLearner,
    horizon: int,
    rng: random.Random,
) -> List[Path]:
    positions = list(task.starts)
    trajectories = [[pos] for pos in positions]
    for _ in range(horizon):
        positions, _, _, _, _ = step_joint_policy(task.grid, positions, task.goals, learner, rng, train=False)
        for agent, pos in enumerate(positions):
            trajectories[agent].append(pos)
        if all(pos == goal for pos, goal in zip(positions, task.goals)):
            return normalize_paths(trajectories)
    out = []
    for agent, pos in enumerate(positions):
        if pos == task.goals[agent]:
            out.append(trajectories[agent])
            continue
        suffix = shortest_path_static(task.grid, pos, task.goals[agent])
        if suffix is None:
            out.append(trajectories[agent])
        else:
            out.append(trajectories[agent] + suffix[1:])
    return normalize_paths(out)


def independent_shortest_paths(task: MAPFTask) -> List[Path]:
    out = []
    for start, goal in zip(task.starts, task.goals):
        path = shortest_path_static(task.grid, start, goal)
        if path is None:
            raise ValueError(f"Unreachable pair on map {task.map_name}")
        out.append(path)
    return normalize_paths(out)


def first_conflict_time(paths: Sequence[Path], agent: int) -> int:
    max_t = max(len(path) for path in paths)
    for t in range(max_t):
        pos = path_position(paths[agent], t)
        for other in range(len(paths)):
            if other == agent:
                continue
            if path_position(paths[other], t) == pos:
                return t
            if t + 1 < max_t:
                if (
                    path_position(paths[agent], t) == path_position(paths[other], t + 1)
                    and path_position(paths[agent], t + 1) == path_position(paths[other], t)
                ):
                    return t
    return max_t


def select_neighborhood(
    task: MAPFTask,
    paths: Sequence[Path],
    neighborhood_size: int,
    learner: Optional[SharedQLearner],
) -> List[int]:
    conflicts = detect_conflicts(paths)
    if not conflicts:
        return []
    counts = Counter()
    for conflict in conflicts:
        counts[conflict["a1"]] += 1
        counts[conflict["a2"]] += 1
    if learner is None:
        ranked = [agent for agent, _ in counts.most_common()]
    else:
        scored = []
        for agent, count in counts.items():
            t = first_conflict_time(paths, agent)
            pos = path_position(paths[agent], min(t, len(paths[agent]) - 1))
            others = [path_position(paths[o], min(t, len(paths[o]) - 1)) for o in range(len(paths)) if o != agent]
            key = learner.state_key(task.grid, pos, task.goals[agent], others)
            urgency = -float(np.max(learner.q[key]))
            scored.append((-(count + 0.25 * urgency), agent))
        scored.sort()
        ranked = [agent for _, agent in scored]
    seed_agents = ranked[: max(1, min(2, len(ranked)))]
    chosen = list(seed_agents)
    while len(chosen) < neighborhood_size:
        best_agent = None
        best_dist = math.inf
        for candidate in ranked:
            if candidate in chosen:
                continue
            dist = min(
                manhattan(task.starts[candidate], task.starts[anchor]) + manhattan(task.goals[candidate], task.goals[anchor])
                for anchor in seed_agents
            )
            if dist < best_dist:
                best_dist = dist
                best_agent = candidate
        if best_agent is None:
            break
        chosen.append(best_agent)
    return chosen


def repair_with_pp(
    task: MAPFTask,
    paths: List[Path],
    neighborhood: Sequence[int],
    rng: random.Random,
    order_trials: int = 4,
) -> Optional[List[Path]]:
    fixed = {idx: path for idx, path in enumerate(paths) if idx not in neighborhood}
    best_candidate = None
    neighborhood = list(neighborhood)
    base_order = sorted(neighborhood, key=lambda a: manhattan(task.starts[a], task.goals[a]), reverse=True)
    trial_orders = [base_order]
    for _ in range(order_trials - 1):
        trial = list(base_order)
        rng.shuffle(trial)
        trial_orders.append(trial)
    for order in trial_orders:
        candidate = dict(fixed)
        ok = True
        for agent in order:
            reserved = [candidate[idx] for idx in sorted(candidate)]
            path = spacetime_astar(task.grid, task.starts[agent], task.goals[agent], reserved)
            if path is None:
                ok = False
                break
            candidate[agent] = path
        if ok:
            full_paths = [candidate[idx] for idx in range(len(paths))]
            full_paths = normalize_paths(full_paths)
            if best_candidate is None or collision_count(full_paths) < collision_count(best_candidate):
                best_candidate = full_paths
                if collision_count(full_paths) == 0:
                    break
    return best_candidate


def lns_solve(
    task: MAPFTask,
    init_paths: List[Path],
    rng: random.Random,
    learner: Optional[SharedQLearner] = None,
    max_iterations: int = 40,
    neighborhood_size: int = 6,
) -> SolverResult:
    start_time = time.perf_counter()
    current = normalize_paths(init_paths)
    current_conflicts = collision_count(current)
    best = current
    best_conflicts = current_conflicts
    for iteration in range(1, max_iterations + 1):
        if best_conflicts == 0:
            runtime = time.perf_counter() - start_time
            return SolverResult(
                success=True,
                paths=best,
                runtime=runtime,
                sum_of_costs=sum_of_costs(best),
                makespan=makespan(best),
                collision_count=0,
                iterations=iteration - 1,
                method="LNS",
            )
        neighborhood = select_neighborhood(task, current, neighborhood_size, learner if iteration <= max_iterations // 2 else None)
        candidate = repair_with_pp(task, current, neighborhood, rng)
        if candidate is None:
            continue
        cand_conflicts = collision_count(candidate)
        if cand_conflicts <= current_conflicts:
            current = candidate
            current_conflicts = cand_conflicts
        if cand_conflicts < best_conflicts or (
            cand_conflicts == best_conflicts and sum_of_costs(candidate) < sum_of_costs(best)
        ):
            best = candidate
            best_conflicts = cand_conflicts
    runtime = time.perf_counter() - start_time
    success = best_conflicts == 0
    return SolverResult(
        success=success,
        paths=best if success else current,
        runtime=runtime,
        sum_of_costs=sum_of_costs(best if success else current),
        makespan=makespan(best if success else current),
        collision_count=best_conflicts if success else current_conflicts,
        iterations=max_iterations,
        method="LNS",
    )
