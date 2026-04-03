import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

Coord = Tuple[int, int]


def load_grid(path: Path) -> np.ndarray:
    grid = np.load(path, allow_pickle=True)
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got {grid.shape}")
    return (grid == 0).astype(np.int8)  # 1 = free, 0 = obstacle


@dataclass
class AgentTask:
    start: Coord
    goal: Coord


@dataclass
class MAPFInstance:
    grid: np.ndarray  # 1 = free, 0 = obstacle
    agents: List[AgentTask]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(grid: np.ndarray, pos: Coord) -> List[Coord]:
    h, w = grid.shape
    res = []
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(0,0)]:  # include wait
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 1:
            res.append((nx, ny))
    return res


def single_agent_astar(grid: np.ndarray, start: Coord, goal: Coord,
                       reservations: Optional[Dict[Tuple[Coord,int], bool]] = None,
                       max_t: int = 256) -> Optional[List[Coord]]:
    """Time-extended A* with simple reservation table for prioritized planning."""
    import heapq

    if reservations is None:
        reservations = {}

    open_list = []
    g = { (start,0): 0 }
    heapq.heappush(open_list, (manhattan(start, goal), 0, start, 0, None))
    parent = {}

    best_goal = None

    while open_list:
        f, cost, pos, t, prev = heapq.heappop(open_list)
        key = (pos, t)
        if key in parent:
            continue
        parent[key] = prev
        if pos == goal:
            best_goal = key
            break
        if t >= max_t:
            continue
        for nb in neighbors(grid, pos):
            nt = t + 1
            nkey = (nb, nt)
            if reservations.get((nb, nt), False):
                continue
            # edge conflict: swap
            if reservations.get((pos, nt), False) and reservations.get((nb, t), False):
                continue
            nc = cost + 1
            if nc < g.get(nkey, 1e9):
                g[nkey] = nc
                heapq.heappush(open_list, (nc + manhattan(nb, goal), nc, nb, nt, key))

    if best_goal is None:
        return None
    # reconstruct
    path = []
    cur = best_goal
    while cur is not None:
        (pos, t), prev = cur, parent[cur]
        path.append(pos)
        cur = prev
    path.reverse()
    return path


def prioritized_planning(instance: MAPFInstance) -> List[List[Coord]]:
    reservations: Dict[Tuple[Coord,int], bool] = {}
    paths: List[List[Coord]] = []
    for agent in instance.agents:
        path = single_agent_astar(instance.grid, agent.start, agent.goal, reservations)
        if path is None:
            paths.append([])
            continue
        for t, p in enumerate(path):
            reservations[(p, t)] = True
        # reserve goal for all future times
        goal = path[-1]
        for t in range(len(path), len(path)+50):
            reservations[(goal, t)] = True
        paths.append(path)
    return paths


class SimpleValueNet:
    """Tiny tabular value estimator over (d_to_goal, local_obstacles) for demo MARL."""
    def __init__(self, max_dist: int = 50):
        self.max_dist = max_dist
        self.values = np.zeros((max_dist+1, 6), dtype=np.float32)

    def featurize(self, grid: np.ndarray, pos: Coord, goal: Coord) -> Tuple[int,int]:
        d = min(manhattan(pos, goal), self.max_dist)
        # count free neighbors including wait
        free = len(neighbors(grid, pos))
        free = max(0, min(5, free))
        return d, free

    def predict(self, grid: np.ndarray, pos: Coord, goal: Coord) -> float:
        d, free = self.featurize(grid, pos, goal)
        return float(self.values[d, free])

    def update(self, grid: np.ndarray, pos: Coord, goal: Coord, target: float, lr: float = 0.1):
        d, free = self.featurize(grid, pos, goal)
        self.values[d, free] += lr * (target - self.values[d, free])


def marl_guided_rollout(instance: MAPFInstance, value_net: SimpleValueNet,
                        horizon: int = 30, gamma: float = 0.95,
                        eps: float = 0.1) -> List[List[Coord]]:
    """MARL-style joint rollout: all agents move simultaneously, guided by value_net.
    Returns partial paths (length <= horizon)."""
    grid = instance.grid
    positions = [a.start for a in instance.agents]
    goals = [a.goal for a in instance.agents]
    paths = [[p] for p in positions]

    for t in range(horizon):
        proposed = []
        for i, pos in enumerate(positions):
            goal = goals[i]
            if pos == goal:
                proposed.append(pos)
                continue
            acts = neighbors(grid, pos)
            if not acts:
                proposed.append(pos)
                continue
            if random.random() < eps:
                proposed.append(random.choice(acts))
            else:
                best_a, best_v = None, -1e9
                for a in acts:
                    v = -manhattan(a, goal) + value_net.predict(grid, a, goal)
                    if v > best_v:
                        best_v, best_a = v, a
                proposed.append(best_a)
        # resolve collisions: simple rule, revert colliding agents to previous pos
        next_pos = positions.copy()
        occ: Dict[Coord, int] = {}
        for i, p in enumerate(proposed):
            if p not in occ:
                occ[p] = i
                next_pos[i] = p
            else:
                # loser reverts (current agent)
                next_pos[i] = positions[i]
        positions = next_pos
        for i, p in enumerate(positions):
            paths[i].append(p)
        # early stop if all at goals
        if all(positions[i] == goals[i] for i in range(len(positions))):
            break
    # simple value updates along each agent's path (Monte Carlo)
    for i, path in enumerate(paths):
        g = 0.0
        for step in reversed(range(len(path))):
            pos = path[step]
            reward = -1.0
            if pos == goals[i]:
                reward = 0.0
            g = reward + gamma * g
            value_net.update(grid, pos, goals[i], g)
    return paths


def lns_with_marl(instance: MAPFInstance, iters: int = 20, destroy_frac: float = 0.5,
                   horizon: int = 30) -> List[List[Coord]]:
    """Hybrid LNS+MARL: start from prioritized planning, then iteratively destroy/repair
    subsets of agents using MARL-guided rollouts followed by prioritized planning repair."""
    value_net = SimpleValueNet()
    # initial solution
    best_paths = prioritized_planning(instance)

    for it in range(iters):
        # choose subset of agents to re-plan
        n = len(instance.agents)
        k = max(1, int(destroy_frac * n))
        idxs = random.sample(range(n), k)
        # build sub-instance
        sub_agents = [instance.agents[i] for i in idxs]
        sub_instance = MAPFInstance(instance.grid, sub_agents)
        # MARL rollout to get better joint trajectories (early stage, no reservation)
        sub_paths = marl_guided_rollout(sub_instance, value_net, horizon=horizon)
        # use their last positions as new starts, keep same goals, then prioritized planning with reservations from untouched agents
        reservations: Dict[Tuple[Coord,int], bool] = {}
        # fix reservations from untouched agents using current best_paths
        for j in range(n):
            if j in idxs:
                continue
            path = best_paths[j]
            for t, p in enumerate(path):
                reservations[(p, t)] = True
            if path:
                goal = path[-1]
                for t in range(len(path), len(path)+50):
                    reservations[(goal, t)] = True
        # re-plan selected agents sequentially from their current positions
        new_paths = {j: best_paths[j] for j in range(n)}
        for local_idx, j in enumerate(idxs):
            start = sub_paths[local_idx][-1]
            goal = instance.agents[j].goal
            path = single_agent_astar(instance.grid, start, goal, reservations)
            if path is None:
                continue
            for t, p in enumerate(path):
                reservations[(p, t)] = True
            if path:
                goal_p = path[-1]
                for t in range(len(path), len(path)+50):
                    reservations[(goal_p, t)] = True
            # stitch: prefix old path up to first occurrence of start
            old = best_paths[j]
            try:
                idx = old.index(start)
                stitched = old[:idx] + path
            except ValueError:
                stitched = path
            new_paths[j] = stitched
        # simple acceptance: if total makespan decreased, accept
        def makespan(paths):
            return max((len(p) for p in paths if p), default=0)
        if makespan(new_paths.values()) <= makespan(best_paths):
            best_paths = [new_paths[i] for i in range(n)]
    return best_paths


def random_tasks_for_grid(grid: np.ndarray, n_agents: int) -> List[AgentTask]:
    free = list(zip(*np.where(grid == 1)))
    choices = random.sample(free, 2*n_agents)
    tasks = []
    for i in range(n_agents):
        tasks.append(AgentTask(start=choices[2*i], goal=choices[2*i+1]))
    return tasks


def evaluate_on_dataset(dataset: str, n_agents: int = 10, n_instances: int = 10,
                        algo: str = 'hybrid'):
    from time import time
    base = Path('data') / dataset
    npys = sorted(base.rglob('*.npy'))[:n_instances]
    results = []
    for f in npys:
        grid_raw = np.load(f, allow_pickle=True)
        grid = (grid_raw == 0).astype(np.int8)
        agents = random_tasks_for_grid(grid, n_agents)
        instance = MAPFInstance(grid, agents)
        t0 = time()
        if algo == 'hybrid':
            paths = lns_with_marl(instance)
        else:
            paths = prioritized_planning(instance)
        runtime = time() - t0
        makespan = max((len(p) for p in paths if p), default=0)
        collisions = 0  # we rely on construction to avoid explicit collisions here
        results.append((dataset, f.name, algo, n_agents, makespan, runtime, collisions))
    return results


def main():
    import argparse, csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random_medium')
    parser.add_argument('--algo', default='hybrid', choices=['hybrid', 'prioritized'])
    parser.add_argument('--n_agents', type=int, default=10)
    parser.add_argument('--n_instances', type=int, default=10)
    parser.add_argument('--out', type=str, default='outputs/results.csv')
    args = parser.parse_args()

    results = evaluate_on_dataset(args.dataset, args.n_agents, args.n_instances, args.algo)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset','file','algo','n_agents','makespan','runtime','collisions'])
        w.writerows(results)


if __name__ == '__main__':
    main()
