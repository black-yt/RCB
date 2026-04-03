"""
MARL-Inspired Cooperative Heuristic for MAPF.

Simulates the behavior of a trained MARL policy using:
1. Congestion-aware pathfinding: agents penalize cells with high traffic
2. Multi-round cooperative replanning: agents iteratively update paths
   based on others' planned paths (similar to decentralized MARL iterations)
3. Conflict-driven priority updates: agents with more conflicts get
   replanned with higher priority

This approximates the collision-reduction behavior of MARL policies
without requiring actual neural network training.
"""

import random
import time
from typing import List, Tuple, Optional, Dict
import numpy as np

from mapf_core import (MAPFInstance, Path, detect_conflicts, count_conflicts,
                        get_agent_conflicts)
from astar import space_time_astar_with_congestion, space_time_astar


def build_congestion_map(instance: MAPFInstance,
                          paths: List[Optional[Path]],
                          agent_id: int) -> Dict[Tuple[int, int], float]:
    """
    Build a congestion map based on all agents' paths (excluding current agent).
    Higher value = more agents pass through this cell = more congested.
    """
    congestion = {}
    for i, path in enumerate(paths):
        if i == agent_id or path is None:
            continue
        for pos in path:
            congestion[pos] = congestion.get(pos, 0.0) + 1.0

    # Normalize
    if congestion:
        max_val = max(congestion.values())
        if max_val > 0:
            for k in congestion:
                congestion[k] /= max_val

    return congestion


def marl_inspired_planning(instance: MAPFInstance,
                             max_rounds: int = 3,
                             congestion_weight: float = 1.0,
                             time_limit: float = 30.0,
                             seed: int = 42) -> Tuple[List[Optional[Path]], Dict]:
    """
    MARL-inspired cooperative pathfinding.

    Simulates multi-agent reinforcement learning behavior through:
    1. Initial independent pathfinding
    2. Iterative replanning with congestion awareness
    3. Conflict-driven priority re-ordering

    Returns paths and statistics.
    """
    rng = random.Random(seed)
    start_time = time.time()
    n = instance.n_agents

    stats = {
        'initial_conflicts': 0,
        'final_conflicts': 0,
        'rounds': 0,
        'time': 0.0,
        'success': False,
        'conflict_history': []
    }

    # Round 0: Independent A* paths (no cooperation)
    paths = []
    for i in range(n):
        path = space_time_astar(instance, instance.starts[i], instance.goals[i],
                                 {'vertex': {}, 'edge': {}}, max_time=300)
        if path is None:
            path = [instance.starts[i]]
        paths.append(path)

    initial_conflicts = count_conflicts(paths)
    stats['initial_conflicts'] = initial_conflicts
    stats['conflict_history'].append(initial_conflicts)

    best_paths = list(paths)
    best_conflicts = initial_conflicts

    if best_conflicts == 0:
        stats['success'] = True
        stats['final_conflicts'] = 0
        stats['time'] = time.time() - start_time
        return best_paths, stats

    # Iterative cooperative replanning rounds
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_limit:
            break

        # Determine agent priorities based on conflicts
        agent_conflicts = get_agent_conflicts(paths)
        # Sort: most conflicted agents replan first (like high-priority agents in MARL)
        agent_order = sorted(range(n), key=lambda i: -agent_conflicts.get(i, 0))

        new_paths = list(paths)

        for agent_id in agent_order:
            if time.time() - start_time > time_limit:
                break

            # Build congestion map from OTHER agents' current paths
            congestion = build_congestion_map(instance, new_paths, agent_id)

            # Build constraints from higher-priority agents (already replanned)
            from mapf_core import build_constraint_table
            higher_priority = agent_order[:agent_order.index(agent_id)]
            hp_paths = [new_paths[i] for i in higher_priority]
            constraints = build_constraint_table(hp_paths, higher_priority)

            # Replan with congestion awareness
            path = space_time_astar_with_congestion(
                instance,
                instance.starts[agent_id],
                instance.goals[agent_id],
                constraints,
                congestion,
                congestion_weight=congestion_weight,
                max_time=300
            )

            if path is not None:
                new_paths[agent_id] = path

        new_conflicts = count_conflicts(new_paths)
        stats['conflict_history'].append(new_conflicts)

        if new_conflicts < best_conflicts:
            best_paths = list(new_paths)
            best_conflicts = new_conflicts

        paths = new_paths
        stats['rounds'] = round_idx + 1

        if best_conflicts == 0:
            break

    stats['final_conflicts'] = best_conflicts
    stats['success'] = (best_conflicts == 0)
    stats['time'] = time.time() - start_time

    return best_paths, stats
