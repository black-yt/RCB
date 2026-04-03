"""
Prioritized Planning (PP) for MAPF.
Agents plan one-by-one in priority order; each agent avoids previously planned paths.
"""

import random
from typing import List, Tuple, Optional, Dict
import numpy as np

from mapf_core import MAPFInstance, Path, build_constraint_table
from astar import space_time_astar


def prioritized_planning(instance: MAPFInstance,
                          agent_order: Optional[List[int]] = None,
                          max_time: int = 500,
                          seed: int = 42) -> List[Optional[Path]]:
    """
    Basic Prioritized Planning: plan agents one by one in priority order.

    Returns a list of paths (None if no path found for that agent).
    """
    rng = random.Random(seed)
    n = instance.n_agents

    if agent_order is None:
        # Default: order by shortest individual path (proxy for difficulty)
        from astar import astar
        path_lengths = []
        for i in range(n):
            path = astar(instance, instance.starts[i], instance.goals[i])
            path_lengths.append(len(path) if path else float('inf'))
        agent_order = sorted(range(n), key=lambda i: path_lengths[i])

    paths = [None] * n
    planned_paths = []
    planned_ids = []

    for agent_id in agent_order:
        start = instance.starts[agent_id]
        goal = instance.goals[agent_id]

        # Build constraints from already-planned paths
        constraints = build_constraint_table(planned_paths, planned_ids)

        path = space_time_astar(instance, start, goal, constraints, max_time)

        if path is None:
            # No path found - assign trivial stay-in-place path
            paths[agent_id] = [start] * (max_time // 10)
        else:
            paths[agent_id] = path
            planned_paths.append(path)
            planned_ids.append(agent_id)

    return paths


def prioritized_planning_subset(instance: MAPFInstance,
                                 subset_agents: List[int],
                                 fixed_paths: List[Optional[Path]],
                                 agent_order: Optional[List[int]] = None,
                                 max_time: int = 500) -> List[Optional[Path]]:
    """
    Run prioritized planning on a subset of agents, treating others as fixed obstacles.

    subset_agents: agents to replan
    fixed_paths: current paths for ALL agents (subset agents will be replanned)
    """
    n = instance.n_agents
    paths = list(fixed_paths)  # copy

    if agent_order is None:
        agent_order = subset_agents

    # Build constraints from fixed agents (not in subset)
    fixed_agent_ids = [i for i in range(n) if i not in set(subset_agents)]
    fixed_agent_paths = [fixed_paths[i] for i in fixed_agent_ids if fixed_paths[i] is not None]

    # Build initial constraint table from fixed agents
    base_constraints_v = {}
    base_constraints_e = {}

    for path in fixed_agent_paths:
        if path is None:
            continue
        for t, pos in enumerate(path):
            base_constraints_v[(pos[0], pos[1], t)] = True
            # Stay at goal
            if t == len(path) - 1:
                for future_t in range(t + 1, t + max_time):
                    base_constraints_v[(pos[0], pos[1], future_t)] = True
        for t in range(1, len(path)):
            prev = path[t - 1]
            curr = path[t]
            base_constraints_e[(curr[0], curr[1], prev[0], prev[1], t)] = True

    # Plan each subset agent in order
    newly_planned = []
    newly_planned_ids = []

    for agent_id in agent_order:
        start = instance.starts[agent_id]
        goal = instance.goals[agent_id]

        # Constraints from fixed agents + previously replanned subset agents
        constraints_v = dict(base_constraints_v)
        constraints_e = dict(base_constraints_e)

        for path, aid in zip(newly_planned, newly_planned_ids):
            if path is None:
                continue
            for t, pos in enumerate(path):
                constraints_v[(pos[0], pos[1], t)] = True
                if t == len(path) - 1:
                    for future_t in range(t + 1, t + max_time):
                        constraints_v[(pos[0], pos[1], future_t)] = True
            for t in range(1, len(path)):
                prev = path[t - 1]
                curr = path[t]
                constraints_e[(curr[0], curr[1], prev[0], prev[1], t)] = True

        constraints = {'vertex': constraints_v, 'edge': constraints_e}
        path = space_time_astar(instance, start, goal, constraints, max_time)

        if path is None:
            paths[agent_id] = [start] * 10
        else:
            paths[agent_id] = path
            newly_planned.append(path)
            newly_planned_ids.append(agent_id)

    return paths
