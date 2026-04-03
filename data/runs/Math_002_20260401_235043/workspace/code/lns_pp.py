"""
Large Neighborhood Search with Prioritized Planning (LNS-PP) for MAPF.
Based on MAPF-LNS2: Fast Repairing for Multi-Agent Path Finding via Large Neighborhood Search.

Key idea: Start with a PP solution (may have collisions), then iteratively
repair neighborhoods of conflicting agents using Prioritized Planning.
"""

import random
import time
from typing import List, Tuple, Optional, Dict
import numpy as np

from mapf_core import (MAPFInstance, Path, detect_conflicts, count_conflicts,
                        get_agent_conflicts, is_solution_valid)
from prioritized_planning import prioritized_planning, prioritized_planning_subset


def select_neighborhood_collision(paths: List[Optional[Path]],
                                   n_agents: int,
                                   neighborhood_size: int,
                                   rng: random.Random) -> List[int]:
    """
    Select a neighborhood of agents to replan based on collision agents.
    Prioritizes agents involved in the most conflicts.
    """
    agent_conflicts = get_agent_conflicts([p for p in paths if p is not None])
    # Map back to original indices
    valid_indices = [i for i, p in enumerate(paths) if p is not None]

    # Sort by conflict count (most conflicted first)
    conflicted = [(i, agent_conflicts.get(j, 0))
                  for j, i in enumerate(valid_indices)
                  if agent_conflicts.get(j, 0) > 0]

    if not conflicted:
        # No conflicts - select random neighborhood
        return rng.sample(range(n_agents), min(neighborhood_size, n_agents))

    # Pick the most conflicted agent as seed
    conflicted.sort(key=lambda x: -x[1])
    seed_agent = conflicted[0][0]

    # Build neighborhood around seed agent
    neighborhood = {seed_agent}

    # Add agents conflicting with seed
    conflicts = detect_conflicts(paths)
    related_agents = set()
    for c in conflicts:
        if c.agent1 == seed_agent:
            related_agents.add(c.agent2)
        elif c.agent2 == seed_agent:
            related_agents.add(c.agent1)

    for a in related_agents:
        if len(neighborhood) >= neighborhood_size:
            break
        neighborhood.add(a)

    # Fill remaining with other conflicted agents
    for a, _ in conflicted:
        if len(neighborhood) >= neighborhood_size:
            break
        neighborhood.add(a)

    # Fill remaining with random agents
    all_agents = list(range(n_agents))
    rng.shuffle(all_agents)
    for a in all_agents:
        if len(neighborhood) >= neighborhood_size:
            break
        neighborhood.add(a)

    return list(neighborhood)


def lns_pp(instance: MAPFInstance,
           initial_paths: Optional[List[Optional[Path]]] = None,
           max_iterations: int = 100,
           neighborhood_size: int = 5,
           time_limit: float = 60.0,
           seed: int = 42) -> Tuple[List[Optional[Path]], Dict]:
    """
    LNS with Prioritized Planning.

    Args:
        instance: MAPF problem instance
        initial_paths: Initial solution (if None, use PP to generate)
        max_iterations: Maximum LNS iterations
        neighborhood_size: Number of agents to replan per iteration
        time_limit: Time limit in seconds
        seed: Random seed

    Returns:
        (best_paths, stats_dict)
    """
    rng = random.Random(seed)
    start_time = time.time()
    stats = {
        'initial_conflicts': 0,
        'final_conflicts': 0,
        'iterations': 0,
        'improvements': 0,
        'time': 0.0,
        'success': False,
        'conflict_history': []
    }

    # Generate initial solution
    if initial_paths is None:
        initial_paths = prioritized_planning(instance, seed=seed)

    best_paths = list(initial_paths)
    best_conflicts = count_conflicts([p for p in best_paths if p is not None])
    stats['initial_conflicts'] = best_conflicts
    stats['conflict_history'].append(best_conflicts)

    if best_conflicts == 0:
        stats['success'] = True
        stats['final_conflicts'] = 0
        stats['time'] = time.time() - start_time
        return best_paths, stats

    # LNS main loop
    current_paths = list(best_paths)
    current_conflicts = best_conflicts

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break

        # Select neighborhood
        neighborhood = select_neighborhood_collision(
            current_paths, instance.n_agents, neighborhood_size, rng)

        # Replan neighborhood using PP
        new_paths = prioritized_planning_subset(
            instance, neighborhood, current_paths,
            agent_order=None, max_time=300)

        new_conflicts = count_conflicts([p for p in new_paths if p is not None])

        # Accept if improvement (or equal)
        if new_conflicts <= current_conflicts:
            current_paths = new_paths
            current_conflicts = new_conflicts

            if new_conflicts < best_conflicts:
                best_paths = list(new_paths)
                best_conflicts = new_conflicts
                stats['improvements'] += 1

        stats['conflict_history'].append(best_conflicts)
        stats['iterations'] = iteration + 1

        if best_conflicts == 0:
            break

    stats['final_conflicts'] = best_conflicts
    stats['success'] = (best_conflicts == 0)
    stats['time'] = time.time() - start_time

    return best_paths, stats
