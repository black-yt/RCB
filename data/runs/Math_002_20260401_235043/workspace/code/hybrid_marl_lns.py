"""
Hybrid MARL-LNS Algorithm for MAPF.

Core Innovation: Combines MARL-inspired cooperative planning (Phase 1)
with LNS-Prioritized Planning (Phase 2) in a two-phase hybrid approach.

Phase 1 (MARL-inspired):
    - Use congestion-aware cooperative pathfinding to generate a high-quality
      initial solution with fewer collisions
    - Multiple cooperative rounds reduce initial conflict count significantly
    - This mirrors how trained MARL policies produce collision-aware paths

Phase 2 (LNS-PP):
    - Take the MARL-inspired solution as initial solution for LNS
    - Apply LNS with Prioritized Planning to repair remaining collisions
    - Starting from a lower-collision solution means faster convergence

The key insight: MARL-inspired Phase 1 gives LNS a better starting point,
leading to fewer iterations needed in Phase 2 and higher overall success rates.
"""

import time
import random
from typing import List, Tuple, Optional, Dict
import numpy as np

from mapf_core import MAPFInstance, Path, count_conflicts, is_solution_valid
from marl_inspired import marl_inspired_planning
from lns_pp import lns_pp
from prioritized_planning import prioritized_planning


def hybrid_marl_lns(instance: MAPFInstance,
                     # Phase 1 parameters (MARL-inspired)
                     marl_rounds: int = 3,
                     marl_congestion_weight: float = 1.0,
                     marl_time_budget: float = 0.4,  # fraction of total time
                     # Phase 2 parameters (LNS-PP)
                     lns_max_iterations: int = 100,
                     lns_neighborhood_size: int = 5,
                     # Overall parameters
                     time_limit: float = 60.0,
                     seed: int = 42) -> Tuple[List[Optional[Path]], Dict]:
    """
    Hybrid MARL-LNS algorithm.

    Args:
        instance: MAPF problem instance
        marl_rounds: Number of cooperative replanning rounds in Phase 1
        marl_congestion_weight: Weight for congestion penalty in Phase 1
        marl_time_budget: Fraction of total time allocated to Phase 1
        lns_max_iterations: Max LNS iterations in Phase 2
        lns_neighborhood_size: Neighborhood size for LNS
        time_limit: Total time limit in seconds
        seed: Random seed

    Returns:
        (best_paths, stats)
    """
    start_time = time.time()
    stats = {
        'phase1_conflicts': 0,
        'phase1_time': 0.0,
        'phase2_conflicts': 0,
        'phase2_time': 0.0,
        'final_conflicts': 0,
        'total_time': 0.0,
        'success': False,
        'phase1_initial_conflicts': 0,
        'conflict_history': [],
        'phase1_rounds': 0,
        'phase2_iterations': 0,
        'phase2_improvements': 0
    }

    # ===== PHASE 1: MARL-Inspired Cooperative Planning =====
    phase1_time_limit = time_limit * marl_time_budget
    phase1_paths, phase1_stats = marl_inspired_planning(
        instance,
        max_rounds=marl_rounds,
        congestion_weight=marl_congestion_weight,
        time_limit=phase1_time_limit,
        seed=seed
    )

    stats['phase1_initial_conflicts'] = phase1_stats['initial_conflicts']
    stats['phase1_conflicts'] = phase1_stats['final_conflicts']
    stats['phase1_time'] = phase1_stats['time']
    stats['phase1_rounds'] = phase1_stats['rounds']
    stats['conflict_history'].extend(phase1_stats['conflict_history'])

    if phase1_stats['success']:
        stats['final_conflicts'] = 0
        stats['success'] = True
        stats['total_time'] = time.time() - start_time
        return phase1_paths, stats

    # ===== PHASE 2: LNS with Prioritized Planning =====
    remaining_time = time_limit - (time.time() - start_time)

    if remaining_time <= 0:
        stats['final_conflicts'] = phase1_stats['final_conflicts']
        stats['total_time'] = time.time() - start_time
        return phase1_paths, stats

    phase2_paths, phase2_stats = lns_pp(
        instance,
        initial_paths=phase1_paths,
        max_iterations=lns_max_iterations,
        neighborhood_size=lns_neighborhood_size,
        time_limit=remaining_time,
        seed=seed
    )

    stats['phase2_conflicts'] = phase2_stats['final_conflicts']
    stats['phase2_time'] = phase2_stats['time']
    stats['phase2_iterations'] = phase2_stats['iterations']
    stats['phase2_improvements'] = phase2_stats['improvements']
    stats['conflict_history'].extend(phase2_stats['conflict_history'][1:])  # Skip duplicate

    # Keep best solution
    phase1_final_conflicts = count_conflicts([p for p in phase1_paths if p is not None])
    phase2_final_conflicts = phase2_stats['final_conflicts']

    if phase2_final_conflicts <= phase1_final_conflicts:
        best_paths = phase2_paths
        stats['final_conflicts'] = phase2_final_conflicts
    else:
        best_paths = phase1_paths
        stats['final_conflicts'] = phase1_final_conflicts

    stats['success'] = (stats['final_conflicts'] == 0)
    stats['total_time'] = time.time() - start_time

    return best_paths, stats


def run_all_algorithms(instance: MAPFInstance,
                       time_limit: float = 30.0,
                       seed: int = 42) -> Dict:
    """
    Run all algorithms on a MAPF instance and return comparative results.

    Algorithms:
    1. PP (Prioritized Planning baseline)
    2. LNS-PP (LNS with Prioritized Planning)
    3. MARL-Inspired (Phase 1 only)
    4. Hybrid MARL-LNS (proposed)

    Returns dict with results for each algorithm.
    """
    results = {}

    # 1. Prioritized Planning (baseline)
    t0 = time.time()
    pp_paths = prioritized_planning(instance, seed=seed)
    pp_time = time.time() - t0
    pp_conflicts = count_conflicts([p for p in pp_paths if p is not None])

    results['PP'] = {
        'paths': pp_paths,
        'conflicts': pp_conflicts,
        'time': pp_time,
        'success': pp_conflicts == 0
    }

    # 2. LNS-PP
    lns_paths, lns_stats = lns_pp(
        instance,
        initial_paths=None,
        max_iterations=100,
        neighborhood_size=max(5, instance.n_agents // 5),
        time_limit=time_limit,
        seed=seed
    )
    results['LNS-PP'] = {
        'paths': lns_paths,
        'conflicts': lns_stats['final_conflicts'],
        'time': lns_stats['time'],
        'success': lns_stats['success'],
        'iterations': lns_stats['iterations'],
        'initial_conflicts': lns_stats['initial_conflicts'],
        'improvements': lns_stats['improvements'],
        'conflict_history': lns_stats['conflict_history']
    }

    # 3. MARL-Inspired only (Phase 1)
    marl_paths, marl_stats = marl_inspired_planning(
        instance,
        max_rounds=5,
        congestion_weight=1.5,
        time_limit=time_limit,
        seed=seed
    )
    results['MARL-Inspired'] = {
        'paths': marl_paths,
        'conflicts': marl_stats['final_conflicts'],
        'time': marl_stats['time'],
        'success': marl_stats['success'],
        'rounds': marl_stats['rounds'],
        'initial_conflicts': marl_stats['initial_conflicts'],
        'conflict_history': marl_stats['conflict_history']
    }

    # 4. Hybrid MARL-LNS (proposed)
    hybrid_paths, hybrid_stats = hybrid_marl_lns(
        instance,
        marl_rounds=3,
        marl_congestion_weight=1.5,
        marl_time_budget=0.3,
        lns_max_iterations=100,
        lns_neighborhood_size=max(5, instance.n_agents // 5),
        time_limit=time_limit,
        seed=seed
    )
    results['Hybrid-MARL-LNS'] = {
        'paths': hybrid_paths,
        'conflicts': hybrid_stats['final_conflicts'],
        'time': hybrid_stats['total_time'],
        'success': hybrid_stats['success'],
        'phase1_conflicts': hybrid_stats['phase1_conflicts'],
        'phase2_conflicts': hybrid_stats['phase2_conflicts'],
        'conflict_history': hybrid_stats['conflict_history'],
        'phase1_time': hybrid_stats['phase1_time'],
        'phase2_time': hybrid_stats['phase2_time']
    }

    return results
