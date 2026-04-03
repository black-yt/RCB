"""
Main experiment runner for MAPF hybrid algorithm evaluation.

Compares:
1. PP (Prioritized Planning) - baseline
2. LNS-PP (Large Neighborhood Search with PP) - MAPF-LNS2 style
3. MARL-Inspired - Phase 1 only (cooperative heuristic)
4. Hybrid-MARL-LNS - Proposed method (Phase 1 + Phase 2)

Experiments are run across multiple map categories, sizes, and agent counts.
"""

import os
import sys
import json
import time
import random
import numpy as np
from typing import List, Dict, Tuple

# Add code directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_instance, get_dataset_info, get_map_files
from hybrid_marl_lns import run_all_algorithms
from mapf_core import solution_cost, solution_makespan, is_solution_valid

WORKSPACE = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_experiment_batch(category: str,
                          map_dirs: List[str],
                          agent_counts: List[int],
                          n_maps_per_dir: int = 3,
                          time_limit: float = 30.0,
                          seeds: List[int] = None) -> List[Dict]:
    """
    Run all algorithms on a batch of instances.

    Returns list of result records.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    results = []
    total_instances = len(map_dirs) * n_maps_per_dir * len(agent_counts) * len(seeds)
    instance_count = 0

    for map_dir in map_dirs:
        map_files = get_map_files(DATA_DIR, map_dir, n_maps_per_dir)

        for map_file in map_files:
            map_name = os.path.basename(map_file)

            for n_agents in agent_counts:
                for seed in seeds:
                    instance_count += 1
                    print(f"  [{instance_count}/{total_instances}] {map_dir}/{map_name} "
                          f"n_agents={n_agents} seed={seed}", end=' ', flush=True)

                    try:
                        instance = load_instance(map_file, n_agents, seed=seed)
                        algo_results = run_all_algorithms(instance, time_limit=time_limit, seed=seed)

                        for algo_name, algo_result in algo_results.items():
                            paths = algo_result.get('paths', [])
                            valid_paths = [p for p in paths if p is not None]

                            record = {
                                'category': category,
                                'map_dir': map_dir,
                                'map_file': map_name,
                                'n_agents': n_agents,
                                'seed': seed,
                                'algorithm': algo_name,
                                'success': algo_result['success'],
                                'conflicts': algo_result['conflicts'],
                                'time': algo_result['time'],
                                'cost': solution_cost(valid_paths) if valid_paths else -1,
                                'makespan': solution_makespan(valid_paths) if valid_paths else -1,
                            }

                            # Algorithm-specific stats
                            if 'iterations' in algo_result:
                                record['iterations'] = algo_result['iterations']
                            if 'rounds' in algo_result:
                                record['rounds'] = algo_result['rounds']
                            if 'initial_conflicts' in algo_result:
                                record['initial_conflicts'] = algo_result['initial_conflicts']
                            if 'phase1_conflicts' in algo_result:
                                record['phase1_conflicts'] = algo_result['phase1_conflicts']

                            results.append(record)

                        # Quick summary
                        successes = sum(1 for r in results[-4:] if r['success'])
                        print(f"-> {successes}/4 succeeded")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    return results


def run_all_experiments():
    """Run the complete experiment suite."""
    print("=" * 70)
    print("MAPF Hybrid MARL-LNS Experiment Suite")
    print("=" * 70)

    datasets, _ = get_dataset_info()

    # Experiment configuration
    # For each category, define which subdirs and agent counts to use
    experiment_config = {
        'random_small': {
            'dirs': datasets['random_small']['dirs'][:2],
            'agent_counts': [3, 5, 8, 10],
            'n_maps': 3,
            'time_limit': 20.0
        },
        'random_medium': {
            'dirs': datasets['random_medium']['dirs'][:2],
            'agent_counts': [5, 10, 15, 20],
            'n_maps': 3,
            'time_limit': 30.0
        },
        'random_large': {
            'dirs': datasets['random_large']['dirs'][:1],
            'agent_counts': [10, 20, 30],
            'n_maps': 2,
            'time_limit': 45.0
        },
        'empty': {
            'dirs': datasets['empty']['dirs'][:1],
            'agent_counts': [10, 20, 30],
            'n_maps': 3,
            'time_limit': 30.0
        },
        'maze': {
            'dirs': datasets['maze']['dirs'][:1],
            'agent_counts': [5, 8, 12],
            'n_maps': 3,
            'time_limit': 30.0
        },
        'room': {
            'dirs': datasets['room']['dirs'][:1],
            'agent_counts': [5, 10, 15],
            'n_maps': 3,
            'time_limit': 30.0
        },
        'warehouse': {
            'dirs': datasets['warehouse']['dirs'][:1],
            'agent_counts': [5, 10, 15],
            'n_maps': 3,
            'time_limit': 30.0
        }
    }

    all_results = []
    seeds = [42, 123, 456]

    for category, cfg in experiment_config.items():
        print(f"\n--- Category: {category.upper()} ---")
        print(f"  Map dirs: {cfg['dirs']}")
        print(f"  Agent counts: {cfg['agent_counts']}")
        print(f"  Time limit: {cfg['time_limit']}s per instance")

        results = run_experiment_batch(
            category=category,
            map_dirs=cfg['dirs'],
            agent_counts=cfg['agent_counts'],
            n_maps_per_dir=cfg['n_maps'],
            time_limit=cfg['time_limit'],
            seeds=seeds
        )

        all_results.extend(results)
        print(f"  Completed {len(results)} result records")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print(f"Total result records: {len(all_results)}")

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results: List[Dict]):
    """Print a summary of experiment results."""
    from collections import defaultdict

    algorithms = ['PP', 'LNS-PP', 'MARL-Inspired', 'Hybrid-MARL-LNS']

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Overall success rate
    print("\nOverall Success Rate by Algorithm:")
    print("-" * 50)
    for algo in algorithms:
        algo_results = [r for r in results if r['algorithm'] == algo]
        if algo_results:
            success_rate = sum(1 for r in algo_results if r['success']) / len(algo_results)
            avg_conflicts = np.mean([r['conflicts'] for r in algo_results])
            avg_time = np.mean([r['time'] for r in algo_results])
            print(f"  {algo:20s}: SR={success_rate:.2%}  "
                  f"Avg Conflicts={avg_conflicts:.1f}  Avg Time={avg_time:.2f}s")

    # By category
    print("\nSuccess Rate by Category:")
    print("-" * 70)
    categories = sorted(set(r['category'] for r in results))
    header = f"{'Category':15s}" + "".join(f"{a:18s}" for a in algorithms)
    print(header)
    print("-" * len(header))

    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        row = f"{cat:15s}"
        for algo in algorithms:
            algo_cat = [r for r in cat_results if r['algorithm'] == algo]
            if algo_cat:
                sr = sum(1 for r in algo_cat if r['success']) / len(algo_cat)
                row += f"  {sr:.1%} ({len(algo_cat):3d})"
            else:
                row += f"  N/A"
        print(row)


if __name__ == '__main__':
    all_results = run_all_experiments()
