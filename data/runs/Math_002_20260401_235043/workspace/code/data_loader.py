"""
Data loading utilities for MAPF benchmarks.
Loads maps and generates agent configurations.
"""

import os
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from mapf_core import MAPFInstance


def load_map(map_path: str) -> np.ndarray:
    """Load a map from a .npy file."""
    return np.load(map_path, allow_pickle=True).astype(np.int32)


def get_free_cells(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Get all free cells in a grid."""
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0]


def generate_agents(grid: np.ndarray, n_agents: int,
                    seed: int = 42) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Generate random start and goal positions for agents.
    Ensures all starts and goals are on free cells and are distinct.
    """
    rng = random.Random(seed)
    free_cells = get_free_cells(grid)

    if len(free_cells) < 2 * n_agents:
        # Not enough free cells - use maximum possible agents
        n_agents = len(free_cells) // 2

    # Sample distinct starts and goals
    selected = rng.sample(free_cells, 2 * n_agents)
    starts = selected[:n_agents]
    goals = selected[n_agents:]

    return starts, goals


def load_instance(map_path: str, n_agents: int, seed: int = 42) -> MAPFInstance:
    """Load a map and create a MAPF instance with random agent configurations."""
    grid = load_map(map_path)
    starts, goals = generate_agents(grid, n_agents, seed)
    return MAPFInstance(grid, starts, goals)


def get_dataset_info() -> Dict:
    """Return dataset configuration for all benchmark categories."""
    base = os.path.join(os.path.dirname(__file__), '..', 'data')

    datasets = {
        'random_small': {
            'dirs': [
                'random_small/maps_50_10_10_0.175',
                'random_small/maps_55_10_10_0.175',
                'random_small/maps_60_10_10_0.175',
                'random_small/maps_65_10_10_0.175',
            ],
            'map_size': (10, 10),
            'agent_counts': [3, 5, 7, 10],
            'description': '10x10 random maps with 17.5% obstacles'
        },
        'random_medium': {
            'dirs': [
                'random_medium/maps_312_25_25_0.175',
                'random_medium/maps_344_25_25_0.175',
                'random_medium/maps_375_25_25_0.175',
                'random_medium/maps_406_25_25_0.175',
            ],
            'map_size': (25, 25),
            'agent_counts': [5, 10, 15, 20],
            'description': '25x25 random maps with 17.5% obstacles'
        },
        'random_large': {
            'dirs': [
                'random_large/maps_1250_50_50_0.175',
                'random_large/maps_1375_50_50_0.175',
            ],
            'map_size': (50, 50),
            'agent_counts': [10, 20, 30],
            'description': '50x50 random maps with 17.5% obstacles'
        },
        'empty': {
            'dirs': [
                'empty/empty_maps_453_25_25',
                'empty/empty_maps_469_25_25',
            ],
            'map_size': (25, 25),
            'agent_counts': [10, 20, 30],
            'description': '25x25 empty maps (no obstacles)'
        },
        'maze': {
            'dirs': [
                'maze/maze_maps_125_25_25',
                'maze/maze_maps_141_25_25',
            ],
            'map_size': (25, 25),
            'agent_counts': [5, 10, 15],
            'description': '25x25 maze maps with complex corridors'
        },
        'room': {
            'dirs': [
                'room/room_maps_250_25_25',
                'room/room_maps_281_25_25',
            ],
            'map_size': (25, 25),
            'agent_counts': [5, 10, 15],
            'description': '25x25 room maps with connected chambers'
        },
        'warehouse': {
            'dirs': [
                'warehouse/warehouse_maps_266_25_25',
                'warehouse/warehouse_maps_281_25_25',
            ],
            'map_size': (25, 25),
            'agent_counts': [5, 10, 15],
            'description': '25x25 warehouse maps with shelf layouts'
        }
    }

    return datasets, base


def get_map_files(data_dir: str, dataset_dir: str, n_files: int = 5) -> List[str]:
    """Get n_files map file paths from a dataset directory."""
    full_dir = os.path.join(data_dir, dataset_dir)
    files = sorted([f for f in os.listdir(full_dir) if f.endswith('.npy')])
    # Use evenly spaced samples
    if len(files) <= n_files:
        return [os.path.join(full_dir, f) for f in files]
    indices = [int(i * len(files) / n_files) for i in range(n_files)]
    return [os.path.join(full_dir, files[i]) for i in indices]


def describe_dataset(base_dir: str) -> Dict:
    """Analyze and describe all available datasets."""
    info = {}
    datasets, _ = get_dataset_info()

    for category, cfg in datasets.items():
        category_info = {
            'description': cfg['description'],
            'map_size': cfg['map_size'],
            'agent_counts_tested': cfg['agent_counts'],
            'subdirs': []
        }

        for d in cfg['dirs']:
            full_path = os.path.join(base_dir, d)
            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.endswith('.npy')]
                # Analyze a sample map
                if files:
                    sample = np.load(os.path.join(full_path, sorted(files)[0]))
                    free_count = int((sample == 0).sum())
                    obs_count = int((sample == -1).sum())
                    category_info['subdirs'].append({
                        'path': d,
                        'n_files': len(files),
                        'sample_free_cells': free_count,
                        'sample_obstacle_cells': obs_count,
                        'sample_obstacle_density': obs_count / (free_count + obs_count)
                    })

        info[category] = category_info

    return info
