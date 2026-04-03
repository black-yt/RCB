"""
Core MAPF data structures and utilities.
Multi-Agent Path Finding - Core Module
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class State:
    """A space-time state for an agent."""
    row: int
    col: int
    time: int

    def pos(self):
        return (self.row, self.col)

    def __hash__(self):
        return hash((self.row, self.col, self.time))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col and self.time == other.time

    def __lt__(self, other):
        return self.time < other.time


# Agent path: list of (row, col) positions, one per timestep
Path = List[Tuple[int, int]]


@dataclass
class Conflict:
    """Represents a conflict between two agents."""
    type: str          # 'vertex' or 'edge'
    agent1: int
    agent2: int
    time: int
    pos1: Tuple[int, int]
    pos2: Optional[Tuple[int, int]] = None  # For edge conflicts


class MAPFInstance:
    """Represents a MAPF problem instance."""

    def __init__(self, grid: np.ndarray, starts: List[Tuple[int, int]], goals: List[Tuple[int, int]]):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.starts = starts
        self.goals = goals
        self.n_agents = len(starts)

    def is_free(self, row: int, col: int) -> bool:
        """Check if a cell is traversable."""
        return (0 <= row < self.rows and 0 <= col < self.cols
                and self.grid[row, col] == 0)

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighbors (including stay-in-place)."""
        moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        return [(row + dr, col + dc) for dr, dc in moves
                if self.is_free(row + dr, col + dc)]

    def free_cells(self) -> List[Tuple[int, int]]:
        """Get all free cells."""
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols) if self.grid[r, c] == 0]


def detect_conflicts(paths: List[Path]) -> List[Conflict]:
    """Detect all vertex and edge conflicts in a solution."""
    conflicts = []
    n = len(paths)
    if n == 0:
        return conflicts

    max_t = max(len(p) for p in paths)

    for i in range(n):
        for j in range(i + 1, n):
            path_i = paths[i]
            path_j = paths[j]

            for t in range(max_t):
                pos_i = path_i[min(t, len(path_i) - 1)]
                pos_j = path_j[min(t, len(path_j) - 1)]

                # Vertex conflict
                if pos_i == pos_j:
                    conflicts.append(Conflict('vertex', i, j, t, pos_i))

                # Edge conflict (swap)
                if t > 0:
                    prev_i = path_i[min(t - 1, len(path_i) - 1)]
                    prev_j = path_j[min(t - 1, len(path_j) - 1)]
                    if pos_i == prev_j and pos_j == prev_i:
                        conflicts.append(Conflict('edge', i, j, t, pos_i, pos_j))

    return conflicts


def count_conflicts(paths: List[Path]) -> int:
    """Count total number of conflicts."""
    return len(detect_conflicts(paths))


def get_agent_conflicts(paths: List[Path]) -> Dict[int, int]:
    """Get conflict count per agent."""
    conflicts = detect_conflicts(paths)
    agent_conflicts = {i: 0 for i in range(len(paths))}
    for c in conflicts:
        agent_conflicts[c.agent1] += 1
        agent_conflicts[c.agent2] += 1
    return agent_conflicts


def build_constraint_table(paths: List[Path], agent_ids: List[int]) -> Dict:
    """Build constraint table from existing paths for space-time A*."""
    constraints = {}  # (row, col, time) -> True: vertex constraint
    edge_constraints = {}  # (row1, col1, row2, col2, time) -> True: edge constraint

    for aid, path in zip(agent_ids, paths):
        for t, pos in enumerate(path):
            constraints[(pos[0], pos[1], t)] = True
            # Stay-in-goal constraint (agent stays at goal forever)
            if t == len(path) - 1:
                for future_t in range(t + 1, t + 200):
                    constraints[(pos[0], pos[1], future_t)] = True

        # Edge constraints
        for t in range(1, len(path)):
            prev = path[t - 1]
            curr = path[t]
            edge_constraints[(curr[0], curr[1], prev[0], prev[1], t)] = True

    return {'vertex': constraints, 'edge': edge_constraints}


def solution_cost(paths: List[Path]) -> int:
    """Sum of individual path costs."""
    return sum(len(p) - 1 for p in paths)


def solution_makespan(paths: List[Path]) -> int:
    """Maximum path length (makespan)."""
    return max(len(p) - 1 for p in paths) if paths else 0


def is_solution_valid(paths: List[Path]) -> bool:
    """Check if the solution has no conflicts."""
    return len(detect_conflicts(paths)) == 0
