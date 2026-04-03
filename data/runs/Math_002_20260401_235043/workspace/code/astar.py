"""
A* and Space-Time A* search algorithms for single-agent pathfinding.
"""

import heapq
from typing import List, Tuple, Dict, Optional, Set
from mapf_core import MAPFInstance, Path


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar(instance: MAPFInstance, start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[Path]:
    """Basic A* for single-agent pathfinding (ignores other agents)."""
    if start == goal:
        return [start]

    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, goal), 0, start, [start]))
    visited = {start: 0}

    while open_set:
        f, g, pos, path = heapq.heappop(open_set)

        if pos == goal:
            return path

        if g > visited.get(pos, float('inf')):
            continue

        for npos in instance.get_neighbors(*pos):
            ng = g + 1
            if ng < visited.get(npos, float('inf')):
                visited[npos] = ng
                h = manhattan_distance(npos, goal)
                heapq.heappush(open_set, (ng + h, ng, npos, path + [npos]))

    return None  # No path found


def space_time_astar(instance: MAPFInstance, start: Tuple[int, int],
                     goal: Tuple[int, int],
                     constraints: Dict,
                     max_time: int = 500) -> Optional[Path]:
    """
    Space-Time A* for single-agent pathfinding with constraints.

    constraints: dict with keys 'vertex' {(r,c,t): True} and 'edge' {(r1,c1,r2,c2,t): True}
    """
    vertex_constraints = constraints.get('vertex', {})
    edge_constraints = constraints.get('edge', {})

    if start == goal and not vertex_constraints.get((start[0], start[1], 0), False):
        # Check if goal is clear at all times
        goal_blocked = any(vertex_constraints.get((goal[0], goal[1], t), False)
                          for t in range(max_time))
        if not goal_blocked:
            return [start]

    # State: (f, g, pos, time)
    open_set = []
    h0 = manhattan_distance(start, goal)
    heapq.heappush(open_set, (h0, 0, 0, start, [start]))

    # visited: (pos, time) -> min_g
    visited = {(start, 0): 0}

    while open_set:
        f, g, t, pos, path = heapq.heappop(open_set)

        if t > max_time:
            continue

        # Check if we reached goal and can stay there
        if pos == goal:
            # Make sure agent can wait at goal without blocking
            return path

        key = (pos, t)
        if g > visited.get(key, float('inf')):
            continue

        # Expand neighbors
        for npos in instance.get_neighbors(*pos):
            nt = t + 1
            if nt > max_time:
                continue

            # Check vertex constraint
            if vertex_constraints.get((npos[0], npos[1], nt), False):
                continue

            # Check edge constraint (swap)
            if edge_constraints.get((npos[0], npos[1], pos[0], pos[1], nt), False):
                continue

            ng = g + 1
            nkey = (npos, nt)
            if ng < visited.get(nkey, float('inf')):
                visited[nkey] = ng
                h = manhattan_distance(npos, goal)
                heapq.heappush(open_set, (ng + h, ng, nt, npos, path + [npos]))

    return None  # No path found


def space_time_astar_with_congestion(instance: MAPFInstance,
                                      start: Tuple[int, int],
                                      goal: Tuple[int, int],
                                      constraints: Dict,
                                      congestion_map: Dict[Tuple, float],
                                      congestion_weight: float = 0.5,
                                      max_time: int = 500) -> Optional[Path]:
    """
    Space-Time A* with congestion penalty to avoid crowded cells.
    Used in the MARL-inspired component.

    congestion_map: {(row, col): congestion_score}
    """
    vertex_constraints = constraints.get('vertex', {})
    edge_constraints = constraints.get('edge', {})

    if start == goal:
        return [start]

    open_set = []
    h0 = manhattan_distance(start, goal)
    heapq.heappush(open_set, (h0, 0.0, 0, start, [start]))

    visited = {(start, 0): 0.0}

    while open_set:
        f, g, t, pos, path = heapq.heappop(open_set)

        if t > max_time:
            continue

        if pos == goal:
            return path

        key = (pos, t)
        if g > visited.get(key, float('inf')):
            continue

        for npos in instance.get_neighbors(*pos):
            nt = t + 1
            if nt > max_time:
                continue

            if vertex_constraints.get((npos[0], npos[1], nt), False):
                continue

            if edge_constraints.get((npos[0], npos[1], pos[0], pos[1], nt), False):
                continue

            # Add congestion penalty
            congestion_penalty = congestion_weight * congestion_map.get(npos, 0.0)
            ng = g + 1.0 + congestion_penalty

            nkey = (npos, nt)
            if ng < visited.get(nkey, float('inf')):
                visited[nkey] = ng
                h = manhattan_distance(npos, goal)
                heapq.heappush(open_set, (ng + h, ng, nt, npos, path + [npos]))

    return None
