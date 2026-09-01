"""Utility functions for deterministic maze generation."""

from collections import deque

import numpy as np


EMPTY = 0
WALL = 1
PLAYER = 2
GOAL = 3

CARDINAL_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

MAZE_DIRECTIONS = tuple(
    (2 * row_change, 2 * col_change)
    for row_change, col_change in CARDINAL_DIRECTIONS
)


def create_rng(seed):
    """Create a local random generator so a seed always recreates a maze."""
    return np.random.default_rng(seed)


def create_wall_grid(width, height):
    """Create the initial grid, in which every position is a wall."""
    if width < 3 or height < 3:
        raise ValueError("Maze dimensions must be at least 3x3.")

    return np.full((height, width), WALL, dtype=np.int8)


def get_logical_cells(width, height, row_offset=0, col_offset=0):
    """Return the cells that DFS connects for one lattice alignment."""
    if row_offset not in (0, 1) or col_offset not in (0, 1):
        raise ValueError("Logical-cell offsets must be either 0 or 1.")

    return [
        (row, col)
        for row in range(row_offset, height, 2)
        for col in range(col_offset, width, 2)
    ]


def choose_random_position(positions, rng):
    """Choose one position using the supplied seeded random generator."""
    if not positions:
        raise ValueError("Cannot choose a position from an empty collection.")

    return positions[int(rng.integers(len(positions)))]


def choose_lattice_offsets(width, height, rng):
    """Choose a valid seeded alignment so no border side is always reserved."""
    valid_offsets = []

    for row_offset in (0, 1):
        for col_offset in (0, 1):
            cells = get_logical_cells(
                width,
                height,
                row_offset,
                col_offset,
            )
            if len(cells) >= 2: # when the size is 3x3
                valid_offsets.append((row_offset, col_offset))

    return choose_random_position(valid_offsets, rng)


def carve_passages(grid, logical_cells, start, rng):
    """Connect all logical cells using randomized depth-first search."""
    cell_set = set(logical_cells)
    if start not in cell_set:
        raise ValueError("The generation start must be a logical maze cell.")

    grid[start] = EMPTY
    visited = {start}
    stack = [start]

    while stack:
        row, col = stack[-1]
        unvisited_neighbors = []

        for row_change, col_change in MAZE_DIRECTIONS:
            neighbor = (row + row_change, col + col_change)
            if neighbor in cell_set and neighbor not in visited:
                unvisited_neighbors.append(neighbor)

        if not unvisited_neighbors:
            stack.pop()
            continue

        next_cell = choose_random_position(unvisited_neighbors, rng)

        visited.add(next_cell)
        stack.append(next_cell)

        # Open the wall between the current logical cell and the next one.
        next_row, next_col = next_cell
        grid[(row + next_row) // 2, (col + next_col) // 2] = EMPTY
        grid[next_cell] = EMPTY

    return grid


def get_empty_positions(grid):
    """Return all currently empty grid positions as coordinate tuples."""
    return [tuple(position) for position in np.argwhere(grid == EMPTY)]


def find_reachable_positions(grid, start, passable_values=(EMPTY,)):
    """Return the shortest distance from start to every reachable position."""
    height, width = grid.shape
    passable_values = set(passable_values)

    if int(grid[start]) not in passable_values:
        raise ValueError("The search start must be on a passable tile.")

    queue = deque([start])
    distances = {start: 0}

    while queue:
        row, col = queue.popleft()

        for row_change, col_change in CARDINAL_DIRECTIONS:
            next_row = row + row_change
            next_col = col + col_change
            next_position = (next_row, next_col)

            inside_grid = 0 <= next_row < height and 0 <= next_col < width
            if (
                inside_grid
                and int(grid[next_position]) in passable_values
                and next_position not in distances
            ):
                distances[next_position] = distances[(row, col)] + 1
                queue.append(next_position)

    return distances


def choose_goal_position(distances, player_position, rng):
    """Choose a farthest reachable position, never the player's position."""
    candidates = [
        position for position in distances if position != player_position
    ]
    if not candidates:
        raise RuntimeError("The generated maze has no reachable goal position.")

    greatest_distance = max(distances[position] for position in candidates)
    farthest_positions = [
        position
        for position in candidates
        if distances[position] == greatest_distance
    ]
    return choose_random_position(farthest_positions, rng)


def place_player_and_goal(grid, player_position, goal_position):
    """Place the player and goal after their connectivity has been proven."""
    result = grid.copy()
    result[player_position] = PLAYER
    result[goal_position] = GOAL
    return result


def validate_player_goal_path(grid, player_position, goal_position):
    """Raise an error unless at least one player-to-goal path exists."""
    distances = find_reachable_positions(
        grid,
        player_position,
        passable_values=(EMPTY, PLAYER, GOAL),
    )
    if goal_position not in distances:
        raise RuntimeError("No path exists from the player to the goal.")

    return distances[goal_position]

def filter_repeated_mazes(mazes, ignore_player_and_goal=False):
    """Return unique mazes while preserving their original order.

    By default, the complete grid is compared. If ``ignore_player_and_goal``
    is true, mazes with the same walls and passages are considered equal even
    when their player or goal positions differ.
    """
    mazes = np.asarray(mazes)
    if mazes.ndim != 3:
        raise ValueError("mazes must have shape (count, height, width)")

    seen = set()
    unique_indices = []

    for index, maze in enumerate(mazes):
        if ignore_player_and_goal:
            comparable = maze.copy()
            comparable[(comparable == PLAYER) | (comparable == GOAL)] = EMPTY
        else:
            comparable = maze

        # Include shape and dtype so the byte representation is unambiguous.
        key = (comparable.shape, comparable.dtype.str, comparable.tobytes())
        if key not in seen:
            seen.add(key)
            unique_indices.append(index)

    return mazes[np.asarray(unique_indices, dtype=np.intp)]
