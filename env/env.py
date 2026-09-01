import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from visualization import plot_maze

try:
    from .util import (
        EMPTY,
        create_rng,
        create_wall_grid,
        get_logical_cells,
        choose_random_position,
        choose_lattice_offsets,
        carve_passages,
        get_empty_positions,
        find_reachable_positions,
        choose_goal_position,
        place_player_and_goal,
        validate_player_goal_path,
        filter_repeated_mazes
    )
except ImportError:  # Support running this file directly.
    from util import (
        EMPTY,
        create_rng,
        create_wall_grid,
        get_logical_cells,
        choose_random_position,
        choose_lattice_offsets,
        carve_passages,
        get_empty_positions,
        find_reachable_positions,
        choose_goal_position,
        place_player_and_goal,
        validate_player_goal_path,
        filter_repeated_mazes
    )

class Maze(gym.Env):
    def __init__(self, width=10, height=10, total_samples=1000):
        self.width = width
        self.height = height
        self.total_samples = total_samples

        self.observation_space = ...
        self.action_space = ...

        self.generate_maze()

    def _generate_maze(self, seed):
        """
        Generate one deterministic maze from `seed`.

        Cell meanings:
            0 = empty
            1 = wall
            2 = player
            3 = goal
        """
        rng = create_rng(seed)
        maze = create_wall_grid(self.width, self.height)

        row_offset, col_offset = choose_lattice_offsets(
            self.width,
            self.height,
            rng,
        )
        logical_cells = get_logical_cells(
            self.width,
            self.height,
            row_offset,
            col_offset,
        )
        generation_start = choose_random_position(logical_cells, rng)
        carve_passages(maze, logical_cells, generation_start, rng)

        # The player's position is independent of the DFS generation start.
        empty_positions = get_empty_positions(maze)
        player_position = choose_random_position(empty_positions, rng)

        reachable_positions = find_reachable_positions(
            maze,
            player_position,
            passable_values=(EMPTY,),
        )
        goal_position = choose_goal_position(
            reachable_positions,
            player_position,
            rng,
        )

        maze = place_player_and_goal(
            maze,
            player_position,
            goal_position,
        )
        validate_player_goal_path(maze, player_position, goal_position)

        return maze

    def generate_maze(self,):
        '''
        Generate mazes with the number specified by total_samples. 
        Each maze is generated with a different random seed to ensure diversity. 
        The generated mazes are stored in an array for later use.
        '''
        mazes = np.empty((self.total_samples, self.height, self.width), dtype=np.int8)
        for i in tqdm(range(self.total_samples), desc="Generating mazes"):
            maze = self._generate_maze(seed=i)
            mazes[i] = maze
        print(f"Generated {self.total_samples} mazes of size {self.width}x{self.height}.")
        self.mazes = filter_repeated_mazes(mazes)
        print(f"Filtered to {len(self.mazes)} unique mazes after removing duplicates.")


if __name__ == "__main__":
    os.mkdir("mazes_pictures-7x7") if not os.path.exists("mazes_pictures-7x7") else None
    os.chdir("mazes_pictures-7x7")
    env = Maze(width=7, height=7, total_samples=10)
    for i, maze in enumerate(env.mazes):
        plot_maze(maze, show=False, save_path=f"maze_{i}.png")
