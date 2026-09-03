import os
from typing import Tuple

import numpy as np
import gymnasium as gym

try:
    from .util import (
        generate_all_single_solution_mazes,
        generate_maze_variations,
    )
except ImportError:  # Support running this file directly.
    from util import generate_all_single_solution_mazes, generate_maze_variations


class Maze(gym.Env):
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        self.observation_space = ...
        self.action_space = ...

        # self.generate_maze()

    def generate_maze(self,):
        '''
        Here is the description of the maze generation process:
            https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe?from=from_copylink

        Cell meanings:
            0 = empty
            1 = wall
            2 = player
            3 = goal
        '''
        original_set = self.pathfinding()
        original_training_set, original_evaluation_set = self.train_test_split(original_set, test_size=0.2, random_state=42)

        self.training_mazes = self.maze_variation(original_training_set)
        self.evaluation_mazes = self.maze_variation(original_evaluation_set)

    def pathfinding(self) -> np.ndarray:
        """
        Generate every maze whose open cells form one unique player-goal path.

        For detailed description, please refer to the following link:
            https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe#share-SDypdpOnto1ZJ1xoNO8jeEj5pIc

        Returns:
            3D numpy array of shape (num_mazes, height, width) containing the generated mazes.
        """
        return generate_all_single_solution_mazes(self.width, self.height)

    def train_test_split(self, mazes: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the generated mazes into training and evaluation sets.

        Args:
            mazes: 3D numpy array of shape (num_mazes, height, width) containing the generated mazes.
            test_size: Proportion of the dataset to include in the evaluation set.
        """
        rng = np.random.default_rng(random_state)
        num_mazes = mazes.shape[0]
        indices = rng.permutation(num_mazes)
        split_idx = int(num_mazes * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        return mazes[train_indices], mazes[test_indices]

    def maze_variation(
        self,
        mazes: np.ndarray,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Generate variations of the given mazes to increase diversity.

        For detailed description, please refer to the following link:
            https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe#share-K32KdcnikoxKdoxjQ5wjFpecpJe

        Args:
            mazes: 3D numpy array of shape (num_mazes, height, width) containing the generated mazes.
            random_state: Optional seed for reproducible variations.

        Returns:
            3D numpy array of shape (num_variations, height, width) containing the varied mazes.
        """
        return generate_maze_variations(
            mazes,
            expected_shape=(self.height, self.width),
            random_state=random_state,
        )

    def reset(self, seed=None, is_evaluation=False):
        """
        Sample a maze from the training set or evaluation set.
        """
        if is_evaluation:
            # Sample from the evaluation set
            pass
        else:
            # Sample from the training set
            pass

    def step(self, action):
        """
        Execute the action in the environment and return the next state, reward, done, and info.
        """
        pass
    
if __name__ == "__main__":
    from visualization import plot_maze

    #os.mkdir("mazes_pictures-4x4") if not os.path.exists("mazes_pictures-4x4") else None
    #os.chdir("mazes_pictures-4x4")
    #env = Maze(width=6, height=6) # Total number of mazes generated: 669072
    #env = Maze(width=7, height=7) # Total number of mazes generated: 23093748
    #print(f"Total number of mazes generated: {len(env.pathfinding())}")
    #for i, maze in enumerate(env.evaluation_mazes):
    #    plot_maze(maze, show=False, save_path=f"maze_{i}.png")
