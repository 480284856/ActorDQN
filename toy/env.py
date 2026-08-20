from typing import Optional, List
import numpy as np
import gymnasium as gym

# def designGrid():
#     grid = [
#                 ["#", "#", " ",  " ", "#", "#"],
#                 ["S", " ", "#",  " ", " ",   1],
#                 ["#", " ",  " ", " ", "#", "#"],
#             ]
#     return grid
def designGrid():
    grid = [
                ["#", "#", "#",  "#",],
                ["S", " ", "#",  "#",],
                ["#", " ",  " ", 1,],
            ]
    return grid

class GridWorldEnv(gym.Env):

    def __init__(self, grid_draft:List[List[str|int]]=None):
        # The size of the square grid (5x5 by default)
        if grid_draft is None:
            grid_draft = designGrid()
        self.grid = np.array(grid_draft)
        self.row_size, self.col_size = self.grid.shape
        

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0,  1, shape=(15,), dtype=np.float32),   # [x, y] coordinates
                "target": gym.spaces.Box(0, max(self.grid.shape) - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([-1, 0]),   # Move up (column - 1)
            1: np.array([0, 1]),   # Move right (column + 1)
            2: np.array([1, 0]),   # Move down (row + 1)
            3: np.array([0, -1]),  # Move left (column - 1)
        }

        self.obstacle_locations = {
            tuple(position)
            for position in np.argwhere(self.grid == "#")
        }

    def _encode_cell(self, position):
        """
        o_t = [ f(current_location), 
                f(up+curr_location), 
                f(right+curr_location)]
                f(down+curr_location), 
                f(left+curr_location), 
        
        f(x) =  [1,0,0] if x is achievable,
                [0,1,0] if x is obstacle,
                [0,0,1] if x is target
        """
        row, col = position

        # 地图之外也当作墙
        if not (0 <= row < self.row_size and 0 <= col < self.col_size):
            return np.array([0, 1, 0])

        cell = self.grid[row, col]

        if cell == "#":
            return np.array([0, 1, 0])

        if cell == "1":
            return np.array([0, 0, 1])

        # "S" 和普通空格都视为可通行
        return np.array([1, 0, 0])  

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """

        self._OBSERVATION_OFFSETS = [
            np.array([0, 0]),    # current
            np.array([-1, 0]),   # up
            np.array([0, 1]),    # right
            np.array([1, 0]),    # down
            np.array([0, -1]),   # left
        ]
        features = [
            self._encode_cell(self._agent_location + offset)
            for offset in self._OBSERVATION_OFFSETS
        ]

        return {"agent": np.concatenate(features).astype(np.float32), "target": self._target_location}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "agent_location": self._agent_location,
            "target_location": self._target_location,
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        row,col = np.where(self.grid == "S")
        self._agent_location = np.array([row[0], col[0]], dtype=int)

        row,col = np.where(self.grid == '1')
        self._target_location = np.array([row[0], col[0]], dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        next_position = np.clip(self._agent_location + direction, [0, 0], [self.row_size - 1, self.col_size - 1])
        if tuple(next_position) not in self.obstacle_locations:
            self._agent_location = next_position

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else -0.01  # small penalty for each step to encourage faster solutions

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

