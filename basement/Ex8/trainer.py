import gymnasium as gym
from ActorDQN.basement.Ex8.agent import DQNAgent
from ActorDQN.basement.Ex8.replay_buffer import ReplayMemory
from tqdm import tqdm
import torch


class Trainer:
    # This trainer is still very similar to the one you know, the main difference is that we are now operating
    # with a replay buffer instead of using the transitions directly
    def __init__(
        self,
        train_environment: gym.Env,
        evaluation_environment: gym.Env,
        agent: DQNAgent,
        memory: ReplayMemory,
        batch_size: int,
    ) -> None:
        self.train_environment = train_environment
        self.evaluation_environment = evaluation_environment
        self.agent = agent
        self.memory = memory
        self.batch_size = batch_size

    def train(self, num_episodes: int):
        state, info = self.train_environment.reset()
        # Gymnasium environments usually use numpy arrays. To use pytorch we need torch.tensors therefore we
        # need make sure to convert everything accordingly.
        # Also we want to add an extra dimension to the data as we are now working with batches, that is what .unsqueeze() does
        # Let's say we have an state with four dimensions than it would have a shape of (4,) we want to change it to (1,4)
        # such that we can stack data along the first dimension (batch_size, 4)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = (False, False)
        for _ in tqdm(range(num_episodes)):
            while not (terminated or truncated):
                action = self.agent.getAction(state)
                next_state, reward, terminated, truncated, info = (
                    self.train_environment.step(action.item())
                )
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(
                    0
                )  # add batch dimension as explained above
                # reward has a shape of () therefore we need to add two dimensions to it, you can unsqueeze it twice, or use reshape like this
                reward = torch.tensor(reward, dtype=torch.float32).reshape(1, 1)
                terminated = torch.tensor(terminated, dtype=torch.bool).unsqueeze(0)
                # add the transition to the replay buffer
                self.memory.push(state, action, reward, next_state, terminated)
                state = next_state

                # We only update when we have at least a full batch worth of data
                if len(self.memory) >= self.batch_size:
                    (
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        terminated_batch,
                    ) = self.memory.sample(self.batch_size)
                    self.agent.update(
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        terminated_batch,
                    )

            # reset the environment when the episode is done
            state, info = self.train_environment.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminated, truncated = (False, False)

    def evaluate(self, num_episodes: int):
        state, info = self.evaluation_environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = (False, False)
        for _ in tqdm(range(num_episodes)):
            while not (terminated or truncated):
                action = self.agent.getPolicy(state)
                state, reward, terminated, truncated, info = (
                    self.evaluation_environment.step(action.item())
                )
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state, info = self.evaluation_environment.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminated, truncated = (False, False)
