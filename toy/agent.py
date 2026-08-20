import copy
import math
import random
import torch
import gymnasium as gym
from torch import nn


class DQNAgent:
    def __init__(
        self,
        dqn: torch.nn.Module,
        optimizer: torch.nn.Module,
        epsilon:float,
        discount: float,
        action_space: gym.spaces.Space,

    ) -> None:

        self.dqn = dqn
        self.optimizer = optimizer

        # Same thing as the Huber loss with delta equal to 1
        self.criterion = nn.MSELoss()

        # We use this to calculate the current epsilon
        self.steps_done = 0

        self.epsilon = epsilon
        self.discount = discount
        self.action_space = action_space

    def getPolicy(self, state: torch.tensor) -> int:
        with torch.no_grad():
            return self.dqn(state).argmax().unsqueeze(0).unsqueeze(0)

    def getAction(self, state: torch.tensor) -> int:
        sample = random.random()

        self.steps_done += 1
        if sample > self.epsilon:
            return self.getPolicy(state=state)
        else:
            return torch.tensor([[self.action_space.sample()]], dtype=torch.long)

    def update(
        self,
        state: torch.tensor,
        action: torch.tensor,
        next_state: torch.tensor,
        reward: torch.tensor,
        terminated: bool,
    ):

        # =prediction[action]
        prediction = self.dqn(state).gather(dim=1, index=action)
        with torch.no_grad():
            if not terminated:
                TD_target = reward + self.discount * self.dqn(next_state).max(axis=1, keepdim=True).values
            else:
                TD_target = reward

        # TODO calculate the loss using the given self.criterion, your prediction and the target
        loss = self.criterion(TD_target, prediction)

        # TODO clear all previous gradients from the optimizer
        self.optimizer.zero_grad()

        # TODO Let pytorch calculate all gradients of the parameters with respect to the loss
        loss.backward()

        # TODO for stability reasons clip all gradients to an absolute value of 100
        # torch.nn.utils.clip_grad_value_(self.dqn.parameters(), clip_value=100)

        # TODO Finally let the optimizer perform an optimization step based on the gradients
        self.optimizer.step()
        # self._update_target_net()

