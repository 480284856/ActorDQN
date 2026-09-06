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
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        action_space: gym.spaces.Space,
        discount: float,
        tau: float,
    ) -> None:

        self.dqn = dqn
        # Our target_net starts of as a copy of our online net, but as the online net changes
        # the target_net is only updated in a delayed manner and therefore they will differ
        self.target_net = copy.deepcopy(self.dqn)

        self.optimizer = optimizer

        # We use an decaying epsilon such that there is more exploration in the beginning
        # and more exploitation as time goes an
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.discount = discount
        # tau is the percentage by with we update our target_net towards our online_net in each update step
        self.tau = tau
        self.action_space = action_space

        # Same thing as the Huber loss with delta equal to 1
        self.criterion = nn.SmoothL1Loss()

        # We use this to calculate the current epsilon
        self.steps_done = 0

    def getPolicy(self, state: torch.tensor) -> int:
        with torch.no_grad():
            return self.dqn(state).argmax().unsqueeze(0).unsqueeze(0)

    def getAction(self, state: torch.tensor) -> int:
        sample = random.random()

        # calculate decaying epsilon
        eps_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)

        self.steps_done += 1
        if sample > eps_threshold:
            return self.getPolicy(state=state)
        else:
            return torch.tensor([[self.action_space.sample()]], dtype=torch.long)

    def update(
        self,
        state_batch: torch.tensor,
        action_batch: torch.tensor,
        reward_batch: torch.tensor,
        next_state_batch: torch.tensor,
        terminated_batch: torch.tensor,
    ):

        # a little help
        not_terminated = ~terminated_batch

        # TODO calculate the current estimates for the given state action pairs the .gather() function from pytorch might be very useful.
        prediction = torch.gather(
            input=self.dqn(state_batch),
            dim=1,
            index=action_batch
        )
        # prediction: (bs, 1)

        next_state_values = torch.zeros_like(prediction)
        # This flag indicates to pytorch to not track gradients within the scope, this saves computation
        # and is also necessary as we want to implement the semi-gradient variant
        with torch.no_grad():
            # TODO calculate the estimates for the next state (action) pairs, deep neural networks are computationally expensive (especially when running on the cpu)
            # try not to pass things to the network you don't really need, the not_terminated booleans should help you with this.
            # Also you might need to add a Batch dimension again back to your result like (batch_size, 1)
            inputs = next_state_batch[not_terminated]
            # prediction_next_state = self.dqn(inputs).max(axis=1, keepdim=True).values
            # user target net instead
            prediction_next_state = self.target_net(inputs).max(axis=1, keepdim=True).values

            next_state_values[not_terminated] = prediction_next_state

        # TODO using the next_state values calculate the target value
        TD_target = reward_batch + self.discount*next_state_values
        # 128 x 1

        # TODO calculate the loss using the given self.criterion, your prediction and the target
        loss = self.criterion(TD_target, prediction)

        # TODO clear all previous gradients from the optimizer
        self.optimizer.zero_grad()

        # TODO Let pytorch calculate all gradients of the parameters with respect to the loss
        loss.backward()

        # TODO for stability reasons clip all gradients to an absolute value of 100
        torch.nn.utils.clip_grad_value_(self.dqn.parameters(), clip_value=100)

        # TODO Finally let the optimizer perform an optimization step based on the gradients
        self.optimizer.step()
        self._update_target_net()

    def _update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.dqn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
