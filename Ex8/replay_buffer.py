from collections import deque, namedtuple
import random

import torch


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminated")
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminated_batch = torch.cat(batch.terminated)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminated_batch,
        )

    def __len__(self):
        return len(self.memory)
