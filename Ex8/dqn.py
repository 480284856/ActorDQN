from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, observation_dimensions, number_of_actions):
        super(DQN, self).__init__()
        # TODO implement a neural network using pytorch with:
        # observation_dimensions input neurons
        # two hidden layers of 128 neurons
        # number_of_actions output neurons
        # And Rectified Linear Units (ReLU) activations in between (only in between, not after the last layer)
        # There are different ways of doing this
        self.model = nn.Sequential(
            nn.Linear(observation_dimensions, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_actions)
        )

    def forward(self, x):
        # TODO write the according forward function
        return self.model(x)
