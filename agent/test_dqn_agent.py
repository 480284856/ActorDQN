"""Focused tests for the Q-network and DQN module boundaries."""

import math
import unittest

import numpy as np
import torch

try:
    from .QNetwork import QNetwork
    from .dqn_agent import DQNAgent
except ImportError:  # Support discovery with ``-s agent``.
    from QNetwork import QNetwork
    from dqn_agent import DQNAgent


class OneStepEnvironment:
    """A tiny deterministic Gymnasium-style environment."""

    def __init__(self) -> None:
        self.reset_options = []

    def reset(self, *, options=None):
        self.reset_options.append(options)
        return np.array([1.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        succeeded = action == 1
        return (
            np.array([0.0, 1.0], dtype=np.float32),
            1.0 if succeeded else -1.0,
            succeeded,
            not succeeded,
            {},
        )


class QNetworkTests(unittest.TestCase):
    def test_flattens_single_and_batched_maze_observations(self):
        network = QNetwork((4, 8), output_dim=4)

        self.assertEqual(network(torch.zeros(4, 8)).shape, (4,))
        self.assertEqual(network(torch.zeros(3, 4, 8)).shape, (3, 4))

    def test_output_layer_has_no_activation(self):
        network = QNetwork(2, output_dim=2, hidden_dims=())
        with torch.no_grad():
            network.model[0].weight.zero_()
            network.model[0].bias.copy_(torch.tensor([-2.0, 3.0]))

        torch.testing.assert_close(
            network(torch.zeros(2)), torch.tensor([-2.0, 3.0])
        )


class DQNAgentTests(unittest.TestCase):
    def make_agent(self, **overrides):
        environment = OneStepEnvironment()
        arguments = {
            "input_dim": 2,
            "output_dim": 2,
            "hidden_dims": (),
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay": 10.0,
            "batch_size": 1,
            "replay_capacity": 4,
            "target_update_frequency": 1,
            "tau": 0.25,
            "device": "cpu",
            "seed": 4,
            "total_timesteps": 3,
            "environment": environment,
            "eval_environment": environment,
            "eval_num_episodes": 3,
            "max_episode_steps": 1,
            "max_episode_steps_eval": 1,
        }
        arguments.update(overrides)
        return DQNAgent(**arguments)

    def test_target_starts_as_an_exact_copy(self):
        agent = self.make_agent()

        for online_parameter, target_parameter in zip(
            agent.q_network.parameters(), agent.target_network.parameters()
        ):
            torch.testing.assert_close(online_parameter, target_parameter)
            self.assertFalse(target_parameter.requires_grad)

    def test_epsilon_decays_exponentially_with_action_selections(self):
        agent = self.make_agent(
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=10.0,
        )

        self.assertEqual(agent._epsilon_threshold(), 1.0)
        agent.steps_done = 10
        self.assertAlmostEqual(
            agent._epsilon_threshold(),
            0.1 + (1.0 - 0.1) * math.exp(-1.0),
        )

    def test_only_training_action_selection_advances_epsilon(self):
        agent = self.make_agent()
        state = np.zeros(2, dtype=np.float32)

        agent.action_selection(state)
        self.assertEqual(agent.steps_done, 1)

        agent.action_selection(state, greedy=True)
        self.assertEqual(agent.steps_done, 1)

    def test_target_update_uses_polyak_averaging(self):
        agent = self.make_agent(tau=0.25)
        with torch.no_grad():
            for parameter in agent.q_network.parameters():
                parameter.fill_(10.0)
            for parameter in agent.target_network.parameters():
                parameter.fill_(2.0)

        agent.update_target_network()

        for target_parameter in agent.target_network.parameters():
            torch.testing.assert_close(
                target_parameter, torch.full_like(target_parameter, 4.0)
            )

    def test_terminal_bellman_target_does_not_bootstrap(self):
        agent = self.make_agent(gamma=0.5)
        with torch.no_grad():
            agent.target_network.model[0].weight.zero_()
            agent.target_network.model[0].bias.fill_(10.0)

        targets = agent._bellman_targets(
            rewards=torch.tensor([[2.0], [2.0]]),
            next_states=torch.zeros(2, 2),
            dones=torch.tensor([[True], [False]]),
        )

        torch.testing.assert_close(targets, torch.tensor([[2.0], [7.0]]))

    def test_train_collects_replay_and_updates_the_network(self):
        agent = self.make_agent()
        environment = OneStepEnvironment()

        agent.environment = environment
        agent.train()

        self.assertEqual(agent.total_steps, 3)
        self.assertEqual(len(agent.replay_buffer), 3)
        self.assertEqual(agent.update_steps, 3)

    def test_evaluation_is_greedy_and_reports_metrics(self):
        agent = self.make_agent(epsilon_start=1.0, epsilon_end=1.0)
        with torch.no_grad():
            agent.q_network.model[0].weight.zero_()
            agent.q_network.model[0].bias.copy_(torch.tensor([0.0, 1.0]))
        environment = OneStepEnvironment()

        agent.eval_environment = environment
        metrics = agent.evaluation()

        self.assertEqual(metrics, (1.0, 1.0, 1.0))
        self.assertTrue(
            all(options == {"is_evaluation": True} for options in environment.reset_options)
        )


if __name__ == "__main__":
    unittest.main()
