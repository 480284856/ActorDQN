"""A modular Deep Q-Network agent for discrete-action Gymnasium environments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

try:
    from .QNetwork import QNetwork
except ImportError:  # Support running this module directly.
    from QNetwork import QNetwork

from torch.utils.tensorboard import SummaryWriter

@dataclass(frozen=True)
class Transition:
    """One immutable interaction stored in replay memory."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size replay memory with uniformly random sampling."""

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        self._transitions: deque[Transition] = deque(maxlen=capacity)
        self._rng = rng or random.Random()

    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Store copies so later environment mutation cannot alter experience."""
        self._transitions.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a uniform sample without replacement."""
        if batch_size > len(self):
            raise ValueError("cannot sample more transitions than the buffer contains")
        return self._rng.sample(list(self._transitions), batch_size)

    def __len__(self) -> int:
        return len(self._transitions)


class DQNAgent:
    """Coordinate exploration, replay learning, target updates, and evaluation.

    Public methods correspond to the logical modules in the supplied diagram:
    :meth:`action_selection`, :meth:`rollout`, :meth:`update`,
    :meth:`evaluation`, and :meth:`update_target_network`. :meth:`train` is the
    top-level orchestration layer that connects those modules.
    """

    def __init__(
        self,
        input_dim: int | Sequence[int],
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 1000.0,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        learning_starts: int | None = None,
        target_update_frequency: int = 1,
        tau: float = 0.005,
        gradient_clip: float | None = 10.0,
        device: str | torch.device | None = None,
        seed: int | None = None,

        total_timesteps: int, 
        environment: Any,
        eval_environment: Any,
        eval_num_episodes: int = 10, 
        max_episode_steps: int|None = 100_000,
        max_episode_steps_eval: int|None = None,
        evaluation_frequency: int | None = None,

        tensorboard_log_dir:str = "logs/dqn_maze",
    ) -> None:
        '''
        Args:
            input_dim: The shape of one **flattened** observation, or the number of features.
            output_dim: The number of discrete actions.
            hidden_dims: The number of units in each hidden layer.
            epsilon_start: Exploration probability at the start of training.
            epsilon_end: Minimum exploration probability approached over time.
            epsilon_decay: Exponential-decay time constant measured in action
                selections.
            tau: The fraction of online-network weights mixed into the target
                network after each scheduled target update.
            
            total_timesteps: The total number of timesteps to call env.step().
            eval_environment: Another new env but used for evaluation.
            eval_num_episodes: The number of episodes to run for evaluation.
            max_episode_steps: The maximum number of steps per episode.
            evaluation_frequency: How often to evaluate the agent during training, in steps.
        '''
        _validate_hyperparameters(
            output_dim=output_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            tau=tau,
            batch_size=batch_size,
            target_update_frequency=target_update_frequency,
            max_episode_steps=max_episode_steps,
            evaluation_frequency=evaluation_frequency,
            total_timesteps=total_timesteps,
            eval_num_episodes=eval_num_episodes,
        )

        output_dim = int(output_dim)
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if learning_starts is not None and learning_starts > replay_capacity:
            raise ValueError("learning_starts cannot exceed replay_capacity")
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.q_network = QNetwork(input_dim, output_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(input_dim, output_dim, hidden_dims).to(self.device)
        # The target must start as an exact copy. Later updates use Polyak
        # averaging controlled by ``tau``.
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_capacity, random.Random(seed))

        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.batch_size = batch_size
        self.learning_starts = batch_size if learning_starts is None else learning_starts
        if self.learning_starts < batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        
        self.gradient_clip = gradient_clip
        self._rng = random.Random(seed)
        self.total_steps = 0
        self.update_steps = 0

        self.tensorboard_writer =  SummaryWriter(log_dir=tensorboard_log_dir)

        self.total_timesteps = total_timesteps
        self.environment = environment
        self.eval_environment = eval_environment
        self.eval_num_episodes = eval_num_episodes
        self.max_episode_steps = max_episode_steps
        self.evaluation_frequency = evaluation_frequency
        self.max_episode_steps_eval = max_episode_steps_eval


    def train(self) -> None:
        """Train by repeatedly rolling out, replaying, and resetting episodes."""
        if self.total_timesteps < 0:
            raise ValueError("total_timesteps cannot be negative")

        self.q_network.train()
        state, _ = _reset_environment(self.environment, evaluation=False)
        episode_steps = 0

        for step in range(self.total_timesteps):
            next_state, episode_ended, _, _ = self.rollout(self.environment, state)
            loss = self.update()

            if self.tensorboard_writer is not None:
                if loss is not None:
                    self.tensorboard_writer.add_scalar(
                        "training/loss", loss, step
                    )
            episode_steps += 1

            # Reset the environment if the episode ended or 
            # it takes too many steps.
            reached_step_limit = (
                self.max_episode_steps is not None
                and episode_steps >= self.max_episode_steps
            )
            if episode_ended or reached_step_limit:
                state, _ = _reset_environment(self.environment, evaluation=False)
                episode_steps = 0
            else:
                state = next_state

            if self.evaluation_frequency is not None and self.total_steps % self.evaluation_frequency == 0:
                eval_avg_return, eval_avg_episode_length, eval_success_rate = self.evaluation(evaluation=True)
                train_avg_return, train_avg_episode_length, train_success_rate = self.evaluation(evaluation=False)
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar(
                        "eval/return", eval_avg_return, step
                    )
                    self.evaluation_writer.add_scalar(
                        "eval/steps", eval_avg_episode_length, step
                    )
                    self.evaluation_writer.add_scalar(
                        "eval/success_rate", eval_success_rate, step
                    )
                    self.tensorboard_writer.add_scalar(
                        "training/return", train_avg_return, step
                    )
                    self.evaluation_writer.add_scalar(
                        "training/steps", train_avg_episode_length, step
                    )
                    self.evaluation_writer.add_scalar(
                        "training/success_rate", train_success_rate, step
                    )

    def action_selection(
        self,
        state: Any,
        *,
        greedy: bool = False,
    ) -> int:
        """Select epsilon-greedily with exponential decay, or greedily for evaluation."""
        if greedy:
            return self._greedy_action(state)

        epsilon_threshold = self._epsilon_threshold()
        self.steps_done += 1
        if self._rng.random() < epsilon_threshold:
            # Randomly select an action with uniform probability.
            return self._rng.randrange(self.output_dim)
        return self._greedy_action(state)

    def _epsilon_threshold(self) -> float:
        """Return the exploration probability for the current training step."""
        return self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)

    def rollout(
        self,
        environment: Any,
        state: Any | None = None,
    ) -> tuple[Any, bool, bool, float]:
        """Take one exploratory environment step and save it to replay memory.

        Returns ``(next_state, episode_ended, succeeded, reward)``. If no
        state is supplied, the environment is reset first.
        """
        if state is None:
            state, _ = _reset_environment(environment, evaluation=False)

        action = self.action_selection(state)
        next_state, reward, terminated, truncated, _ = environment.step(action)
        episode_ended = bool(terminated or truncated)
        self.replay_buffer.add(
            state, action, reward, next_state, done=episode_ended
        )
        self.total_steps += 1
        return next_state, episode_ended, bool(terminated), float(reward)

    def update(self) -> float | None:
        """Learn from one replay batch and return its loss, if replay is ready."""
        if len(self.replay_buffer) < self.learning_starts:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._tensorize_batch(batch)
        predicted_q_values = self._chosen_action_values(states, actions)
        target_q_values = self._bellman_targets(rewards, next_states, dones)
        loss = self._optimize(predicted_q_values, target_q_values)

        self.update_steps += 1
        if self.update_steps % self.target_update_frequency == 0:
            self.update_target_network()
        return loss

    def evaluation(
        self,
        evaluation: bool = True,
    ) -> tuple[float, float, float]:
        """Evaluate greedily and return average return, steps, and success rate."""
        if self.eval_num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        step_limit = (
            -1
            if self.max_episode_steps_eval is None
            else self.max_episode_steps_eval
        )

        returns: list[float] = []
        step_counts: list[int] = []
        successes = 0
        was_training = self.q_network.training
        self.q_network.eval()

        try:
            for _ in range(self.eval_num_episodes):
                state, _ = _reset_environment(self.eval_environment, evaluation=evaluation)
                episode_return = 0.0
                episode_steps = 0

                if step_limit == -1:
                    while True:
                        action = self.action_selection(state, greedy=True)
                        state, reward, terminated, truncated, _ = self.eval_environment.step(action)
                        episode_return += float(reward)
                        episode_steps += 1
                        if terminated:
                            successes += 1
                        if terminated or truncated:
                            break
                else:
                    for episode_steps in range(1, step_limit + 1):
                        action = self.action_selection(state, greedy=True)
                        state, reward, terminated, truncated, _ = self.eval_environment.step(action)
                        episode_return += float(reward)
                        if terminated:
                            successes += 1
                        if terminated or truncated:
                            break

                returns.append(episode_return)
                step_counts.append(episode_steps)
        finally:
            self.q_network.train(was_training)

        return (
            float(np.mean(returns)),
            float(np.mean(step_counts)),
            successes / self.eval_num_episodes,
        )

    def update_target_network(self) -> None:
        """Move the target network a ``tau`` fraction toward the online network."""
        target_state_dict = self.target_network.state_dict()
        online_state_dict = self.q_network.state_dict()
        with torch.no_grad():
            for key in online_state_dict:
                target_state_dict[key] = (
                    online_state_dict[key] * self.tau
                    + target_state_dict[key] * (1.0 - self.tau)
                )
        self.target_network.load_state_dict(target_state_dict)

    def _greedy_action(self, state: Any) -> int:
        """Select the action with the largest online-network Q-value."""
        state_tensor = self._state_tensor(state)
        with torch.no_grad():
            return int(self.q_network(state_tensor).argmax(dim=1).item())

    def _state_tensor(self, state: Any) -> torch.Tensor:
        """Convert one observation to a float tensor with a batch dimension."""
        return torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device
        ).unsqueeze(0)

    def _tensorize_batch(
        self, batch: list[Transition]
    ) -> tuple[torch.Tensor, ...]:
        """Convert sampled transitions into device-resident training tensors."""
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.bool, device=self.device
        ).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def _chosen_action_values(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Gather Q(s, a) for the actions that were actually taken."""
        return self.q_network(states).gather(1, actions)

    def _bellman_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fixed-target one-step Bellman values without gradients."""
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True).values
            return rewards + self.gamma * next_q_values * (~dones).float()

    def _optimize(
        self, predicted_q_values: torch.Tensor, target_q_values: torch.Tensor
    ) -> float:
        """Perform one stable gradient update of the online network."""
        loss = self.loss_fn(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip is not None:
            nn.utils.clip_grad_value_(
                self.q_network.parameters(), self.gradient_clip
            )
        self.optimizer.step()
        return float(loss.detach().item())


def _reset_environment(environment: Any, *, evaluation: bool) -> tuple[Any, dict]:
    """Reset Gymnasium environments, selecting an evaluation split when supported."""
    try:
        return environment.reset(options={"is_evaluation": evaluation})
    except TypeError:
        return environment.reset()


def _validate_hyperparameters(**values: Any) -> None:
    """Validate configuration at the agent boundary rather than during training."""
    integer_names = {
        "output_dim",
        "batch_size",
        "target_update_frequency",
        "max_episode_steps",
        "evaluation_frequency",
        "total_timesteps",
        "eval_num_episodes",

    }
    for name in integer_names:
        value = values[name]
        if name in {"max_episode_steps", "evaluation_frequency"} and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if values["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= values["gamma"] <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if not 0.0 <= values["epsilon_end"] <= values["epsilon_start"] <= 1.0:
        raise ValueError(
            "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1"
        )
    if values["epsilon_decay"] <= 0:
        raise ValueError("epsilon_decay must be positive")
    if not 0.0 <= values["tau"] <= 1.0:
        raise ValueError("tau must be between 0 and 1")
