from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
import random
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm
from torch import nn

try:
    from .QNetwork import QNetwork
    from .dqn_agent import DQNAgent, _reset_environment
except ImportError:  # Support running this module directly.
    from QNetwork import QNetwork
    from dqn_agent import DQNAgent, _reset_environment


class ActorDQNAgent(DQNAgent):
    """A DQN agent that uses an actor network to select actions."""

    def __init__(
        self,
        input_dim: int | Sequence[int],
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        exploration_rate: float = 0.1,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        learning_starts: int | None = None,
        target_update_frequency: int = 2000,
        gradient_clip: float | None = 10.0,
        device: str | torch.device | None = None,
        seed: int | None = None,

        total_timesteps: int, 
        environment: Any,
        eval_environment: Any,
        actor_eval_environment: Any,
        eval_num_episodes: int = 10, 
        max_episode_steps: int|None = 100_000,
        max_episode_steps_eval: int|None = None,
        evaluation_frequency: int | None = None,
    ):
        '''
        I use two different env for different models, to ensure they are evaluated in the same episode sequence.

        If we use only one eval environment, for both models, env will give different episodes to two models, causing
        multi-variable situation.

        Use two independent envs can ensure two model are evaluated in the same episode sequence.

        Args:
            eval_environment: evaluation env for normal dqn
            actor_eval_environment: evaluation env for actor dqn
        '''
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_rate=exploration_rate,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            learning_starts=learning_starts,
            target_update_frequency=target_update_frequency,
            gradient_clip=gradient_clip,
            device=device,
            seed=seed,
            total_timesteps=total_timesteps,
            environment=environment,
            eval_num_episodes=eval_num_episodes,
            max_episode_steps=max_episode_steps,
            max_episode_steps_eval=max_episode_steps_eval,
            evaluation_frequency=evaluation_frequency,
            eval_environment=eval_environment,
        )

        self.actor_q_network = QNetwork(input_dim, output_dim, hidden_dims).to(self.device)
        self.optimizer_for_actor = torch.optim.Adam(self.actor_q_network.parameters(), lr=learning_rate)
        self.loss_fn_for_actor = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification


        self.actor_eval_environment = actor_eval_environment

    def train(self) -> None:
            """Train by repeatedly rolling out, replaying, and resetting episodes."""
            if self.total_timesteps < 0:
                raise ValueError("total_timesteps cannot be negative")
    
            self.q_network.train()
            self.actor_q_network.train()

            state, _ = _reset_environment(self.environment, evaluation=False)
            episode_steps = 0

            bar = tqdm(range(self.total_timesteps), desc="Running experiments")
            for step in bar:
                next_state, episode_ended, _, _ = self.rollout(self.environment, state)
                update_result = self.update()
                if update_result is None:
                    loss = loss_actor = None
                else:
                    loss, loss_actor = update_result
    
                if self.tensorboard_writer is not None:
                    if loss is not None:
                        self.tensorboard_writer.add_scalar(
                            "training/loss", loss, step
                        )
                    if loss_actor is not None:
                        self.tensorboard_writer.add_scalar(
                            "training/loss_actor", loss_actor, step
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
                    # metrics recording for vanilla dqn
                    eval_avg_return, eval_avg_episode_length, eval_success_rate = self.evaluation(evaluation=True)
                    train_avg_return, train_avg_episode_length, train_success_rate = self.evaluation(evaluation=False)
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar(
                            "eval/return", eval_avg_return, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "eval/steps", eval_avg_episode_length, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "eval/success_rate", eval_success_rate, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "training/return", train_avg_return, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "training/steps", train_avg_episode_length, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "training/success_rate", train_success_rate, step
                        )

                    # metrics recording for actor dqn
                    eval_avg_return, eval_avg_episode_length, eval_success_rate = self.actor_evaluation(evaluation=True)
                    train_avg_return, train_avg_episode_length, train_success_rate = self.actor_evaluation(evaluation=False)
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar(
                            "eval/return_actor", eval_avg_return, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "eval/steps_actor", eval_avg_episode_length, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "eval/success_rate_actor", eval_success_rate, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "training/return_actor", train_avg_return, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "training/steps_actor", train_avg_episode_length, step
                        )
                        self.tensorboard_writer.add_scalar(
                            "training/success_rate_actor", train_success_rate, step
                        )

    def update(self) -> tuple[float, float] | None:
        """Learn from one replay batch and return its loss and predicted label, if replay is ready."""
        if len(self.replay_buffer) < self.learning_starts:
            return None

        # Update for vanilla DQN.
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._tensorize_batch(batch)
        predicted_q_values = self._chosen_action_values(states, actions)
        target_q_values = self._bellman_targets(rewards, next_states, dones)
        loss = self._optimize(predicted_q_values, target_q_values)


        # Update for actor DQN.
        with torch.no_grad():
            labels = self.q_network(states).argmax(dim=1)
        predicted_actor_logits = self.actor_q_network(states)
        loss_actor = self._optimize_for_actor(predicted_actor_logits, labels)

        self.update_steps += 1
        if self.update_steps % self.target_update_frequency == 0:
            self.update_target_network()
        return loss,loss_actor
    
    def _optimize_for_actor(
        self, predicted_q_values: torch.Tensor, label: torch.Tensor
    ) -> float:
        """Perform one stable gradient update of the actor network."""
        loss = self.loss_fn_for_actor(predicted_q_values, label)
        self.optimizer_for_actor.zero_grad()
        loss.backward()
        if self.gradient_clip is not None:
            nn.utils.clip_grad_value_(
                self.actor_q_network.parameters(), self.gradient_clip
            )
        self.optimizer_for_actor.step()
        return float(loss.detach().item())

    def _actor_greedy_action(self, state: Any) -> int:
        """Sample the action with the probability distribution from the actor dqn network."""
        state_tensor = self._state_tensor(state)
        with torch.no_grad():
            q_values = self.actor_q_network(state_tensor)
            action_probs = torch.softmax(q_values, dim=1)
            return int(torch.multinomial(action_probs, num_samples=1).item())

    def actor_action_selection(
        self,
        state: Any,
        *,
        greedy: bool = False,
    ) -> int:
        """Select with the fixed exploration rate, or greedily for evaluation."""
        exploration_rate = 0.0 if greedy else self.exploration_rate
        if self._rng.random() < exploration_rate:
            # Randomly select an action with uniform probability.
            return self._rng.randrange(self.output_dim)
        return self._actor_greedy_action(state)
    
    def actor_evaluation(
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
        was_training = self.actor_q_network.training
        self.actor_q_network.eval()

        try:
            for _ in range(self.eval_num_episodes):
                state, _ = _reset_environment(self.actor_eval_environment, evaluation=evaluation)
                episode_return = 0.0
                episode_steps = 0

                if step_limit == -1:
                    while True:
                        action = self.actor_action_selection(state, greedy=True)
                        state, reward, terminated, truncated, _ = self.actor_eval_environment.step(action)
                        episode_return += float(reward)
                        episode_steps += 1
                        if terminated:
                            successes += 1
                        if terminated or truncated:
                            break
                else:
                    for episode_steps in range(1, step_limit + 1):
                        action = self.actor_action_selection(state, greedy=True)
                        state, reward, terminated, truncated, _ = self.actor_eval_environment.step(action)
                        episode_return += float(reward)
                        if terminated:
                            successes += 1
                        if terminated or truncated:
                            break

                returns.append(episode_return)
                step_counts.append(episode_steps)
        finally:
            self.actor_q_network.train(was_training)

        return (
            float(np.mean(returns)),
            float(np.mean(step_counts)),
            successes / self.eval_num_episodes,
        )        

