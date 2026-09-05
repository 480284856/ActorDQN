"""Train and evaluate the Actor-DQN agent on the maze environment."""

from __future__ import annotations

import logging
import argparse
import random
from typing import Sequence

import numpy as np
import torch

from agent.actor_dqn_agent import ActorDQNAgent
from env.env import Maze



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a training run."""
    parser = argparse.ArgumentParser(
        description="Train the Actor-DQN agent on generated mazes."
    )
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--max-episode-steps-eval", type=int, default=16)
    parser.add_argument("--evaluation-frequency", type=int, default=10000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=2048)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=10000.0)
    parser.add_argument("--tensorboard_log_dir", type=str, default="logs/dqn3")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Fraction of online-network weights mixed into the target per update.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Training device (default: CUDA when available, otherwise CPU).",
    )
    return parser.parse_args(argv)


def create_environments(width: int, height: int, seed: int, logger) -> tuple[Maze, Maze]:
    """Create independent environments backed by one generated maze dataset."""
    training_environment = Maze(width=width, height=height)
    evaluation_environment = Maze(width=width, height=height)
    actor_eval_environment = Maze(width=width, height=height)

    # Generation is deterministic for a seed and can be expensive. Both
    # environments may share these arrays because reset() copies a sampled maze
    # before changing it.
    maze_generator_code = training_environment.generate_maze.__func__.__code__
    logger.warning(
        "Maze generation is starting and may take a while (%s:%d).",
        maze_generator_code.co_filename,
        maze_generator_code.co_firstlineno,
    )
    training_environment.generate_maze(seed=seed)
    evaluation_environment.training_mazes = training_environment.training_mazes
    evaluation_environment.evaluation_mazes = training_environment.evaluation_mazes
    actor_eval_environment.training_mazes = training_environment.training_mazes
    actor_eval_environment.evaluation_mazes = training_environment.evaluation_mazes

    # Seed each environment's independent Gymnasium random-number generator.
    training_environment.reset(seed=seed, options={"is_evaluation": False})
    evaluation_environment.reset(seed=seed + 1, options={"is_evaluation": True})
    actor_eval_environment.reset(seed=seed + 1, options={"is_evaluation": True})
    return training_environment, evaluation_environment, actor_eval_environment


def format_metrics(name: str, metrics: tuple[float, float, float]) -> str:
    """Format an evaluation result for the console."""
    average_return, average_steps, success_rate = metrics
    return (
        f"{name}: return={average_return:.3f}, "
        f"steps={average_steps:.1f}, success_rate={success_rate:.1%}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one reproducible Actor-DQN training experiment."""
    logger = logging.getLogger(__name__)
    args = parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    training_environment, evaluation_environment, actor_eval_environment = create_environments(
        width=args.width,
        height=args.height,
        seed=args.seed,
        logger=logger,
    )
    agent = ActorDQNAgent(
        input_dim=training_environment.observation_space.shape,
        output_dim=training_environment.action_space.n,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        tau=args.tau,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
        seed=args.seed,
        total_timesteps=args.timesteps,
        environment=training_environment,
        eval_environment=evaluation_environment,
        eval_num_episodes=args.evaluation_episodes,
        max_episode_steps=args.max_episode_steps,
        max_episode_steps_eval=args.max_episode_steps_eval,
        evaluation_frequency=args.evaluation_frequency,
        actor_eval_environment=actor_eval_environment,
        tensorboard_log_dir=args.tensorboard_log_dir,
    )

    try:
        logger.info(
            f"Training on {args.width}x{args.height} mazes for "
            f"{args.timesteps:,} timesteps using {agent.device}."
        )
        agent.train()
        logger.info(format_metrics("DQN", agent.evaluation(evaluation=True)))
        logger.info(format_metrics("Actor-DQN", agent.actor_evaluation(evaluation=True)))
    finally:
        if agent.tensorboard_writer is not None:
            agent.tensorboard_writer.close()
        training_environment.close()
        evaluation_environment.close()
        actor_eval_environment.close()


if __name__ == "__main__":
    main()
