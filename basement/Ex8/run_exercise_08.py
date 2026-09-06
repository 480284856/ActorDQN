import gymnasium as gym
import torch

from ActorDQN.basement.Ex8.agent import DQNAgent
from ActorDQN.basement.Ex8.dqn import DQN
from ActorDQN.basement.Ex8.replay_buffer import ReplayMemory
from ActorDQN.basement.Ex8.trainer import Trainer


def main():
    # Initializing all components, setting all hyperparameter
    # environment_name = "LunarLander-v3"
    environment_name = "CartPole-v1"
    train_environment = gym.make(environment_name, render_mode=None)
    evaluation_environment = gym.make(environment_name, render_mode="human")

    # Initializing the neural network
    deep_q_network = DQN(
        observation_dimensions=train_environment.observation_space.shape[0],
        number_of_actions=train_environment.action_space.n,
    )

    # Initializing the optimizer and telling it which parameters to optimize,
    # what objective is being optimized is determined by the gradients these parameters receive
    optimizer = torch.optim.Adam(params=deep_q_network.parameters(), lr=1e-4)

    dqn_agent = DQNAgent(
        dqn=deep_q_network,
        optimizer=optimizer,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=1000,
        tau=0.005,
        action_space=train_environment.action_space,
        discount=0.99,
    )

    replay_buffer = ReplayMemory(capacity=10000)

    trainer = Trainer(
        train_environment=train_environment,
        evaluation_environment=evaluation_environment,
        agent=dqn_agent,
        memory=replay_buffer,
        batch_size=128,
    )

    trainer.train(num_episodes=200)

    trainer.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
