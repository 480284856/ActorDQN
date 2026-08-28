from stable_baselines3 import DQN


def make_agent(env, seed=42, tensorboard_log="./logs/"):
    return DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        max_grad_norm=10.0,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device="auto",
        verbose=1,
    )