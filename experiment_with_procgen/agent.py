from stable_baselines3 import DQN


def make_agent(env, seed=42, tensorboard_log="./logs/"):
    return DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,                    # how many sample should be sampled from the buffer at each optimization stage.
        gamma=0.99,
        train_freq=4,                      # the interval of steps between two training stages.
        gradient_steps=2,                  # In each training stage, how many optimization steps should be performed.
        target_update_interval=10_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        max_grad_norm=10.0,                # scaler used in gradient clipping. If the norm of gradient is larger than 10,
                                           # scale each parameter by 10/||G||
        tensorboard_log=tensorboard_log,
        seed=seed,
        device="auto",
        verbose=1,
    )