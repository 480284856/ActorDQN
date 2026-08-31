from agent import make_agent
from env import env_factory, N_ENVS
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

def main():
    train_env = SubprocVecEnv(
        [env_factory(rank, 'maze', 200) for rank in range(N_ENVS)],
        start_method="spawn",
    )
    eval_env = SubprocVecEnv(
            [env_factory(rank, 'maze', 0) for rank in range(N_ENVS)],
            start_method="spawn",
        )
    train_env = VecMonitor(train_env, info_keywords=("is_success",))
    eval_env = VecMonitor(eval_env, info_keywords=("is_success",))

    train_freq = 4
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/maze",
                             log_path="./logs/maze", eval_freq=10000*train_freq+1,
                             n_eval_episodes=100,
                             deterministic=True, render=False)
    
    model = make_agent(train_env, seed=42)

    model.learn(
        total_timesteps=25_000_000,
        progress_bar=True,
        callback=eval_callback
    )

    model.save("checkpoints/dqn_procgen_maze")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()