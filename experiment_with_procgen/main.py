from agent import make_agent
from env import env_factory, N_ENVS
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

def main(base_seed, name_of_game):
    train_env = SubprocVecEnv(
        [env_factory(rank, name_of_game, 0, 200, base_seed) for rank in range(N_ENVS)],
        start_method="spawn",
    )
    eval_env = SubprocVecEnv(
            [env_factory(rank, name_of_game, 201, 0, base_seed) for rank in range(N_ENVS)],
            start_method="spawn",
    )
    train_env = VecMonitor(train_env, info_keywords=("is_success",))
    eval_env = VecMonitor(eval_env, info_keywords=("is_success",))

    train_freq = 4
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{name_of_game}-{base_seed}",
                             log_path=f"./logs/{name_of_game}-{base_seed}", eval_freq=5000*train_freq+1,
                             n_eval_episodes=128,
                             deterministic=True, render=False)
    
    model = make_agent(train_env, seed=base_seed, tensorboard_log=f"./logs/{name_of_game}-{base_seed}")

    model.learn(
        total_timesteps=25_000_000,
        progress_bar=True,
        callback=eval_callback
    )

    model.save(f"./logs/{name_of_game}-{base_seed}/checkpoints/")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    BASE_SEED=42

    for game in ["coinrun", "maze"]:
        for i in range(4):
            main(base_seed=BASE_SEED+i, name_of_game=game)