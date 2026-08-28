from agent import make_agent
from env import env_factory, N_ENVS
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def main():
    env = SubprocVecEnv(
        [env_factory(rank) for rank in range(N_ENVS)],
        start_method="spawn",
    )
    env = VecMonitor(env)

    model = make_agent(env, seed=42)

    model.learn(
        total_timesteps=25_000_000,
        progress_bar=True,
    )

    model.save("checkpoints/dqn_procgen_maze")
    env.close()


if __name__ == "__main__":
    main()