from agent import make_agent
from env import make_procgen_env


def main():
    env = make_procgen_env(
        game="maze",
        start_level=0,
        num_levels=200,
        distribution_mode="easy",
    )

    model = make_agent(env, seed=42)

    model.learn(
        total_timesteps=1_000_00,
        progress_bar=True,
    )

    model.save("checkpoints/dqn_procgen_maze")
    env.close()


if __name__ == "__main__":
    main()