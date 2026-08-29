import gym
import gymnasium

class ProcgenGymnasiumAdapter(gymnasium.Env):
    """Expose Procgen's legacy Gym API through the Gymnasium API."""

    def __init__(self, legacy_env):
        super().__init__()
        self.env = legacy_env

        old_obs = self.env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=old_obs.low,
            high=old_obs.high,
            shape=old_obs.shape,
            dtype=old_obs.dtype,
        )
        self.action_space = gymnasium.spaces.Discrete(
            self.env.action_space.n
        )

    def reset(self, seed=None, options=None):
        # Procgen's legacy ToGymEnv.seed() does not accept a seed argument.
        # Level generation is controlled by start_level/num_levels instead.
        super().reset(seed=seed)
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:
            info["is_success"] = bool(
                info.get("prev_level_complete", False)
            )

        return obs, float(reward), bool(done), False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

def make_procgen_env(
    game="maze",
    start_level=0,
    num_levels=200,
    distribution_mode="easy",
    render_mode=None,
    seed=42,
):
    legacy_env = gym.make(
        f"procgen:procgen-{game}-v0",
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=distribution_mode,
        render_mode=render_mode,
        rand_seed=seed,
    )

    return ProcgenGymnasiumAdapter(legacy_env)

N_ENVS = 16
BASE_SEED = 42

def env_factory(rank, game_name, num_levels):
    def create_env():
        return make_procgen_env(
            game=game_name,
            start_level=0,
            num_levels=num_levels,
            distribution_mode="easy",
            # each environment will sample levels differently from pool
            seed=BASE_SEED + rank,
        )
    return create_env


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = make_procgen_env()
    check_env(env)
