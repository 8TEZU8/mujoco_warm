from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points
_load_env_plugins()


# Classic
# ----------------------------------------

register(
    id="Warm-v0",
    entry_point="warm:WarmEnv",
    max_episode_steps=1000,
    reward_threshold=1000.0,
)