"""Gymnasium registration and convenience imports for City Drive RL."""

from gymnasium.envs.registration import register, registry

ENV_ID = "CityDrive-v0"

# Importing this package registers the environment once. This makes the normal
# Gymnasium workflow work: import city_drive_rl; env = gym.make("CityDrive-v0").
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="ai.training_env:RacingEnv",
        order_enforce=True,
        disable_env_checker=False,
    )

from ai.training_env import (  # noqa: E402
    ActionMode,
    ObservationMode,
    RacingEnv,
    TrainingConfig,
    make_racing_env,
)

__all__ = [
    "ENV_ID",
    "ActionMode",
    "ObservationMode",
    "RacingEnv",
    "TrainingConfig",
    "make_racing_env",
]
