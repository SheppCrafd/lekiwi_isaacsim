"""Gym registration for both cone-nav variants. Import this package before gym.make()."""

import gymnasium as gym

from .cone_nav_env import LekiwiConeNavEnv
from .env_cfg_camera import LekiwiCameraConeNavEnvCfg
from .env_cfg_lidar import LekiwiLidarConeNavEnvCfg

gym.register(
    id="Lekiwi-ConeNav-Camera-v0",
    entry_point=f"{LekiwiConeNavEnv.__module__}:{LekiwiConeNavEnv.__name__}",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LekiwiCameraConeNavEnvCfg,
        "rsl_rl_cfg_entry_point": "isaac_lab.lekiwi_tasks.cone_nav.agents.rsl_rl_ppo_camera_cfg:LekiwiCameraPPORunnerCfg",
    },
)

gym.register(
    id="Lekiwi-ConeNav-Lidar-v0",
    entry_point=f"{LekiwiConeNavEnv.__module__}:{LekiwiConeNavEnv.__name__}",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LekiwiLidarConeNavEnvCfg,
        "rsl_rl_cfg_entry_point": "isaac_lab.lekiwi_tasks.cone_nav.agents.rsl_rl_ppo_lidar_cfg:LekiwiLidarPPORunnerCfg",
    },
)
