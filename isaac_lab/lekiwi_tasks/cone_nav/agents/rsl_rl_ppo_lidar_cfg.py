"""
RSL-RL PPO runner config, lidar variant. Every hyperparameter here is copied from the
real, verified reference plan.md Phase 3 pulled its numbers from --
MuammerBay/isaac_so_arm101's actual reach_env_cfg.py/rsl_rl_ppo_cfg.py (BSD-3-Clause) --
not guessed. [64,64] MLP applies directly here since lidar_ranges + base_pose_2d +
base_velocity_2d (mdp/observations.py) concatenate into one flat vector
(env_cfg_lidar.py's ObservationsCfg sets concatenate_terms=True for exactly this),
the same shape of problem SO-101's joint-state observations are.

CONFIRMED (not just written from memory): this cfg shape -- single `policy:
RslRlPpoActorCriticCfg` field, `actor_hidden_dims`/`critic_hidden_dims`, `init_noise_std`
-- was checked against real Isaac Lab source across every tagged release from v2.1.0
through the current v2.3.0 GA (fetched `_modules/isaaclab_rl/rsl_rl/rl_cfg.html` for
each) and matches exactly. The one open question is forward-compatibility: Isaac Lab's
`main` branch has already moved on to a newer actor/critic-split API (see
agents/nature_cnn_actor_critic.py's docstring for the detail) -- if your Phase 1 install
is newer than 2.3.0, this file may need to move to that shape too. `isaaclab_tasks.utils`
(used in scripts/train.py and scripts/play.py) is confirmed real and current; the
`isaaclab_tasks.utils.wrappers.rsl_rl` path this comment used to mention as an older-
install fallback was not confirmed to exist and shouldn't be assumed.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


LekiwiLidarPPORunnerCfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=3000,  # generous ceiling for a 50-hour budget (plan.md Phase 7); real stopping point is convergence, watched live, not this number
    save_interval=50,
    experiment_name="lekiwi_conenav_lidar",
    empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)
