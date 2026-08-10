"""
RSL-RL PPO runner config, camera variant. Algorithm hyperparameters are the same
verified plan.md Phase 3 numbers as the lidar variant (rsl_rl_ppo_lidar_cfg.py) --
copying MuammerBay/isaac_so_arm101's reference doesn't change based on observation
modality, only the policy network does.

Targets the OLD rsl_rl ActorCritic API (confirmed via source research to match every
tagged Isaac Lab release through 2.3.0 GA) via `policy_class_name`/a custom class --
see nature_cnn_actor_critic.py's docstring FIRST: Isaac Lab's own `main` branch already
pins `rsl-rl-lib==5.0.1` and has a native `RslRlCNNModelCfg`/`RslRlMLPModelCfg` API that
would replace this whole file with a plain cfg (no custom class, no `policy_class_name`
uncertainty). Check which one your Phase 1 install actually has before trusting this
file's shape.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .nature_cnn_actor_critic import NatureCnnActorCritic

LekiwiCameraPPORunnerCfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=3000,
    save_interval=50,
    experiment_name="lekiwi_conenav_camera",
    empirical_normalization=False,  # normalizing raw pixel batches isn't meaningful the way it is for proprioceptive state -- image is scaled to [0,1] inside NatureCnnActorCritic._split_obs instead
    policy_class_name="NatureCnnActorCritic",
    policy=dict(
        class_=NatureCnnActorCritic,
        num_proprio_obs=7,  # base_pose_2d (4) + base_velocity_2d (3), mdp/observations.py
        image_shape=(3, 480, 640),
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
        init_noise_std=1.0,
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
