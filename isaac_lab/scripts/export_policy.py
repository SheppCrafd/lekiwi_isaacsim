"""
Export a trained checkpoint's actor network to ONNX + TorchScript (plan.md Phase 10:
"export the trained policy in a runtime the Pi can actually execute"). Run after Phase 8
eval passes, before Phase 10's on-robot deployment.

    python export_policy.py --task lidar --checkpoint logs/rsl_rl/lekiwi_conenav_lidar/model_2500.pt --out lidar_policy

UPDATE (2026-08-10): "the lidar variant exports cleanly" is no longer true as written.
rsl_rl.modules.ActorCritic (what the lidar path targets) is CONFIRMED missing from
both the latest rsl-rl-lib (5.4.2) and the exact version Isaac Lab's main branch pins
(5.0.1) -- verified by directly installing and inspecting each, not assumed. See
_load_rsl_rl_actor_critic()'s docstring below for the detail and the real fix path
(isaaclab_rl.rsl_rl's native RslRlCNNModelCfg/RslRlMLPModelCfg). This script now fails
loudly with an actionable message when that import doesn't exist, rather than either a
bare ImportError or a false claim that this path works.

The camera variant's export is the other part of this script that inherits
nature_cnn_actor_critic.py's own flagged uncertainty -- if act_inference's dict-vs-flat
observation handling needed a fix in Phase 2 for training to work at all, apply the
same fix to CameraPolicyWrapper.forward below before trusting this export. Unlike the
lidar path, NatureCnnActorCritic is this project's own class (not a third-party import
that can vanish out from under it), so it's not affected by the rsl_rl removal above --
just by its own separately-documented uncertainty.
"""

from __future__ import annotations

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["camera", "lidar"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="Output path prefix, e.g. 'lidar_policy' -> lidar_policy.onnx / .pt")
args = parser.parse_args()

# lidar variant observation width: 360 lidar_ranges + 4 base_pose_2d + 3 base_velocity_2d
# (mdp/observations.py), concatenated (env_cfg_lidar.py's concatenate_terms=True). 360
# comes from LidarPatternCfg(horizontal_res=1.0) over the full -180..180 sweep
# (env_cfg_lidar.py) -- update this constant too if that pattern cfg ever changes.
LIDAR_NUM_RAYS = 360
NUM_PROPRIO_OBS = 7  # base_pose_2d (4) + base_velocity_2d (3), both variants
NUM_ACTIONS = 3  # vx, vy, omega -- mdp/actions.py:BodyVelocityAction.action_dim


def _load_rsl_rl_actor_critic():
    """
    CONFIRMED BROKEN (2026-08-10) -- not a hypothetical forward-compat risk anymore.
    Verified directly (pip install + inspect, not assumed): rsl_rl.modules.ActorCritic
    does not exist in the latest rsl-rl-lib release (5.4.2) OR the exact version Isaac
    Lab's main branch pins (5.0.1, per agents/nature_cnn_actor_critic.py's own
    research) -- both were installed and checked. It's been fully replaced by a
    modular MLP/CNN/GaussianDistribution split (rsl_rl.modules), with the actual
    actor-critic orchestration apparently living inside isaaclab_rl.rsl_rl's own
    wrapper (RslRlCNNModelCfg/RslRlMLPModelCfg) -- not verifiable further here since
    isaaclab_rl isn't pip-installable standalone (it needs Isaac Lab itself). Raises a
    clear, actionable error here instead of a bare ImportError, so this is unmistakable
    the moment Phase 1 hits it rather than a confusing stack trace.
    """
    try:
        from rsl_rl.modules import ActorCritic

        return ActorCritic
    except ImportError as e:
        raise ImportError(
            "rsl_rl.modules.ActorCritic does not exist in your installed rsl-rl-lib "
            "(confirmed missing in both 5.4.2 and 5.0.1 -- checked directly, 2026-08-10). "
            "This script's lidar-variant export path targets the OLD single-class API and "
            "needs to move to isaaclab_rl.rsl_rl's native RslRlCNNModelCfg/RslRlMLPModelCfg "
            "path instead (see agents/nature_cnn_actor_critic.py's module docstring for "
            "the shape). Check `python -c \"from isaaclab_rl.rsl_rl import RslRlMLPModelCfg\"` "
            "first."
        ) from e


class LidarPolicyWrapper(torch.nn.Module):
    """Flat-vector-in, action-out -- exports as a single graph, no branching needed."""

    def __init__(self, actor: torch.nn.Sequential):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class CameraPolicyWrapper(torch.nn.Module):
    """image (N,3,480,640) + proprio (N,7) in, action out. See module docstring."""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        features = self.actor_critic.actor_cnn(image.float() / 255.0)
        return self.actor_critic.actor_head(torch.cat([features, proprio], dim=-1))


def main() -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # rsl_rl checkpoint format: {"model_state_dict": ..., "optimizer_state_dict": ..., ...}
    # -- verify this key against your installed rsl_rl version's actual save format
    # (OnPolicyRunner.save) before assuming it matches.
    state_dict = checkpoint["model_state_dict"]

    if args.task == "lidar":
        from lekiwi_tasks.cone_nav.agents.rsl_rl_ppo_lidar_cfg import LekiwiLidarPPORunnerCfg

        ActorCritic = _load_rsl_rl_actor_critic()
        lidar_obs_dim = LIDAR_NUM_RAYS + NUM_PROPRIO_OBS
        # num_actor_obs/num_critic_obs/num_actions are NOT in LekiwiLidarPPORunnerCfg.policy
        # (rsl_rl_ppo_lidar_cfg.py's RslRlPpoActorCriticCfg only holds hidden_dims/
        # activation/init_noise_std) -- BUGFIX (2026-08-10): this used to call
        # ActorCritic(**policy.to_dict()) alone, missing these three required
        # constructor args entirely. At real training time Isaac Lab's own runner
        # injects them from the live env's actual observation/action space; standalone
        # reconstruction here needs them supplied explicitly instead. Symmetric
        # actor/critic obs (both = lidar_obs_dim) since this task has no privileged
        # critic-only observation (plan.md Phase 3's observation policy applies to
        # both actor and critic identically). Whether RslRlPpoActorCriticCfg.to_dict()
        # includes any extra Isaac-Lab-only metadata field incompatible with rsl_rl's
        # raw constructor is still unverified (same category of risk this call already
        # carried before this fix, not a new one introduced by it).
        ac = ActorCritic(
            num_actor_obs=lidar_obs_dim,
            num_critic_obs=lidar_obs_dim,
            num_actions=NUM_ACTIONS,
            **LekiwiLidarPPORunnerCfg.policy.to_dict(),
        )
        ac.load_state_dict(state_dict)
        wrapper = LidarPolicyWrapper(ac.actor)
        dummy_input = (torch.zeros(1, lidar_obs_dim),)
        input_names = ["observation"]
    else:
        from lekiwi_tasks.cone_nav.agents.nature_cnn_actor_critic import NatureCnnActorCritic
        from lekiwi_tasks.cone_nav.agents.rsl_rl_ppo_camera_cfg import LekiwiCameraPPORunnerCfg

        # Sourced from LekiwiCameraPPORunnerCfg.policy rather than re-hardcoded here --
        # BUGFIX (2026-08-10): this used to duplicate num_proprio_obs/image_shape/
        # hidden_dims as separate literals, which would silently drift from the actual
        # trained architecture if that cfg's hyperparameters ever changed without a
        # matching edit here. "class_" is Isaac Lab's own routing key (which class to
        # instantiate), not a NatureCnnActorCritic constructor kwarg -- popped before
        # unpacking. num_actions isn't in the cfg dict (env-derived at real training
        # time, same reasoning as the lidar path above), so it's supplied explicitly.
        policy_kwargs = dict(LekiwiCameraPPORunnerCfg.policy)
        policy_kwargs.pop("class_", None)
        policy_kwargs.setdefault("num_actions", NUM_ACTIONS)
        ac = NatureCnnActorCritic(**policy_kwargs)
        ac.load_state_dict(state_dict)
        wrapper = CameraPolicyWrapper(ac)
        dummy_input = (torch.zeros(1, 3, 480, 640), torch.zeros(1, NUM_PROPRIO_OBS))
        input_names = ["image", "proprio"]

    wrapper.eval()

    torch.onnx.export(
        wrapper, dummy_input, f"{args.out}.onnx",
        input_names=input_names, output_names=["action"],
        dynamic_axes={name: {0: "batch"} for name in input_names} | {"action": {0: "batch"}},
        opset_version=17,
    )
    scripted = torch.jit.trace(wrapper, dummy_input)
    scripted.save(f"{args.out}.pt")

    print(f"Exported {args.out}.onnx and {args.out}.pt")
    print("Sanity-check both against the live rsl_rl policy's output on the same input before trusting either for deployment.")


if __name__ == "__main__":
    main()
