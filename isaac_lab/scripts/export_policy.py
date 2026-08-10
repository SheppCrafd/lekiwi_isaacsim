"""
Export a trained checkpoint's actor network to ONNX + TorchScript (plan.md Phase 10:
"export the trained policy in a runtime the Pi can actually execute"). Run after Phase 8
eval passes, before Phase 10's on-robot deployment.

    python export_policy.py --task lidar --checkpoint logs/rsl_rl/lekiwi_conenav_lidar/model_2500.pt --out lidar_policy

The lidar variant exports cleanly (plain MLP on a flat observation vector). The camera
variant's export is the one part of this script that inherits
nature_cnn_actor_critic.py's own flagged uncertainty -- if act_inference's dict-vs-flat
observation handling needed a fix in Phase 2 for training to work at all, apply the
same fix to CameraPolicyWrapper.forward below before trusting this export.
"""

from __future__ import annotations

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["camera", "lidar"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="Output path prefix, e.g. 'lidar_policy' -> lidar_policy.onnx / .pt")
args = parser.parse_args()


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
        from rsl_rl.modules import ActorCritic

        ac = ActorCritic(**LekiwiLidarPPORunnerCfg.policy.to_dict())
        ac.load_state_dict(state_dict)
        wrapper = LidarPolicyWrapper(ac.actor)
        dummy_input = (torch.zeros(1, ac.actor[0].in_features),)
        input_names = ["observation"]
    else:
        from lekiwi_tasks.cone_nav.agents.nature_cnn_actor_critic import NatureCnnActorCritic

        ac = NatureCnnActorCritic(num_proprio_obs=7, num_actions=3, image_shape=(3, 480, 640))
        ac.load_state_dict(state_dict)
        wrapper = CameraPolicyWrapper(ac)
        dummy_input = (torch.zeros(1, 3, 480, 640), torch.zeros(1, 7))
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
