"""
Custom action term: body-frame (vx, vy, omega) -> base_x/base_y/base_theta joint
velocity targets.

Why this needs to be custom rather than a stock JointVelocityActionCfg: the real
LeKiwi firmware takes body-frame vx/vy/omega and does its own wheel inverse-kinematics
(isaac_sim/README_lekiwi_variants.md). But base_x/base_y are WORLD-frame prismatic
joints in this asset's actual joint chain -- `world -> base_x (prismatic X) -> base_y
(prismatic Y) -> base_theta (revolute Z) -> base` (same file, "Important architectural
fact" section) -- base_theta's rotation is applied *after* the X/Y translation, so
commanding base_x_vel/base_y_vel directly moves the robot along world X/Y regardless of
its current heading, not the direction it's facing. Sending a policy's raw
(vx, vy, omega) straight to those three joints would silently train it against a
world-frame interface the real robot doesn't have.

This term does the body->world rotation explicitly (using the robot's own current
base_theta each step) so the policy's action space matches the real vx/vy/omega
interface it'll actually be given at deployment (plan.md Phase 10's stated goal --
doing the rotation here instead of punting it to deployment removes exactly the gap
Phase 10 flags as "the first point where the sim/real command interfaces actually have
to agree").

This class's base-class contract (`action_dim`/`raw_actions`/`processed_actions`
properties, `process_actions(actions)`/`apply_actions()`/`reset(env_ids)` methods) was
confirmed against Isaac Lab's real `isaaclab/managers/action_manager.py` source
(fetched directly, not guessed) -- matches exactly.

Also owns two Phase 6 domain-randomization hooks that only make sense inside the
action pipeline, not as scene EventTerms:
  - Actuation latency: actions are applied `latency_steps` control-steps late, per env.
  - Wheel-ground "slip": since locomotion is a virtual planar joint, not simulated
    wheel-ground contact (same README section), a physics_material friction
    randomization on the decorative fixed-joint wheels would do *nothing* to
    locomotion -- there's no contact-driven rolling to affect. Slip is instead injected
    directly into the action-to-motion mapping, as a per-env efficiency multiplier on
    commanded velocity. Both are randomized once per episode by
    mdp/events.py:randomize_actuation (not per-step -- that would make the dynamics
    non-Markovian in a way the latency buffer here is specifically built to handle;
    true per-step actuation noise is a further extension, not implemented here).
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ._pure_math import body_to_world_velocity


class BodyVelocityAction(ActionTerm):
    cfg: "BodyVelocityActionCfg"

    def __init__(self, cfg: "BodyVelocityActionCfg", env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(["base_x", "base_y", "base_theta"])

        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=env.device)

        # Ring buffer for actuation-latency injection. max_latency_steps sized for the
        # worst case events.py will ever sample (see mdp/events.py:randomize_actuation);
        # per-env latency draws an index into this buffer, so no per-env-shaped dynamic
        # allocation is needed at reset time.
        self.max_latency_steps = cfg.max_latency_steps
        self._action_history = torch.zeros(env.num_envs, self.max_latency_steps + 1, 3, device=env.device)
        self.latency_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.slip_factor = torch.ones(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clipped = torch.clamp(actions, -1.0, 1.0)
        vx_b = clipped[:, 0] * self.cfg.max_lin_vel
        vy_b = clipped[:, 1] * self.cfg.max_lin_vel
        omega = clipped[:, 2] * self.cfg.max_ang_vel

        # Push this step's commanded (body-frame) velocity into the latency ring buffer.
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._action_history[:, -1, 0] = vx_b
        self._action_history[:, -1, 1] = vy_b
        self._action_history[:, -1, 2] = omega

        # Read back whatever was commanded `latency_steps` control-steps ago, per env.
        idx = (self.max_latency_steps - self.latency_steps).clamp(0, self.max_latency_steps)
        delayed = torch.gather(
            self._action_history, 1, idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)

        theta = self._asset.data.joint_pos[:, self._joint_ids[2]]
        vx_w, vy_w, omega_w = body_to_world_velocity(delayed[:, 0], delayed[:, 1], delayed[:, 2], theta)

        self._processed_actions[:, 0] = vx_w * self.slip_factor
        self._processed_actions[:, 1] = vy_w * self.slip_factor
        self._processed_actions[:, 2] = omega_w * self.slip_factor

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._action_history[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        # latency_steps / slip_factor are NOT reset here -- events.py:randomize_actuation
        # (mode="reset") sets fresh values for the new episode; resetting them to a
        # default here would just get overwritten, but doing it there keeps "what
        # varies per episode" in one place (events.py) instead of split across two files.


@configclass
class BodyVelocityActionCfg(ActionTermCfg):
    class_type: type = BodyVelocityAction
    max_lin_vel: float = 0.5  # m/s -- placeholder, tune against real firmware limits once hardware exists (Phase 9)
    max_ang_vel: float = 2.0  # rad/s -- placeholder, same caveat
    max_latency_steps: int = 6  # at 30Hz control (Phase 3), 6 steps = 200ms ceiling
