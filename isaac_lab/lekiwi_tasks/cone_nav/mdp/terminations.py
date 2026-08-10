"""
Termination terms (plan.md Phase 5). plan.md's original spec only had two: cone hit
(fail) and full-footprint-in-goal (success). Its own review flagged both a missing
timeout ("without one, an episode that never hits a cone or reaches the goal never
resets, which breaks batched on-policy training") and a missing out-of-bounds check --
both added here as their own explicit terms rather than folded into the other two, so
the reward/log code can tell *why* an episode ended, not just that it did.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ._pure_math import out_of_bounds_mask, success_mask


def cone_collision(env, contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact")) -> torch.Tensor:
    """Same contact threshold as rewards.cone_collision_penalty -- keep both in sync."""
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    return contact_sensor.data.net_forces_w.norm(dim=-1).max(dim=1)[0] > 1.0


def goal_reached_and_held(
    env,
    hold_steps_required: int = 15,
    robot_footprint_radius: float = 0.16,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Mirrors rewards.success_bonus's hold-time logic (both read env.success_hold_steps,
    which rewards.success_bonus is what actually increments -- ensure that reward term
    runs in the same step's manager pass so this sees an up-to-date count, not a
    one-step-stale one).
    """
    asset: Articulation = env.scene[robot_cfg.name]
    pos = asset.data.root_pos_w[:, :2]
    dist = torch.linalg.norm(pos - env.goal_pos_w[:, :2], dim=-1)
    inside = success_mask(dist, robot_footprint_radius, env.goal_radius)
    return inside & (env.success_hold_steps >= hold_steps_required)


def out_of_bounds(
    env,
    margin_m: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Fail condition for driving outside the generated course's open area without
    touching a cone or the goal -- plan.md: "those episodes just burn to timeout
    uselessly" without this. Bounds come from the per-env course layout written by
    mdp/events.py's regenerate_course (env.course_bounds, set alongside goal/cone
    state), not a single global constant, since course footprint size varies per
    generated environment (course_generator.CourseGeneratorCfg.width_range_m /
    length_range_m).
    """
    asset: Articulation = env.scene[robot_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return out_of_bounds_mask(pos, env.course_bounds, margin_m)
