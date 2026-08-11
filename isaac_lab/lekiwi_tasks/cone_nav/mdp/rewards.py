"""
Reward terms (plan.md Phase 5). Every term here exists because plan.md's own review
flagged the original "small shaping + one large success reward" spec as incomplete for
actually training PPO -- each docstring notes which gap it closes.

Privileged state (env.goal_pos_w) is used freely in here -- rewards are allowed to
cheat with ground truth (plan.md Phase 3). Never import anything from this module into
observations.py.

BUGFIX (2026-08-10): approach_goal_potential and success_bonus used to read the
robot's position via a local _robot_pos_xy() that returned asset.data.root_pos_w --
a per-env CONSTANT for this asset's fixed-base joint chain, not the robot's actual
driven position (see mdp/_robot_state.py:robot_local_xy's docstring for the full
root-cause). approach_goal_potential's shaping reward was therefore always ~zero
(distance never changed step to step) and success_bonus could basically never fire --
the dominant part of the reward signal for the actual navigation task was dead
regardless of where the robot drove. Now read via mdp/_robot_state.py, shared with
terminations.py (which had the identical bug, same root cause).
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from ._pure_math import potential_shaping, success_mask, update_hold_counter
from ._robot_state import robot_world_xy


def approach_goal_potential(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Potential-based shaping: reward = dist_last_step - dist_this_step (positive for
    getting closer, negative for retreating), NOT raw `-distance`. Raw-distance shaping
    is the reward-hacking magnet plan.md calls out explicitly -- an agent can camp near
    the goal boundary farming incremental reward instead of finishing. Potential-based
    shaping sums to a bounded telescoping total over an episode regardless of path
    length, which raw distance doesn't.

    env.prev_dist_to_goal is updated here (not read-only) since this is the one place
    in the reward graph that needs "distance as of last step" -- termination checks
    (terminations.py) recompute their own current-step distance independently rather
    than reusing this term's side effect, so term ordering in the reward manager cfg
    doesn't matter for correctness.
    """
    pos = robot_world_xy(env, robot_cfg)
    dist = torch.linalg.norm(pos - env.goal_pos_w[:, :2], dim=-1)
    shaped = potential_shaping(env.prev_dist_to_goal, dist)
    env.prev_dist_to_goal = dist
    return shaped


def success_bonus(
    env,
    hold_steps_required: int = 15,  # ~0.5s at 30Hz control rate (plan.md Phase 3)
    robot_footprint_radius: float = 0.16,  # matches course_generator.CourseGeneratorCfg.robot_radius_m
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    One large reward, but only once success has actually been *held*, not on
    instantaneous full-containment -- plan.md flags a policy that "blasts through the
    area" as a real failure mode without a hold requirement. env.success_hold_steps
    tracks consecutive in-goal steps per env (reset to 0 the instant the robot leaves
    the goal region); this term fires the bonus exactly once, on the step the hold
    threshold is first crossed, using a mask rather than `>=` alone so it can't re-fire
    every subsequent step the robot happens to stay parked.
    """
    pos = robot_world_xy(env, robot_cfg)
    dist = torch.linalg.norm(pos - env.goal_pos_w[:, :2], dim=-1)
    inside = success_mask(dist, robot_footprint_radius, env.goal_radius)

    env.success_hold_steps = update_hold_counter(inside, env.success_hold_steps)
    just_reached = env.success_hold_steps == hold_steps_required
    return just_reached.float()


def cone_collision_penalty(env, contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact")) -> torch.Tensor:
    """
    Explicit negative reward on cone contact -- plan.md: "a reset alone is weak
    signal... add a real penalty so failure is distinguishable from ran out of time."
    Reads the same contact sensor terminations.py's cone_collision uses, so the penalty
    and the episode-ending termination agree on what counts as a hit.
    """
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    has_contact = contact_sensor.data.net_forces_w.norm(dim=-1).max(dim=1)[0] > 1.0
    return -has_contact.float()


def action_smoothness_penalty(env) -> torch.Tensor:
    """
    Energy/smoothness penalty on action deltas -- without this, PPO commonly finds
    jerky bang-bang control that's fine in sim but doesn't survive real motor/torque
    limits (plan.md Phase 5). Penalizes step-to-step action change, not raw action
    magnitude, so a robot holding a steady velocity isn't penalized for the velocity
    itself, only for jerking it around.
    """
    return -torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1)
