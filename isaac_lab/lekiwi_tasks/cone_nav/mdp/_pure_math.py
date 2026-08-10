"""
Pure-tensor math (torch only, zero isaaclab/omni dependency) factored out of
actions.py, rewards.py, and terminations.py specifically so it can be unit-tested
without a real Isaac Sim install -- see isaac_lab/tests/test_mdp_math.py, which
imports this module directly and actually runs it. The isaaclab-dependent files import
these functions rather than reimplementing the formulas inline, so there's one source
of truth for each formula, not a copy that could silently drift out of sync.
"""

from __future__ import annotations

import torch


def body_to_world_velocity(
    vx_b: torch.Tensor, vy_b: torch.Tensor, omega_b: torch.Tensor, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rotate body-frame (vx, vy) into world-frame given current heading theta (standard
    2D rotation matrix, right-hand rule / counterclockwise-positive about Z). omega
    passes through unchanged -- yaw rate is the same in body and world frame for a
    planar robot. See mdp/actions.py's module docstring for why this rotation exists
    at all (base_x/base_y are world-frame joints, the real firmware's command interface
    is body-frame).
    """
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    vx_w = vx_b * cos_t - vy_b * sin_t
    vy_w = vx_b * sin_t + vy_b * cos_t
    return vx_w, vy_w, omega_b


def potential_shaping(prev_dist: torch.Tensor, current_dist: torch.Tensor) -> torch.Tensor:
    """reward = dist_last_step - dist_this_step. Positive when approaching, negative when retreating."""
    return prev_dist - current_dist


def update_hold_counter(inside: torch.Tensor, hold_steps: torch.Tensor) -> torch.Tensor:
    """Consecutive-steps-inside-goal counter: +1 while inside, reset to 0 the instant it isn't."""
    return torch.where(inside, hold_steps + 1, torch.zeros_like(hold_steps))


def success_mask(dist_to_goal_center: torch.Tensor, robot_footprint_radius: float, goal_radius: torch.Tensor) -> torch.Tensor:
    """Full robot footprint (approximated as a bounding circle) entirely inside the goal circle."""
    return dist_to_goal_center + robot_footprint_radius <= goal_radius


def apply_lidar_dropout(ranges: torch.Tensor, dropout_prob: float, max_range: float = 12.0) -> torch.Tensor:
    """Each ray independently drops to max_range (a real "no return") with probability dropout_prob."""
    if dropout_prob <= 0.0:
        return ranges
    dropped = torch.rand_like(ranges) < dropout_prob
    return torch.where(dropped, torch.full_like(ranges, max_range), ranges)


def apply_lidar_angular_jitter(ranges: torch.Tensor, jitter_std_deg: float) -> torch.Tensor:
    """Whole-scan circular shift by a small random number of ray-bins, per env (row)."""
    if jitter_std_deg <= 0.0:
        return ranges
    num_rays = ranges.shape[-1]
    deg_per_ray = 360.0 / num_rays
    shift_rays = torch.round(torch.randn(ranges.shape[0]) * jitter_std_deg / deg_per_ray).long()
    return torch.stack([torch.roll(ranges[i], shifts=int(shift_rays[i].item())) for i in range(ranges.shape[0])])


def linear_anneal(start: float, end: float, progress: float) -> float:
    """progress in [0, 1] -- NOT clamped here (caller's job, e.g. mdp/curriculum.py
    clamps env.common_step_counter / num_steps_to_anneal before calling this)."""
    return start + progress * (end - start)


def out_of_bounds_mask(pos_xy: torch.Tensor, bounds: torch.Tensor, margin_m: float) -> torch.Tensor:
    """
    bounds is (..., 4) = (xmin, xmax, ymin, ymax), matching course_generator.CourseLayout.bounds
    and cone_nav_env.py's course_bounds buffer. pos_xy is (..., 2).
    """
    xmin, xmax, ymin, ymax = bounds.unbind(dim=-1)
    outside_x = (pos_xy[..., 0] < xmin - margin_m) | (pos_xy[..., 0] > xmax + margin_m)
    outside_y = (pos_xy[..., 1] < ymin - margin_m) | (pos_xy[..., 1] > ymax + margin_m)
    return outside_x | outside_y
