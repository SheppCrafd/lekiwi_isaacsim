"""
Observation terms for the cone-avoidance nav task.

Hard rule (plan.md Phase 3): nothing in here may read env.goal_pos_w / env.cone_pos_w
or anything derived from them. Those are privileged, reward-function-only state (see
cone_nav_env.py's docstring) -- the real robot has no such signal at deployment, so the
policy can't be trained to rely on one either. Everything below reads only sensor data
(camera/lidar) or proprioceptive base state, both of which exist on the real hardware.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCaster

from ._pure_math import apply_lidar_angular_jitter, apply_lidar_dropout


def base_pose_2d(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Proprioceptive (x, y, cos(theta), sin(theta)) from the robot's own odometry.

    This is the sim analogue of what the real firmware already tracks from wheel
    odometry -- it's the robot's own estimate of where it is, not simulator ground
    truth about the goal/cones, so it's fine for the policy to see. sin/cos encoding
    avoids the angle-wraparound discontinuity a raw theta value would create.
    """
    asset: Articulation = env.scene[robot_cfg.name]
    x = asset.data.joint_pos[:, asset.find_joints("base_x")[0][0]]
    y = asset.data.joint_pos[:, asset.find_joints("base_y")[0][0]]
    theta = asset.data.joint_pos[:, asset.find_joints("base_theta")[0][0]]
    return torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)


def base_velocity_2d(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Proprioceptive (vx, vy, omega) from the planar joint velocities."""
    asset: Articulation = env.scene[robot_cfg.name]
    vx = asset.data.joint_vel[:, asset.find_joints("base_x")[0][0]]
    vy = asset.data.joint_vel[:, asset.find_joints("base_y")[0][0]]
    omega = asset.data.joint_vel[:, asset.find_joints("base_theta")[0][0]]
    return torch.stack([vx, vy, omega], dim=-1)


def front_camera_rgb(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("front_camera")) -> torch.Tensor:
    """Raw RGB frame, (N, H, W, 3) uint8-range float. Camera-variant policy only."""
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    return sensor.data.output["rgb"][..., :3]


def lidar_ranges(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("rplidar"),
    dropout_prob: float = 0.02,
    angular_jitter_std_deg: float = 0.5,
) -> torch.Tensor:
    """
    2D range scan, (N, num_rays). Distance from sensor origin to each ray hit,
    clamped to the RPLIDAR A1M8's real 0.15-12m valid range (out-of-range hits report
    the sensor's max raycast distance, not real returns, so they're clamped rather than
    trusted past the datasheet's own spec -- see isaac_sim/attach_sensors.py for the
    same real hardware numbers used here).

    Also implements the two Phase 6 sensor-noise items Isaac Lab's stock GaussianNoiseCfg
    (env_cfg_lidar.py's range-noise term) can't express, since that's a simple additive
    op on the whole tensor, not built for zeroing a subset of rays or perturbing the
    angular sample grid itself:
      - dropout_prob: each ray independently has this probability of reporting "no
        return" -- set to max range (12.0m), matching how a real 2D lidar actually
        reports a missed/invalid return, NOT zero (zero would look like "obstacle
        touching the sensor," which is physically backwards and would badly mislead
        the policy). 2% per ray is a starting default, not a datasheet spec -- the
        A1M8's own datasheet doesn't publish a dropout rate.
      - angular_jitter_std_deg: modeled as a small random circular shift of the whole
        360-ray array (not independent per-ray angle noise, which isn't recoverable
        from a post-hoc range vector without the raw per-ray geometry) -- approximates
        the sensor's whole angular reference frame being very slightly offset, a
        real effect of mechanical assembly tolerance. Documented approximation, not a
        physically exact per-ray angular noise model.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    origins = sensor.data.pos_w.unsqueeze(1)
    hits = sensor.data.ray_hits_w
    ranges = torch.linalg.norm(hits - origins, dim=-1)
    ranges = torch.clamp(ranges, 0.15, 12.0)
    ranges = apply_lidar_dropout(ranges, dropout_prob)
    ranges = apply_lidar_angular_jitter(ranges, angular_jitter_std_deg)
    return ranges
