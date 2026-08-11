"""
Shared robot base-position readers. Isaaclab-dependent (needs Articulation/
SceneEntityCfg), unlike _pure_math.py -- that's why this isn't there instead.

Factored out after a real bug (2026-08-10): rewards.py and terminations.py each had
their OWN inline "get the robot's position" reading asset.data.root_pos_w, while
observations.py's independently-written version (reading joint_pos directly) was
correct. Two reimplementations of the same fact, silently drifted apart -- see
robot_local_xy's docstring for the actual root-cause. One function now, imported
everywhere a robot xy position is needed, so there's exactly one place left to get it
wrong.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def robot_local_xy(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Robot's (x, y) in its own course-local frame, read directly from the base_x/base_y
    prismatic joint positions -- NOT asset.data.root_pos_w.

    Why root_pos_w is wrong for this asset (confirmed against the real USD's joint
    graph via pxr, 2026-08-10, not assumed): root_pos_w is the pose of the
    articulation's ROOT link, which for LeKiwi is `/LeKiwi/world` -- it carries
    RigidBodyAPI and its own FixedJoint has an EMPTY body0 (i.e. anchored directly to
    the global inertial frame), while `/LeKiwi` itself (where ArticulationRootAPI
    actually sits) is a bare Xform with no RigidBodyAPI of its own -- so PhysX
    resolves the real root body to `world`. `world` sits at the very TOP of the
    kinematic chain (world -> base_x_link -> base_y_link -> base_theta_link -> base ->
    wheels, confirmed via each joint's body0/body1) and is never driven by anything --
    base_x/base_y/base_theta only move bodies DOWNSTREAM of it. root_pos_w is
    therefore a per-env CONSTANT (robots/lekiwi.py's ArticulationCfg.InitialStateCfg.pos,
    (0,0,0.05) local -- never updated by the joints), not the robot's actual position.
    Confirmed identical topology in both usd/lekiwi_camera.usd and usd/lekiwi_lidar.usd.

    This silently broke three things before the fix: rewards.py's approach_goal_potential
    (distance never changed -> ~zero shaping signal), terminations.py's
    goal_reached_and_held (robot always read as sitting near its spawn-frame origin,
    essentially never actually inside the goal), and terminations.py's out_of_bounds
    (same reason, essentially never triggered either) -- the entire task-completion
    reward/termination signal was effectively dead regardless of where the robot
    actually drove.

    Safe to read base_x/base_y directly as course-local coordinates because
    mdp/events.py:regenerate_course initializes them to spawn_x/spawn_y (course-local,
    from course_generator.CourseLayout) via write_joint_state_to_sim, and they're
    world-aligned prismatic joints (mdp/actions.py's module docstring) -- so their
    value at any later step IS the robot's current position in that same local frame,
    exactly matching CourseLayout's goal_x/goal_y/cone x/y. This is the same read
    mdp/observations.py:base_pose_2d already did correctly (independently written,
    never had this bug) -- factored out here so both share one implementation.
    """
    asset: Articulation = env.scene[robot_cfg.name]
    x = asset.data.joint_pos[:, asset.find_joints("base_x")[0][0]]
    y = asset.data.joint_pos[:, asset.find_joints("base_y")[0][0]]
    return torch.stack([x, y], dim=-1)


def robot_world_xy(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    robot_local_xy converted to world frame (env_origins + local) -- matches how
    mdp/events.py:regenerate_course stores env.goal_pos_w/env.cone_pos_w (also
    origin + local course coordinates), so distance to those can be computed directly
    without a separate frame conversion at each call site.
    """
    return env.scene.env_origins[:, :2] + robot_local_xy(env, robot_cfg)
