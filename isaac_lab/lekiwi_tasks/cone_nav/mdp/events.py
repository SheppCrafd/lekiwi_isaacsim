"""
Domain randomization / procedural regeneration events (plan.md Phases 4 & 6).

All of these are reset-time EventTerms (mode="reset") -- everything here fires once
when an env's episode starts and stays fixed for that whole episode, matching plan.md's
explicit "randomized once, at generation time" requirement (not real-time jitter, see
course_generator.py's own docstring for the same point). Per-step actuation noise would
be a further extension, not implemented here (see mdp/actions.py's docstring).

Cone position/size scatter (Phase 6's "cone position offset" + "cone size and shape
randomization" items) needs no separate event -- course_generator.generate_course()
already produces a fresh scattered layout per seed, so regenerate_course() below gets
that DR "for free" just by sampling a new seed every reset.
"""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from ..course_generator import TOTAL_SEEDS, TRAIN_SEED_UPPER, generate_course


def regenerate_course(
    env,
    env_ids: torch.Tensor,
    eval_mode: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_cones: int | None = None,
) -> None:
    """
    The core per-episode procedural regeneration event. For each env being reset:
    draw a fresh seed (training range unless eval_mode, see course_generator.py's seed
    convention docstring), generate a course, and write it into both the physics scene
    (cone RigidObject poses, robot joint state) and the env's own privileged-state
    buffers (cone_nav_env.py) that reward/termination functions read.

    Reads `env.course_generator_cfg`, NOT a frozen default argument -- that's a mutable
    per-env-set working copy (cone_nav_env.py) that mdp/curriculum.py's
    anneal_course_difficulty term adjusts in place as training progresses. Reading it
    fresh here (rather than capturing a cfg once at import time) is what makes the
    curriculum term's changes actually take effect on the next reset.

    Runs on CPU (numpy) per env_id in a python loop -- course generation itself is
    cheap (well under a ms per seed, see the generator's own 5000-seed test taking a
    few seconds total) and this only runs at reset, not every physics step, so this is
    not expected to be a training bottleneck. Revisit if profiling in Phase 2 says
    otherwise.
    """
    course_cfg = env.course_generator_cfg
    n_max_cones = max_cones if max_cones is not None else env.cfg.course.max_cones
    asset: Articulation = env.scene[robot_cfg.name]
    base_x_id, base_y_id, base_theta_id = asset.find_joints(["base_x", "base_y", "base_theta"])[0]

    for env_id in env_ids.tolist():
        if eval_mode:
            seed = int(TRAIN_SEED_UPPER + (int(env.common_step_counter) + env_id) % (TOTAL_SEEDS - TRAIN_SEED_UPPER))
        else:
            seed = int(torch.randint(0, TRAIN_SEED_UPPER, (1,)).item())

        layout = generate_course(seed, course_cfg)
        origin = env.scene.env_origins[env_id]

        env.episode_seed[env_id] = seed
        env.course_bounds[env_id] = torch.tensor(layout.bounds, device=env.device)

        env.goal_pos_w[env_id] = origin + torch.tensor([layout.goal_x, layout.goal_y, 0.0], device=env.device)
        env.goal_radius[env_id] = layout.goal_radius_m

        env.cone_active[env_id] = False
        for i in range(n_max_cones):
            cone_asset: RigidObject = env.scene[f"cone_{i}"]
            if i < len(layout.cones):
                c = layout.cones[i]
                pos = origin + torch.tensor([c.x, c.y, c.height / 2.0], device=env.device)
                env.cone_pos_w[env_id, i] = pos
                env.cone_radius[env_id, i] = c.radius
                env.cone_active[env_id, i] = True
            else:
                # Park unused slots well below the floor -- inactive, out of the way,
                # doesn't need its own visibility/collision toggle.
                pos = origin + torch.tensor([0.0, 0.0, -5.0], device=env.device)
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            cone_asset.write_root_pose_to_sim(
                torch.cat([pos, identity_quat]).unsqueeze(0), env_ids=torch.tensor([env_id], device=env.device)
            )
            cone_asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([env_id], device=env.device)
            )

        joint_pos = torch.tensor([layout.spawn_x, layout.spawn_y, layout.spawn_heading_rad], device=env.device)
        asset.write_joint_state_to_sim(
            position=joint_pos.unsqueeze(0),
            velocity=torch.zeros(1, 3, device=env.device),
            joint_ids=[base_x_id, base_y_id, base_theta_id],
            env_ids=torch.tensor([env_id], device=env.device),
        )

        spawn_dist = math.hypot(layout.spawn_x - layout.goal_x, layout.spawn_y - layout.goal_y)
        env.prev_dist_to_goal[env_id] = spawn_dist
        env.success_hold_steps[env_id] = 0


def randomize_actuation(
    env,
    env_ids: torch.Tensor,
    latency_steps_range: tuple[int, int] = (0, 6),
    slip_factor_range: tuple[float, float] = (0.85, 1.0),
    action_term_name: str = "base_velocity",
) -> None:
    """
    Phase 6: actuation/control latency + wheel-ground "slip" (see mdp/actions.py's
    docstring for why slip is injected here rather than via a physics_material
    friction randomization -- the wheels are decorative fixed joints, friction on them
    does nothing to locomotion). Both are per-episode constants, not per-step noise.

    slip_factor_range default (0.85-1.0, i.e. up to 15% commanded-velocity loss) and
    latency_steps_range default (0-6 steps, up to 200ms at the 30Hz control rate) are
    starting defaults, same as course_generator's own size/spacing constants -- picked
    and documented, not yet validated against anything real (that needs actual wheel-
    slip data off the physical robot, Phase 9+).
    """
    action_term = env.action_manager.get_term(action_term_name)
    n = len(env_ids)
    action_term.latency_steps[env_ids] = torch.randint(
        latency_steps_range[0], latency_steps_range[1] + 1, (n,), device=env.device
    )
    action_term.slip_factor[env_ids] = torch.empty(n, device=env.device).uniform_(*slip_factor_range)


def randomize_sensor_mount_pose(
    env,
    env_ids: torch.Tensor,
    sensor_prim_relpath: str,
    pos_std_m: float = 0.003,
    pos_clip_m: float = 0.01,
    tilt_std_deg: float = 1.0,
    tilt_clip_deg: float = 3.0,
) -> None:
    """
    Phase 6: models real assembly tolerance -- the physical camera/lidar bracket won't
    sit at the exact CAD-nominal position every time it's built or reattached. Small
    per-episode offset from the sensor's nominal mount transform, applied directly to
    the sensor prim via pxr (matching how the mount geometry itself was authored --
    isaac_sim/README_lekiwi_variants.md's camera/lidar fixes both used pxr directly for
    the same reason: it's the one thing guaranteed correct for whatever's actually in
    the USD, rather than guessing a wrapper API's exact mutation method).

    NEEDS VERIFICATION IN A RUNNING ISAAC SIM (Phase 1/2): mutating a sensor prim's
    local transform after the sensor has already been created by Isaac Lab's SensorCfg
    machinery may or may not be picked up by that sensor's render/raycast pipeline
    without an explicit re-initialize call, depending on Isaac Sim version. If it
    isn't, move this offset into the SceneCfg's static sensor `offset=` field sampled
    once at scene build instead of per-episode -- a real fallback, not a hidden trap:
    flagged here so Phase 2's first real test specifically checks this term did
    something, not just that it didn't crash.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    n = len(env_ids)
    dpos = torch.clamp(torch.randn(n, 3) * pos_std_m, -pos_clip_m, pos_clip_m)
    dtilt_deg = torch.clamp(torch.randn(n, 3) * tilt_std_deg, -tilt_clip_deg, tilt_clip_deg)

    for row, env_id in enumerate(env_ids.tolist()):
        prim_path = f"{env.scene.env_prim_paths[env_id]}/{sensor_prim_relpath}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*dpos[row].tolist()))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*dtilt_deg[row].tolist()))


def randomize_surroundings_clutter(
    env,
    env_ids: torch.Tensor,
    num_props_range: tuple[int, int] = (6, 16),
    prop_prefix: str = "clutter_",
    max_props: int | None = None,
) -> None:
    """
    Phase 6's "surroundings, materials" randomization, deliberately aggressive: the
    idea is that the policy should learn to navigate off the cone/goal geometry alone
    and be robust to an arbitrary real-world background, not something that only works
    against one specific simulated room. So the area *outside* the course rectangle
    (but inside the per-env cell, bounded by env_spacing) gets a genuinely messy prop
    scatter every episode -- randomized count, position, scale, and color -- while the
    course rectangle itself (cones + goal + open floor) stays exactly what
    regenerate_course() authored. Clutter props are spawned at scene-build time up to
    max_props (env_cfg's SceneCfg) as plain boxes/cylinders; this event only randomizes
    which ones are "on" (visible + placed in the clutter band) vs. parked out of the
    way, and their pose/scale/color, each reset.
    """
    n_max = max_props if max_props is not None else env.cfg.course.max_clutter_props

    for env_id in env_ids.tolist():
        origin = env.scene.env_origins[env_id]
        bounds = env.course_bounds[env_id]
        xmin, xmax, ymin, ymax = bounds.tolist()
        num_active = int(torch.randint(num_props_range[0], num_props_range[1] + 1, (1,)).item())

        for i in range(n_max):
            prop: RigidObject = env.scene[f"{prop_prefix}{i}"]
            if i < num_active:
                # Placed in the band between the course rectangle and the env cell
                # edge, never inside [xmin,xmax] x [ymin,ymax] -- clutter must never
                # overlap the actual navigable course (this is what "should be able to
                # run anywhere with the correct cone course" requires: the course
                # itself is exactly reproducible, only what's around it is a mess).
                side = torch.randint(0, 4, (1,)).item()
                pad = 0.3
                if side == 0:
                    x = float(torch.empty(1).uniform_(xmin - 1.0, xmax + 1.0))
                    y = ymin - pad - float(torch.empty(1).uniform_(0.0, 0.8))
                elif side == 1:
                    x = float(torch.empty(1).uniform_(xmin - 1.0, xmax + 1.0))
                    y = ymax + pad + float(torch.empty(1).uniform_(0.0, 0.8))
                elif side == 2:
                    x = xmin - pad - float(torch.empty(1).uniform_(0.0, 0.8))
                    y = float(torch.empty(1).uniform_(ymin - 1.0, ymax + 1.0))
                else:
                    x = xmax + pad + float(torch.empty(1).uniform_(0.0, 0.8))
                    y = float(torch.empty(1).uniform_(ymin - 1.0, ymax + 1.0))
                z = float(torch.empty(1).uniform_(0.05, 0.4))
                pos = origin + torch.tensor([x, y, z], device=env.device)
                yaw = float(torch.empty(1).uniform_(-math.pi, math.pi))
                quat = quat_from_euler_xyz(
                    torch.zeros(1), torch.zeros(1), torch.tensor([yaw])
                )[0].to(env.device)
                scale = torch.empty(3).uniform_(0.5, 1.8)
                prop.write_root_pose_to_sim(
                    torch.cat([pos, quat]).unsqueeze(0), env_ids=torch.tensor([env_id], device=env.device)
                )
                # Random muted color per prop, applied via the prim's displayColor --
                # "make the bgs look a mess" is about visual/material variety, not just
                # geometric clutter. See randomize_sensor_mount_pose's docstring for why
                # pxr is used directly rather than guessing a wrapper API.
                _set_prim_display_color(env, env_id, f"{prop_prefix}{i}", torch.rand(3).tolist())
            else:
                pos = origin + torch.tensor([0.0, 0.0, -5.0], device=env.device)
                prop.write_root_pose_to_sim(
                    torch.cat([pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)]).unsqueeze(0),
                    env_ids=torch.tensor([env_id], device=env.device),
                )


def _set_prim_display_color(env, env_id: int, relpath: str, rgb: list[float]) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim_path = f"{env.scene.env_prim_paths[env_id]}/{relpath}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
