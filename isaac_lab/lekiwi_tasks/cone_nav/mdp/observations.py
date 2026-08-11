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

from ._pure_math import (
    apply_dead_pixels,
    apply_exposure_white_balance,
    apply_lens_glare,
    apply_lens_smudge,
    apply_lidar_angular_jitter_variable_std,
    apply_lidar_dropout,
    apply_lidar_freeze,
    apply_lidar_misreads,
    apply_motion_blur,
    apply_pixel_shake,
    blur_weight_from_speed,
    glare_intensity_from_heading,
    shake_std_from_speed,
    speed_metric,
)
from ._robot_state import robot_local_xy


def _base_speed_metric(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), omega_to_linear_m: float = 0.15
) -> torch.Tensor:
    """
    Shared by front_camera_rgb and lidar_ranges: both need "how fast is this env
    currently moving" as a single scalar to drive their live shake/blur/jitter
    magnitude (see _pure_math.speed_metric's docstring for the omega_to_linear_m
    judgment call). Recomputed here rather than reused from base_velocity_2d's
    ObsTerm output because ObsTerm functions are independent, order-unspecified calls
    -- there's no guarantee base_velocity_2d has already run this step when this
    function is called, so this reads the same underlying joint_vel state directly.
    """
    asset: Articulation = env.scene[robot_cfg.name]
    vx = asset.data.joint_vel[:, asset.find_joints("base_x")[0][0]]
    vy = asset.data.joint_vel[:, asset.find_joints("base_y")[0][0]]
    omega = asset.data.joint_vel[:, asset.find_joints("base_theta")[0][0]]
    return speed_metric(vx, vy, omega, omega_to_linear_m)


def _base_heading(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Current base_theta, used by front_camera_rgb's live glare term. Same
    direct-joint-read reasoning as _base_speed_metric -- don't assume base_pose_2d
    has already run this step."""
    asset: Articulation = env.scene[robot_cfg.name]
    return asset.data.joint_pos[:, asset.find_joints("base_theta")[0][0]]


def base_pose_2d(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Proprioceptive (x, y, cos(theta), sin(theta)) from the robot's own odometry.

    This is the sim analogue of what the real firmware already tracks from wheel
    odometry -- it's the robot's own estimate of where it is, not simulator ground
    truth about the goal/cones, so it's fine for the policy to see. sin/cos encoding
    avoids the angle-wraparound discontinuity a raw theta value would create.

    x/y come from mdp/_robot_state.py:robot_local_xy, shared with rewards.py/
    terminations.py (2026-08-10) -- this function's own version of that read was
    already correct (joint_pos, not root_pos_w), so this is a pure dedup, not a
    behavior change; see that module's docstring for why the shared version exists.
    """
    asset: Articulation = env.scene[robot_cfg.name]
    xy = robot_local_xy(env, robot_cfg)
    theta = asset.data.joint_pos[:, asset.find_joints("base_theta")[0][0]]
    return torch.stack([xy[:, 0], xy[:, 1], torch.cos(theta), torch.sin(theta)], dim=-1)


def base_velocity_2d(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Proprioceptive (vx, vy, omega) from the planar joint velocities."""
    asset: Articulation = env.scene[robot_cfg.name]
    vx = asset.data.joint_vel[:, asset.find_joints("base_x")[0][0]]
    vy = asset.data.joint_vel[:, asset.find_joints("base_y")[0][0]]
    omega = asset.data.joint_vel[:, asset.find_joints("base_theta")[0][0]]
    return torch.stack([vx, vy, omega], dim=-1)


def front_camera_rgb(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("front_camera"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    shake_still_px: float = 0.15,
    shake_moving_px: float = 3.0,
    shake_speed_at_moving_px: float = 0.4,
    blur_still_weight: float = 1.0,
    blur_moving_weight: float = 0.35,
    blur_speed_at_moving_weight: float = 0.4,
) -> torch.Tensor:
    """
    Raw RGB frame, (N, H, W, 3) uint8-range float, with two live per-step effects
    layered on top of the raw render. Both close a gap plan.md's open-decisions list
    explicitly flagged as unbuilt: "motion blur ... specifically ... would need
    dedicated image-augmentation code, not done."

    1. MECHANICAL SHAKE (apply_pixel_shake): a small per-step circular pixel shift,
       std given by shake_std_from_speed -- "very slight when still, intensifies
       greatly when moving," a real function of the robot's current body speed
       (_base_speed_metric). Distinct from mdp/events.py:randomize_sensor_mount_pose,
       which is a fixed-for-the-whole-episode offset modeling assembly tolerance (the
       bracket sits 2mm off nominal for its whole physical life) -- this is the same
       bracket vibrating in real time while the robot is actually rolling, a
       different physical effect with different statistics, so it's layered on top
       of that per-episode offset, not a replacement for it.

    2. MOTION BLUR (apply_motion_blur): an exponential temporal blend against the
       previous step's blurred frame, weight given by blur_weight_from_speed --
       sharp when still, progressively smeared as speed increases. Needs state
       across steps (env.prev_camera_frame / env.camera_frame_valid, owned by
       cone_nav_env.py) since a single frame alone carries no motion information --
       unlike shake/dropout/jitter (stateless per-step corruptions), this is why a
       plain per-step function call alone wasn't enough for this one item.
       env.camera_frame_valid is cleared by mdp/events.py:reset_camera_frame_state
       (mode="reset") so an env's first frame after a course reset never blends
       against the PREVIOUS episode's last frame (different course, different robot
       pose) -- when invalid, blend weight is forced to 1.0 (pure sharp frame) for
       that one step, then normal blending resumes.

    Both run BEFORE env_cfg_camera.py's existing UniformNoiseCfg (+/-12 pixel-value
    noise), since that's Isaac Lab's own post-hoc noise wrapper applied to whatever
    this function returns -- shake/blur affect image CONTENT, the existing noise
    affects pixel VALUES, applied in physical order (shake the camera and integrate
    the blur first, then apply sensor read noise on top of what was actually
    captured).

    "Anything the robot might endure" pass (2026-08-10) adds four more, all per-episode
    physical properties of one camera unit (randomized once per episode by
    mdp/events.py:randomize_camera_defects, see that function's and cone_nav_env.py's
    docstrings for why these are episode-constants rather than live per-step effects
    like shake/blur above), applied in physical capture order -- light hits the lens
    (glare), passes through whatever's stuck to the lens (smudge), is captured by the
    sensor's own exposure/white-balance response, and finally any permanently stuck
    photosites override whatever the rest of the pipeline computed for their exact
    pixel, since a dead photosite ignores incoming light entirely:

    3. LENS GLARE: uniform brightness wash-out (apply_lens_glare), intensity LIVE
       per-step (glare_intensity_from_heading -- depends on the CURRENT heading
       relative to this episode's fixed sun_azimuth_rad), harshness fixed per episode
       (glare_max_brightness_add). Bright when facing the light source, absent when
       facing away -- the one item in this group with a live component, since "how
       much glare" genuinely does change every step as the robot turns, even though
       the light source's position doesn't.
    4. LENS SMUDGE: a static darkened patch at a fixed per-episode screen position
       (apply_lens_smudge) -- most episodes have none at all (smudge_prob in
       randomize_camera_defects), matching how most real camera units aren't
       smudged at any given moment.
    5. EXPOSURE / WHITE BALANCE: per-episode global brightness + per-channel color
       gain (apply_exposure_white_balance) -- a real camera's auto-exposure/AWB
       settling to a slightly different operating point each power cycle.
    6. DEAD/HOT PIXELS: a small, usually-zero, fixed-per-episode set of stuck
       photosites (apply_dead_pixels) forced to pure white or pure black regardless
       of everything else in the frame.

    Both this group and the shake/blur group run BEFORE env_cfg_camera.py's existing
    UniformNoiseCfg (+/-12 pixel-value noise) -- that's Isaac Lab's own post-hoc
    wrapper applied to whatever this function returns.

    Still not lens distortion (geometric warp) or compression artifacts -- plan.md's
    own remaining gap after this change, genuinely separate corruptions, not
    implemented here either.
    """
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    frame = sensor.data.output["rgb"][..., :3].float()

    speed = _base_speed_metric(env, robot_cfg)

    shake_std = shake_std_from_speed(speed, shake_still_px, shake_moving_px, shake_speed_at_moving_px)
    shift_x = torch.round(torch.randn_like(speed) * shake_std)
    shift_y = torch.round(torch.randn_like(speed) * shake_std)
    frame = apply_pixel_shake(frame, shift_x, shift_y)

    if env.prev_camera_frame is None:
        env.prev_camera_frame = frame.clone()

    blend_w = blur_weight_from_speed(speed, blur_still_weight, blur_moving_weight, blur_speed_at_moving_weight)
    blend_w = torch.where(env.camera_frame_valid, blend_w, torch.ones_like(blend_w))
    frame = apply_motion_blur(frame, env.prev_camera_frame, blend_w)

    env.prev_camera_frame = frame.detach()
    env.camera_frame_valid[:] = True

    heading = _base_heading(env, robot_cfg)
    glare_intensity = glare_intensity_from_heading(heading, env.sun_azimuth_rad, env.glare_half_width_rad)
    frame = apply_lens_glare(frame, glare_intensity, env.glare_max_brightness_add)

    h, w = frame.shape[1], frame.shape[2]
    smudge_cx = env.smudge_center_frac[:, 0] * (w - 1)
    smudge_cy = env.smudge_center_frac[:, 1] * (h - 1)
    smudge_radius_px = env.smudge_radius_frac * min(h, w)
    frame = apply_lens_smudge(frame, smudge_cx, smudge_cy, smudge_radius_px, env.smudge_opacity)

    frame = apply_exposure_white_balance(frame, env.exposure_gain, env.wb_gain)

    px_x = torch.clamp((env.dead_pixel_uv_frac[..., 0] * (w - 1)).long(), 0, w - 1)
    px_y = torch.clamp((env.dead_pixel_uv_frac[..., 1] * (h - 1)).long(), 0, h - 1)
    frame = apply_dead_pixels(frame, px_x, px_y, env.dead_pixel_value, env.dead_pixel_active)

    return frame


def lidar_ranges(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("rplidar"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dropout_prob: float = 0.02,
    angular_jitter_still_deg: float = 0.15,
    angular_jitter_moving_deg: float = 1.5,
    jitter_speed_at_moving_deg: float = 0.4,
    misread_prob: float = 0.01,
    misread_min_ghost_frac: float = 0.1,
    misread_max_ghost_frac: float = 0.6,
    freeze_prob: float = 0.01,
    angular_res_deg: float = 1.0,
) -> torch.Tensor:
    """
    2D range scan, (N, num_rays). Distance from sensor origin to each ray hit,
    clamped to the RPLIDAR A1M8's real 0.15-12m valid range (out-of-range hits report
    the sensor's max raycast distance, not real returns, so they're clamped rather than
    trusted past the datasheet's own spec -- see isaac_sim/attach_sensors.py for the
    same real hardware numbers used here).

    Also implements the Phase 6 sensor-noise items Isaac Lab's stock GaussianNoiseCfg
    (env_cfg_lidar.py's range-noise term) can't express, since that's a simple additive
    op on the whole tensor, not built for zeroing a subset of rays or perturbing the
    angular sample grid itself:
      - dropout_prob: each ray independently has this probability of reporting "no
        return" -- set to max range (12.0m), matching how a real 2D lidar actually
        reports a missed/invalid return, NOT zero (zero would look like "obstacle
        touching the sensor," which is physically backwards and would badly mislead
        the policy). 2% per ray is a starting default, not a datasheet spec -- the
        A1M8's own datasheet doesn't publish a dropout rate. Deliberately left as a
        flat rate, not speed-scaled -- dropped returns are a signal-quality effect,
        not the "shakiness" the angular-jitter term below now models live.
      - angular jitter: modeled as a small random circular shift of the whole ray
        array (not independent per-ray angle noise, which isn't recoverable from a
        post-hoc range vector without the raw per-ray geometry) -- approximates the
        sensor's whole angular reference frame being very slightly offset. NOW LIVE
        AND SPEED-SCALED (shake_std_from_speed + apply_lidar_angular_jitter_variable_std),
        replacing the old flat angular_jitter_std_deg=0.5 constant: "very slight when
        still, intensifies greatly when moving," the same physical-vibration story as
        front_camera_rgb's pixel-shake term, applied to lidar's own whole-scan-shift
        corruption model instead of a pixel shift. Still a documented approximation,
        not a physically exact per-ray angular noise model. angular_res_deg (below)
        must match env_cfg_lidar.py's LidarPatternCfg(horizontal_res=...) exactly --
        this used to be silently inferred as 360/num_rays inside _pure_math.py, which
        broke the moment the scan stopped spanning a full 360deg circle (2026-08-11,
        when the lidar's FOV was narrowed to match the camera's); now passed explicitly
        instead of guessed from array width.

    "Anything the robot might endure" pass (2026-08-10) adds two more, both LIVE
    per-step effects (unlike front_camera_rgb's episode-constant glare/smudge/
    exposure/dead-pixels group -- these two are closer in spirit to dropout/jitter
    above, real-time sensor glitches rather than fixed physical properties):
      - misread_prob: each ray independently has this probability of reporting a
        spurious SHORT reading -- a real multipath-reflection failure mode off
        glossy/reflective surfaces, and a deliberately different (more dangerous)
        failure direction from dropout: dropout reports NO obstacle, a misread
        reports a WRONG, falsely-CLOSE one, which is what actually risks a policy
        overreacting to a phantom obstacle rather than missing a real one.
      - freeze_prob: each step, each env independently has this probability of the
        WHOLE scan freezing -- reporting an exact repeat of the previous step's scan
        instead of a fresh reading, modeling a real comms/firmware hiccup. Needs
        state across steps (env.prev_lidar_scan / env.lidar_scan_valid, owned by
        cone_nav_env.py, same pattern as front_camera_rgb's motion-blur buffer) --
        cleared by mdp/events.py:reset_lidar_scan_state so an env's first scan after
        reset never freezes onto the PREVIOUS episode's last one.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    origins = sensor.data.pos_w.unsqueeze(1)
    hits = sensor.data.ray_hits_w
    ranges = torch.linalg.norm(hits - origins, dim=-1)
    ranges = torch.clamp(ranges, 0.15, 12.0)
    ranges = apply_lidar_dropout(ranges, dropout_prob)
    ranges = apply_lidar_misreads(ranges, misread_prob, misread_min_ghost_frac, misread_max_ghost_frac)

    speed = _base_speed_metric(env, robot_cfg)
    jitter_std_deg = shake_std_from_speed(speed, angular_jitter_still_deg, angular_jitter_moving_deg, jitter_speed_at_moving_deg)
    ranges = apply_lidar_angular_jitter_variable_std(ranges, jitter_std_deg, deg_per_ray=angular_res_deg)

    if env.prev_lidar_scan is None:
        env.prev_lidar_scan = ranges.clone()
    freeze_roll = torch.rand(ranges.shape[0], device=ranges.device) < freeze_prob
    freeze_mask = freeze_roll & env.lidar_scan_valid
    ranges = apply_lidar_freeze(ranges, env.prev_lidar_scan, freeze_mask)

    env.prev_lidar_scan = ranges.detach()
    env.lidar_scan_valid[:] = True

    return ranges
