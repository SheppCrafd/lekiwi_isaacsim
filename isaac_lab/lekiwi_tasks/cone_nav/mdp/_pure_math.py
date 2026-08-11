"""
Pure-tensor math (torch only, zero isaaclab/omni dependency) factored out of
actions.py, rewards.py, and terminations.py specifically so it can be unit-tested
without a real Isaac Sim install -- see isaac_lab/tests/test_mdp_math.py, which
imports this module directly and actually runs it. The isaaclab-dependent files import
these functions rather than reimplementing the formulas inline, so there's one source
of truth for each formula, not a copy that could silently drift out of sync.
"""

from __future__ import annotations

import math

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


def speed_metric(vx: torch.Tensor, vy: torch.Tensor, omega: torch.Tensor, omega_to_linear_m: float = 0.15) -> torch.Tensor:
    """
    Single scalar "how hard is this env currently shaking" signal per env, combining
    body-frame linear and rotational speed into one number so shake/blur/jitter
    magnitude (below) has a single knob to scale against instead of two separate
    speed axes. omega_to_linear_m converts rad/s into an equivalent linear-speed
    contribution -- roughly the robot's own footprint radius, so a fast in-place spin
    shakes the mount about as much as driving at that speed does. A physically
    reasonable judgment call, not measured off real hardware (nothing to measure yet,
    see plan.md Phase 9) -- same category of picked-not-validated constant as
    mdp/events.py's slip_factor_range/latency_steps_range.
    """
    return torch.hypot(vx, vy) + omega_to_linear_m * torch.abs(omega)


def shake_std_from_speed(speed: torch.Tensor, still_std: float, moving_std: float, speed_at_moving_std: float) -> torch.Tensor:
    """
    Maps a speed metric to a jitter magnitude: still_std at speed=0, ramping linearly
    up to moving_std by speed_at_moving_std (the speed_metric value, not necessarily
    m/s once omega is folded in), clamped flat beyond that. This is what makes
    "very slight when still, intensifies greatly when moving" a real continuous
    function of current speed rather than the single fixed constant Phase 6
    originally shipped with (e.g. lidar_ranges' old flat angular_jitter_std_deg=0.5,
    applied identically whether the robot was stopped or moving at full speed).
    Shared by front_camera_rgb's pixel-shake std and lidar_ranges' angular-jitter std
    -- same shape of problem (map speed -> corruption magnitude), different units of
    magnitude at the call site.
    """
    frac = torch.clamp(speed / speed_at_moving_std, 0.0, 1.0)
    return still_std + frac * (moving_std - still_std)


def blur_weight_from_speed(speed: torch.Tensor, still_weight: float, moving_weight: float, speed_at_moving_weight: float) -> torch.Tensor:
    """
    Temporal-blend weight for apply_motion_blur: the fraction of the CURRENT sharp
    frame that survives into the blended observation. still_weight ~1.0 (no blur at
    rest -- a stationary camera has nothing to smear), ramping down toward
    moving_weight (heavier blend of the previous frame = more visible smear) as speed
    increases, same linear-then-clamp shape as shake_std_from_speed but decreasing
    instead of increasing.
    """
    frac = torch.clamp(speed / speed_at_moving_weight, 0.0, 1.0)
    return still_weight + frac * (moving_weight - still_weight)


def apply_motion_blur(frame: torch.Tensor, prev_blurred: torch.Tensor, blend_weight: torch.Tensor) -> torch.Tensor:
    """
    Exponential temporal blend: output = w*frame + (1-w)*prev_blurred, per env (w is
    (N,), broadcast over the image's H/W/C dims). This is the standard game-engine
    accumulation-buffer approximation of motion blur -- mixing the current sharp
    frame with a running blend of recent ones -- rather than simulating shutter
    integration against real per-pixel optical-flow motion vectors. Deliberately the
    cheap approximation: it's one more elementwise op per step for thousands of
    parallel envs, where a true per-pixel motion-vector blur would not be.

    Caller owns storing the returned tensor as next step's prev_blurred (see
    cone_nav_env.py's prev_camera_frame buffer and mdp/observations.py:
    front_camera_rgb, which also handles the post-reset case where prev_blurred
    belongs to a different, already-ended episode).
    """
    w = blend_weight.view(-1, 1, 1, 1)
    return w * frame + (1.0 - w) * prev_blurred


def apply_pixel_shake(frame: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor) -> torch.Tensor:
    """
    Per-env whole-image circular roll by (shift_x, shift_y) pixels -- represents the
    camera bracket physically vibrating in real time while the robot is rolling
    (image CONTENT moves), a different effect from env_cfg_camera.py's existing
    UniformNoiseCfg (elementwise pixel-VALUE read noise) and from
    mdp/events.py:randomize_sensor_mount_pose (a fixed-for-the-whole-episode offset
    modeling assembly tolerance, not live motion) -- all three are real, independent,
    layered effects, not duplicates of each other.

    Vectorized across the batch via gather-based fancy indexing rather than the
    per-env python loop + torch.roll that apply_lidar_angular_jitter below uses --
    deliberate difference, not an inconsistency: image tensors here (e.g.
    2500 x 480 x 640 x 3) are roughly 1000x larger per-env than a 360-value lidar
    scan, and this runs every physics step for every parallel env, so a python-level
    per-env loop here would be a real bottleneck the lidar version's much smaller
    tensors don't suffer from.

    Wrap-around at the image edge (not border-replicate/black-fill) -- same
    "negligible at the few-pixel shifts this is actually called with" reasoning as
    apply_lidar_angular_jitter's own circular shift.
    """
    n, h, w = frame.shape[0], frame.shape[1], frame.shape[2]
    device = frame.device
    row_idx = ((torch.arange(h, device=device).view(1, h, 1) - shift_y.view(n, 1, 1)) % h).expand(n, h, w).long()
    col_idx = ((torch.arange(w, device=device).view(1, 1, w) - shift_x.view(n, 1, 1)) % w).expand(n, h, w).long()
    batch_idx = torch.arange(n, device=device).view(n, 1, 1).expand(n, h, w)
    return frame[batch_idx, row_idx, col_idx]


def apply_lidar_angular_jitter_variable_std(ranges: torch.Tensor, jitter_std_deg: torch.Tensor) -> torch.Tensor:
    """
    Same whole-scan circular-shift model as apply_lidar_angular_jitter below, but
    with a PER-ENV jitter std (one value per row of `ranges`, shape (N,)) instead of
    one shared float -- needed for the live speed-scaled wiggle
    (mdp/observations.py:lidar_ranges), where each env's current jitter magnitude
    depends on that env's own current speed (shake_std_from_speed), not a single
    training-wide constant.

    Kept as a separate function rather than changing apply_lidar_angular_jitter's own
    signature to accept `float | torch.Tensor` -- that function's existing
    scalar-float contract is exactly what tests/test_mdp_math.py already tests and
    what it's still called with in its own right; adding a second, narrower function
    here avoids any risk of changing behavior under an already-tested, already-relied
    on signature.
    """
    num_rays = ranges.shape[-1]
    deg_per_ray = 360.0 / num_rays
    shift_rays = torch.round(torch.randn(ranges.shape[0], device=ranges.device) * jitter_std_deg / deg_per_ray).long()
    return torch.stack([torch.roll(ranges[i], shifts=int(shift_rays[i].item())) for i in range(ranges.shape[0])])


# --- "anything the robot might endure in the real world" pass, camera-side ------------
# Each of these models ONE specific real failure mode, deliberately not a grab-bag
# "add some more noise" function -- same discipline as the shake/blur pass above.
# Per-episode parameters (sun azimuth, smudge geometry, exposure/WB gain, dead-pixel
# positions) live on env buffers owned by cone_nav_env.py, randomized once per episode
# by mdp/events.py:randomize_camera_defects (mode="reset") -- these are real physical
# properties of one camera unit that don't change mid-episode (a lens smudge doesn't
# move while the robot drives; a light source's position doesn't either), unlike the
# shake/blur pass's LIVE per-step effects. Glare's INTENSITY is still live per-step
# (see glare_intensity_from_heading) since that depends on which way the robot is
# currently facing relative to the fixed light source, not a constant.


def glare_intensity_from_heading(heading_rad: torch.Tensor, sun_azimuth_rad: torch.Tensor, half_width_rad: float) -> torch.Tensor:
    """
    How directly the camera currently faces a fixed bright light source (this
    episode's sun_azimuth_rad, randomized once at reset) -- 1.0 when heading and sun
    azimuth coincide exactly, ramping down to 0.0 by half_width_rad of angular
    separation, matching real lens flare's behavior of appearing/vanishing as a
    camera pans across a light source rather than being an on/off event.
    """
    diff = torch.remainder(heading_rad - sun_azimuth_rad + math.pi, 2 * math.pi) - math.pi
    return torch.clamp(1.0 - torch.abs(diff) / half_width_rad, 0.0, 1.0)


def apply_lens_glare(frame: torch.Tensor, intensity: torch.Tensor, max_brightness_add: torch.Tensor) -> torch.Tensor:
    """
    Uniform whole-frame brightness wash-out proportional to intensity (see
    glare_intensity_from_heading) and this episode's own max_brightness_add (some
    episodes' light source is harsher than others, randomized per episode) --
    approximates a lens flare's dominant visible effect (a bright veiling glow
    washing out contrast) without attempting the more complex radial-streak/ghosting
    geometry a real flare also produces. frame is (N, H, W, 3) in 0-255 float range.
    """
    add = (intensity * max_brightness_add).view(-1, 1, 1, 1)
    return torch.clamp(frame + add, 0.0, 255.0)


def apply_lens_smudge(frame: torch.Tensor, center_x_px: torch.Tensor, center_y_px: torch.Tensor, radius_px: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    """
    Per-env static darkened patch with a soft (smoothstep-ish) edge over the outer
    30% of its radius, not a hard-edged disc -- reads as an optical defect (grease/
    dust on the lens) rather than a UI overlay. opacity=0 for envs with no smudge
    this episode (most real cameras aren't smudged at any given moment -- see
    mdp/events.py:randomize_camera_defects for the per-episode "is there a smudge at
    all" draw). Darkens and desaturates toward 40% of original brightness at full
    opacity+coverage rather than fully occluding -- a smudge degrades a view, it
    doesn't blank it.
    """
    n, h, w = frame.shape[0], frame.shape[1], frame.shape[2]
    device = frame.device
    ys = torch.arange(h, device=device).view(1, h, 1).float()
    xs = torch.arange(w, device=device).view(1, 1, w).float()
    dist = torch.sqrt((xs - center_x_px.view(n, 1, 1)) ** 2 + (ys - center_y_px.view(n, 1, 1)) ** 2)
    r = radius_px.view(n, 1, 1)
    edge_band = torch.clamp(r * 0.3, min=1.0)
    mask = torch.clamp((r - dist) / edge_band, 0.0, 1.0)
    strength = (mask * opacity.view(n, 1, 1)).unsqueeze(-1)
    darkened = frame * 0.4
    return frame * (1.0 - strength) + darkened * strength


def apply_exposure_white_balance(frame: torch.Tensor, exposure_gain: torch.Tensor, wb_gain: torch.Tensor) -> torch.Tensor:
    """
    Per-episode global brightness (exposure_gain, (N,)) and per-channel color
    (wb_gain, (N,3)) multiplicative gain -- models a real camera's auto-exposure/
    auto-white-balance settling to a slightly different operating point each power
    cycle, not something that drifts mid-episode at this sensor's update rate.
    """
    gain = exposure_gain.view(-1, 1, 1, 1) * wb_gain.view(-1, 1, 1, 3)
    return torch.clamp(frame * gain, 0.0, 255.0)


def apply_dead_pixels(frame: torch.Tensor, px_x: torch.Tensor, px_y: torch.Tensor, px_value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """
    Stuck/dead sensor pixels: for each env, up to px_x.shape[1] fixed pixel
    coordinates (px_x/px_y, (N, K) long) are forced to a fixed stuck color
    (px_value, (N, K, 3)) wherever `active` (N, K) bool is True -- a real, common
    cheap-camera-sensor defect (a handful of photosites that always read pure white
    "hot" or pure black "dead" regardless of the actual scene), randomized once per
    episode (mdp/events.py:randomize_camera_defects) since real dead pixels are a
    fixed property of one physical sensor, not something that changes moment to
    moment. Most envs/slots have active=False most of the time (see that event's
    per-slot activation probability) -- absence of dead pixels is the common case,
    matching how most real camera units don't have any.
    """
    out = frame.clone()
    k = px_x.shape[1]
    for i in range(k):
        rows = active[:, i]
        if not rows.any():
            continue
        idx = rows.nonzero(as_tuple=True)[0]
        out[idx, px_y[idx, i], px_x[idx, i]] = px_value[idx, i]
    return out


# --- lidar misreads ---------------------------------------------------------------


def apply_lidar_misreads(ranges: torch.Tensor, misread_prob: float, min_ghost_frac: float, max_ghost_frac: float, min_range: float = 0.15) -> torch.Tensor:
    """
    Each ray independently has misread_prob chance of reporting a spurious SHORT
    reading -- a random fraction (min_ghost_frac..max_ghost_frac) of its TRUE range
    -- modeling multipath reflection off glossy/reflective real-world surfaces. A
    deliberately different failure direction from apply_lidar_dropout (which reports
    NO return, i.e. max range): this reports a WRONG, falsely-CLOSE reading, the more
    dangerous case for a collision-avoidance policy since it looks like a real
    obstacle that isn't actually as close as reported, rather than an obviously
    missing data point.
    """
    if misread_prob <= 0.0:
        return ranges
    misread = torch.rand_like(ranges) < misread_prob
    ghost_frac = torch.empty_like(ranges).uniform_(min_ghost_frac, max_ghost_frac)
    ghost_ranges = torch.clamp(ranges * ghost_frac, min=min_range)
    return torch.where(misread, ghost_ranges, ranges)


def apply_lidar_freeze(ranges: torch.Tensor, prev_ranges: torch.Tensor, freeze_mask: torch.Tensor) -> torch.Tensor:
    """
    freeze_mask: (N,) bool, True where this env's scan should freeze -- repeat
    prev_ranges verbatim instead of this step's fresh reading, modeling a real 2D
    lidar occasionally missing an update cycle (comms/firmware hiccup) and reporting
    a stale repeated frame rather than a corrupted one. Caller owns deciding
    freeze_mask (a per-step probability roll) and storing the returned tensor as next
    step's prev_ranges, same state-ownership pattern as apply_motion_blur.
    """
    return torch.where(freeze_mask.view(-1, 1), prev_ranges, ranges)


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
