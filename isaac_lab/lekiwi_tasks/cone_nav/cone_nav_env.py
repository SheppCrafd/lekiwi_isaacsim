"""
Custom ManagerBasedRLEnv subclass adding per-env procedural cone-course state.

Isaac Lab's base ManagerBasedRLEnv has no concept of "this env's goal is currently
here" or "this env's cones are currently there" -- that's this task's own state, not
a generic RL-env concept, so it's owned here rather than forced into a manager that
doesn't fit it. mdp/events.py's `regenerate_course` EventTerm is what actually calls
course_generator.generate_course() and fills these tensors in (at env creation and at
every episode reset -- each reset draws a new procedural seed, which is how a training
run gets exposed to plan.md's ~1,000,000-environment seed space over 50 hours without
literally instantiating a million env slots).

Privileged vs. policy-observable split (plan.md Phase 3, load-bearing sim-to-real
decision): goal_pos_w and everything derived from it are populated here and are
readable by reward/termination functions, but mdp/observations.py's policy
observation group deliberately never reads them -- only sensor data (camera/lidar) and
proprioceptive base state go into the policy's own observations. Keep it that way; the
real LeKiwi has no ground-truth-position sensor to fall back on at deployment.
"""

from __future__ import annotations

import copy

import torch
from isaaclab.envs import ManagerBasedRLEnv

# Fixed slot count for per-episode dead/hot sensor pixels (mdp/observations.py:
# front_camera_rgb, mdp/events.py:randomize_camera_defects) -- same padded-slots
# pattern as MAX_CONES/max_clutter_props (env_cfg_base.py): a real physical sensor
# has some small, fixed-per-unit number of stuck photosites, not a per-episode-varying
# count, but tensors need a static shape, so most slots are simply inactive most of
# the time (see randomize_camera_defects' own per-slot activation probability).
MAX_DEAD_PIXELS = 5


class LekiwiConeNavEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        device = self.device
        n = self.num_envs
        max_cones = self.cfg.course.max_cones

        # Mutable working copy of the course generator cfg -- mdp/curriculum.py's
        # anneal_course_difficulty term adjusts fields on THIS copy in place as
        # training progresses (env.common_step_counter), not on self.cfg.course.generator
        # itself, so the original cfg (the eventual full-difficulty target) stays intact
        # for reference/logging. mdp/events.py:regenerate_course reads this copy, not
        # a frozen default argument, so curriculum changes actually take effect.
        self.course_generator_cfg = copy.deepcopy(self.cfg.course.generator)

        # Privileged goal state -- reward/termination functions only, never exposed to
        # the policy's observation group.
        self.goal_pos_w = torch.zeros(n, 3, device=device)
        self.goal_radius = torch.zeros(n, device=device)
        # (xmin, xmax, ymin, ymax) of the generated course, in each env's own local
        # frame (i.e. relative to that env's origin) -- terminations.out_of_bounds
        # subtracts env_origins before comparing against this, so it lines up.
        self.course_bounds = torch.zeros(n, 4, device=device)

        # Cone state -- also reward/termination-only in the current design (collision
        # is checked via contact sensors / distance, not "the policy sees cone
        # coordinates"). Real perception of cones comes entirely through the
        # camera/lidar sensor observations, matching the real robot.
        self.cone_pos_w = torch.zeros(n, max_cones, 3, device=device)
        self.cone_radius = torch.zeros(n, max_cones, device=device)
        self.cone_active = torch.zeros(n, max_cones, dtype=torch.bool, device=device)

        # Potential-based reward shaping state (Phase 5: reward = dist_last - dist_this,
        # not raw -distance, to avoid the oscillate-near-goal reward-hacking failure
        # mode plan.md flags explicitly).
        self.prev_dist_to_goal = torch.zeros(n, device=device)

        # Success requires full-footprint-in-goal AND a brief hold, not instantaneous
        # containment (Phase 5: "a policy could learn to blast through the area and
        # still register success"). Counts consecutive steps inside the goal region.
        self.success_hold_steps = torch.zeros(n, dtype=torch.long, device=device)

        # Seed actually used for each env's current episode -- kept around so
        # scripts/play.py can request the held-out eval range (course_generator's
        # TRAIN_SEED_UPPER split) instead of the training range.
        self.episode_seed = torch.zeros(n, dtype=torch.long, device=device)

        # Motion-blur temporal state for mdp/observations.py:front_camera_rgb.
        # prev_camera_frame starts as None rather than a pre-sized zero tensor -- this
        # class doesn't know the camera's (H, W) (only env_cfg_camera.py's CameraCfg
        # does), so it's allocated lazily on first use from the real sensor output.
        # camera_frame_valid is a small per-env bool buffer marking whether
        # prev_camera_frame currently holds a frame from THIS episode -- cleared by
        # mdp/events.py:reset_camera_frame_state on reset so a fresh episode's first
        # frame never blends against the previous episode's last one. Both fields are
        # allocated unconditionally (cheap even unused) rather than guarded behind a
        # camera-variant check -- lidar_ranges never touches them.
        self.prev_camera_frame: torch.Tensor | None = None
        self.camera_frame_valid = torch.zeros(n, dtype=torch.bool, device=device)

        # Per-episode camera defect state ("anything the robot might endure" pass,
        # 2026-08-10) -- randomized once per episode by
        # mdp/events.py:randomize_camera_defects, applied every step by
        # mdp/observations.py:front_camera_rgb. Real physical properties of one
        # camera unit (a light source's position, whether this lens currently has a
        # smudge, this unit's exposure/white-balance operating point, which
        # photosites are stuck) that don't change mid-episode -- same "per-episode
        # constant" category as goal_pos_w/cone_pos_w above, just camera-specific
        # rather than course-specific. Allocated unconditionally (cheap, ~n*40 floats
        # total) even for the lidar variant, which never reads them.
        self.sun_azimuth_rad = torch.zeros(n, device=device)
        self.glare_half_width_rad = torch.zeros(n, device=device)
        self.glare_max_brightness_add = torch.zeros(n, device=device)
        # Smudge center/radius stored as FRACTIONS of frame width/height, not pixels
        # -- this class doesn't know the camera's (H, W) any more than
        # prev_camera_frame's docstring above does; front_camera_rgb converts to
        # pixels at call time using the sensor's real resolution.
        self.smudge_center_frac = torch.zeros(n, 2, device=device)
        self.smudge_radius_frac = torch.zeros(n, device=device)
        self.smudge_opacity = torch.zeros(n, device=device)
        self.exposure_gain = torch.ones(n, device=device)
        self.wb_gain = torch.ones(n, 3, device=device)
        self.dead_pixel_uv_frac = torch.zeros(n, MAX_DEAD_PIXELS, 2, device=device)
        self.dead_pixel_value = torch.zeros(n, MAX_DEAD_PIXELS, 3, device=device)
        self.dead_pixel_active = torch.zeros(n, MAX_DEAD_PIXELS, dtype=torch.bool, device=device)

        # Freeze-glitch temporal state for mdp/observations.py:lidar_ranges -- same
        # lazy-allocation-plus-valid-flag pattern as prev_camera_frame/
        # camera_frame_valid above, mirrored for the lidar variant's own live-glitch
        # effect (a stale repeated scan instead of a fresh one, mdp/_pure_math.py:
        # apply_lidar_freeze). Cleared by mdp/events.py:reset_lidar_scan_state.
        self.prev_lidar_scan: torch.Tensor | None = None
        self.lidar_scan_valid = torch.zeros(n, dtype=torch.bool, device=device)
