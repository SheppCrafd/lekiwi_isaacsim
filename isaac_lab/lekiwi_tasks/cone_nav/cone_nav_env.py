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
