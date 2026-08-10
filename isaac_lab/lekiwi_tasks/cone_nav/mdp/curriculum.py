"""
Curriculum / Automatic Domain Randomization ramp (plan.md Phase 6's last unchecked
item: "Consider annealing in obstacle density/domain-randomization range progressively
... rather than starting training against the full 1,000,000-environment variety
immediately -- full randomization from scratch is a much harder learning problem than
a staged ramp-up").

Confirmed real API (source research, not guessed): Isaac Lab's `CurriculumTermCfg`/
`CurriculumManager` call `func(env, env_ids, **params)` once per env at reset and expect
a return value usable as curriculum state for logging.

What's annealed: course *difficulty*, not the domain-randomization noise terms
(sensor/latency/slip -- those stay at their full configured range from step 0, plan.md
didn't ask to anneal those specifically, just cone/obstacle density). Two knobs, both on
`env.course_generator_cfg` (cone_nav_env.py's mutable per-env working copy of the
generator cfg, read fresh by mdp/events.py:regenerate_course on every reset):
  - `min_cone_spacing_m`: starts loose (more room to maneuver), tightens to the
    original target as training progresses.
  - `num_cones_range`: starts narrow/low (fewer obstacles), widens to the full
    configured range.
Cone SIZE/SHAPE ranges and the course footprint (width/length) ranges are NOT annealed
here -- plan.md's own curriculum item specifically calls out "obstacle density" and (in
the old design) "cone-position-offset range", both of which map onto spacing/count, not
size or footprint; left alone as a reasonable scope boundary, not an oversight.
"""

from __future__ import annotations

import torch

from ._pure_math import linear_anneal


def anneal_course_difficulty(
    env,
    env_ids: torch.Tensor,
    num_steps_to_anneal: int = 500_000,
    start_min_cone_spacing_m: float = 1.2,
    end_min_cone_spacing_m: float = 0.5,  # matches CourseGeneratorCfg's own default target
    start_num_cones_range: tuple[int, int] = (6, 8),
    end_num_cones_range: tuple[int, int] = (6, 14),  # matches CourseGeneratorCfg's own default target
) -> torch.Tensor:
    """
    Linear ramp keyed off env.common_step_counter (a real, existing ManagerBasedRLEnv
    counter -- already relied on elsewhere in this codebase, e.g. scripts/play.py's
    held-out-seed rotation). `num_steps_to_anneal` default (500k steps) is a starting
    guess sized against Phase 7's ~24 steps-per-env-per-update x thousands of updates
    over a 50-hour run -- not tuned against a real training curve, adjust once Phase 7
    shows how fast the success rate actually climbs.

    Mutates env.course_generator_cfg in place (shared across all envs -- this is a
    *training-progress* curriculum, not a per-env one, so every env anneals together).
    Called once per env_id at reset, but it's idempotent and cheap, so recomputing the
    same global progress value redundantly per env_id in the batch is fine, not worth
    optimizing.
    """
    progress = min(1.0, env.common_step_counter / num_steps_to_anneal)

    cfg = env.course_generator_cfg
    cfg.min_cone_spacing_m = linear_anneal(start_min_cone_spacing_m, end_min_cone_spacing_m, progress)
    lo = round(linear_anneal(start_num_cones_range[0], end_num_cones_range[0], progress))
    hi = round(linear_anneal(start_num_cones_range[1], end_num_cones_range[1], progress))
    cfg.num_cones_range = (lo, max(lo, hi))

    return torch.tensor(progress)
