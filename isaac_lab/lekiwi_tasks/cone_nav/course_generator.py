"""
Procedural cone-avoidance course generator (plan.md Phase 4 + Phase 6).

Pure numpy, no Isaac Sim / omni imports -- deliberately, so it can be unit-tested
outside Isaac Sim (see isaac_lab/tests/test_course_generator.py, added 2026-08-10 --
an earlier version of this docstring pointed here before that file actually existed,
a real dangling reference caught during a bug scan) and so the same generator can be
reused by both the Isaac Lab reset-event code (events.py) and any offline tooling
(course previews, held-out eval seed export for Phase 8).

"~1,000,000 procedurally generated environments" (plan.md) isn't 1M baked assets --
it's this generator's seed space. A course is fully determined by an integer seed, so
"generate environment #k" just means generate_course(seed=k, cfg). Everything about a
course (open-area size, cone count/position/size/shape, goal location, robot spawn
pose) is drawn once from that seed and never changes again -- matching plan.md's
explicit "randomized once, at generation time" requirement, not real-time jitter.

Seed convention (also see scripts/play.py, Phase 8's held-out eval requirement):
  seeds [0, TRAIN_SEED_UPPER)              -> training pool
  seeds [TRAIN_SEED_UPPER, TOTAL_SEEDS)     -> held-out eval pool, never seen in training
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOTAL_SEEDS = 1_000_000
TRAIN_SEED_UPPER = 900_000  # seeds >= this are held out for Phase 8 eval, never sampled during training

MIN_OPEN_AREA_SQM = 9.2903  # 100 sq ft, per plan.md's explicit area guarantee

CONE_SHAPES = ("cone", "pylon", "barrel")
# One shared (radius_m, height_m) range applied to every generated cone regardless of
# its cosmetic shape label -- shape is texture/label variety only now (see
# _generate_candidate), not a separate size bracket per shape. Explicit user-specified
# range: 0.25-2 ft tall (0.0762-0.6096m), 0.125-1 ft diameter i.e. 0.0625-0.5 ft radius
# (0.01905-0.1524m). Wide on purpose -- from a small pylon-sized nub to a tall drum --
# tune down once Phase 8 eval shows whether the policy generalizes across the full range.
CONE_SIZE_RANGE_M = {"radius": (0.01905, 0.1524), "height": (0.0762, 0.6096)}


@dataclass
class CourseGeneratorCfg:
    """Tunable ranges. Defaults satisfy MIN_OPEN_AREA_SQM by construction."""

    width_range_m: tuple[float, float] = (3.5, 6.0)
    length_range_m: tuple[float, float] = (3.5, 7.0)
    num_cones_range: tuple[int, int] = (6, 14)

    # Cones are scattered independently at random across the whole open area (no fixed
    # lane/slalom template) -- min_spacing_m is the only thing keeping them apart, a
    # rejection-sampled minimum center-to-center distance so a fully random scatter
    # can't drop two cones on top of each other. Kept modest relative to num_cones_range
    # so a busy 14-cone draw in a small course still has room to place all of them.
    cone_wall_margin_m: float = 0.3
    min_cone_spacing_m: float = 0.5
    max_placement_attempts_per_cone: int = 200

    goal_radius_m: float = 0.35
    # Robot's own bounding-circle clearance radius. The base plate spans roughly
    # +/-0.10m and the wheels sit at a 117.6mm closest-approach-to-center
    # (isaac_sim/README_lekiwi_variants.md), so 0.16m covers the physical footprint;
    # +0.04m safety margin below (robot_clearance_margin_m) accounts for approach
    # trajectories, not just static overlap.
    robot_radius_m: float = 0.16
    robot_clearance_margin_m: float = 0.04

    reachability_grid_resolution_m: float = 0.05
    max_generation_attempts: int = 64


@dataclass
class ConeSpec:
    x: float
    y: float
    radius: float
    height: float
    shape: str


@dataclass
class CourseLayout:
    seed: int
    width_m: float
    length_m: float  # +X is the nav direction: spawn near x=0, goal near x=length_m
    cones: list[ConeSpec] = field(default_factory=list)
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_radius_m: float = 0.35
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_heading_rad: float = 0.0  # yaw only -- robot spawns flat on the floor (Phase 3 rotation-axis fix)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) where x is the nav/length axis, y is the lateral/width axis."""
        return (0.0, self.length_m, 0.0, self.width_m)

    @property
    def area_sqm(self) -> float:
        return self.width_m * self.length_m


class CourseGenerationError(RuntimeError):
    """Raised when max_generation_attempts is exhausted without a valid layout."""


def generate_course(seed: int, cfg: CourseGeneratorCfg | None = None) -> CourseLayout:
    # cfg defaults to None, not CourseGeneratorCfg() directly -- a mutable dataclass
    # instance as a function default is evaluated ONCE at import time and shared
    # across every call that doesn't pass cfg explicitly (found by static analysis,
    # not by reading the code -- every real call site in this repo already passes
    # cfg explicitly, so this was a latent footgun, not an active bug: nothing here
    # ever mutates cfg in place, but a future change that did would silently corrupt
    # state across unrelated calls). Constructing fresh per call is the standard fix.
    if cfg is None:
        cfg = CourseGeneratorCfg()
    rng = np.random.default_rng(seed)
    for _ in range(cfg.max_generation_attempts):
        layout = _generate_candidate(seed, rng, cfg)
        ok, _reason = validate_layout(layout, cfg)
        if ok:
            return layout
    raise CourseGenerationError(
        f"seed={seed}: no valid course found in {cfg.max_generation_attempts} attempts"
    )


def _generate_candidate(seed: int, rng: np.random.Generator, cfg: CourseGeneratorCfg) -> CourseLayout:
    width = rng.uniform(*cfg.width_range_m)
    length = rng.uniform(*cfg.length_range_m)
    # width_range/length_range are chosen so their minimum product already clears
    # MIN_OPEN_AREA_SQM, but assert it rather than assume the cfg was edited correctly.
    assert width * length >= MIN_OPEN_AREA_SQM, (
        f"width_range_m/length_range_m can produce {width * length:.2f} sqm < "
        f"required {MIN_OPEN_AREA_SQM:.2f} sqm -- widen the ranges"
    )

    num_cones = int(rng.integers(cfg.num_cones_range[0], cfg.num_cones_range[1] + 1))
    margin = 0.5  # keep the goal end clear of the last cone

    # Goal/end-zone placement is the one thing that does NOT get randomized per course
    # -- always the far end, centered laterally. Keeping it fixed regardless of how
    # cones/spawn are re-rolled is a deliberate choice (matches the physical hero
    # course: paint the end zone once, permanently, and only the cones/robot start move
    # between runs).
    goal_x = length - margin * 0.5
    goal_y = width / 2.0

    # Cones scattered independently at random across the whole open area -- no fixed
    # lane/slalom template. Each cone is rejection-sampled against every cone already
    # placed in *this* candidate (min_cone_spacing_m apart, center to center) so a
    # fully random scatter doesn't stack cones on each other; it is NOT checked against
    # spawn/goal here (that's validate_layout's job, on the whole candidate, with
    # generate_course() retrying on a fresh seed-derived candidate if it fails -- same
    # retry loop as every other rejection in this generator, not a special case).
    # If a cone can't find a free spot within max_placement_attempts_per_cone (a dense
    # cone count in a small course), placement stops early with fewer cones rather than
    # overlapping one -- validate_layout still runs on whatever was placed.
    cones: list[ConeSpec] = []
    for _ in range(num_cones):
        shape = CONE_SHAPES[rng.integers(0, len(CONE_SHAPES))]
        radius = rng.uniform(*CONE_SIZE_RANGE_M["radius"])
        height = rng.uniform(*CONE_SIZE_RANGE_M["height"])

        for _attempt in range(cfg.max_placement_attempts_per_cone):
            cx = float(rng.uniform(cfg.cone_wall_margin_m + radius, length - cfg.cone_wall_margin_m - radius))
            cy = float(rng.uniform(cfg.cone_wall_margin_m + radius, width - cfg.cone_wall_margin_m - radius))
            if all(math.hypot(cx - o.x, cy - o.y) >= cfg.min_cone_spacing_m for o in cones):
                cones.append(ConeSpec(x=cx, y=cy, radius=radius, height=height, shape=shape))
                break
        # else: couldn't fit this one, move on -- fewer cones than requested, not a stacked pair

    # Spawn can be anywhere in the open area, not just a fixed near-wall point --
    # matches plan.md's own original task spec ("robot starting (x, y) ... randomized
    # once, at generation time"), which a fixed spawn formula didn't actually satisfy.
    # Sampled uniformly across the full course footprint (validate_layout's spawn-vs-
    # cone / spawn-vs-goal / reachability checks below reject anything that lands too
    # close to an obstacle or cuts off the goal -- generate_course() just resamples a
    # fresh candidate on rejection, same retry loop as everything else).
    wall_margin = cfg.robot_radius_m + cfg.robot_clearance_margin_m
    spawn_x = float(rng.uniform(wall_margin, length - wall_margin))
    spawn_y = float(rng.uniform(wall_margin, width - wall_margin))
    # Full-circle heading -- with spawn position no longer fixed near one wall facing
    # the goal, there's no more "roughly toward the goal" default orientation to bias
    # around. The robot sits flat at spawn either way; only Z/yaw varies (Phase 3's
    # rotation-axis bugfix), never X/Y tilt.
    spawn_heading = float(rng.uniform(-math.pi, math.pi))

    return CourseLayout(
        seed=seed,
        width_m=width,
        length_m=length,
        cones=cones,
        goal_x=goal_x,
        goal_y=goal_y,
        goal_radius_m=cfg.goal_radius_m,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_heading_rad=spawn_heading,
    )


def validate_layout(layout: CourseLayout, cfg: CourseGeneratorCfg) -> tuple[bool, str]:
    """Phase 4's three explicit reset-safety checks. Returns (ok, reason_if_not)."""
    if layout.area_sqm < MIN_OPEN_AREA_SQM:
        return False, f"area {layout.area_sqm:.2f} sqm below {MIN_OPEN_AREA_SQM:.2f} sqm minimum"

    clearance = cfg.robot_radius_m + cfg.robot_clearance_margin_m

    for cone in layout.cones:
        d = math.hypot(layout.spawn_x - cone.x, layout.spawn_y - cone.y)
        if d < clearance + cone.radius:
            return False, f"spawn overlaps cone at ({cone.x:.2f}, {cone.y:.2f})"

    d_goal = math.hypot(layout.spawn_x - layout.goal_x, layout.spawn_y - layout.goal_y)
    if d_goal < layout.goal_radius_m + clearance:
        return False, "spawn overlaps goal area"

    if not _goal_reachable(layout, cfg):
        return False, "goal unreachable from spawn given cone placement"

    return True, ""


def _goal_reachable(layout: CourseLayout, cfg: CourseGeneratorCfg) -> bool:
    """
    Grid flood-fill from spawn to goal, treating each cone (+ robot clearance) as a
    blocked disc. Catches the case Phase 4 calls out explicitly: cone position
    randomization can produce an arrangement that blocks the only path to the goal
    even when the canonical (un-offset) arrangement was solvable.

    Grid axis convention matches ConeSpec/CourseLayout: index `u` runs along the
    nav/length axis (cone.x, spawn_x, goal_x), index `v` along the lateral/width axis
    (cone.y, spawn_y, goal_y). Keeping u/v and x/y strictly paired here matters --
    an earlier version of this function built the grid with width/length swapped
    relative to cone.x/cone.y, which silently checked reachability against the wrong
    extents whenever width_m != length_m. Caught by testing against real generated
    layouts, not by inspection.
    """
    res = cfg.reachability_grid_resolution_m
    clearance = cfg.robot_radius_m + cfg.robot_clearance_margin_m

    nu = max(2, int(math.ceil(layout.length_m / res)))  # nav axis (x)
    nv = max(2, int(math.ceil(layout.width_m / res)))  # lateral axis (y)

    us = (np.arange(nu) + 0.5) * (layout.length_m / nu)
    vs = (np.arange(nv) + 0.5) * (layout.width_m / nv)
    gu, gv = np.meshgrid(us, vs, indexing="ij")  # gu ~ x (nav), gv ~ y (lateral)

    blocked = np.zeros((nu, nv), dtype=bool)
    for cone in layout.cones:
        blocked |= (gu - cone.x) ** 2 + (gv - cone.y) ** 2 < (cone.radius + clearance) ** 2

    def to_cell(x: float, y: float) -> tuple[int, int]:
        iu = int(np.clip(x / (layout.length_m / nu), 0, nu - 1))
        iv = int(np.clip(y / (layout.width_m / nv), 0, nv - 1))
        return iu, iv

    start = to_cell(layout.spawn_x, layout.spawn_y)
    goal_cell = to_cell(layout.goal_x, layout.goal_y)
    if blocked[start] or blocked[goal_cell]:
        return False

    visited = np.zeros_like(blocked)
    stack = [start]
    visited[start] = True
    while stack:
        cu, cv = stack.pop()
        if (cu, cv) == goal_cell:
            return True
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nu_, nv_ = cu + du, cv + dv
            if 0 <= nu_ < nu and 0 <= nv_ < nv and not visited[nu_, nv_] and not blocked[nu_, nv_]:
                visited[nu_, nv_] = True
                stack.append((nu_, nv_))
    return False
