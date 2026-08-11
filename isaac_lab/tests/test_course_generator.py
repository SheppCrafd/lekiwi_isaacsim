"""
Unit tests for lekiwi_tasks/cone_nav/course_generator.py -- pure numpy, no isaaclab/omni
needed, so this actually runs (same category as tests/test_mdp_math.py). Persists what
was previously only an ad-hoc, non-repo verification pass: course_generator.py's own
module docstring pointed here before this file existed (a real, now-fixed dangling
reference, found during a 2026-08-10 bug scan) and isaac_lab/README.md's "actually run:
5000+ seeds tested" claim had no persisted script backing it up. An independent 8000-seed
run during that same scan (not part of this file, to keep routine test time short)
reproduced that claim with 0 failures -- this file keeps a smaller, fast slice of the
same checks runnable on every normal test pass, not the full 8000.

Run: python tests/test_course_generator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lekiwi_tasks" / "cone_nav"))

import numpy as np
from course_generator import (
    ConeSpec,
    CourseGeneratorCfg,
    CourseLayout,
    TOTAL_SEEDS,
    TRAIN_SEED_UPPER,
    _goal_reachable,
    generate_course,
    validate_layout,
)

STRESS_SEED_COUNT = 300  # keep routine runs fast; a larger one-off 8000-seed pass was separately verified clean


def test_seed_convention_ranges_are_sane():
    assert 0 < TRAIN_SEED_UPPER < TOTAL_SEEDS


def test_many_seeds_generate_without_error_and_pass_their_own_validation():
    cfg = CourseGeneratorCfg()
    for seed in range(STRESS_SEED_COUNT):
        layout = generate_course(seed, cfg)  # raises CourseGenerationError on failure -- let it propagate
        ok, reason = validate_layout(layout, cfg)
        assert ok, f"seed={seed}: generate_course returned a layout failing its own validate_layout: {reason}"


def test_generation_is_deterministic():
    cfg = CourseGeneratorCfg()
    for seed in (0, 1, 12345, 999_999):
        a = generate_course(seed, cfg)
        b = generate_course(seed, cfg)
        assert (a.spawn_x, a.spawn_y, a.goal_x, a.goal_y, len(a.cones)) == (
            b.spawn_x, b.spawn_y, b.goal_x, b.goal_y, len(b.cones)
        )
        for ca, cb in zip(a.cones, b.cones, strict=True):
            assert (ca.x, ca.y, ca.radius, ca.height, ca.shape) == (cb.x, cb.y, cb.radius, cb.height, cb.shape)


def test_everything_stays_within_course_bounds():
    cfg = CourseGeneratorCfg()
    for seed in range(STRESS_SEED_COUNT):
        layout = generate_course(seed, cfg)
        assert 0.0 <= layout.spawn_x <= layout.length_m
        assert 0.0 <= layout.spawn_y <= layout.width_m
        assert 0.0 <= layout.goal_x <= layout.length_m
        assert 0.0 <= layout.goal_y <= layout.width_m
        for cone in layout.cones:
            assert 0.0 <= cone.x <= layout.length_m
            assert 0.0 <= cone.y <= layout.width_m


def test_validate_layout_rejects_spawn_overlapping_a_cone():
    cfg = CourseGeneratorCfg()
    layout = CourseLayout(
        seed=0, width_m=5.0, length_m=5.0,
        cones=[ConeSpec(x=2.0, y=2.0, radius=0.15, height=0.4, shape="cone")],
        goal_x=4.5, goal_y=2.5, goal_radius_m=0.35,
        spawn_x=2.0, spawn_y=2.0,  # dead center of the cone -- must be rejected
    )
    ok, reason = validate_layout(layout, cfg)
    assert not ok
    assert "cone" in reason


def test_validate_layout_rejects_spawn_overlapping_goal():
    cfg = CourseGeneratorCfg()
    layout = CourseLayout(
        seed=0, width_m=5.0, length_m=5.0, cones=[],
        goal_x=2.5, goal_y=2.5, goal_radius_m=0.35,
        spawn_x=2.5, spawn_y=2.5,  # dead center of the goal -- must be rejected
    )
    ok, reason = validate_layout(layout, cfg)
    assert not ok
    assert "goal" in reason


def test_goal_reachable_true_with_no_obstacles():
    cfg = CourseGeneratorCfg()
    layout = CourseLayout(
        seed=0, width_m=5.0, length_m=5.0, cones=[],
        goal_x=4.5, goal_y=2.5, goal_radius_m=0.35,
        spawn_x=0.5, spawn_y=2.5,
    )
    assert _goal_reachable(layout, cfg)


def test_goal_reachable_false_behind_a_wall_of_cones():
    # A dense row of cones spanning the full width, spacing well under 2*radius,
    # sitting between spawn (x=0.5) and goal (x=4.5) -- no gap to slip through.
    cfg = CourseGeneratorCfg()
    wall_x = 2.5
    cones = [ConeSpec(x=wall_x, y=y, radius=0.3, height=0.4, shape="cone") for y in np.arange(0.0, 5.01, 0.4)]
    layout = CourseLayout(
        seed=0, width_m=5.0, length_m=5.0, cones=cones,
        goal_x=4.5, goal_y=2.5, goal_radius_m=0.35,
        spawn_x=0.5, spawn_y=2.5,
    )
    assert not _goal_reachable(layout, cfg)


def test_goal_reachable_respects_width_length_axis_convention():
    # Regression guard for the exact axis-swap bug course_generator.py's own
    # docstring describes catching during real testing: a non-square course (width
    # != length) where a wall of cones blocks the nav (x) axis specifically -- if the
    # reachability grid ever swapped width/length again, this wall would silently
    # sit in the wrong place and this test would start seeing "reachable" incorrectly.
    cfg = CourseGeneratorCfg()
    width, length = 3.0, 6.0
    wall_x = 3.0  # partway along the LONGER (length) axis
    cones = [ConeSpec(x=wall_x, y=y, radius=0.3, height=0.4, shape="cone") for y in np.arange(0.0, width + 0.01, 0.4)]
    layout = CourseLayout(
        seed=0, width_m=width, length_m=length, cones=cones,
        goal_x=length - 0.5, goal_y=width / 2, goal_radius_m=0.35,
        spawn_x=0.5, spawn_y=width / 2,
    )
    assert not _goal_reachable(layout, cfg)


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
