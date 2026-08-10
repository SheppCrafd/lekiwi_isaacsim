"""
Unit tests for lekiwi_tasks/cone_nav/mdp/_pure_math.py -- the tensor math extracted out
of actions.py/rewards.py/terminations.py specifically so it's testable without a real
Isaac Sim install (those three files import isaaclab at module level and can't be
imported here at all). torch is a real, local dependency for this -- no isaaclab/omni
needed, so this actually runs, unlike almost everything else in isaac_lab/.

Run: python tests/test_mdp_math.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lekiwi_tasks" / "cone_nav" / "mdp"))

import torch
from _pure_math import (
    apply_lidar_angular_jitter,
    apply_lidar_dropout,
    body_to_world_velocity,
    linear_anneal,
    out_of_bounds_mask,
    potential_shaping,
    success_mask,
    update_hold_counter,
)


def test_body_to_world_identity_at_zero_heading():
    vx_w, vy_w, omega_w = body_to_world_velocity(
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.5, -0.5]), torch.tensor([0.0, 0.0])
    )
    assert torch.allclose(vx_w, torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(vy_w, torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(omega_w, torch.tensor([0.5, -0.5]))  # omega passes through unchanged


def test_body_to_world_quarter_turn():
    # Facing +90deg (pi/2): body-forward (vx_b=1, vy_b=0) should point world +Y.
    vx_w, vy_w, _ = body_to_world_velocity(
        torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([math.pi / 2])
    )
    assert torch.allclose(vx_w, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(vy_w, torch.tensor([1.0]), atol=1e-6)


def test_body_to_world_preserves_magnitude():
    rng = torch.Generator().manual_seed(0)
    vx_b = torch.rand(200, generator=rng) * 2 - 1
    vy_b = torch.rand(200, generator=rng) * 2 - 1
    theta = torch.rand(200, generator=rng) * 2 * math.pi - math.pi
    vx_w, vy_w, _ = body_to_world_velocity(vx_b, vy_b, torch.zeros(200), theta)
    mag_b = torch.hypot(vx_b, vy_b)
    mag_w = torch.hypot(vx_w, vy_w)
    assert torch.allclose(mag_b, mag_w, atol=1e-5), "rotation must be norm-preserving"


def test_potential_shaping_sign():
    assert potential_shaping(torch.tensor(5.0), torch.tensor(3.0)).item() > 0  # got closer -> positive
    assert potential_shaping(torch.tensor(3.0), torch.tensor(5.0)).item() < 0  # retreated -> negative
    assert potential_shaping(torch.tensor(3.0), torch.tensor(3.0)).item() == 0


def test_potential_shaping_telescopes_over_episode():
    # Sum of per-step shaped rewards over an episode should equal start_dist - end_dist,
    # regardless of the path taken in between (the whole point of potential-based shaping).
    dists = torch.tensor([10.0, 8.0, 9.0, 4.0, 4.0, 1.0])
    total = sum(potential_shaping(dists[i], dists[i + 1]).item() for i in range(len(dists) - 1))
    assert math.isclose(total, (dists[0] - dists[-1]).item(), abs_tol=1e-6)


def test_update_hold_counter_increments_and_resets():
    hold = torch.zeros(3, dtype=torch.long)
    inside_sequence = [
        torch.tensor([True, False, True]),
        torch.tensor([True, False, True]),
        torch.tensor([False, True, True]),  # env0 leaves -> resets; env1 enters; env2 still in
    ]
    for inside in inside_sequence:
        hold = update_hold_counter(inside, hold)
    assert hold.tolist() == [0, 1, 3]


def test_success_mask_boundary():
    # dist + footprint <= goal_radius -> inside. Exactly on the boundary counts as inside.
    assert success_mask(torch.tensor(0.5), 0.16, torch.tensor(0.66)).item() is True
    assert success_mask(torch.tensor(0.51), 0.16, torch.tensor(0.66)).item() is False
    assert success_mask(torch.tensor(0.0), 0.16, torch.tensor(0.35)).item() is True  # robot at goal center, well inside


def test_out_of_bounds_mask():
    bounds = torch.tensor([[0.0, 5.0, 0.0, 3.0]])  # xmin,xmax,ymin,ymax
    inside = torch.tensor([[2.5, 1.5]])
    just_outside_within_margin = torch.tensor([[5.05, 1.5]])
    clearly_outside = torch.tensor([[6.0, 1.5]])

    assert out_of_bounds_mask(inside, bounds, margin_m=0.1).item() is False
    assert out_of_bounds_mask(just_outside_within_margin, bounds, margin_m=0.1).item() is False
    assert out_of_bounds_mask(clearly_outside, bounds, margin_m=0.1).item() is True


def test_out_of_bounds_mask_batched_different_bounds_per_env():
    # Regression guard for exactly the kind of per-env-course-size bug this project has
    # already hit once (course_generator.py's axis-swap bug) -- each env has ITS OWN
    # course footprint (varies per generated environment), so bounds must broadcast
    # per-row, not against a single shared box.
    bounds = torch.tensor([
        [0.0, 3.0, 0.0, 3.0],  # small course
        [0.0, 7.0, 0.0, 6.0],  # large course
    ])
    pos = torch.tensor([
        [3.5, 1.0],  # outside the SMALL course's xmax=3.0
        [3.5, 1.0],  # well inside the LARGE course
    ])
    result = out_of_bounds_mask(pos, bounds, margin_m=0.1)
    assert result.tolist() == [True, False]


def test_linear_anneal_endpoints_and_midpoint():
    assert linear_anneal(1.2, 0.5, 0.0) == 1.2
    assert linear_anneal(1.2, 0.5, 1.0) == 0.5
    assert math.isclose(linear_anneal(1.2, 0.5, 0.5), 0.85, abs_tol=1e-9)


def test_linear_anneal_direction_agnostic():
    # Curriculum widens num_cones_range (increasing) while tightening min_spacing
    # (decreasing) -- same function, both directions must work.
    assert linear_anneal(6, 14, 0.5) == 10
    assert linear_anneal(14, 6, 0.5) == 10


def test_lidar_dropout_rate_roughly_matches_probability():
    torch.manual_seed(0)
    ranges = torch.full((1, 10000), 3.0)
    corrupted = apply_lidar_dropout(ranges, dropout_prob=0.1)
    dropped_frac = (corrupted == 12.0).float().mean().item()
    assert 0.08 < dropped_frac < 0.12, f"expected ~10% dropout, got {dropped_frac:.3f}"
    assert (corrupted[corrupted != 12.0] == 3.0).all(), "non-dropped rays must be untouched"


def test_lidar_dropout_zero_prob_is_noop():
    ranges = torch.rand(5, 360) * 10
    assert torch.equal(apply_lidar_dropout(ranges, dropout_prob=0.0), ranges)


def test_lidar_angular_jitter_is_a_permutation_not_data_loss():
    torch.manual_seed(1)
    ranges = torch.arange(360, dtype=torch.float32).unsqueeze(0)  # distinct value per ray, easy to track
    jittered = apply_lidar_angular_jitter(ranges, jitter_std_deg=5.0)
    # A circular shift must be a permutation -- same multiset of values, just reordered.
    assert torch.equal(torch.sort(jittered[0])[0], torch.sort(ranges[0])[0])


def test_lidar_angular_jitter_zero_std_is_noop():
    ranges = torch.rand(5, 360) * 10
    assert torch.equal(apply_lidar_angular_jitter(ranges, jitter_std_deg=0.0), ranges)


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
