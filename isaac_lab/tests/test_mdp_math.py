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
    apply_dead_pixels,
    apply_exposure_white_balance,
    apply_lens_glare,
    apply_lens_smudge,
    apply_lidar_angular_jitter,
    apply_lidar_angular_jitter_variable_std,
    apply_lidar_dropout,
    apply_lidar_freeze,
    apply_lidar_misreads,
    apply_motion_blur,
    apply_pixel_shake,
    blur_weight_from_speed,
    body_to_world_velocity,
    glare_intensity_from_heading,
    linear_anneal,
    out_of_bounds_mask,
    potential_shaping,
    shake_std_from_speed,
    speed_metric,
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


def test_speed_metric_combines_linear_and_angular():
    # Pure linear: matches hypot exactly regardless of omega_to_linear_m.
    s = speed_metric(torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([0.0]))
    assert torch.allclose(s, torch.tensor([5.0]))
    # Pure rotation contributes omega_to_linear_m * |omega|.
    s2 = speed_metric(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([2.0]), omega_to_linear_m=0.15)
    assert torch.allclose(s2, torch.tensor([0.3]))
    # Sign of omega doesn't matter -- spinning either direction shakes the mount the same.
    s3 = speed_metric(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([-2.0]), omega_to_linear_m=0.15)
    assert torch.allclose(s3, s2)


def test_shake_std_from_speed_still_vs_moving_endpoints():
    speed = torch.tensor([0.0, 0.2, 0.4, 1.0])
    std = shake_std_from_speed(speed, still_std=0.1, moving_std=3.0, speed_at_moving_std=0.4)
    assert math.isclose(std[0].item(), 0.1, abs_tol=1e-6)  # stationary -> still_std exactly
    assert math.isclose(std[2].item(), 3.0, abs_tol=1e-6)  # at the reference speed -> moving_std exactly
    assert math.isclose(std[3].item(), 3.0, abs_tol=1e-6)  # past the reference speed -> clamped, not extrapolated
    assert std[0] < std[1] < std[2], "must ramp monotonically between the two endpoints"


def test_blur_weight_from_speed_decreases_with_speed():
    speed = torch.tensor([0.0, 0.2, 0.4, 1.0])
    w = blur_weight_from_speed(speed, still_weight=1.0, moving_weight=0.35, speed_at_moving_weight=0.4)
    assert math.isclose(w[0].item(), 1.0, abs_tol=1e-6)   # stationary -> no blur
    assert math.isclose(w[2].item(), 0.35, abs_tol=1e-6)  # at the reference speed -> full moving blur
    assert math.isclose(w[3].item(), 0.35, abs_tol=1e-6)  # clamped past the reference speed
    assert w[0] > w[1] > w[2], "blend weight must ramp DOWN as speed increases (more blur, not less)"


def test_apply_motion_blur_blend_math():
    frame = torch.full((2, 2, 2, 3), 10.0)
    prev = torch.zeros(2, 2, 2, 3)
    w = torch.tensor([1.0, 0.25])  # env0: no blur (pure current frame); env1: mostly previous
    out = apply_motion_blur(frame, prev, w)
    assert torch.allclose(out[0], torch.full((2, 2, 3), 10.0))
    assert torch.allclose(out[1], torch.full((2, 2, 3), 2.5))  # 0.25*10 + 0.75*0


def test_apply_pixel_shake_zero_shift_is_noop():
    frame = torch.rand(3, 8, 10, 3)
    zero = torch.zeros(3)
    out = apply_pixel_shake(frame, zero, zero)
    assert torch.equal(out, frame)


def test_apply_pixel_shake_matches_manual_roll():
    frame = torch.arange(2 * 4 * 5 * 1, dtype=torch.float32).reshape(2, 4, 5, 1)
    shift_x = torch.tensor([2.0, 0.0])
    shift_y = torch.tensor([0.0, 1.0])
    out = apply_pixel_shake(frame, shift_x, shift_y)
    expected0 = torch.roll(frame[0], shifts=2, dims=1)  # x -> width dim
    expected1 = torch.roll(frame[1], shifts=1, dims=0)  # y -> height dim
    assert torch.equal(out[0], expected0)
    assert torch.equal(out[1], expected1)


def test_apply_pixel_shake_is_content_preserving_permutation():
    frame = torch.rand(2, 6, 7, 3)
    out = apply_pixel_shake(frame, torch.tensor([3.0, -2.0]), torch.tensor([-1.0, 4.0]))
    for i in range(2):
        assert torch.allclose(torch.sort(out[i].flatten())[0], torch.sort(frame[i].flatten())[0])


def test_apply_lidar_angular_jitter_variable_std_zero_is_noop():
    ranges = torch.rand(4, 360) * 10
    out = apply_lidar_angular_jitter_variable_std(ranges, torch.zeros(4))
    assert torch.equal(out, ranges)


def test_apply_lidar_angular_jitter_variable_std_per_env_magnitude():
    torch.manual_seed(2)
    ranges = torch.arange(360, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    # env0 gets zero jitter std, env1 gets a large one -- env0 must stay exactly put,
    # env1 is still a permutation of the same values (circular shift, no data loss).
    out = apply_lidar_angular_jitter_variable_std(ranges, torch.tensor([0.0, 20.0]))
    assert torch.equal(out[0], ranges[0])
    assert torch.equal(torch.sort(out[1])[0], torch.sort(ranges[1])[0])


def test_apply_lidar_angular_jitter_deg_per_ray_scales_shift_magnitude():
    """
    Regression test for a real bug fixed 2026-08-11: deg_per_ray used to be silently
    inferred as 360.0/num_rays inside this function, which only happened to be correct
    while the lidar's FOV was a full 360deg sweep (360 rays / 360deg = 1deg/ray). Once
    the FOV was narrowed to a forward-facing 90deg window (env_cfg_lidar.py, to match
    the camera variant's FOV) that inference would have silently computed 360/90 =
    4deg/ray for a sensor whose REAL resolution is still 1deg/ray -- understating
    jitter shifts by 4x. Proves the now-explicit deg_per_ray parameter actually
    controls shift magnitude as expected, not just that the function runs.
    """
    torch.manual_seed(4)
    n = 2000
    num_rays = 90
    ranges = torch.arange(num_rays, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    jitter_std_deg = torch.full((n,), 4.0)
    out_correct = apply_lidar_angular_jitter_variable_std(ranges.clone(), jitter_std_deg, deg_per_ray=1.0)
    out_old_buggy = apply_lidar_angular_jitter_variable_std(ranges.clone(), jitter_std_deg, deg_per_ray=4.0)

    def mean_abs_shift(shifted: torch.Tensor) -> float:
        # Recover each row's shift by finding where the original ray-0 value (0.0) landed.
        positions = (shifted == 0.0).float().argmax(dim=-1)
        signed = torch.where(positions > num_rays // 2, positions - num_rays, positions)
        return signed.abs().float().mean().item()

    mean_correct = mean_abs_shift(out_correct)
    mean_old_buggy = mean_abs_shift(out_old_buggy)
    assert mean_correct > 2.5 * mean_old_buggy, (
        f"expected deg_per_ray=1.0 to shift meaningfully more ray-bins than the old "
        f"buggy 360/num_rays=4.0 inference at the same jitter_std_deg, got "
        f"mean_correct={mean_correct:.2f} mean_old_buggy={mean_old_buggy:.2f}"
    )


def test_glare_intensity_peaks_facing_source_and_vanishes_beyond_half_width():
    heading = torch.tensor([0.0, 0.1, 0.3, math.pi])
    intensity = glare_intensity_from_heading(heading, sun_azimuth_rad=torch.zeros(4), half_width_rad=0.2)
    assert math.isclose(intensity[0].item(), 1.0, abs_tol=1e-6)  # dead-on -> full intensity
    assert 0.0 < intensity[1].item() < 1.0
    assert intensity[2].item() == 0.0  # past half_width -> clamped to zero
    assert intensity[3].item() == 0.0  # facing directly away


def test_glare_intensity_wraps_around_2pi():
    # Heading just past 2*pi and sun azimuth near 0 are only 0.1 rad apart (true
    # angular gap), not the ~6.18 rad a naive unwrapped subtraction would compute --
    # use a wide half_width so that small true gap reads as high intensity, proving
    # the wraparound is handled rather than just checking a threshold that happens to
    # pass either way.
    heading = torch.tensor([2 * math.pi - 0.05])
    intensity = glare_intensity_from_heading(heading, sun_azimuth_rad=torch.tensor([0.05]), half_width_rad=2.0)
    assert intensity.item() > 0.9  # true gap 0.1 rad / half_width 2.0 -> ~0.95


def test_apply_lens_glare_zero_intensity_is_noop():
    frame = torch.rand(3, 4, 4, 3) * 255
    out = apply_lens_glare(frame, torch.zeros(3), torch.full((3,), 150.0))
    assert torch.allclose(out, frame)


def test_apply_lens_glare_brightens_and_clamps():
    frame = torch.full((2, 2, 2, 3), 200.0)
    out = apply_lens_glare(frame, torch.tensor([1.0, 0.5]), torch.full((2,), 100.0))
    assert torch.allclose(out[0], torch.full((2, 2, 3), 255.0))  # 200+100 clamped to 255
    assert torch.allclose(out[1], torch.full((2, 2, 3), 250.0))  # 200+50, no clamp needed


def test_apply_lens_smudge_zero_opacity_is_noop():
    frame = torch.rand(2, 20, 20, 3) * 255
    out = apply_lens_smudge(frame, torch.tensor([10.0, 5.0]), torch.tensor([10.0, 5.0]), torch.tensor([5.0, 5.0]), torch.zeros(2))
    assert torch.allclose(out, frame)


def test_apply_lens_smudge_darkens_center_more_than_far_edge():
    frame = torch.full((1, 40, 40, 3), 200.0)
    out = apply_lens_smudge(frame, torch.tensor([20.0]), torch.tensor([20.0]), torch.tensor([8.0]), torch.tensor([1.0]))
    center_val = out[0, 20, 20, 0].item()
    far_val = out[0, 0, 0, 0].item()
    assert center_val < far_val
    assert math.isclose(far_val, 200.0, abs_tol=1e-4)  # untouched far outside the smudge radius


def test_apply_exposure_white_balance_identity_gain_is_noop():
    frame = torch.rand(2, 3, 3, 3) * 255
    out = apply_exposure_white_balance(frame, torch.ones(2), torch.ones(2, 3))
    assert torch.allclose(out, frame, atol=1e-4)


def test_apply_exposure_white_balance_scales_and_clamps():
    frame = torch.full((1, 2, 2, 3), 100.0)
    out = apply_exposure_white_balance(frame, torch.tensor([3.0]), torch.ones(1, 3))
    assert torch.allclose(out, torch.full((1, 2, 2, 3), 255.0))  # 300 clamped to 255


def test_apply_dead_pixels_sets_stuck_value_only_where_active():
    frame = torch.zeros(2, 5, 5, 3)
    px_x = torch.tensor([[2], [2]])
    px_y = torch.tensor([[3], [3]])
    px_value = torch.tensor([[[255.0, 255.0, 255.0]], [[0.0, 0.0, 0.0]]])
    active = torch.tensor([[True], [False]])
    out = apply_dead_pixels(frame, px_x, px_y, px_value, active)
    assert torch.equal(out[0, 3, 2], torch.tensor([255.0, 255.0, 255.0]))  # active -> stuck
    assert torch.equal(out[1, 3, 2], torch.tensor([0.0, 0.0, 0.0]))  # inactive -> untouched (was already 0 here)
    assert out[0, 0, 0].sum().item() == 0.0  # everywhere else untouched


def test_apply_lidar_misreads_zero_prob_is_noop():
    ranges = torch.rand(4, 360) * 10
    out = apply_lidar_misreads(ranges, misread_prob=0.0, min_ghost_frac=0.1, max_ghost_frac=0.5)
    assert torch.equal(out, ranges)


def test_apply_lidar_misreads_only_shortens_never_lengthens():
    torch.manual_seed(3)
    ranges = torch.full((1, 5000), 5.0)
    out = apply_lidar_misreads(ranges, misread_prob=0.3, min_ghost_frac=0.1, max_ghost_frac=0.6)
    assert (out <= ranges + 1e-6).all(), "misreads must never report a LONGER range than truth"
    frac_changed = (out != ranges).float().mean().item()
    assert 0.2 < frac_changed < 0.4, f"expected ~30% misread rate, got {frac_changed:.3f}"


def test_apply_lidar_freeze_mask_semantics():
    ranges = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prev = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    out = apply_lidar_freeze(ranges, prev, torch.tensor([True, False]))
    assert torch.equal(out[0], prev[0])   # frozen -> repeats previous
    assert torch.equal(out[1], ranges[1])  # not frozen -> fresh reading passes through


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
