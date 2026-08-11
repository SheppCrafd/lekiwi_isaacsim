# isaac_lab/ — cone-avoidance nav task code

Everything plan.md Phases 3–8 and 10 describe as buildable *without* a running Isaac
Sim (i.e. everything that isn't literally "load the USD and press Play" or "acquire
physical hardware") — written, not yet run. **Nothing in here has touched a real Isaac
Sim instance.** Phase 1/2 (real GPU access, first real load) is still the hard blocker
plan.md always said it was; this code is what's ready to throw at it once that exists.

## What's real vs. what's a documented guess

Consistent with how the rest of this project treats verification (see
`isaac_sim/README_lekiwi_variants.md`'s own corrections history): most of this was
first written against Isaac Lab 2.x API conventions from memory, then **checked against
real Isaac Lab/rsl_rl source and docs** (fetched directly — `isaaclab_rl`'s actual
`rl_cfg.html` across every tagged release from v2.1.0 through v2.3.0 GA, the real
`action_manager.py`/`noise_cfg.py`/`shapes.py` source, rsl_rl's own GitHub repo, and
Isaac Lab's `main`-branch `setup.py` for its actual pinned `rsl-rl-lib` version) — not a
live install (no Isaac Sim exists in this environment), but no longer pure guesswork
either. Three things are independently *run*, not just read against source, and don't
share any of this risk:

- **`lekiwi_tasks/cone_nav/course_generator.py`** — pure numpy, no Isaac Sim
  dependency, actually run: 5000+ seeds tested (safety checks, reachability,
  determinism), including after two real bugs were caught by that testing, not by
  inspection (an axis-swap bug in the reachability grid, and a spawn-placement gap
  versus plan.md's own task spec). See its module docstring for the seed convention.
  **`tests/test_course_generator.py`** (added 2026-08-10, during a bug scan) persists a
  fast slice of this (300 seeds + two hand-built regression tests for the axis-swap
  bug specifically) as a real, routinely-runnable test — the module docstring used to
  point at this file before it existed, a real dangling reference, now fixed.
- **`scripts/render_course_map.py`** — also pure numpy + matplotlib, generates
  `hero_course.png`, a dimensioned real-world build reference (12ft × 20ft, 8 identical
  cones, fixed end zone, spawn-anywhere safe zone) for building a physical course.
- **`tests/test_mdp_math.py`** — pure-tensor unit tests (torch, no isaaclab import) for
  the reward/termination/action math itself, isolated from the Isaac Lab plumbing
  around it. See that file for what it does and doesn't cover.

**Confirmed correct against real source** (no longer flagged as guesses):
`mdp/actions.py`'s `ActionTerm` base-class contract; `env_cfg_camera.py`/
`env_cfg_lidar.py`'s `GaussianNoiseCfg`/`UniformNoiseCfg` field names;
`env_cfg_base.py`'s `ConeCfg`/`CuboidCfg` spawner fields; `mdp/events.py`'s
`quat_from_euler_xyz` signature; `agents/rsl_rl_ppo_lidar_cfg.py`'s entire cfg shape
(matches every tagged Isaac Lab release v2.1.0–v2.3.0 exactly); `scripts/train.py`/
`play.py`'s use of `isaaclab_tasks.utils.parse_env_cfg`/`load_cfg_from_registry` (Isaac
Lab's own real entry-point resolution helpers, replacing an earlier hand-rolled
`gym.spec(...).kwargs[...]` approach that worked but wasn't how Isaac Lab's own scripts
actually do it).

**Sensor stats/position also re-checked directly against the real hardware** (Seeed's
LeKiwi Kit and X10 camera product pages, a reseller page, Seeed's own LeRobot wiki, and
the SIGRobotics-UIUC community assembly guide): the RPLIDAR A1M8 numbers throughout this
project were already sourced from Slamtec's own official datasheet — confirmed as the
highest-confidence sensor spec here, nothing to change. The X10 camera's *position* was
already photo-verified against Seeed's real product photo in the previous session — also
holds. The X10 camera's *optics* (focal length, aperture, FOV baked into the USD Camera
prim) are still borrowed from an unrelated LightwheelAI/leisaac camera config, not the
real X10 — re-confirmed as a genuine, currently-unresolvable gap: three independent
sources checked today, none publish FOV/focal-length/sensor-size data for this camera.
Documented directly in `env_cfg_camera.py` now, not just the older `isaac_sim/README`.

**One confirmed real bug, found and fixed by this research, not by running anything:**
`env_cfg_lidar.py` originally used a plain `RayCasterCfg` against the per-env cones —
Isaac Lab's own docs state plain `RayCaster` mesh data "only works for literally static
meshes." The cones are dynamic `RigidObject`s repositioned every episode
(`mdp/events.py:regenerate_course`), so the lidar would very likely have silently never
seen a single cone. Now uses `MultiMeshRayCasterCfg`, added specifically for raycasting
against tracked moving meshes — confirmed to exist, not a guess, though its
`RaycastTargetCfg.is_shared`/`merge_prim_meshes` field semantics are still unconfirmed
(see the file's own comment).

Remaining risk, real and current, not resolved by this pass:

| File | Risk |
|---|---|
| `cone_nav/agents/nature_cnn_actor_critic.py` + `rsl_rl_ppo_camera_cfg.py` + `scripts/export_policy.py` | Targets rsl_rl's OLD single-class ActorCritic API — **CONFIRMED GONE (2026-08-10), not just a forward-compat guess**: pip-installed and inspected both the latest rsl-rl-lib (5.4.2) and the exact 5.0.1 Isaac Lab's `main` pins; `rsl_rl.modules.ActorCritic` doesn't exist in either. The NATIVE `RslRlCNNModelCfg`/`RslRlMLPModelCfg` path (confirmed to exist, in `isaaclab_rl` which isn't pip-installable standalone to verify further) is now the one to check FIRST in Phase 1, not a fallback — its multi-observation-group routing (image group vs. proprioceptive group) still isn't confirmed. `export_policy.py`'s lidar path now fails with a clear, actionable error if it hits the missing import, rather than a bare `ImportError`. |
| `cone_nav/mdp/events.py` | `randomize_sensor_mount_pose` mutates a sensor prim's transform post-creation via raw `pxr` — may need a re-initialize call depending on Isaac Sim version, or may need to move to a static per-scene-build offset instead of per-episode. Not researchable further without a live Kit process to test against. |
| `cone_nav/env_cfg_lidar.py` | `MultiMeshRayCasterCfg` confirmed to exist (see above), but `RaycastTargetCfg`'s exact field semantics (`is_shared`, `merge_prim_meshes`) are a best-effort guess, not confirmed. |
| `scripts/train.py` | `extras["log"]` termination-rate key names (e.g. `Episode_Termination/success`) are still an assumed format, not confirmed against a real run — print one dict and check if you rely on that logging path. (`play.py` no longer needs this guess, 2026-08-10 — it now reads `env.unwrapped.termination_manager.get_term(name)` directly, a real per-env signal rather than a parsed log key, though that call itself is still unverified against a real install.) |

None of this is a reason not to use the code — it's the same "here's what's provisional,
here's why" discipline the rest of the project already runs on, now with a lot less
actually left to that category than at the end of the previous session. Phase 2 ("Fix
whatever this step reveals — something will", per plan.md) is exactly the step that
resolves everything still in the table above; treat it as a checklist for Phase 2, not
a blocker before then.

## Second pass (2026-08-10, same day, after the API research above)

Everything not requiring Isaac Sim or the physical robot got another round: cone
placement is now a genuine random scatter (no lane template) with a much wider size
range (0.25–2ft tall, 0.125–1ft diameter — `course_generator.CONE_SIZE_RANGE_M`);
spawn is randomized across the whole open floor, not a fixed point; a Phase 6
curriculum/ADR ramp is implemented (`mdp/curriculum.py`, anneals cone density/spacing
over training); the two remaining sensor-noise gaps (dropped lidar returns, angular
jitter) are closed (`mdp/observations.py:lidar_ranges`); the reward/termination/action
tensor math was factored out into `mdp/_pure_math.py` and actually unit-tested
(`tests/test_mdp_math.py`, 15/15 passing, torch-only, no isaaclab needed); camera/lidar
sensor stats and position were re-checked directly against Seeed's product pages, a
reseller page, and Seeed's own wiki (lidar confirmed already correct against Slamtec's
datasheet; camera position still photo-verified; camera optics remain a genuine,
currently-unresolvable gap — no published source anywhere gives the X10's real
FOV/focal length); the hero course build was regenerated with 14 cones instead of 8;
and `blender_verify_12_views.py`'s flat/unlit bug got a real root-cause fix (see
plan.md Phase 0) even though it can't be run here to confirm.

## Third pass (2026-08-10, same day, live sensor effects)

Closed the specific gap the second pass's own notes above flagged as still open:
camera-side structured corruption (motion blur) and the difference between the
existing per-EPISODE sensor mount jitter (`mdp/events.py:randomize_sensor_mount_pose`,
a fixed offset for the whole episode, modeling assembly tolerance) and genuinely LIVE,
per-step jitter that reacts to the robot's current speed. Added:

- `mdp/_pure_math.py`: `speed_metric` (combines body-frame linear + angular speed into
  one scalar), `shake_std_from_speed`/`blur_weight_from_speed` (map that speed to a
  still/moving-interpolated magnitude), `apply_pixel_shake` (vectorized per-env
  circular image roll — gather-indexing, not a python loop, since image tensors are
  ~1000x larger per-env than a lidar scan and this runs every step for thousands of
  envs), `apply_motion_blur` (exponential temporal frame blend), and
  `apply_lidar_angular_jitter_variable_std` (the existing angular-jitter model, but
  with a per-env std instead of one shared float). All six are pure torch, unit-tested
  in `tests/test_mdp_math.py` (9 new tests, 24/24 total passing) — same "actually run,
  no isaacsim needed" category as the rest of that file, not just reasoned through.
- `mdp/observations.py:front_camera_rgb` now applies live mechanical shake (image
  content shift) and motion blur (temporal blend against `env.prev_camera_frame`) on
  every step, magnitude driven by current base speed — sharp and steady at rest,
  progressively shaky/smeared while moving. `mdp/observations.py:lidar_ranges`'s
  angular jitter is now the same speed-scaled shape instead of a flat constant.
- New state this required: `cone_nav_env.py` owns `prev_camera_frame` (lazily
  allocated, this class doesn't know the camera's resolution) and
  `camera_frame_valid` (a bool per env); `mdp/events.py:reset_camera_frame_state`
  (mode="reset", camera-variant only) clears the latter so a fresh episode's first
  frame never blends against the previous episode's last one.
- Also folded in, same day, same session: the wheel-mounting-radius correction
  (118mm → 117.6mm, all three wheels, both USD files) described in
  `isaac_sim/README_lekiwi_variants.md`'s wheel-geometry section — not code in this
  directory, but the same "close out a still-open gap found this session" pattern.

**New risk, same category as the rest of this table:** `env.prev_camera_frame` stores
a full `(N, H, W, 3)` float tensor per camera-variant training run (2500 envs × 480×640×3
× 4 bytes ≈ 2.9GB at default resolution) — additional VRAM pressure on top of the
camera-variant memory question Phase 3/4 already flagged as unconfirmed at 2500 envs on
an A100 40GB. Untested, not just unvalidated-constants risk like the rest of this pass's
additions — this is the first thing to check in Phase 2 if the camera variant doesn't
fit in memory, before dropping `num_envs` for an unrelated reason.

## Fourth pass (2026-08-10, same day, "anything the robot might endure")

Explicit ask: simulate real-world failure modes beyond generic noise — camera glare,
lens smudges, lidar misreads, voltage spikes, "anything." Scoped to a bounded,
physically-grounded set (not literally infinite) and implemented the same way as the
third pass: pure, unit-tested math in `_pure_math.py`, wired through
`observations.py`/`actions.py`/`events.py`.

- **Camera** (`front_camera_rgb`, mostly per-episode constants from
  `mdp/events.py:randomize_camera_defects`): lens glare (`apply_lens_glare` +
  `glare_intensity_from_heading` — brightness wash-out, LIVE per-step since it depends
  on current heading vs. this episode's fixed random sun azimuth), lens smudge
  (`apply_lens_smudge` — static per-episode patch, absent in 70% of episodes by
  default), exposure/white-balance drift (`apply_exposure_white_balance`), dead/hot
  pixels (`apply_dead_pixels` — 0-3 stuck photosites/episode). Applied in physical
  capture order: shake/blur (third pass) → glare → smudge → exposure/WB → dead pixels
  (a stuck photosite ignores everything upstream of it) → Isaac Lab's own pixel-value
  noise wrapper.
- **Lidar** (`lidar_ranges`, both LIVE per-step): spurious short-range misreads
  (`apply_lidar_misreads` — simulated multipath reflection; deliberately the more
  dangerous falsely-CLOSE direction, distinct from dropout's falsely-far/no-return),
  whole-scan freeze glitches (`apply_lidar_freeze` — repeats the previous scan
  verbatim, modeling a comms/firmware hiccup; needs `env.prev_lidar_scan`/
  `lidar_scan_valid`, same temporal-buffer pattern as the camera's motion-blur state,
  cleared each reset by `mdp/events.py:reset_lidar_scan_state`).
- **Actuation:** a voltage/wattage brownout term was built here (a live per-step
  transient power-loss effect in `mdp/actions.py`'s `BodyVelocityAction`) and then
  **removed the same day, by explicit direction**: an unexpected power drop on the
  real robot is a wiring/electrical-integration fault on the builder's own end, not
  an environmental condition worth training the policy to tolerate. Not implemented.
- All 8 remaining new pure functions unit-tested (`tests/test_mdp_math.py`, 12 new
  tests, **36/36 total passing**).
- Genuinely NOT covered, by scope choice, not oversight: lens distortion (geometric
  warp), compression artifacts, humidity/condensation, thermal derating, and (per the
  removal above) any actuation-side power/voltage failure. "Anything the robot might
  endure" was interpreted as a bounded, physically-reasoned set of *environmental*
  conditions worth actually building well — not a claim of literal completeness, and
  deliberately excluding failure modes that are really about build/wiring quality
  rather than the world the robot operates in.
- **Same-day cross-check, not new code:** the X10 camera's real resolution is now
  confirmed (Seeed's own product page: 1080p / 1920x1080) — `env_cfg_camera.py` and
  `deploy/lekiwi_policy_runner.py` both already deliberately operate at 640x480 for
  training-time VRAM reasons; now documented as a deliberate downsample in both files
  rather than an unexplained mismatch. FOV/focal-length/sensor-size are still
  genuinely unpublished anywhere, re-confirmed against the exact same product page.

## Fifth pass (2026-08-10, same day, bug scan + fixes)

A deliberate "find real bugs without a running Isaac Sim" pass: compiled every `.py`
file in the project, ran the full test suite, independently stress-tested
`course_generator.py` (8000 fresh seeds via a throwaway script, 0 failures — see below
for where a slice of this got persisted), and read every remaining file by hand,
cross-referencing signatures/shapes/the real USD's joint graph across file boundaries.
Found and fixed:

- **SEVERE, now fixed: `rewards.py`/`terminations.py` read a robot position that never
  moved.** All of `approach_goal_potential`, `success_bonus`, `goal_reached_and_held`,
  and `out_of_bounds` computed "the robot's position" from `asset.data.root_pos_w` —
  the articulation ROOT's world pose. Verified directly against the real USD's joint
  graph (`pxr`, both `usd/lekiwi_camera.usd` and `usd/lekiwi_lidar.usd`): the root link
  is `/LeKiwi/world`, anchored to the global frame via a `FixedJoint` with an empty
  `body0` and carrying `RigidBodyAPI` (while `/LeKiwi` itself, where
  `ArticulationRootAPI` sits, has none) — so PhysX resolves the real root body to
  `world`, which sits at the *top* of the chain (`world -> base_x_link -> base_y_link
  -> base_theta_link -> base`) and is never driven by anything. `root_pos_w` was
  therefore a per-env constant (`robots/lekiwi.py`'s `InitialStateCfg.pos`), not the
  robot's actual driven position — meaning reward shaping gave ~zero signal and both
  terminations essentially never fired, regardless of where the robot actually drove.
  `mdp/observations.py:base_pose_2d` (written independently, at a different time) had
  already gotten this right, reading `joint_pos` directly — the two implementations of
  the same fact had silently drifted apart. Fixed by factoring out one shared
  `mdp/_robot_state.py` (`robot_local_xy`/`robot_world_xy`), now used by all three
  files including `observations.py` (a pure dedup there, not a behavior change).
- **`scripts/play.py`'s success/collision counting was wrong.** It read
  `extras["log"]` — a batch-level aggregate logged once per `env.step()` — inside a
  per-`env_id` loop as if it were that env's own outcome, over-crediting every env in
  `done_ids` whenever multiple eval envs finished the same step with mixed outcomes
  (the common case at `num_eval_envs=256`). Also had a dead `env_id in done_ids` check
  (always true — it's iterating `done_ids` itself) and declared `collisions`/
  `out_of_bounds` counters that were never incremented or printed. Fixed by reading
  each termination term's real per-env boolean straight from
  `env.unwrapped.termination_manager.get_term(name)` right after `env.step()`, and by
  actually reporting collision/out-of-bounds/timeout rates alongside success rate.
- **`rsl_rl.modules.ActorCritic` is confirmed gone, not a hypothetical risk.**
  `agents/nature_cnn_actor_critic.py`'s own docstring already flagged this as an open
  forward-compat question ("Isaac Lab's `main` branch already pins `rsl-rl-lib==5.0.1`
  and has already deleted `actor_critic.py`...") — this pass actually pip-installed
  and inspected both the latest release (5.4.2) and 5.0.1 specifically. Confirmed:
  `ActorCritic` doesn't exist in either; `rsl_rl.modules` only exposes the new modular
  `MLP`/`CNN`/`GaussianDistribution` split in both. `scripts/export_policy.py`'s
  lidar-variant export path (the only place that actually imports the old class) now
  fails with a clear, actionable error instead of a bare `ImportError` if it hits this
  — not rewritten to the new native `RslRlCNNModelCfg`/`RslRlMLPModelCfg` path, since
  that lives in `isaaclab_rl` (not pip-installable standalone, can't verify its exact
  shape from here) — separately, fixed `export_policy.py`'s reconstruction of both
  variants to supply `num_actor_obs`/`num_critic_obs`/`num_actions` (missing
  entirely before — these are normally injected by Isaac Lab's runner from the live
  env at training time, absent when reconstructing the network standalone) and reduced
  the camera path's hardcoded-hyperparameter duplication by sourcing from
  `LekiwiCameraPPORunnerCfg.policy` directly instead of a second hand-copied literal.
- **`course_generator.py`'s own module docstring pointed at a test file that didn't
  exist** (`isaac_lab/tests/test_course_generator.py`) — the "5000+ seeds tested" claim
  in this README had no persisted script backing it up either. Fixed by adding that
  file for real: 300 seeds (fast enough for routine runs) plus two hand-built
  regression tests targeting the exact axis-swap bug class `_goal_reachable`'s own
  docstring already describes catching once. The larger 8000-seed throwaway
  verification from this pass isn't itself persisted (kept routine test time short)
  but reproduced 0 failures, matching the original claim.

## Sixth pass (2026-08-10, same day, camera-spec matching)

Asked to make the simulated camera match the real X10's spec exactly. Re-checked
every claim `env_cfg_camera.py`'s own confidence breakdown makes, one item at a time,
rather than assuming the prior pass's conclusions still held:

- **Resolution, position, frame rate: already correct, re-confirmed, nothing to
  change.** 640×480 is a deliberate, matched downsample from the X10's real native
  1920×1080 on both the sim side (`env_cfg_camera.py`) and the real deploy path
  (`deploy/lekiwi_policy_runner.py`'s `CameraSensorReader`); both also operate at
  30fps (sim's `update_period=1/30`, real `CONTROL_HZ=30` driving the read loop).
  Camera position is photo-verified against Seeed's real product photo.
- **Optics: a real bug found, not just re-confirmed as unknown.** The USD Camera
  prim's `focal_length=36.5mm`/`horizontal_aperture=36.83mm` (still copied from an
  unrelated LightwheelAI/leisaac camera — the real X10's optics remain genuinely
  unpublished, now checked against five independent sources including the actual
  datasheet PDF, not three) mathematically produce **~53.5° horizontal FOV**, not the
  "~75°" this project's own comments claimed in two files
  (`env_cfg_camera.py`, `isaac_sim/README_lekiwi_variants.md`). Grepped the rest of
  the codebase first to confirm nothing (reward shaping, course generation) actually
  depends on 75° being true — nothing does, it was pure documentation arithmetic
  error, not a load-bearing constant. Fixed by correcting the comments to the number
  the existing values actually produce, rather than changing the values themselves to
  hit 75° — there's no real X10 FOV to target either way, so retargeting the numbers
  would just be trading one unverified guess for another.
- **Net effect:** the sim camera now matches every real, currently-knowable X10 spec
  (resolution, position, frame rate) exactly, and its one remaining unverifiable spec
  (FOV/focal length/sensor size — never published by Seeed anywhere, reconfirmed) is
  now at least documented *accurately* for what it currently is, instead of carrying
  a second, independent error on top of being an admitted placeholder.

## Seventh pass (2026-08-11, lidar + camera FOV matching)

Asked (science-fair framing: does a 360° lidar vs. ~70-90° camera FOV mismatch affect
the experiment?) whether the two sensor variants' FOVs should be matched. Short answer
given first: no, it doesn't invalidate anything — the two are trained and evaluated as
fully independent policies, never compared head-to-head, so mismatched FOV was never a
confound. Then asked to do it anyway (cheaper than buying new hardware, and a real fix
for "53.5° is tiny for nav" either way):

- **Lidar:** `env_cfg_lidar.py`'s `horizontal_fov_range` narrowed from the full
  `(-180, 180)` sweep to a forward-facing `(-45, 45)` (90°) window. The RPLIDAR still
  physically spins 360° — this only changes what the trained policy is shown.
- **Camera:** the USD Camera prim's baked-in `horizontal_aperture`/`vertical_aperture`
  changed (via direct `pxr` edit, confirmed persisted by reopening the file from disk
  after writing) from `36.83mm`/`15.29mm` (~53.5° horizontal FOV) to `73.0mm`/`54.75mm`
  (exactly 90° horizontal FOV, `vertical_aperture` now also actually matching the
  640×480 render resolution's 4:3 aspect — the old values never did, a real latent
  stretch bug fixed as a side effect of touching this).
- **Real bug caught along the way:** `mdp/_pure_math.py`'s
  `apply_lidar_angular_jitter`/`_variable_std` derived the sensor's angular resolution
  as `360.0 / num_rays`, silently assuming the ray array always spans a full 360°
  circle. True by coincidence at the old FOV (360 rays / 360° = 1°/ray, matching the
  real `horizontal_res=1.0`), but would have silently computed a 4x-too-coarse
  `4°/ray` the moment the FOV narrowed to 90° (90 rays / 360° per the old formula).
  Fixed by threading the sensor's real angular resolution through explicitly
  (`deg_per_ray` param, `observations.py`'s new `angular_res_deg` obs-term param)
  instead of inferring it from array width. Caught by reasoning through the change's
  downstream effects before shipping it, then proven with a new regression test
  (`test_apply_lidar_angular_jitter_deg_per_ray_scales_shift_magnitude`) that
  statistically confirms the old inferred value would have understated jitter by
  ~4x — not just that the fixed function runs without crashing.
- Hardcoded ray-count assumptions elsewhere (`scripts/export_policy.py`'s
  `LIDAR_NUM_RAYS`, `deploy/lekiwi_policy_runner.py`'s `_num_rays`) updated 360 → 90 to
  match, both flagged as needing reconfirmation once Phase 2's real `LidarPatternCfg`
  is available (exact endpoint-inclusive ray count at a 90° window, not verified here).
- This is a simulation/training-config change only — it doesn't touch what FOV the
  real physical X10 or a real RPLIDAR actually have. Whether the real X10 gets
  physically replaced with a wider-FOV webcam is a separate hardware decision — see
  `BoM.md`/`plan.md` Phase 9 for whether that was pursued this session.

## Layout

```
lekiwi_tasks/
  robots/lekiwi.py            ArticulationCfg for usd/lekiwi_camera.usd & lekiwi_lidar.usd
  cone_nav/
    course_generator.py       procedural cone-course generator (tested, no Isaac Sim dep)
    cone_nav_env.py           custom ManagerBasedRLEnv subclass, owns per-env course state
    env_cfg_base.py           shared scene/action/reward/termination/event wiring
    env_cfg_camera.py         + camera sensor, image observation group
    env_cfg_lidar.py          + MultiMeshRayCaster lidar sensor, range observation group
    mdp/
      _pure_math.py            tensor math with zero isaaclab dependency -- unit-tested, see tests/
      observations.py         policy obs (sensor + proprioceptive only — no privileged state); live camera shake/blur/glare + lidar jitter/misreads/freeze live here
      rewards.py               potential-based approach shaping, success-with-hold, collision, smoothness
      terminations.py          cone collision, success, out-of-bounds (time_out is Isaac Lab's built-in)
      events.py                course regeneration, actuation latency/slip, per-episode sensor mount jitter + camera defects, background clutter, temporal-buffer resets
      actions.py                body-frame vx/vy/omega -> world-frame base_x/y/theta joint targets
      curriculum.py             Phase 6 ADR ramp: anneals cone density/spacing over training
    agents/
      rsl_rl_ppo_lidar_cfg.py   verified hyperparameters (copied from MuammerBay/isaac_so_arm101)
      rsl_rl_ppo_camera_cfg.py  same hyperparameters + custom CNN policy
      nature_cnn_actor_critic.py
scripts/
  train.py                    Phase 7, with spot-preemption checkpointing/auto-resume
  play.py                     Phase 8, held-out eval seed range
  export_policy.py            Phase 10, ONNX/TorchScript export
  render_course_map.py        offline course preview + physical hero-course build spec (tested)
deploy/
  lekiwi_policy_runner.py     Phase 10-11 on-robot inference loop SKELETON (needs real LeRobot API + hardware, Phase 9)
tests/
  test_mdp_math.py            unit tests for mdp/_pure_math.py -- torch only, actually run (37/37 passing)
```

## Running once Phase 1/2 access exists

```bash
# on the Google Cloud Isaac Sim instance, inside its Isaac Sim python env:
cd isaac_lab
pip install -e .   # see setup.py -- makes lekiwi_tasks importable
python scripts/train.py --task lidar --headless
python scripts/train.py --task camera --headless
python scripts/play.py --task lidar --checkpoint logs/rsl_rl/lekiwi_conenav_lidar/model_XXXX.pt
python scripts/export_policy.py --task lidar --checkpoint ... --out lidar_policy
```

## Building the physical hero course

```bash
python scripts/render_course_map.py --hero --out hero_course.png
```

Prints a tape-measure build spec (feet-inches + meters) and saves the dimensioned PNG.
8 identical 18in/10in-dia traffic cones, 12ft × 20ft footprint, end zone fixed at the
far end (never moves between runs), robot can start anywhere else on the floor.
