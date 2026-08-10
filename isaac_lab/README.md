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
| `cone_nav/agents/nature_cnn_actor_critic.py` + `rsl_rl_ppo_camera_cfg.py` | Targets rsl_rl's OLD single-class ActorCritic API (confirmed to match every tagged release through 2.3.0). But Isaac Lab's `main` branch already pins `rsl-rl-lib==5.0.1` and has a NATIVE `RslRlCNNModelCfg`/`RslRlMLPModelCfg` path that would replace this entire custom class — confirmed to exist, but its multi-observation-group routing (image group vs. proprioceptive group) isn't confirmed. **Check for `RslRlCNNModelCfg` first in Phase 1** before using this file — see its docstring for the concrete cfg shape to try. |
| `cone_nav/mdp/events.py` | `randomize_sensor_mount_pose` mutates a sensor prim's transform post-creation via raw `pxr` — may need a re-initialize call depending on Isaac Sim version, or may need to move to a static per-scene-build offset instead of per-episode. Not researchable further without a live Kit process to test against. |
| `cone_nav/env_cfg_lidar.py` | `MultiMeshRayCasterCfg` confirmed to exist (see above), but `RaycastTargetCfg`'s exact field semantics (`is_shared`, `merge_prim_meshes`) are a best-effort guess, not confirmed. |
| `scripts/train.py` / `play.py` | `extras["log"]` termination-rate key names (e.g. `Episode_Termination/success`) are still an assumed format, not confirmed against a real run — print one dict and check before trusting `play.py`'s printed success rate. |

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
      observations.py         policy obs (sensor + proprioceptive only — no privileged state)
      rewards.py               potential-based approach shaping, success-with-hold, collision, smoothness
      terminations.py          cone collision, success, out-of-bounds (time_out is Isaac Lab's built-in)
      events.py                course regeneration, actuation latency/slip, sensor mount jitter, background clutter
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
  test_mdp_math.py            unit tests for mdp/_pure_math.py -- torch only, actually run (15/15 passing)
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
