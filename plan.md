# LeKiwi: Sim Assets → Trained Policy → Physical Robot

Full path from where we are right now to a real LeKiwi robot running a policy trained in Isaac Sim. Every phase notes what's already done, what's genuinely open/undecided, and where money or hardware [...]

**Current state:** `usd/lekiwi_camera.usd` and `usd/lekiwi_lidar.usd` exist, derived from a real verified LightwheelAI/leisaac asset, arm removed, real physics/articulation, real camera prim, real [...]

**Task decided:** procedurally-generated cone-avoidance navigation. ~1,000,000 procedurally generated ugly environments, each guaranteeing at least 100 sq ft of open space with a cone arrangement a[...]

**Two training runs, not one:** a full 50-hour session for the camera variant and a separate full 50-hour session for the lidar variant — two independently trained policies, not a single run that[...]

**"Most powerful $1-2/hr spot GPU" — checked against real current pricing, not assumed:** on Google Cloud, spot A100 40GB runs ~$1.10/hr (down from $3.67/hr on-demand); A100 80GB likely lands ne[...]

---

## Phase 0 — Close out current sim-asset verification

- [ ] Re-run `blender_verify_12_views.py`, confirm the lidar world-to-local fix actually holds in the live "Modeling" viewport (not just the flat silhouette renders, which already proved they can [...]
- [ ] Spot-check the camera variant the same way — same class of bug (world/local frame mixups) could theoretically exist wherever positions were computed relative to a bbox rather than copied v[...]
- [ ] Decide whether to keep `urdf/lekiwi_cam.urdf` / `urdf/lekiwi_lidar.urdf` (the earlier hand-built, physically-simulated-wheel approach) around as a documented alternative, or archive them —[...]

## Phase 1 — Get real Isaac Sim access on a Google Cloud spot GPU

You were explicit: no payment info for *me*, nothing on your own machine (can't handle it) or mine (doesn't exist). This phase is where that changes — you're paying for this yourself, deliberate[...]

- [ ] Pick the actual spot GPU SKU — **A100 40GB is the realistic target** (verified spot pricing ~$1.10/hr, see above); confirm current price/availability in your chosen region at request time,[...]
- [ ] Install Isaac Sim (pip-installable `isaacsim` package for 4.x+, or the Omniverse Launcher install) on the instance, plus Isaac Lab for the RL environment/training stack this task needs.
- [ ] Confirm the Isaac Sim version, since it changes which lidar sensor API is live — `attach_sensors.py` already branches for this (RTX Lidar → deprecated PhysX Lidar → Isaac Sim 6.0+ `Ray[...]
- [ ] **Checkpointing + auto-resume before any real training run.** Spot instances get preempted with short notice — without periodic checkpoint saves (model + optimizer state) and a resume-on-r[...]

## Phase 2 — First real load (the tests I could never actually run)

- [ ] Load `usd/lekiwi_camera.usd`, press Play. Confirm: doesn't explode/jitter on spawn, articulation reports correctly in the Stage/Physics panels, `base_x`/`base_y`/`base_theta` actually move t[...]
- [ ] Confirm the `front_camera` prim produces a real RGB image.
- [ ] Load `usd/lekiwi_lidar.usd`, press Play, same stability check. Run `isaac_sim/attach_sensors.py` and confirm real scan data comes back (ranges, not zeros/NaNs).
- [ ] Fix whatever this step reveals — something will. Every prior "verification" in this project was static (USD inspection, Blender stills); this is the first time real PhysX and real sensor p[...]

## Phase 3 — Decisions still required before the cone task is buildable

- [ ] **Build two separate environment/training configs, one per sensor** — camera and lidar are each getting their own full 50-hour run, so this isn't a single shared config with a sensor flag;[...]
- [ ] **Fix the rotation-randomization axis.** "x-y rotation (not z)" is very likely backwards for a wheeled ground robot: randomize **yaw (Z)** for starting heading, keep roll/pitch (X/Y) at zero[...]
- [ ] **Decide what the policy is actually allowed to observe.** The "closer to success area" reward can use privileged ground-truth simulator position (rewards are allowed to cheat) — but the p[...]
- [x] **Physics timestep, control frequency, RL library, and network architecture — settled by copying `MuammerBay/isaac_so_arm101`** (a real, working Isaac Lab RL project for the SO-ARM101, the[...]
  - `sim.dt = 1/60s` (60Hz physics), `decimation = 2` → policy acts every 2 physics steps, i.e. a **30Hz control rate**. Still need to confirm this against the real LeKiwi's actual control loop [...] 
  - RL library: **rsl_rl** (`RslRlOnPolicyRunnerCfg` / `RslRlPpoActorCriticCfg` / `RslRlPpoAlgorithmCfg`).
  - PPO hyperparameters: `num_steps_per_env=24`, `learning_rate=1e-3` (adaptive schedule), `clip_param=0.2`, `entropy_coef=0.001`, `num_learning_epochs=8`, `num_mini_batches=4`, `gamma=0.99`, `lam[...]
  - Network: `actor_hidden_dims=[64, 64]`, `critic_hidden_dims=[64, 64]`, `elu` activation — a plain 2-layer MLP. **This directly applies to the lidar variant** (its observations — ranges + `b[...]
  - **Camera variant CNN — no SO-101 reference exists for this** (its reach task has no camera at all, pose/joint-state only), so per "pick what you think is best": use the standard "Nature CNN"[...]
  - `num_envs` — SO-101's reference used 4096, but landed on **2500 (a 50×50 grid)** instead, deliberately smaller. This resolves the earlier "25 environments is too few" flag either way, but 2[...]

## Phase 4 — Build the Isaac Lab environment

- [x] **Parallel environment count — 2500 (50×50 grid)**, a deliberately smaller number than the SO-101 reference's 4096 (Phase 3). Confirm in Phase 1/2 that the chosen A100 instance actually h[...]
- [ ] **Per-env isolation in the vectorized layout.** With many cloned environments running simultaneously, each needs proper origin offsetting and collision-group separation so one env's cones/ro[...]
- [ ] Write the procedural environment generator: guarantees ≥100 sq ft open space, places the cone arrangement and success area, randomizes surroundings/materials/cone size+shape.
- [ ] **Reset-safety checks in the generator** — validate every generated instance before use:
  - Robot never spawns already overlapping a cone (instant, meaningless fail).
  - Robot never spawns already inside the success area (instant, meaningless success).
  - The success area is actually reachable given wherever the cones landed (procedural generation can produce unsolvable layouts without an explicit check).
- [ ] Write the scene config (`InteractiveSceneCfg`) wiring the robot USD, chosen sensor, and generated environment.
- [ ] Write the observation config per the Phase 3 privileged-vs-real-sensor split above.
- [ ] Write the action config: `base_x`/`base_y`/`base_theta` velocity or position targets.

## Phase 5 — Reward and termination design

Termination conditions as specified (cone hit = fail, full-footprint-in-success-area = success) are missing pieces needed to actually train:

- [ ] **Episode timeout / truncation.** No max-episode-length exists yet. Without one, an episode that never hits a cone or reaches the goal never resets, which breaks batched on-policy training [...] 
- [ ] **Out-of-bounds termination.** What happens if the robot drives outside the 100 sq ft area without touching a cone or the goal? Needs its own fail condition, or those episodes just burn to t[...]
- [ ] **Explicit negative reward on cone collision.** A reset alone is weak signal — add a real penalty so failure is distinguishable from "ran out of time," not just inferred from episode lengt[...]
- [ ] **Action-smoothness / energy penalty.** Without one, policies commonly learn jerky, high-acceleration control that's fine in sim but doesn't survive real motor/torque limits, and is harder o[...]
- [ ] **Potential-based approach shaping, not raw distance.** `reward = distance_last_step − distance_this_step`, not `reward = -distance`. Raw-distance shaping is a known reward-hacking magnet [...]
- [ ] **Define "success" precisely.** Full robot footprint (collision bounding box) entirely inside the success-area polygon — should this also require near-zero velocity / a brief hold, or does[...]
- [ ] **Decide whether cones are static/kinematic or dynamic rigid bodies.** Real traffic cones are light and get knocked over/moved on contact — a rigid static cone behaves differently under ne[...]

## Phase 6 — Domain randomization spec

What's already decided (surroundings, materials, cone size/shape, start x/y, sensor deficiencies, voltage/wattage spikes) needs each item made concrete, plus these additions:

- [ ] **Sensor noise, spelled out per sensor** — "random sensor deficiencies" is a category, not a spec:
  - Camera: motion blur, exposure/white-balance jitter, lens distortion, compression artifacts.
  - Lidar: range noise, dropped/missing returns, angular jitter.
- [ ] **Actuation/control latency randomization.** Real motor control loops have delay the sim doesn't have by default; training with randomized latency meaningfully helps transfer.
- [ ] **Wheel-ground and floor friction randomization** — not just visual floor materials, the actual physics friction coefficient. This matters more than usual here because the sim's locomotion[...]
- [ ] **Curriculum, not full randomization from step 0.** Consider annealing in obstacle density/domain-randomization range progressively (e.g. NVIDIA's Automatic Domain Randomization pattern) rat[...]

## Phase 7 — Train (two runs)

- [ ] Run 1: camera-variant policy, 50-hour spot session. Kick off the PPO run with the chosen library/hyperparameters (Phase 3), scaled parallel env count (Phase 4), and full reward/termination/D[...]
- [ ] Run 2: lidar-variant policy, separate 50-hour spot session, same rigor.
- [ ] Track real metrics during each run, not just the reward curve: episode success rate, collision rate, average episode length — needed to actually know when training has converged/plateaued [...]
- [ ] Expect to revisit reward shaping and DR ranges at least once per sensor — these are two independent training problems (different observation noise characteristics, different failure modes)[...]
- [ ] This is real GPU-hours on a metered spot instance, twice — the checkpointing from Phase 1 is what makes 100 total hours affordable to iterate on rather than a single expensive all-or-nothi[...]

## Phase 8 — Evaluate in sim before touching real hardware

- [ ] Roll out each trained policy on a **held-out set of generated environments/seeds, separate from training** — currently missing entirely. Without this there's no way to measure real general[...]
- [ ] Check for the obvious failure signatures per policy: overfitting to training-distribution room layouts, brittleness to lighting/camera-angle changes outside the DR range, reward hacking (age[...]
- [ ] Don't skip this — deploying an untested policy to real hardware is how you damage the robot or waste the first real-world attempts on bugs that were cheap to catch in sim.

## Phase 9 — Acquire the physical robot

Nothing physical has been ordered yet — this whole project has been sim assets only. **See `BoM.md` for verified component links, pricing, and stock status as of August 2026.**

- [ ] Order the Seeed LeKiwi Kit (mobile base, 3D printed parts, battery) — **$179.00 @ [Seeed Studio JP](https://jp.seeedstudio.com/mobile-base-c-2676.html)** (pre-order/limited at most US retailers)
- [ ] Order Raspberry Pi 5 8GB (bare board) — **$175–$200 @ [Adafruit](https://www.adafruit.com/product/5813)** (in stock; official MSRP $95, but retail pricing reflects high demand)
- [ ] Order Seeed X10 USB Camera 1080p (front RGB sensor) — **$12.99 @ [Seeed Studio](https://www.seeedstudio.com/X10-USB-wired-camera-p-6506.html)** (in stock)
- [ ] Order RPLIDAR A1M8-R6 360° LiDAR (12m range 2D scanner) — **$119.98 @ [Walmart](https://business.walmart.com/ip/RPLIDAR-A1M8-2D-360-Degree-12-Meters-Scanning-Radius-LIDAR-Sensor-Scanner-for-Obstacle-Avoidance-and-Navigation-of-Robots/14747563594)** (in stock; cheapest verified option with free US shipping)
- [ ] Assemble per Seeed's own build instructions (their wiki confirmed the base webcam mount, Pi mounting location, etc. — already cross-referenced against the sim assets this session).
  - Seeed Wiki: https://wiki.seeedstudio.com/
- [ ] Install LeRobot + the robot's control firmware/software stack on the Pi.
  - LeRobot / LeKiwi software (GitHub): https://github.com/huggingface/lerobot
- [ ] **First real cost checkpoint that isn't cloud-GPU-shaped**: actual money for actual hardware (~$487–$512 + shipping/taxes), whenever you're ready for it — not a blocker for Phases 1-8, which all happen in sim first.

## Phase 10 — Sim-to-real transfer

- [ ] Calibrate the real camera/lidar against what the sim assets assumed (real X10 FOV/mount angle vs. the sim's copied optics, real A1M8 scan parameters vs. whatever `attach_sensors.py` actuall[...]
- [ ] Export the trained policy in a runtime the Pi can actually execute (ONNX/TorchScript, or LeRobot's own deployment path).
- [ ] Build the real-time inference loop on-robot: sensor read → policy forward pass → `base_x/y/theta`-equivalent motor command translation (the real robot's firmware does its own wheel inve[...]

## Phase 11 — Real-world testing and iteration

- [ ] First runs: tethered/supervised, low-stakes physical space, kill-switch ready. A cone-collision policy trained purely on the reward as specified has no explicit incentive to approach slowly[...]
- [ ] Expect the sim-to-real gap to show up as *something* — lighting differences, real sensor noise, real wheel slip the virtual planar joint never modeled, latency the sim loop didn't have. N[...]
- [ ] Iterate: more real-world fine-tuning data, reward/observation adjustments, or another sim training round with whatever the real-world gap revealed.

---

## Open decisions this plan still can't make for you

- Whether 2500 parallel environments (Phases 1 & 4) actually fits in an A100 40GB's VRAM for the camera variant specifically — settled as a starting number, not confirmed to actually run yet; o[...]
- Exact reward magnitudes (how "small" vs. how "large") and whether success requires a velocity/hold condition (Phase 5) — needs real tuning, not just relative sizing.
- Whether you want a camera+lidar combined variant on top of the two separate-sensor policies — doesn't exist yet, would be new build work.
