# LeKiwi: Sim Assets → Trained Policy → Physical Robot

Full path from where we are right now to a real LeKiwi robot running a policy trained in Isaac Sim. Every phase notes what's already done, what's genuinely open/undecided, and where money or hardware becomes unavoidable.

**Current state (updated 2026-08-09):** `usd/lekiwi_camera.usd` derived from a real verified LightwheelAI/leisaac asset, arm removed, real physics/articulation. Two real bugs found by comparing Blender renders against Seeed's own product photo and fixed directly in the USD: a baked 180°-Z root rotation that mirrored the whole robot (paired wheels/single wheel were on the wrong sides), and a wrong GoPro-pedestal camera mount replaced with a photo-verified flush bracket + repositioned camera sensor prim. Wheel geometry itself was audited via forward kinematics against the CAD-derived URDF and found already correct — not changed. See `isaac_sim/README_lekiwi_variants.md` for the full writeup.

`usd/lekiwi_lidar.usd` was **deleted and rebuilt from scratch** this session (the earlier lidar variant predated the camera fixes above and was never brought in line with them) — no camera, a real RPLIDAR A1M8 mount designed from Slamtec's own datasheet dimensions on a new 3D-printable block (`urdf/meshes/lidar_mount_block_v1.stl`), real base-plate hole positions confirmed by cross-sectioning the actual mesh (not assumed from spec). Full design writeup, including the tradeoff of offsetting the mount off dead-center to clear the existing compute-tower, in `isaac_sim/README_lekiwi_variants.md`.

Both variants verified via raw USD inspection + real (Store-installed, GUI-only) Blender renders so far. **Neither has ever been opened in an actual running Isaac Sim** — that's the next step and the first thing that can still fail in a way we haven't seen.

**Task decided:** procedurally-generated cone-avoidance navigation. ~1,000,000 procedurally generated ugly environments, each guaranteeing at least 100 sq ft of open space with a cone arrangement and a success area inside it. Randomized **once, at generation time, when each of the 1M environments is created** — none of this changes during an episode: surroundings, materials, cone size/shape, cone positions (a small random offset per cone around the canonical arrangement, baked into that environment's layout — still just one more thing the generator varies, not movement during a run), robot starting (x, y) and starting **heading**, sensor deficiencies, voltage/wattage spikes. Episode ends on cone collision (fail) or the robot's full footprint inside the success area (success). Reward: small shaping for approaching the success area, one large reward for fully succeeding. Trained on Google Cloud spot GPU ($1-2/hr tier) via Isaac Sim / Isaac Lab.

**Two training runs, not one:** a full 50-hour session for the camera variant and a separate full 50-hour session for the lidar variant — two independently trained policies, not a single run that picks one sensor. This resolves what was previously an open decision (Phase 3 used to ask "camera or lidar" — it's now "both, separately"). Budget accordingly: roughly 100 GPU-hours total.

**"Most powerful $1-2/hr spot GPU" — checked against real current pricing, not assumed:** on Google Cloud, spot A100 40GB runs ~$1.10/hr (down from $3.67/hr on-demand); A100 80GB likely lands near the top of the $1-2/hr range. H100-class hardware does **not** fit this budget even on spot (~$3.30/hr minimum) — so the honest ceiling for "$1-2/hr" is an A100, not literally the most powerful GPU Google Cloud sells. Total for both 50-hour runs at that tier: roughly $110-190, before any spot-preemption wall-clock overhead.

---

## Phase 0 — Close out current sim-asset verification

- [ ] Re-render both variants with `blender_verify_12_views.py` (now covers `lekiwi_camera.usd` and the rebuilt `lekiwi_lidar.usd`) and confirm visually against Seeed's product photo: wheel layout no longer mirrored, camera bracket reads as a flush low-profile box (not a pedestal), lidar mount block + RPLIDAR sit correctly on the -Y side of the base without clipping the compute tower.
- [ ] The 12-view silhouette renders are flat/unlit (a pre-existing quirk of the render script, not fixed this session) — good enough for layout/shape checks but not fine surface detail. Worth fixing the lighting setup if finer visual verification is needed later.
- [ ] Decide whether to keep `urdf/lekiwi_cam.urdf` / `urdf/lekiwi_lidar.urdf` (the hand-built, physically-simulated-wheel URDFs) around as a documented alternative, or archive them — they're not the source for the USD deliverables, but the lidar mount's design (block position, base-plate hole grid, RPLIDAR dimensions) was worked out and documented in the lidar URDF first, so it's currently load-bearing documentation, not just an alternative.
- [ ] `lidar_mount_block_v1.stl` needs an actual test print before trusting the fit — the base-plate hole grid was verified from the real mesh and the RPLIDAR dimensions are from Slamtec's own datasheet, but only 2 of the RPLIDAR's 4 stock mounting holes were used (the other 2 sat in an illegible part of the datasheet's mechanical drawing) and the block's placement was shifted off dead-center to clear the compute tower — both worth confirming physically before ordering hardware at volume.

## Phase 1 — Get real Isaac Sim access on a Google Cloud spot GPU

You were explicit: no payment info for *me*, nothing on your own machine (can't handle it) or mine (doesn't exist). This phase is where that changes — you're paying for this yourself, deliberately, at the $1-2/hr spot tier.

- [ ] Pick the actual spot GPU SKU — **A100 40GB is the realistic target** (verified spot pricing ~$1.10/hr, see above); confirm current price/availability in your chosen region at request time, since spot pricing floats day to day.
- [ ] Install Isaac Sim (pip-installable `isaacsim` package for 4.x+, or the Omniverse Launcher install) on the instance, plus Isaac Lab for the RL environment/training stack this task needs.
- [ ] Confirm the Isaac Sim version, since it changes which lidar sensor API is live — `attach_sensors.py` already branches for this (RTX Lidar → deprecated PhysX Lidar → Isaac Sim 6.0+ `RaycastSensor`).
- [ ] **Checkpointing + auto-resume before any real training run.** Spot instances get preempted with short notice — without periodic checkpoint saves (model + optimizer state) and a resume-on-restart path, a reclaim mid-run loses training progress and the money already spent on it. This matters even more across two separate 50-hour sessions than it would for one.

## Phase 2 — First real load (the tests I could never actually run)

- [ ] Load `usd/lekiwi_camera.usd`, press Play. Confirm: doesn't explode/jitter on spawn, articulation reports correctly in the Stage/Physics panels, `base_x`/`base_y`/`base_theta` actually move the robot when commanded (not the wheel joints — those are fixed/decorative, documented in `isaac_sim/README_lekiwi_variants.md`).
- [ ] Confirm the `front_camera` prim produces a real RGB image.
- [ ] Load `usd/lekiwi_lidar.usd`, press Play, same stability check. Run `isaac_sim/attach_sensors.py` and confirm real scan data comes back (ranges, not zeros/NaNs).
- [ ] Fix whatever this step reveals — something will. Every prior "verification" in this project was static (USD inspection, Blender stills); this is the first time real PhysX and real sensor pipelines actually touch these files.

## Phase 3 — Decisions still required before the cone task is buildable

- [ ] **Build two separate environment/training configs, one per sensor** — camera and lidar are each getting their own full 50-hour run, so this isn't a single shared config with a sensor flag; each needs its own observation space and policy architecture (CNN encoder for camera, MLP/1D-conv for lidar ranges). Note this is still two *separate-sensor* policies, not one policy using both at once — a combined camera+lidar variant still doesn't exist and would be a third, additional build.
- [ ] **Fix the rotation-randomization axis.** "x-y rotation (not z)" is very likely backwards for a wheeled ground robot: randomize **yaw (Z)** for starting heading, keep roll/pitch (X/Y) at zero — the robot sits flat on a floor at spawn. Randomizing X/Y rotation spawns it already tipped over, which isn't a valid starting pose.
- [ ] **Decide what the policy is actually allowed to observe.** The "closer to success area" reward can use privileged ground-truth simulator position (rewards are allowed to cheat) — but the policy's *observations* cannot include ground-truth (x, y) distance-to-goal, because the real robot has no such signal available at deployment. The policy has to navigate from only what its actual sensor (camera or lidar) can perceive. This needs to be an explicit, separate wiring: privileged state → reward function only; sensor data (+ proprioceptive `base_x/y/theta` state) → policy observation.
- [x] **Physics timestep, control frequency, RL library, and network architecture — settled by copying `MuammerBay/isaac_so_arm101`** (a real, working Isaac Lab RL project for the SO-ARM101, the same arm the broader leisaac ecosystem is built around; GitHub, BSD-3-Clause). Verified by pulling its actual `reach_env_cfg.py` / `rsl_rl_ppo_cfg.py`, not guessed:
  - `sim.dt = 1/60s` (60Hz physics), `decimation = 2` → policy acts every 2 physics steps, i.e. a **30Hz control rate**. Still need to confirm this against the real LeKiwi's actual control loop rate once hardware is in hand (Phase 9) — copying SO-101's number is a reasonable starting point, not a guarantee it matches LeKiwi's firmware.
  - RL library: **rsl_rl** (`RslRlOnPolicyRunnerCfg` / `RslRlPpoActorCriticCfg` / `RslRlPpoAlgorithmCfg`).
  - PPO hyperparameters: `num_steps_per_env=24`, `learning_rate=1e-3` (adaptive schedule), `clip_param=0.2`, `entropy_coef=0.001`, `num_learning_epochs=8`, `num_mini_batches=4`, `gamma=0.99`, `lam=0.95` (GAE), `desired_kl=0.01`, `max_grad_norm=1.0`.
  - Network: `actor_hidden_dims=[64, 64]`, `critic_hidden_dims=[64, 64]`, `elu` activation — a plain 2-layer MLP. **This directly applies to the lidar variant** (its observations — ranges + `base_x/y/theta` state — are a flat vector, same shape of problem as SO-101's joint-state observations).
  - **Camera variant CNN — no SO-101 reference exists for this** (its reach task has no camera at all, pose/joint-state only), so per "pick what you think is best": use the standard "Nature CNN" encoder (3 conv layers, channels 32→64→64, strides 4/2/1, ReLU activations, flatten) ahead of the same `[64, 64]` actor/critic MLP heads used above. This is the long-established default for RL-from-pixels (originating from the DQN paper, still the common baseline choice in rl_games/skrl/rsl_rl camera setups) — not a novel choice, just not one with a direct SO-101 citation to point at.
  - `num_envs` — SO-101's reference used 4096, but landed on **2500 (a 50×50 grid)** instead, deliberately smaller. This resolves the earlier "25 environments is too few" flag either way, but 2500 doesn't automatically solve the camera-rendering VRAM question raised when checking real A100 pricing — it's a reasonable, more conservative starting point, not a guarantee the camera variant fits in memory at that count on an A100 40GB. Validate empirically in Phase 1/4 (actually launch it and watch VRAM usage) rather than assuming 2500 is small enough; drop further if it isn't. `env_spacing = 2.5` from the same reference.

## Phase 4 — Build the Isaac Lab environment

- [x] **Parallel environment count — 2500 (50×50 grid)**, a deliberately smaller number than the SO-101 reference's 4096 (Phase 3). Confirm in Phase 1/2 that the chosen A100 instance actually holds 2500 environments in memory, especially for the camera variant (rendering RGB for thousands of parallel envs is far more memory-hungry than the lidar variant's plain range data) — the two 50-hour runs may still end up needing different env counts if the camera variant can't fit 2500, even though it's more conservative than 4096.
- [ ] **Per-env isolation in the vectorized layout.** With many cloned environments running simultaneously, each needs proper origin offsetting and collision-group separation so one env's cones/robot don't physically or (for the camera variant) visually bleed into a neighboring env's frame.
- [ ] Write the procedural environment generator: guarantees ≥100 sq ft open space, places the cone arrangement and success area, randomizes surroundings/materials/cone size+shape/cone position offset (see Phase 6 for exact scope) — all of this happens once, when a given environment instance is generated, not during a run.
- [ ] **Reset-safety checks in the generator** — validate every generated instance before use:
  - Robot never spawns already overlapping a cone (instant, meaningless fail).
  - Robot never spawns already inside the success area (instant, meaningless success).
  - The success area is actually reachable given wherever the cones landed (procedural generation can produce unsolvable layouts without an explicit check) — **more important now that cone positions also vary**, since an offset cone could land in a position that blocks the only path to the goal even if the canonical arrangement was solvable.
- [ ] Write the scene config (`InteractiveSceneCfg`) wiring the robot USD, chosen sensor, and generated environment.
- [ ] Write the observation config per the Phase 3 privileged-vs-real-sensor split above.
- [ ] Write the action config: `base_x`/`base_y`/`base_theta` velocity or position targets.

## Phase 5 — Reward and termination design

Termination conditions as specified (cone hit = fail, full-footprint-in-success-area = success) are missing pieces needed to actually train:

- [ ] **Episode timeout / truncation.** No max-episode-length exists yet. Without one, an episode that never hits a cone or reaches the goal never resets, which breaks batched on-policy training (PPO needs bounded episodes).
- [ ] **Out-of-bounds termination.** What happens if the robot drives outside the 100 sq ft area without touching a cone or the goal? Needs its own fail condition, or those episodes just burn to timeout uselessly.
- [ ] **Explicit negative reward on cone collision.** A reset alone is weak signal — add a real penalty so failure is distinguishable from "ran out of time," not just inferred from episode length.
- [ ] **Action-smoothness / energy penalty.** Without one, policies commonly learn jerky, high-acceleration control that's fine in sim but doesn't survive real motor/torque limits, and is harder on the real hardware even when it "succeeds."
- [ ] **Potential-based approach shaping, not raw distance.** `reward = distance_last_step − distance_this_step`, not `reward = -distance`. Raw-distance shaping is a known reward-hacking magnet — the agent can learn to oscillate near the goal boundary collecting incremental reward instead of ever finishing.
- [ ] **Define "success" precisely.** Full robot footprint (collision bounding box) entirely inside the success-area polygon — should this also require near-zero velocity / a brief hold, or does instantaneous full-containment count? Without a hold requirement, a policy could learn to blast through the area and still register success, which won't be a usable real-world stop-and-park behavior.
- [ ] **Decide whether cones are static/kinematic or dynamic rigid bodies.** Real traffic cones are light and get knocked over/moved on contact — a rigid static cone behaves differently under near-misses than dynamic ones would, and changes what "hitting a cone" even means physically (position-overlap check vs. real contact-force event). **Worth being explicit this is separate from Phase 6's cone-position randomization** — that's a one-time offset baked into each generated environment's starting layout; this item is about whether cones can *additionally* move during a run because the robot bumped one (real physics, dynamic bodies), which is a different, optional feature, not something implied by the position randomization existing.

## Phase 6 — Domain randomization spec

What's already decided (surroundings, materials, cone size/shape, start x/y, sensor deficiencies, voltage/wattage spikes) needs each item made concrete, plus these additions:

- [ ] **Cone position offset — new item, generation-time only, not real-time movement.** The original task spec described a fixed canonical cone arrangement (only size/shape varied between environments); cone *positions* now also get randomized as part of the same one-time generation step every other randomized property already goes through — a small offset per cone around the canonical arrangement, baked into that environment before the episode ever starts, not a fully independent re-layout each time and not anything that changes once an episode is running. Needs an actual magnitude decided (e.g. ± some cm/inches per cone, sampled independently per cone so the arrangement's general shape survives but individual gaps between cones shrink/widen slightly) — "changing a bit" isn't yet a number. This is a second, meaningfully different randomization axis from cone size/shape below and should be sampled independently of it, not conflated into one "cone randomization" knob.
- [ ] **Cone size and shape randomization**, already decided in the original task spec but still needs the actual ranges specified — e.g. a real min/max radius and height range for size, and which discrete shape variants exist (all real traffic cones, or also stand-ins like small pylons/barrels) for shape. "Randomized" isn't yet a number either.
- [ ] **Sensor noise, spelled out per sensor** — "random sensor deficiencies" is a category, not a spec:
  - Camera: motion blur, exposure/white-balance jitter, lens distortion, compression artifacts.
  - Lidar: range noise, dropped/missing returns, angular jitter.
- [ ] **Sensor mount position/orientation randomization — new item.** Small random offset to where the camera/lidar sits and how it's oriented relative to its nominal mount transform (a few mm of translation, a degree or two of tilt), sampled once at environment-reset time like the other DR properties, not moving during an episode. This models real assembly tolerance (the physical camera bracket/lidar mount won't sit at the exact CAD-nominal position every time it's built or reattached) and gives the policy some robustness to it, rather than training against a sensor pose that's pixel/ray-perfect every single episode. Needs an actual magnitude decided per sensor, same as the other DR items above.
- [ ] **Actuation/control latency randomization.** Real motor control loops have delay the sim doesn't have by default; training with randomized latency meaningfully helps transfer.
- [ ] **Wheel-ground and floor friction randomization** — not just visual floor materials, the actual physics friction coefficient. This matters more than usual here because the sim's locomotion is a virtual `base_x/y/theta` planar joint (found this session), not physically-simulated wheel-ground contact — real wheel slip is something the sim never naturally produces, so it likely needs to be injected as artificial noise into the action-to-motion mapping during training, or transfer will suffer on real floors.
- [ ] **Curriculum, not full randomization from step 0.** Consider annealing in obstacle density/domain-randomization range progressively (e.g. NVIDIA's Automatic Domain Randomization pattern) rather than starting training against the full 1,000,000-environment variety immediately — full randomization from scratch is a much harder learning problem than a staged ramp-up. **The cone-position-offset range is a good candidate to widen gradually too** (start environments with a near-zero offset range, widen it over training) rather than generating at the full offset magnitude from step 0 — this is still about how spread-out the *pool of generated environments* is over time, not anything moving within a single run.

## Phase 7 — Train (two runs)

- [ ] Run 1: camera-variant policy, 50-hour spot session. Kick off the PPO run with the chosen library/hyperparameters (Phase 3), scaled parallel env count (Phase 4), and full reward/termination/DR spec (Phases 5-6).
- [ ] Run 2: lidar-variant policy, separate 50-hour spot session, same rigor.
- [ ] Track real metrics during each run, not just the reward curve: episode success rate, collision rate, average episode length — needed to actually know when training has converged/plateaued vs. just producing a rising number.
- [ ] Expect to revisit reward shaping and DR ranges at least once per sensor — these are two independent training problems (different observation noise characteristics, different failure modes) even though the task and reward design are shared, not a single kickoff-and-wait.
- [ ] This is real GPU-hours on a metered spot instance, twice — the checkpointing from Phase 1 is what makes 100 total hours affordable to iterate on rather than a single expensive all-or-nothing shot.

## Phase 8 — Evaluate in sim before touching real hardware

- [ ] Roll out each trained policy on a **held-out set of generated environments/seeds, separate from training** — currently missing entirely. Without this there's no way to measure real generalization vs. training-curve reward, for either policy.
- [ ] Check for the obvious failure signatures per policy: overfitting to training-distribution room layouts, brittleness to lighting/camera-angle changes outside the DR range, reward hacking (agent found a way to score without doing the intended task).
- [ ] Don't skip this — deploying an untested policy to real hardware is how you damage the robot or waste the first real-world attempts on bugs that were cheap to catch in sim.

## Phase 9 — Acquire the physical robot

Nothing physical has been ordered yet — this whole project has been sim assets only. **See `BoM.md` for verified component links, pricing, and stock status as of August 2026.**

- [ ] Order the Seeed LeKiwi Kit (mobile base, 3D printed parts, battery) — **$179.00 @ [Seeed Studio JP](https://jp.seeedstudio.com/mobile-base-c-2676.html)** (pre-order/limited at most US retailers)
- [ ] Order Raspberry Pi 5 8GB (bare board) — **$175–$200 @ [Adafruit](https://www.adafruit.com/product/5813)** (in stock; official MSRP $95, but retail pricing reflects high demand)
- [ ] Order Seeed X10 USB Camera 1080p (front RGB sensor) — **$12.99 @ [Seeed Studio](https://www.seeedstudio.com/X10-USB-wired-camera-p-6506.html)** (in stock)
- [ ] Order RPLIDAR A1M8-R6 360° LiDAR (12m range 2D scanner) — **$119.98 @ [Walmart](https://business.walmart.com/ip/RPLIDAR-A1M8-2D-360-Degree-12-Meters-Scanning-Radius-LIDAR-Sensor-Scanner-for-Obstacle-Avoidance-and-Navigation-of-Robots/14747563594)** (in stock; cheapest verified option with free US shipping)
- [ ] **3D print `urdf/meshes/lidar_mount_block_v1.stl`** (the RPLIDAR mount, 110×80×25mm) — normal ~15-20% infill, not solid (solid would be ~245g in PLA). Order the mount hardware too: M3 + M2.5 heat-set inserts, M3 x 12mm and M2.5 x 8mm screws, and a soldering-iron insert-tip kit (not a heat gun) — full verified links in `BoM.md`'s "Lidar Mount Hardware" section, ~$55-75 total.
- [ ] Assemble per Seeed's own build instructions (their wiki confirmed the base webcam mount, Pi mounting location, etc. — already cross-referenced against the sim assets this session).
  - Seeed Wiki: https://wiki.seeedstudio.com/
- [ ] Install LeRobot + the robot's control firmware/software stack on the Pi.
  - LeRobot / LeKiwi software (GitHub): https://github.com/huggingface/lerobot
- [ ] **First real cost checkpoint that isn't cloud-GPU-shaped**: actual money for actual hardware (~$487–$512 + shipping/taxes), whenever you're ready for it — not a blocker for Phases 1-8, which all happen in sim first.

## Phase 10 — Sim-to-real transfer

- [ ] Calibrate the real camera/lidar against what the sim assets assumed (real X10 FOV/mount angle vs. the sim's copied optics, real A1M8 scan parameters vs. whatever `attach_sensors.py` actually configured in Phase 2).
- [ ] Export the trained policy in a runtime the Pi can actually execute (ONNX/TorchScript, or LeRobot's own deployment path).
- [ ] Build the real-time inference loop on-robot: sensor read → policy forward pass → `base_x/y/theta`-equivalent motor command translation (the real robot's firmware does its own wheel inverse-kinematics from vx/vy/omega, matching how the sim's virtual planar joint was set up — should translate directly, but is the first point where the sim/real command interfaces actually have to agree).

## Phase 11 — Real-world testing and iteration

- [ ] First runs: tethered/supervised, low-stakes physical space, kill-switch ready. A cone-collision policy trained purely on the reward as specified has no explicit incentive to approach slowly — verify real approach speed near cones/goal is actually safe for the hardware before running unsupervised.
- [ ] Expect the sim-to-real gap to show up as *something* — lighting differences, real sensor noise, real wheel slip the virtual planar joint never modeled, latency the sim loop didn't have. Normal, not a sign the earlier phases were wrong.
- [ ] Iterate: more real-world fine-tuning data, reward/observation adjustments, or another sim training round with whatever the real-world gap revealed.

---

## Open decisions this plan still can't make for you

- Whether 2500 parallel environments (Phases 1 & 4) actually fits in an A100 40GB's VRAM for the camera variant specifically — settled as a starting number, not confirmed to actually run yet; only real testing in Phase 2 answers this.
- Exact reward magnitudes (how "small" vs. how "large") and whether success requires a velocity/hold condition (Phase 5) — needs real tuning, not just relative sizing.
- Exact cone-position-offset magnitude and exact cone size/shape ranges (Phase 6) — both are now "randomized" in principle but not yet a number.
- Whether you want a camera+lidar combined variant on top of the two separate-sensor policies — doesn't exist yet, would be new build work.
