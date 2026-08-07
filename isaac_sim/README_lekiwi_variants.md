# lekiwi_camera.usd / lekiwi_lidar.usd

Two Isaac Sim-ready robot assets, each the LeKiwi mobile base with **no arm**:

- `usd/lekiwi_camera.usd` — base + wheels + a real base-mounted camera sensor.
- `usd/lekiwi_lidar.usd` — base + wheels + a Slamtec RPLIDAR A1M8 mount, no camera.

## What was found vs. what was built

**Found and used as the real source**, not built from scratch: a genuine, actively
maintained, Apache-2.0 Isaac Sim/Isaac Lab asset for LeKiwi —
[`LightwheelAI/leisaac`](https://github.com/LightwheelAI/leisaac) (GitHub) with its
binary robot USD hosted at
[`LightwheelAI/leisaac_env`](https://huggingface.co/LightwheelAI/leisaac_env/tree/main)
on HuggingFace (`assets/robots/lekiwi.usd`, 35.7MB). This is the asset behind
leisaac's own `LEKIWI_CFG` (`source/leisaac/leisaac/assets/robots/lerobot.py`), used
for real LeKiwi teleoperation and imitation-learning training in IsaacLab.

Verified before trusting it:
- Its 6 arm joint limits (`shoulder_pan`: ±110°, `shoulder_lift`: ±100°, `elbow_flex`:
  -100°/90°, `wrist_flex`: ±95°, `wrist_roll`: ±160°, `gripper`: -10°/100°) exactly match
  the same repo's own `SO101_FOLLOWER_USD_JOINT_LIMLITS` — independent confirmation
  it's the genuine, correctly-matched asset, not a stale or mismatched one.
- Structurally inspected directly with `pxr` (not just skimmed): real
  `ArticulationRootAPI` on `/LeKiwi`, a real `/physicsScene` (gravity, PhysX TGS
  solver, CCD enabled), real `PhysicsRevoluteJoint`/`PhysicsFixedJoint`/
  `PhysicsPrismaticJoint` prims with real `DriveAPI` gains, real collision groups,
  real PBR materials — not placeholder geometry.

**Built from this source**: both variants are that real asset with the entire SO-101
arm surgically removed (6 joints, 6 rigid bodies, their visual/collision/mesh groups —
`shoulder`, `upper_arm`, `lower_arm`, `wrist`, `gripper`, `jaw`), verified afterward with
zero dangling joint references and the correct remaining joint/rigid-body counts. The
camera variant then got a real `UsdGeom.Camera` prim added; the lidar variant also had
the original base-camera mesh removed and a RigidBody+Collision RPLIDAR mount added.

**Not found anywhere**: a pre-built no-arm, camera-or-lidar-only LeKiwi variant. NVIDIA's
own official Isaac Sim robot asset registry does not list LeKiwi at all. The full-robot
asset above was the closest real prior art, hence deriving from it rather than building
the whole articulation from scratch.

## Important architectural fact (not obvious, worth knowing before scripting)

**The 3 physical omni-wheels are not simulated at all.** They're attached to `base` via
`PhysicsFixedJoint`s (`fix_back_wheel`, `fix_left_wheel`, `fix_right_wheel`) — rigid,
non-rotating, decorative/collision-only geometry. Real locomotion instead comes from a
virtual 3-DOF planar joint stack at the root: `world` → (`base_x`, prismatic X) →
(`base_y`, prismatic Y) → (`base_theta`, revolute Z) → `base`, each with very stiff
drive gains (stiffness 1e6, damping 1e4, unlimited range) that make it behave like a
directly-commanded velocity interface rather than a passive dynamic joint.

This is the real production pattern for training holonomic-base policies (matches
LeKiwi's actual firmware, which takes `vx`/`vy`/`omega` body-frame commands and does its
own wheel inverse-kinematics on the physical robot — the sim doesn't need to replicate
that inner loop). **To drive the robot, command `base_x`/`base_y`/`base_theta` joint
targets or velocities directly** — not per-wheel commands. If your training setup
specifically needs individually-simulated omni-wheel rolling physics (e.g. for
wheel-slip research), this asset is not that; say so and a different, wheel-physics-based
model would need to be built instead (this project also has a hand-built
URDF-derived version with real 3-wheel drivetrain joints — `urdf/lekiwi_cam.urdf` /
`urdf/lekiwi_lidar.urdf` — which takes that different, physically-simulated-wheel
approach, though it wasn't carried through to a physics-authored USD).

## Camera (lekiwi_camera.usd)

A real `UsdGeom.Camera` prim at `/LeKiwi/base/front_camera` — position and optics copied
directly from leisaac's own verified `TiledCameraCfg` for this exact mount point
(`source/leisaac/leisaac/tasks/template/lekiwi_env_cfg.py`), not derived independently:

- Position `(0.0, 0.13, 0.025)`, orientation quaternion `(0.64279, 0.76604, 0.0, 0.0)`
  (wxyz) relative to `/LeKiwi/base`.
- `focal_length=36.5mm`, `horizontal_aperture=36.83mm` (~75° horizontal FOV),
  `clipping_range=(0.01, 50.0)`.
- Recommended render resolution `640×480 @ 30fps` (matches the reference config;
  resolution/fps are render-product settings, not attributes on the Camera prim itself).

The base camera's original visual/collision mesh (`Camera_Mount_v8` / `Camera_Model_v3`
— the real Seeed-matching bracket+webcam geometry from the original CAD) is preserved
unchanged from the source asset.

## Lidar (lekiwi_lidar.usd)

A real `RigidBody` + `Collision` cylinder at `/LeKiwi/rplidar_a1m8`, fixed-jointed to
`/LeKiwi/base` the same way the asset's own 3 wheels are (`fix_rplidar_a1m8`), sized to
the actual Slamtec RPLIDAR A1M8 (datasheet rev 3.0, 2020-10-15): 70mm disc diameter
(0.035m radius), 170g mass, mounted at `z=0.16m` — checked against the base body's own
real bounding box (max z = 0.138m across its whole footprint) for genuine clearance, not
eyeballed. The original base camera mesh was removed for this variant.

**This authors the physical mount, not the sensor's scan behavior.** The RTX/PhysX Lidar
sensor schema (`OmniLidar` prim, `OmniSensorGenericLidarCoreAPI`) is a Kit
extension-internal schema, not part of core pip-installed USD — its core scan attributes
(FOV/range/resolution namespace) could not be fully verified against NVIDIA's docs from
outside a running Isaac Sim, so guessing exact raw attribute names was avoided. Run
`isaac_sim/attach_sensors.py` inside Isaac Sim after loading this USD to attach the real
working sensor via Isaac Sim's own command API (which is guaranteed correct for whatever
version is actually installed) — it tries RTX Lidar first, falls back to the deprecated
PhysX SDK Lidar, and notes the Isaac Sim 6.0+ `RaycastSensor` replacement.

## Manual steps on the Google Cloud Isaac Sim instance

1. Copy `usd/lekiwi_camera.usd` and `usd/lekiwi_lidar.usd` to the instance. Each is
   fully self-contained (flattened — no external file references, all meshes/materials
   embedded), so nothing else needs to travel with them.
2. Reference or open directly in Isaac Sim (`Isaac Utils` → open the USD, or
   `add_reference_to_stage()` in a standalone script). Default prim is `/LeKiwi`.
3. For `lekiwi_lidar.usd`, run `isaac_sim/attach_sensors.py` inside Isaac Sim once the
   stage is loaded to get a real working lidar sensor (see above — geometry alone isn't
   a functioning sensor).
4. For `lekiwi_camera.usd`, the camera is already fully authored — either read it
   directly via a standard USD Camera / render product, or call
   `attach_front_camera()` from the same script for a ready-made
   `omni.isaac.sensor.Camera` handle.
5. To drive the robot: set joint position/velocity targets on `base_x`, `base_y`,
   `base_theta` for locomotion (not the wheel joints, which are fixed/inert — see
   above), and on the arm joints if you re-add an arm later.
6. If you're using IsaacLab specifically (not plain Isaac Sim), consider using
   leisaac's own `LEKIWI_CFG`-style `ArticulationCfg` wrapper instead of/alongside these
   raw USD files — its `ImplicitActuatorCfg` actuator gains (e.g. arm stiffness=12.8,
   damping=1.2) override whatever's baked into the USD at scene-load time, so if you
   want IsaacLab's own tuned gains rather than this asset's baked-in ones, go through
   that config path.
