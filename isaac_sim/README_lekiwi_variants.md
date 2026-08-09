# lekiwi_camera.usd

An Isaac Sim-ready robot asset: the LeKiwi mobile base with **no arm** and a real
base-mounted camera sensor.

A RPLIDAR A1M8 variant (`lekiwi_lidar.usd`/`.urdf`) previously lived alongside this
one. It's been removed while this camera variant's geometry is being brought in line
with the real [Seeed LeKiwi Kit](https://www.seeedstudio.com/Lekiwi-Kit-p-6501.html) —
see "Corrections made against the real Seeed Kit" below. The lidar variant will come
back once that's done.

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

**Built from this source**: the SO-101 arm was surgically removed (6 joints, 6 rigid
bodies, their visual/collision/mesh groups — `shoulder`, `upper_arm`, `lower_arm`,
`wrist`, `gripper`, `jaw`), verified afterward with zero dangling joint references and
the correct remaining joint/rigid-body counts. A real `UsdGeom.Camera` prim was then
added (see below).

## Corrections made against the real Seeed Kit (2026-08-09)

A side-by-side Blender render vs. the real product photo on Seeed's own listing
surfaced two real problems, fixed directly in the USD with `pxr`:

1. **Whole robot mirrored 180°.** The root `/LeKiwi` prim had a baked
   `xformOp:orient` of `(real≈0, imaginary=(0,0,1))` — a 180° rotation about Z —
   inherited from the leisaac source asset. This put the paired front wheels at the
   back of the world frame and the single back wheel at the front, the mirror image
   of the real kit's layout (paired wheels + camera on one side, single wheel
   opposite). Fixed by zeroing that root orientation to identity. Verified after the
   fix: `wheel_left`/`wheel_right` world Y ≈ 0 to +0.12 (same side as the camera),
   `wheel_back` world Y ≈ -0.12 to -0.16 (opposite side) — matches the product photo.

2. **Camera mount and camera model were the wrong shape, in the wrong place.** The
   leisaac source's `Camera_Mount_v8` mesh is a raised, GoPro-style pedestal — the
   real Seeed kit has no such pedestal. A cropped close-up of Seeed's own hero photo
   shows the camera sitting on a plain, flush rectangular bracket **in the ~50mm gap
   between the two base plates**, not above the top plate. This project's
   `urdf/lekiwi_cam.urdf` had already independently reconstructed the correct
   bracket + camera placement from that same photo (see its `Camera_Bracket` /
   `X10_USB_Camera` links, both simple box placeholders — no accurate mesh for the
   real Seeed X10 USB Camera (the actual part, ~$13, confirmed in `plan.md`/`BoM.md`)
   was found anywhere, so boxes stand in until one is sourced). That placeholder
   design was ported into this USD:
   - `Camera_Mount_v8` / `Camera_Model_v3` (visual + collision) deactivated
     (`SetActive(False)`, not deleted — reversible).
   - New `Camera_Bracket` box (0.035 × 0.030 × 0.040m) added at
     `(0, 0.085, 0.025)` relative to `/LeKiwi/base`, and a new `X10_USB_Camera` box
     (0.025 × 0.015 × 0.020m) at `(0, 0.1075, 0.025)`, both visual + collision,
     bound to the same materials the old geometry used (`material_base_black` for
     the bracket, `material_camera` for the camera box).
   - The real `UsdGeom.Camera` sensor prim (`/LeKiwi/base/front_camera`) moved from
     `(0, 0.13, 0.025)` — copied from leisaac's `TiledCameraCfg` for a mount point
     that doesn't match this asset's actual geometry — to `(0, 0.115, 0.025)`, the
     new camera box's front (lens) face. Its orientation quaternion
     `(0.64279, 0.76604, 0, 0)` (forward + slight upward tilt) was left unchanged;
     it was independently verified to already point in a sane direction.

   These boxes are **placeholder geometry**, not an accurate mesh of the real X10
   camera or its bracket. If a real mesh turns up (Seeed hasn't published STEP/STL
   for either part as of this writing), swap it in at the same transforms.

3. **Wheel geometry was checked and found correct — not changed.** Given the visual
   mismatch, the 3 wheels' placement was suspected too. A full forward-kinematics
   pass through `urdf/lekiwi_cam.urdf`'s joint chain (base plate → drive motor mount
   → servo → wheel mount → wheel body, at zero joint angle) was compared directly
   against every wheel transform baked into this USD — outer mount pose *and* the
   inner mount-to-wheel-body offset, for all three wheels. They matched to
   floating-point precision (both position and orientation, including the wheel-body
   visual mesh's own local offset). The apparent "front wheels phasing into the
   base / back wheel sticking out too far" impression was very likely just fix #1
   (the 180° mirror) putting the large-radius back-wheel mount (radius ≈0.119m) in
   the visual "slot" where the eye expects the smaller-radius front pair
   (radius ≈0.099–0.100m), and vice versa. Re-render after fix #1 before assuming
   wheel geometry itself needs work.

## Camera (lekiwi_camera.usd)

A real `UsdGeom.Camera` prim at `/LeKiwi/base/front_camera`, position corrected as
described above (was leisaac's copied `TiledCameraCfg` offset, now the real Seeed
bracket's lens-face position):

- Position `(0.0, 0.115, 0.025)`, orientation quaternion `(0.64279, 0.76604, 0.0, 0.0)`
  (wxyz) relative to `/LeKiwi/base`.
- `focal_length=36.5mm`, `horizontal_aperture=36.83mm` (~75° horizontal FOV),
  `clipping_range=(0.01, 50.0)` — unchanged, still from leisaac's verified
  `TiledCameraCfg`.
- Recommended render resolution `640×480 @ 30fps` (matches the reference config;
  resolution/fps are render-product settings, not attributes on the Camera prim itself).

## Manual steps on the Google Cloud Isaac Sim instance

1. Copy `usd/lekiwi_camera.usd` to the instance. It's fully self-contained (flattened
   — no external file references, all meshes/materials embedded), so nothing else
   needs to travel with it.
2. Reference or open directly in Isaac Sim (`Isaac Utils` → open the USD, or
   `add_reference_to_stage()` in a standalone script). Default prim is `/LeKiwi`.
3. The camera is already fully authored — either read it directly via a standard USD
   Camera / render product, or call `attach_front_camera()` from
   `isaac_sim/attach_sensors.py` for a ready-made `omni.isaac.sensor.Camera` handle.
4. To drive the robot: set joint position/velocity targets on `base_x`, `base_y`,
   `base_theta` for locomotion (the 3 physical omni-wheels are fixed/decorative, see
   below — not the wheel joints), and on the arm joints if you re-add an arm later.
5. If you're using IsaacLab specifically (not plain Isaac Sim), consider using
   leisaac's own `LEKIWI_CFG`-style `ArticulationCfg` wrapper instead of/alongside this
   raw USD file — its `ImplicitActuatorCfg` actuator gains (e.g. arm stiffness=12.8,
   damping=1.2) override whatever's baked into the USD at scene-load time, so if you
   want IsaacLab's own tuned gains rather than this asset's baked-in ones, go through
   that config path.

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
URDF-derived version with real 3-wheel drivetrain joints — `urdf/lekiwi_cam.urdf` —
which takes that different, physically-simulated-wheel approach, though it wasn't
carried through to a physics-authored USD).
