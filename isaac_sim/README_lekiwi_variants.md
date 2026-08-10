# lekiwi_camera.usd / lekiwi_lidar.usd

Two Isaac Sim-ready robot assets, each the LeKiwi mobile base with **no arm**:

- `usd/lekiwi_camera.usd` — base + wheels + a real base-mounted camera sensor.
- `usd/lekiwi_lidar.usd` — base + wheels + a real RPLIDAR A1M8 mount (own design, real
  Slamtec datasheet dimensions), no camera.

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

## Corrections made against the real Seeed Kit, camera variant (2026-08-09)

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

3. **Wheel geometry — the standard is the real Seeed kit, not CAD-source consensus.**
   This one flip-flopped during investigation, so the reasoning is worth keeping, not
   just the conclusion:
   - `urdf/lekiwi_cam.urdf`, this USD (leisaac-derived), and a third independent
     source — [`kabilankb/lekiwi_isaacsim`](https://github.com/kabilankb/lekiwi_isaacsim)
     (this project's own upstream)'s real `urdf/lekiwi/configuration/lekiwi_base.usd`,
     a genuine 35MB Isaac-Sim URDF import, not a stub — **all three agree** on wheel
     mount radii of 99.2mm / 100.0mm / 119.2mm. Three-way agreement looks like strong
     evidence, and it was initially read that way.
   - It isn't. All three trace back to the same shared, generic, open-source LeKiwi
     CAD lineage. Three-way agreement only proves a common ancestor, not that the
     ancestor matches what Seeed actually ships — and we already have a proven
     counterexample: that same upstream source also contains the wrong `Camera_Mount_v8`
     GoPro-pedestal mesh (see fix #2), which every one of these three sources shares
     and which definitively does *not* match Seeed's real product photo. Shared CAD
     lineage has already been shown, on this exact project, to diverge from the real
     shipped kit.
   - Directly measuring Seeed's own product photo (circle-fit the disc edge and the
     black wheel-roller pixels) put the real back wheel's closest approach to center
     at 93.4% of the disc radius vs. this shared-CAD model's 109.8% — a real error,
     direction confirmed, but the same photo-measurement method gave a nonsensical
     left/right split on the front wheels (87% vs 60%, which shouldn't exist in a
     mirror-symmetric design) — real noise from photo perspective, compression, and
     each wheel's mount bracket projecting differently at its own 120°-rotated
     orientation, precise enough to trust the *direction* of an error but not an exact
     per-wheel target number.
   - **Final call (2026-08-10): explicit design target, not a per-wheel photo
     measurement.** All three wheels' closest-approach-to-center set to exactly
     **118mm** (`wheel_left`/`wheel_right`/`wheel_back` Xform translate, and each
     matching `fix_*_wheel` joint's `physics:localPos0`, updated together — same
     requirement as every other wheel-position change in this file). Radial scale
     only, per wheel's own existing angle preserved exactly, so the 120° kiwi-drive
     symmetry isn't disturbed. Mount origin and closest-mesh-point-to-center were
     verified to coincide exactly for all three wheels before scaling (same value to
     the mm), which is what makes a clean radial scale valid here instead of an
     approximation. Verified after: 118.000 / 117.998 / 117.995mm — within 5µm of
     target on all three, effectively exact.
   - **The drivetrain hardware between the base plate and each wheel needed the same
     scale, and was missed on the first pass.** `drive_motor_mount_v11*` and
     `ST3215_Servo_Motor_v1*` (visual + collision, 12 prims total) are separate
     prims parented under `/LeKiwi/base`, not children of the `wheel_*` Xforms —
     moving only `wheel_left`/`wheel_right`/`wheel_back` left these floating at their
     old radius while the wheel moved out to 118mm, disconnecting the visual
     drivetrain chain. Fixed by applying each wheel's own scale factor to its
     matching motor+mount pair (matched by naming: `_2`→back, no suffix→left,
     `_1`→right, confirmed by comparing their angular position to each wheel's).
     Verified after with an independent render (matplotlib+trimesh, outside
     Blender/USD's own renderer): motor mount → servo → wheel mount → wheel body
     forms one continuous connected chain from the deck edge outward, no gaps.
   - The "front wheels phasing into the plate" visual impression that started this
     whole investigation traces to the separately-fixed 180°-mirror bug, confirmed by
     a tight zoomed-in screenshot showing nothing beyond the normal look of a
     wheel-mounted-near-the-edge design in a flat top-orthographic view with no depth
     cues — not a wheel-geometry bug, and not touched.
   - The upstream USD's `/LeKiwi/joints` scope is empty and nothing in it has
     `PhysicsRigidBodyAPI` applied — pure visual/geometry, no articulation — so it's a
     reference point for checking geometry, not a candidate to replace this project's
     physics-ready asset.

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

## Lidar (lekiwi_lidar.usd), built from scratch (2026-08-09)

Unlike the camera variant (which started from leisaac's real asset), there was no
pre-existing lidar mount to fix — this was designed from nothing, using Slamtec's own
official RPLIDAR A1M8 datasheet
(`LD108_SLAMTEC_rplidar_datasheet_A1M8_v3.0_en.pdf`) and this USD's own real geometry,
not eyeballed.

**The literal robot center is occupied — by something the first attempt at this got
wrong.** The first version of this design checked for collisions against
`urdf/lekiwi_cam.urdf`'s `Base_08q_v1` compute-mount tower and, thinking that was the
only obstruction, placed the block at `(0, -0.044)`. That URDF is stale and not a
reliable source for this USD's actual geometry, and worse, `Base_08q_v1` doesn't even
exist in this USD's lineage (verified directly: no such prim). The real obstruction was
found by exhaustively bbox-checking every mesh in the file above deck height: it's
`Bottom_V2_v3_visual` + `Top_V2_v2_visual` — the robot's motor-controller board (a real
PCB-detail mesh, "M" logo and vent hatching, matching the panel visible in Seeed's own
product photo) — at `x[-0.046, 0.0474] y[-0.104, -0.0338] z[0.057, 0.086]`. That's the
**-Y side**, the opposite side from where the first attempt was checking. The block is
now at `(0, 0.015)` — only 15mm off dead-center, not 44mm — clearing the real
obstruction by a verified 8.8mm margin. Checked three ways before trusting it: a
pairwise 3D bbox-overlap test (`False`), and two independent renders of the actual USD
mesh geometry (matplotlib+trimesh, entirely outside Blender/USD's own renderer) from
multiple angles, confirming a visible gap between the two parts.

**The mounting block** (`urdf/meshes/lidar_mount_block_v1.stl`, generated with
`trimesh` + `manifold3d`, not hand-modeled) is a plain box, 110 × 80 × 25mm:
- Height (25mm) was **71.6mm in the first two passes** — matched to
  `urdf/lekiwi_cam.urdf`'s `Base_08q_v1` tower height, a real measurement but from that
  same unreliable URDF, for a tower that doesn't exist in this USD and that the block
  no longer sits near anyway (it's beside the real obstruction now, not stacked above
  it). Once repositioned, the only real requirement is clearing the tallest thing
  actually on the robot — the controller board at 29mm above the deck — which the
  RPLIDAR's own 51mm height does on its own with room to spare. 25mm is sized for the
  insert bosses (8mm/6mm deep) and real structural thickness, not a tower-matching
  target. Total assembly height above the deck: 75mm (down from 122.6mm in the
  71.6mm-block version).
- 4× M3 bottom mounting holes (for heat-set inserts, pressed in from the block's
  underside) at `(±40, 0)` and `(±40, 40)` in the base plate's frame — real
  positions on the base plate's actual hole grid, confirmed by cross-sectioning the
  real `base_plate_layername_vname.stl` mesh with `trimesh` (not assumed from the
  "20mm grid" spec alone): 95 holes detected, confirming a clean 3.5mm-hole, 20mm-pitch
  grid spanning ±100mm in X and Y, centered on the plate's own origin. All 4 chosen
  holes are clear of the plate's structural standoffs and the servo controller mount.
- A shallow 3mm registration pocket (99 × 73mm) sized to the RPLIDAR A1M8's real
  96.74 × 70.28mm base footprint, for lateral registration, plus 2× M2.5 insert holes
  (28mm off the unit's center axis, 56mm apart) matching the ONE stock mounting-hole
  pair read with confidence off the datasheet's mechanical drawing (Figure 5-2). The
  drawing shows 4 stock holes total, but the other 2 sit in a part of the drawing that
  wasn't legible enough to trust for a real printed part — the pocket carries the
  lateral registration on its own, so 2 screws is still a secure mount, just not the
  full 4-point stock pattern.

**The RPLIDAR A1M8 itself, corrected 2026-08-09 (second pass).** The first version used
a plain box for the base and centered the cap on the whole footprint — wrong on both
counts, caught by re-tracing Slamtec's own side-view drawing (Figure 5-2) carefully
instead of skimming it:
- **Base (0–21mm local)**: not a box. It's a "paddle" — a Ø70mm circle (the main body)
  unioned with a ~Ø32mm circular boss (the motor spindle/connector housing) extending
  toward one end, reaching the real 96.74mm total length. Built as an actual boolean
  union of two cylinders (`trimesh` + `manifold3d`), not approximated as a box.
  Main-circle center derived directly from the drawing's own "35.14mm from the left
  edge" dimension (35.14 ≈ the circle's own 35mm radius — its left edge sits almost
  exactly at the footprint's left edge). Boss sized to the drawing's Ø32 spec and
  positioned so its right edge lands exactly on the real 96.74mm length.
- **Cap (21–51mm local)**: a plain Ø70mm cylinder, correct as before, but now positioned
  concentric with the *main circle only* (offset toward one end of the footprint) —
  not centered on the full 96.74×70.28mm bounding box like the first pass had it. This
  matches the real unit: the round rotating head sits over the main body, not over the
  connector boss.
- 190g total mass (datasheet MISC spec), split proportionally by the *actual* mesh
  volumes of the paddle-base and the cap (not a box-vs-cylinder approximation) —
  86.3g / 103.7g. Mass/COM/inertia for the whole assembly (block + base + cap) computed
  programmatically from real mesh volumes and the parallel-axis theorem, not by hand.

This is still a real simplification of the true, more detailed 2-tier housing (screw
bosses, connector nub, exact fillets aren't modeled) — but the topology and major
dimensions now genuinely match the datasheet, not just the overall envelope.

**Physics setup mirrors the wheels' pattern exactly**: `/LeKiwi/lidar_assembly` is one
rigid body (`PhysicsRigidBodyAPI` + `PhysicsMassAPI`, combined mass/COM/inertia across
all 3 parts) with `visuals`/`collisions` children (the latter with
`PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI`, `approximation=convexDecomposition`),
welded to `/LeKiwi/base` via a new `PhysicsFixedJoint` (`fix_lidar_assembly`) — the
same structure as `fix_back_wheel`/`fix_left_wheel`/`fix_right_wheel`.

**This authors the physical mount, not the sensor's scan behavior.** The RTX/PhysX
Lidar sensor schema (`OmniLidar` prim, `OmniSensorGenericLidarCoreAPI`) is a Kit
extension-internal schema, not part of core pip-installed USD — its core scan
attributes (FOV/range/resolution namespace) could not be fully verified against
NVIDIA's docs from outside a running Isaac Sim, so guessing exact raw attribute names
was avoided. Run `isaac_sim/attach_sensors.py` inside Isaac Sim after loading this USD
to attach the real working sensor via Isaac Sim's own command API (which is guaranteed
correct for whatever version is actually installed) — it tries RTX Lidar first, falls
back to the deprecated PhysX SDK Lidar, and notes the Isaac Sim 6.0+ `RaycastSensor`
replacement. Real hardware specs used there: 360° FOV, 0.15–12m range, 5.5Hz typical
scan rate (10Hz max), ≤1° angular resolution — all from Slamtec's own datasheet.

**What's still an approximation, worth knowing before you print:**
- The block and RPLIDAR housing are placeholder-accuracy geometry (primitives and
  boolean unions of primitives), not exact meshes — same caveat as the camera variant's
  bracket. The RPLIDAR shape is dimensionally correct (real circle sizes/positions from
  the datasheet) but doesn't model fine details like screw bosses or the connector nub.
- Only 2 of the RPLIDAR's 4 stock mounting holes are used (see above).
- Mass/inertia for the new parts are reasonable estimates (box/cylinder formulas,
  volume-proportional mass split), not measured.
- The block's infill matters for real-world weight — it's ~198cm³ of material if
  printed solid (roughly 245g in solid PLA); print at normal (~15-20%) infill unless
  there's a specific reason to go solid.
- The block/rplidar-base/rplidar-cap stack is authored with a deliberate 0.5-1mm
  overlap at each junction rather than an exact touching seam — exactly-coincident
  flat faces between separate mesh objects z-fight in Blender's lit viewport (visible
  as a garbled/glitchy look at the seam) even when the underlying geometry is
  completely valid; invisible at print scale, purely a rendering-cleanliness fix.

## Manual steps on the Google Cloud Isaac Sim instance

1. Copy `usd/lekiwi_camera.usd` and `usd/lekiwi_lidar.usd` to the instance. Each is
   fully self-contained (flattened — no external file references, all meshes/materials
   embedded), so nothing else needs to travel with them.
2. Reference or open directly in Isaac Sim (`Isaac Utils` → open the USD, or
   `add_reference_to_stage()` in a standalone script). Default prim is `/LeKiwi`.
3. For `lekiwi_lidar.usd`, run `isaac_sim/attach_sensors.py` inside Isaac Sim once the
   stage is loaded to get a real working lidar sensor (see above — geometry alone
   isn't a functioning sensor).
4. For `lekiwi_camera.usd`, the camera is already fully authored — either read it
   directly via a standard USD Camera / render product, or call
   `attach_front_camera()` from the same script for a ready-made
   `omni.isaac.sensor.Camera` handle.
5. To drive the robot: set joint position/velocity targets on `base_x`, `base_y`,
   `base_theta` for locomotion (the 3 physical omni-wheels are fixed/decorative, see
   below — not the wheel joints), and on the arm joints if you re-add an arm later.
6. If you're using IsaacLab specifically (not plain Isaac Sim), consider using
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
model would need to be built instead (this project also has hand-built URDF-derived
versions with real 3-wheel drivetrain joints — `urdf/lekiwi_cam.urdf` /
`urdf/lekiwi_lidar.urdf` — which take that different, physically-simulated-wheel
approach, though it wasn't carried through to the physics-authored USDs).
