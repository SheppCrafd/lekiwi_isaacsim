"""
Run this INSIDE Isaac Sim (Script Editor, or `python.sh this_file.py` with the
stage already open) AFTER referencing usd/lekiwi_camera.usd or
usd/lekiwi_lidar.usd into your scene (Create > Payload/Reference, or
add_reference_to_stage in a standalone script).

These two USD files were built directly from the real, community-maintained
LeKiwi Isaac Sim asset (LightwheelAI/leisaac_env on HuggingFace,
assets/robots/lekiwi.usd -- the asset behind leisaac's own LEKIWI_CFG), with
the arm stripped out (and for the lidar variant, the base camera mesh too).
Real physics/articulation/materials/collision groups all came from that
asset -- see README_lekiwi_variants.md for the full provenance.

lekiwi_camera.usd already has a REAL UsdGeom.Camera prim baked in at
/LeKiwi/base/front_camera (real position + optics, copied from leisaac's own
verified TiledCameraCfg for this exact mount point) -- attach_front_camera()
below just wraps it to get a live Isaac Sim sensor handle, it does not need
to create anything.

lekiwi_lidar.usd has a real RigidBody + Collision cylinder at
/LeKiwi/rplidar_a1m8 sized to the actual Slamtec A1M8 (70mm diameter, 170g),
fixed-jointed to /LeKiwi/base the same way the asset's own 3 wheels are
fixed-jointed -- that's the physical mount, not the sensor. The RTX/PhysX
Lidar SENSOR schema (OmniLidar / OmniSensorGenericLidarCoreAPI) is a Kit
extension-internal schema, not part of core pip-installed USD, and its core
scan attributes (FOV/range/resolution namespace) could not be fully verified
against NVIDIA's docs from outside a running Isaac Sim -- so the sensor
itself is created here via Isaac Sim's own command API (guaranteed correct
for whatever's actually installed) rather than hand-authored in the USD.

Isaac Sim's lidar API has moved twice: PhysX SDK RangeSensor Lidar (now
DEPRECATED as of Isaac Sim 6.0, replaced by
isaacsim.sensors.experimental.physics.RaycastSensor) -> RTX Lidar via
IsaacSensorCreateRtxLidar (current, GPU-accelerated, what NVIDIA recommends
for new projects). This script tries RTX first, falls back to the deprecated
PhysX path for older installs, and notes the RaycastSensor path for very new
ones -- check your Google Cloud image's Isaac Sim version and pick the one
that actually exists in it.

Real hardware specs used below:
  Slamtec RPLIDAR A1M8 (datasheet rev 3.0, 2020-10-15):
    - 360 deg field of view, range 0.15m-12m
    - scan rate: 5.5Hz typical, up to 10Hz max
    - angular resolution: <=1 deg (at 5.5Hz / 400 samples per rotation)
    - sample rate: ~8000Hz (up to 8010Hz)
"""

LEKIWI_LIDAR_MOUNT_PRIM = "/LeKiwi/rplidar_a1m8"
LEKIWI_FRONT_CAMERA_PRIM = "/LeKiwi/base/front_camera"


def attach_rplidar_rtx(lidar_parent_prim: str = LEKIWI_LIDAR_MOUNT_PRIM):
    """Attach a real RTX Lidar configured to match RPLIDAR A1M8 specs."""
    import omni.kit.commands

    _, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/rplidar_a1m8_sensor",
        parent=lidar_parent_prim,
        config="Slamtec_RPLIDAR_A1",  # falls back to Example_Rotary if this
        # profile isn't in your Isaac Sim build's sensor config library --
        # in that case use attach_rplidar_physx() below instead, or clone
        # Example_Rotary.json and edit the fields to match the specs above.
        translation=(0, 0, 0),
        orientation=(1, 0, 0, 0),
    )
    return sensor


def attach_rplidar_physx(lidar_parent_prim: str = LEKIWI_LIDAR_MOUNT_PRIM):
    """Fallback for Isaac Sim builds before 6.0 (PhysX SDK Lidar deprecated at 6.0)."""
    from omni.isaac.sensor import RotatingLidarPhysX

    lidar = RotatingLidarPhysX(
        prim_path=f"{lidar_parent_prim}/rplidar_a1m8_sensor",
        name="rplidar_a1m8",
        rotation_frequency=10.0,        # Hz, A1M8 max scan rate
        fov=(360.0, 0.0),               # full 360 deg horizontal, no vertical spread (2D lidar)
        resolution=(1.0, 0.0),          # ~1 deg angular resolution
        valid_range=(0.15, 12.0),       # meters, per datasheet
    )
    lidar.initialize()
    lidar.add_range_data_to_frame()
    lidar.add_point_cloud_data_to_frame()
    lidar.enable_visualization()
    return lidar


def attach_rplidar_raycast_experimental(lidar_parent_prim: str = LEKIWI_LIDAR_MOUNT_PRIM):
    """
    For Isaac Sim 6.0+, where PhysX SDK Lidar was removed. NVIDIA's own docs
    named this as the replacement but did not show a full usage example in
    what was fetchable outside a running Isaac Sim -- check
    isaacsim.sensors.experimental.physics.RaycastSensor's actual constructor
    signature in your installed version before trusting this call verbatim.
    """
    from isaacsim.sensors.experimental.physics import RaycastSensor

    return RaycastSensor(
        prim_path=f"{lidar_parent_prim}/rplidar_a1m8_sensor",
        rotation_frequency=10.0,
        fov=(360.0, 0.0),
        resolution=(1.0, 0.0),
        valid_range=(0.15, 12.0),
    )


def attach_front_camera(camera_prim: str = LEKIWI_FRONT_CAMERA_PRIM):
    """
    Wraps the REAL camera prim already baked into lekiwi_camera.usd -- does
    not create new geometry. Position/optics were already authored directly
    on the USD Camera prim (copied from leisaac's own verified
    TiledCameraCfg), so this just gets you a live sensor handle to pull
    rgb/depth frames from during a script or training loop.
    """
    from omni.isaac.sensor import Camera

    camera = Camera(
        prim_path=camera_prim,
        resolution=(640, 480),  # matches leisaac's own TiledCameraCfg for this exact mount
    )
    camera.initialize()
    return camera


if __name__ == "__main__":
    # For lekiwi_lidar.usd:
    try:
        attach_rplidar_rtx()
    except Exception as e:
        print(f"RTX lidar attach failed ({e}), trying deprecated PhysX lidar")
        try:
            attach_rplidar_physx()
        except Exception as e2:
            print(f"PhysX lidar also failed ({e2}), trying experimental RaycastSensor (Isaac Sim 6.0+)")
            attach_rplidar_raycast_experimental()

    # For lekiwi_camera.usd:
    attach_front_camera()
