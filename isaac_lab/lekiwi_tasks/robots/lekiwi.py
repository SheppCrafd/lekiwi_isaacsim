"""
ArticulationCfg for the two arm-free LeKiwi variants built in usd/lekiwi_camera.usd
and usd/lekiwi_lidar.usd. See isaac_sim/README_lekiwi_variants.md for how those USDs
were built and verified -- this file only wires them into Isaac Lab.

Written against Isaac Lab 2.x namespaces (isaaclab.*). If your Google Cloud image has
an older install, the equivalent APIs live under omni.isaac.lab.* instead -- same
shapes, different import path (plan.md Phase 1 asks you to confirm the Isaac Sim
version for exactly this reason; attach_sensors.py already branches the same way for
the lidar sensor API).

The only real articulated DOF on either variant is the 3-joint planar stack
(base_x prismatic, base_y prismatic, base_theta revolute) -- the 3 physical omni-wheels
are PhysicsFixedJoints (rigid welds, not real USD Physics joints/DOFs), so they don't
appear here at all. Stiffness/damping below (1e6 / 1e4) mirror the drive gains already
baked into the USD (README_lekiwi_variants.md, "Important architectural fact") so the
Isaac Lab articulation's implicit actuator behaves the same way the raw USD does if
loaded outside Isaac Lab: an effectively-velocity-commanded interface, not a springy
passive joint.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Repo layout: isaac_lab/lekiwi_tasks/robots/lekiwi.py -> ../../../usd/
LEKIWI_ASSETS_DIR = os.environ.get(
    "LEKIWI_ASSETS_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd")),
)

# Both USDs were flattened at build time (no external file references, everything
# embedded) per README_lekiwi_variants.md's "Manual steps" section -- copying just the
# .usd file to the training instance is enough, nothing else needs to travel with it.
LEKIWI_CAMERA_USD_PATH = os.path.join(LEKIWI_ASSETS_DIR, "lekiwi_camera.usd")
LEKIWI_LIDAR_USD_PATH = os.path.join(LEKIWI_ASSETS_DIR, "lekiwi_lidar.usd")

_BASE_JOINT_NAMES = ["base_x", "base_y", "base_theta"]

# Drive gains copied from the USD's own baked PhysicsDriveAPI values (see
# README_lekiwi_variants.md) rather than re-derived, so Isaac Lab's implicit actuator
# reproduces the same "stiff enough to behave like a commanded velocity interface"
# response the raw USD already has.
_BASE_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=_BASE_JOINT_NAMES,
    stiffness=1.0e6,
    damping=1.0e4,
    effort_limit=1.0e5,
    velocity_limit=2.0,  # m/s (base_x/base_y) / rad-s^-1 (base_theta) -- generous placeholder, tune against Phase 9's real firmware vx/vy/omega limits once hardware exists
)

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
)

_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
)

# Spawn a few cm off the floor, matching the base's own local origin offset baked into
# the USD (the base plate isn't exactly at Z=0 in the source asset's own frame).
_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.05),
    joint_pos={name: 0.0 for name in _BASE_JOINT_NAMES},
)

LEKIWI_CAMERA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEKIWI_CAMERA_USD_PATH,
        activate_contact_sensors=True,  # needed for the cone-collision termination (Phase 5)
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    ),
    init_state=_INIT_STATE,
    actuators={"base": _BASE_ACTUATOR},
)

LEKIWI_LIDAR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEKIWI_LIDAR_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    ),
    init_state=_INIT_STATE,
    actuators={"base": _BASE_ACTUATOR},
)

# Prim paths sensors attach to, matching isaac_sim/attach_sensors.py's constants exactly
# so the same mount points are used whether a sensor is created by that standalone
# script or wired declaratively through an Isaac Lab SensorCfg (cone_nav/env_cfg_*.py).
FRONT_CAMERA_PRIM = "front_camera"  # relative to {ENV_REGEX_NS}/Robot/base
LIDAR_MOUNT_PRIM = "lidar_assembly"  # relative to {ENV_REGEX_NS}/Robot
