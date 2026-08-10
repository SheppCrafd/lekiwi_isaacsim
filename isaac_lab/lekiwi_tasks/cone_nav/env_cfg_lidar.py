"""Lidar-variant env cfg. See env_cfg_base.py for everything sensor-agnostic."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

from . import mdp
from .env_cfg_base import LekiwiConeNavEnvCfgBase, LekiwiSceneCfgBase
from ..robots.lekiwi import LEKIWI_LIDAR_CFG

# Must match isaac_sim/attach_sensors.py's LIDAR_SCAN_HEIGHT_LOCAL exactly -- both are
# "block height + RPLIDAR base height + partway into the cap" measured the same way
# against the same lekiwi_lidar.usd geometry (isaac_sim/README_lekiwi_variants.md).
LIDAR_SCAN_HEIGHT_LOCAL = 0.061


@configclass
class LekiwiLidarSceneCfg(LekiwiSceneCfgBase):
    robot = LEKIWI_LIDAR_CFG

    # RPLIDAR A1M8 real specs (Slamtec's own official datasheet, also
    # isaac_sim/attach_sensors.py): 360 deg FOV, <=1deg angular resolution -- 360 rays
    # at 1deg spacing approximates that (the datasheet's ~400 samples/rotation @ 5.5Hz
    # is a bit finer; 360 is a clean round number close enough for a first pass, tune
    # later if needed). Unlike the camera variant's optics (env_cfg_camera.py), this is
    # a real manufacturer-published spec for the exact part actually being used, not
    # borrowed from an unrelated reference -- highest-confidence sensor spec in this
    # whole project, position AND stats both real.
    #
    # Uses MultiMeshRayCasterCfg, not the plain RayCasterCfg an earlier version of this
    # file used -- confirmed via source research (isaaclab.sensors changelog/docs), not
    # a guess: plain RayCaster's mesh data is loaded once at sensor init and "only works
    # for literally static meshes" (its own docs' wording), which would have silently
    # never seen the cones at all -- they're per-env dynamic RigidObjects, moved every
    # episode by mdp/events.py:regenerate_course. MultiMeshRayCaster was added
    # specifically to support raycasting against tracked moving meshes.
    # STILL UNVERIFIED: the exact semantics of RaycastTargetCfg's `is_shared` /
    # `merge_prim_meshes` fields (below, `is_shared=False` guessed for "each env's cone
    # is independently posed, not literally shared mesh data" -- confirm against your
    # installed version rather than trusting this guess). If MultiMeshRayCaster doesn't
    # exist in your installed version at all (older Isaac Sim), fall back to
    # isaac_sim/attach_sensors.py's attach_rplidar_rtx()/attach_rplidar_physx() runtime
    # attachment instead of this declarative SensorCfg.
    rplidar: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_assembly",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, LIDAR_SCAN_HEIGHT_LOCAL)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=12.0,  # RPLIDAR A1M8 max range
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground", is_shared=True, track_mesh_transforms=False
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/cone_.*", is_shared=False, track_mesh_transforms=True
            ),
        ],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Phase 6 "sensor noise, spelled out per sensor" -- all three listed items now
        # implemented: range noise (RPLIDAR A1M8's own datasheet doesn't publish a
        # distance-noise std, so +/-2cm gaussian is a plausible-for-a-cheap-2D-lidar
        # starting number, not a datasheet spec) via Isaac Lab's stock noise cfg;
        # dropped returns (2% per-ray) and angular jitter (0.5deg std, modeled as a
        # whole-scan circular shift) via mdp/observations.py:lidar_ranges's own params,
        # since Isaac Lab's stock noise cfgs can't express either (see that function's
        # docstring) -- unit-tested in tests/test_mdp_math.py (the underlying corruption
        # math, not the ObsTerm wiring itself, which still needs Isaac Sim to exercise).
        ranges: ObsTerm = ObsTerm(
            func=mdp.observations.lidar_ranges,
            params={"dropout_prob": 0.02, "angular_jitter_std_deg": 0.5},
            noise=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        )
        base_pose: ObsTerm = ObsTerm(func=mdp.observations.base_pose_2d)
        base_velocity: ObsTerm = ObsTerm(func=mdp.observations.base_velocity_2d)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True  # all three are flat vectors -- fine to concat into one observation for the [64,64] MLP (agents/rsl_rl_ppo_lidar_cfg.py)

    policy: PolicyCfg = PolicyCfg()


@configclass
class LekiwiLidarConeNavEnvCfg(LekiwiConeNavEnvCfgBase):
    scene: LekiwiLidarSceneCfg = LekiwiLidarSceneCfg(num_envs=2500, env_spacing=9.0)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_sensor_mount = EventTerm(
            func=mdp.events.randomize_sensor_mount_pose,
            mode="reset",
            params={"sensor_prim_relpath": "Robot/lidar_assembly"},
        )
