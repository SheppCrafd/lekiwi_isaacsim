"""Lidar-variant env cfg. See env_cfg_base.py for everything sensor-agnostic."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

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
    # isaac_sim/attach_sensors.py): the PHYSICAL unit spins a full 360deg, <=1deg
    # angular resolution (the datasheet's ~400 samples/rotation @ 5.5Hz is a bit finer;
    # 1deg spacing is a clean round number close enough for a first pass). This is a
    # real manufacturer-published spec for the exact part actually being used, not
    # borrowed from an unrelated reference -- highest-confidence sensor spec in this
    # whole project, position AND stats both real.
    #
    # horizontal_fov_range narrowed to a forward-facing 100deg window (-50..50), NOT the
    # sensor's full 360deg physical sweep -- a deliberate 2026-08-11 decision, not a
    # hardware change. The RPLIDAR still spins the full circle; this just discards
    # everything outside the forward 100deg before the policy ever sees it. 100deg
    # (not the initial 90deg pass) matches the real Arducam IMX291 board camera's own
    # published "100 Degree Wide Angle" spec (the real front camera as of 2026-08-11,
    # see BoM.md -- it replaced the Seeed X10, whose FOV was never published anywhere).
    # env_cfg_camera.py's baked USD optics were updated to the same 100deg the same day,
    # so real camera spec = sim camera FOV = lidar FOV, all three in agreement -- not
    # just two of them arbitrarily matched to each other, which is what the initial
    # 90deg pass was (there was no real spec to target at the time). FOV was never
    # something a real head-to-head between the camera and lidar variants needed to
    # control for (they're trained and evaluated as fully independent policies, not
    # compared against each other), but matching it removes one variable from the
    # comparison. Trivially reversible -- flip back to (-180.0, 180.0) for a full-circle
    # lidar if the two variants are ever meant to be compared on sensing coverage
    # specifically, not just task performance.
    # At horizontal_res=1.0 this yields ~100 rays, not confirmed against the real
    # LidarPatternCfg's exact bin-count semantics (inclusive vs. exclusive endpoint) --
    # same "unverified until Phase 2" caveat as everything else touching a live sensor
    # cfg in this file. deploy/lekiwi_policy_runner.py and scripts/export_policy.py's
    # hardcoded ray counts were updated to match (100, not 360) -- keep all three in sync
    # if this range or horizontal_res ever changes again.
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
            horizontal_fov_range=(-50.0, 50.0),
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
        # dropped returns (2% per-ray, flat rate) and angular jitter via
        # mdp/observations.py:lidar_ranges's own params, since Isaac Lab's stock noise
        # cfgs can't express either (see that function's docstring). Angular jitter is
        # now LIVE and speed-scaled -- 0.15deg std at rest, ramping to 1.5deg by the
        # time the speed_metric hits 0.4 (roughly the robot's own max_lin_vel from
        # mdp/actions.py) -- replacing the old flat 0.5deg-regardless-of-motion
        # constant, matching front_camera_rgb's own still/moving shake magnitudes.
        # Unit-tested in tests/test_mdp_math.py (the underlying corruption math, not
        # the ObsTerm wiring itself, which still needs Isaac Sim to exercise).
        #
        # "Anything the robot might endure" pass (2026-08-10) adds two live per-step
        # misread effects on top of the above: misread_prob is a per-ray chance of a
        # spurious falsely-CLOSE reading (multipath reflection off glossy surfaces --
        # a more dangerous failure direction than dropout's falsely-far/no-return,
        # since it can look like a real obstacle). freeze_prob is a per-step,
        # per-env chance of the whole scan repeating the previous step's reading
        # verbatim (comms/firmware hiccup) -- needs env.prev_lidar_scan /
        # env.lidar_scan_valid (cone_nav_env.py), cleared each reset by
        # reset_lidar_scan_state below.
        ranges: ObsTerm = ObsTerm(
            func=mdp.observations.lidar_ranges,
            params={
                "dropout_prob": 0.02,
                "angular_jitter_still_deg": 0.15,
                "angular_jitter_moving_deg": 1.5,
                "jitter_speed_at_moving_deg": 0.4,
                "misread_prob": 0.01,
                "misread_min_ghost_frac": 0.1,
                "misread_max_ghost_frac": 0.6,
                "freeze_prob": 0.01,
                "angular_res_deg": 1.0,  # must match LidarPatternCfg(horizontal_res=...) above
            },
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
        # Companion reset for lidar_ranges' live freeze-glitch temporal buffer -- see
        # mdp.events.reset_lidar_scan_state's docstring. Lidar-variant only.
        self.events.reset_lidar_scan = EventTerm(
            func=mdp.events.reset_lidar_scan_state,
            mode="reset",
        )
