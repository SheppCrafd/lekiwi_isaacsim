"""Camera-variant env cfg. See env_cfg_base.py for everything sensor-agnostic."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from . import mdp
from .env_cfg_base import LekiwiConeNavEnvCfgBase, LekiwiSceneCfgBase
from ..robots.lekiwi import LEKIWI_CAMERA_CFG


@configclass
class LekiwiCameraSceneCfg(LekiwiSceneCfgBase):
    robot = LEKIWI_CAMERA_CFG

    # Position/optics already baked into the USD's own Camera prim
    # (isaac_sim/README_lekiwi_variants.md) -- this SensorCfg wraps that existing prim
    # rather than authoring new optics, matching isaac_sim/attach_sensors.py's
    # attach_front_camera() (same 640x480 resolution).
    #
    # Confidence levels are NOT uniform, worth knowing before trusting this sensor's
    # output against the real X10 (re-checked directly against Seeed's own product page,
    # a reseller page, and Seeed's own LeRobot wiki -- all three lack FOV/focal-length/
    # sensor-size data, confirming this isn't solvable by more searching):
    #   - POSITION: photo-verified against Seeed's real product hero photo
    #     (README_lekiwi_variants.md) -- reasonable confidence.
    #   - SHAPE/SIZE (bracket + camera body geometry): placeholder boxes, no real X10
    #     mesh exists anywhere -- low confidence, dimensionally approximate.
    #   - OPTICS (focal_length=36.5mm, horizontal_aperture=36.83mm, ~75deg horizontal
    #     FOV, baked into the USD Camera prim): copied from LightwheelAI/leisaac's
    #     TiledCameraCfg for an unrelated camera, NOT the real Seeed X10's actual specs
    #     -- lowest confidence of the three. No published source (checked 2026-08-10)
    #     gives the X10's real FOV/focal length, so this is the best available stand-in,
    #     not a placeholder to feel bad about -- just don't mistake it for a verified
    #     number if e.g. tuning reward shaping against expected visible range.
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,  # prim already exists in the USD, don't author new optics over it
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Order matters for the CNN+MLP split in agents/rsl_rl_ppo_camera_cfg.py --
        # image first (goes through the CNN encoder), proprioceptive state after (goes
        # straight to the MLP head). Ground-truth goal/cone state deliberately absent --
        # see mdp/observations.py's module docstring.
        # Phase 6 "sensor noise, spelled out per sensor": +/-12 (of 255) uniform
        # per-pixel noise is a real number in place of the vague "motion blur,
        # exposure/white-balance jitter, lens distortion, compression artifacts" list --
        # but it's only a rough stand-in for that whole list, not each item individually.
        # Motion blur/lens distortion/compression artifacts are structured, correlated
        # corruptions a single elementwise noise op can't produce; those would need
        # dedicated image-augmentation code, not implemented here. Flagged as a real
        # gap, not silently approximated as "done."
        image: ObsTerm = ObsTerm(
            func=mdp.observations.front_camera_rgb,
            noise=UniformNoiseCfg(n_min=-12.0, n_max=12.0, operation="add"),
        )
        base_pose: ObsTerm = ObsTerm(func=mdp.observations.base_pose_2d)
        base_velocity: ObsTerm = ObsTerm(func=mdp.observations.base_velocity_2d)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False  # image needs its own tensor shape, not flattened in with the 4+3-dim proprioceptive vector

    policy: PolicyCfg = PolicyCfg()


@configclass
class LekiwiCameraConeNavEnvCfg(LekiwiConeNavEnvCfgBase):
    scene: LekiwiCameraSceneCfg = LekiwiCameraSceneCfg(num_envs=2500, env_spacing=9.0)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_sensor_mount = EventTerm(
            func=mdp.events.randomize_sensor_mount_pose,
            mode="reset",
            params={"sensor_prim_relpath": "Robot/base/front_camera"},
        )
        # Camera rendering is far more memory-hungry per-env than the lidar variant's
        # plain range data (plan.md Phase 3/4's explicit open question) -- start with
        # the same 2500 default as the base cfg, but this is the first thing to drop if
        # Phase 2's real VRAM usage doesn't fit an A100 40GB with all 2500 rendering.
