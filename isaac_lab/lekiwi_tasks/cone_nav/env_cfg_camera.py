"""Camera-variant env cfg. See env_cfg_base.py for everything sensor-agnostic."""

from __future__ import annotations

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
    # output against the real X10 (re-checked directly against Seeed's own product page
    # https://www.seeedstudio.com/X10-USB-wired-camera-p-6506.html, a reseller page, and
    # Seeed's own LeRobot wiki -- all three still lack FOV/focal-length/sensor-size
    # data, confirming this isn't solvable by more searching):
    #   - POSITION: photo-verified against Seeed's real product hero photo
    #     (README_lekiwi_variants.md) -- reasonable confidence.
    #   - SHAPE/SIZE (bracket + camera body geometry): placeholder boxes, no real X10
    #     mesh exists anywhere -- low confidence, dimensionally approximate.
    #   - RESOLUTION: real spec IS confirmed -- Seeed's own product page names the
    #     part "X10 USB Camera 1080p," i.e. native 1920x1080. height=480/width=640
    #     below is a DELIBERATE downsample, not an unverified guess like the two items
    #     above -- 2500 parallel envs rendering RGB at native 1080p (~6.75x the pixel
    #     throughput of 640x480) would very likely blow the project's own A100 40GB
    #     VRAM budget (plan.md Phase 3/4's already-flagged camera-variant memory
    #     concern) for no training benefit a policy actually needs. This matches
    #     `isaac_lab/deploy/lekiwi_policy_runner.py`'s real-hardware capture path
    #     (`CameraSensorReader(width=640, height=480)`, requesting that resolution
    #     directly from the UVC driver rather than capturing 1080p and downsampling in
    #     software) -- sim and real deployment both operate at 640x480 as the actual
    #     resolution the policy was trained on and will run against, so there's no
    #     train/inference resolution mismatch even though it's below the sensor's max.
    #   - OPTICS (focal_length=36.5mm, horizontal_aperture=36.83mm, baked into the USD
    #     Camera prim): copied from LightwheelAI/leisaac's TiledCameraCfg for an
    #     unrelated camera, NOT the real Seeed X10's actual specs -- lowest confidence
    #     of everything here. These two numbers actually produce ~53.5deg horizontal
    #     FOV (2*atan(36.83/(2*36.5)) = 2*atan(0.5045) = 53.5deg -- the standard
    #     USD/photographic aperture-and-focal-length formula), corrected 2026-08-10 from
    #     a prior version of this comment that claimed "~75deg" -- that number didn't
    #     match these values under the same formula and nothing else in this codebase
    #     depends on 75deg specifically (grepped: only this comment ever said it), so
    #     it was simply wrong arithmetic, not a differently-sourced figure. Fixing the
    #     comment rather than retargeting the numbers to hit 75deg, since no real X10
    #     FOV exists to target either way -- see below.
    #     Re-verified 2026-08-10 across five independent sources (two more than the
    #     three checked in the prior pass): Seeed's product page, a reseller page,
    #     Seeed's own LeRobot/LeKiwi wiki (all as before), PLUS this time the actual
    #     X10 datasheet PDF (Seeed's own "Industrial Product Datasheet", fetched and
    #     read directly, not skimmed as HTML) and a second reseller (OpenELAB). The
    #     datasheet's full spec table is: Product name, Operating Temperature (0-40C),
    #     Communication Interface (USB), Applications, Part List, and compliance
    #     HSCODEs -- literally nothing optical. Same conclusion as before, now on
    #     firmer ground: there is genuinely no published FOV/focal-length/sensor-size
    #     data for this camera anywhere, not a search gap. Best available stand-in, not
    #     a placeholder to feel bad about -- just don't mistake it for a verified
    #     number (real OR now-corrected-math) if e.g. tuning reward shaping against
    #     expected visible range.
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
        # per-pixel noise (UniformNoiseCfg below) covers pixel-VALUE read noise.
        # mdp.observations.front_camera_rgb itself now ALSO applies live, speed-scaled
        # mechanical shake (pixel-content shift) and motion blur (temporal frame
        # blend) -- see that function's docstring for the full reasoning and its
        # still/moving default magnitudes. Only lens distortion and compression
        # artifacts remain genuinely unbuilt after this change, not the whole
        # "motion blur/lens distortion/compression artifacts" list this comment used
        # to flag as one undifferentiated gap.
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
        # Companion reset for front_camera_rgb's live motion-blur temporal buffer --
        # see mdp.events.reset_camera_frame_state's docstring. Camera-variant only,
        # the lidar cfg never registers this.
        self.events.reset_camera_frame = EventTerm(
            func=mdp.events.reset_camera_frame_state,
            mode="reset",
        )
        # "Anything the robot might endure" pass (2026-08-10): draws this episode's
        # glare direction/harshness, lens smudge (if any), exposure/white-balance
        # operating point, and dead-pixel positions -- see
        # mdp.events.randomize_camera_defects's own docstring for each default range.
        self.events.randomize_camera_defects = EventTerm(
            func=mdp.events.randomize_camera_defects,
            mode="reset",
        )
        # Camera rendering is far more memory-hungry per-env than the lidar variant's
        # plain range data (plan.md Phase 3/4's explicit open question) -- start with
        # the same 2500 default as the base cfg, but this is the first thing to drop if
        # Phase 2's real VRAM usage doesn't fit an A100 40GB with all 2500 rendering.
