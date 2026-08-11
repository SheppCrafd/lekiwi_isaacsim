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
    #   - OPTICS (focal_length=36.5mm, horizontal_aperture=73.0mm, vertical_aperture=
    #     54.75mm, baked into the USD Camera prim): focal_length is still the number
    #     originally copied from LightwheelAI/leisaac's TiledCameraCfg for an unrelated
    #     camera, NOT the real Seeed X10's actual spec -- lowest confidence of
    #     everything here, unchanged. horizontal_aperture and vertical_aperture were
    #     DELIBERATELY CHANGED 2026-08-11 (were 36.83mm / 15.29mm, ~53.5deg horizontal /
    #     23.7deg vertical FOV) to 73.0mm / 54.75mm -- exactly 90deg horizontal FOV
    #     (2*atan(73.0/(2*36.5)) = 2*atan(1.0) = 90deg) with vertical_aperture set to
    #     match this sensor's actual 640x480 (4:3) render resolution
    #     (54.75 = 73.0 * 480/640), which the OLD values never did either (15.29/36.83 =
    #     0.415, nowhere near 480/640 = 0.75 -- a real latent aspect-ratio bug inherited
    #     from the borrowed reference camera's own different resolution, silently
    #     stretching the image, fixed as part of this same change since touching
    #     horizontal_aperture meant touching this relationship regardless).
    #
    #     Reasoning for 90deg specifically: no real X10 FOV spec exists to target either
    #     way (see the re-verification below, unchanged), so the old ~53.5deg was never
    #     more "correct" than any other number -- it was just an accident of which
    #     unrelated camera got borrowed from. 90deg was chosen to (a) match the lidar
    #     variant's FOV, narrowed to the same forward-facing 90deg window the same day
    #     (env_cfg_lidar.py) -- removing FOV as an uncontrolled difference between the
    #     two sensor variants, even though they're trained/evaluated as fully
    #     independent policies and were never required to match -- and (b) directly fix
    #     the standing complaint that ~53.5deg is genuinely tiny for a nav task. This is
    #     a SIMULATION-ONLY change -- the real physical Seeed X10 still has whatever FOV
    #     it actually has (still unknown, still unpublished); seeing a real replacement
    #     webcam that's ACTUALLY ~90deg is a separate, physical-hardware decision (see
    #     plan.md Phase 9 / BoM.md for whether that was pursued and what was found).
    #     Re-verified 2026-08-10 across five independent sources (two more than the
    #     three checked in the prior pass): Seeed's product page, a reseller page,
    #     Seeed's own LeRobot/LeKiwi wiki (all as before), PLUS this time the actual
    #     X10 datasheet PDF (Seeed's own "Industrial Product Datasheet", fetched and
    #     read directly, not skimmed as HTML) and a second reseller (OpenELAB). The
    #     datasheet's full spec table is: Product name, Operating Temperature (0-40C),
    #     Communication Interface (USB), Applications, Part List, and compliance
    #     HSCODEs -- literally nothing optical. Same conclusion as before, now on
    #     firmer ground: there is genuinely no published FOV/focal-length/sensor-size
    #     data for this camera anywhere, not a search gap.
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
