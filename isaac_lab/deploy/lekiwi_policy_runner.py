"""
On-robot real-time inference loop skeleton (plan.md Phases 10-11). Runs on the
Raspberry Pi 5 once actual hardware exists (Phase 9 -- nothing physical has been
ordered yet, see BoM.md) and a policy has passed Phase 8 held-out eval.

Base-control wiring is now against the REAL `lerobot.robots.lekiwi.LeKiwi` class
(huggingface/lerobot, fetched directly from source, not guessed) -- see
LeKiwiRobotInterface's docstring for the method names/units confirmed this way and the
one real integration gap that research surfaced: the stock class assumes the FULL kit
(arm + gripper + 3 wheels, 9 motors), and this project's build has no arm. Camera/lidar
sensor reading and the kill switch are still genuinely unconfirmed -- no LeRobot lidar
support exists at all (confirmed by reading the class: it only has `cameras`, no lidar
concept), and there's no way to check GPIO/kill-switch specifics without the actual Pi
in hand. Every remaining `# TODO(hardware)` is a real gap, not a stylistic placeholder.

Loop structure matches the sim's own control rate (plan.md Phase 3: 30Hz, i.e. 1/60s
physics with decimation=2) and the sim's action interface (mdp/actions.py's
BodyVelocityAction: body-frame vx/vy/omega, [-1,1] normalized then scaled by
max_lin_vel/max_ang_vel) -- so the exported ONNX policy's output can be sent to the
real firmware's own vx/vy/omega command with the same scaling used in training, not a
different one invented at deployment time.

Uses onnxruntime rather than full torch -- lighter dependency for a Pi 5, and
export_policy.py already produces a .onnx file for exactly this.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

CONTROL_HZ = 30.0  # must match plan.md Phase 3 / isaac_lab env_cfg_base.py's decimation-derived control rate
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ

# Same normalized-action scaling as mdp/actions.py:BodyVelocityActionCfg -- keep these
# two files' numbers in sync if either changes; deployment must replay the exact
# mapping the policy was trained against.
MAX_LIN_VEL_MPS = 0.5
MAX_ANG_VEL_RADPS = 2.0


@dataclass
class KillSwitch:
    """
    Phase 11: "kill-switch ready" -- polled every control loop, not just checked once
    at startup. TODO(hardware): wire is_triggered() to whatever's actually available --
    a GPIO button, a keyboard listener over SSH, or a simple "does this file exist"
    dead-man's-switch file another terminal can touch. Don't ship the NotImplementedError
    default to a real run.
    """

    def is_triggered(self) -> bool:
        raise NotImplementedError("Wire this to a real kill switch before running on hardware.")


class SensorReader:
    """TODO(hardware): implement one of these two, matching whichever policy variant you trained."""

    def read(self) -> np.ndarray:
        raise NotImplementedError


class CameraSensorReader(SensorReader):
    """
    lerobot.robots.lekiwi.LeKiwi.get_observation() would give us a camera frame too
    (via each configured Camera's .read_latest(), confirmed real API) -- deliberately
    NOT used here even though LeKiwiRobotInterface already holds a connected LeKiwi
    instance, because that class's default camera config (config_lekiwi.py) hardcodes
    a "front" camera at /dev/video0 640x480 rotated 180deg AND a "wrist" camera at
    /dev/video2 -- this project's build has no wrist/arm camera, so the stock config
    doesn't match. Reading the X10 directly via cv2 (below) sidesteps needing a custom
    LeKiwiConfig just for camera enumeration. Revisit once hardware is in hand and it's
    clear whether a trimmed LeKiwiConfig (single "front" camera, no wrist) is easier
    than this standalone path.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        # width=640/height=480 is not the X10's native resolution -- Seeed's own
        # product page confirms it's a 1080p sensor (1920x1080) -- it's a deliberate
        # match to what the policy was actually TRAINED on
        # (env_cfg_camera.py:LekiwiCameraSceneCfg.front_camera, downsampled from
        # native 1080p there for training-time VRAM reasons, see that file's own
        # comment). Requesting 640x480 directly from the UVC driver via
        # CAP_PROP_FRAME_WIDTH/HEIGHT below (if the driver honors it) is preferred
        # over capturing at 1080p and downsampling in software here -- less work per
        # frame on the Pi, and avoids a second place this resolution could drift out
        # of sync with what training used.
        # TODO(hardware): the Seeed X10 is a standard UVC USB camera per BoM.md, so
        # cv2.VideoCapture(device_index) is the likely path -- confirm the actual
        # /dev/videoN index and that V4L2 actually honors the requested 640x480 mode
        # (rather than silently capturing at 1080p and needing a software resize
        # added here) once the camera is in hand.
        import cv2

        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        import cv2

        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb.transpose(2, 0, 1)[None].astype(np.float32)  # (1,3,H,W), matches export_policy.py's CameraPolicyWrapper input


class LidarSensorReader(SensorReader):
    """
    CONFIRMED (not just assumed): lerobot's own LeKiwi robot class has no lidar concept
    at all -- read directly from source (lekiwi.py), its `cameras` dict is the only
    sensor abstraction, RPLIDAR isn't mentioned anywhere in that module. So there's no
    "wrong LeRobot API" risk here to research further -- a standalone RPLIDAR driver
    (below) is the only path, not a shortcut that skipped checking a real integration.
    """

    def __init__(self, port: str = "/dev/ttyUSB0"):
        # TODO(hardware): confirm which RPLIDAR python package you're actually using
        # (e.g. PyPI "rplidar" / "rplidar-roboticia") and its real API -- constructor
        # args and iterator shape both vary by package/version.
        from rplidar import RPLidar  # placeholder import name -- confirm the real package

        self._lidar = RPLidar(port)
        # Matches env_cfg_lidar.py's LidarPatternCfg(horizontal_res=1.0,
        # horizontal_fov_range=(-50, 50)) -- 100 rays over a forward-facing 100deg
        # window, NOT the RPLIDAR's full 360deg physical sweep (narrowed 2026-08-11 to
        # match the real Arducam camera's published 100deg FOV; was 360 before). The
        # real unit still spins the full circle -- see read()'s TODO below, which now
        # also has to DISCARD everything outside the forward 100deg, not just rebin
        # onto a 1deg grid.
        self._num_rays = 100

    def read(self) -> np.ndarray:
        # TODO(hardware): real RPLIDAR scans arrive as a stream of (quality, angle,
        # distance) samples, not a clean fixed-bin array -- bin/interpolate onto the
        # same 1-degree grid the sim's RayCaster produces (env_cfg_lidar.py), THEN crop
        # to the same forward-facing -50..50deg window the sim was trained on (readings
        # outside that window must be dropped, not fed to the policy -- it never saw
        # anything outside that window during training), or the observation won't match
        # what the policy trained on.
        raise NotImplementedError


class LeKiwiRobotInterface:
    """
    Wraps the REAL `lerobot.robots.lekiwi.LeKiwi` class (huggingface/lerobot,
    src/lerobot/robots/lekiwi/lekiwi.py) -- fetched and read directly, not guessed.
    Confirmed real method names, action-dict schema, and units:

      - `LeKiwi(LeKiwiConfig(port=...)).connect()` / `.disconnect()`
      - `send_action(action: dict)` -- filters keys by suffix, base velocity keys are
        `"x.vel"` (m/s), `"y.vel"` (m/s), `"theta.vel"` -- **degrees/s, NOT radians/s**
        (`LeKiwiConfig.use_degrees` defaults to `True`; `_body_to_wheel_raw`'s own
        theta param is documented in deg/s). This project's sim action space
        (mdp/actions.py) outputs omega in **rad/s** -- converted explicitly below
        (`math.degrees(...)`). Missing this conversion would silently send an omega
        ~57x too small (or, sent the other way, ~57x too large) to the real robot --
        exactly the kind of unit mismatch this research pass exists to catch before
        it happens on hardware, not after.
      - `stop_base()` -- the real e-stop primitive (`sync_write("Goal_Velocity", 0)`
        on all base motors), used as this class's `emergency_stop()`.
      - `get_observation()` -- returns camera frames (`cam.read_latest()` per
        configured camera) AND the same state-feature keys as `action_features`
        (arm `.pos` + base `.vel`) -- whether the base `.vel` entries are a true
        odometry readback or just mirror the last commanded velocity is NOT
        confirmed from source alone; print one real observation dict once hardware
        exists and check before trusting it for the policy's proprioceptive input
        (mdp/observations.py's base_pose_2d/base_velocity_2d).

    REAL INTEGRATION GAP, confirmed by reading the class, not assumed: `LeKiwiConfig`
    has no base-only mode -- `LeKiwi.__init__` configures a `FeetechMotorsBus` for the
    full kit's 9 motors (5 arm joints + 1 gripper + 3 wheels), matching physical servo
    IDs on the bus. This project's build has NO arm (removed from the sim asset,
    isaac_sim/README_lekiwi_variants.md), so connecting with a stock, unmodified
    `LeKiwiConfig` against a bus that's only ever had 3 wheel motors wired to it will
    very likely fail at connect()/calibration time (it'll try to ping motor IDs that
    don't physically exist). Two real options once hardware is in hand, neither
    attempted here since neither is checkable without it:
      (a) A custom `LeKiwiConfig`/motor-ID subclass trimmed to 3 wheel motors only.
      (b) Skip the high-level `LeKiwi` class for the base entirely and drive
          `FeetechMotorsBus` directly, reusing `_body_to_wheel_raw`'s real, confirmed
          signature -- `_body_to_wheel_raw(x, y, theta, wheel_radius=0.05,
          base_radius=0.125, max_raw=3000)` -- as a reference for the wheel
          inverse-kinematics math instead of re-deriving it. Side note, not a bug:
          LeRobot's own default `base_radius=0.125m` (125mm) lands close to this
          project's independently photo/measurement-derived wheel radius (117.6mm,
          isaac_sim/README_lekiwi_variants.md) -- a reassuring cross-check between two
          unrelated sources, not a discrepancy to resolve.

    Given that gap, this class wraps `LeKiwi` but does NOT instantiate it in __init__
    with a bare default config -- `connect()` takes an explicit, pre-built `LeKiwi`
    instance so whichever of (a)/(b) above turns out to be necessary happens at the
    call site once hardware exists, not silently assumed here.
    """

    def __init__(self, robot):
        """`robot` must already be a connected `lerobot.robots.lekiwi.LeKiwi` instance
        (or a compatible stand-in providing send_action/get_observation/stop_base) --
        constructed at the call site per the class docstring's (a)/(b) options, not
        here, since which option applies isn't decidable without real hardware."""
        self._robot = robot

    def send_body_velocity(self, vx_mps: float, vy_mps: float, omega_radps: float) -> None:
        self._robot.send_action({
            "x.vel": vx_mps,
            "y.vel": vy_mps,
            "theta.vel": math.degrees(omega_radps),  # rad/s (sim) -> deg/s (real LeKiwiConfig default)
        })

    def get_observation(self) -> dict:
        return self._robot.get_observation()

    def emergency_stop(self) -> None:
        self._robot.stop_base()


def run(
    onnx_path: str,
    sensor: SensorReader,
    robot: LeKiwiRobotInterface,
    kill_switch: KillSwitch,
    max_lin_vel: float = MAX_LIN_VEL_MPS,
    max_ang_vel: float = MAX_ANG_VEL_RADPS,
) -> None:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    print(f"[lekiwi_policy_runner] Loaded {onnx_path}, control loop at {CONTROL_HZ} Hz. Ctrl+C or kill switch to stop.")

    # Phase 11: "no explicit incentive to approach slowly -- verify real approach speed
    # near cones/goal is actually safe... before running unsupervised." Ramp commanded
    # velocity up over the first few seconds rather than snapping to full policy output
    # immediately on start -- cheap first line of defense for the very first real run,
    # not a substitute for actually watching it with a hand near the kill switch.
    ramp_start = time.monotonic()
    ramp_duration_s = 5.0

    try:
        while True:
            loop_start = time.monotonic()

            if kill_switch.is_triggered():
                print("[lekiwi_policy_runner] Kill switch triggered, stopping.")
                robot.emergency_stop()
                break

            obs = sensor.read()
            # TODO(hardware): also feed real proprioceptive base_pose/base_velocity
            # (from robot.get_observation()'s base ".vel"/".pos" keys -- see
            # LeKiwiRobotInterface's docstring for what's confirmed vs. still open
            # about those) alongside the sensor frame -- this skeleton only wires the
            # sensor input so far.
            outputs = session.run(None, {input_names[0]: obs})
            action = np.clip(outputs[0][0], -1.0, 1.0)

            ramp = min(1.0, (time.monotonic() - ramp_start) / ramp_duration_s)
            vx = float(action[0]) * max_lin_vel * ramp
            vy = float(action[1]) * max_lin_vel * ramp
            omega = float(action[2]) * max_ang_vel * ramp

            robot.send_body_velocity(vx, vy, omega)

            elapsed = time.monotonic() - loop_start
            sleep_s = CONTROL_PERIOD_S - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                print(f"[lekiwi_policy_runner] WARNING: control loop overran by {-sleep_s*1000:.1f}ms")
    except KeyboardInterrupt:
        pass
    finally:
        robot.emergency_stop()


def build_robot(port: str):
    """
    TODO(hardware): construct the real base-control connection here, once hardware
    exists (Phase 9) -- per LeKiwiRobotInterface's docstring, this project's arm-free
    build needs either (a) a trimmed base-only LeKiwiConfig/motor-ID subclass, or
    (b) a direct FeetechMotorsBus wrapper reusing the real, confirmed
    `_body_to_wheel_raw(x, y, theta, wheel_radius=0.05, base_radius=0.125,
    max_raw=3000)` math instead of the stock `LeKiwi` class. Not implemented here --
    guessing which one fits, or the exact real motor IDs on the bus, means real motors
    doing the wrong thing on first power-up, not just a crashed script. Once decided:

        from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig
        robot = LeKiwi(LeKiwiConfig(port=port, cameras={}))  # cameras={} -- see
        # CameraSensorReader's docstring for why this file reads the X10 directly
        # instead of through LeKiwi's own camera config
        robot.connect()
        return robot
    """
    raise NotImplementedError(f"Build a real base-control connection on {port} -- see this function's docstring.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--sensor", choices=["camera", "lidar"], required=True)
    parser.add_argument("--port", default="/dev/ttyACM0", help="Feetech bus serial port (lerobot.robots.lekiwi.LeKiwiConfig.port)")
    args = parser.parse_args()

    sensor: SensorReader = CameraSensorReader() if args.sensor == "camera" else LidarSensorReader()
    robot = LeKiwiRobotInterface(build_robot(args.port))
    run(args.onnx, sensor, robot=robot, kill_switch=KillSwitch())


if __name__ == "__main__":
    main()
