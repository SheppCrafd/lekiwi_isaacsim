"""
On-robot real-time inference loop skeleton (plan.md Phases 10-11). Runs on the
Raspberry Pi 5 once actual hardware exists (Phase 9 -- nothing physical has been
ordered yet, see BoM.md) and a policy has passed Phase 8 held-out eval.

This is deliberately a SKELETON, not a finished deployment script -- the real LeRobot
LeKiwi robot class's actual method names for reading sensors and sending motor commands
aren't something to guess at from here (no repo access to verify against). Every
`# TODO(hardware)` below is a real integration point, not a stylistic placeholder --
wire each to the actual LeRobot API once you're looking at it, don't assume this file's
guessed method names are right.

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
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        # TODO(hardware): the Seeed X10 is a standard UVC USB camera per BoM.md, so
        # cv2.VideoCapture(device_index) is the likely path -- confirm the actual
        # /dev/videoN index and that V4L2 exposes 640x480 without needing a format
        # conversion once the camera is in hand.
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
    def __init__(self, port: str = "/dev/ttyUSB0"):
        # TODO(hardware): confirm which RPLIDAR python package you're actually using
        # (e.g. PyPI "rplidar" / "rplidar-roboticia") and its real API -- constructor
        # args and iterator shape both vary by package/version.
        from rplidar import RPLidar  # placeholder import name -- confirm the real package

        self._lidar = RPLidar(port)
        self._num_rays = 360  # matches env_cfg_lidar.py's LidarPatternCfg(horizontal_res=1.0)

    def read(self) -> np.ndarray:
        # TODO(hardware): real RPLIDAR scans arrive as a stream of (quality, angle,
        # distance) samples, not a clean fixed-360-bin array -- bin/interpolate onto
        # the same 1-degree grid the sim's RayCaster produces (env_cfg_lidar.py) before
        # handing this to the policy, or the observation distribution won't match what
        # it trained on.
        raise NotImplementedError


class LeKiwiRobotInterface:
    """
    TODO(hardware): replace this whole class with LeRobot's actual LeKiwi robot
    interface (https://github.com/huggingface/lerobot, plan.md Phase 9) once it's
    installed against the real hardware. Method names here (`send_body_velocity`,
    `emergency_stop`) are this skeleton's own naming, not LeRobot's real API -- don't
    assume they match without checking.
    """

    def send_body_velocity(self, vx: float, vy: float, omega: float) -> None:
        raise NotImplementedError

    def emergency_stop(self) -> None:
        raise NotImplementedError


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
            # (from the robot's own odometry, matching mdp/observations.py's
            # base_pose_2d/base_velocity_2d) alongside the sensor frame -- this
            # skeleton only wires the sensor input, proprioception plumbing depends on
            # what LeRobot's LeKiwi interface actually exposes for odometry.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--sensor", choices=["camera", "lidar"], required=True)
    args = parser.parse_args()

    sensor: SensorReader = CameraSensorReader() if args.sensor == "camera" else LidarSensorReader()
    run(args.onnx, sensor, robot=LeKiwiRobotInterface(), kill_switch=KillSwitch())


if __name__ == "__main__":
    main()
