# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pose integration for keyboard teleoperation.

This module holds the whole teleoperation state machine and deliberately
imports neither ``dora`` nor the WebRTC stack, so it can be exercised without
a dataflow or a browser.  ``main`` supplies the events and publishes the
results.

Held keys are read as velocities: each ``step`` advances the target pose of
the key's own arm by ``speed * dt`` along every axis whose key is down, so
both arms can move at once.  Holding Shift makes the motion keys drive
rotation instead of translation.  Orientation is integrated in the **tool
frame** (``r_new = r_cur * delta``), which keeps roll, pitch and yaw
meaningful relative to the gripper rather than the world.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .keymap import (
    GRIP_KEYS,
    LEFT,
    MOTION_KEYS,
    RIGHT,
    ROTATION_KEY,
)

# End-effector pose of the scene's ``home`` keyframe, expressed in its
# ``arm_origin`` site frame.  The IK and MuJoCo nodes both start from that same
# keyframe, so starting anywhere else would make IK yank both arms across the
# workspace on the very first tick.  Regenerate with ``dev/home_pose.py``.
DEFAULT_HOME_RIGHT = np.array([0.216, -0.1535, -0.22], dtype=np.float64)
DEFAULT_HOME_LEFT = np.array([0.216, 0.1535, -0.22], dtype=np.float64)
DEFAULT_HOME_RPY = (0.0, -90.0, 0.0)

DEFAULT_LINEAR_SPEED = 0.05  # m/s
DEFAULT_ANGULAR_SPEED = 0.5  # rad/s
DEFAULT_GRIP_SPEED = 2.0  # fraction/s

DEFAULT_POS_MIN = np.array([-0.8, -0.8, -0.8], dtype=np.float64)
DEFAULT_POS_MAX = np.array([0.8, 0.8, 0.8], dtype=np.float64)

# Fully open gripper angle, mirrored between the arms.  Kept identical to
# dora-openarm-vr so a dataflow can swap one teleoperation source for the other.
_GRIPPER_OPEN_ANGLE = 1.57 / 2.0


def map_grip_to_angle(grip: float, side: str) -> float:
    """Map a grip fraction (0 open, 1 closed) to a gripper joint angle."""
    magnitude = _GRIPPER_OPEN_ANGLE * (1.0 - grip)
    return magnitude if side == LEFT else -magnitude


class KeyState:
    """Tracks which motion keys are down."""

    def __init__(self) -> None:
        """Start with nothing held."""
        self._held: set[str] = set()

    @property
    def held(self) -> set[str]:
        """Keys currently down, as a copy safe to iterate while events arrive."""
        return set(self._held)

    def press(self, key: str) -> None:
        """Mark a key down."""
        self._held.add(key)

    def release(self, key: str) -> None:
        """Mark a key up."""
        self._held.discard(key)

    def clear(self) -> None:
        """Release every key currently held."""
        self._held.clear()


class ArmState:
    """Target pose of a single arm."""

    def __init__(self, home_pos: np.ndarray, home_rot: Rotation) -> None:
        """Start the arm at its home pose with the gripper fully open."""
        self.home_pos = np.asarray(home_pos, dtype=np.float64).copy()
        self.home_rot = home_rot
        self.pos = self.home_pos.copy()
        self.rot = home_rot
        self.grip = 0.0

    def step_home(self, max_distance: float, max_angle: float) -> bool:
        """Move the target toward home by one bounded step.

        Return whether the arm is now home.  The gripper is left where it is:
        an arm that is holding something carries it home rather than dropping
        it on the way.
        """
        at_home = True

        offset = self.home_pos - self.pos
        distance = float(np.linalg.norm(offset))
        if distance <= max_distance:
            self.pos = self.home_pos.copy()
        else:
            self.pos = self.pos + offset * (max_distance / distance)
            at_home = False

        # Integrated in the tool frame, like the manual rotation keys, so the
        # target follows the same geodesic it would under manual control.
        offset_rotvec = (self.rot.inv() * self.home_rot).as_rotvec()
        angle = float(np.linalg.norm(offset_rotvec))
        if angle <= max_angle:
            self.rot = self.home_rot
        else:
            self.rot = self.rot * Rotation.from_rotvec(
                offset_rotvec * (max_angle / angle)
            )
            at_home = False

        return at_home


class TeleopState:
    """Integrates held keys into a pair of end-effector pose targets.

    Every motion and gripper key names the arm it drives, so the two arms are
    independent and can be moved at the same time.
    """

    def __init__(
        self,
        home_right: np.ndarray | None = None,
        home_left: np.ndarray | None = None,
        home_rotation: Rotation | None = None,
        linear_speed: float = DEFAULT_LINEAR_SPEED,
        angular_speed: float = DEFAULT_ANGULAR_SPEED,
        grip_speed: float = DEFAULT_GRIP_SPEED,
        pos_min: np.ndarray | None = None,
        pos_max: np.ndarray | None = None,
    ) -> None:
        """Build both arms at their home poses."""
        if home_right is None:
            home_right = DEFAULT_HOME_RIGHT
        if home_left is None:
            home_left = DEFAULT_HOME_LEFT
        if home_rotation is None:
            home_rotation = Rotation.from_euler("xyz", DEFAULT_HOME_RPY, degrees=True)

        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.grip_speed = grip_speed
        self.pos_min = np.asarray(
            DEFAULT_POS_MIN if pos_min is None else pos_min, dtype=np.float64
        )
        self.pos_max = np.asarray(
            DEFAULT_POS_MAX if pos_max is None else pos_max, dtype=np.float64
        )

        self.arms = {
            RIGHT: ArmState(home_right, home_rotation),
            LEFT: ArmState(home_left, home_rotation),
        }
        self.enabled = True
        self.homing = False

    def start_home(self) -> None:
        """Begin walking both targets back to their home poses."""
        self.homing = True

    def reset_home(self) -> None:
        """Atomically reset both targets and grippers between episodes."""
        self.homing = False
        for arm in self.arms.values():
            arm.pos = arm.home_pos.copy()
            arm.rot = arm.home_rot
            arm.grip = 0.0

    def cancel_home(self) -> None:
        """Abort a home return, leaving both targets where they are."""
        self.homing = False

    def step(self, dt: float, held_keys: set[str]) -> None:
        """Advance both targets by one timestep of the currently held keys."""
        if dt <= 0.0 or not self.enabled:
            return

        if self.homing:
            # A home return owns both targets and moves them at the same speed
            # manual control would, so the arms come back at a speed the
            # operator has already accepted.  Nothing is held while it runs:
            # pressing a motion key cancels it and hands control straight back.
            reached = [
                arm.step_home(self.linear_speed * dt, self.angular_speed * dt)
                for arm in self.arms.values()
            ]
            self.homing = not all(reached)
            return

        # Shift is momentary: the motion keys drive rotation only while it is
        # down, so releasing it always returns to translation.
        rotating = ROTATION_KEY in held_keys
        linear = {RIGHT: np.zeros(3), LEFT: np.zeros(3)}
        angular = {RIGHT: np.zeros(3), LEFT: np.zeros(3)}
        grip = {RIGHT: 0.0, LEFT: 0.0}

        for key in held_keys:
            motion = MOTION_KEYS.get(key)
            if motion is not None:
                side, linear_axis, angular_axis, sign = motion
                if rotating:
                    angular[side][angular_axis] += sign
                else:
                    linear[side][linear_axis] += sign
                continue
            gripper = GRIP_KEYS.get(key)
            if gripper is not None:
                side, sign = gripper
                grip[side] += sign

        for side, arm in self.arms.items():
            arm.pos = np.clip(
                arm.pos + linear[side] * self.linear_speed * dt,
                self.pos_min,
                self.pos_max,
            )
            rotvec = angular[side] * self.angular_speed * dt
            if rotvec.any():
                arm.rot = arm.rot * Rotation.from_rotvec(rotvec)
            if grip[side]:
                arm.grip = float(
                    np.clip(arm.grip + grip[side] * self.grip_speed * dt, 0.0, 1.0)
                )

    def pose(self, side: str) -> np.ndarray:
        """Return ``[px, py, pz, qw, qx, qy, qz, gripper_angle]`` as float32."""
        arm = self.arms[side]
        qx, qy, qz, qw = arm.rot.as_quat()
        return np.array(
            [
                arm.pos[0],
                arm.pos[1],
                arm.pos[2],
                qw,
                qx,
                qy,
                qz,
                map_grip_to_angle(arm.grip, side),
            ],
            dtype=np.float32,
        )

    def describe(self) -> str:
        """One-line summary of the current state, for status logging."""
        parts = [f"enabled={self.enabled}", f"homing={self.homing}"]
        for side in (RIGHT, LEFT):
            arm = self.arms[side]
            roll, pitch, yaw = arm.rot.as_euler("xyz", degrees=True)
            parts.append(
                f"{side}: p=({arm.pos[0]:+.3f},{arm.pos[1]:+.3f},{arm.pos[2]:+.3f}) "
                f"rpy=({roll:+.0f},{pitch:+.0f},{yaw:+.0f}) grip={arm.grip:.2f}"
            )
        return " | ".join(parts)
