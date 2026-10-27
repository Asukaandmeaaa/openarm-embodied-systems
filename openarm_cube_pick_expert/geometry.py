"""Pure grasp geometry contracts."""

from __future__ import annotations

import math
import numpy as np


RIGHT_GRIPPER_RANGE = (-0.7854, 0.0)
LEFT_GRIPPER_RANGE = (0.0, 0.7854)


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert an external xyzw quaternion to MuJoCo's wxyz order."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    return q[[3, 0, 1, 2]]


def wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    return q[[1, 2, 3, 0]]


def yaw_candidates(cube_yaw: float) -> tuple[float, ...]:
    """Return the four square-object symmetries, normalized to [-pi, pi)."""
    return tuple((cube_yaw + k * math.pi / 2 + math.pi) % (2 * math.pi) - math.pi for k in range(4))


def right_gripper_target(
    object_width: float,
    *,
    preload: float = 0.002,
    contact_radius: float = 0.070,
    reference_width: float = 0.040,
    reference_angle: float = -0.10,
) -> float:
    """Map commanded aperture to the right finger angle using jaw arc geometry."""
    if object_width <= 0 or preload < 0 or contact_radius <= 0:
        raise ValueError("invalid gripper geometry")
    aperture = max(0.0, object_width - preload)
    # Calibrate the arc model at the real-robot 40 mm/-0.10 rad datum.
    angle = reference_angle + (reference_width - aperture) / (2.0 * contact_radius)
    return float(np.clip(angle, *RIGHT_GRIPPER_RANGE))


def left_gripper_target_from_right(right_target: float) -> float:
    """Mirror the physical aperture command into the left joint convention."""
    return float(np.clip(-float(right_target), *LEFT_GRIPPER_RANGE))


def combine_action(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    right = np.asarray(right, dtype=np.float32)
    left = np.asarray(left, dtype=np.float32)
    if right.shape != (8,) or left.shape != (8,):
        raise ValueError("each arm command must have shape (8,)")
    return np.concatenate((right, left)).astype(np.float32, copy=False)
