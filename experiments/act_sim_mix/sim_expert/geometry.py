"""Pure grasp geometry helpers, ported from openarm_cube_pick_expert/geometry.py."""
from __future__ import annotations

import math

import numpy as np

RIGHT_GRIPPER_RANGE = (-0.7854, 0.0)
LEFT_GRIPPER_RANGE = (0.0, 0.7854)


def wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    return q[[1, 2, 3, 0]]


def yaw_candidates(cube_yaw: float) -> tuple[float, ...]:
    return tuple(
        (cube_yaw + k * math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        for k in range(4)
    )


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
