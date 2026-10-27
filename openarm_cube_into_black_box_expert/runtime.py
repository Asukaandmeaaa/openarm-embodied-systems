"""Runtime policy for the fixed-box experiment."""

from __future__ import annotations

import mujoco
import numpy as np


PARKED_ARM = "left"
ACTIVE_ARM = "right"


def park_left_arm(model: mujoco.MjModel) -> dict[str, int]:
    """Hide and remove left-arm collision for the new task runtime.

    The left arm remains in the 16-value state shape for dataset compatibility,
    but the collector must keep its command at the reset/home value.
    """
    disabled = 0
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "left" not in name and "left" not in body_name:
            continue
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        model.geom_rgba[geom_id, 3] = 0.0
        disabled += 1
    return {"left_geoms_disabled": disabled, "left_geoms_hidden": disabled}


def parked_left_command(mapper, qpos: np.ndarray) -> np.ndarray:
    joints, gripper = mapper.get_driver(qpos, PARKED_ARM)
    return np.asarray(np.r_[joints, gripper], dtype=np.float32).copy()


def assert_left_is_parked(mapper, qpos: np.ndarray, home_command: np.ndarray, tolerance: float = 1e-6) -> None:
    current = parked_left_command(mapper, qpos)
    expected = np.asarray(home_command, dtype=np.float32)
    if expected.shape != (8,) or not np.allclose(current, expected, atol=tolerance, rtol=0):
        raise AssertionError("left arm moved despite fixed parked-arm contract")
