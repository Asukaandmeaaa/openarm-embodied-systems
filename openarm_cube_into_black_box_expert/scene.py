"""Runtime scene adjustment for the middle-box experiment."""

from __future__ import annotations

import mujoco
import numpy as np

BLACK_BOX_CENTER_XY_M = (0.40, 0.0)


def move_black_box_to_middle(model: mujoco.MjModel) -> tuple[float, float]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "black_frame")
    if body < 0:
        raise ValueError("scene is missing black_frame")
    model.body_pos[body, :2] = np.asarray(BLACK_BOX_CENTER_XY_M, dtype=float)
    return BLACK_BOX_CENTER_XY_M
