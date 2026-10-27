"""Teaching-ready target: where the arm must be before a human starts teaching.

This is deliberately not a grasp pose. It is the safe, comfortable preparation
position the arm is driven to before manual kinesthetic teaching begins.

The target position is table-relative so the same definition works in simulation and
on hardware:

    target_xyz = table_top_centre + (offset_x, offset_y, height_above_table)

The orientation is never an invented Euler triple. It is taken by forward kinematics
from the project's already-validated episode Home pose, and the prompt is explicit
that this pose is an orientation/seed reference only - the actual joint goal always
comes from a fresh collision-aware IK solve against the real table-relative target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .table_geometry import SOURCE_CONFIGURED_PRIOR, SOURCE_RECOMMENDED, TableGeometry

ORIENTATION_FROM_FK = "fk_reference"


class TeachingReadyError(ValueError):
    """Raised when the teaching-ready definition is unusable."""


@dataclass(frozen=True)
class TeachingReadyTarget:
    position_world_m: np.ndarray
    orientation_matrix: np.ndarray
    orientation_source: str
    table_xy_m: np.ndarray
    height_above_table_m: float
    reference_pose: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tcp_position_world_m": self.position_world_m.tolist(),
            "tcp_orientation_source": self.orientation_source,
            "tcp_orientation_matrix": self.orientation_matrix.tolist(),
            "tcp_orientation_xyzw": rotation_to_quaternion_xyzw(self.orientation_matrix),
            "table_xy_m": self.table_xy_m.tolist(),
            "height_above_table_m": float(self.height_above_table_m),
            "reference_pose": self.reference_pose,
        }


def rotation_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    import mujoco

    wxyz = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(wxyz, np.asarray(matrix, dtype=float).reshape(9))
    return [float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])]


def rotation_error_rad(current: np.ndarray, target: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in radians."""
    relative = np.asarray(current, dtype=float).reshape(3, 3).T @ np.asarray(target, dtype=float).reshape(3, 3)
    cos_angle = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.arccos(min(1.0, max(-1.0, cos_angle))))


def reference_orientation(model, data, mapping, reference_q: np.ndarray) -> np.ndarray:
    """Grasp-site orientation at the reference pose, without changing current state."""
    import mujoco

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    try:
        data.qpos[mapping.right_qpos] = np.asarray(reference_q, dtype=float)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return np.asarray(data.site_xmat[mapping.grasp_site], dtype=float).reshape(3, 3).copy()
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)


def teaching_ready_target(
    model,
    data,
    mapping,
    geometry: TableGeometry,
    config: Mapping[str, Any],
    reference_q: np.ndarray,
) -> TeachingReadyTarget:
    block = config.get("teaching_ready")
    if not isinstance(block, Mapping):
        raise TeachingReadyError("teaching_ready: required mapping is missing")
    for key in ("offset_x_m", "offset_y_m", "height_above_table_m"):
        if not isinstance(block.get(key), Mapping) or "value" not in block[key]:
            raise TeachingReadyError(f"teaching_ready.{key} must be a mapping with a `value` field")
    orientation_block = block.get("orientation", {})
    if orientation_block.get("source") != ORIENTATION_FROM_FK:
        raise TeachingReadyError(
            "teaching_ready.orientation.source must be fk_reference: an invented Euler triple is not acceptable"
        )
    reference_pose = str(orientation_block.get("reference_pose", "right_task_ready"))

    offset_x = float(block["offset_x_m"]["value"])
    offset_y = float(block["offset_y_m"]["value"])
    height = float(block["height_above_table_m"]["value"])
    if not np.isfinite([offset_x, offset_y, height]).all():
        raise TeachingReadyError("teaching-ready offsets must be finite")
    if height <= 0.0:
        raise TeachingReadyError("teaching-ready height above the table must be positive")

    table_xy = geometry.center[:2] + np.asarray([offset_x, offset_y], dtype=float)
    if not geometry.contains_xy(table_xy):
        raise TeachingReadyError(
            "teaching-ready offset leaves the table footprint; use offsets inside the prior table"
        )
    position = np.asarray([table_xy[0], table_xy[1], geometry.top_z + height], dtype=float)

    orientation = reference_orientation(model, data, mapping, np.asarray(reference_q, dtype=float))

    return TeachingReadyTarget(
        position_world_m=position,
        orientation_matrix=orientation,
        orientation_source=ORIENTATION_FROM_FK,
        table_xy_m=np.asarray(table_xy, dtype=float),
        height_above_table_m=height,
        reference_pose=reference_pose,
    )


def teaching_ready_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    block = config.get("teaching_ready", {})
    return {
        "offset_x": block.get("offset_x_m", {}).get("source", SOURCE_CONFIGURED_PRIOR),
        "offset_y": block.get("offset_y_m", {}).get("source", SOURCE_CONFIGURED_PRIOR),
        "height_above_table": block.get("height_above_table_m", {}).get("source", SOURCE_RECOMMENDED),
        "orientation": str(block.get("orientation", {}).get("source", ORIENTATION_FROM_FK)),
    }
