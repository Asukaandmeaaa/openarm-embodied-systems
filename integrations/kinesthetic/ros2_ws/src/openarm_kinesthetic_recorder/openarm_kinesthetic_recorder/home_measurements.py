"""Measured-pose and table-height contracts for fixed lifted HOME commissioning.

This module has no ROS imports.  In particular, a posture read from CAN is evidence about
the current posture only; it never authorizes motion and is never treated as motor zero.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


RIGHT_JOINT_NAMES = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
LEFT_JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
Q_MOTOR_ZERO = (0.0,) * 7
Q_TABLE_SUPPORTED_RIGHT = (
    0.204280,
    -0.005531,
    -0.032616,
    0.426299,
    0.025750,
    -0.798238,
    -0.129892,
)
Q_TABLE_SUPPORTED_LEFT = (
    -0.093652,
    -0.072290,
    0.069238,
    0.616655,
    -0.108911,
    0.778782,
    -0.015831,
)
Q_TABLE_SUPPORTED_RIGHT_GRIPPER = -0.342374
Q_TABLE_SUPPORTED_LEFT_GRIPPER = 0.219921
# Fixed right-arm target selected by the completed +10 cm study.  Runtime HOME
# planning consumes this value directly: it is not recomputed by IK and is not
# loaded from a candidate-ranking report.
Q_LIFTED_HOME_RIGHT = (
    -0.1450262670,
    -0.0151076284,
    -0.0060887956,
    1.3751977241,
    -0.0031428430,
    -0.1984151663,
    -0.1222898194,
)
# Operator-confirmed bimanual contract for the right-TCP +10 cm HOME study.  Keep this
# separate from Q_TABLE_SUPPORTED_LEFT so a CAN snapshot of the table-supported left arm
# can never be mistaken for the stationary left-arm planning state.
Q_LIFTED_HOME_LEFT_FIXED = Q_MOTOR_ZERO


def _seven(values: Sequence[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 7 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain seven finite radian values")
    return result


def posture_snapshot() -> dict:
    """Return immutable provenance; deliberately label both poses externally supported."""

    return {
        "q_motor_zero_rad": list(Q_MOTOR_ZERO),
        "right": {
            "joint_names": list(RIGHT_JOINT_NAMES),
            "can_j1_j7_rad": list(Q_TABLE_SUPPORTED_RIGHT),
            "measured_gripper_rad": Q_TABLE_SUPPORTED_RIGHT_GRIPPER,
            "semantic_role": "real_measured_posture_snapshot_only",
            "externally_supported_pose": True,
            "allowed_uses": [
                "fk_reference",
                "collision_validation_reference",
                "side_prehome_search_seed_or_reference",
            ],
            "forbidden_uses": ["motor_zero", "act_home", "automatic_execution_target"],
        },
        "left": {
            "joint_names": list(LEFT_JOINT_NAMES),
            "can_j1_j7_rad": list(Q_TABLE_SUPPORTED_LEFT),
            "measured_gripper_rad": Q_TABLE_SUPPORTED_LEFT_GRIPPER,
            "semantic_role": "real_measured_posture_snapshot_only",
            "externally_supported_pose": True,
            "forbidden_uses": ["motor_zero", "automatic_execution_target"],
        },
    }


def compare_can_to_ros(
    can_j1_j7_rad: Sequence[float],
    ros_positions_by_name: Mapping[str, float],
    joint_names: Sequence[str] = RIGHT_JOINT_NAMES,
) -> dict:
    """Compare simultaneous CAN/ROS samples in canonical order.

    The audited OpenArmHW arm path copies DAMIAO ``get_position()`` directly into the
    J1..J7 ros2_control position state.  Consequently the expected transform is identity
    in radians.  This function still requires actual ROS values and never manufactures a
    comparison from that source-level expectation.
    """

    can = _seven(can_j1_j7_rad, "CAN J1-J7")
    names = tuple(joint_names)
    if len(names) != 7 or any(name not in ros_positions_by_name for name in names):
        missing = [name for name in names if name not in ros_positions_by_name]
        raise ValueError(f"ROS /joint_states J1-J7 unavailable: {missing}")
    ros = _seven([ros_positions_by_name[name] for name in names], "ROS J1-J7")
    return {
        "joint_names": list(names),
        "can_j1_j7_rad": can,
        "ros_joint_states_j1_j7_rad": ros,
        "difference_ros_minus_can_rad": [r - c for c, r in zip(can, ros)],
        "expected_source_transform": {
            "ordering": "motor vector index 0..6 -> openarm_{side}_joint1..7",
            "sign": [1.0] * 7,
            "offset_rad": [0.0] * 7,
            "unit": "radian",
            "status": "source_audited_identity; simultaneous_live_sample_still_required",
        },
    }


def require_current_table_collision(config: Mapping) -> dict:
    """Return the configured NEW-table guard after geometry/epoch validation."""

    heights = config.get("table_height_calibration")
    if not isinstance(heights, Mapping):
        raise RuntimeError("NEW_TABLE_HEIGHT_MEASUREMENT_REQUIRED")
    old = heights.get("old_table_top_z_m")
    new = heights.get("new_measured_table_top_z_m")
    if not isinstance(old, (int, float)) or not math.isfinite(float(old)):
        raise ValueError("OLD_TABLE_HEIGHT must be a finite measured value")
    if new is None:
        raise RuntimeError("NEW_TABLE_HEIGHT_MEASUREMENT_REQUIRED")
    if not isinstance(new, (int, float)) or not math.isfinite(float(new)):
        raise ValueError("NEW_MEASURED_TABLE_HEIGHT must be a finite measured value")
    collision = config.get("static_collision")
    if not isinstance(collision, Mapping):
        raise RuntimeError("NEW_TABLE_COLLISION_GUARD_REQUIRED")
    if collision.get("height_epoch") != "NEW_MEASURED_TABLE_HEIGHT":
        raise RuntimeError("NEW_TABLE_COLLISION_GUARD_REQUIRED")
    if not math.isclose(
        float(collision.get("top_z_m", math.nan)), float(new), rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError("NEW_TABLE_COLLISION_GUARD_HEIGHT_MISMATCH")
    station = config.get("station_frame")
    station_matches = (
        isinstance(station, Mapping)
        and station.get("frame_id") == collision.get("frame_id")
        and station.get("measurement_epoch") == "NEW_MEASURED_TABLE_HEIGHT"
    )
    if not station_matches:
        raise RuntimeError("NEW_TABLE_STATION_FRAME_VERIFICATION_REQUIRED")
    result = dict(collision)
    result["authorization_scope"] = "CURRENT_CONFIGURED_GUARD"
    return result
