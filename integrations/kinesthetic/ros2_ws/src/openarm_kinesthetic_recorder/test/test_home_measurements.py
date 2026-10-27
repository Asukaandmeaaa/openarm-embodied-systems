"""Safety regressions for the lowered-table and CAN snapshot update."""

from pathlib import Path

import pytest
import yaml

from openarm_kinesthetic_recorder.home_measurements import (
    Q_LIFTED_HOME_LEFT_FIXED,
    Q_LIFTED_HOME_RIGHT,
    Q_MOTOR_ZERO,
    Q_TABLE_SUPPORTED_LEFT,
    Q_TABLE_SUPPORTED_LEFT_GRIPPER,
    Q_TABLE_SUPPORTED_RIGHT,
    Q_TABLE_SUPPORTED_RIGHT_GRIPPER,
    RIGHT_JOINT_NAMES,
    compare_can_to_ros,
    posture_snapshot,
    require_current_table_collision,
)


CONFIG = Path(__file__).resolve().parents[1] / "config/home_initialization.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_can_snapshot_is_not_motor_zero_or_execution_target() -> None:
    snapshot = posture_snapshot()["right"]
    assert tuple(snapshot["can_j1_j7_rad"]) == Q_TABLE_SUPPORTED_RIGHT
    assert tuple(snapshot["can_j1_j7_rad"]) != Q_MOTOR_ZERO
    assert snapshot["semantic_role"] == "real_measured_posture_snapshot_only"
    assert snapshot["externally_supported_pose"] is True
    assert snapshot["measured_gripper_rad"] == pytest.approx(
        Q_TABLE_SUPPORTED_RIGHT_GRIPPER
    )
    assert posture_snapshot()["left"]["measured_gripper_rad"] == pytest.approx(
        Q_TABLE_SUPPORTED_LEFT_GRIPPER
    )
    assert "motor_zero" in snapshot["forbidden_uses"]
    assert "automatic_execution_target" in snapshot["forbidden_uses"]


def test_lifted_right_home_study_keeps_left_arm_at_motor_zero() -> None:
    assert Q_LIFTED_HOME_LEFT_FIXED == Q_MOTOR_ZERO
    assert Q_LIFTED_HOME_LEFT_FIXED != Q_TABLE_SUPPORTED_LEFT


def test_lifted_right_home_is_the_fixed_reviewed_joint_target() -> None:
    assert Q_LIFTED_HOME_RIGHT == (
        -0.1450262670,
        -0.0151076284,
        -0.0060887956,
        1.3751977241,
        -0.0031428430,
        -0.1984151663,
        -0.1222898194,
    )


def test_old_and_new_table_heights_are_explicit_and_not_aliased() -> None:
    calibration = _config()["table_height_calibration"]
    assert calibration["old_table_top_z_m"] == pytest.approx(0.340)
    assert calibration["new_measured_table_top_z_m"] == pytest.approx(0.270)
    assert calibration["new_table_measurement_status"] == (
        "APPROXIMATE_GUARD_REAL_MOTION_VALIDATED_2026_09_21"
    )
    assert calibration["runtime_measurement_confirmed"] is False


def test_operator_near_edge_measurement_is_not_misused_as_box_center() -> None:
    config = _config()
    calibration = config["table_height_calibration"]
    assert calibration["table_near_edge_forward_from_robot_center_m"] == pytest.approx(0.175)
    assert calibration["registered_table_center_forward_m"] == pytest.approx(0.382)
    assert calibration["registered_table_center_lateral_m"] == pytest.approx(0.016)
    assert calibration["registered_center_implied_near_edge_forward_m"] == pytest.approx(0.182)
    assert calibration["near_edge_registration_residual_m"] == pytest.approx(0.007)
    assert config["static_collision"]["center_xyz_m"][:2] == pytest.approx(
        [0.382, 0.016]
    )
    assert config["static_collision"]["center_xyz_m"][0] != pytest.approx(0.175)


def test_missing_new_table_height_fails_closed_without_guessing() -> None:
    config = _config()
    config["table_height_calibration"]["new_measured_table_top_z_m"] = None
    with pytest.raises(RuntimeError, match="NEW_TABLE_HEIGHT_MEASUREMENT_REQUIRED"):
        require_current_table_collision(config)


def test_current_guard_needs_geometry_but_no_authorization_flags() -> None:
    config = _config()
    assert "runtime_authorized" not in config["static_collision"]
    assert "runtime_authorized" not in config["station_frame"]
    guard = require_current_table_collision(config)
    assert guard["authorization_scope"] == "CURRENT_CONFIGURED_GUARD"


def test_new_height_still_requires_matching_current_guard() -> None:
    config = _config()
    config["table_height_calibration"]["new_measured_table_top_z_m"] = 0.300
    with pytest.raises(RuntimeError, match="HEIGHT_MISMATCH"):
        require_current_table_collision(config)


def test_can_ros_comparison_orders_names_and_computes_difference() -> None:
    ros = {
        name: value + 0.001 * (index + 1)
        for index, (name, value) in enumerate(zip(RIGHT_JOINT_NAMES, Q_TABLE_SUPPORTED_RIGHT))
    }
    report = compare_can_to_ros(Q_TABLE_SUPPORTED_RIGHT, ros)
    assert report["joint_names"] == list(RIGHT_JOINT_NAMES)
    assert report["difference_ros_minus_can_rad"] == pytest.approx(
        [0.001 * (index + 1) for index in range(7)]
    )
    assert report["expected_source_transform"]["offset_rad"] == [0.0] * 7


def test_can_ros_comparison_refuses_missing_live_ros_values() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        compare_can_to_ros(Q_TABLE_SUPPORTED_RIGHT, {})
