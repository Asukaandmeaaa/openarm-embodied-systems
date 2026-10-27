"""Plan and execute the fixed lifted HOME without camera or table detection."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
import rclpy
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml
from rosidl_runtime_py.convert import message_to_ordereddict

from .execution_guard import audit_joint_trajectory
from .home_audit import ExecutionTrackingAudit, path_quality, validate_station_axes
from .home_measurements import Q_LIFTED_HOME_RIGHT, require_current_table_collision
from .home_runtime import (
    HomeRuntimeNode,
    trajectory_sha256,
    validate_real_execution_scaling,
)


RIGHT_ARM_JOINT_NAMES = [f"openarm_right_joint{index}" for index in range(1, 8)]
RIGHT_GRIPPER_JOINT_NAME = "openarm_right_finger_joint1"
STAGE_A_TRANSIT_JOINT_TOLERANCE_RAD = 0.04
FIXED_LIFTED_HOME = list(Q_LIFTED_HOME_RIGHT)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_home_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("HOME configuration must have schema_version 1")
    names = payload.get("right_joint_names")
    goal = payload.get("right_home_joint_positions_rad")
    if names != RIGHT_ARM_JOINT_NAMES:
        raise ValueError("HOME configuration must use right J1-J7 in canonical order")
    if not isinstance(goal, list) or len(goal) != 7 or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in goal
    ):
        raise ValueError("right_home_joint_positions_rad must contain seven finite values")
    if any(
        abs(float(actual) - expected) > 1.0e-9
        for actual, expected in zip(goal, FIXED_LIFTED_HOME)
    ):
        raise ValueError("HOME configuration must contain the fixed lifted HOME")
    if payload.get("gripper_policy") != "open_after_arm_home":
        raise ValueError("HOME initialization must open J8 after arm HOME")
    gripper = payload.get("home_gripper")
    if not isinstance(gripper, dict):
        raise ValueError("home_gripper is required")
    if gripper.get("joint_name") != RIGHT_GRIPPER_JOINT_NAME:
        raise ValueError("home_gripper.joint_name must be the canonical right J8")
    for key in (
        "open_position_rad",
        "position_tolerance_rad",
        "command_duration_s",
        "server_timeout_s",
        "result_timeout_s",
        "stable_duration_s",
    ):
        value = gripper.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"home_gripper.{key} must be finite")
    if float(gripper["open_position_rad"]) >= 0.0:
        raise ValueError("home_gripper.open_position_rad must be negative")
    if any(
        float(gripper[key]) <= 0.0
        for key in (
            "position_tolerance_rad",
            "command_duration_s",
            "server_timeout_s",
            "result_timeout_s",
            "stable_duration_s",
        )
    ):
        raise ValueError("home_gripper timing and tolerance values must be positive")
    if float(gripper["stable_duration_s"]) > float(gripper["result_timeout_s"]):
        raise ValueError("home_gripper stable duration cannot exceed result timeout")
    station = payload.get("station_frame")
    if not isinstance(station, dict) or station.get("frame_id") != "world":
        raise ValueError("station_frame in world is required")
    validate_station_axes(
        {
            "forward": station.get("forward_world", []),
            "lateral": station.get("lateral_world", []),
            "vertical": station.get("vertical_world", []),
        }
    )
    side = payload.get("side_prehome")
    if not isinstance(side, dict) or not side.get("candidate_lateral_offsets_m"):
        raise ValueError("side_prehome candidate search configuration is required")
    selected = side.get("selected_joint_positions_rad")
    if (
        not isinstance(selected, list)
        or len(selected) != 7
        or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in selected)
    ):
        raise ValueError(
            "side_prehome.selected_joint_positions_rad must contain seven finite values"
        )
    if not isinstance(side.get("selected_candidate_id"), str):
        raise ValueError("side_prehome.selected_candidate_id is required")
    gates = side.get("path_quality_gates")
    required_gates = (
        "minimum_clearance_m",
        "maximum_backward_excursion_m",
        "maximum_lateral_excursion_m",
        "maximum_vertical_excursion_m",
        "maximum_absolute_j4_rad",
    )
    if not isinstance(gates, dict) or any(
        key not in gates or not math.isfinite(float(gates[key])) or float(gates[key]) < 0.0
        for key in required_gates
    ):
        raise ValueError("side_prehome.path_quality_gates is invalid")
    collision = payload.get("static_collision")
    if not isinstance(collision, dict):
        raise ValueError("static_collision is required")
    calibration = payload.get("table_height_calibration")
    if not isinstance(calibration, dict) or "old_table_top_z_m" not in calibration or "new_measured_table_top_z_m" not in calibration:
        raise ValueError("OLD_TABLE_HEIGHT and NEW_MEASURED_TABLE_HEIGHT must be explicit")
    old_top = float(calibration["old_table_top_z_m"])
    if not math.isfinite(old_top):
        raise ValueError("OLD_TABLE_HEIGHT must be finite")
    new_top = calibration["new_measured_table_top_z_m"]
    if new_top is not None and (not isinstance(new_top, (int, float)) or not math.isfinite(float(new_top))):
        raise ValueError("NEW_MEASURED_TABLE_HEIGHT must be null or finite")
    for key, length in (
        ("center_xyz_m", 3),
        ("orientation_xyzw", 4),
        ("dimensions_xyz_m", 3),
    ):
        values = collision.get(key)
        if not isinstance(values, list) or len(values) != length or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"static_collision.{key} is invalid")
    if any(float(value) <= 0.0 for value in collision["dimensions_xyz_m"]):
        raise ValueError("static collision dimensions must be positive")
    quaternion = [float(value) for value in collision["orientation_xyzw"]]
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(quaternion_norm - 1.0) > 1.0e-6:
        raise ValueError("static collision orientation must be a unit quaternion")
    expected_top = (
        float(collision["center_xyz_m"][2])
        + 0.5 * float(collision["dimensions_xyz_m"][2])
    )
    if not math.isclose(
        float(collision.get("top_z_m", math.nan)),
        expected_top,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("static collision top_z_m disagrees with center and height")
    for block, keys in (
        ("planning", ("velocity_scaling", "acceleration_scaling", "service_timeout_s")),
        ("final_acceptance", ("joint_tolerance_rad", "settle_timeout_s", "stable_duration_s")),
    ):
        values = payload.get(block)
        if not isinstance(values, dict):
            raise ValueError(f"{block} is required")
        for key in keys:
            value = float(values.get(key, 0.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{block}.{key} must be positive")
    validate_real_execution_scaling(
        float(payload["planning"]["velocity_scaling"]),
        float(payload["planning"]["acceleration_scaling"]),
    )
    if (
        float(payload["final_acceptance"]["stable_duration_s"])
        > float(payload["final_acceptance"]["settle_timeout_s"])
    ):
        raise ValueError("stable_duration_s cannot exceed settle_timeout_s")
    return payload


class HomeInitializationNode(HomeRuntimeNode):
    def __init__(self) -> None:
        self.gripper_position: float | None = None
        self.gripper_state_monotonic_s: float | None = None
        self.gripper_goal_sent = False
        super().__init__()
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.tracking_audit: ExecutionTrackingAudit | None = None
        self.create_subscription(
            JointTrajectoryControllerState,
            "/right_joint_trajectory_controller/controller_state",
            self._controller_state,
            100,
        )
        self.gripper_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_gripper_controller/follow_joint_trajectory",
        )

    def _joint_state(self, message: JointState) -> None:
        super()._joint_state(message)
        indices = {name: index for index, name in enumerate(message.name)}
        index = indices.get(RIGHT_GRIPPER_JOINT_NAME)
        if index is None or index >= len(message.position):
            return
        position = float(message.position[index])
        if not math.isfinite(position):
            return
        self.gripper_position = position
        self.gripper_state_monotonic_s = time.monotonic()

    def open_home_gripper(self, config: dict) -> dict:
        gripper = config["home_gripper"]
        target = float(gripper["open_position_rad"])
        tolerance = float(gripper["position_tolerance_rad"])
        server_timeout = float(gripper["server_timeout_s"])
        result_timeout = float(gripper["result_timeout_s"])
        stable_required = float(gripper["stable_duration_s"])
        if not self.gripper_action_client.wait_for_server(timeout_sec=server_timeout):
            raise RuntimeError("right gripper trajectory action is unavailable")

        trajectory = JointTrajectory()
        trajectory.joint_names = [RIGHT_GRIPPER_JOINT_NAME]
        point = JointTrajectoryPoint()
        point.positions = [target]
        duration_s = float(gripper["command_duration_s"])
        point.time_from_start.sec = int(duration_s)
        point.time_from_start.nanosec = int((duration_s % 1.0) * 1.0e9)
        trajectory.points = [point]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        sent = self.gripper_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=server_timeout)
        handle = sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("right gripper OPEN goal was not accepted")
        self.gripper_goal_sent = True
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=result_timeout)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("right gripper OPEN action timed out")
        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            raise RuntimeError(
                "right gripper OPEN action failed: "
                f"status={wrapped.status} error_code={wrapped.result.error_code}"
            )

        started = time.monotonic()
        deadline = started + result_timeout
        stable_since = None
        samples = 0
        final_position = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            if (
                self.gripper_position is None
                or self.gripper_state_monotonic_s is None
                or now - self.gripper_state_monotonic_s > 0.5
            ):
                stable_since = None
                continue
            samples += 1
            final_position = float(self.gripper_position)
            if abs(final_position - target) <= tolerance:
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= stable_required:
                    return {
                        "accepted": True,
                        "joint_name": RIGHT_GRIPPER_JOINT_NAME,
                        "target_position_rad": target,
                        "measured_position_rad": final_position,
                        "position_error_rad": final_position - target,
                        "position_tolerance_rad": tolerance,
                        "stable_duration_s": stable_required,
                        "sample_count": samples,
                        "action_status": int(wrapped.status),
                        "controller_error_code": int(wrapped.result.error_code),
                        "command_trajectory": message_to_ordereddict(trajectory),
                    }
            else:
                stable_since = None
        raise RuntimeError(
            "right gripper OPEN measured acceptance failed: "
            f"target={target:.3f} measured={final_position!r} tolerance={tolerance:.3f}"
        )

    def _controller_state(self, message: JointTrajectoryControllerState) -> None:
        """Record the controller's actual set point and feedback in canonical order."""
        if self.tracking_audit is None:
            return
        indices = {name: index for index, name in enumerate(message.joint_names)}
        if any(name not in indices for name in RIGHT_ARM_JOINT_NAMES):
            return
        if len(message.reference.positions) != len(message.joint_names):
            return
        if len(message.feedback.positions) != len(message.joint_names):
            return
        commanded = [
            float(message.reference.positions[indices[name]])
            for name in RIGHT_ARM_JOINT_NAMES
        ]
        measured = [
            float(message.feedback.positions[indices[name]])
            for name in RIGHT_ARM_JOINT_NAMES
        ]
        stamp = message.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        self.tracking_audit.add_sample(
            timestamp_s,
            commanded,
            measured,
            measured_velocity=self.current_velocities,
            measured_effort=self.current_efforts,
        )

    def begin_tracking_audit(self) -> None:
        self.tracking_audit = ExecutionTrackingAudit()

    def finish_tracking_audit(self) -> dict:
        audit = self.tracking_audit or ExecutionTrackingAudit()
        self.tracking_audit = None
        return audit.as_dict()

    def execution_table_guard_failure(self, now=None) -> str:
        # This workflow owns an immutable fixed collision object rather than a visual
        # status heartbeat.  Its exact PlanningScene snapshot is checked immediately
        # before sending the action goal.
        return ""

    def apply_static_collision(self, config: dict, timeout_s: float) -> dict:
        require_current_table_collision(config)
        collision = config["static_collision"]
        if not self.apply_scene_client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("/apply_planning_scene is unavailable")
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(value) for value in collision["dimensions_xyz_m"]]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [
            float(value) for value in collision["center_xyz_m"]
        ]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = [float(value) for value in collision["orientation_xyzw"]]
        item = CollisionObject()
        item.header.frame_id = str(collision["frame_id"])
        # In MoveIt 2 the primitive pose is relative to CollisionObject.pose.
        # geometry_msgs/Pose defaults to the invalid all-zero quaternion, so keep
        # the object frame explicitly identical to the header frame.
        item.pose.orientation.w = 1.0
        item.id = str(collision["object_id"])
        item.primitives = [primitive]
        item.primitive_poses = [pose]
        item.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [item]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("MoveIt refused the fixed inflated table collision object")
        return {
            "id": item.id,
            "frame_id": item.header.frame_id,
            "center_xyz_m": list(collision["center_xyz_m"]),
            "orientation_xyzw": list(collision["orientation_xyzw"]),
            "dimensions_xyz_m": list(collision["dimensions_xyz_m"]),
            "top_z_m": float(collision["top_z_m"]),
            "source": str(collision["source"]),
        }


def verify_static_collision_snapshot(snapshot: dict, configured: dict) -> None:
    """Require the effective PlanningScene geometry to equal the fixed guard.

    MoveIt may normalize ``CollisionObject.pose`` and ``primitive_poses`` by
    moving the primitive transform into the object pose.  Those encodings are
    geometrically equivalent, so compare their composed header-frame pose.
    """

    effective = effective_primitive_pose(snapshot)
    checks = (
        ("id", snapshot.get("id"), configured["object_id"]),
        ("frame_id", snapshot.get("frame_id"), configured["frame_id"]),
        ("dimensions_m", snapshot.get("dimensions_m"), configured["dimensions_xyz_m"]),
        (
            "effective_position_m",
            effective["position_m"],
            configured["center_xyz_m"],
        ),
    )
    for label, actual, expected in checks:
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected) or any(
                not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-6)
                for a, b in zip(actual, expected)
            ):
                raise RuntimeError(
                    f"fixed collision snapshot {label}={actual!r}, expected {expected!r}"
                )
        elif actual != expected:
            raise RuntimeError(
                f"fixed collision snapshot {label}={actual!r}, expected {expected!r}"
            )
    actual_orientation = effective["orientation_xyzw"]
    expected_orientation = [float(value) for value in configured["orientation_xyzw"]]
    dot = abs(sum(a * b for a, b in zip(actual_orientation, expected_orientation)))
    if not math.isclose(dot, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise RuntimeError(
            "fixed collision snapshot effective_orientation_xyzw="
            f"{actual_orientation!r}, expected {expected_orientation!r}"
        )


def _unit_quaternion(values, label: str) -> list[float]:
    quaternion = [float(value) for value in values]
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise RuntimeError(f"fixed collision snapshot {label} is invalid")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-12:
        raise RuntimeError(f"fixed collision snapshot {label} has zero norm")
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise RuntimeError(f"fixed collision snapshot {label} is not normalized")
    return [value / norm for value in quaternion]


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _rotate_vector(quaternion: list[float], vector: list[float]) -> list[float]:
    pure = [float(vector[0]), float(vector[1]), float(vector[2]), 0.0]
    conjugate = [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
    return _quaternion_multiply(
        _quaternion_multiply(quaternion, pure), conjugate
    )[:3]


def effective_primitive_pose(snapshot: dict) -> dict:
    """Compose object and primitive poses into the collision header frame."""

    object_pose = snapshot.get("object_pose") or {}
    primitive_pose = snapshot.get("primitive_pose") or {}
    object_position = object_pose.get("position_m")
    primitive_position = primitive_pose.get("position_m")
    if (
        not isinstance(object_position, list)
        or len(object_position) != 3
        or not isinstance(primitive_position, list)
        or len(primitive_position) != 3
    ):
        raise RuntimeError("fixed collision snapshot pose positions are invalid")
    if not all(
        math.isfinite(float(value))
        for value in [*object_position, *primitive_position]
    ):
        raise RuntimeError("fixed collision snapshot pose positions are non-finite")
    object_orientation = _unit_quaternion(
        object_pose.get("orientation_xyzw", []), "object orientation"
    )
    primitive_orientation = _unit_quaternion(
        primitive_pose.get("orientation_xyzw", []), "primitive orientation"
    )
    rotated = _rotate_vector(
        object_orientation, [float(value) for value in primitive_position]
    )
    orientation = _unit_quaternion(
        _quaternion_multiply(object_orientation, primitive_orientation),
        "effective orientation",
    )
    return {
        "position_m": [
            float(object_position[index]) + rotated[index] for index in range(3)
        ],
        "orientation_xyzw": orientation,
    }


def wait_for_home_settle(
    node,
    goal: list[float],
    config: dict,
    *,
    joint_tolerance_rad: float | None = None,
) -> dict:
    acceptance = config["final_acceptance"]
    timeout_s = float(acceptance["settle_timeout_s"])
    stable_required = float(acceptance["stable_duration_s"])
    tolerance = (
        float(acceptance["joint_tolerance_rad"])
        if joint_tolerance_rad is None
        else float(joint_tolerance_rad)
    )
    service_timeout = float(config["planning"]["service_timeout_s"])
    started = time.monotonic()
    deadline = started + timeout_s
    stable_since = None
    first = best = final = None
    samples = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 0.0 and final is not None:
            break
        measured = node.refresh_rigid_state(
            min(service_timeout, max(remaining, 0.05))
        )
        now = time.monotonic()
        errors = [float(actual) - target for actual, target in zip(measured, goal)]
        sample = {
            "elapsed_s": now - started,
            "measured_positions_rad": measured,
            "joint_errors_rad": errors,
            "max_joint_error_rad": max(abs(value) for value in errors),
        }
        samples += 1
        first = deepcopy(sample) if first is None else first
        final = sample
        if best is None or sample["max_joint_error_rad"] < best["max_joint_error_rad"]:
            best = deepcopy(sample)
        if sample["max_joint_error_rad"] <= tolerance:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= stable_required:
                return {
                    "accepted": True,
                    "joint_tolerance_rad": tolerance,
                    "sample_count": samples,
                    "time_to_acceptance_s": now - started,
                    "first_sample": first,
                    "best_sample": best,
                    "final_sample": final,
                }
        else:
            stable_since = None
        if now >= deadline:
            break
    return {
        "accepted": False,
        "joint_tolerance_rad": tolerance,
        "sample_count": samples,
        "time_to_acceptance_s": None,
        "first_sample": first,
        "best_sample": best,
        "final_sample": final,
    }


def _sample_trajectory_positions(trajectory, maximum: int = 80) -> list[list[float]]:
    points = trajectory.joint_trajectory.points
    if not points:
        return []
    if len(points) <= maximum:
        return [[float(value) for value in point.positions] for point in points]
    indices = sorted(
        {round(index * (len(points) - 1) / (maximum - 1)) for index in range(maximum)}
    )
    return [
        [float(value) for value in points[index].positions] for index in indices
    ]


def trajectory_path_quality(node, trajectory, config: dict, timeout_s: float) -> dict:
    q_path = _sample_trajectory_positions(trajectory)
    payload = {
        "joint_names": list(RIGHT_ARM_JOINT_NAMES),
        "world_frame": config["station_frame"]["frame_id"],
        "end_effector_link": "openarm_right_ee_base_link",
    }
    tcp = []
    for positions in q_path:
        pose = node.fk_pose(payload, positions, timeout_s).pose.position
        tcp.append([float(pose.x), float(pose.y), float(pose.z)])
    station = config["station_frame"]
    result = path_quality(
        q_path,
        tcp,
        {
            "forward": station["forward_world"],
            "lateral": station["lateral_world"],
            "vertical": station["vertical_world"],
        },
        minimum_clearance_m=None,
    )
    result["minimum_clearance"] = {
        "value_m": None,
        "status": "unavailable_from_MoveIt_ROS_service_API",
        "authorization_source": "real_motion_validated_fixed_side_route",
    }
    return result


def directional_quality_gate(metrics: dict, config: dict) -> dict:
    gates = config["side_prehome"]["path_quality_gates"]
    checks = {
        "maximum_backward_excursion_m": metrics["maximum_backward_excursion_m"]
        <= float(gates["maximum_backward_excursion_m"]),
        "maximum_lateral_excursion_m": metrics["maximum_lateral_excursion_m"]
        <= float(gates["maximum_lateral_excursion_m"]),
        "maximum_vertical_excursion_m": metrics["maximum_vertical_excursion_m"]
        <= float(gates["maximum_vertical_excursion_m"]),
        "maximum_absolute_j4_rad": max(abs(value) for value in metrics["j4_range_rad"])
        <= float(gates["maximum_absolute_j4_rad"]),
    }
    return {"accepted": all(checks.values()), "checks": checks, "configured_gates": gates}


def parse_arguments(argv=None):
    share = Path(get_package_share_directory("openarm_kinesthetic_recorder"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=share / "config/home_initialization.yaml")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("~/.ros/openarm_kinesthetic/home_initialization.execution.json").expanduser(),
    )
    parser.add_argument("--allow-real-execution", action="store_true")
    return parser.parse_args(argv)


def parse_process_arguments(argv=None):
    process_argv = sys.argv if argv is None else argv
    return parse_arguments(remove_ros_args(args=process_argv)[1:])


def main() -> int:
    args = parse_process_arguments()
    config = load_home_config(args.config)
    report = {
        "schema_version": 1,
        "workflow": "FIXED_LIFTED_HOME",
        "state": "VALIDATING",
        "success": False,
        "j8_policy": "open_after_arm_home",
        "j8_commanded_by_initialization": False,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config_snapshot": deepcopy(config),
        "started_wall_time_s": time.time(),
    }
    rclpy.init()
    node = HomeInitializationNode()
    try:
        if not args.allow_real_execution:
            raise RuntimeError("--allow-real-execution is required")
        # Validate that the fixed guard matches the current table epoch.  Physical
        # approval metadata is not an execution prerequisite; the essential motion
        # gates below still apply the guard and verify its exact PlanningScene readback.
        require_current_table_collision(config)
        report["runtime_authorization_scope"] = "ATTENDED_REAL_TEST_CURRENT_GUARD"
        planning = config["planning"]
        timeout_s = float(planning["service_timeout_s"])
        goal = [float(value) for value in config["right_home_joint_positions_rad"]]
        names = list(config["right_joint_names"])
        report["controllers"] = {
            "right_arm": node.verify_controller_active(
                "right_joint_trajectory_controller", timeout_s
            ),
            "right_gripper": node.verify_controller_active(
                "right_gripper_controller", timeout_s
            ),
        }
        selected = config["side_prehome"]["selected_joint_positions_rad"]
        selected_candidate_id = config["side_prehome"]["selected_candidate_id"]
        prehome = [float(value) for value in selected]
        start = node.refresh_rigid_state(timeout_s)
        start_error = max(abs(float(value)) for value in start)
        if start_error > float(config["maximum_start_from_zero_error_rad"]):
            raise RuntimeError(
                f"HOME_START_NOT_ZERO: {start_error:.6f} rad exceeds "
                f"{float(config['maximum_start_from_zero_error_rad']):.6f}"
            )
        collision = node.apply_static_collision(config, timeout_s)
        snapshot = node.guard_snapshot(
            collision["id"], collision["dimensions_xyz_m"], timeout_s
        )
        verify_static_collision_snapshot(snapshot, config["static_collision"])
        effective_snapshot = {
            "id": snapshot["id"],
            "frame_id": snapshot["frame_id"],
            "dimensions_m": snapshot["dimensions_m"],
            "effective_pose": effective_primitive_pose(snapshot),
        }
        report["stages"] = []
        report.update(
            {
                "state": "STAGE_A_PLANNING",
                "start_positions_rad": start,
                "start_max_abs_rad": start_error,
                "safe_side_prehome_positions_rad": prehome,
                "safe_side_prehome_candidate_id": selected_candidate_id,
                "side_prehome_source": "fixed_joint_target_fresh_planned_at_runtime",
                "home_goal_positions_rad": goal,
                "static_collision": collision,
                "static_collision_snapshot": snapshot,
                "static_collision_effective_geometry": effective_snapshot,
                "static_collision_sha256": _hash(effective_snapshot),
            }
        )

        def plan_and_execute_stage(
            stage_name: str,
            stage_start: list[float],
            stage_goal: list[float],
            *,
            acceptance_tolerance_rad: float,
        ) -> dict:
            stage = {
                "name": stage_name,
                "state": "PLANNING",
                "start_positions_rad": list(stage_start),
                "goal_positions_rad": list(stage_goal),
                "acceptance_tolerance_rad": float(acceptance_tolerance_rad),
            }
            report["stages"].append(stage)
            # Every stage begins with fresh controller and guard readback.
            stage["controller_before_plan"] = node.verify_controller_active(
                "right_joint_trajectory_controller", timeout_s
            )
            plan_snapshot = node.guard_snapshot(
                collision["id"], collision["dimensions_xyz_m"], timeout_s
            )
            verify_static_collision_snapshot(plan_snapshot, config["static_collision"])
            payload = {
                "joint_names": names,
                "start_positions_rad": list(stage_start),
                "move_group": "right_arm",
            }
            response = node.plan(
                payload,
                stage_goal,
                timeout_s,
                velocity_scaling=float(planning["velocity_scaling"]),
                acceleration_scaling=float(planning["acceleration_scaling"]),
            )
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(
                    f"{stage_name} MoveIt planning failed: {response.error_code.val}"
                )
            trajectory = deepcopy(response.trajectory)
            audit = audit_joint_trajectory(
                trajectory.joint_trajectory, names, stage_start, stage_goal, 0.01
            )
            quality = trajectory_path_quality(node, trajectory, config, timeout_s)
            quality_gate = directional_quality_gate(quality, config)
            if not quality_gate["accepted"]:
                raise RuntimeError(f"{stage_name} path-quality gate failed")
            node.publish_display(response)
            stage.update(
                {
                    "state": "PLANNED",
                    "trajectory": audit.as_dict(),
                    "trajectory_sha256": trajectory_sha256(trajectory),
                    # Preserve the exact timed plan sent to ExecuteTrajectory.
                    # Path-quality samples and a hash cannot reconstruct it.
                    "robot_trajectory": message_to_ordereddict(trajectory),
                    "path_quality": quality,
                    "path_quality_gate": quality_gate,
                    "collision_guard_snapshot_before_plan": plan_snapshot,
                }
            )
            _write_report(args.report, report)
            stage["controller_before_execution"] = node.verify_controller_active(
                "right_joint_trajectory_controller", timeout_s
            )
            current = node.refresh_rigid_state(timeout_s)
            changed = max(abs(a - b) for a, b in zip(current, stage_start))
            if changed > 0.01:
                raise RuntimeError(f"{stage_name}_START_CHANGED: {changed:.6f} rad")
            current_snapshot = node.guard_snapshot(
                collision["id"], collision["dimensions_xyz_m"], timeout_s
            )
            verify_static_collision_snapshot(current_snapshot, config["static_collision"])
            current_effective_snapshot = {
                "id": current_snapshot["id"],
                "frame_id": current_snapshot["frame_id"],
                "dimensions_m": current_snapshot["dimensions_m"],
                "effective_pose": effective_primitive_pose(current_snapshot),
            }
            if _hash(current_effective_snapshot) != _hash(effective_snapshot):
                raise RuntimeError("fixed inflated table collision changed before execution")
            stage["state"] = "EXECUTING"
            stage["automatic_execution_authorized"] = True
            _write_report(args.report, report)
            node.begin_tracking_audit()
            stage["execute_called_wall_time_ns"] = time.time_ns()
            stage["execute_called_monotonic_ns"] = time.monotonic_ns()
            try:
                execution = node.execute_same_trajectory(
                    trajectory, audit.duration_s, timeout_margin_s=30.0
                )
            finally:
                stage["execute_returned_wall_time_ns"] = time.time_ns()
                stage["execute_returned_monotonic_ns"] = time.monotonic_ns()
                stage["execution_tracking_audit"] = node.finish_tracking_audit()
            stage["execution"] = execution
            if (
                execution["action_status"] != GoalStatus.STATUS_SUCCEEDED
                or execution["moveit_error_code"] != MoveItErrorCodes.SUCCESS
            ):
                raise RuntimeError(f"{stage_name} MoveIt execution failed")
            settle = wait_for_home_settle(
                node,
                stage_goal,
                config,
                joint_tolerance_rad=acceptance_tolerance_rad,
            )
            stage["final_settling"] = settle
            if not settle["accepted"]:
                raise RuntimeError(f"{stage_name} final measured joint acceptance failed")
            stage["state"] = "ACCEPTED"
            _write_report(args.report, report)
            return stage

        stage_a = plan_and_execute_stage(
            "STAGE_A_ZERO_TO_SAFE_SIDE_PREHOME",
            start,
            prehome,
            acceptance_tolerance_rad=STAGE_A_TRANSIT_JOINT_TOLERANCE_RAD,
        )
        # Explicitly discard the commanded endpoint.  A new measured state is the only
        # permitted Stage B start, even after Stage A acceptance succeeded.
        stage_b_start = node.refresh_rigid_state(timeout_s)
        report["stage_b_fresh_measured_start_positions_rad"] = stage_b_start
        report["state"] = "STAGE_B_PLANNING"
        _write_report(args.report, report)
        stage_b = plan_and_execute_stage(
            "STAGE_B_SAFE_SIDE_PREHOME_TO_LIFTED_HOME",
            stage_b_start,
            goal,
            acceptance_tolerance_rad=float(
                config["final_acceptance"]["joint_tolerance_rad"]
            ),
        )
        report["trajectory_sha256"] = _hash(
            [stage_a["trajectory_sha256"], stage_b["trajectory_sha256"]]
        )
        report["execution_tracking_audit"] = {
            "stage_a": stage_a["execution_tracking_audit"],
            "stage_b": stage_b["execution_tracking_audit"],
        }
        report["final_settling"] = stage_b["final_settling"]
        report["execution_client_goal_sent"] = bool(node.execution_goal_sent)
        report["state"] = "OPENING_HOME_GRIPPER"
        _write_report(args.report, report)
        report["home_gripper"] = node.open_home_gripper(config)
        report["post_h2_measured_arm_positions_rad"] = list(node.current_positions)
        report["post_h2_arm_feedback_age_s"] = time.monotonic() - node.current_state_monotonic_s
        report["post_h2_measured_gripper_position_rad"] = node.gripper_position
        report["j8_commanded_by_initialization"] = bool(node.gripper_goal_sent)
        report["state"] = "HOME_READY"
        report["success"] = True
        report["completed_wall_time_s"] = time.time()
        _write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("HOME_READY: J8 is OPEN; teaching and recording may start")
        return 0
    except Exception as error:
        report["state"] = "FAIL_RIGID"
        report["success"] = False
        report["failure_reason"] = str(error)
        report["completed_wall_time_s"] = time.time()
        report["execution_client_goal_sent"] = bool(node.execution_goal_sent)
        report["j8_commanded_by_initialization"] = bool(node.gripper_goal_sent)
        _write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"FAIL_RIGID: {error}", file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
