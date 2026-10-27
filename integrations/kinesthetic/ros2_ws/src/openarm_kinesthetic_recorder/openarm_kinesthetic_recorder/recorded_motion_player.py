"""Validate, plan, and optionally execute a recorded right-arm motion."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetStateValidity
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .execution_guard import audit_joint_trajectory, duration_seconds
from .home_guarded_executor import verify_static_collision_snapshot
from .home_measurements import Q_LIFTED_HOME_RIGHT
from .home_runtime import HomeRuntimeNode, validate_real_execution_scaling
from .recorded_motion import (
    COLLECTION_REPLAY_VELOCITY_LIMITS_RAD_S,
    GRIPPER_NAME,
    JOINT_NAMES,
    audit_recorded_path,
    interpolated_arm_states,
    is_keyframe_motion,
    is_marked_motion,
    load_recorded_motion,
    marker_summary,
    marker_segment_ranges,
    playback_time_scale,
    validate_collection_trajectory,
    validate_recorded_motion,
)


ARM_NAMES = JOINT_NAMES[:7]
GRIPPER_OPEN_RAD = -0.35
GRIPPER_CLOSE_RAD = -0.10


def replay_provenance(
    trajectory_path: Path,
    payload: dict,
    playback_rate: float,
    ranges: list[dict],
    keyframe_settle_tolerance_rad: float = 0.03,
) -> dict:
    """Build the immutable provenance sent to an active episode recorder."""
    resolved = trajectory_path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    if validate_recorded_motion(json.loads(raw)) != payload:
        raise RuntimeError("trajectory file changed after it was loaded")
    first_sample = int(ranges[0]["start_sample_index"])
    last_sample = int(ranges[-1]["end_sample_index"])
    selected_markers = [
        marker
        for segment in ranges
        for marker in segment["markers"]
    ]
    return {
        "schema_version": 1,
        "trajectory_path": str(resolved),
        "trajectory_sha256": hashlib.sha256(raw).hexdigest(),
        "trajectory_schema_version": payload["schema_version"],
        "trajectory_mode": payload["mode"],
        "trajectory_trial_id": payload.get("trial_id"),
        "trajectory_duration_s": float(payload["duration_s"]),
        "trajectory_processing": payload.get("processing"),
        "playback_rate": float(playback_rate),
        "keyframe_settle_tolerance_rad": float(keyframe_settle_tolerance_rad),
        "marker_summary": marker_summary(payload),
        "selected_marker_indices": [
            int(marker["marker_index"]) for marker in selected_markers
        ],
        "selected_sample_range": [first_sample, last_sample],
        "full_replay": (
            first_sample == 0
            and last_sample == len(payload["samples"]) - 1
            and len(selected_markers) == len(payload["markers"])
        ),
    }


def _duration(seconds: float):
    from builtin_interfaces.msg import Duration

    nanoseconds = int(round(float(seconds) * 1e9))
    return Duration(
        sec=nanoseconds // 1_000_000_000,
        nanosec=nanoseconds % 1_000_000_000,
    )


def build_replay_trajectories(
    payload: dict, start_delay_s: float = 0.20, time_scale: float = 1.0
):
    """Build the legacy continuous replay trajectories unchanged."""
    if time_scale < 1.0 or not math.isfinite(time_scale):
        raise ValueError("replay time_scale must be finite and at least one")
    arm = JointTrajectory()
    arm.joint_names = list(ARM_NAMES)
    gripper = JointTrajectory()
    gripper.joint_names = [GRIPPER_NAME]
    for sample in payload["samples"]:
        timestamp = start_delay_s + time_scale * float(sample["time_from_start_s"])
        positions = [float(value) for value in sample["positions_rad"]]
        arm_point = JointTrajectoryPoint()
        arm_point.positions = positions[:7]
        arm_point.time_from_start = _duration(timestamp)
        arm.points.append(arm_point)
        gripper_point = JointTrajectoryPoint()
        gripper_point.positions = [positions[7]]
        gripper_point.time_from_start = _duration(timestamp)
        gripper.points.append(gripper_point)
    return arm, gripper


def build_dense_arm_trajectory(
    samples: list[dict], start_sample_index: int, end_sample_index: int,
    *, time_scale: float, start_delay_s: float = 0.20,
) -> JointTrajectory:
    """Build a controller trajectory containing every taught sample in a range."""
    if time_scale < 1.0 or not math.isfinite(time_scale):
        raise ValueError("time_scale must be finite and at least one")
    if not 0 <= start_sample_index <= end_sample_index < len(samples):
        raise ValueError("dense trajectory sample range is invalid")
    trajectory = JointTrajectory()
    trajectory.joint_names = list(ARM_NAMES)
    base_time = float(samples[start_sample_index]["time_from_start_s"])
    selected = samples[start_sample_index : end_sample_index + 1]
    if len(selected) == 1:
        selected = [selected[0], selected[0]]
    for offset, sample in enumerate(selected):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in sample["positions_rad"][:7]]
        elapsed = float(sample["time_from_start_s"]) - base_time
        if offset and elapsed <= 0.0:
            elapsed = 0.02
        point.time_from_start = _duration(start_delay_s + time_scale * elapsed)
        trajectory.points.append(point)
    return trajectory


def verify_home_provenance(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "HOME_READY" or payload.get("success") is not True:
        raise RuntimeError("latest H2 report is not HOME_READY")
    goal = payload.get("home_goal_positions_rad")
    if not isinstance(goal, list) or len(goal) != 7 or max(
        abs(float(a) - float(b)) for a, b in zip(goal, Q_LIFTED_HOME_RIGHT)
    ) > 1e-9:
        raise RuntimeError("H2 report does not contain the fixed lifted HOME")
    return payload


class RecordedMotionPlayer(HomeRuntimeNode):
    def __init__(self, node_name: str = "recorded_motion_player") -> None:
        super().__init__(node_name)
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.declare_parameter("joint_state_startup_timeout_s", 10.0)
        self.declare_parameter("joint_state_freshness_s", 0.5)
        self.declare_parameter("joint_state_required_consecutive_samples", 3)
        self.keyframe_settle_tolerance_rad = 0.03
        self.joint_state_startup_timeout_s = float(
            self.get_parameter("joint_state_startup_timeout_s").value
        )
        self.joint_state_freshness_s = float(
            self.get_parameter("joint_state_freshness_s").value
        )
        self.joint_state_required_consecutive_samples = int(
            self.get_parameter("joint_state_required_consecutive_samples").value
        )
        if self.joint_state_startup_timeout_s <= 0.0 or self.joint_state_freshness_s <= 0.0 or self.joint_state_required_consecutive_samples < 1:
            raise ValueError("joint-state gate parameters are invalid")
        self.all_positions: dict[str, float] = {}
        self.all_received_at: float | None = None
        self.last_message_received_at: float | None = None
        self.last_complete_source_timestamp_s: float | None = None
        self.total_joint_state_messages = 0
        self.complete_joint_state_messages = 0
        self.consecutive_complete_samples = 0
        self.last_missing_joint_names = list(JOINT_NAMES)
        self.software_stop_latched = False
        self.operator_interrupt_requested = False
        self.active_arm_goal = None
        self.active_gripper_goal = None
        self.publish_reference = False
        self.gripper_reference: float | None = None
        self.expert_action_pub = self.create_publisher(
            JointState, "/openarm_kinesthetic/expert_action", qos
        )
        self.gripper_event_pub = self.create_publisher(
            String, "/openarm_kinesthetic/expert_gripper_event", qos
        )
        replay_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.replay_status_pub = self.create_publisher(
            String, "/openarm_kinesthetic/expert_replay_status", replay_qos
        )
        self.replay_provenance: dict | None = None
        self.create_subscription(
            JointState, "/joint_states", self.on_all_joint_state, qos_profile_sensor_data
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            "/right_joint_trajectory_controller/controller_state",
            self.on_controller_state,
            qos,
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            "/right_gripper_controller/controller_state",
            self.on_gripper_controller_state,
            qos,
        )
        self.create_subscription(
            String,
            "/openarm_kinesthetic/gamepad_status",
            self.on_gamepad_status,
            qos,
        )
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")
        self.gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_gripper_controller/follow_joint_trajectory",
        )
        self.recorder_failure = self.create_client(
            Trigger, "/real_episode_recorder/finish_failure"
        )

    def on_all_joint_state(self, message: JointState) -> None:
        now = time.monotonic()
        self.total_joint_state_messages += 1
        self.last_message_received_at = now
        indices = {str(name): index for index, name in enumerate(message.name)}
        missing = [name for name in JOINT_NAMES if name not in indices or indices[name] >= len(message.position)]
        self.last_missing_joint_names = missing
        if missing:
            self.consecutive_complete_samples = 0
            return
        positions = {name: float(message.position[indices[name]]) for name in JOINT_NAMES}
        if not all(math.isfinite(value) for value in positions.values()):
            self.consecutive_complete_samples = 0
            return
        stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1.0e-9
        if not math.isfinite(stamp) or stamp <= 0.0 or (
            self.last_complete_source_timestamp_s is not None
            and stamp <= self.last_complete_source_timestamp_s
        ):
            self.consecutive_complete_samples = 0
            return
        self.all_positions = positions
        self.all_received_at = now
        self.last_complete_source_timestamp_s = stamp
        self.complete_joint_state_messages += 1
        self.consecutive_complete_samples += 1
        if self.gripper_reference is None:
            self.gripper_reference = self.all_positions[GRIPPER_NAME]

    def on_controller_state(self, message: JointTrajectoryControllerState) -> None:
        if not self.publish_reference or self.gripper_reference is None:
            return
        indices = {name: index for index, name in enumerate(message.joint_names)}
        if any(name not in indices for name in ARM_NAMES):
            return
        if len(message.reference.positions) != len(message.joint_names):
            return
        output = JointState()
        output.header = message.header
        output.name = list(JOINT_NAMES)
        output.position = [
            float(message.reference.positions[indices[name]]) for name in ARM_NAMES
        ] + [float(self.gripper_reference)]
        self.expert_action_pub.publish(output)

    def on_gripper_controller_state(
        self, message: JointTrajectoryControllerState
    ) -> None:
        indices = {name: index for index, name in enumerate(message.joint_names)}
        if (
            GRIPPER_NAME in indices
            and len(message.reference.positions) == len(message.joint_names)
        ):
            value = float(message.reference.positions[indices[GRIPPER_NAME]])
            if math.isfinite(value):
                self.gripper_reference = value

    def on_gamepad_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if payload.get("connected") is True and payload.get("stop_latched") is True:
            self.software_stop_latched = True

    def execution_table_guard_failure(self, now: float | None = None) -> str:
        if self.operator_interrupt_requested:
            return "operator interrupt requested"
        if self.software_stop_latched:
            return "gamepad RT/RB software stop latched"
        return ""

    def request_operator_interrupt(self) -> None:
        self.operator_interrupt_requested = True

    def stop_reason(self) -> str:
        if getattr(self, "operator_interrupt_requested", False):
            return "OPERATOR_INTERRUPT"
        if getattr(self, "software_stop_latched", False):
            return "SOFTWARE_STOP"
        return ""

    def cancel_goal(self, handle, label: str, timeout_s: float = 2.0) -> bool:
        if handle is None:
            return True
        try:
            future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
            response = future.result() if future.done() else None
            accepted = bool(response is not None and response.goals_canceling)
        except Exception as error:
            self.get_logger().error(f"{label}_CANCEL_FAILED: {error}")
            return False
        if accepted:
            self.get_logger().warn(f"{label}_CANCEL_ACCEPTED")
        else:
            self.get_logger().error(f"{label}_CANCEL_NOT_CONFIRMED")
        return accepted

    def cancel_active_goals(self) -> bool:
        arm = self.cancel_goal(self.active_arm_goal, "RIGHT_ARM_GOAL")
        gripper = self.cancel_goal(self.active_gripper_goal, "RIGHT_GRIPPER_GOAL")
        return arm and gripper

    def joint_state_gate_diagnostic(self) -> str:
        now = time.monotonic()
        age = None if self.last_message_received_at is None else now - self.last_message_received_at
        complete_age = None if self.all_received_at is None else now - self.all_received_at
        publisher_count = self.count_publishers("/joint_states")
        if publisher_count == 0:
            reason = "no_publisher"
        elif self.total_joint_state_messages == 0:
            reason = "no_messages"
        elif self.complete_joint_state_messages == 0:
            reason = "incomplete_or_invalid_messages"
        elif self.consecutive_complete_samples == 0:
            reason = "timestamp_not_increasing_or_incomplete"
        elif complete_age is not None and complete_age > self.joint_state_freshness_s:
            reason = "complete_state_stale"
        else:
            reason = "insufficient_consecutive_samples"
        return (
            "reason=%s publisher_count=%d total_messages_received=%d "
            "complete_messages_received=%d consecutive_complete_samples=%d "
            "missing_joint_names=%s last_message_age_s=%s last_complete_message_age_s=%s "
            "subscriber_qos=sensor_data(best_effort,volatile) timeout_s=%.3f freshness_s=%.3f required_consecutive=%d"
            % (
                reason, publisher_count, self.total_joint_state_messages,
                self.complete_joint_state_messages, self.consecutive_complete_samples,
                self.last_missing_joint_names,
                "none" if age is None else f"{age:.3f}",
                "none" if complete_age is None else f"{complete_age:.3f}",
                self.joint_state_startup_timeout_s, self.joint_state_freshness_s,
                self.joint_state_required_consecutive_samples,
            )
        )

    def wait_for_full_joint_state(self, timeout_s: float | None = None) -> None:
        timeout_s = self.joint_state_startup_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            reason = RecordedMotionPlayer.stop_reason(self)
            if reason:
                raise RuntimeError(reason)
            if (
                self.consecutive_complete_samples >= self.joint_state_required_consecutive_samples
                and self.all_received_at is not None
                and time.monotonic() - self.all_received_at <= self.joint_state_freshness_s
            ):
                return
        raise RuntimeError(
            "fresh right J1-J8 joint state unavailable: " + self.joint_state_gate_diagnostic()
        )

    def validate_state(self, positions: list[float], label: str) -> None:
        if not self.validity.wait_for_service(timeout_sec=1.0):
            raise RuntimeError("/check_state_validity unavailable")
        state = dict(self.all_positions)
        state.update(zip(JOINT_NAMES, positions))
        request = GetStateValidity.Request()
        request.group_name = "right_arm"
        request.robot_state = RobotState()
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(state)
        request.robot_state.joint_state.position = list(state.values())
        future = self.validity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        response = future.result() if future.done() else None
        if response is None:
            raise RuntimeError(f"state validity timed out at {label}")
        if not response.valid:
            raise RuntimeError(f"collision/constraint rejection at {label}")

    def verify_guard(self, home_report: dict, timeout_s: float = 10.0) -> dict:
        configured = home_report.get("static_collision")
        if not isinstance(configured, dict):
            raise RuntimeError("H2 report has no static collision contract")
        snapshot = self.guard_snapshot(
            str(configured["id"]), list(configured["dimensions_xyz_m"]), timeout_s
        )
        expected = {
            "object_id": configured["id"],
            "frame_id": configured["frame_id"],
            "dimensions_xyz_m": configured["dimensions_xyz_m"],
            "center_xyz_m": configured["center_xyz_m"],
            "orientation_xyzw": configured["orientation_xyzw"],
        }
        verify_static_collision_snapshot(snapshot, expected)
        return snapshot

    def plan_segment(
        self,
        start: list[float],
        goal: list[float],
        velocity_scaling: float,
        acceleration_scaling: float,
    ):
        response = self.plan(
            {
                "joint_names": list(ARM_NAMES),
                "start_positions_rad": list(start),
                "move_group": "right_arm",
            },
            list(goal),
            10.0,
            velocity_scaling,
            acceleration_scaling,
        )
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt planning failed: {response.error_code.val}")
        trajectory = deepcopy(response.trajectory)
        audit = audit_joint_trajectory(
            trajectory.joint_trajectory,
            ARM_NAMES,
            start,
            goal,
            0.01,
        )
        for point_index, point in enumerate(trajectory.joint_trajectory.points):
            state = [*map(float, point.positions), float(self.gripper_reference or 0.0)]
            self.validate_state(state, f"planned point {point_index}")
        self.publish_display(response)
        return trajectory, audit

    def plan_keyframes(
        self,
        payload: dict,
        home_report: dict,
        velocity_scaling: float,
        acceleration_scaling: float,
    ) -> list[tuple[object, object]]:
        self.verify_guard(home_report)
        planned = []
        for index in range(len(payload["samples"]) - 1):
            start = [float(value) for value in payload["samples"][index]["positions_rad"][:7]]
            goal = [float(value) for value in payload["samples"][index + 1]["positions_rad"][:7]]
            try:
                trajectory, audit = self.plan_segment(
                    start, goal, velocity_scaling, acceleration_scaling
                )
            except Exception as error:
                raise RuntimeError(f"KEYFRAME_SEGMENT_{index}_FAILED: {error}") from error
            planned.append((trajectory, audit))
            self.get_logger().warn(
                "KEYFRAME_SEGMENT_PLANNED index=%d from=%d to=%d duration_s=%.3f points=%d"
                % (index, index, index + 1, audit.duration_s, audit.point_count)
            )
        return planned

    def command_gripper(self, event: str) -> None:
        if event == "none":
            return
        target = GRIPPER_CLOSE_RAD if event == "close" else GRIPPER_OPEN_RAD
        if not self.gripper_action.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("right gripper FollowJointTrajectory action unavailable")
        trajectory = JointTrajectory()
        trajectory.joint_names = [GRIPPER_NAME]
        point = JointTrajectoryPoint()
        point.positions = [target]
        point.time_from_start = _duration(0.5)
        trajectory.points = [point]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.gripper_reference = target
        output = String()
        output.data = json.dumps(
            {
                "event": event,
                "target_position_rad": target,
                "timestamp_s": self.get_clock().now().nanoseconds * 1e-9,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self.gripper_event_pub.publish(output)
        goal_future = self.gripper_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=3.0)
        handle = goal_future.result() if goal_future.done() else None
        if handle is None or not handle.accepted:
            raise RuntimeError(f"right gripper {event} goal rejected")
        self.active_gripper_goal = handle
        result = handle.get_result_async()
        deadline = time.monotonic() + 5.0
        cancel_requested = False
        try:
            while not result.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                reason = self.stop_reason()
                if reason:
                    cancel_requested = True
                    self.cancel_goal(handle, "RIGHT_GRIPPER_GOAL")
                    raise RuntimeError(
                        f"right gripper {event} stopped: {reason}"
                    )
            if not result.done():
                cancel_requested = True
                self.cancel_goal(handle, "RIGHT_GRIPPER_GOAL")
                raise RuntimeError(f"right gripper {event} execution timed out")
            if result.result().status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"right gripper {event} execution failed")
        except BaseException:
            if not cancel_requested and not result.done():
                self.cancel_goal(handle, "RIGHT_GRIPPER_GOAL")
            raise
        finally:
            self.active_gripper_goal = None
        self.get_logger().warn(f"KEYFRAME_GRIPPER_EXECUTED event={event}")

    def wait_for_arm_target(
        self, target: list[float], timeout_s: float = 2.0, tolerance_rad: float = 0.03
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (
                self.all_received_at is not None
                and time.monotonic() - self.all_received_at <= 0.5
                and all(name in self.all_positions for name in ARM_NAMES)
            ):
                error = max(
                    abs(self.all_positions[name] - float(required))
                    for name, required in zip(ARM_NAMES, target)
                )
                if error <= tolerance_rad:
                    return
        errors = [
            abs(self.all_positions[name] - float(required))
            for name, required in zip(ARM_NAMES, target)
        ]
        details = ", ".join(
            "%s(actual=%.6f,target=%.6f,error=%.6f)"
            % (name, self.all_positions[name], float(required), error)
            for name, required, error in zip(ARM_NAMES, target, errors)
        )
        raise RuntimeError(
            "measured arm did not settle at keyframe target: "
            f"max_error_rad={max(errors):.6f} tolerance_rad={tolerance_rad:.6f} "
            f"joints=[{details}]"
        )

    def publish_replay_status(self, state: str) -> None:
        if self.replay_provenance is None:
            raise RuntimeError("replay provenance is unavailable")
        message = String()
        message.data = json.dumps(
            {
                **self.replay_provenance,
                "state": state,
                "timestamp_s": time.time(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self.replay_status_pub.publish(message)
        # Give DDS discovery/acknowledgement a chance before this short-lived
        # player process exits; TRANSIENT_LOCAL also protects a late callback.
        rclpy.spin_once(self, timeout_sec=0.05)

    def execute_keyframes(
        self,
        payload: dict,
        planned: list[tuple[object, object]],
        start_waypoint: int,
        stop_after_waypoint: int | None,
        home_report: dict,
    ) -> None:
        if start_waypoint < 0 or start_waypoint >= len(payload["samples"]):
            raise ValueError("start_waypoint is outside the recorded waypoint range")
        final_waypoint = len(payload["samples"]) - 1
        if stop_after_waypoint is not None:
            if stop_after_waypoint <= start_waypoint or stop_after_waypoint > final_waypoint:
                raise ValueError("stop_after_waypoint must be after start_waypoint")
            final_waypoint = stop_after_waypoint
        current = [self.all_positions[name] for name in ARM_NAMES]
        required = payload["samples"][start_waypoint]["positions_rad"][:7]
        error = max(abs(float(a) - float(b)) for a, b in zip(current, required))
        if error > 0.06:
            raise RuntimeError(
                f"current arm differs from start waypoint {start_waypoint}: {error:.6f} rad"
            )
        self.verify_controller_active("right_joint_trajectory_controller", 10.0)
        self.verify_controller_active("right_gripper_controller", 10.0)
        self.publish_reference = True
        try:
            for segment_index in range(start_waypoint, final_waypoint):
                self.verify_guard(home_report)
                current = self.refresh_rigid_state(10.0)
                expected = payload["samples"][segment_index]["positions_rad"][:7]
                changed = max(abs(float(a) - float(b)) for a, b in zip(current, expected))
                if changed > 0.06:
                    raise RuntimeError(
                        f"KEYFRAME_SEGMENT_{segment_index}_START_CHANGED: {changed:.6f} rad"
                    )
                trajectory, audit = planned[segment_index]
                result = self.execute_same_trajectory(
                    trajectory, audit.duration_s, timeout_margin_s=30.0
                )
                if (
                    result["action_status"] != GoalStatus.STATUS_SUCCEEDED
                    or result["moveit_error_code"] != MoveItErrorCodes.SUCCESS
                ):
                    raise RuntimeError(
                        f"KEYFRAME_SEGMENT_{segment_index}_EXECUTION_FAILED"
                    )
                self.wait_for_arm_target(
                    [
                        float(value)
                        for value in payload["samples"][segment_index + 1][
                            "positions_rad"
                        ][:7]
                    ],
                    tolerance_rad=self.keyframe_settle_tolerance_rad,
                )
                event = str(
                    payload["samples"][segment_index + 1].get("gripper_event", "none")
                )
                self.command_gripper(event)
        finally:
            self.publish_reference = False

    def validate_marked_path(
        self, payload: dict, home_report: dict, playback_rate: float,
        maximum_sample_gap_s: float, maximum_sample_step_rad: float,
        required_start_sample_index: int = 0,
        required_end_sample_index: int | None = None,
    ):
        """Validate and preserve the selected taught samples and recorded timing."""
        self.verify_guard(home_report)
        if required_end_sample_index is None:
            required_end_sample_index = len(payload["samples"]) - 1
        if not (
            0 <= required_start_sample_index
            <= required_end_sample_index
            < len(payload["samples"])
        ):
            raise ValueError("selected marked sample range is invalid")
        first = [
            float(value)
            for value in payload["samples"][required_start_sample_index]["positions_rad"]
        ]
        current_error = max(
            abs(self.all_positions[name] - target)
            for name, target in zip(ARM_NAMES, first[:7])
        )
        if current_error > 0.06:
            raise RuntimeError(
                f"current arm state differs from recording start: {current_error:.6f} rad"
            )
        selected_samples = payload["samples"][
            required_start_sample_index : required_end_sample_index + 1
        ]
        audit = audit_recorded_path(
            selected_samples,
            maximum_gap_s=maximum_sample_gap_s,
            maximum_step_rad=maximum_sample_step_rad,
        )
        effective_velocities = tuple(
            value * playback_rate for value in audit.maximum_velocity_rad_s
        )
        for joint_index, (velocity, limit) in enumerate(zip(
            effective_velocities, COLLECTION_REPLAY_VELOCITY_LIMITS_RAD_S
        )):
            if velocity > limit:
                raise RuntimeError(
                    "recorded-timing replay exceeds velocity guard: "
                    f"joint=J{joint_index + 1} "
                    f"measured={velocity:.6f} "
                    f"limit={limit:.6f} rad/s; "
                    "reduce --playback-rate"
                )
        checked = 0
        for state, elapsed in interpolated_arm_states(selected_samples):
            reason = RecordedMotionPlayer.stop_reason(self)
            if reason:
                raise RuntimeError(reason)
            self.validate_state(state, f"sample_path_time={elapsed:.6f}s")
            checked += 1
        return audit, checked

    def execute_dense_arm_segment(
        self, payload: dict, start_sample: int, end_sample: int, time_scale: float,
        segment_index: int,
    ) -> None:
        expected = [float(value) for value in payload["samples"][start_sample]["positions_rad"][:7]]
        current = [self.all_positions[name] for name in ARM_NAMES]
        changed = max(abs(a - b) for a, b in zip(current, expected))
        if changed > 0.06:
            raise RuntimeError(
                f"MARKED_SEGMENT_{segment_index}_START_CHANGED measured={changed:.6f} limit=0.060000"
            )
        if start_sample == end_sample:
            return
        trajectory = build_dense_arm_trajectory(
            payload["samples"], start_sample, end_sample, time_scale=time_scale
        )
        arm_client = ActionClient(
            self, FollowJointTrajectory,
            "/right_joint_trajectory_controller/follow_joint_trajectory",
        )
        if not arm_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("right arm FollowJointTrajectory action unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        future = arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            raise RuntimeError(f"MARKED_SEGMENT_{segment_index}_GOAL_REJECTED")
        self.active_arm_goal = handle
        result = handle.get_result_async()
        segment_duration = duration_seconds(trajectory.points[-1].time_from_start)
        deadline = time.monotonic() + segment_duration + 5.0
        cancel_requested = False
        try:
            while not result.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                reason = self.stop_reason()
                if reason:
                    cancel_requested = True
                    self.cancel_goal(handle, "RIGHT_ARM_GOAL")
                    raise RuntimeError(
                        f"MARKED_SEGMENT_{segment_index}_{reason}"
                    )
            if not result.done():
                cancel_requested = True
                self.cancel_goal(handle, "RIGHT_ARM_GOAL")
                raise RuntimeError(f"MARKED_SEGMENT_{segment_index}_TIMEOUT")
            if result.result().status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"MARKED_SEGMENT_{segment_index}_EXECUTION_FAILED")
        except BaseException:
            if not cancel_requested and not result.done():
                self.cancel_goal(handle, "RIGHT_ARM_GOAL")
            raise
        finally:
            self.active_arm_goal = None
        target = [float(value) for value in payload["samples"][end_sample]["positions_rad"][:7]]
        self.wait_for_arm_target(
            target, tolerance_rad=self.keyframe_settle_tolerance_rad
        )

    def execute_marked(
        self, payload: dict, ranges: list[dict], time_scale: float,
        home_report: dict,
    ) -> None:
        self.verify_controller_active("right_joint_trajectory_controller", 10.0)
        self.verify_controller_active("right_gripper_controller", 10.0)
        self.publish_reference = True
        try:
            for segment_index, segment in enumerate(ranges):
                self.wait_for_full_joint_state()
                self.verify_guard(home_report)
                start_sample = int(segment["start_sample_index"])
                end_sample = int(segment["end_sample_index"])
                self.execute_dense_arm_segment(
                    payload, start_sample, end_sample, time_scale, segment_index
                )
                for marker in segment["markers"]:
                    marker_type = str(marker["type"])
                    if marker_type == "gripper_open":
                        self.command_gripper("open")
                    elif marker_type == "gripper_close":
                        self.command_gripper("close")
                    self.get_logger().warn(
                        "MARKED_EVENT_EXECUTED marker_index=%d type=%s sample_index=%d"
                        % (marker["marker_index"], marker_type, marker["sample_index"])
                    )
        finally:
            self.publish_reference = False

    def ensure_collection_fixed_home(self, home_report: dict) -> None:
        """Complete a collection replay at fixed HOME before reporting success."""
        self.wait_for_full_joint_state()
        target = list(Q_LIFTED_HOME_RIGHT)
        actual = [self.all_positions[name] for name in ARM_NAMES]
        maximum_error = max(abs(a - b) for a, b in zip(actual, target))
        if maximum_error <= 0.03:
            self.get_logger().warn(
                f"COLLECTION_FIXED_HOME_ACCEPTED max_error_rad={maximum_error:.6f} correction=false"
            )
            return
        self.verify_guard(home_report)
        plan_started = time.monotonic()
        trajectory, audit = self.plan_segment(actual, target, 0.10, 0.10)
        self.get_logger().warn(
            "COLLECTION_FIXED_HOME_CORRECTION_PLANNED planning_wall_s=%.3f duration_s=%.3f points=%d safety=PASS"
            % (time.monotonic() - plan_started, audit.duration_s, audit.point_count)
        )
        self.verify_controller_active("right_joint_trajectory_controller", 10.0)
        self.publish_reference = True
        try:
            result = self.execute_same_trajectory(
                trajectory, audit.duration_s, timeout_margin_s=30.0
            )
        finally:
            self.publish_reference = False
        if (
            result["action_status"] != GoalStatus.STATUS_SUCCEEDED
            or result["moveit_error_code"] != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError("fixed HOME correction execution failed")
        self.wait_for_arm_target(target, timeout_s=3.0, tolerance_rad=0.03)
        actual = [self.all_positions[name] for name in ARM_NAMES]
        maximum_error = max(abs(a - b) for a, b in zip(actual, target))
        self.get_logger().warn(
            f"COLLECTION_FIXED_HOME_ACCEPTED max_error_rad={maximum_error:.6f} correction=true"
        )

    def align_gripper_start(self, target: float) -> None:
        error = abs(self.all_positions[GRIPPER_NAME] - target)
        if error <= 0.02:
            self.gripper_reference = target
            return
        event = "open" if abs(target - GRIPPER_OPEN_RAD) < abs(target - GRIPPER_CLOSE_RAD) else "close"
        self.command_gripper(event)

    def execute_continuous(self, payload: dict, time_scale: float) -> None:
        arm, gripper = build_replay_trajectories(payload, time_scale=time_scale)
        if not self.execute_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("MoveIt /execute_trajectory action unavailable")
        # Preserve the commissioned legacy controller path.
        arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_joint_trajectory_controller/follow_joint_trajectory",
        )
        if not arm_client.wait_for_server(timeout_sec=2.0) or not self.gripper_action.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("replay FollowJointTrajectory action unavailable")
        arm_goal = FollowJointTrajectory.Goal(); arm_goal.trajectory = arm
        gripper_goal = FollowJointTrajectory.Goal(); gripper_goal.trajectory = gripper
        self.publish_reference = True
        try:
            arm_future = arm_client.send_goal_async(arm_goal)
            gripper_future = self.gripper_action.send_goal_async(gripper_goal)
            rclpy.spin_until_future_complete(self, arm_future, timeout_sec=3.0)
            rclpy.spin_until_future_complete(self, gripper_future, timeout_sec=3.0)
            arm_handle = arm_future.result() if arm_future.done() else None
            gripper_handle = gripper_future.result() if gripper_future.done() else None
            if arm_handle is None or not arm_handle.accepted:
                raise RuntimeError("right arm replay goal rejected")
            if gripper_handle is None or not gripper_handle.accepted:
                self.cancel_goal(arm_handle, "RIGHT_ARM_GOAL")
                raise RuntimeError("right gripper replay goal rejected")
            self.active_arm_goal = arm_handle
            self.active_gripper_goal = gripper_handle
            arm_result = arm_handle.get_result_async()
            gripper_result = gripper_handle.get_result_async()
            deadline = time.monotonic() + time_scale * float(payload["duration_s"]) + 5.0
            while (not arm_result.done() or not gripper_result.done()) and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                reason = self.stop_reason()
                if reason:
                    self.cancel_goal(arm_handle, "RIGHT_ARM_GOAL")
                    self.cancel_goal(gripper_handle, "RIGHT_GRIPPER_GOAL")
                    raise RuntimeError(f"recorded replay stopped: {reason}")
            if not arm_result.done() or not gripper_result.done():
                self.cancel_goal(arm_handle, "RIGHT_ARM_GOAL")
                self.cancel_goal(gripper_handle, "RIGHT_GRIPPER_GOAL")
                raise RuntimeError("recorded replay timed out")
            if not arm_result.done() or arm_result.result().status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError("right arm replay execution failed")
            if not gripper_result.done() or gripper_result.result().status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError("right gripper replay execution failed")
        finally:
            self.active_arm_goal = None
            self.active_gripper_goal = None
            self.publish_reference = False

    def fail_active_recorder(self) -> None:
        if not self.recorder_failure.wait_for_service(timeout_sec=0.25):
            return
        future = self.recorder_failure.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument(
        "--home-report",
        type=Path,
        default=Path("~/.ros/openarm_kinesthetic/home_initialization.execution.json").expanduser(),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--return-home", action="store_true")
    parser.add_argument("--allow-real-execution", action="store_true")
    parser.add_argument("--velocity-scaling", type=float, default=0.20)
    parser.add_argument("--acceleration-scaling", type=float, default=0.20)
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help=(
            "continuous_marked recorded-time rate in (0, 1]; 1.0 reproduces "
            "the taught timing and 0.5 plays it twice as slowly"
        ),
    )
    parser.add_argument("--start-waypoint", type=int, default=0)
    parser.add_argument("--stop-after-waypoint", type=int)
    parser.add_argument("--start-marker", type=int)
    parser.add_argument("--stop-after-marker", type=int)
    parser.add_argument("--maximum-sample-gap-s", type=float, default=0.50)
    parser.add_argument("--maximum-sample-step-rad", type=float, default=0.20)
    parser.add_argument(
        "--keyframe-settle-tolerance-rad",
        type=float,
        default=0.03,
        help=(
            "arm settling tolerance at marked/keyframe targets; "
            "fixed HOME acceptance remains 0.03 rad"
        ),
    )
    process_argv = __import__("sys").argv if argv is None else argv
    args = parser.parse_args(remove_ros_args(args=process_argv)[1:])
    if args.return_home == (args.trajectory is not None):
        parser.error("provide exactly one of --trajectory or --return-home")
    if not args.plan_only and not args.allow_real_execution:
        parser.error("execution requires --allow-real-execution; use --plan-only otherwise")
    validate_real_execution_scaling(args.velocity_scaling, args.acceleration_scaling)
    if not math.isfinite(args.playback_rate) or not 0.0 < args.playback_rate <= 1.0:
        parser.error("--playback-rate must be finite and in (0, 1]")
    if args.maximum_sample_gap_s <= 0.0 or args.maximum_sample_step_rad <= 0.0:
        parser.error("sample gap and step limits must be positive")
    if (
        not math.isfinite(args.keyframe_settle_tolerance_rad)
        or not 0.0 < args.keyframe_settle_tolerance_rad <= 0.06
    ):
        parser.error("--keyframe-settle-tolerance-rad must be in (0, 0.06]")
    return args


def main() -> int:
    args = parse_arguments()
    home_report = verify_home_provenance(args.home_report)
    payload = None if args.return_home else load_recorded_motion(args.trajectory)
    rclpy.init()
    node = RecordedMotionPlayer()
    node.keyframe_settle_tolerance_rad = args.keyframe_settle_tolerance_rad
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        if node.operator_interrupt_requested:
            raise KeyboardInterrupt
        node.request_operator_interrupt()

    signal.signal(signal.SIGINT, request_stop)
    try:
        node.wait_for_full_joint_state()
        if args.return_home:
            node.verify_guard(home_report)
            start = [node.all_positions[name] for name in ARM_NAMES]
            plan_started = time.monotonic()
            trajectory, audit = node.plan_segment(
                start,
                list(Q_LIFTED_HOME_RIGHT),
                args.velocity_scaling,
                args.acceleration_scaling,
            )
            node.get_logger().warn(
                "RETURN_HOME_PLANNED planning_wall_s=%.3f duration_s=%.3f points=%d safety=PASS"
                % (time.monotonic() - plan_started, audit.duration_s, audit.point_count)
            )
            if not args.plan_only:
                node.verify_controller_active("right_joint_trajectory_controller", 10.0)
                execution_started = time.monotonic()
                node.publish_reference = True
                try:
                    result = node.execute_same_trajectory(
                        trajectory, audit.duration_s, timeout_margin_s=30.0
                    )
                finally:
                    node.publish_reference = False
                if (
                    result["action_status"] != GoalStatus.STATUS_SUCCEEDED
                    or result["moveit_error_code"] != MoveItErrorCodes.SUCCESS
                ):
                    raise RuntimeError("runtime return HOME execution failed")
                node.wait_for_arm_target(
                    list(Q_LIFTED_HOME_RIGHT), timeout_s=3.0, tolerance_rad=0.03
                )
                node.get_logger().warn(
                    "RUNTIME_RETURN_HOME_COMPLETE execution_wall_s=%.3f"
                    % (time.monotonic() - execution_started)
                )
            return 0

        first = [float(value) for value in payload["samples"][0]["positions_rad"]]
        start_home_error = max(abs(a - b) for a, b in zip(first[:7], Q_LIFTED_HOME_RIGHT))
        if start_home_error > 0.06:
            raise RuntimeError(
                f"recorded motion does not start at lifted HOME: {start_home_error:.6f}"
            )
        if is_keyframe_motion(payload):
            planned = node.plan_keyframes(
                payload,
                home_report,
                args.velocity_scaling,
                args.acceleration_scaling,
            )
            total = sum(audit.duration_s for _, audit in planned)
            node.get_logger().warn(
                "KEYFRAME_PLAN_VALIDATED segments=%d duration_s=%.3f safety=PASS plan_only=%s"
                % (len(planned), total, args.plan_only)
            )
            if not args.plan_only:
                node.execute_keyframes(
                    payload,
                    planned,
                    args.start_waypoint,
                    args.stop_after_waypoint,
                    home_report,
                )
                node.get_logger().warn("KEYFRAME_MOTION_REPLAY_COMPLETE")
            return 0

        if is_marked_motion(payload):
            ranges = marker_segment_ranges(
                payload, args.start_marker, args.stop_after_marker
            )
            node.replay_provenance = replay_provenance(
                args.trajectory,
                payload,
                args.playback_rate,
                ranges,
                args.keyframe_settle_tolerance_rad,
            )
            start_sample = int(ranges[0]["start_sample_index"])
            end_sample = int(ranges[-1]["end_sample_index"])
            audit, checked = node.validate_marked_path(
                payload,
                home_report,
                args.playback_rate,
                args.maximum_sample_gap_s,
                args.maximum_sample_step_rad,
                start_sample,
                end_sample,
            )
            velocity_summary = ",".join(
                f"J{index + 1}={value:.6f}"
                for index, value in enumerate(audit.maximum_velocity_rad_s)
            )
            acceleration_summary = ",".join(
                f"J{index + 1}={value:.6f}"
                for index, value in enumerate(audit.maximum_acceleration_rad_s2)
            )
            selected_marker_count = sum(len(item["markers"]) for item in ranges)
            replay_duration = audit.duration_s / args.playback_rate
            node.get_logger().warn(
                "MARKED_TRAJECTORY_VALIDATED selected_samples=%d total_samples=%d "
                "selected_markers=%d total_markers=%d segments=%d "
                "recorded_duration_s=%.3f replay_duration_s=%.3f "
                "playback_rate=%.3f timing=%s safety=PASS dense_path=true "
                "checked_states=%d max_recorded_velocity_rad_s=[%s] "
                "raw_acceleration_diagnostic_rad_s2=[%s]"
                % (
                    end_sample - start_sample + 1,
                    len(payload["samples"]),
                    selected_marker_count,
                    len(payload["markers"]),
                    len(ranges),
                    audit.duration_s,
                    replay_duration,
                    args.playback_rate,
                    "processed" if payload.get("processing") else "recorded",
                    checked,
                    velocity_summary, acceleration_summary,
                )
            )
            if not args.plan_only:
                node.publish_replay_status("REPLAY_STARTED")
                node.execute_marked(
                    payload, ranges, 1.0 / args.playback_rate, home_report
                )
                if node.replay_provenance["full_replay"]:
                    try:
                        validate_collection_trajectory(payload)
                    except ValueError:
                        pass
                    else:
                        node.ensure_collection_fixed_home(home_report)
                node.publish_replay_status("REPLAY_COMPLETE")
                node.get_logger().warn("MARKED_TRAJECTORY_REPLAY_COMPLETE")
            return 0

        if args.plan_only:
            # Legacy v1 continuous validation now follows its dense path too,
            # but preserves the commissioned legacy execution behavior.
            current_arm_error = max(
                abs(node.all_positions[name] - target)
                for name, target in zip(ARM_NAMES, first[:7])
            )
            if current_arm_error > 0.06:
                raise RuntimeError(
                    f"current arm state differs from recording start: {current_arm_error:.6f}"
                )
            checked = 0
            for state, elapsed in interpolated_arm_states(payload["samples"]):
                node.validate_state(state, f"t={elapsed:.3f}s")
                checked += 1
            time_scale = playback_time_scale(payload["samples"])
            node.get_logger().warn(
                "REPLAY_VALIDATED states=%d recorded_duration_s=%.3f time_scale=%.3f safety=PASS plan_only=True"
                % (checked, payload["duration_s"], time_scale)
            )
            return 0
        current_arm_error = max(
            abs(node.all_positions[name] - target)
            for name, target in zip(ARM_NAMES, first[:7])
        )
        if current_arm_error > 0.06:
            raise RuntimeError(
                f"current arm state differs from recording start: {current_arm_error:.6f}"
            )
        checked = 0
        for state, elapsed in interpolated_arm_states(payload["samples"]):
            node.validate_state(state, f"t={elapsed:.3f}s")
            checked += 1
        time_scale = playback_time_scale(payload["samples"])
        replay_duration = time_scale * float(payload["duration_s"])
        node.get_logger().warn(
            "REPLAY_VALIDATED states=%d recorded_duration_s=%.3f time_scale=%.3f replay_duration_s=%.3f"
            % (checked, payload["duration_s"], time_scale, replay_duration)
        )
        node.align_gripper_start(first[7])
        node.execute_continuous(payload, time_scale)
        node.get_logger().warn("RECORDED_MOTION_REPLAY_COMPLETE")
        return 0
    except KeyboardInterrupt:
        canceled = node.cancel_active_goals()
        node.fail_active_recorder()
        node.get_logger().error(
            "RECORDED_MOTION_REPLAY_INTERRUPTED: action_cancel_confirmed=%s"
            % canceled
        )
        return 130
    except Exception as error:
        node.cancel_active_goals()
        node.fail_active_recorder()
        node.get_logger().error(f"RECORDED_MOTION_REPLAY_FAILED: {error}")
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
