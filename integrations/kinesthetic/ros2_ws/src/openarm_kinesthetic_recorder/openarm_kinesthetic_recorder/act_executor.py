"""Guarded executor for ACT action chunks written by the Nano inference process.

The file is a transport boundary, not a trajectory player.  It is consumed only
when this node is explicitly enabled; merely leaving ``latest_action.json`` on
disk never commands the robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import ListControllers
import rclpy
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Trigger
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .recorded_motion import GRIPPER_NAME, JOINT_NAMES
from .recorded_motion_player import ARM_NAMES, RecordedMotionPlayer, _duration
from .act_trajectory import OfficialActShaper, max_abs_difference
from .act_contract import (
    ACTION_ORDER,
    ACTION_SCHEMA,
    EXPECTED_INFERENCE_API,
    validate_action_chunk,
)


def build_act_trajectories(right_actions: list[list[float]], start_delay_s: float) -> tuple[JointTrajectory, JointTrajectory]:
    arm = JointTrajectory()
    arm.joint_names = list(ARM_NAMES)
    gripper = JointTrajectory()
    gripper.joint_names = [GRIPPER_NAME]
    for index, values in enumerate(right_actions):
        point_time = start_delay_s + index / 30.0
        arm_point = JointTrajectoryPoint(positions=list(values[:7]), time_from_start=_duration(point_time))
        grip_point = JointTrajectoryPoint(positions=[values[7]], time_from_start=_duration(point_time))
        arm.points.append(arm_point)
        gripper.points.append(grip_point)
    return arm, gripper


def build_official_dense_trajectories(
    right_actions: list[list[float]],
    start_delay_s: float,
    shaper: OfficialActShaper,
    hold_s: float,
) -> tuple[JointTrajectory, JointTrajectory]:
    """Adapt the official 250 Hz position stream to the existing FJT actions."""
    arm, gripper, _ = _build_official_dense_trajectories_with_targets(
        right_actions, start_delay_s, shaper, hold_s
    )
    return arm, gripper


def _build_official_dense_trajectories_with_targets(
    right_actions: list[list[float]],
    start_delay_s: float,
    shaper: OfficialActShaper,
    hold_s: float,
) -> tuple[JointTrajectory, JointTrajectory, list[list[float]]]:
    """Build dense controller points and return the exact filtered target stream."""
    shaped = shaper.shape(right_actions)
    arm = JointTrajectory()
    arm.joint_names = list(ARM_NAMES)
    gripper = JointTrajectory()
    gripper.joint_names = [GRIPPER_NAME]
    control_dt = shaper.upsampler.control_dt
    for index, values in enumerate(shaped):
        point_time = start_delay_s + index * control_dt
        arm.points.append(JointTrajectoryPoint(
            positions=list(values[:7]), time_from_start=_duration(point_time)
        ))
        gripper.points.append(JointTrajectoryPoint(
            positions=[values[7]], time_from_start=_duration(point_time)
        ))
    last_time = start_delay_s + (len(shaped) - 1) * control_dt
    if hold_s < last_time:
        raise ValueError(
            f"trajectory_hold_s={hold_s:.6f} is shorter than the dense trajectory "
            f"horizon {last_time:.6f}"
        )
    if hold_s > last_time:
        final = shaped[-1]
        arm.points.append(JointTrajectoryPoint(
            positions=list(final[:7]), time_from_start=_duration(hold_s)
        ))
        gripper.points.append(JointTrajectoryPoint(
            positions=[final[7]], time_from_start=_duration(hold_s)
        ))
    return arm, gripper, shaped


class ActExecutor(RecordedMotionPlayer):
    def __init__(self) -> None:
        super().__init__("act_executor")
        self.declare_parameter("action_path", "/dev/shm/act_live/latest_action.json")
        self.declare_parameter("enabled", False)
        self.declare_parameter("allow_real_execution", False)
        self.declare_parameter("poll_period_s", 0.02)
        self.declare_parameter("action_start_delay_s", 0.02)
        self.declare_parameter("action_timeout_s", 0.75)
        self.declare_parameter("diagnostics_path", "/dev/shm/act_live/act_executor_diagnostics.jsonl")
        self.declare_parameter("controller_state_diagnostics_hz", 20.0)
        self.declare_parameter("joint_state_diagnostics_hz", 20.0)
        self.declare_parameter("trajectory_mode", "official_dense")
        self.declare_parameter("execution_interface", "trajectory_topic")
        self.declare_parameter("control_hz", 250.0)
        self.declare_parameter("filter_cutoff_hz", 15.0)
        self.declare_parameter("trajectory_hold_s", 0.50)
        self.action_path = Path(str(self.get_parameter("action_path").value)).expanduser()
        self.enabled = bool(self.get_parameter("enabled").value)
        self.allow_real_execution = bool(self.get_parameter("allow_real_execution").value)
        self.poll_period_s = float(self.get_parameter("poll_period_s").value)
        self.act_joint_freshness_s = min(0.20, float(self.get_parameter("joint_state_freshness_s").value))
        self.start_delay_s = float(self.get_parameter("action_start_delay_s").value)
        self.action_timeout_s = float(self.get_parameter("action_timeout_s").value)
        self.diagnostics_path = Path(str(self.get_parameter("diagnostics_path").value)).expanduser()
        self.controller_state_diagnostics_hz = float(
            self.get_parameter("controller_state_diagnostics_hz").value
        )
        self.joint_state_diagnostics_hz = float(
            self.get_parameter("joint_state_diagnostics_hz").value
        )
        self.trajectory_mode = str(self.get_parameter("trajectory_mode").value)
        self.execution_interface = str(self.get_parameter("execution_interface").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.filter_cutoff_hz = float(self.get_parameter("filter_cutoff_hz").value)
        self.trajectory_hold_s = float(self.get_parameter("trajectory_hold_s").value)
        if (
            self.poll_period_s <= 0
            or self.act_joint_freshness_s <= 0
            or self.start_delay_s < 0
            or self.action_timeout_s <= 0
            or self.controller_state_diagnostics_hz <= 0
            or self.joint_state_diagnostics_hz <= 0
            or self.trajectory_mode not in ("raw", "official_dense")
            or self.execution_interface not in ("trajectory_topic", "fjt_action")
            or self.control_hz <= 0
            or self.filter_cutoff_hz <= 0
            or self.trajectory_hold_s <= 0
        ):
            raise ValueError("ACT timing parameters must be positive")
        if self.enabled and not self.allow_real_execution:
            raise ValueError("enabled=true requires allow_real_execution=true")
        self.arm_action = ActionClient(self, FollowJointTrajectory,
                                       "/right_joint_trajectory_controller/follow_joint_trajectory")
        self.arm_trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/right_joint_trajectory_controller/joint_trajectory",
            10,
        )
        self.gripper_trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/right_gripper_controller/joint_trajectory",
            10,
        )
        self.last_file_sequence = -1
        self.last_observation_sequence = -1
        self.last_file_mtime_ns = None
        self.last_valid_action_at = None
        self.last_sent_actions: list[list[float]] | None = None
        self.last_sent_monotonic_s: float | None = None
        self.last_goal_accepted_monotonic_s: float | None = None
        self.last_goal_start_monotonic_s: float | None = None
        self.last_chunk_received_monotonic_s: float | None = None
        self.last_controller_diag_monotonic = {"arm": float("-inf"), "gripper": float("-inf")}
        self.last_joint_state_diag_monotonic = float("-inf")
        self.last_raw_right_actions: list[list[float]] | None = None
        self.arm_result_future = None
        self.gripper_result_future = None
        self.pending_phase = ""
        self.pending_payload = None
        self.pending_generation = 0
        self.pending_controller_future = None
        self.pending_cancel_futures = []
        self.pending_cancel_labels = {}
        self.pending_cancel_handles = {}
        self.pending_arm_send = None
        self.pending_gripper_send = None
        self.pending_arm_handle = None
        self.pending_gripper_handle = None
        self.fault = ""
        self.cleanup_pending = False
        self.cleanup_label = ""
        self.cleanup_failed = ""
        self.run_generation = 0
        self._result_logged = set()
        self._expected_canceled_results = set()
        self._retired_result_futures = []
        self.act_shaper = OfficialActShaper(
            control_hz=self.control_hz, cutoff_hz=self.filter_cutoff_hz
        )
        self.exclusive_status_seen = False
        self.gamepad_busy = False
        self.create_subscription(String, "/openarm_kinesthetic/gamepad_status", self._gamepad_status, 20)
        self.start_service = self.create_service(Trigger, "~/start", self._start)
        self.stop_service = self.create_service(Trigger, "~/stop", self._stop)
        self.timer = self.create_timer(self.poll_period_s, self._tick)
        self._diag("executor_started", enabled=self.enabled, action_path=str(self.action_path))
        if self.enabled:
            self.get_logger().warn("ACT executor enabled; waiting for a fresh validated action chunk")
        else:
            self.get_logger().warn("ACT executor armed but disabled; call ~/start or set enabled:=true with allow_real_execution:=true")

    def _gamepad_status(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.exclusive_status_seen = True
        self.gamepad_busy = bool(payload.get("active")) or payload.get("arm_motion_source") not in (None, "idle")
        if payload.get("stop_latched") is True:
            self.software_stop_latched = True

    def _start(self, request, response):
        if not self.allow_real_execution:
            response.success = False
            response.message = "allow_real_execution is false"
            return response
        if self.fault:
            response.success = False
            response.message = f"latched fault: {self.fault}"
            return response
        if self.cleanup_pending or self.pending_phase or self.active_arm_goal is not None or self.active_gripper_goal is not None:
            response.success = False
            response.message = "previous ACT goal cleanup is still pending or was not confirmed"
            return response
        self.enabled = True
        self.cleanup_pending = False
        self.cleanup_failed = ""
        self.act_shaper.reset()
        self.run_generation += 1
        self._diag("start_requested", generation=self.run_generation)
        response.success = True
        response.message = "ACT enabled; waiting for a fresh validated action chunk"
        return response

    def _stop(self, request, response):
        self.enabled = False
        self._diag("stop_requested", execution_interface=self.execution_interface)
        cancelled = self._cancel_all("ACT_STOP")
        response.success = cancelled
        if self.execution_interface == "trajectory_topic":
            response.message = (
                "ACT disabled and empty trajectory published; verify controller feedback"
                if cancelled else "ACT disabled, but trajectory stop publication did not complete"
            )
        else:
            response.message = (
                "ACT disabled and active goals cancelled"
                if cancelled else "ACT disabled, but one or more goal cancellations were not confirmed"
            )
        return response

    @staticmethod
    def _diag_value(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (list, tuple)):
            return [ActExecutor._diag_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): ActExecutor._diag_value(item) for key, item in value.items()}
        return value

    def _diag(self, event: str, **fields) -> None:
        """Append one machine-readable event using the same monotonic clock."""
        record = {
            "diagnostics_schema": "openarm.act_executor.v2",
            "event": event,
            "monotonic_s": time.monotonic(),
            **{key: self._diag_value(value) for key, value in fields.items()},
        }
        try:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as error:
            self.get_logger().warn(f"ACT_DIAGNOSTICS_WRITE_FAILED: {error}")

    def on_controller_state(self, message: JointTrajectoryControllerState) -> None:
        super().on_controller_state(message)
        self._record_controller_state("arm", message)

    def on_gripper_controller_state(self, message: JointTrajectoryControllerState) -> None:
        super().on_gripper_controller_state(message)
        self._record_controller_state("gripper", message)

    def on_all_joint_state(self, message) -> None:
        """Keep the complete joint feedback stream in the same JSONL clock domain."""
        super().on_all_joint_state(message)
        now = time.monotonic()
        minimum_period = 1.0 / self.joint_state_diagnostics_hz
        if now - self.last_joint_state_diag_monotonic < minimum_period:
            return
        indices = {str(name): index for index, name in enumerate(message.name)}
        if any(name not in indices or indices[name] >= len(message.position) for name in JOINT_NAMES):
            return
        positions = [float(message.position[indices[name]]) for name in JOINT_NAMES]
        if not all(math.isfinite(value) for value in positions):
            return
        self.last_joint_state_diag_monotonic = now
        self._diag(
            "joint_feedback",
            source_stamp_s=self._header_stamp_s(message),
            joint_names=list(JOINT_NAMES),
            feedback_positions=positions,
        )

    @staticmethod
    def _header_stamp_s(message) -> float | None:
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return None
        value = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1.0e-9
        return value if math.isfinite(value) and value > 0.0 else None

    def _record_controller_state(self, controller: str, message: JointTrajectoryControllerState) -> None:
        now = time.monotonic()
        minimum_period = 1.0 / self.controller_state_diagnostics_hz
        if now - self.last_controller_diag_monotonic.get(controller, float("-inf")) < minimum_period:
            return
        self.last_controller_diag_monotonic[controller] = now
        names = list(message.joint_names)
        index = {name: i for i, name in enumerate(names)}
        reference = list(message.reference.positions)
        feedback = list(message.feedback.positions)
        if len(reference) != len(names):
            reference = []
        if len(feedback) != len(names):
            feedback = []
        errors = []
        if reference and feedback:
            for name in names:
                i = index[name]
                if math.isfinite(float(reference[i])) and math.isfinite(float(feedback[i])):
                    errors.append(abs(float(reference[i]) - float(feedback[i])))
        self._diag(
            "controller_state",
            controller=controller,
            source_stamp_s=self._header_stamp_s(message),
            joint_names=names,
            reference_positions=reference,
            feedback_positions=feedback,
            max_reference_feedback_error=(max(errors) if errors else None),
        )

    def _cancel_all(self, label: str) -> bool:
        self.enabled = False
        self.run_generation += 1
        self.cleanup_pending = True
        self.cleanup_label = label
        self.cleanup_failed = ""
        self.pending_payload = None
        self._diag("cleanup_requested", label=label, generation=self.run_generation)
        if self.execution_interface == "trajectory_topic":
            self._publish_empty_trajectories(label)
            return self._progress_cleanup()
        self._start_goal_cancels(label)
        return self._progress_cleanup()

    def _publish_empty_trajectories(self, label: str) -> None:
        arm = JointTrajectory()
        arm.joint_names = list(ARM_NAMES)
        gripper = JointTrajectory()
        gripper.joint_names = [GRIPPER_NAME]
        self.arm_trajectory_pub.publish(arm)
        self.gripper_trajectory_pub.publish(gripper)
        self._diag(
            "trajectory_stop_published",
            label=label,
            execution_interface=self.execution_interface,
            target_time_from_start_s=[],
            interpolated_filtered_right_actions=[],
        )

    def _start_goal_cancels(self, label: str) -> None:
        """Submit at most one cancel request per live goal.

        The response is consumed from the timer callback.  This is deliberate:
        calling spin_until_future_complete from a timer or service callback
        deadlocks the single-threaded ROS executor.
        """
        for role, handle, result_future in (
            ("arm", self.active_arm_goal, self.arm_result_future),
            ("gripper", self.active_gripper_goal, self.gripper_result_future),
        ):
            if handle is None or (result_future is not None and result_future.done()):
                continue
            if any(item is handle for item in self.pending_cancel_handles.values()):
                continue
            try:
                if result_future is not None:
                    self._expected_canceled_results.add(result_future)
                future = handle.cancel_goal_async()
                self.pending_cancel_futures.append(future)
                self.pending_cancel_labels[id(future)] = role
                self.pending_cancel_handles[id(future)] = handle
                self._diag("goal_cancel_requested", role=role, label=label)
            except Exception as error:
                self.cleanup_failed = f"{role} cancel request failed: {error}"
                self._diag("goal_cancel_error", role=role, error=str(error))

    def _progress_cleanup(self) -> bool:
        """Drain controller query/send/cancel futures after stop or fault."""
        if self.pending_controller_future is not None:
            if not self.pending_controller_future.done():
                return False
            try:
                self.pending_controller_future.result()
            except Exception as error:
                self._diag("controller_query_error", error=str(error))
            self.pending_controller_future = None
            if self.pending_phase == "controllers":
                self.pending_phase = ""

        for attr, role in (("pending_arm_send", "arm"), ("pending_gripper_send", "gripper")):
            future = getattr(self, attr)
            if future is None:
                continue
            if not future.done():
                return False
            try:
                handle = future.result()
            except Exception as error:
                self._diag("goal_send_error", role=role, error=str(error))
                handle = None
            if handle is not None and getattr(handle, "accepted", False):
                self._diag("goal_accepted", role=role, cleanup=True)
                if role == "arm":
                    self.active_arm_goal = handle
                    self.arm_result_future = handle.get_result_async()
                else:
                    self.active_gripper_goal = handle
                    self.gripper_result_future = handle.get_result_async()
                self._start_goal_cancels(self.cleanup_label)
            else:
                self._diag("goal_rejected", role=role, cleanup=True)
            setattr(self, attr, None)

        remaining_cancel = []
        for future in self.pending_cancel_futures:
            if not future.done():
                remaining_cancel.append(future)
                continue
            role = self.pending_cancel_labels.pop(id(future), "unknown")
            self.pending_cancel_handles.pop(id(future), None)
            try:
                response = future.result()
                confirmed = bool(response is not None and response.goals_canceling)
            except Exception as error:
                confirmed = False
                self.cleanup_failed = f"{role} cancel failed: {error}"
                self._diag("goal_cancel_result", role=role, confirmed=False, error=str(error))
            else:
                self._diag("goal_cancel_result", role=role, confirmed=confirmed)
                if not confirmed:
                    self.cleanup_failed = f"{role} goal cancellation was not confirmed"
            if confirmed:
                # Keep the result future for a later terminal result record, but
                # no longer consider this goal a source that can be replaced.
                if role == "arm":
                    self.active_arm_goal = None
                elif role == "gripper":
                    self.active_gripper_goal = None
        self.pending_cancel_futures = remaining_cancel

        if self.pending_phase == "goals" and (
            self.pending_arm_send is not None or self.pending_gripper_send is not None
        ):
            return False
        if self.pending_cancel_futures:
            return False
        self.pending_phase = ""
        self.pending_payload = None
        self.last_sent_actions = None
        self.last_sent_monotonic_s = None
        self.last_goal_accepted_monotonic_s = None
        self.last_goal_start_monotonic_s = None
        self.last_valid_action_at = None
        self.last_raw_right_actions = None
        self.stationary_target_since = None
        self.stationary_target_warned = False
        self.act_shaper.reset()
        if self.arm_result_future is not None:
            self._retired_result_futures.append(("arm", self.arm_result_future))
        if self.gripper_result_future is not None:
            self._retired_result_futures.append(("gripper", self.gripper_result_future))
        self.arm_result_future = None
        self.gripper_result_future = None
        self.cleanup_pending = False
        self._diag(
            "cleanup_complete",
            label=self.cleanup_label,
            confirmed=not bool(self.cleanup_failed),
            error=self.cleanup_failed or None,
        )
        return not self.cleanup_failed

    def _fault(self, reason: str) -> None:
        if not self.fault:
            self.fault = reason
            self.enabled = False
            self._diag("fault", reason=reason)
            self._cancel_all("ACT_FAULT")
            self.get_logger().error(f"ACT_EXECUTOR_FAULT: {reason}")

    def _read_chunk(self, now: float):
        try:
            stat = self.action_path.stat()
            if self.last_file_mtime_ns == stat.st_mtime_ns:
                return None
            payload = json.loads(self.action_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            self.get_logger().warn(f"ACT_ACTION_FILE_RETRY: {error}")
            return None
        try:
            normalized = validate_action_chunk(
                payload, now, self.last_file_sequence, self.last_observation_sequence
            )
        except ValueError as error:
            # Remember this inode version so a torn/expired publication is not
            # parsed on every timer tick.  The inference process publishes a
            # new atomic file for the next usable chunk.
            self.last_file_mtime_ns = stat.st_mtime_ns
            sequence = payload.get("action_sequence") if isinstance(payload, dict) else None
            message = str(error)
            event = (
                "action_chunk_expired"
                if "clock window is invalid or expired" in message
                else "action_file_rejected"
            )
            self._diag(
                event,
                error=message,
                action_sequence=sequence,
                published_monotonic_s=(payload.get("published_monotonic_s")
                                       if isinstance(payload, dict) else None),
                valid_until_monotonic_s=(payload.get("valid_until_monotonic_s")
                                         if isinstance(payload, dict) else None),
                received_monotonic_s=now,
                safety_action="wait_for_fresh_chunk" if event == "action_chunk_expired" else "fault",
            )
            raise
        self.last_file_mtime_ns = stat.st_mtime_ns
        normalized["_executor_received_monotonic_s"] = now
        previous_raw_right_actions = self.last_raw_right_actions
        normalized["_previous_raw_right_actions"] = previous_raw_right_actions
        interval = (
            None if self.last_chunk_received_monotonic_s is None
            else now - self.last_chunk_received_monotonic_s
        )
        self.last_chunk_received_monotonic_s = now
        self._diag(
            "chunk_received",
            action_sequence=normalized["action_sequence"],
            observation_sequence=normalized["source_observation_sequence"],
            published_monotonic_s=normalized["published_monotonic_s"],
            valid_until_monotonic_s=normalized["valid_until_monotonic_s"],
            source_observation_monotonic_s=normalized.get("source_observation_monotonic_s"),
            source_observation_age_ms=normalized.get("source_observation_age_ms"),
            inference_api=normalized.get("inference_api"),
            chunk_interval_s=interval,
            raw_cross_chunk_first_point_jump_max_rad=(
                None
                if previous_raw_right_actions is None
                else max_abs_difference(
                    normalized["right_actions"][0], previous_raw_right_actions[-1]
                )
            ),
            raw_actions=normalized["actions"],
            raw_right_actions=normalized["right_actions"],
        )
        self.last_raw_right_actions = [list(values) for values in normalized["right_actions"]]
        return normalized

    def _prepare_chunk(self, payload: dict) -> dict:
        right_actions = payload["right_actions"]
        now = time.monotonic()
        # Match the official executor's cancellation/blending semantics.  The
        # remaining raw positions from the previous chunk fade into the new
        # prediction instead of restarting at new[0] and producing a command
        # discontinuity at every inference update.
        consumed = 0
        elapsed_since_goal_start = None
        estimated_hold = None
        original_right_actions = [list(values) for values in right_actions]
        if self.last_sent_actions is not None and self.last_goal_start_monotonic_s is not None:
            elapsed_since_goal_start = now - self.last_goal_start_monotonic_s
            estimated_hold = max(0.0, elapsed_since_goal_start - (len(self.last_sent_actions) - 1) / 30.0)
            consumed = min(10, max(0, int(elapsed_since_goal_start * 30.0)))
            remaining = self.last_sent_actions[consumed:]
            blend_count = min(len(remaining), len(right_actions))
            if blend_count:
                blended = []
                for index in range(blend_count):
                    old_weight = 1.0 if blend_count == 1 else 1.0 - index / (blend_count - 1)
                    blended.append([
                        old_weight * float(old) + (1.0 - old_weight) * float(new)
                        for old, new in zip(remaining[index], right_actions[index])
                    ])
                right_actions = blended + right_actions[blend_count:]
        prepared = {
            **payload,
            "right_actions": right_actions,
            "_blend_consumed_raw_steps": consumed,
            "_blend_original_right_actions": original_right_actions,
        }
        self._diag(
            "chunk_prepared",
            action_sequence=payload["action_sequence"],
            consumed_raw_steps=consumed,
            elapsed_since_previous_goal_start_s=elapsed_since_goal_start,
            estimated_previous_goal_hold_s=estimated_hold,
            raw_actions=payload.get("actions"),
            raw_right_actions=original_right_actions,
            raw_first_point_jump_max_rad=(
                None
                if payload.get("_previous_raw_right_actions") is None
                else max_abs_difference(
                    original_right_actions[0], payload["_previous_raw_right_actions"][-1]
                )
            ),
            blended_right_actions=right_actions,
            blended_first_point_jump_max_rad=(
                None
                if self.last_sent_actions is None
                else max_abs_difference(right_actions[0], self.last_sent_actions[-1])
            ),
        )
        return prepared

    def _begin_send(self, payload: dict) -> None:
        if self.cleanup_pending or not self.enabled:
            return
        if self.execution_interface == "fjt_action":
            if not self.arm_action.server_is_ready() or not self.gripper_action.server_is_ready():
                raise RuntimeError("right FollowJointTrajectory action server unavailable")
        if not self.controller_client.service_is_ready():
            raise RuntimeError("/controller_manager/list_controllers unavailable")
        self.pending_payload = self._prepare_chunk(payload)
        self.pending_generation = self.run_generation
        self.pending_controller_future = self.controller_client.call_async(ListControllers.Request())
        self.pending_phase = "controllers"
        self._diag("controller_query_requested", action_sequence=payload["action_sequence"])

    def _send_goals(self) -> None:
        if self.cleanup_pending or not self.enabled or self.pending_generation != self.run_generation:
            return
        payload = self.pending_payload
        if self.trajectory_mode == "official_dense":
            arm_trajectory, grip_trajectory, filtered_targets = _build_official_dense_trajectories_with_targets(
                payload["right_actions"],
                self.start_delay_s,
                self.act_shaper,
                self.trajectory_hold_s,
            )
        else:
            arm_trajectory, grip_trajectory = build_act_trajectories(
                payload["right_actions"], self.start_delay_s
            )
            filtered_targets = [list(values) for values in payload["right_actions"]]
        dense_target_times_s = [
            float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1.0e-9
            for point in arm_trajectory.points
        ]
        if self.execution_interface == "trajectory_topic":
            self.arm_trajectory_pub.publish(arm_trajectory)
            self.gripper_trajectory_pub.publish(grip_trajectory)
            self._diag(
                "trajectory_published",
                action_sequence=payload["action_sequence"],
                execution_interface=self.execution_interface,
                trajectory_point_count=len(arm_trajectory.points),
                target_time_from_start_s=dense_target_times_s,
                interpolated_filtered_right_actions=filtered_targets,
                raw_right_actions=payload.get("_blend_original_right_actions"),
                blended_right_actions=payload["right_actions"],
            )
            self._mark_chunk_published(payload)
            return
        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory = arm_trajectory
        grip_goal = FollowJointTrajectory.Goal()
        grip_goal.trajectory = grip_trajectory
        self.pending_arm_send = self.arm_action.send_goal_async(arm_goal)
        self.pending_gripper_send = self.gripper_action.send_goal_async(grip_goal)
        self.pending_phase = "goals"
        self._diag(
            "goal_send_requested",
            action_sequence=payload["action_sequence"],
            received_monotonic_s=payload.get("_executor_received_monotonic_s"),
            trajectory_start_delay_s=self.start_delay_s,
            trajectory_raw_span_s=(len(payload["right_actions"]) - 1) / 30.0,
            trajectory_duration_s=(
                self.trajectory_hold_s
                if self.trajectory_mode == "official_dense"
                else self.start_delay_s + (len(payload["right_actions"]) - 1) / 30.0
            ),
            trajectory_mode=self.trajectory_mode,
            control_hz=self.control_hz,
            filter_cutoff_hz=self.filter_cutoff_hz,
            trajectory_point_count=len(arm_trajectory.points),
            target_time_from_start_s=dense_target_times_s,
            interpolated_filtered_right_actions=filtered_targets,
            raw_right_actions=payload.get("_blend_original_right_actions"),
            blended_right_actions=payload["right_actions"],
            arm_points=[list(point.positions) for point in arm_trajectory.points],
            gripper_points=[list(point.positions) for point in grip_trajectory.points],
        )

    def _mark_chunk_published(self, payload: dict) -> None:
        accepted_at = time.monotonic()
        self.last_valid_action_at = accepted_at
        self.last_sent_actions = [list(values) for values in payload["right_actions"]]
        self.last_goal_accepted_monotonic_s = accepted_at
        self.last_goal_start_monotonic_s = accepted_at + self.start_delay_s
        self.last_sent_monotonic_s = self.last_goal_start_monotonic_s
        self.last_file_sequence = payload["action_sequence"]
        self.last_observation_sequence = payload["source_observation_sequence"]
        self.get_logger().info(
            f"ACT_CHUNK_PUBLISHED action_sequence={self.last_file_sequence} "
            f"observation_sequence={self.last_observation_sequence} "
            f"interface={self.execution_interface}"
        )
        self._diag(
            "chunk_published",
            action_sequence=self.last_file_sequence,
            observation_sequence=self.last_observation_sequence,
            published_monotonic_s=accepted_at,
            execution_interface=self.execution_interface,
            trajectory_horizon_s=(
                self.trajectory_hold_s
                if self.trajectory_mode == "official_dense"
                else self.start_delay_s + (len(payload["right_actions"]) - 1) / 30.0
            ),
        )
        self.pending_phase = ""
        self.pending_payload = None
        self.pending_controller_future = None

    def _progress_pending(self) -> bool:
        if not self.pending_phase:
            return False
        if self.cleanup_pending:
            self._progress_cleanup()
            return True
        if self.pending_phase == "controllers":
            if not self.pending_controller_future.done():
                return True
            response = self.pending_controller_future.result()
            self._diag("controller_query_result", controllers=[
                {"name": item.name, "state": item.state} for item in response.controller
            ])
            controllers = {item.name: item for item in response.controller}
            for name in ("right_joint_trajectory_controller", "right_gripper_controller"):
                if name not in controllers or controllers[name].state != "active":
                    raise RuntimeError(f"required controller {name!r} is not active")
            if self.execution_interface == "trajectory_topic":
                self._send_goals()
                return True
            self.pending_cancel_futures = []
            self.pending_cancel_labels = {}
            self.pending_cancel_handles = {}
            for role, handle, result_future in (
                ("arm", self.active_arm_goal, self.arm_result_future),
                ("gripper", self.active_gripper_goal, self.gripper_result_future),
            ):
                if handle is not None and (result_future is None or not result_future.done()):
                    future = handle.cancel_goal_async()
                    if result_future is not None:
                        self._expected_canceled_results.add(result_future)
                    self.pending_cancel_futures.append(future)
                    self.pending_cancel_labels[id(future)] = role
                    self.pending_cancel_handles[id(future)] = handle
                    self._diag("goal_cancel_requested", role=role, label="chunk_replace")
            if self.pending_cancel_futures:
                self.pending_phase = "cancel"
            else:
                self._send_goals()
            return True
        if self.pending_phase == "cancel":
            if not all(future.done() for future in self.pending_cancel_futures):
                return True
            confirmations = []
            for future in self.pending_cancel_futures:
                role = self.pending_cancel_labels.pop(id(future), "unknown")
                self.pending_cancel_handles.pop(id(future), None)
                try:
                    result = future.result()
                    confirmed = bool(result is not None and result.goals_canceling)
                except Exception as error:
                    confirmed = False
                    self._diag("goal_cancel_result", role=role, confirmed=False, error=str(error))
                else:
                    self._diag("goal_cancel_result", role=role, confirmed=confirmed)
                confirmations.append(confirmed)
            if not all(confirmations):
                raise RuntimeError("ACT previous goal cancellation was not confirmed")
            if self.arm_result_future is not None:
                self._retired_result_futures.append(("arm", self.arm_result_future))
            if self.gripper_result_future is not None:
                self._retired_result_futures.append(("gripper", self.gripper_result_future))
            self.active_arm_goal = None
            self.active_gripper_goal = None
            self.arm_result_future = None
            self.gripper_result_future = None
            self._send_goals()
            return True
        if self.pending_phase == "goals":
            if not self.pending_arm_send.done() or not self.pending_gripper_send.done():
                return True
            arm_handle = self.pending_arm_send.result()
            grip_handle = self.pending_gripper_send.result()
            arm_accepted = arm_handle is not None and arm_handle.accepted
            grip_accepted = grip_handle is not None and grip_handle.accepted
            if arm_accepted:
                self.active_arm_goal = arm_handle
                self.arm_result_future = arm_handle.get_result_async()
            if grip_accepted:
                self.active_gripper_goal = grip_handle
                self.gripper_result_future = grip_handle.get_result_async()
            if not arm_accepted or not grip_accepted:
                if arm_accepted or grip_accepted:
                    self._start_goal_cancels("partial_goal_acceptance")
                raise RuntimeError(
                    "right arm ACT goal rejected" if not arm_accepted
                    else "right gripper ACT goal rejected"
                )
            self._diag("goal_accepted", role="arm", action_sequence=self.pending_payload["action_sequence"])
            self._diag("goal_accepted", role="gripper", action_sequence=self.pending_payload["action_sequence"])
            payload = self.pending_payload
            accepted_at = time.monotonic()
            self.last_valid_action_at = accepted_at
            self.last_sent_actions = [list(values) for values in payload["right_actions"]]
            self.last_goal_accepted_monotonic_s = accepted_at
            self.last_goal_start_monotonic_s = accepted_at + self.start_delay_s
            self.last_sent_monotonic_s = self.last_goal_start_monotonic_s
            self.last_file_sequence = payload["action_sequence"]
            self.last_observation_sequence = payload["source_observation_sequence"]
            self.get_logger().info(
                f"ACT_CHUNK_SENT action_sequence={self.last_file_sequence} "
                f"observation_sequence={self.last_observation_sequence}"
            )
            self._diag(
                "chunk_goals_accepted",
                action_sequence=self.last_file_sequence,
                observation_sequence=self.last_observation_sequence,
                accepted_monotonic_s=accepted_at,
                goal_start_monotonic_s=self.last_goal_start_monotonic_s,
                trajectory_horizon_s=(len(payload["right_actions"]) - 1) / 30.0 + self.start_delay_s,
            )
            self.pending_phase = ""
            self.pending_payload = None
            self.pending_controller_future = None
            self.pending_cancel_futures = []
            self.pending_cancel_labels = {}
            self.pending_cancel_handles = {}
            self.pending_arm_send = None
            self.pending_gripper_send = None
            return True
        raise RuntimeError(f"unknown ACT pending phase {self.pending_phase!r}")

    def _progress_results(self) -> None:
        for role, handle_attr, future_attr in (
            ("arm", "active_arm_goal", "arm_result_future"),
            ("gripper", "active_gripper_goal", "gripper_result_future"),
        ):
            future = getattr(self, future_attr)
            if future is None or not future.done() or future in self._result_logged:
                continue
            self._result_logged.add(future)
            expected_cancel = future in self._expected_canceled_results
            self._expected_canceled_results.discard(future)
            try:
                wrapped = future.result()
                status = None if wrapped is None else int(wrapped.status)
                self._diag("goal_result", role=role, status=status)
            except Exception as error:
                self._diag("goal_result", role=role, status=None, error=str(error))
                if not self.cleanup_pending and not self.fault:
                    raise RuntimeError(f"right {role} ACT result unavailable: {error}") from error
                continue
            if wrapped is None or (
                wrapped.status != GoalStatus.STATUS_SUCCEEDED
                and not (expected_cancel and wrapped.status == GoalStatus.STATUS_CANCELED)
            ):
                if not self.cleanup_pending and not self.fault:
                    raise RuntimeError(f"right {role} ACT goal failed (status={status})")
            # A terminal result cannot be cancelled or replaced, so release the
            # handle.  The result has already been retained in diagnostics.
            setattr(self, handle_attr, None)
            setattr(self, future_attr, None)

        remaining_retired = []
        for role, future in self._retired_result_futures:
            if future is None or future in self._result_logged:
                continue
            if not future.done():
                remaining_retired.append((role, future))
                continue
            self._result_logged.add(future)
            expected_cancel = future in self._expected_canceled_results
            self._expected_canceled_results.discard(future)
            try:
                wrapped = future.result()
                status = None if wrapped is None else int(wrapped.status)
                self._diag("goal_result", role=role, status=status, retired=True)
            except Exception as error:
                self._diag("goal_result", role=role, status=None, retired=True, error=str(error))
                continue
            if wrapped is None or (
                wrapped.status != GoalStatus.STATUS_SUCCEEDED
                and not (expected_cancel and wrapped.status == GoalStatus.STATUS_CANCELED)
            ):
                self._diag(
                    "retired_goal_unexpected_result",
                    role=role,
                    status=status,
                    expected_cancel=expected_cancel,
                )
        self._retired_result_futures = remaining_retired

    def _tick(self) -> None:
        try:
            self._progress_results()
            if self.cleanup_pending:
                self._progress_cleanup()
                return
        except Exception as error:
            if not self.fault:
                self._fault(str(error))
            return
        if not self.enabled or self.fault:
            return
        if self.software_stop_latched or self.operator_interrupt_requested:
            self._fault(self.stop_reason() or "software stop")
            return
        if self.gamepad_busy:
            self._fault("gamepad is an active competing motion source")
            return
        try:
            if self._progress_pending():
                return
        except Exception as error:
            self._fault(str(error))
            return
        if self.all_received_at is None or time.monotonic() - self.all_received_at > self.act_joint_freshness_s:
            if self.last_valid_action_at is not None:
                self._fault("/joint_states stale during ACT execution")
            return
        now = time.monotonic()
        try:
            payload = self._read_chunk(now)
            if payload is not None:
                self._begin_send(payload)
            elif self.last_valid_action_at is not None and now - self.last_valid_action_at > self.action_timeout_s:
                self._fault("ACT action chunk timeout")
        except RuntimeError as error:
            self._fault(str(error))
        except ValueError as error:
            # An expired file is rejected, then the next atomic publication is
            # consumed.  At startup this is normal if the executor was enabled
            # between two inference chunks; active execution is still guarded
            # by action_timeout_s below.
            if "monotonic clock window is invalid or expired" in str(error):
                self.get_logger().warn("ACT_ACTION_WAITING_FOR_FRESH_CHUNK")
                return
            self._fault(str(error))
        except (KeyError, TypeError) as error:
            self._fault(str(error))

    def request_operator_interrupt(self) -> None:
        super().request_operator_interrupt()
        if self.enabled or self.active_arm_goal is not None or self.active_gripper_goal is not None:
            self._fault("operator interrupt requested")
        else:
            self.enabled = False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(remove_ros_args(args=__import__("sys").argv if argv is None else argv)[1:])
    rclpy.init()
    node = ActExecutor()
    def request_stop(_signum, _frame):
        node.request_operator_interrupt()
        # The executor owns the foreground spin loop.  Shutting down the
        # context here lets Ctrl-C return to the shell after goals are
        # cancelled instead of leaving a disabled node stuck in spin().
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, request_stop)
    try:
        rclpy.spin(node)
    finally:
        node._cancel_all("ACT_SHUTDOWN")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
