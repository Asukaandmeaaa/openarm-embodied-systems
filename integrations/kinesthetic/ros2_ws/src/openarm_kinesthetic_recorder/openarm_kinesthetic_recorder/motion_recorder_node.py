"""Record measured right J1-J8 motion between two presses of gamepad X."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool
from openarm_kinesthetic.recording import allocate_trial_id

from .home_measurements import Q_LIFTED_HOME_RIGHT
from .recorded_motion import (
    GRIPPER_TARGETS,
    JOINT_NAMES,
    KIND,
    MARKED_KIND,
    MARKED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    validate_collection_trajectory,
    validate_recorded_motion,
)


class MotionRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("right_motion_recorder")
        self.declare_parameter(
            "output_directory", "~/openarm_kinesthetic_data/collections/trajectories"
        )
        self.declare_parameter(
            "toggle_topic", "/openarm_kinesthetic/motion_record_toggle"
        )
        self.declare_parameter("minimum_sample_period_s", 0.02)
        self.declare_parameter("joint_state_watchdog_s", 0.50)
        self.declare_parameter("home_start_tolerance_rad", 0.06)
        self.declare_parameter("recording_mode", "continuous_marked")
        self.declare_parameter(
            "keyframe_event_topic", "/openarm_kinesthetic/keyframe_event"
        )
        self.declare_parameter(
            "motion_marker_event_topic", "/openarm_kinesthetic/motion_marker_event"
        )
        self.declare_parameter("marker_binding_tolerance_s", 0.25)
        self.declare_parameter("maximum_sample_gap_s", 0.50)
        self.declare_parameter("auto_return_home_on_second_x", False)
        self.declare_parameter("auto_home_execute_timeout_s", 180.0)
        self.declare_parameter("auto_home_velocity_scaling", 0.10)
        self.declare_parameter("auto_home_acceleration_scaling", 0.10)
        self.output_directory = Path(
            str(self.get_parameter("output_directory").value)
        ).expanduser()
        self.failed_output_directory = self.output_directory.parent / "trajectories_failed"
        self.minimum_sample_period = float(
            self.get_parameter("minimum_sample_period_s").value
        )
        self.joint_watchdog = float(self.get_parameter("joint_state_watchdog_s").value)
        self.home_tolerance = float(
            self.get_parameter("home_start_tolerance_rad").value
        )
        self.recording_mode = str(self.get_parameter("recording_mode").value)
        if self.recording_mode not in ("continuous", "keyframe", "continuous_marked"):
            raise ValueError("recording_mode must be continuous, keyframe, or continuous_marked")
        self.marker_binding_tolerance = float(
            self.get_parameter("marker_binding_tolerance_s").value
        )
        self.maximum_sample_gap = float(
            self.get_parameter("maximum_sample_gap_s").value
        )
        self.auto_return_home_on_second_x = bool(
            self.get_parameter("auto_return_home_on_second_x").value
        )
        self.auto_home_execute_timeout = float(
            self.get_parameter("auto_home_execute_timeout_s").value
        )
        self.auto_home_velocity_scaling = float(
            self.get_parameter("auto_home_velocity_scaling").value
        )
        self.auto_home_acceleration_scaling = float(
            self.get_parameter("auto_home_acceleration_scaling").value
        )
        if (
            self.minimum_sample_period <= 0.0
            or self.marker_binding_tolerance <= 0.0
            or self.maximum_sample_gap <= 0.0
            or not math.isfinite(self.auto_home_execute_timeout)
            or self.auto_home_execute_timeout <= 0.0
            or not 0.0 < self.auto_home_velocity_scaling <= 0.8
            or not 0.0 < self.auto_home_acceleration_scaling <= 0.8
        ):
            raise ValueError("sample period, gap, and marker binding limits must be positive")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # /joint_states is usually much faster than the recorded 50 Hz stream.
        # Keep enough DDS history through MoveIt process startup and controller
        # handoff; the default sensor-data depth of five can drop 0.3 s of data
        # while this single-threaded executor is briefly busy.
        joint_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, joint_qos)
        self.create_subscription(
            String,
            str(self.get_parameter("toggle_topic").value),
            self.on_toggle,
            qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("motion_marker_event_topic").value),
            self.on_motion_marker_event,
            qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("keyframe_event_topic").value),
            self.on_keyframe_event,
            qos,
        )
        self.status_pub = self.create_publisher(
            String, "/openarm_kinesthetic/motion_record_status", qos
        )
        self.handoff_client = self.create_client(
            SetBool, "/right_kinesthetic_bridge/automatic_home_handoff"
        )
        self.auto_home_stage = "IDLE"
        self.auto_home_future = None
        self.auto_home_process: subprocess.Popen | None = None
        self.auto_home_deadline = 0.0
        self.auto_home_settle_count = 0
        self.auto_home_last_stamp: float | None = None
        self.last_saved_path = ""
        self.last_failed_path = ""
        self.invalid_marker_reason: str | None = None
        self.current_trial_id: str | None = None
        self.create_timer(0.1, self.service_auto_home)
        self.latest_positions: list[float] | None = None
        self.latest_received_at: float | None = None
        self.latest_joint_timestamp_s: float | None = None
        self.recording = False
        self.started_monotonic = 0.0
        self.last_sample_at = 0.0
        self.samples: list[dict] = []
        self.markers: list[dict] = []
        self.first_source_timestamp_s: float | None = None
        self.last_accepted_source_timestamp_s: float | None = None
        self.quality: dict[str, int | float] = {}
        self.last_toggle_sequence_by_session: dict[str, int] = {}
        self.last_keyframe_sequence_by_session: dict[str, int] = {}
        self.get_logger().info(
            "motion recorder ready: X starts/stops and saves J1-J8 "
            f"mode={self.recording_mode}"
        )

    def on_joint_state(self, message: JointState) -> None:
        index = {name: offset for offset, name in enumerate(message.name)}
        if any(name not in index for name in JOINT_NAMES):
            if self.recording and self.recording_mode == "continuous_marked":
                self.quality["rejected_incomplete_count"] += 1
            return
        if any(index[name] >= len(message.position) for name in JOINT_NAMES):
            if self.recording and self.recording_mode == "continuous_marked":
                self.quality["rejected_incomplete_count"] += 1
            return
        positions = [float(message.position[index[name]]) for name in JOINT_NAMES]
        if not all(math.isfinite(value) for value in positions):
            if self.recording and self.recording_mode == "continuous_marked":
                self.quality["rejected_nonfinite_count"] += 1
            return
        now = time.monotonic()
        source_stamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        if self.recording_mode == "continuous_marked" and (
            not math.isfinite(source_stamp) or source_stamp <= 0.0
        ):
            if self.recording:
                self.quality["rejected_invalid_timestamp_count"] += 1
            return
        self.latest_positions = positions
        self.latest_received_at = now
        self.latest_joint_timestamp_s = source_stamp if source_stamp > 0.0 else time.time()
        if (
            self.recording
            and self.recording_mode in ("continuous", "continuous_marked")
        ):
            if self.recording_mode == "continuous_marked":
                assert self.first_source_timestamp_s is not None
                assert self.last_accepted_source_timestamp_s is not None
                if source_stamp <= self.last_accepted_source_timestamp_s:
                    self.quality["timestamp_regression_count"] += 1
                    return
                if source_stamp - self.last_accepted_source_timestamp_s < self.minimum_sample_period:
                    return
                self._append_marked_sample(positions, source_stamp)
            elif now - self.last_sample_at >= self.minimum_sample_period:
                self.samples.append(
                    {"time_from_start_s": now - self.started_monotonic, "positions_rad": list(positions)}
                )
                self.last_sample_at = now

    def _append_marked_sample(self, positions: list[float], source_stamp: float) -> None:
        assert self.first_source_timestamp_s is not None
        elapsed = source_stamp - self.first_source_timestamp_s
        if self.samples:
            gap = elapsed - float(self.samples[-1]["time_from_start_s"])
            if gap > float(self.quality["maximum_sample_gap_s"]):
                self.quality["maximum_sample_gap_s"] = gap
                self.quality["maximum_sample_gap_sample_index"] = len(self.samples)
                self.quality["maximum_sample_gap_start_s"] = float(
                    self.samples[-1]["time_from_start_s"]
                )
                self.quality["maximum_sample_gap_stage"] = getattr(
                    self, "auto_home_stage", "IDLE"
                )
            self.quality["estimated_dropped_sample_count"] += max(
                0, int(gap / self.minimum_sample_period + 1.0e-9) - 1
            )
        self.samples.append(
            {
                "sample_index": len(self.samples),
                "time_from_start_s": elapsed,
                "source_timestamp_s": source_stamp,
                "positions_rad": list(positions),
            }
        )
        self.last_accepted_source_timestamp_s = source_stamp
        self.last_sample_at = time.monotonic()
        self.quality["accepted_sample_count"] = len(self.samples)

    def on_motion_marker_event(self, message: String) -> None:
        if (
            self.recording_mode != "continuous_marked"
            or not self.recording
            or getattr(self, "auto_home_stage", "IDLE") != "IDLE"
        ):
            return
        try:
            event = json.loads(message.data)
            if event.get("schema_version") != 1:
                raise ValueError("invalid marker-event schema")
            session = str(event["session_id"])
            sequence = int(event["sequence"])
            marker_type = str(event["type"])
            source_button = str(event["source_button"])
            event_stamp = float(event["source_timestamp_s"])
            if marker_type not in ("keypoint", "gripper_open", "gripper_close"):
                raise ValueError("unsupported marker type")
            if sequence < 0 or not source_button or not math.isfinite(event_stamp) or event_stamp <= 0.0:
                raise ValueError("invalid marker event fields")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().error(f"MOTION_MARKER_REJECTED: {error}")
            return
        if sequence <= self.last_keyframe_sequence_by_session.get(session, -1):
            return
        self.last_keyframe_sequence_by_session[session] = sequence
        if not self.samples or self.first_source_timestamp_s is None:
            self.get_logger().error("MOTION_MARKER_REJECTED: NO_ACCEPTED_SAMPLE")
            return
        sample = self.samples[-1]
        marker_time = max(0.0, event_stamp - self.first_source_timestamp_s)
        if self.markers:
            marker_time = max(marker_time, float(self.markers[-1]["time_from_start_s"]))
        binding_error = abs(marker_time - float(sample["time_from_start_s"]))
        if binding_error > self.marker_binding_tolerance:
            self.get_logger().error(
                "MOTION_MARKER_REJECTED: STALE_SAMPLE binding_error_s=%.6f limit_s=%.6f"
                % (binding_error, self.marker_binding_tolerance)
            )
            return
        marker = {
            "marker_index": len(self.markers),
            "type": marker_type,
            "sample_index": int(sample["sample_index"]),
            "time_from_start_s": marker_time,
            "source_timestamp_s": event_stamp,
            "source_button": source_button,
            "binding_error_s": binding_error,
        }
        if marker_type.startswith("gripper_"):
            marker["target_position_rad"] = float(event.get("target_position_rad", GRIPPER_TARGETS[marker_type]))
            marker["result"] = str(event.get("result", "requested"))
        self.markers.append(marker)
        self.get_logger().warn(
            "MOTION_MARKER_APPENDED marker_index=%d sample_index=%d time=%.6f type=%s"
            % (marker["marker_index"], marker["sample_index"], marker_time, marker_type)
        )

    def on_keyframe_event(self, message: String) -> None:
        if self.recording_mode != "keyframe" or not self.recording:
            return
        try:
            payload = json.loads(message.data)
            session = str(payload["session_id"])
            sequence = int(payload["sequence"])
            event = str(payload["event"])
            if payload.get("schema_version") != 1 or sequence < 0:
                raise ValueError("invalid keyframe-event schema")
            if event not in ("capture", "close", "open"):
                raise ValueError(f"unsupported keyframe event: {event}")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().error(f"keyframe event rejected: {error}")
            return
        if sequence <= self.last_keyframe_sequence_by_session.get(session, -1):
            return
        self.last_keyframe_sequence_by_session[session] = sequence
        if event == "capture":
            self.capture_keyframe()
            return
        if not self.samples:
            self.get_logger().error("KEYFRAME_EVENT_REJECTED: NO_WAYPOINT")
            return
        self.samples[-1]["gripper_event"] = event
        self.get_logger().warn(
            f"KEYFRAME_GRIPPER_EVENT waypoint={len(self.samples) - 1} event={event}"
        )

    def capture_keyframe(self) -> None:
        now = time.monotonic()
        if (
            self.latest_positions is None
            or self.latest_received_at is None
            or now - self.latest_received_at > self.joint_watchdog
        ):
            self.get_logger().error("KEYFRAME_CAPTURE_REJECTED: JOINT_STATE_STALE")
            return
        self.samples.append(
            {
                "time_from_start_s": now - self.started_monotonic,
                "positions_rad": list(self.latest_positions),
                "gripper_event": "none",
                "waypoint_index": len(self.samples),
                "timestamp_s": self.latest_joint_timestamp_s,
                "captured_wall_time_s": time.time(),
            }
        )
        self.get_logger().warn(
            f"KEYFRAME_CAPTURED waypoint={len(self.samples) - 1}"
        )

    def on_toggle(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            key = (str(payload["session_id"]), int(payload["sequence"]))
            if payload.get("schema_version") != 1 or key[1] < 1:
                raise ValueError("invalid toggle schema")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().error(f"motion record toggle rejected: {error}")
            return
        if key[1] <= self.last_toggle_sequence_by_session.get(key[0], 0):
            return
        self.last_toggle_sequence_by_session[key[0]] = key[1]
        if self.recording:
            if self.recording_mode == "continuous_marked" and self.auto_return_home_on_second_x:
                self.request_auto_home_finish()
            else:
                self.finish_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.auto_home_stage != "IDLE":
            self.get_logger().error("RECORD_REJECTED: automatic HOME handoff is unresolved")
            return
        now = time.monotonic()
        if (
            self.latest_positions is None
            or self.latest_received_at is None
            or now - self.latest_received_at > self.joint_watchdog
        ):
            self.get_logger().error("RECORD_REJECTED: fresh right J1-J8 unavailable")
            return
        home_errors = [
            abs(actual - target)
            for actual, target in zip(self.latest_positions[:7], Q_LIFTED_HOME_RIGHT)
        ]
        home_error = max(home_errors)
        if home_error > self.home_tolerance:
            joint_index = home_errors.index(home_error)
            self.get_logger().error(
                "RECORD_REJECTED: start is not lifted HOME "
                f"(J{joint_index + 1} error={home_error:.6f} rad, "
                f"actual={self.latest_positions[joint_index]:.6f}, "
                f"required={Q_LIFTED_HOME_RIGHT[joint_index]:.6f})"
            )
            return
        self.recording = True
        try:
            self.current_trial_id = allocate_trial_id(self.output_directory.parent)
        except (OSError, ValueError) as error:
            self.recording = False
            self.get_logger().error(f"RECORD_REJECTED: trial ID allocation failed: {error}")
            return
        self.last_saved_path = ""
        self.last_failed_path = ""
        self.invalid_marker_reason = None
        self.started_monotonic = now
        self.last_sample_at = now
        first = {"time_from_start_s": 0.0, "positions_rad": list(self.latest_positions)}
        self.markers = []
        self.quality = {
            "accepted_sample_count": 0,
            "rejected_incomplete_count": 0,
            "rejected_nonfinite_count": 0,
            "rejected_invalid_timestamp_count": 0,
            "timestamp_regression_count": 0,
            "estimated_dropped_sample_count": 0,
            "maximum_sample_gap_s": 0.0,
            "maximum_sample_gap_sample_index": None,
            "maximum_sample_gap_start_s": None,
            "maximum_sample_gap_stage": None,
        }
        if self.recording_mode == "continuous_marked":
            if self.latest_joint_timestamp_s is None or self.latest_joint_timestamp_s <= 0.0:
                self.recording = False
                self.get_logger().error("RECORD_REJECTED: valid joint-state timestamp unavailable")
                return
            self.first_source_timestamp_s = self.latest_joint_timestamp_s
            self.last_accepted_source_timestamp_s = self.latest_joint_timestamp_s
            first.update(
                {
                    "sample_index": 0,
                    "source_timestamp_s": self.latest_joint_timestamp_s,
                }
            )
            self.quality["accepted_sample_count"] = 1
        if self.recording_mode == "keyframe":
            first.update(
                {
                    "gripper_event": "none",
                    "waypoint_index": 0,
                    "timestamp_s": self.latest_joint_timestamp_s,
                    "captured_wall_time_s": time.time(),
                }
            )
        self.samples = [first]
        self.publish_status("RECORDING", "")
        self.get_logger().warn(
            "KEYFRAME_RECORDING_STARTED HOME=waypoint_0"
            if self.recording_mode == "keyframe"
            else "MARKED_TRAJECTORY_RECORDING_STARTED"
            if self.recording_mode == "continuous_marked"
            else "MOTION_RECORDING_STARTED"
        )

    def finish_recording(self) -> None:
        now = time.monotonic()
        if self.recording_mode == "continuous" and self.latest_positions is not None and (
            not self.samples or now - self.last_sample_at > 1e-6
        ):
            self.samples.append(
                {
                    "time_from_start_s": now - self.started_monotonic,
                    "positions_rad": list(self.latest_positions),
                }
            )
        elif self.recording_mode == "continuous_marked":
            if (
                self.latest_positions is None
                or self.latest_received_at is None
                or now - self.latest_received_at > self.joint_watchdog
                or self.latest_joint_timestamp_s is None
            ):
                self.recording = False
                self.samples = []
                self.markers = []
                self.get_logger().error(
                    "RECORD_REJECTED: fresh terminal right J1-J8 unavailable"
                )
                self.publish_status("REJECTED", "")
                return
            if self.latest_joint_timestamp_s > float(
                self.samples[-1]["source_timestamp_s"]
            ):
                # Keep the actual stop pose even when it lands inside the normal
                # down-sampling interval. The schema only requires monotonic time.
                self._append_marked_sample(
                    self.latest_positions, self.latest_joint_timestamp_s
                )
        self.recording = False
        if len(self.samples) < 2 or (
            self.recording_mode in ("continuous", "continuous_marked")
            and self.samples[-1]["time_from_start_s"] < 0.10
        ):
            self.samples = []
            self.get_logger().error("RECORD_REJECTED: motion is shorter than 0.10 s")
            self.publish_status("REJECTED", "")
            return
        if (
            self.recording_mode == "continuous_marked"
            and float(self.quality["maximum_sample_gap_s"]) > self.maximum_sample_gap
        ):
            maximum_gap = float(self.quality["maximum_sample_gap_s"])
            self.samples = []
            self.markers = []
            self.get_logger().error(
                "RECORD_REJECTED: sample gap measured=%.6f limit=%.6f "
                "sample_index=%s start_s=%s stage=%s"
                % (
                    maximum_gap, self.maximum_sample_gap,
                    self.quality.get("maximum_sample_gap_sample_index"),
                    self.quality.get("maximum_sample_gap_start_s"),
                    self.quality.get("maximum_sample_gap_stage"),
                )
            )
            self.publish_status("REJECTED", "")
            return
        self.output_directory.mkdir(parents=True, exist_ok=True)
        wall_time_ns = time.time_ns()
        path = self.output_directory / f"{self.current_trial_id}.json"
        if path.exists():
            self.get_logger().error(f"RECORD_REJECTED: trajectory path already exists: {path}")
            self.publish_status("REJECTED", "")
            return
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "created_wall_time_s": wall_time_ns / 1e9,
            "trial_id": self.current_trial_id,
            "joint_names": JOINT_NAMES,
            "required_start_home_positions_rad": list(Q_LIFTED_HOME_RIGHT),
            "duration_s": float(self.samples[-1]["time_from_start_s"]),
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        if self.recording_mode == "keyframe":
            payload["mode"] = "keyframe"
            payload["waypoint_count"] = len(self.samples)
        elif self.recording_mode == "continuous_marked":
            payload = {
                "schema_version": MARKED_SCHEMA_VERSION,
                "kind": MARKED_KIND,
                "mode": "continuous_marked",
                "arm": "right",
                "created_wall_time_s": wall_time_ns / 1e9,
                "trial_id": self.current_trial_id,
                "joint_names": JOINT_NAMES,
                "required_start_home_positions_rad": list(Q_LIFTED_HOME_RIGHT),
                "sample_period_target_s": self.minimum_sample_period,
                "marker_binding_tolerance_s": self.marker_binding_tolerance,
                "maximum_allowed_sample_gap_s": self.maximum_sample_gap,
                "duration_s": float(self.samples[-1]["time_from_start_s"]),
                "samples": self.samples,
                "markers": self.markers,
                "quality": self.quality,
        }
        try:
            if self.recording_mode == "continuous_marked":
                collection = validate_collection_trajectory(payload)
            else:
                validate_recorded_motion(payload)
        except ValueError as error:
            self.samples = []
            self.markers = []
            self.get_logger().error(f"RECORD_REJECTED: {error}")
            self.publish_status("REJECTED", "")
            return
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(self.output_directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.samples = []
        self.markers = []
        self.last_saved_path = str(path)
        self.publish_status("SAVED", str(path))
        if self.recording_mode == "continuous_marked":
            self.get_logger().warn(
                "MOTION_RECORDING_SAVED path=%s markers=%d "
                "end_home_max_error_rad=%.6f"
                % (
                    path,
                    collection["marker_count"],
                    collection["end_home_max_error_rad"],
                )
            )
        else:
            self.get_logger().warn(f"MOTION_RECORDING_SAVED path={path}")

    def publish_status(self, state: str, path: str) -> None:
        message = String()
        message.data = json.dumps(
            {"state": state, "path": path, "trial_id": self.current_trial_id},
            separators=(",", ":"), sort_keys=True
        )
        self.status_pub.publish(message)

    def finish_failed_marker_recording(self) -> None:
        """Persist a marker-invalid trial after the robot has returned HOME."""
        if self.recording_mode != "continuous_marked" or not self.samples:
            raise ValueError("no continuous marked recording to preserve")
        if (
            self.latest_positions is None
            or self.latest_joint_timestamp_s is None
            or self.latest_received_at is None
        ):
            raise ValueError("fresh terminal joint state unavailable")
        if self.latest_joint_timestamp_s > float(self.samples[-1]["source_timestamp_s"]):
            self._append_marked_sample(self.latest_positions, self.latest_joint_timestamp_s)
        payload = {
            "schema_version": MARKED_SCHEMA_VERSION,
            "kind": MARKED_KIND,
            "mode": "continuous_marked",
            "arm": "right",
            "created_wall_time_s": time.time(),
            "trial_id": self.current_trial_id,
            "joint_names": JOINT_NAMES,
            "required_start_home_positions_rad": list(Q_LIFTED_HOME_RIGHT),
            "sample_period_target_s": self.minimum_sample_period,
            "marker_binding_tolerance_s": self.marker_binding_tolerance,
            "maximum_allowed_sample_gap_s": self.maximum_sample_gap,
            "duration_s": float(self.samples[-1]["time_from_start_s"]),
            "samples": self.samples,
            "markers": self.markers,
            "quality": self.quality,
            "failure_reason": self.invalid_marker_reason,
            "failed_at_fixed_home": True,
        }
        validate_recorded_motion(payload)
        self.failed_output_directory.mkdir(parents=True, exist_ok=True)
        path = self.failed_output_directory / f"{self.current_trial_id}.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.last_failed_path = str(path)
        self.recording = False
        self.samples = []
        self.markers = []
        self.publish_status("FAILED_SAVED", str(path))
        self.get_logger().warn(
            f"MOTION_RECORDING_FAILED_SAVED path={path} reason={self.invalid_marker_reason}"
        )

    def request_auto_home_finish(self) -> None:
        """Keep sampling while the bridge yields and a fresh HOME path executes."""
        if self.auto_home_stage != "IDLE":
            self.get_logger().warn("automatic HOME already in progress; X ignored")
            return
        markers = [
            (item.get("type"), item.get("source_button"), item.get("result"))
            for item in self.markers
        ]
        if markers != [
            ("gripper_close", "Y", "semantic_only"),
            ("gripper_open", "A", "semantic_only"),
        ]:
            self.invalid_marker_reason = (
                f"expected Y=CLOSE,A=OPEN; got {markers}"
            )
            self.get_logger().error(
                f"automatic HOME marker check failed; HOME recovery will continue: {markers}"
            )
        if not self.handoff_client.service_is_ready():
            self.get_logger().error("automatic HOME rejected: bridge handoff service unavailable")
            return
        self.auto_home_stage = "HANDOFF"
        self.auto_home_deadline = time.monotonic() + 5.0
        self.auto_home_future = self.handoff_client.call_async(SetBool.Request(data=True))
        self.get_logger().warn("SECOND_X_ACCEPTED: recording continues; requesting rigid bridge handoff")

    def start_auto_home_player(self) -> None:
        # The player plans, validates, then executes the very same HOME plan.
        # A separate --plan-only subprocess planned a second path and added
        # several seconds of stationary samples to the recorded trajectory.
        command = [
            "ros2", "run", "openarm_kinesthetic_recorder", "recorded_motion_player",
            "--return-home", "--allow-real-execution",
            "--velocity-scaling", str(self.auto_home_velocity_scaling),
            "--acceleration-scaling", str(self.auto_home_acceleration_scaling),
        ]
        self.auto_home_process = subprocess.Popen(command, start_new_session=True)
        self.auto_home_stage = "EXECUTE"
        self.auto_home_deadline = time.monotonic() + self.auto_home_execute_timeout
        self.get_logger().warn("AUTO_HOME_%s_STARTED: %s" % (
            self.auto_home_stage, " ".join(command)
        ))

    def reject_auto_home(self, reason: str) -> None:
        process = self.auto_home_process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            self.auto_home_stage = "ABORTING"
            self.auto_home_deadline = time.monotonic() + 5.0
        else:
            self.auto_home_process = None
            self.auto_home_stage = "FAILED"
        self.recording = False
        self.samples = []
        self.markers = []
        self.get_logger().error(
            f"AUTO_HOME_FAILED: {reason}; recording discarded; bridge handoff remains locked"
        )
        self.publish_status("REJECTED", "")

    def release_rejected_recording_at_home(self) -> None:
        """Release handoff after a failed save only when HOME was measured."""
        self.get_logger().error(
            "AUTO_HOME_FAILED: trajectory validation rejected after measured HOME; "
            "recording discarded"
        )
        if not self.handoff_client.service_is_ready():
            self.auto_home_stage = "FAILED"
            self.get_logger().error("AUTO_HOME_RELEASE_FAILED: bridge service unavailable")
            return
        self.auto_home_future = self.handoff_client.call_async(
            SetBool.Request(data=False)
        )
        self.auto_home_stage = "RELEASE_REJECTED"
        self.auto_home_deadline = time.monotonic() + 5.0

    def service_auto_home(self) -> None:
        stage = self.auto_home_stage
        if stage == "IDLE" or stage == "FAILED":
            return
        now = time.monotonic()
        if stage == "ABORTING":
            process = self.auto_home_process
            if process is None or process.poll() is not None:
                self.auto_home_process = None
                self.auto_home_stage = "FAILED"
                self.get_logger().error("AUTO_HOME_PLAYER_EXITED_AFTER_CANCEL")
            elif now > self.auto_home_deadline:
                self.get_logger().error(
                    "AUTO_HOME_PLAYER_STILL_RUNNING: keep bridge locked and use the physical emergency stop if motion persists"
                )
                self.auto_home_deadline = now + 5.0
            return
        if stage == "HANDOFF":
            if self.auto_home_future is not None and self.auto_home_future.done():
                try:
                    result = self.auto_home_future.result()
                    if result is None or not result.success:
                        if result is not None and result.message.startswith(
                            "release LT/LB"
                        ):
                            self.auto_home_stage = "IDLE"
                            self.get_logger().warn(
                                "SECOND_X_WAITING: %s; then press X again" % result.message
                            )
                            return
                        raise RuntimeError("bridge rejected handoff: " + str(result.message if result else "no response"))
                    self.start_auto_home_player()
                except Exception as error:
                    self.reject_auto_home(str(error))
            elif now > self.auto_home_deadline:
                self.reject_auto_home("bridge handoff timed out")
            return
        if stage == "EXECUTE":
            process = self.auto_home_process
            if process is None:
                self.reject_auto_home("player process missing")
                return
            exit_code = process.poll()
            if exit_code is None:
                if now > self.auto_home_deadline:
                    self.reject_auto_home(f"{stage.lower()} timed out")
                return
            self.auto_home_process = None
            if exit_code != 0:
                self.reject_auto_home(f"{stage.lower()} failed with exit code {exit_code}")
                return
            self.auto_home_stage = "SETTLE"
            self.auto_home_deadline = now + 5.0
            self.auto_home_settle_count = 0
            self.auto_home_last_stamp = None
            self.get_logger().warn("AUTO_HOME_EXECUTE_COMPLETE: waiting for measured J1-J7 HOME")
            return
        if stage == "SETTLE":
            if (
                self.latest_positions is not None
                and self.latest_received_at is not None
                and now - self.latest_received_at <= self.joint_watchdog
                and self.latest_joint_timestamp_s is not None
                and (
                    self.auto_home_last_stamp is None
                    or self.latest_joint_timestamp_s > self.auto_home_last_stamp
                )
            ):
                self.auto_home_last_stamp = self.latest_joint_timestamp_s
                error = max(
                    abs(actual - target)
                    for actual, target in zip(self.latest_positions[:7], Q_LIFTED_HOME_RIGHT)
                )
                self.auto_home_settle_count = (
                    self.auto_home_settle_count + 1 if error <= 0.03 else 0
                )
                if self.auto_home_settle_count >= 3:
                    try:
                        if getattr(self, "invalid_marker_reason", None):
                            self.finish_failed_marker_recording()
                        else:
                            self.finish_recording()
                    except Exception as error:
                        self.reject_auto_home(f"trajectory save failed: {error}")
                        return
                    if getattr(self, "invalid_marker_reason", None) and not self.last_failed_path:
                        self.reject_auto_home("failed marker trajectory was not saved")
                        return
                    if not getattr(self, "invalid_marker_reason", None) and not self.last_saved_path:
                        self.release_rejected_recording_at_home()
                        return
                    self.auto_home_stage = (
                        "RELEASE_FAILED" if getattr(self, "invalid_marker_reason", None) else "RELEASE"
                    )
                    self.auto_home_deadline = now + 5.0
                    self.auto_home_future = self.handoff_client.call_async(
                        SetBool.Request(data=False)
                    )
                    return
            if now > self.auto_home_deadline:
                self.reject_auto_home("measured HOME did not settle within 0.03 rad")
            return
        if stage == "RELEASE_FAILED":
            if self.auto_home_future is not None and self.auto_home_future.done():
                try:
                    result = self.auto_home_future.result()
                except Exception as error:
                    result = None
                    self.get_logger().error(f"AUTO_HOME_FAILED_RELEASE_ERROR: {error}")
                if result is not None and result.success:
                    self.auto_home_stage = "IDLE"
                    self.get_logger().warn(
                        "AUTO_HOME_FAILED_RECORDING_COMPLETE: "
                        f"saved={self.last_failed_path}; bridge manual control restored"
                    )
                else:
                    self.auto_home_stage = "FAILED"
                    self.get_logger().error(
                        "AUTO_HOME_FAILED_RELEASE_FAILED: bridge handoff remains locked"
                    )
            elif now > self.auto_home_deadline:
                self.auto_home_stage = "FAILED"
                self.get_logger().error(
                    "AUTO_HOME_FAILED_RELEASE_TIMEOUT: bridge handoff remains locked"
                )
            return
        if stage == "RELEASE_REJECTED":
            if self.auto_home_future is not None and self.auto_home_future.done():
                try:
                    result = self.auto_home_future.result()
                except Exception as error:
                    result = None
                    self.get_logger().error(f"AUTO_HOME_RELEASE_FAILED: {error}")
                if result is not None and result.success:
                    self.auto_home_stage = "IDLE"
                    self.get_logger().warn(
                        "AUTO_HOME_REJECTED_RECORDING_RELEASED: rigid HOME verified; "
                        "bridge manual control restored"
                    )
                else:
                    self.auto_home_stage = "FAILED"
                    self.get_logger().error(
                        "AUTO_HOME_RELEASE_FAILED: bridge handoff remains locked"
                    )
            elif now > self.auto_home_deadline:
                self.auto_home_stage = "FAILED"
                self.get_logger().error(
                    "AUTO_HOME_RELEASE_TIMEOUT: bridge handoff remains locked"
                )
            return
        if stage == "RELEASE":
            if self.auto_home_future is not None and self.auto_home_future.done():
                try:
                    result = self.auto_home_future.result()
                except Exception as error:
                    self.auto_home_stage = "FAILED"
                    self.get_logger().error(
                        f"trajectory saved, but bridge release failed: {error}"
                    )
                    return
                if result is not None and result.success:
                    self.auto_home_stage = "IDLE"
                    self.get_logger().warn("AUTO_HOME_RECORDING_COMPLETE: bridge manual control restored")
                else:
                    self.auto_home_stage = "FAILED"
                    self.get_logger().error("trajectory saved, but bridge handoff remains locked")
            elif now > self.auto_home_deadline:
                self.auto_home_stage = "FAILED"
                self.get_logger().error("trajectory saved, but bridge release timed out")


def main() -> int:
    rclpy.init()
    node = MotionRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        process = node.auto_home_process
        if process is not None and process.poll() is None:
            try:
                if node.auto_home_stage != "ABORTING":
                    os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                node.get_logger().error("automatic HOME player did not confirm exit; keep bridge handoff locked")
        if node.recording:
            node.get_logger().warn("active motion recording cancelled during shutdown")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
