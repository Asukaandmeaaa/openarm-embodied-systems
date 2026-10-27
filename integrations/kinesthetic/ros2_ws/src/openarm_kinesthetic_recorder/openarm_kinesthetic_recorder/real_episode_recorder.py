"""Crash-safe attended real episode capture after fixed HOME initialization."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from openarm_kinesthetic.recording import (
    EpisodeRecorder,
    RecorderStateError,
    audit_real_episode_directory,
    valid_trial_id,
)

from .recorded_motion import COLLECTION_END_HOME_TOLERANCE_RAD
from .capture_preflight import capture_preflight


JOINT_NAMES = [
    *[f"openarm_right_joint{i}" for i in range(1, 8)],
    "openarm_right_finger_joint1",
    *[f"openarm_left_joint{i}" for i in range(1, 8)],
    "openarm_left_finger_joint1",
]
RIGHT_JOINT_NAMES = JOINT_NAMES[:8]
TRIAL_METADATA_FIELDS = (
    "batch_id",
    "trial_id",
    "position_id",
    "cube_forward_m",
    "cube_lateral_m",
    "placement_measurement_method",
    "plan_only_passed",
    "dry_replay_passed",
)


def ordered_joint_positions(message: JointState) -> list[float] | None:
    index = {name: position for position, name in enumerate(message.name)}
    if any(
        name not in index or index[name] >= len(message.position)
        for name in JOINT_NAMES
    ):
        return None
    try:
        values = [float(message.position[index[name]]) for name in JOINT_NAMES]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def validate_trial_metadata(payload: object) -> dict:
    """Validate operator-supplied position identity without adding ROS parameters."""
    if not isinstance(payload, dict):
        raise ValueError("trial metadata must be a JSON object")
    missing = [key for key in TRIAL_METADATA_FIELDS if key not in payload]
    if missing:
        raise ValueError(f"trial metadata missing fields: {missing}")
    result = {key: payload[key] for key in TRIAL_METADATA_FIELDS}
    for key in ("batch_id", "trial_id", "position_id", "placement_measurement_method"):
        value = result[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"trial metadata {key} must be a nonempty string")
        result[key] = value.strip()
    for key in ("cube_forward_m", "cube_lateral_m"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"trial metadata {key} must be finite")
        result[key] = float(value)
        if not math.isfinite(result[key]):
            raise ValueError(f"trial metadata {key} must be finite")
    result.update(capture_preflight(
        payload.get("capture_validation_mode", "prior_verified"),
        result["plan_only_passed"], result["dry_replay_passed"],
    ))
    source_trial_id = payload.get("source_trajectory_trial_id")
    if source_trial_id is not None:
        if not isinstance(source_trial_id, str) or not valid_trial_id(source_trial_id):
            raise ValueError("trial metadata source_trajectory_trial_id is invalid")
        result["source_trajectory_trial_id"] = source_trial_id
    return result


def jpeg_dimensions(data: bytes) -> list[int] | None:
    """Read JPEG SOF dimensions without decoding or adding a runtime dependency."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    offset = 2
    start_of_frame = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return None
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            return None
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            return None
        if marker in start_of_frame:
            if length < 7:
                return None
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return [width, height] if width > 0 and height > 0 else None
        offset += length
    return None


def joint_error_details(actual: list[float], target: list[float]) -> tuple[list[float], str]:
    errors = [abs(float(a) - float(b)) for a, b in zip(actual[:7], target)]
    details = ", ".join(
        "J%d(actual=%.6f,target=%.6f,error=%.6f)" % (index, a, b, error)
        for index, (a, b, error) in enumerate(
            zip(actual[:7], target, errors), start=1
        )
    )
    return errors, details


def demonstrated_action(
    measured: list[float], parked_left: list[float]
) -> list[float]:
    if len(measured) != 16 or len(parked_left) != 8:
        raise ValueError("real demonstration action requires right8 and parked left8")
    return [*measured[:8], *parked_left]


def load_home_provenance(path: Path, maximum_age_s: float) -> dict:
    try:
        raw = path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"HOME report unavailable/invalid: {path}: {error}") from error
    if report.get("workflow") != "FIXED_LIFTED_HOME":
        raise RuntimeError("HOME report has the wrong workflow")
    if report.get("state") != "HOME_READY" or report.get("success") is not True:
        raise RuntimeError("HOME report does not prove HOME_READY")
    gripper = report.get("home_gripper")
    if (
        report.get("j8_policy") != "open_after_arm_home"
        or report.get("j8_commanded_by_initialization") is not True
        or not isinstance(gripper, dict)
        or gripper.get("accepted") is not True
    ):
        raise RuntimeError("HOME report does not prove that J8 is open")
    goal = report.get("home_goal_positions_rad")
    if (
        not isinstance(goal, list)
        or len(goal) != 7
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in goal
        )
    ):
        raise RuntimeError("HOME report has no valid seven-joint HOME target")
    completed = report.get("completed_wall_time_s")
    if not isinstance(completed, (int, float)) or not math.isfinite(float(completed)):
        raise RuntimeError("HOME report has no valid completion time")
    age = time.time() - float(completed)
    if age < -5.0 or age > maximum_age_s:
        raise RuntimeError(
            f"HOME report is not from this session (age={age:.1f}s, limit={maximum_age_s:.1f}s)"
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "completed_wall_time_s": float(completed),
        "trajectory_sha256": str(report.get("trajectory_sha256", "")),
        "config_sha256": str(report.get("config_sha256", "")),
        "home_goal_positions_rad": [float(value) for value in goal],
        "home_gripper": gripper,
    }


def stamp_seconds(message) -> float:
    stamp = message.header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    return value if value > 0.0 else time.time()


class RealEpisodeRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("real_episode_recorder")
        self.declare_parameter("output_root", "~/openarm_kinesthetic_data/collections")
        self.declare_parameter(
            "ceiling_topic", "/camera/global_camera/color/image_raw/compressed"
        )
        self.declare_parameter(
            "wrist_topic", "/camera/wrist_right/image_raw/compressed"
        )
        self.declare_parameter("maximum_source_age_s", 1.0)
        self.declare_parameter(
            "home_report",
            "~/.ros/openarm_kinesthetic/home_initialization.execution.json",
        )
        self.declare_parameter("maximum_home_report_age_s", 300.0)
        self.declare_parameter("maximum_episode_start_home_error_rad", 0.06)
        self.declare_parameter("require_episode_start_home", True)
        self.output_root = Path(
            str(self.get_parameter("output_root").value)
        ).expanduser()
        self.maximum_age = float(self.get_parameter("maximum_source_age_s").value)
        self.home_report = Path(
            str(self.get_parameter("home_report").value)
        ).expanduser()
        self.maximum_home_report_age = float(
            self.get_parameter("maximum_home_report_age_s").value
        )
        self.maximum_episode_start_home_error = float(
            self.get_parameter("maximum_episode_start_home_error_rad").value
        )
        self.require_episode_start_home = bool(
            self.get_parameter("require_episode_start_home").value
        )
        if self.maximum_age <= 0.0:
            raise ValueError("maximum_source_age_s must be positive")
        if self.maximum_home_report_age <= 0.0:
            raise ValueError("maximum_home_report_age_s must be positive")
        if self.maximum_episode_start_home_error <= 0.0:
            raise ValueError("maximum_episode_start_home_error_rad must be positive")
        # Bind this recorder process to the HOME execution that caused launch to
        # start it.  Do not age this out between episodes in the same session.
        self.home_provenance = load_home_provenance(
            self.home_report, self.maximum_home_report_age
        )
        self.recorder = EpisodeRecorder(self.output_root)
        self.latest_state: list[float] | None = None
        self.latest_state_wall_s: float | None = None
        self.latest_reference_wall_s: float | None = None
        self.parked_left_action: list[float] | None = None
        self.passive: bool | None = None
        self.latest_passive_wall_s: float | None = None
        self.latest_camera_wall_s: dict[str, float] = {}
        self.camera_index = {"ceiling": 0, "wrist_right": 0}
        self.camera_topics = {
            "ceiling": str(self.get_parameter("ceiling_topic").value),
            "wrist_right": str(self.get_parameter("wrist_topic").value),
        }
        self.trial_metadata: dict | None = None
        self.active_trial_id: str | None = None
        self.active_source_trajectory_trial_id: str | None = None
        self.active_replay: dict | None = None
        self.active_replay_complete = False
        self.observation_count = 0
        self.reference_count = 0
        self.gripper_events: list[str] = []
        self.camera_stats: dict[str, dict] = {}
        self.status_pub = self.create_publisher(
            String, "/openarm_kinesthetic/recording_status", 10
        )
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        replay_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        passive_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, qos)
        self.create_subscription(
            JointState,
            "/openarm_kinesthetic/expert_action",
            self.on_expert_action,
            qos,
        )
        self.create_subscription(
            String,
            "/openarm_kinesthetic/expert_gripper_event",
            self.on_gripper_event,
            qos,
        )
        self.create_subscription(
            Bool,
            "/openarm_right_hardware/passive_state",
            self.on_passive,
            passive_qos,
        )
        self.create_subscription(
            String,
            "/openarm_kinesthetic/gamepad_events",
            self.on_gamepad_events,
            qos,
        )
        self.create_subscription(
            String,
            "/openarm_kinesthetic/trial_metadata",
            self.on_trial_metadata,
            qos,
        )
        self.create_subscription(
            String,
            "/openarm_kinesthetic/expert_replay_status",
            self.on_replay_status,
            replay_qos,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("ceiling_topic").value),
            lambda message: self.on_camera("ceiling", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("wrist_topic").value),
            lambda message: self.on_camera("wrist_right", message),
            qos_profile_sensor_data,
        )
        self.create_service(Trigger, "~/start", self.start)
        self.create_service(Trigger, "~/finish_success", self.finish_success)
        self.create_service(Trigger, "~/finish_auto_review", self.finish_auto_review)
        self.create_service(Trigger, "~/finish_failure", self.finish_failure)
        self.create_service(Trigger, "~/cancel", self.cancel)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().warn(
            "Real recorder ready. Operator success is saved under "
            "episodes_pending_review and cannot enter training before review."
        )

    @property
    def active(self) -> bool:
        return self.recorder.active is not None

    def abort_active(self, reason: str) -> None:
        if not self.active:
            return
        try:
            path = self.recorder.finish_failure(reason)
            self.get_logger().error(f"episode aborted and saved as failed: {path}")
        except RecorderStateError as error:
            self.get_logger().error(f"episode abort failed: {error}")

    def on_passive(self, message: Bool) -> None:
        self.passive = bool(message.data)
        self.latest_passive_wall_s = time.time()

    def on_joint_state(self, message: JointState) -> None:
        values = ordered_joint_positions(message)
        if values is None:
            return
        timestamp = stamp_seconds(message)
        self.latest_state = values
        self.latest_state_wall_s = time.time()
        if not self.active:
            return
        try:
            self.recorder.append(
                "joint_observation", timestamp, {"positions": values, "order": JOINT_NAMES}
            )
            self.observation_count += 1
        except RecorderStateError as error:
            self.abort_active(f"joint_observation_recording_failure: {error}")

    def on_expert_action(self, message: JointState) -> None:
        index = {name: offset for offset, name in enumerate(message.name)}
        if any(
            name not in index or index[name] >= len(message.position)
            for name in RIGHT_JOINT_NAMES
        ):
            return
        try:
            right = [
                float(message.position[index[name]]) for name in RIGHT_JOINT_NAMES
            ]
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in right):
            return
        self.latest_reference_wall_s = time.time()
        if not self.active or self.parked_left_action is None:
            return
        action = [*right, *self.parked_left_action]
        try:
            self.recorder.append(
                "demonstration_action",
                stamp_seconds(message),
                {
                    "positions": action,
                    "order": JOINT_NAMES,
                    "source": "recorded_motion_player_controller_reference",
                },
            )
            self.reference_count += 1
        except RecorderStateError as error:
            self.abort_active(f"reference_action_recording_failure: {error}")

    def on_gripper_event(self, message: String) -> None:
        if not self.active:
            return
        try:
            payload = json.loads(message.data)
            timestamp = float(payload.pop("timestamp_s"))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().error(f"gripper event rejected: {error}")
            return
        try:
            self.recorder.append("gripper_event", timestamp, payload)
            self.gripper_events.append(str(payload.get("event", "")))
        except RecorderStateError as error:
            self.abort_active(f"gripper_event_recording_failure: {error}")

    def on_camera(self, key: str, message: CompressedImage) -> None:
        timestamp = stamp_seconds(message)
        self.latest_camera_wall_s[key] = time.time()
        if not self.active:
            return
        active = self.recorder.active
        if active is None:
            return
        directory = active / "cameras" / key
        directory.mkdir(parents=True, exist_ok=True)
        sequence = self.camera_index[key]
        self.camera_index[key] += 1
        filename = f"{sequence:08d}_{int(timestamp * 1e9)}.jpg"
        path = directory / filename
        temporary = path.with_suffix(".jpg.tmp")
        try:
            image_bytes = bytes(message.data)
            temporary.write_bytes(image_bytes)
            temporary.replace(path)
            self.recorder.append(
                "camera",
                timestamp,
                {
                    "key": key,
                    "path": str(path.relative_to(active)),
                    "format": str(message.format),
                },
            )
            stats = self.camera_stats[key]
            stats["count"] += 1
            stats["first_timestamp_s"] = (
                timestamp
                if stats["first_timestamp_s"] is None
                else stats["first_timestamp_s"]
            )
            stats["last_timestamp_s"] = timestamp
            dimensions = jpeg_dimensions(image_bytes)
            if dimensions is not None:
                stats["resolutions"].add(tuple(dimensions))
        except (OSError, RecorderStateError) as error:
            self.abort_active(f"camera_{key}_recording_failure: {error}")

    def on_gamepad_events(self, message: String) -> None:
        if not self.active:
            return
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            payload = {"raw": message.data}
        try:
            self.recorder.append("gamepad_events", time.time(), payload)
        except RecorderStateError as error:
            self.abort_active(f"gamepad_event_recording_failure: {error}")

    def on_trial_metadata(self, message: String) -> None:
        if self.active:
            self.get_logger().error("trial metadata rejected while an episode is active")
            return
        try:
            self.trial_metadata = validate_trial_metadata(json.loads(message.data))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.trial_metadata = None
            self.get_logger().error(f"trial metadata rejected: {error}")
            return
        self.get_logger().warn(
            "TRIAL_METADATA_READY batch_id=%s trial_id=%s position_id=%s"
            % (
                self.trial_metadata["batch_id"],
                self.trial_metadata["trial_id"],
                self.trial_metadata["position_id"],
            )
        )

    def on_replay_status(self, message: String) -> None:
        if not self.active:
            return
        try:
            payload = json.loads(message.data)
            state = str(payload["state"])
            digest = str(payload["trajectory_sha256"])
            playback_rate = float(payload["playback_rate"])
            marker_summary = payload["marker_summary"]
            if state not in ("REPLAY_STARTED", "REPLAY_COMPLETE"):
                raise ValueError("invalid replay state")
            if (
                payload.get("schema_version") != 1
                or payload.get("trajectory_mode") != "continuous_marked"
                or payload.get("full_replay") is not True
                or not isinstance(payload.get("trajectory_path"), str)
                or not payload["trajectory_path"]
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not math.isfinite(playback_rate)
                or abs(playback_rate - 1.0) > 1.0e-9
                or not isinstance(marker_summary, list)
                or not all(isinstance(item, dict) for item in marker_summary)
                or [item.get("type") for item in marker_summary]
                != ["gripper_close", "gripper_open"]
                or [item.get("source_button") for item in marker_summary]
                != ["Y", "A"]
                or any(
                    item.get("result") != "semantic_only"
                    for item in marker_summary
                )
            ):
                raise ValueError(
                    "replay must be full-rate Y=CLOSE then A=OPEN collection data"
                )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.abort_active(f"replay_provenance_failure: {error}")
            return
        if self.active_replay is not None and (
            self.active_replay["trajectory_sha256"] != digest
            or self.active_replay["trajectory_path"] != payload["trajectory_path"]
        ):
            self.abort_active("replay_provenance_failure: trajectory changed during episode")
            return
        expected_trajectory_trial_id = (
            getattr(self, "active_source_trajectory_trial_id", None)
            or getattr(self, "active_trial_id", None)
        )
        if valid_trial_id(expected_trajectory_trial_id or "") and (
            payload.get("trajectory_trial_id") != expected_trajectory_trial_id
        ):
            self.abort_active("replay_provenance_failure: trajectory trial_id does not match episode")
            return
        self.active_replay = payload
        self.active_replay_complete = state == "REPLAY_COMPLETE"
        try:
            self.recorder.update_metadata(
                {
                    "trajectory_path": payload["trajectory_path"],
                    "trajectory_sha256": digest,
                    "trajectory_schema_version": payload["trajectory_schema_version"],
                    "trajectory_mode": payload["trajectory_mode"],
                    "trajectory_duration_s": payload["trajectory_duration_s"],
                    "marker_summary": payload["marker_summary"],
                    "playback_rate": playback_rate,
                    "keyframe_settle_tolerance_rad": float(
                        payload.get("keyframe_settle_tolerance_rad", 0.03)
                    ),
                    "full_replay": True,
                    "replay_complete": self.active_replay_complete,
                    "replay_status_timestamp_s": float(payload["timestamp_s"]),
                }
            )
        except (KeyError, TypeError, ValueError, RecorderStateError) as error:
            self.abort_active(f"replay_metadata_recording_failure: {error}")

    def sources_ready(self) -> tuple[bool, str]:
        now = time.time()
        if self.latest_state is None or self.latest_state_wall_s is None:
            return False, "joint state unavailable"
        if now - self.latest_state_wall_s > self.maximum_age:
            return False, "joint state stale"
        missing = [
            key
            for key in ("ceiling", "wrist_right")
            if key not in self.latest_camera_wall_s
            or now - self.latest_camera_wall_s[key] > self.maximum_age
        ]
        if missing:
            return False, f"camera streams unavailable/stale: {missing}"
        return True, "sources ready"

    def start(self, _request, response):
        if self.active:
            response.success = False
            response.message = f"episode already active: {self.recorder.episode_id}"
            return response
        ready, detail = self.sources_ready()
        if not ready:
            response.success = False
            response.message = detail
            return response
        trial_metadata = getattr(self, "trial_metadata", None)
        if trial_metadata is None:
            response.success = False
            response.message = "validated trial metadata is required before episode start"
            return response
        now = time.time()
        if (
            self.passive is not False
            or self.latest_passive_wall_s is None
            or self.count_publishers("/openarm_right_hardware/passive_state") < 1
        ):
            response.success = False
            response.message = "live hardware publisher and rigid passive_state=false are required at episode start"
            return response
        start_home_error = max(
            abs(actual - target)
            for actual, target in zip(
                self.latest_state[:7],
                self.home_provenance["home_goal_positions_rad"],
            )
        )
        if (
            getattr(self, "require_episode_start_home", True)
            and start_home_error > self.maximum_episode_start_home_error
        ):
            response.success = False
            response.message = (
                f"right arm must return to HOME before recording: {start_home_error:.6f} rad"
            )
            return response
        self.parked_left_action = list(self.latest_state[8:])
        self.camera_index = {"ceiling": 0, "wrist_right": 0}
        episode_id = (
            trial_metadata["trial_id"]
            if valid_trial_id(trial_metadata["trial_id"])
            else f"real_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        )
        metadata = {
            **trial_metadata,
            "source": "real_right_continuous_marked_replay",
            "task_contract": "openarm_kinesthetic_right_blue_cube_v1",
            "rate_hz_target": 30,
            "state_action_order": JOINT_NAMES,
            "episode_home": "fixed_lifted_home",
            "initialization_trajectory_in_training_data": False,
            "initial_positions": list(self.latest_state),
            "observation_source": "measured_joint_states",
            "action_source": "recorded_motion_player_controller_reference",
            "j8_policy": "measured_observation_and_commanded_reference_action",
            "left_action_policy": "fixed_at_episode_start",
            "automatic_task_contract_passed": False,
            "home_initialization": dict(self.home_provenance),
            "episode_start_home_error_rad": start_home_error,
            "episode_start_home_required": getattr(
                self, "require_episode_start_home", True
            ),
            "camera_config": {
                key: {
                    "topic": topic,
                    "raw_encoding": "jpeg",
                }
                for key, topic in getattr(
                    self,
                    "camera_topics",
                    {
                        "ceiling": "/camera/global_camera/color/image_raw/compressed",
                        "wrist_right": "/camera/wrist_right/image_raw/compressed",
                    },
                ).items()
            },
        }
        try:
            self.recorder.begin(metadata, episode_id=episode_id)
        except RecorderStateError as error:
            response.success = False
            response.message = str(error)
            return response
        self.trial_metadata = None
        self.active_trial_id = trial_metadata["trial_id"]
        self.active_source_trajectory_trial_id = trial_metadata.get(
            "source_trajectory_trial_id", trial_metadata["trial_id"]
        )
        self.active_replay = None
        self.active_replay_complete = False
        self.observation_count = 0
        self.reference_count = 0
        self.latest_reference_wall_s = None
        self.gripper_events = []
        self.camera_stats = {
            key: {
                "count": 0,
                "first_timestamp_s": None,
                "last_timestamp_s": None,
                "resolutions": set(),
            }
            for key in ("ceiling", "wrist_right")
        }
        response.success = True
        response.message = f"recording started: {episode_id}"
        return response

    def finish_success(self, _request, response):
        return self._finish_review(response, operator_confirmed=True)

    def finish_auto_review(self, _request, response):
        return self._finish_review(response, operator_confirmed=False)

    def _finish_review(self, response, *, operator_confirmed: bool):
        if not self.active:
            response.success = False
            response.message = "no active transaction"
            return response
        failures = []
        if not self.active_replay_complete:
            failures.append("full marked replay did not report REPLAY_COMPLETE")
        if self.observation_count < 2:
            failures.append("joint observation stream has fewer than two samples")
        if self.reference_count < 2:
            failures.append("controller-reference action stream has fewer than two samples")
        if self.gripper_events != ["close", "open"]:
            failures.append(
                f"gripper events must be exactly close/open, got {self.gripper_events}"
            )
        for key in ("ceiling", "wrist_right"):
            stats = self.camera_stats.get(key, {})
            if int(stats.get("count", 0)) < 1:
                failures.append(f"camera {key} has no frames")
            if len(stats.get("resolutions", set())) != 1:
                failures.append(f"camera {key} does not have one stable JPEG resolution")
        if self.latest_state is None:
            failures.append("final joint state unavailable")
            home_errors, home_details = [], ""
        else:
            if (
                self.latest_state_wall_s is None
                or time.time() - self.latest_state_wall_s > self.maximum_age
            ):
                failures.append("final joint state is stale")
            home_errors, home_details = joint_error_details(
                self.latest_state,
                self.home_provenance["home_goal_positions_rad"],
            )
            if max(home_errors) > COLLECTION_END_HOME_TOLERANCE_RAD:
                failures.append(
                    "final measured HOME rejection: "
                    f"max_error={max(home_errors):.6f} "
                    f"limit={COLLECTION_END_HOME_TOLERANCE_RAD:.6f} "
                    f"joints=[{home_details}]"
                )
        if failures:
            response.success = False
            response.message = "; ".join(failures)
            return response
        camera_observation = {}
        for key, stats in self.camera_stats.items():
            duration = float(stats["last_timestamp_s"] - stats["first_timestamp_s"])
            camera_observation[key] = {
                "frame_count": stats["count"],
                "first_timestamp_s": stats["first_timestamp_s"],
                "last_timestamp_s": stats["last_timestamp_s"],
                "observed_fps": (
                    (stats["count"] - 1) / duration if duration > 0.0 else None
                ),
                "resolution_wh": list(next(iter(stats["resolutions"]))),
            }
        try:
            self.recorder.update_metadata(
                {
                    "episode_end_home_max_error_rad": max(home_errors),
                    "episode_end_home_joint_errors_rad": home_errors,
                    "episode_end_home_tolerance_rad": COLLECTION_END_HOME_TOLERANCE_RAD,
                    "stream_counts": {
                        "joint_observation": self.observation_count,
                        "demonstration_action": self.reference_count,
                        "gripper_event": len(self.gripper_events),
                    },
                    "camera_observation": camera_observation,
                }
            )
            audit = audit_real_episode_directory(
                self.recorder.active, JOINT_NAMES
            )
            self.recorder.update_metadata({"automatic_episode_audit": audit})
            if not audit["passed"]:
                response.success = False
                response.message = "automatic episode audit failed: " + "; ".join(
                    audit["failures"]
                )
                return response
            if operator_confirmed:
                path = self.recorder.finish_operator_success(
                    "operator marked success; strict review still required"
                )
            else:
                path = self.recorder.finish_automatic_pending_review(
                    "full replay and raw audit passed; physical task and images require human review"
                )
        except RecorderStateError as error:
            response.success = False
            response.message = str(error)
            return response
        response.success = True
        response.message = f"saved pending review: {path}"
        return response

    def finish_failure(self, _request, response):
        try:
            path = self.recorder.finish_failure("operator_marked_failure")
        except RecorderStateError as error:
            response.success = False
            response.message = str(error)
            return response
        response.success = True
        response.message = f"saved failed episode: {path}"
        return response

    def cancel(self, _request, response):
        try:
            self.recorder.cancel()
        except RecorderStateError as error:
            response.success = False
            response.message = str(error)
            return response
        response.success = True
        response.message = "active episode deleted"
        return response

    def publish_status(self) -> None:
        message = String()
        message.data = json.dumps(
            {
                "active": self.active,
                "episode_id": self.recorder.episode_id,
                "passive": self.passive,
                "sources_ready": self.sources_ready()[0],
                "output_root": str(self.output_root),
                "reference_action_seen": self.latest_reference_wall_s is not None,
                "trial_metadata_ready": self.trial_metadata is not None,
                "pending_trial_id": (
                    self.trial_metadata["trial_id"] if self.trial_metadata else None
                ),
                "replay_complete": self.active_replay_complete,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self.status_pub.publish(message)


def main() -> int:
    rclpy.init()
    node = RealEpisodeRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.abort_active("recorder_shutdown_before_episode_commit")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
