import json
import time
from types import MethodType, SimpleNamespace

import pytest
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from openarm_kinesthetic_recorder.motion_recorder_node import MotionRecorderNode
from openarm_kinesthetic_recorder.home_measurements import Q_LIFTED_HOME_RIGHT
from openarm_kinesthetic_recorder.recorded_motion import JOINT_NAMES


class _Logger:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)


def _message(stamp: float, names=JOINT_NAMES) -> JointState:
    message = JointState()
    seconds = int(stamp)
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = int(round((stamp - seconds) * 1e9))
    message.name = list(names)
    message.position = [0.01 * index for index in range(len(names))]
    return message


def _recorder() -> SimpleNamespace:
    logger = _Logger()
    first = {
        "sample_index": 0,
        "time_from_start_s": 0.0,
        "source_timestamp_s": 1000.0,
        "positions_rad": [0.0] * 8,
    }
    node = SimpleNamespace(
        recording=True,
        recording_mode="continuous_marked",
        quality={
            "accepted_sample_count": 1,
            "rejected_incomplete_count": 0,
            "rejected_nonfinite_count": 0,
            "rejected_invalid_timestamp_count": 0,
            "timestamp_regression_count": 0,
            "estimated_dropped_sample_count": 0,
            "maximum_sample_gap_s": 0.0,
        },
        latest_positions=None,
        latest_received_at=None,
        latest_joint_timestamp_s=None,
        first_source_timestamp_s=1000.0,
        last_accepted_source_timestamp_s=1000.0,
        minimum_sample_period=0.02,
        last_sample_at=0.0,
        samples=[first],
        markers=[],
        marker_binding_tolerance=0.25,
        last_keyframe_sequence_by_session={},
        get_logger=lambda: logger,
    )
    node._append_marked_sample = MethodType(MotionRecorderNode._append_marked_sample, node)
    return node


def test_continuous_marked_sampling_uses_complete_timestamped_frames_at_target_rate() -> None:
    node = _recorder()
    MotionRecorderNode.on_joint_state(node, _message(1000.010))
    assert len(node.samples) == 1
    MotionRecorderNode.on_joint_state(node, _message(1000.021))
    assert len(node.samples) == 2
    assert node.samples[-1]["sample_index"] == 1
    assert node.samples[-1]["time_from_start_s"] == pytest.approx(0.021)

    MotionRecorderNode.on_joint_state(node, _message(1000.050, JOINT_NAMES[:-1]))
    assert len(node.samples) == 2
    assert node.quality["rejected_incomplete_count"] == 1
    MotionRecorderNode.on_joint_state(node, _message(1000.020))
    assert node.quality["timestamp_regression_count"] == 1


def test_markers_bind_latest_accepted_sample_and_append_without_overwrite() -> None:
    node = _recorder()
    node._append_marked_sample([0.1] * 8, 1000.02)
    for sequence, marker_type, button, target in (
        (1, "keypoint", "LB", None),
        (2, "gripper_close", "Y", -0.10),
    ):
        payload = {
            "schema_version": 1,
            "session_id": "session",
            "sequence": sequence,
            "type": marker_type,
            "source_button": button,
            "source_timestamp_s": 1000.021 + sequence * 0.001,
        }
        if target is not None:
            payload.update(target_position_rad=target, result="requested")
        MotionRecorderNode.on_motion_marker_event(
            node, String(data=json.dumps(payload))
        )
    assert [marker["marker_index"] for marker in node.markers] == [0, 1]
    assert [marker["sample_index"] for marker in node.markers] == [1, 1]
    assert node.markers[0]["type"] == "keypoint"
    assert node.markers[1]["type"] == "gripper_close"
    assert node.markers[1]["target_position_rad"] == -0.10


def test_save_rejects_excessive_sample_gap_before_writing() -> None:
    logger = _Logger()
    statuses = []
    node = SimpleNamespace(
        recording_mode="continuous_marked",
        latest_positions=[0.0] * 8,
        samples=[
            {"time_from_start_s": 0.0, "source_timestamp_s": 1000.0},
            {"time_from_start_s": 0.3, "source_timestamp_s": 1000.3},
        ],
        markers=[{"marker_index": 0}],
        quality={"maximum_sample_gap_s": 0.3},
        maximum_sample_gap=0.25,
        latest_received_at=time.monotonic(),
        latest_joint_timestamp_s=1000.3,
        joint_watchdog=0.2,
        recording=True,
        get_logger=lambda: logger,
        publish_status=lambda state, path: statuses.append((state, path)),
    )
    MotionRecorderNode.finish_recording(node)
    assert node.recording is False
    assert node.samples == []
    assert node.markers == []
    assert statuses == [("REJECTED", "")]
    assert "sample gap" in logger.errors[-1]


def test_gap_diagnostic_records_sample_and_stage() -> None:
    node = _recorder()
    node.auto_home_stage = "EXECUTE"
    node._append_marked_sample([0.0] * 8, 1000.35)

    assert node.quality["maximum_sample_gap_s"] == pytest.approx(0.35)
    assert node.quality["maximum_sample_gap_sample_index"] == 1
    assert node.quality["maximum_sample_gap_start_s"] == pytest.approx(0.0)
    assert node.quality["maximum_sample_gap_stage"] == "EXECUTE"


def test_failed_save_at_measured_home_releases_bridge_handoff() -> None:
    logger = _Logger()
    requests = []
    future = SimpleNamespace(
        done=lambda: True,
        result=lambda: SimpleNamespace(success=True),
    )
    client = SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda request: requests.append(request.data) or future,
    )
    node = SimpleNamespace(
        handoff_client=client,
        auto_home_stage="SETTLE",
        get_logger=lambda: logger,
    )
    node.release_rejected_recording_at_home = MethodType(
        MotionRecorderNode.release_rejected_recording_at_home, node
    )

    node.release_rejected_recording_at_home()
    assert requests == [False]
    assert node.auto_home_stage == "RELEASE_REJECTED"
    # The production timer handles the response without treating it as a save.
    node.auto_home_future = future
    node.auto_home_deadline = time.monotonic() + 5.0
    MotionRecorderNode.service_auto_home(node)
    assert node.auto_home_stage == "IDLE"
    assert any("REJECTED_RECORDING_RELEASED" in item for item in logger.warnings)


def test_second_x_keeps_sampling_until_automatic_home_is_measured(monkeypatch) -> None:
    class _Future:
        def __init__(self):
            self.response = SimpleNamespace(success=True, message="ready")

        def done(self):
            return True

        def result(self):
            return self.response

    class _Client:
        def __init__(self):
            self.requests = []

        def service_is_ready(self):
            return True

        def call_async(self, request):
            self.requests.append(request.data)
            return _Future()

    class _Process:
        commands = []

        def __init__(self, command, **_kwargs):
            self.command = command
            self.commands.append(command)

        def poll(self):
            return 0

    monkeypatch.setattr(
        "openarm_kinesthetic_recorder.motion_recorder_node.subprocess.Popen",
        _Process,
    )
    logger = _Logger()
    client = _Client()
    saves = []
    node = SimpleNamespace(
        recording=True,
        recording_mode="continuous_marked",
        auto_return_home_on_second_x=True,
        auto_home_stage="IDLE",
        auto_home_execute_timeout=180.0,
        auto_home_velocity_scaling=0.10,
        auto_home_acceleration_scaling=0.10,
        auto_home_future=None,
        auto_home_process=None,
        auto_home_settle_count=0,
        auto_home_last_stamp=None,
        handoff_client=client,
        markers=[
            {"type": "gripper_close", "source_button": "Y", "result": "semantic_only"},
            {"type": "gripper_open", "source_button": "A", "result": "semantic_only"},
        ],
        latest_positions=[*Q_LIFTED_HOME_RIGHT, -0.35],
        latest_received_at=time.monotonic(),
        latest_joint_timestamp_s=1000.0,
        joint_watchdog=0.2,
        last_saved_path="",
        get_logger=lambda: logger,
        publish_status=lambda *_args: None,
    )
    node.request_auto_home_finish = MethodType(MotionRecorderNode.request_auto_home_finish, node)
    node.start_auto_home_player = MethodType(MotionRecorderNode.start_auto_home_player, node)
    node.service_auto_home = MethodType(MotionRecorderNode.service_auto_home, node)
    node.finish_recording = lambda: (saves.append(True), setattr(node, "last_saved_path", "/tmp/new.json"))

    MotionRecorderNode.request_auto_home_finish(node)
    assert node.recording and not saves and client.requests == [True]
    for _ in range(2):
        node.service_auto_home()
    assert node.auto_home_stage == "SETTLE" and not saves
    assert len(_Process.commands) == 1
    assert "--allow-real-execution" in _Process.commands[0]
    assert "--plan-only" not in _Process.commands[0]
    for stamp in (1001.0, 1002.0, 1003.0):
        node.latest_joint_timestamp_s = stamp
        node.latest_received_at = time.monotonic()
        node.service_auto_home()
    assert len(saves) == 1
    assert client.requests == [True, False]


def test_active_continuous_recording_routes_second_x_to_auto_home() -> None:
    calls = []
    node = SimpleNamespace(
        recording=True,
        recording_mode="continuous_marked",
        auto_return_home_on_second_x=True,
        last_toggle_sequence_by_session={},
        request_auto_home_finish=lambda: calls.append("home"),
        finish_recording=lambda: calls.append("save"),
    )
    toggle = String(data=json.dumps({
        "schema_version": 1, "session_id": "pad", "sequence": 1,
    }))
    MotionRecorderNode.on_toggle(node, toggle)
    MotionRecorderNode.on_toggle(node, toggle)
    assert calls == ["home"]


def test_auto_home_timeout_keeps_player_tracked_until_cancel_exits(monkeypatch) -> None:
    class _Process:
        pid = 31415
        exit_code = None

        def poll(self):
            return self.exit_code

    process = _Process()
    signals = []
    monkeypatch.setattr(
        "openarm_kinesthetic_recorder.motion_recorder_node.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    logger = _Logger()
    statuses = []
    node = SimpleNamespace(
        auto_home_process=process,
        auto_home_stage="EXECUTE",
        recording=True,
        samples=[{"sample_index": 0}],
        markers=[{"marker_index": 0}],
        get_logger=lambda: logger,
        publish_status=lambda state, path: statuses.append((state, path)),
    )
    MotionRecorderNode.reject_auto_home(node, "execution timeout")
    assert node.auto_home_stage == "ABORTING"
    assert node.auto_home_process is process
    assert not node.recording and statuses == [("REJECTED", "")]
    assert len(signals) == 1

    process.exit_code = 2
    MotionRecorderNode.service_auto_home(node)
    assert node.auto_home_stage == "FAILED"
    assert node.auto_home_process is None
