from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
from std_msgs.msg import String

# The ROS adapter deliberately reuses the workspace's validated protocol and
# Linux joystick implementation.  Colcon's isolated pytest environment does
# not inherit the workspace-level PYTHONPATH used by the deployed launch.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from openarm_kinesthetic_recorder import gamepad_node


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warn(self, message):
        self.warning_messages.append(message)


def _fake_node(device="/dev/input/js0"):
    logger = _Logger()
    node = SimpleNamespace(
        joystick=None,
        last_connect_attempt=0.0,
        last_connect_error="",
        last_missing_log=0.0,
        device_request=device,
        config={},
        mapper=SimpleNamespace(button={"hand_drag_lb": 6}),
        get_logger=lambda: logger,
    )
    return node, logger


def test_local_gamepad_connects_to_explicit_nano_device(monkeypatch) -> None:
    node, logger = _fake_node()
    joystick = SimpleNamespace(
        connected=True,
        path="/dev/input/js0",
        name="Gamepad MACHENIKE G3V2",
        button_index_for_code=lambda _code: 6,
    )
    monkeypatch.setattr(
        gamepad_node.LinuxJoystick,
        "open_configured",
        lambda requested, config: joystick,
    )

    gamepad_node.RealGamepadNode.try_connect(node, 10.0)

    assert node.joystick is joystick
    assert node.last_connect_error == ""
    assert "gamepad connected" in logger.info_messages[-1]
    assert "LB_button=6 source=kernel_BTN_TL" in logger.info_messages[-1]


def test_local_gamepad_keeps_reviewed_lb_fallback_without_kernel_map(
    monkeypatch,
) -> None:
    node, logger = _fake_node()
    joystick = SimpleNamespace(
        connected=True,
        path="/dev/input/js0",
        name="Gamepad MACHENIKE G3V2",
        button_index_for_code=lambda _code: None,
    )
    monkeypatch.setattr(
        gamepad_node.LinuxJoystick,
        "open_configured",
        lambda requested, config: joystick,
    )

    gamepad_node.RealGamepadNode.try_connect(node, 10.0)

    assert node.mapper.button["hand_drag_lb"] == 6
    assert "LB_button=6 source=yaml_fallback" in logger.info_messages[-1]


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("missing"), PermissionError("permission denied")],
)
def test_local_gamepad_connection_failure_stays_fail_rigid(
    monkeypatch,
    error,
) -> None:
    node, logger = _fake_node()

    def fail(_requested, _config):
        raise error

    monkeypatch.setattr(gamepad_node.LinuxJoystick, "open_configured", fail)

    gamepad_node.RealGamepadNode.try_connect(node, 10.0)

    assert node.joystick is None
    assert str(error) in node.last_connect_error
    assert "bridge remains rigid" in logger.warning_messages[-1]


def test_keyframe_gripper_is_semantic_only_while_recorder_is_active() -> None:
    logger = _Logger()
    node = SimpleNamespace(
        keyframe_recording_active=False,
        get_logger=lambda: logger,
    )

    message = String()
    message.data = json.dumps({"state": "RECORDING"})
    gamepad_node.RealGamepadNode.on_motion_record_status(node, message)
    assert node.keyframe_recording_active is True
    assert "do not drive real gripper" in logger.warning_messages[-1]

    message.data = json.dumps({"state": "SAVED"})
    gamepad_node.RealGamepadNode.on_motion_record_status(node, message)
    assert node.keyframe_recording_active is False
    assert "control real gripper" in logger.warning_messages[-1]


def test_drag_l3_is_not_disabled_by_keyframe_semantic_mode() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "openarm_kinesthetic_recorder"
        / "gamepad_node.py"
    ).read_text(encoding="utf-8")
    condition = "if physical_drag_intent and command.hand_drag_gripper_toggle_pressed:"
    assert condition in source


def test_continuous_marked_event_is_append_only_and_records_target(monkeypatch) -> None:
    published = []
    logger = _Logger()
    node = SimpleNamespace(
        marker_sequence=0,
        session_id="session",
        motion_marker_publisher=SimpleNamespace(publish=published.append),
        get_logger=lambda: logger,
    )
    monkeypatch.setattr(gamepad_node.time, "time", lambda: 1234.5)
    gamepad_node.RealGamepadNode.publish_motion_marker(
        node, "gripper_close", "Y", 10.0, -0.10
    )
    gamepad_node.RealGamepadNode.publish_motion_marker(
        node, "keypoint", "LB", 10.1
    )
    first, second = [json.loads(message.data) for message in published]
    assert first["sequence"] == 1
    assert first["target_position_rad"] == -0.10
    assert first["result"] == "requested"
    assert second["sequence"] == 2
    assert second["type"] == "keypoint"
    assert "target_position_rad" not in second


def test_continuous_marked_status_keeps_real_gripper_enabled() -> None:
    logger = _Logger()
    node = SimpleNamespace(
        keyframe_recording_active=False,
        motion_recording_mode="continuous_marked",
        get_logger=lambda: logger,
    )
    message = String(data=json.dumps({"state": "RECORDING"}))
    gamepad_node.RealGamepadNode.on_motion_record_status(node, message)
    assert node.keyframe_recording_active is True
    assert "semantic_marker_only" in logger.warning_messages[-1]


def test_continuous_marked_defaults_to_semantic_ay_and_no_lb_keypoints() -> None:
    publish_marker, drive_gripper = gamepad_node.continuous_marked_gripper_policy(
        "continuous_marked", True, False
    )
    assert publish_marker is True
    assert drive_gripper is False
    assert not gamepad_node.continuous_marked_lb_enabled(
        "continuous_marked", True, False
    )


def test_continuous_marked_legacy_controls_can_be_explicitly_enabled() -> None:
    publish_marker, drive_gripper = gamepad_node.continuous_marked_gripper_policy(
        "continuous_marked", True, True
    )
    assert publish_marker is True
    assert drive_gripper is True
    assert gamepad_node.continuous_marked_lb_enabled(
        "continuous_marked", True, True
    )
