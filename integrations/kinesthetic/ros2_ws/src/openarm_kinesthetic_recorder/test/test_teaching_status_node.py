from openarm_kinesthetic_recorder.teaching_status_node import combined_status


def test_combined_status_starts_unknown_without_faking_readiness() -> None:
    status = combined_status(None, None, None)

    assert all(value is None for value in status.values())


def test_combined_status_preserves_all_safety_signals() -> None:
    status = combined_status(
        {
            "connected": True,
            "deadman_pressed": False,
            "control_mode": "idle",
        },
        {
            "command_fresh": True,
            "operator_armed": True,
            "joint_state_fresh": True,
            "passive_enabled": False,
            "interlock_latched": False,
            "stop_reason": None,
            "control_mode": "idle",
        },
        False,
    )

    assert status == {
        "gamepad_connected": True,
        "deadman_pressed": False,
        "control_mode": "idle",
        "operator_command_fresh": True,
        "operator_armed": True,
        "joint_state_fresh": True,
        "passive_enabled": False,
        "hardware_passive": False,
        "interlock_latched": False,
        "stop_reason": None,
    }


def test_combined_status_exposes_interlock_and_passive_state() -> None:
    status = combined_status(
        {"connected": True, "deadman_pressed": True},
        {
            "command_fresh": True,
            "operator_armed": True,
            "joint_state_fresh": True,
            "passive_enabled": True,
            "interlock_latched": True,
            "stop_reason": "operator_stop_latched",
        },
        True,
    )

    assert status["hardware_passive"] is True
    assert status["passive_enabled"] is True
    assert status["interlock_latched"] is True
    assert status["stop_reason"] == "operator_stop_latched"
