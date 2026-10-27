"""The fail-rigid shutdown path must never wait forever for a ROS service."""

import math
from pathlib import Path
import threading
import time
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from openarm_human_retarget.kinematics import OpenArmRightKinematics
from openarm_kinesthetic.real import OperatorPacket, OperatorWatchdog
from openarm_kinesthetic_recorder.hardware_bridge import (
    PassiveState,
    RightKinestheticBridge,
    bounded_cartesian_command_seed,
    cartesian_direction_fidelity,
    cartesian_joint_velocity,
    directional_joint_limit_step,
    measured_position_is_recoverable,
)
from std_srvs.srv import SetBool
from std_msgs.msg import String


class _Publisher:
    def __init__(self, subscribers=1):
        self.subscribers = subscribers
        self.messages = []

    def get_subscription_count(self):
        return self.subscribers

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)

    def warn(self, message):
        self.warning_messages.append(message)


def _fake_gripper_bridge(measured=-0.10):
    logger = _Logger()
    bridge = SimpleNamespace(
        gripper_target=None,
        gripper_target_label="",
        gripper_requested_at=None,
        gripper_last_publish_at=None,
        measured_gripper=measured,
        automatic_home_handoff=False,
        gripper_pub=_Publisher(),
        gripper_name="openarm_right_finger_joint1",
        gripper_duration=0.5,
        get_logger=lambda: logger,
    )
    bridge.publish_gripper = MethodType(RightKinestheticBridge.publish_gripper, bridge)
    bridge.service_gripper_command = MethodType(
        RightKinestheticBridge.service_gripper_command, bridge
    )
    return bridge, logger


def test_hardware_service_call_has_a_finite_timeout() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "openarm_kinesthetic_recorder"
        / "hardware_bridge.py"
    ).read_text(encoding="utf-8")

    assert "hardware_client.call(request, timeout_sec=" in source
    assert "hardware_client.call(request)" not in source
    assert "callback_group=self.service_group" in source
    assert source.index("self.service_group = ReentrantCallbackGroup()") < source.index(
        "self.on_hardware_state"
    )


def test_automatic_home_handoff_requires_rigid_idle_controller() -> None:
    operator_state = SimpleNamespace(
        command_fresh=True, control_mode="idle", stop_reason=None,
    )
    bridge = SimpleNamespace(
        control_mode="idle",
        passive_enabled=False,
        hardware_passive=False,
        interlock_latched=False,
        automatic_home_handoff=False,
        transition_lock=threading.Lock(),
        operator=SimpleNamespace(decision=lambda: operator_state),
        gripper_target=-0.1,
        cartesian_commanded=[0.0],
        controller_state=lambda: ("active", "none"),
        joint_state_is_fresh=lambda: True,
    )
    response = RightKinestheticBridge.on_automatic_home_handoff(
        bridge, SetBool.Request(data=True), SetBool.Response()
    )
    assert response.success and bridge.automatic_home_handoff
    assert bridge.gripper_target is None and bridge.cartesian_commanded is None

    operator_state.control_mode = "physical_kinesthetic"
    response = RightKinestheticBridge.on_automatic_home_handoff(
        bridge, SetBool.Request(data=False), SetBool.Response()
    )
    assert not response.success and bridge.automatic_home_handoff

    operator_state.control_mode = "idle"
    operator_state.command_fresh = False
    response = RightKinestheticBridge.on_automatic_home_handoff(
        bridge, SetBool.Request(data=False), SetBool.Response()
    )
    assert not response.success and bridge.automatic_home_handoff

    operator_state.command_fresh = True
    response = RightKinestheticBridge.on_automatic_home_handoff(
        bridge, SetBool.Request(data=False), SetBool.Response()
    )
    assert response.success and not bridge.automatic_home_handoff

    bridge.control_mode = "physical_kinesthetic"
    response = RightKinestheticBridge.on_automatic_home_handoff(
        bridge, SetBool.Request(data=True), SetBool.Response()
    )
    assert not response.success and not bridge.automatic_home_handoff


def test_handoff_tracks_live_buttons_without_issuing_manual_commands() -> None:
    manual_commands = []
    bridge = SimpleNamespace(
        automatic_home_handoff=True,
        operator=OperatorWatchdog(),
        interlock_latched=False,
        apply_control_mode=lambda mode: manual_commands.append(mode),
        request_gripper=lambda *_args: manual_commands.append("gripper"),
        reconcile=lambda *_args: manual_commands.append("reconcile"),
    )
    for sequence, active, mode in (
        (0, False, "idle"),
        (1, True, "physical_kinesthetic"),
    ):
        packet = OperatorPacket(
            schema_version=1,
            session_id="pad",
            sequence=sequence,
            sent_monotonic_s=float(sequence),
            connected=True,
            deadman_pressed=active,
            active=active,
            stop_latched=False,
            open_sequence=0,
            close_sequence=0,
            arm_motion_source=mode,
        )
        RightKinestheticBridge.on_operator_command(
            bridge, String(data=packet.to_json())
        )
    assert bridge.operator.decision(time.monotonic()).control_mode == "physical_kinesthetic"
    assert manual_commands == []


def test_hardware_driver_synchronizes_measured_target_before_kp_restore() -> None:
    source = (
        Path(__file__).resolve().parents[5]
        / "openarm_ws/src/openarm_ros2/openarm_hardware/src/"
        "openarm_simple_hardware.cpp"
    ).read_text(encoding="utf-8")
    sync = source.index("pos_commands_[i] = pos_states_[i]")
    transition = source.index("if (passive != passive_applied_)", sync)
    gain = source.index("const double effective_kp", transition)
    assert sync < transition < gain
    assert "if (!passive && passive_applied_)" in source
    assert "double gkp = gripper_kp_" in source
    assert "double gripper_command = pos_commands_[ARM_DOF]" in source
    assert "J8 remains controlled" in source


def test_gripper_edge_publishes_exactly_once_until_measured_target_is_reached() -> None:
    bridge, logger = _fake_gripper_bridge()
    RightKinestheticBridge.request_gripper(bridge, -0.35, "OPEN", now=1.0)
    assert len(bridge.gripper_pub.messages) == 1
    assert bridge.gripper_target == -0.35

    bridge.service_gripper_command(1.05)
    assert len(bridge.gripper_pub.messages) == 1
    bridge.service_gripper_command(1.61)
    assert len(bridge.gripper_pub.messages) == 1

    bridge.measured_gripper = -0.345
    bridge.service_gripper_command(1.12)
    assert bridge.gripper_target is None
    assert any("OPEN reached" in message for message in logger.info_messages)


def test_gripper_command_reports_missing_controller_then_times_out() -> None:
    bridge, logger = _fake_gripper_bridge()
    bridge.gripper_pub.subscribers = 0
    RightKinestheticBridge.request_gripper(bridge, -0.35, "OPEN", now=1.0)
    assert any("controller missing" in message for message in logger.error_messages)

    bridge.service_gripper_command(6.01)
    assert bridge.gripper_target is None
    assert any("timed out" in message for message in logger.error_messages)


def _fake_passive_exit(measured, controller_state="inactive"):
    logger = _Logger()
    messages = []
    bridge = SimpleNamespace(
        passive_enabled=True,
        hardware_passive=True,
        passive_state=PassiveState.PASSIVE,
        transition_target=None,
        transition_lock=threading.Lock(),
        state_lock=threading.Lock(),
        measured=list(measured),
        last_published=None,
        last_stop_reason="none",
        interlock_latched=False,
        rigid_sync_delay=0.0,
        rigid_target_tolerance=0.002,
        kinematics=SimpleNamespace(
            lower_limits=np.full(7, -1.0), upper_limits=np.full(7, 1.0)
        ),
        heartbeat_pub=_Publisher(),
        get_logger=lambda: logger,
    )
    bridge.joint_state_is_fresh = lambda *_args: True
    bridge.measured_snapshot = MethodType(
        RightKinestheticBridge.measured_snapshot, bridge
    )
    bridge.controller_state = lambda: (controller_state, "none")
    bridge.switch_arm_controller = lambda activate: (True, "none")

    def publish_current():
        bridge.last_published = list(bridge.measured)
        messages.append(list(bridge.measured))
        return True

    bridge.publish_arm_current = publish_current
    bridge.call_hardware = lambda enabled: (
        setattr(bridge, "hardware_passive", bool(enabled)) is None,
        "ok",
    )
    bridge.wait_hardware_state = lambda expected: bridge.hardware_passive == expected
    bridge.keep_passive_fail_safe = MethodType(
        RightKinestheticBridge.keep_passive_fail_safe, bridge
    )
    return bridge, messages


def test_passive_stale_target_exit_first_commands_fresh_measured_pose() -> None:
    q2 = np.asarray([0.2, -0.1, 0.3, 0.4, 0.0, -0.2, 0.1])
    bridge, messages = _fake_passive_exit(q2)
    bridge.last_published = [0.8] * 7  # stale HOME/controller target

    RightKinestheticBridge.reconcile(bridge, False)

    assert messages and np.allclose(messages[0], q2)
    assert np.allclose(messages[-1], q2)
    assert bridge.passive_state is PassiveState.RIGID_HOLD_CURRENT
    assert not bridge.hardware_passive


def test_unconfigured_controller_refuses_rigid_restore() -> None:
    bridge, messages = _fake_passive_exit(np.zeros(7), "unconfigured")
    hardware_calls = []
    bridge.call_hardware = lambda enabled: hardware_calls.append(enabled)

    RightKinestheticBridge.reconcile(bridge, False)

    assert not messages and not hardware_calls
    assert bridge.hardware_passive and bridge.passive_enabled
    assert bridge.passive_state is PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED


def test_inactive_controller_before_passive_enable_enters_recovery_once() -> None:
    logger = _Logger()
    bridge = SimpleNamespace(
        automatic_home_handoff=False,
        passive_enabled=False,
        hardware_passive=False,
        transition_lock=threading.Lock(),
        joint_state_is_fresh=lambda: True,
        controller_state=lambda: ("inactive", "none"),
        get_logger=lambda: logger,
        heartbeat_pub=_Publisher(),
        last_stop_reason="none",
    )
    bridge.keep_passive_fail_safe = MethodType(
        RightKinestheticBridge.keep_passive_fail_safe, bridge
    )

    RightKinestheticBridge.reconcile(bridge, True)
    RightKinestheticBridge.reconcile(bridge, True)

    assert bridge.passive_enabled
    assert bridge.passive_state is PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED
    assert len(logger.error_messages) == 1


def test_failed_passive_enable_and_controller_restore_stays_in_recovery() -> None:
    logger = _Logger()
    bridge = SimpleNamespace(
        automatic_home_handoff=False,
        passive_enabled=False,
        hardware_passive=False,
        transition_lock=threading.Lock(),
        joint_state_is_fresh=lambda: True,
        controller_state=lambda: ("active", "none"),
        switch_arm_controller=lambda activate: (
            (False, "activation_failed") if activate else (True, "none")
        ),
        call_hardware=lambda _enabled: (False, "passive_service_failed"),
        get_logger=lambda: logger,
        heartbeat_pub=_Publisher(),
        last_stop_reason="none",
    )
    bridge.keep_passive_fail_safe = MethodType(
        RightKinestheticBridge.keep_passive_fail_safe, bridge
    )

    RightKinestheticBridge.reconcile(bridge, True)

    assert bridge.passive_enabled
    assert bridge.passive_state is PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED
    assert "activation_failed" in bridge.last_stop_reason


def test_finite_pose_outside_urdf_planning_limit_can_restore_rigid_hold() -> None:
    measured = np.zeros(7)
    measured[1] = -0.183299
    valid, reason = measured_position_is_recoverable(measured)
    assert valid
    assert reason == "none"


def test_nonfinite_passive_pose_still_refuses_rigid_restore() -> None:
    measured = np.zeros(7)
    measured[1] = math.nan
    valid, reason = measured_position_is_recoverable(measured)
    assert not valid
    assert reason == "measured_position_invalid"


def test_cartesian_mapping_is_bounded_at_lifted_home() -> None:
    kinematics = OpenArmRightKinematics.from_current_xacro()
    home = np.asarray(
        [-0.1450262670, -0.0151076284, -0.0060887956, 1.3751977241,
         -0.0031428430, -0.1984151663, -0.1222898194]
    )
    qdot, condition = cartesian_joint_velocity(
        kinematics,
        home,
        np.asarray([0.02, 0.0, 0.0, 0.0, 0.0, 0.10]),
        damping=0.08,
        maximum_joint_velocity=0.60,
    )
    assert np.all(np.isfinite(qdot))
    assert np.max(np.abs(qdot)) <= 0.60
    assert np.linalg.norm(qdot) > 0.0
    assert condition < 80.0


def test_cartesian_velocity_limit_preserves_joint_solution_direction() -> None:
    class Kinematics:
        @staticmethod
        def forward(_measured):
            return SimpleNamespace(ee=np.eye(4))

        @staticmethod
        def jacobian(_measured):
            return np.hstack((np.eye(6), np.zeros((6, 1))))

    qdot, _condition = cartesian_joint_velocity(
        Kinematics(),
        np.zeros(7),
        np.asarray([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        damping=0.0,
        maximum_joint_velocity=0.8,
    )
    assert np.isclose(qdot[0], 0.8)
    assert np.isclose(qdot[1], 0.4)


def test_cartesian_rolling_target_cannot_run_away_from_encoder_feedback() -> None:
    measured = np.zeros(7)
    commanded = np.asarray([0.20, -0.20, 0.01, 0.0, 0.0, 0.0, 0.0])
    seed = bounded_cartesian_command_seed(measured, commanded, 0.025)
    assert np.allclose(seed[:3], [0.025, -0.025, 0.01])
    assert np.max(np.abs(seed - measured)) <= 0.025


def test_cartesian_fidelity_exposes_cross_axis_motion() -> None:
    requested = np.asarray([0.20, 0.0, 0.0, 0.0, 0.0, 0.0])
    achieved = np.asarray([0.16, 0.0, 0.04, 0.0, 0.0, 0.0])
    parallel, orthogonal = cartesian_direction_fidelity(requested, achieved)
    assert parallel == pytest.approx(0.8)
    assert orthogonal == pytest.approx(0.2)


def test_cartesian_command_accumulates_from_previous_target() -> None:
    kinematics = OpenArmRightKinematics.from_current_xacro()
    commanded = np.asarray(
        [-0.1450262670, -0.0151076284, -0.0060887956, 1.3751977241,
         -0.0031428430, -0.1984151663, -0.1222898194]
    )
    initial_x = float(kinematics.forward(commanded).ee[0, 3])
    twist = np.asarray([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        qdot, _ = cartesian_joint_velocity(
            kinematics, commanded, twist,
            damping=0.08, maximum_joint_velocity=0.60,
        )
        commanded = directional_joint_limit_step(
            commanded, qdot, 0.05,
            kinematics.lower_limits, kinematics.upper_limits, 0.03,
        )
    final_x = float(kinematics.forward(commanded).ee[0, 3])
    assert final_x - initial_x > 0.025


def test_motor_zero_allows_reachable_x_motion_away_from_joint4_limit() -> None:
    kinematics = OpenArmRightKinematics.from_current_xacro()
    commanded = np.zeros(7)
    twist = np.asarray([0.20, 0.0, 0.0, 0.0, 0.0, 0.0])
    qdot, condition = cartesian_joint_velocity(
        kinematics, commanded, twist,
        damping=0.08, maximum_joint_velocity=0.60,
    )
    target = directional_joint_limit_step(
        commanded, qdot, 0.05,
        kinematics.lower_limits, kinematics.upper_limits, 0.03,
    )
    achieved = kinematics.jacobian(commanded) @ ((target - commanded) / 0.05)

    assert math.isinf(condition)  # singular axes must not block reachable X
    assert target[3] == 0.0  # joint 4 may not move toward its nearby lower limit
    assert achieved[0] > 0.15


def test_soft_joint_margin_allows_recovery_away_from_limit() -> None:
    commanded = np.zeros(7)
    lower = np.full(7, -1.0)
    upper = np.full(7, 1.0)
    commanded[1] = -0.99
    toward = directional_joint_limit_step(
        commanded, np.asarray([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        0.05, lower, upper, 0.03,
    )
    away = directional_joint_limit_step(
        commanded, np.asarray([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        0.05, lower, upper, 0.03,
    )

    assert toward[1] == commanded[1]
    assert away[1] > commanded[1]


def test_physical_drag_requires_full_lt_release_before_cartesian_rearm() -> None:
    logger = _Logger()
    reconcile_calls = []
    bridge = SimpleNamespace(
        operator=SimpleNamespace(stop_reason=None),
        control_mode="physical_kinesthetic",
        cartesian_rearm_required=False,
        interlock_latched=False,
        last_stop_reason="none",
        cartesian_commanded=np.zeros(7),
        passive_enabled=True,
        hardware_passive=True,
        automatic_home_handoff=False,
        get_logger=lambda: logger,
        reconcile=lambda passive: reconcile_calls.append(passive),
    )

    RightKinestheticBridge.apply_control_mode(bridge, "cartesian_jog")

    assert bridge.control_mode == "idle"
    assert bridge.cartesian_rearm_required
    assert bridge.last_stop_reason == "release_lt_after_physical_drag"
    assert reconcile_calls == [False]
    assert any("release LT" in message for message in logger.warning_messages)

    # Continuing to hold LT cannot re-enter Cartesian mode.
    RightKinestheticBridge.apply_control_mode(bridge, "cartesian_jog")
    assert bridge.control_mode == "idle"
    assert bridge.cartesian_rearm_required

    # One idle packet proves LT has been physically released.
    RightKinestheticBridge.apply_control_mode(bridge, "idle")
    assert not bridge.cartesian_rearm_required


def test_passive_switch_leaves_gripper_controller_active() -> None:
    requests = []
    client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: True,
        call=lambda request, timeout_sec: (
            requests.append(request) or SimpleNamespace(ok=True, message="")
        ),
    )
    parameters = {"arm_controller": "right_joint_trajectory_controller"}
    bridge = SimpleNamespace(
        switch_controller_client=client,
        controller_switch_timeout=1.0,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        controller_state=lambda: ("inactive", "none"),
    )

    success, detail = RightKinestheticBridge.switch_arm_controller(bridge, False)

    assert success and detail == "none"
    assert requests[0].activate_controllers == []
    assert requests[0].deactivate_controllers == ["right_joint_trajectory_controller"]
