from openarm_kinesthetic.control.gripper import GripperController, GripperState
from openarm_kinesthetic.validation.gate_c import validate_gate_c


def test_keyboard_gripper_latches_targets_and_ignores_repeat() -> None:
    controller = GripperController()
    assert controller.OPEN_TARGET == -0.35
    close = controller.key_down("c", 1.0)
    repeated = controller.key_down("c", 1.01)
    assert close.accepted and close.effective_target == controller.CLOSE_TARGET
    assert not repeated.accepted and repeated.rejection_reason == "key_repeat_ignored"
    controller.update(controller.CLOSE_TARGET)
    assert controller.state == GripperState.HOLDING
    controller.key_up("c")
    controller.key_down("o", 2.0)
    controller.update(controller.OPEN_TARGET)
    assert controller.state == GripperState.OPEN


def test_stop_is_latched_and_requires_explicit_reset() -> None:
    controller = GripperController()
    controller.key_down("Esc", 1.0)
    controller.key_up("Esc")
    assert controller.state == GripperState.STOPPED
    rejected = controller.key_down("C", 2.0)
    assert not rejected.accepted


def test_passive_dynamics_numerical_gate() -> None:
    report = validate_gate_c()
    assert report["status"] == "PASS", report
