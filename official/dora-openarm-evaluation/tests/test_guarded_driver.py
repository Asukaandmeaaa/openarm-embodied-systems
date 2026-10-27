"""Driver-boundary tests for the guarded direct-CAN startup path.

These tests exercise the Python driver's real safety checkers but use a small
in-memory driver/backend double, so importing and running them never opens
SocketCAN.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if str(SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SITE_PACKAGES))

spec = importlib.util.spec_from_file_location(
    "guarded_home_driver_test_module",
    ROOT / "nodes/dora-openarm/src/dora_openarm/guarded_home.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

Guard = module.Guard
_recovery_checker = module._recovery_checker
_send_guarded_position = module._send_guarded_position
_require_can_health = module._require_can_health
_validate_driver_config = module._validate_driver_config


def _config() -> dict:
    return json.loads((ROOT / "guarded-home-right.json").read_text())


def _normal_checker(guard):
    """Build the same checker stack used by SingleArmDriver by default."""
    from openarm_driver.base_safety import CompositeChecker
    from openarm_driver.safety import (
        JointDeltaPosChecker,
        JointPosChecker,
        JointVelocityChecker,
    )

    return CompositeChecker(
        [
            JointPosChecker(np.column_stack((guard.lower, guard.upper))),
            JointDeltaPosChecker(guard.single_jump),
            JointVelocityChecker(guard.velocity),
        ]
    )


class _LinkStats:
    def __init__(self, responses: int = 1, stale: bool = False):
        self.responses = responses
        self._stale = stale

    def is_stale(self, _timeout: float) -> bool:
        return self._stale


class _ArmBackend:
    def __init__(self, links: list[_LinkStats] | None = None):
        self.links = links or [_LinkStats() for _ in range(8)]

    def get_link_stats(self, index: int) -> _LinkStats:
        return self.links[index]


class _OpenArmBackend:
    def __init__(self, links: list[_LinkStats] | None = None):
        self.arm = _ArmBackend(links)

    def get_arm(self) -> _ArmBackend:
        return self.arm


class _FakeDriver:
    """A no-CAN stand-in that applies the real checker protocol."""

    def __init__(self, guard, *, send_error: bool = False, links=None):
        self.can_interface = "can0"
        self.safety_checker = _normal_checker(guard)
        self.last_command = np.zeros(8, dtype=float)
        self.last_command_time_s = time.monotonic() - 1.0
        self.last_command_dispatch_timestamp_ns = None
        self.dispatched: list[np.ndarray] = []
        self.send_error = send_error
        self.openarm = _OpenArmBackend(links)
        self.motor_status = ["ENABLED"] * 8
        self.bus_status = {
            "carrier": True,
            "bus_off": 0,
            "error_passive": 0,
            "error_warning": 0,
            "tx_overflow": 0,
            "rx_overflow": 0,
            "ack_error": 0,
            "tx_timeout": 0,
            "restarted": 0,
            "write_net_down": 0,
            "write_no_buffer": 0,
            "write_other": 0,
        }

    def send_position(self, position) -> bool:
        if self.send_error:
            raise RuntimeError("fake CAN dispatch failure")
        now = time.monotonic()
        dt_s = min(max(now - self.last_command_time_s, 0.0), 0.04)
        result = self.safety_checker.check(
            position,
            driver=self,
            dt_s=dt_s,
        )
        if not result.is_safe:
            if result.force_stop:
                return False
            if result.fixed_joint_positions is not None:
                position = result.fixed_joint_positions
        position = np.asarray(position, dtype=float).copy()
        self.last_command = position
        self.last_command_time_s = now
        self.last_command_dispatch_timestamp_ns = time.time_ns()
        self.dispatched.append(position.copy())
        return True

    def get_health(self):
        return self.motor_status, self.bus_status


def _guard(initial=None) -> Guard:
    if initial is None:
        initial = np.array(
            [0.1157778, -0.0089647, -0.1253147, -0.1955062,
             0.7047761, -0.1199741, 0.1249332, -0.0562676],
            dtype=float,
        )
    return Guard(_config(), np.asarray(initial, dtype=float), 0.0)


def test_recovery_checker_preserves_negative_j4_first_dispatch_and_tracks_actual():
    guard = _guard()
    arm = _FakeDriver(guard)
    recovery_checker = _recovery_checker(guard)

    # A target below the normal J4 lower bound is valid only in H1's explicit
    # activation envelope. The actual checker must preserve it for MIT dispatch.
    first = guard.command.copy()
    # ROS-equivalent startup seeds the driver's previous command from the same
    # measured feedback before the first zero-directed waypoint is emitted.
    arm.last_command = first.copy()
    arm.last_command_time_s = time.monotonic() - 1.0
    _send_guarded_position(arm, guard, first, recovery_checker)

    np.testing.assert_allclose(arm.dispatched[-1], first)
    np.testing.assert_allclose(arm.last_command, first)
    np.testing.assert_allclose(guard.command, arm.last_command)
    assert arm.safety_checker is not recovery_checker


def test_recovery_dispatch_tracks_driver_rate_limited_command():
    guard = _guard(np.zeros(8))
    arm = _FakeDriver(guard)
    recovery_checker = _recovery_checker(guard)
    # Force the real velocity checker to limit this first dispatch by making
    # the driver's previous command timestamp effectively "now".
    arm.last_command_time_s = time.monotonic()
    requested = guard.command.copy()
    requested[0] = 0.5

    _send_guarded_position(arm, guard, requested, recovery_checker)

    assert arm.dispatched[-1][0] < requested[0]
    np.testing.assert_allclose(guard.command, arm.last_command)


def test_normal_checker_restored_after_dispatch_error():
    guard = _guard()
    arm = _FakeDriver(guard, send_error=True)
    normal_checker = arm.safety_checker
    recovery_checker = _recovery_checker(guard)

    with pytest.raises(RuntimeError, match="fake CAN dispatch failure"):
        _send_guarded_position(arm, guard, guard.command, recovery_checker)

    assert arm.safety_checker is normal_checker


def test_normal_phase_rejects_negative_j4_before_driver_dispatch():
    guard = _guard(np.zeros(8))
    guard.recovery_scope = False
    arm = _FakeDriver(guard)
    normal_checker = arm.safety_checker
    recovery_checker = _recovery_checker(guard)
    target = guard.command.copy()
    target[3] = -0.01

    with pytest.raises(ValueError, match="HOME/ACT dispatch"):
        _send_guarded_position(arm, guard, target, recovery_checker)

    assert not arm.dispatched
    assert arm.safety_checker is normal_checker


def test_require_can_health_accepts_live_eight_axis_backend():
    guard = _guard(np.zeros(8))
    arm = _FakeDriver(guard)
    motors, bus = _require_can_health(arm)
    assert motors == ["ENABLED"] * 8
    assert bus["carrier"] is True


@pytest.mark.parametrize(
    "links, bus_change, expected",
    [
        (
            [_LinkStats()] * 7 + [_LinkStats(responses=0)],
            {},
            "CAN feedback stale or missing: J8",
        ),
        (
            [_LinkStats()] * 7 + [_LinkStats(stale=True)],
            {},
            "CAN feedback stale or missing: J8",
        ),
        (
            None,
            {"bus_off": 1},
            "CAN bus not healthy",
        ),
    ],
)
def test_require_can_health_rejects_missing_or_faulted_feedback(links, bus_change, expected):
    guard = _guard(np.zeros(8))
    arm = _FakeDriver(guard, links=links)
    arm.bus_status.update(bus_change)

    with pytest.raises(RuntimeError, match=expected):
        _require_can_health(arm)


def test_validate_driver_config_matches_actual_installed_config():
    from openarm_driver import Config

    guard = Guard(_config(), np.zeros(8), 0.0)
    config = Config(ROOT / "openarm_real_ros_coordinates.yaml")
    _validate_driver_config(config, guard)


def test_validate_driver_config_rejects_coordinate_mismatch():
    from openarm_driver import Config

    guard = Guard(_config(), np.zeros(8), 0.0)
    config = Config(ROOT / "openarm_real_ros_coordinates.yaml")
    config._config["joint_offsets"]["right_arm"][3] = 0.01

    with pytest.raises(ValueError, match="coordinate frame"):
        _validate_driver_config(config, guard)
