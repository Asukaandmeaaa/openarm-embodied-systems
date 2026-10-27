"""Guarded-execution helper tests.

This file used to import a `confirmation_text` helper and assert the phrase
`EXECUTE <nonce>`. That helper belonged to an interactive second-confirmation gate the real
recorder never implemented: the executor's authorization is the single explicit
`--allow-real-execution` flag plus the launch requirement `allow_real_execution:=true`, and
it then runs its automatic gates with no nonce, keypress or RViz prompt (see
docs/NANO_TEACHING_READY_COMMISSIONING.md and docs/TABLE_INITIALIZATION.md). The
helper had no production caller, so keeping a test for it - or inventing a caller to make
the test pass - would have documented a gate that does not exist. The assertion is gone and
the helper was removed with it.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT / "ros2_ws" / "src" / "openarm_kinesthetic_recorder")
)

from openarm_kinesthetic_recorder.execution_guard import (  # noqa: E402
    audit_joint_trajectory,
    pose_errors,
)


def _duration(seconds: float):
    whole = int(seconds)
    return SimpleNamespace(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


def _point(time_s, positions, velocities=None, accelerations=None):
    return SimpleNamespace(
        positions=list(positions),
        velocities=list(velocities or []),
        accelerations=list(accelerations or []),
        effort=[],
        time_from_start=_duration(time_s),
    )


def test_trajectory_audit_accepts_ordered_finite_reviewed_path() -> None:
    names = [f"joint{index}" for index in range(3)]
    start = [0.0, 0.1, -0.1]
    goal = [0.3, 0.2, 0.0]
    trajectory = SimpleNamespace(
        joint_names=names,
        points=[
            _point(0.0, start, [0.0] * 3, [0.0] * 3),
            _point(1.0, [0.15, 0.15, -0.05], [0.15] * 3, [0.2] * 3),
            _point(2.0, goal, [0.0] * 3, [0.0] * 3),
        ],
    )
    audit = audit_joint_trajectory(trajectory, names, start, goal, 0.01)
    assert audit.point_count == 3
    assert audit.duration_s == 2.0
    assert audit.maximum_velocity_rad_s == pytest.approx(0.15)
    assert audit.maximum_acceleration_rad_s2 == pytest.approx(0.2)


def test_trajectory_audit_rejects_joint_reordering_and_nonmonotonic_time() -> None:
    names = ["joint1", "joint2"]
    trajectory = SimpleNamespace(
        joint_names=list(reversed(names)),
        points=[_point(0.0, [0.0, 0.0]), _point(1.0, [0.1, 0.1])],
    )
    with pytest.raises(ValueError, match="joint order"):
        audit_joint_trajectory(trajectory, names, [0.0, 0.0], [0.1, 0.1], 0.01)
    trajectory.joint_names = names
    trajectory.points[1].time_from_start = _duration(0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        audit_joint_trajectory(trajectory, names, [0.0, 0.0], [0.1, 0.1], 0.01)


def test_pose_error_contract() -> None:
    position_error, orientation_error = pose_errors(
        [0.001, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    assert position_error == pytest.approx(0.001)
    assert orientation_error == pytest.approx(0.0)
    _, right_angle = pose_errors(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    assert right_angle == pytest.approx(math.pi / 2.0)
