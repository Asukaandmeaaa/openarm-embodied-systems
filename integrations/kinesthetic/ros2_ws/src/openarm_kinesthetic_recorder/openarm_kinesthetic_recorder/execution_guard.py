"""Pure validation helpers for guarded real initialization execution.

There is deliberately no interactive second-confirmation helper here. The real executor's
authorization is the single explicit `--allow-real-execution` flag plus the launch
requirement `allow_real_execution:=true`, and it proceeds through its automatic gates
without a nonce, a keypress or an RViz prompt (see
docs/NANO_TEACHING_READY_COMMISSIONING.md). An earlier `confirmation_text` helper belonged to a review
gate that was never implemented; it had no caller and was removed rather than kept as an
unused function that implies a gate nobody passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


def duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1.0e-9


@dataclass(frozen=True)
class TrajectoryAudit:
    point_count: int
    duration_s: float
    start_max_delta_rad: float
    goal_max_delta_rad: float
    maximum_velocity_rad_s: float
    maximum_acceleration_rad_s2: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def audit_joint_trajectory(
    trajectory,
    expected_names: Sequence[str],
    captured_start: Sequence[float],
    planned_goal: Sequence[float],
    start_tolerance_rad: float,
    goal_tolerance_rad: float = 0.01,
) -> TrajectoryAudit:
    """Validate ordering, finiteness, timing, and trajectory endpoints."""
    names = list(trajectory.joint_names)
    expected = list(expected_names)
    if names != expected:
        raise ValueError(f"trajectory joint order {names} does not match {expected}")
    if len(captured_start) != len(expected) or len(planned_goal) != len(expected):
        raise ValueError("trajectory start/goal vectors have the wrong length")
    points = list(trajectory.points)
    if len(points) < 2:
        raise ValueError("trajectory must contain at least two points")

    previous_time = -1.0
    maximum_velocity = 0.0
    maximum_acceleration = 0.0
    for index, point in enumerate(points):
        vectors = (
            ("positions", list(point.positions), True),
            ("velocities", list(point.velocities), False),
            ("accelerations", list(point.accelerations), False),
            ("effort", list(point.effort), False),
        )
        for label, values, required in vectors:
            if required and len(values) != len(expected):
                raise ValueError(f"trajectory point {index} has invalid {label} length")
            if values and len(values) != len(expected):
                raise ValueError(f"trajectory point {index} has invalid {label} length")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"trajectory point {index} has non-finite {label}")
        timestamp = duration_seconds(point.time_from_start)
        if timestamp < 0.0 or (index > 0 and timestamp <= previous_time):
            raise ValueError("trajectory timestamps must be nonnegative and strictly increasing")
        previous_time = timestamp
        if point.velocities:
            maximum_velocity = max(
                maximum_velocity, max(abs(float(value)) for value in point.velocities)
            )
        if point.accelerations:
            maximum_acceleration = max(
                maximum_acceleration,
                max(abs(float(value)) for value in point.accelerations),
            )

    start_delta = max(
        abs(float(actual) - float(required))
        for actual, required in zip(points[0].positions, captured_start)
    )
    if start_delta > start_tolerance_rad:
        raise ValueError(
            f"trajectory first point differs from captured start by {start_delta:.6f} rad"
        )
    goal_delta = max(
        abs(float(actual) - float(required))
        for actual, required in zip(points[-1].positions, planned_goal)
    )
    if goal_delta > goal_tolerance_rad:
        raise ValueError(
            f"trajectory final point differs from planned goal by {goal_delta:.6f} rad"
        )
    if previous_time <= 0.0:
        raise ValueError("trajectory duration must be positive")
    return TrajectoryAudit(
        point_count=len(points),
        duration_s=previous_time,
        start_max_delta_rad=start_delta,
        goal_max_delta_rad=goal_delta,
        maximum_velocity_rad_s=maximum_velocity,
        maximum_acceleration_rad_s2=maximum_acceleration,
    )


def pose_errors(
    actual_position: Sequence[float],
    actual_xyzw: Sequence[float],
    target_position: Sequence[float],
    target_xyzw: Sequence[float],
) -> tuple[float, float]:
    if (
        len(actual_position) != 3
        or len(target_position) != 3
        or len(actual_xyzw) != 4
        or len(target_xyzw) != 4
    ):
        raise ValueError("pose vectors have invalid lengths")
    position_error = math.sqrt(
        sum(
            (float(actual) - float(target)) ** 2
            for actual, target in zip(actual_position, target_position)
        )
    )
    actual_norm = math.sqrt(sum(float(value) ** 2 for value in actual_xyzw))
    target_norm = math.sqrt(sum(float(value) ** 2 for value in target_xyzw))
    if actual_norm < 1.0e-12 or target_norm < 1.0e-12:
        raise ValueError("pose quaternion has zero length")
    dot = abs(
        sum(
            float(actual) * float(target)
            for actual, target in zip(actual_xyzw, target_xyzw)
        )
        / (actual_norm * target_norm)
    )
    orientation_error = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
    return position_error, orientation_error
