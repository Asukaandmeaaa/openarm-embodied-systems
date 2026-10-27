"""Pure helpers for fail-rigid HOME execution and path-quality auditing.

This module deliberately has no ROS imports.  The real executor feeds it controller-state
samples and MoveIt FK results; tests can therefore exercise every acceptance rule without
creating an action client or touching hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence


RIGHT_ARM_JOINT_NAMES = tuple(
    f"openarm_right_joint{index}" for index in range(1, 8)
)


def _seven(values: Sequence[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 7 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain seven finite values")
    return result


@dataclass
class ExecutionTrackingAudit:
    """Collect the JTC desired/actual/error trace without weakening acceptance."""

    joint_names: tuple[str, ...] = RIGHT_ARM_JOINT_NAMES
    samples: list[dict] = field(default_factory=list)

    def add_sample(
        self,
        timestamp_s: float,
        commanded: Sequence[float],
        measured: Sequence[float],
        *,
        measured_velocity: Sequence[float] | None = None,
        measured_effort: Sequence[float] | None = None,
    ) -> None:
        command = _seven(commanded, "commanded positions")
        actual = _seven(measured, "measured positions")
        velocity = (
            None
            if measured_velocity is None
            else _seven(measured_velocity, "measured velocity")
        )
        effort = (
            None if measured_effort is None else _seven(measured_effort, "measured effort")
        )
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("tracking timestamp must be finite")
        self.samples.append(
            {
                "timestamp_s": timestamp,
                "commanded_positions_rad": command,
                "measured_positions_rad": actual,
                "tracking_errors_rad": [a - c for a, c in zip(actual, command)],
                "measured_velocities_rad_s": velocity,
                "measured_efforts_nm": effort,
            }
        )

    def as_dict(self) -> dict:
        per_joint = []
        for index, name in enumerate(self.joint_names):
            errors = [sample["tracking_errors_rad"][index] for sample in self.samples]
            goals = [sample["commanded_positions_rad"][index] for sample in self.samples]
            actual = [sample["measured_positions_rad"][index] for sample in self.samples]
            per_joint.append(
                {
                    "joint_name": name,
                    "goal_rad": None if not goals else goals[-1],
                    "actual_rad": None if not actual else actual[-1],
                    "final_error_rad": None if not errors else errors[-1],
                    "maximum_absolute_tracking_error_rad": (
                        None if not errors else max(abs(value) for value in errors)
                    ),
                }
            )
        return {
            "sample_count": len(self.samples),
            "joint_names": list(self.joint_names),
            "per_joint": per_joint,
            "samples": list(self.samples),
            "telemetry": {
                "position": "available:/joint_states_and_JTC_controller_state",
                "velocity": "available_when_published_by_joint_state_broadcaster",
                "torque": "available_as_sensor_msgs/JointState.effort; hardware source DAMIAO get_torque()",
                "current": "unavailable:not_exported_by_openarm_hardware",
                "temperature": "unavailable:not_exported_by_openarm_hardware",
                "motor_error_or_protection_state": "unavailable:not_exported_by_openarm_hardware",
            },
        }


def final_acceptance(
    measured: Sequence[float], goal: Sequence[float], tolerance_rad: float
) -> dict:
    actual = _seven(measured, "measured positions")
    target = _seven(goal, "goal positions")
    tolerance = float(tolerance_rad)
    errors = [a - b for a, b in zip(actual, target)]
    maximum = max(abs(value) for value in errors)
    return {
        "accepted": maximum <= tolerance,
        "joint_errors_rad": errors,
        "maximum_absolute_error_rad": maximum,
        "tolerance_rad": tolerance,
    }


def validate_station_axes(axes: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Validate the configured real-station frame; world X is never assumed."""

    result = {name: [float(value) for value in axes[name]] for name in ("forward", "lateral", "vertical")}
    for name, vector in result.items():
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"station axis {name} must contain three finite values")
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > 1.0e-6:
            raise ValueError(f"station axis {name} must be unit length")
    for left, right in (("forward", "lateral"), ("forward", "vertical"), ("lateral", "vertical")):
        dot = sum(a * b for a, b in zip(result[left], result[right]))
        if abs(dot) > 1.0e-6:
            raise ValueError("station axes must be orthogonal")
    cross = [
        result["forward"][1] * result["lateral"][2] - result["forward"][2] * result["lateral"][1],
        result["forward"][2] * result["lateral"][0] - result["forward"][0] * result["lateral"][2],
        result["forward"][0] * result["lateral"][1] - result["forward"][1] * result["lateral"][0],
    ]
    if sum(a * b for a, b in zip(cross, result["vertical"])) < 1.0 - 1.0e-6:
        raise ValueError("station axes must form a right-handed frame")
    return result


def side_prehome_positions(
    home_xyz: Sequence[float], axes: Mapping[str, Sequence[float]], lateral_offsets_m: Sequence[float]
) -> list[dict]:
    """Generate side candidates from HOME FK and the declared station frame."""

    origin = [float(value) for value in home_xyz]
    if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
        raise ValueError("HOME FK xyz must contain three finite values")
    frame = validate_station_axes(axes)
    candidates = []
    for offset in lateral_offsets_m:
        value = float(offset)
        if not math.isfinite(value) or abs(value) < 1.0e-6:
            raise ValueError("side-prehome lateral offsets must be finite and non-zero")
        candidates.append(
            {
                "candidate_id": f"side_{'positive' if value > 0 else 'negative'}_{abs(value):.3f}m",
                "lateral_offset_m": value,
                "tcp_position_world_m": [
                    origin[index] + value * frame["lateral"][index] for index in range(3)
                ],
            }
        )
    return candidates


def path_quality(
    q_path: Sequence[Sequence[float]],
    tcp_xyz: Sequence[Sequence[float]],
    axes: Mapping[str, Sequence[float]],
    *,
    minimum_clearance_m: float | None,
) -> dict:
    """Compute directional TCP excursions and joint travel for one real path."""

    joints = [_seven(values, "trajectory point") for values in q_path]
    tcp = [[float(value) for value in values] for values in tcp_xyz]
    if not joints or len(joints) != len(tcp) or any(
        len(values) != 3 or not all(math.isfinite(value) for value in values) for values in tcp
    ):
        raise ValueError("q_path and tcp_xyz must contain the same non-empty sample count")
    frame = validate_station_axes(axes)
    start = tcp[0]

    def projection(point: Sequence[float], axis: Sequence[float]) -> float:
        return sum((point[index] - start[index]) * axis[index] for index in range(3))

    forward = [projection(point, frame["forward"]) for point in tcp]
    lateral = [projection(point, frame["lateral"]) for point in tcp]
    vertical = [projection(point, frame["vertical"]) for point in tcp]
    travel = [
        sum(abs(joints[row][index] - joints[row - 1][index]) for row in range(1, len(joints)))
        for index in range(7)
    ]
    return {
        "sample_count": len(joints),
        "tcp_xyz_m": tcp,
        "q_rad": joints,
        "minimum_clearance_m": minimum_clearance_m,
        "maximum_backward_excursion_m": max(0.0, -min(forward)),
        "maximum_forward_excursion_m": max(0.0, max(forward)),
        "maximum_lateral_excursion_m": max(abs(value) for value in lateral),
        "maximum_vertical_excursion_m": max(vertical) - min(vertical),
        "final_delta_forward_m": forward[-1],
        "final_delta_lateral_m": lateral[-1],
        "final_delta_vertical_m": vertical[-1],
        "final_displacement_lateral_dominant": abs(lateral[-1]) >= abs(forward[-1])
        and abs(lateral[-1]) >= abs(vertical[-1]),
        "j4_range_rad": [min(values[3] for values in joints), max(values[3] for values in joints)],
        "joint_travel_rad": travel,
        "total_joint_travel_rad": sum(travel),
    }


def quality_gate(metrics: Mapping, gates: Mapping) -> dict:
    checks = {
        "backward_excursion": float(metrics["maximum_backward_excursion_m"])
        <= float(gates["maximum_backward_excursion_m"]),
        "lateral_excursion": float(metrics["maximum_lateral_excursion_m"])
        <= float(gates["maximum_lateral_excursion_m"]),
        "vertical_excursion": float(metrics["maximum_vertical_excursion_m"])
        <= float(gates["maximum_vertical_excursion_m"]),
        "j4": max(abs(float(value)) for value in metrics["j4_range_rad"])
        <= float(gates["maximum_absolute_j4_rad"]),
    }
    clearance = metrics.get("minimum_clearance_m")
    checks["clearance_available"] = clearance is not None
    checks["clearance"] = clearance is not None and float(clearance) >= float(
        gates["minimum_clearance_m"]
    )
    return {"accepted": all(checks.values()), "checks": checks}


def require_two_stage_authorization(stage_a: Mapping, stage_b: Mapping | None) -> None:
    if not stage_a.get("accepted", False):
        raise RuntimeError("SIDE_ROUTE_NOT_FOUND: Stage A failed; Stage B is forbidden")
    if stage_b is None or not stage_b.get("accepted", False):
        raise RuntimeError("SIDE_ROUTE_NOT_FOUND: Stage B failed")


def plan_two_stage(stage_a_planner, measured_state_reader, stage_b_planner) -> dict:
    """Fail-closed sequencing contract used by tests and commissioning adapters."""
    stage_a = stage_a_planner()
    if not stage_a.get("accepted", False):
        raise RuntimeError("SIDE_ROUTE_NOT_FOUND: Stage A failed; Stage B is forbidden")
    measured_start = _seven(measured_state_reader(), "Stage B measured start")
    stage_b = stage_b_planner(measured_start)
    if not stage_b.get("accepted", False):
        raise RuntimeError("SIDE_ROUTE_NOT_FOUND: Stage B failed")
    return {"stage_a": stage_a, "stage_b_start_measured_rad": measured_start, "stage_b": stage_b}


def select_side_candidate(candidates: Sequence[Mapping]) -> Mapping:
    """Collision or quality failure cannot be ranked into an authorized route."""
    admissible = [
        item
        for item in candidates
        if item.get("ik_valid") is True
        and item.get("collision_free") is True
        and item.get("path_quality_computed") is True
        and item.get("path_quality_accepted") is True
    ]
    if not admissible:
        raise RuntimeError("SIDE_ROUTE_NOT_FOUND")
    return min(admissible, key=lambda item: float(item["score"]))


def verify_guard_not_shrunk(configured: Sequence[float], observed: Sequence[float]) -> None:
    expected = [float(value) for value in configured]
    actual = [float(value) for value in observed]
    if len(expected) != 3 or len(actual) != 3 or any(
        value + 1.0e-9 < required for value, required in zip(actual, expected)
    ):
        raise RuntimeError("collision guard was shrunk")


def validate_home_joint_names(names: Sequence[str]) -> None:
    if tuple(names) != RIGHT_ARM_JOINT_NAMES:
        raise ValueError("HOME trajectory must contain exactly J1-J7; J8 is forbidden")


def home_feasibility_decision(low_load_attempts: Sequence[Mapping]) -> str:
    """Never rewrite HOME: repeated low-load measured failures require review."""

    attempts = list(low_load_attempts)
    if attempts and all(
        attempt.get("path_quality_accepted") is True
        and attempt.get("final_measured_acceptance") is False
        for attempt in attempts
    ):
        return "REAL_ACT_HOME_REEVALUATION_REQUIRED"
    return "HOME_UNRESOLVED" if not attempts else "ACT_HOME_RETAINED"
