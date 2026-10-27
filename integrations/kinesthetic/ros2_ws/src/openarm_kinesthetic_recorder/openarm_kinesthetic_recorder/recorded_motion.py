"""Validated file contract and pure helpers for recorded right-arm motion."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .home_measurements import Q_LIFTED_HOME_RIGHT, RIGHT_JOINT_NAMES


SCHEMA_VERSION = 1  # legacy writer/reader compatibility
MARKED_SCHEMA_VERSION = 2
KIND = "openarm_right_recorded_motion"
MARKED_KIND = "openarm_kinesthetic_recording"
GRIPPER_NAME = "openarm_right_finger_joint1"
JOINT_NAMES = [*RIGHT_JOINT_NAMES, GRIPPER_NAME]
MARKER_TYPES = {"keypoint", "gripper_open", "gripper_close"}
GRIPPER_TARGETS = {"gripper_open": -0.35, "gripper_close": -0.10}
COLLECTION_END_HOME_TOLERANCE_RAD = 0.03
# OpenArm v2.0 right-arm limits from the official MoveIt joint_limits.yaml.
# Keep recording and original-rate playback on the same per-joint contract.
COLLECTION_REPLAY_VELOCITY_LIMITS_RAD_S = (
    16.754666, 16.754666, 5.445426, 5.445426,
    20.943946, 20.943946, 20.943946,
)


def is_keyframe_motion(payload: dict) -> bool:
    return payload.get("schema_version") == 1 and payload.get("mode", "continuous") == "keyframe"


def is_marked_motion(payload: dict) -> bool:
    return payload.get("schema_version") == MARKED_SCHEMA_VERSION


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _validate_home_and_names(payload: dict) -> None:
    if payload.get("joint_names") != JOINT_NAMES:
        raise ValueError("recorded motion must use canonical right J1-J7 plus J8")
    home = payload.get("required_start_home_positions_rad")
    if not isinstance(home, list) or len(home) != 7:
        raise ValueError("recorded motion HOME contract is invalid")
    home_values = [_finite_number(value, "HOME position") for value in home]
    if max(abs(a - b) for a, b in zip(home_values, Q_LIFTED_HOME_RIGHT)) > 1e-9:
        raise ValueError("recorded motion does not use the fixed lifted HOME")


def _validate_v1(payload: dict) -> dict:
    if payload.get("kind") != KIND:
        raise ValueError("unsupported recorded-motion schema")
    _validate_home_and_names(payload)
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("recorded motion requires at least two samples")
    previous = -1.0
    keyframe = is_keyframe_motion(payload)
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"recorded sample {index} is not an object")
        elapsed = _finite_number(sample.get("time_from_start_s"), f"sample_index={index} time")
        if elapsed < 0.0 or (index > 0 and elapsed <= previous):
            raise ValueError("recorded sample times must be finite and strictly increasing")
        positions = sample.get("positions_rad")
        if not isinstance(positions, list) or len(positions) != 8:
            raise ValueError(f"recorded sample {index} must contain eight finite positions")
        for value in positions:
            _finite_number(value, f"sample_index={index} position")
        if keyframe:
            if sample.get("waypoint_index") != index:
                raise ValueError("keyframe waypoint_index must match sample order")
            if sample.get("gripper_event") not in ("none", "close", "open"):
                raise ValueError(f"keyframe waypoint {index} has invalid gripper_event")
            if _finite_number(sample.get("timestamp_s"), "timestamp_s") <= 0.0:
                raise ValueError(f"keyframe waypoint {index} has invalid timestamp_s")
        previous = elapsed
    if abs(float(samples[0]["time_from_start_s"])) > 1e-9:
        raise ValueError("first recorded sample must start at zero")
    duration = _finite_number(payload.get("duration_s"), "duration_s")
    if duration <= 0.0 or abs(duration - previous) > 1e-6:
        raise ValueError("recorded duration does not match the last sample")
    if payload.get("sample_count") != len(samples):
        raise ValueError("recorded sample_count does not match samples")
    if keyframe and payload.get("waypoint_count") != len(samples):
        raise ValueError("recorded waypoint_count does not match samples")
    return payload


def _validate_v2(payload: dict) -> dict:
    if payload.get("kind") != MARKED_KIND or payload.get("mode") != "continuous_marked":
        raise ValueError("schema v2 requires continuous_marked recording kind")
    if payload.get("arm") != "right":
        raise ValueError("schema v2 arm must be right")
    _validate_home_and_names(payload)
    period = _finite_number(payload.get("sample_period_target_s"), "sample_period_target_s")
    if period <= 0.0:
        raise ValueError("sample_period_target_s must be positive")
    binding_tolerance = _finite_number(
        payload.get("marker_binding_tolerance_s"), "marker_binding_tolerance_s"
    )
    if binding_tolerance <= 0.0:
        raise ValueError("marker_binding_tolerance_s must be positive")
    if _finite_number(
        payload.get("maximum_allowed_sample_gap_s"),
        "maximum_allowed_sample_gap_s",
    ) <= 0.0:
        raise ValueError("maximum_allowed_sample_gap_s must be positive")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("recorded motion requires at least two samples")
    previous_time = -1.0
    previous_stamp = -1.0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("sample_index") != index:
            raise ValueError(f"sample_index must be continuous at sample_index={index}")
        elapsed = _finite_number(sample.get("time_from_start_s"), f"sample_index={index} time")
        stamp = _finite_number(sample.get("source_timestamp_s"), f"sample_index={index} source_timestamp_s")
        if elapsed < 0.0 or (index and elapsed <= previous_time):
            raise ValueError(f"sample_index={index} time must be strictly increasing")
        if stamp <= 0.0 or (index and stamp <= previous_stamp):
            raise ValueError(f"sample_index={index} source timestamp must be strictly increasing")
        positions = sample.get("positions_rad")
        if not isinstance(positions, list) or len(positions) != len(JOINT_NAMES):
            raise ValueError(f"sample_index={index} positions dimension must be 8")
        for joint, value in zip(JOINT_NAMES, positions):
            _finite_number(value, f"sample_index={index} joint={joint}")
        previous_time, previous_stamp = elapsed, stamp
    if abs(float(samples[0]["time_from_start_s"])) > 1e-9:
        raise ValueError("first recorded sample must start at zero")
    duration = _finite_number(payload.get("duration_s"), "duration_s")
    if duration <= 0.0 or abs(duration - previous_time) > 1e-6:
        raise ValueError("recorded duration does not match the last sample")

    markers = payload.get("markers")
    if not isinstance(markers, list):
        raise ValueError("schema v2 markers must be a list")
    previous_marker_sample = -1
    previous_marker_time = -1.0
    for index, marker in enumerate(markers):
        if not isinstance(marker, dict) or marker.get("marker_index") != index:
            raise ValueError(f"marker_index must be continuous at marker_index={index}")
        marker_type = marker.get("type")
        if marker_type not in MARKER_TYPES:
            raise ValueError(f"marker_index={index} has invalid type")
        sample_index = marker.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or not 0 <= sample_index < len(samples):
            raise ValueError(f"marker_index={index} reason=invalid_sample_reference sample_index={sample_index}")
        marker_time = _finite_number(marker.get("time_from_start_s"), f"marker_index={index} time")
        marker_stamp = _finite_number(marker.get("source_timestamp_s"), f"marker_index={index} source_timestamp_s")
        binding_error = _finite_number(marker.get("binding_error_s"), f"marker_index={index} binding_error_s")
        if marker_stamp <= 0.0 or binding_error < 0.0:
            raise ValueError(f"marker_index={index} has invalid timestamp/binding error")
        if binding_error > binding_tolerance:
            raise ValueError(
                f"marker_index={index} binding error measured={binding_error:.6f} "
                f"limit={binding_tolerance:.6f}"
            )
        if sample_index < previous_marker_sample or marker_time < previous_marker_time:
            raise ValueError(f"marker_index={index} marker order is reversed")
        expected_error = abs(marker_time - float(samples[sample_index]["time_from_start_s"]))
        if abs(binding_error - expected_error) > 1e-6:
            raise ValueError(f"marker_index={index} binding_error_s is inconsistent")
        if not isinstance(marker.get("source_button"), str) or not marker["source_button"]:
            raise ValueError(f"marker_index={index} source_button is required")
        if marker_type.startswith("gripper_"):
            target = _finite_number(marker.get("target_position_rad"), f"marker_index={index} target")
            if abs(target - GRIPPER_TARGETS[marker_type]) > 1e-6:
                raise ValueError(f"marker_index={index} has invalid gripper target")
            if marker.get("result") not in {
                "semantic_only", "requested", "accepted", "reached"
            }:
                raise ValueError(f"marker_index={index} has invalid gripper result")
        previous_marker_sample, previous_marker_time = sample_index, marker_time

    quality = payload.get("quality")
    if not isinstance(quality, dict) or quality.get("accepted_sample_count") != len(samples):
        raise ValueError("quality.accepted_sample_count does not match samples")
    for key in (
        "rejected_incomplete_count", "rejected_nonfinite_count",
        "rejected_invalid_timestamp_count", "timestamp_regression_count",
        "estimated_dropped_sample_count",
    ):
        value = quality.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"quality.{key} must be a nonnegative integer")
    maximum_gap = _finite_number(quality.get("maximum_sample_gap_s"), "quality.maximum_sample_gap_s")
    actual_gap = max(float(b["time_from_start_s"]) - float(a["time_from_start_s"]) for a, b in zip(samples, samples[1:]))
    if maximum_gap < 0.0 or abs(maximum_gap - actual_gap) > 1e-6:
        raise ValueError("quality.maximum_sample_gap_s does not match samples")
    return payload


def validate_recorded_motion(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("recorded motion must be a JSON object")
    version = payload.get("schema_version")
    if version == SCHEMA_VERSION:
        return _validate_v1(payload)
    if version == MARKED_SCHEMA_VERSION:
        return _validate_v2(payload)
    raise ValueError(f"unsupported recorded-motion schema version: {version!r}")


def marker_summary(payload: dict) -> list[dict]:
    """Return the stable marker fields stored with an episode provenance record."""
    if not is_marked_motion(payload):
        return []
    return [
        {
            key: marker[key]
            for key in (
                "marker_index",
                "type",
                "sample_index",
                "source_button",
                "result",
                "target_position_rad",
            )
            if key in marker
        }
        for marker in payload["markers"]
    ]


def validate_collection_trajectory(
    payload: object,
    *,
    maximum_end_home_error_rad: float = COLLECTION_END_HOME_TOLERANCE_RAD,
) -> dict:
    """Validate the stricter two-marker trajectory used for real ACT capture."""
    motion = validate_recorded_motion(payload)
    if not is_marked_motion(motion):
        raise ValueError("ACT collection requires a continuous_marked trajectory")
    if (
        not math.isfinite(maximum_end_home_error_rad)
        or maximum_end_home_error_rad <= 0.0
    ):
        raise ValueError("end HOME tolerance must be finite and positive")

    markers = motion["markers"]
    expected = (
        ("gripper_close", "Y"),
        ("gripper_open", "A"),
    )
    actual = [(item.get("type"), item.get("source_button")) for item in markers]
    if actual != list(expected):
        raise ValueError(
            "ACT collection markers must be exactly "
            "Y=gripper_close then A=gripper_open"
        )
    if any(item.get("result") != "semantic_only" for item in markers):
        raise ValueError("ACT collection gripper markers must be semantic_only")

    terminal = motion["samples"][-1]["positions_rad"][:7]
    errors = [
        abs(float(actual_value) - float(target))
        for actual_value, target in zip(terminal, Q_LIFTED_HOME_RIGHT)
    ]
    maximum_error = max(errors)
    if maximum_error > maximum_end_home_error_rad:
        joint_index = errors.index(maximum_error)
        raise ValueError(
            "ACT collection trajectory must end at fixed lifted HOME: "
            f"joint=J{joint_index + 1} actual={float(terminal[joint_index]):.6f} "
            f"target={float(Q_LIFTED_HOME_RIGHT[joint_index]):.6f} "
            f"error={maximum_error:.6f} "
            f"limit={maximum_end_home_error_rad:.6f} rad"
        )
    path_audit = audit_recorded_path(
        motion["samples"],
        maximum_gap_s=float(motion["maximum_allowed_sample_gap_s"]),
        maximum_step_rad=0.20,
    )
    for joint_index, (velocity, limit) in enumerate(zip(
        path_audit.maximum_velocity_rad_s, COLLECTION_REPLAY_VELOCITY_LIMITS_RAD_S
    )):
        if velocity <= limit:
            continue
        raise ValueError(
            "ACT collection trajectory exceeds original-rate replay velocity: "
            f"joint=J{joint_index + 1} measured={velocity:.6f} "
            f"limit={limit:.6f} rad/s"
        )
    return {
        "marker_count": len(markers),
        "markers": marker_summary(motion),
        "end_home_max_error_rad": maximum_error,
        "end_home_joint_errors_rad": errors,
        "maximum_recorded_velocity_rad_s": list(path_audit.maximum_velocity_rad_s),
        "maximum_sample_gap_s": path_audit.maximum_sample_gap_s,
        "maximum_joint_step_rad": path_audit.maximum_joint_step_rad,
    }


@dataclass(frozen=True)
class PathDynamics:
    time_scale: float
    duration_s: float
    maximum_velocity_rad_s: tuple[float, ...]
    maximum_acceleration_rad_s2: tuple[float, ...]
    maximum_sample_gap_s: float
    maximum_joint_step_rad: float


@dataclass(frozen=True)
class RecordedPathAudit:
    duration_s: float
    maximum_velocity_rad_s: tuple[float, ...]
    maximum_acceleration_rad_s2: tuple[float, ...]
    maximum_sample_gap_s: float
    maximum_joint_step_rad: float


def audit_recorded_path(
    samples: Sequence[dict], *, maximum_gap_s: float = 0.50,
    maximum_step_rad: float = 0.20,
) -> RecordedPathAudit:
    """Audit a dense path without changing its recorded timing."""
    if not samples:
        raise ValueError("dense path requires at least one sample")
    if maximum_gap_s <= 0.0 or maximum_step_rad <= 0.0:
        raise ValueError("path gap and step limits must be positive")
    maximum_velocities = [0.0] * 7
    maximum_accelerations = [0.0] * 7
    segment_velocities: list[list[float]] = []
    maximum_gap = 0.0
    maximum_step = 0.0
    for local_index, (previous, current) in enumerate(
        zip(samples, samples[1:]), start=1
    ):
        sample_index = int(current.get("sample_index", local_index))
        dt = float(current["time_from_start_s"]) - float(
            previous["time_from_start_s"]
        )
        if dt <= 0.0:
            raise ValueError(
                f"sample_index={sample_index} reason=non_increasing_time dt={dt:.9f}"
            )
        if dt > maximum_gap_s:
            raise ValueError(
                f"sample_index={sample_index} reason=sample_gap "
                f"measured={dt:.6f} limit={maximum_gap_s:.6f}"
            )
        q0, q1 = previous["positions_rad"], current["positions_rad"]
        velocities = []
        for joint_index, (a, b) in enumerate(zip(q0[:7], q1[:7])):
            step = abs(float(b) - float(a))
            if step > maximum_step_rad:
                raise ValueError(
                    f"sample_index={sample_index} joint={JOINT_NAMES[joint_index]} "
                    f"reason=position_step measured={step:.6f} "
                    f"limit={maximum_step_rad:.6f}"
                )
            velocity = (float(b) - float(a)) / dt
            velocities.append(velocity)
            maximum_velocities[joint_index] = max(
                maximum_velocities[joint_index], abs(velocity)
            )
            maximum_step = max(maximum_step, step)
        segment_velocities.append(velocities)
        maximum_gap = max(maximum_gap, dt)
    for segment_index in range(1, len(segment_velocities)):
        t0 = float(samples[segment_index - 1]["time_from_start_s"])
        t2 = float(samples[segment_index + 1]["time_from_start_s"])
        dt = 0.5 * (t2 - t0)
        for joint_index in range(7):
            acceleration = abs(
                segment_velocities[segment_index][joint_index]
                - segment_velocities[segment_index - 1][joint_index]
            ) / dt
            maximum_accelerations[joint_index] = max(
                maximum_accelerations[joint_index], acceleration
            )
    duration = float(samples[-1]["time_from_start_s"]) - float(
        samples[0]["time_from_start_s"]
    )
    return RecordedPathAudit(
        duration,
        tuple(maximum_velocities),
        tuple(maximum_accelerations),
        maximum_gap,
        maximum_step,
    )


def analyze_dense_path(
    samples: Sequence[dict], *, maximum_gap_s: float = 0.50,
    maximum_step_rad: float = 0.20, velocity_limits_rad_s: Sequence[float] | None = None,
    acceleration_limits_rad_s2: Sequence[float] | None = None,
    velocity_scaling: float = 1.0, acceleration_scaling: float = 1.0,
) -> PathDynamics:
    """Audit a measured path and return a conservative global time stretch."""
    if len(samples) < 2:
        raise ValueError("dense path requires at least two samples")
    if not 0.0 < velocity_scaling <= 1.0 or not 0.0 < acceleration_scaling <= 1.0:
        raise ValueError("path scaling factors must be in (0, 1]")
    velocity_limits = list(velocity_limits_rad_s or [0.80] * 7)
    acceleration_limits = list(acceleration_limits_rad_s2 or [0.50] * 7)
    if len(velocity_limits) != 7 or len(acceleration_limits) != 7 or any(v <= 0.0 for v in velocity_limits + acceleration_limits):
        raise ValueError("seven positive arm velocity and acceleration limits are required")
    audit = audit_recorded_path(
        samples,
        maximum_gap_s=maximum_gap_s,
        maximum_step_rad=maximum_step_rad,
    )
    scale = 1.0
    for joint_index in range(7):
        scale = max(
            scale,
            audit.maximum_velocity_rad_s[joint_index]
            / (velocity_limits[joint_index] * velocity_scaling),
        )
        scale = max(
            scale,
            math.sqrt(
                audit.maximum_acceleration_rad_s2[joint_index]
                / (acceleration_limits[joint_index] * acceleration_scaling)
            ),
        )
    return PathDynamics(
        scale,
        scale * audit.duration_s,
        audit.maximum_velocity_rad_s,
        audit.maximum_acceleration_rad_s2,
        audit.maximum_sample_gap_s,
        audit.maximum_joint_step_rad,
    )


def playback_time_scale(samples: list[dict], maximum_arm_velocity_rad_s: float = 0.80, maximum_gripper_velocity_rad_s: float = 0.80) -> float:
    """Legacy v1 continuous global stretch, including its J8 sample stream."""
    if maximum_arm_velocity_rad_s <= 0.0 or maximum_gripper_velocity_rad_s <= 0.0:
        raise ValueError("replay velocity limits must be positive")
    scale = analyze_dense_path(samples, maximum_gap_s=float("inf"), maximum_step_rad=float("inf"), velocity_limits_rad_s=[maximum_arm_velocity_rad_s] * 7, acceleration_limits_rad_s2=[float("inf")] * 7).time_scale
    for previous, current in zip(samples, samples[1:]):
        dt = float(current["time_from_start_s"]) - float(previous["time_from_start_s"])
        scale = max(scale, abs(float(current["positions_rad"][7]) - float(previous["positions_rad"][7])) / dt / maximum_gripper_velocity_rad_s)
    return scale


def marker_segment_ranges(payload: dict, start_marker: int | None = None, stop_after_marker: int | None = None) -> list[dict]:
    """Return inclusive dense sample ranges; no marker is converted into an OMPL leg."""
    markers = payload["markers"]
    if start_marker is not None and not 0 <= start_marker < len(markers):
        raise ValueError("start_marker is outside the marker range")
    if stop_after_marker is not None and not 0 <= stop_after_marker < len(markers):
        raise ValueError("stop_after_marker is outside the marker range")
    if start_marker is not None and stop_after_marker is not None and stop_after_marker < start_marker:
        raise ValueError("stop_after_marker precedes start_marker")
    start_sample = markers[start_marker]["sample_index"] if start_marker is not None else 0
    selected = [
        m for m in markers
        if (start_marker is None or m["marker_index"] >= start_marker)
        and m["sample_index"] >= start_sample
        and (stop_after_marker is None or m["marker_index"] <= stop_after_marker)
    ]
    ranges = []
    cursor = start_sample
    for marker in selected:
        end = int(marker["sample_index"])
        ranges.append({"start_sample_index": cursor, "end_sample_index": end, "markers": [marker]})
        cursor = end
    final_sample = int(markers[stop_after_marker]["sample_index"]) if stop_after_marker is not None else len(payload["samples"]) - 1
    if cursor < final_sample or not ranges:
        ranges.append({"start_sample_index": cursor, "end_sample_index": final_sample, "markers": []})
    return ranges


def load_recorded_motion(path: Path) -> dict:
    return validate_recorded_motion(json.loads(path.read_text(encoding="utf-8")))


def interpolated_arm_states(samples: list[dict], maximum_joint_step_rad: float = 0.02) -> Iterable[tuple[list[float], float]]:
    if maximum_joint_step_rad <= 0.0:
        raise ValueError("maximum_joint_step_rad must be positive")
    yield [float(value) for value in samples[0]["positions_rad"]], 0.0
    for previous, current in zip(samples, samples[1:]):
        q0 = [float(value) for value in previous["positions_rad"]]
        q1 = [float(value) for value in current["positions_rad"]]
        divisions = max(1, math.ceil(max(abs(a - b) for a, b in zip(q0[:7], q1[:7])) / maximum_joint_step_rad))
        t0, t1 = float(previous["time_from_start_s"]), float(current["time_from_start_s"])
        for step in range(1, divisions + 1):
            fraction = step / divisions
            yield [a + fraction * (b - a) for a, b in zip(q0, q1)], t0 + fraction * (t1 - t0)
