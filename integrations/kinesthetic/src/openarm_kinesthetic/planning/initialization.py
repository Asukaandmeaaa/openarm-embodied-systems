"""Compare direct, explicit-waypoint, and collision-aware initialization paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from ..config import load_yaml
from ..paths import workspace_path
from ..scene.model import load_model, reset_task_ready
from .kinematics import check_joint_path, contact_failures, joint_limits


@dataclass
class PathReport:
    name: str
    success: bool
    failure_stage: str | None
    collision_failures: list[str]
    minimum_tcp_table_clearance_m: float
    minimum_joint_margin_normalized: float
    joint_path_length_rad: float
    estimated_max_velocity_rad_s: float
    estimated_max_acceleration_rad_s2: float
    waypoints: list[list[float]]


def _evaluate(model, data, mapping, name: str, nodes: list[np.ndarray]) -> PathReport:
    failures: set[str] = set()
    failure_stage = None
    minimum_clearance = float("inf"); minimum_margin = float("inf"); length = 0.0
    for index, (start, goal) in enumerate(zip(nodes, nodes[1:])):
        segment_failures, clearance, margin = check_joint_path(model, data, mapping, start, goal, step_rad=0.01)
        length += float(np.linalg.norm(goal - start))
        minimum_clearance = min(minimum_clearance, clearance)
        minimum_margin = min(minimum_margin, margin)
        if segment_failures and failure_stage is None:
            failure_stage = f"segment_{index}"
        failures.update(segment_failures)
    # Conservative time scaling for a quintic zero-velocity waypoint blend.
    largest_step = max(float(np.max(np.abs(b - a))) for a, b in zip(nodes, nodes[1:]))
    duration = max(1.0, largest_step / 0.45 * 1.875)
    max_velocity = 1.875 * largest_step / duration
    max_acceleration = 5.8 * largest_step / (duration * duration)
    return PathReport(
        name=name, success=not failures, failure_stage=failure_stage,
        collision_failures=sorted(failures), minimum_tcp_table_clearance_m=minimum_clearance,
        minimum_joint_margin_normalized=minimum_margin, joint_path_length_rad=length,
        estimated_max_velocity_rad_s=max_velocity, estimated_max_acceleration_rad_s2=max_acceleration,
        waypoints=[node.tolist() for node in nodes],
    )


def plan_initialization() -> dict:
    import mujoco

    model = load_model(); data = mujoco.MjData(model); mapping = reset_task_ready(model, data)
    poses = load_yaml("poses.yaml"); scan = load_yaml("layout_scan.yaml")
    motor_zero = np.asarray(poses["motor_zero"]["arm"], dtype=float)
    # The real backend must hold measured current. This collision-free simulation
    # current is the original pedestal home candidate, not mechanical zero.
    safe_current = np.asarray(poses["right_task_ready"]["initial_candidate_arm"], dtype=float)
    ready = np.asarray(poses["right_task_ready"]["candidate_arm"], dtype=float)
    explicit = np.asarray(scan["safe_waypoint_candidates"][0], dtype=float)

    reports = [
        _evaluate(model, data, mapping, "direct_joint_interpolation", [safe_current, ready]),
        _evaluate(model, data, mapping, "explicit_raise_and_fold_waypoint", [safe_current, explicit, ready]),
    ]

    lower, upper = joint_limits(model, mapping, 0.02)
    rng = np.random.default_rng(int(scan["seed"]) + 17)
    planned = None
    for attempt in range(1000):
        alpha = rng.random()
        center = safe_current * (1.0 - alpha) + ready * alpha
        candidate = np.clip(center + rng.normal(0.0, [0.4, 0.5, 0.5, 0.4, 0.5, 0.3, 0.5]), lower, upper)
        data.qpos[mapping.right_qpos] = candidate; mujoco.mj_forward(model, data)
        if contact_failures(model, data):
            continue
        report = _evaluate(model, data, mapping, "collision_aware_sampled_waypoint", [safe_current, candidate, ready])
        if report.success:
            planned = report
            break
    if planned is None:
        planned = PathReport("collision_aware_sampled_waypoint", False, "search", ["no_path"], 0, 0, 0, 0, 0, [])
    reports.append(planned)

    zero_report = _evaluate(model, data, mapping, "motor_zero_negative_control", [motor_zero, ready])
    successes = [report for report in reports if report.success]
    selected = min(successes, key=lambda report: report.joint_path_length_rad) if successes else None
    result = {
        "schema_version": 1, "gate": "B", "seed": int(scan["seed"]) + 17,
        "start_policy": "hold_current_then_plan", "safe_current_simulation_only": safe_current.tolist(),
        "motor_zero_negative_control": asdict(zero_report),
        "candidates": [asdict(report) for report in reports],
        "selected": asdict(selected) if selected else None,
        "status": "PASS" if selected else "FAIL",
        "note": "motor_zero is not assumed collision-free; real startup must hold measured current",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    result = plan_initialization()
    output = workspace_path("results", "layout", "initialization_paths.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
