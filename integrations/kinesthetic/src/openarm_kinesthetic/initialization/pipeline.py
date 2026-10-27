"""Orchestration of the simulation initialization chain, plus the offset scan.

    table geometry -> teaching-ready target -> collision-aware IK -> planner
                                                         -> simulation execution

The scan exists because the teaching-ready offsets are a *choice*, and the prompt is
explicit that the recommended table-centre value must be tuned against MuJoCo
evidence rather than trusted. On this station the table centre turns out to be
unreachable at the project's validated TCP orientation, so the offsets are selected
by scan and recorded with `simulation_scan_selected` provenance.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..config import load_yaml
from ..paths import workspace_path
from ..planning.kinematics import normalized_joint_margin, solve_site_pose
from ..scene.model import ModelMap, load_model, reset_task_ready
from .collision import configuration_valid
from .execute import execute_initialization_trajectory
from .ik import solve_teaching_ready_ik
from .planner import plan_initialization_path
from .states import InitializationStateMachine
from .table_geometry import sim_table_geometry, table_geometry_summary
from .teaching_ready import (
    TeachingReadyTarget,
    reference_orientation,
    rotation_error_rad,
    teaching_ready_provenance,
    teaching_ready_target,
)

SCENE_READY = "SIM_SCENE_READY"


def load_initialization_config() -> dict[str, Any]:
    return load_yaml("initialization.yaml")


def reference_pose_from_poses(config: Mapping[str, Any] | None = None) -> np.ndarray:
    """The episode Home joint values, used only as an orientation / IK-seed reference."""
    poses = load_yaml("poses.yaml")
    return np.asarray(poses["right_task_ready"]["candidate_arm"], dtype=float)


def simulation_start_q(config: Mapping[str, Any]) -> np.ndarray:
    """Configured MuJoCo current state; real hardware always uses /joint_states."""
    poses = load_yaml("poses.yaml")
    pose_name = config.get("simulation", {}).get("initial_pose")
    if pose_name == "motor_zero":
        return np.asarray(poses["motor_zero"]["arm"], dtype=float)
    if pose_name == "right_task_ready_initial_candidate":
        return np.asarray(poses["right_task_ready"]["initial_candidate_arm"], dtype=float)
    raise ValueError(f"unsupported simulation.initial_pose: {pose_name!r}")


def build_scene(config: Mapping[str, Any]):
    """Compile the scene and install the configured current simulation state."""
    import mujoco

    model = load_model(str(config["initialization"]["scene"]))
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    start = simulation_start_q(config)
    data.qpos[mapping.right_qpos] = start
    data.qvel[:] = 0.0
    data.ctrl[mapping.right_actuators] = start
    mujoco.mj_forward(model, data)
    return model, data, mapping


def current_start_q(mapping: ModelMap, data) -> np.ndarray:
    """The planner start state: whatever the right arm is doing right now."""
    return np.asarray(data.qpos[mapping.right_qpos], dtype=float).copy()


def _ik_seeds(reference_q: np.ndarray) -> list[np.ndarray]:
    """Deterministic IK seed list, reference pose first."""
    return [
        np.asarray(reference_q, dtype=float),
        np.asarray([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=float),
        np.asarray([0.44, 0.05, -0.46, 1.17, -0.35, 0.04, 0.32], dtype=float),
        np.asarray([0.70, 0.13, -0.68, 0.96, 0.18, 0.23, 0.37], dtype=float),
        np.asarray([0.22722054, 1.087121, 0.41459577, 2.34218107, -0.24927559, -0.1086842, -0.56711975], dtype=float),
    ]


def scan_teaching_ready_offsets(
    model,
    data,
    mapping,
    geometry,
    config: Mapping[str, Any],
    reference_q: np.ndarray,
    *,
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reproducible reachability scan over table-relative teaching-ready offsets.

    Selection rule (recorded in the config as `teaching_ready.selection_rule`):
    keep offsets at least `minimum_footprint_margin_m` inside the table footprint,
    then maximize the normalized joint-limit margin. This avoids the near-edge
    corner solutions that have a large margin but sit almost off the table.
    """
    rule = config.get("teaching_ready", {}).get("selection_rule", {})
    minimum_footprint_margin = float(rule.get("minimum_footprint_margin_m", 0.05))
    ik_block = config.get("teaching_ready", {}).get("ik", {})
    orientation = reference_orientation(model, data, mapping, reference_q)
    height = float(config["teaching_ready"]["height_above_table_m"]["value"])
    half = geometry.half_extent

    if x_values is None:
        limit = max(0.0, float(half[0]) - minimum_footprint_margin)
        x_values = np.round(np.arange(-limit, limit + 1e-9, 0.02), 6)
    if y_values is None:
        limit = max(0.0, float(half[1]) - minimum_footprint_margin)
        y_values = np.round(np.arange(-limit, limit + 1e-9, 0.02), 6)

    candidates: list[dict[str, Any]] = []
    for offset_x in x_values:
        for offset_y in y_values:
            goal = np.asarray(
                [geometry.center[0] + offset_x, geometry.center[1] + offset_y, geometry.top_z + height],
                dtype=float,
            )
            record: dict[str, Any] = {
                "offset_x_m": float(offset_x),
                "offset_y_m": float(offset_y),
                "target_world_m": goal.tolist(),
                "feasible": False,
                "failure": "TEACH_READY_IK_FAILED",
            }
            best: tuple[float, np.ndarray] | None = None
            for seed in _ik_seeds(reference_q):
                ik = solve_site_pose(
                    model,
                    data,
                    mapping,
                    goal,
                    orientation,
                    seed,
                    tolerance_m=float(ik_block.get("position_tolerance_m", 0.004)),
                    orientation_tolerance_rad=float(ik_block.get("orientation_tolerance_rad", 0.04)),
                    iterations=int(ik_block.get("iterations", 300)),
                    margin_rad=float(ik_block.get("joint_margin_rad", 0.02)),
                )
                if not ik.success:
                    continue
                import mujoco

                data.qpos[mapping.right_qpos] = ik.q
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                position_error = float(np.linalg.norm(goal - data.site_xpos[mapping.grasp_site]))
                orientation_error = rotation_error_rad(
                    np.asarray(data.site_xmat[mapping.grasp_site], dtype=float).reshape(3, 3), orientation
                )
                if position_error > float(ik_block.get("position_tolerance_m", 0.004)):
                    continue
                if orientation_error > float(ik_block.get("orientation_tolerance_rad", 0.04)):
                    continue
                ok, found = configuration_valid(
                    model,
                    data,
                    mapping,
                    ik.q,
                    margin_rad=float(ik_block.get("joint_margin_rad", 0.02)),
                    forbid_cube_contact=bool(config.get("planning", {}).get("forbid_cube_contact", True)),
                )
                if not ok:
                    record["failure"] = "|".join(sorted(found)) or "collision"
                    continue
                margin = normalized_joint_margin(model, mapping, ik.q)
                if best is None or margin > best[0]:
                    best = (margin, ik.q.copy())
            if best is not None:
                record.update(
                    feasible=True,
                    failure=None,
                    normalized_joint_margin=float(best[0]),
                    q_candidate=best[1].tolist(),
                )
            candidates.append(record)

    feasible = [item for item in candidates if item["feasible"]]
    selected = max(feasible, key=lambda item: item["normalized_joint_margin"]) if feasible else None
    return {
        "schema_version": 1,
        "table": table_geometry_summary(geometry),
        "height_above_table_m": height,
        "selection_rule": {
            "minimum_footprint_margin_m": minimum_footprint_margin,
            "objective": str(rule.get("objective", "maximize_normalized_joint_limit_margin")),
        },
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "candidates": candidates,
        "selected": selected,
        "status": "PASS" if selected else "FAIL",
    }


def run_initialization(
    *,
    execute: bool = True,
    start_q: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
    model=None,
    data=None,
    mapping=None,
) -> dict[str, Any]:
    """Run the whole simulation chain and return a report dictionary."""
    config = dict(config) if config is not None else load_initialization_config()
    machine = InitializationStateMachine()
    machine.advance(SCENE_READY)
    if model is None:
        model, data, mapping = build_scene(config)
    geometry = sim_table_geometry(model, data, config)
    machine.advance("SIM_TABLE_READY")

    # Capture before FK/IK use scratch configurations. The planner start must be
    # the state MuJoCo was actually in when initialization began.
    start = current_start_q(mapping, data) if start_q is None else np.asarray(start_q, dtype=float).copy()

    reference_q = reference_pose_from_poses(config)
    machine.advance("SIM_COMPUTE_TEACH_READY")
    target: TeachingReadyTarget = teaching_ready_target(model, data, mapping, geometry, config, reference_q)

    # IK is seeded from the reference pose (as the real acceptance path does), while
    # the planner start state is the measured/current configuration. Those two roles
    # are deliberately not the same thing.
    ik = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference_q), config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "chain": "simulation_initialization_teaching_ready",
        "table": table_geometry_summary(geometry),
        "teaching_ready": {
            **target.as_dict(),
            "provenance": teaching_ready_provenance(config),
        },
        "start_q": start.tolist(),
        "start_state_source": "current_joint_state",
        "ik": ik.as_dict(),
    }
    if not ik.success:
        machine.fail("SIM_TEACH_READY_IK_FAILED")
        report.update(success=False, state=machine.as_dict(), planner=None, execution=None)
        return report
    machine.advance("SIM_IK_READY")

    plan = plan_initialization_path(model, data, mapping, start, ik.q, config, geometry=geometry)
    report["planner"] = plan.as_dict()
    if not plan.success:
        machine.fail("SIM_INITIALIZATION_PLAN_FAILED")
        report.update(success=False, state=machine.as_dict(), execution=None)
        return report
    machine.advance("SIM_PLAN_READY")

    if not execute:
        machine.advance("SIM_EXECUTING_INIT")
        machine.advance("SIM_INITIALIZATION_REACHED")
        machine.advance("SIM_MANUAL_TEACHING")
        report.update(success=True, state=machine.as_dict(), execution=None)
        return report

    # IK and collision checks evaluate scratch configurations in `data`. Restore
    # the captured current state before dynamic playback; otherwise execution
    # would start at the IK goal and command the first waypoint backwards.
    import mujoco

    data.qpos[mapping.right_qpos] = start
    data.qvel[:] = 0.0
    data.ctrl[mapping.right_actuators] = start
    mujoco.mj_forward(model, data)
    machine.advance("SIM_EXECUTING_INIT")
    execution = execute_initialization_trajectory(
        model, data, mapping, plan.waypoints, ik.q, target, geometry, config
    )
    report["execution"] = execution.as_dict()
    if not execution.success:
        machine.fail("SIM_INITIALIZATION_EXECUTION_FAILED")
        report.update(success=False, state=machine.as_dict())
        return report
    machine.advance("SIM_INITIALIZATION_REACHED")
    machine.advance("SIM_MANUAL_TEACHING")
    report.update(success=True, state=machine.as_dict())
    return report


def write_report(report: Mapping[str, Any], name: str = "initialization_sim.json"):
    import json

    output = workspace_path("results", "initialization", name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="skip simulation execution")
    parser.add_argument("--scan", action="store_true", help="also write the teaching-ready offset scan")
    args = parser.parse_args()
    config = load_initialization_config()
    model, data, mapping = build_scene(config)
    if args.scan:
        geometry = sim_table_geometry(model, data, config)
        scan = scan_teaching_ready_offsets(model, data, mapping, geometry, config, reference_pose_from_poses(config))
        write_report(scan, "teaching_ready_offset_scan.json")
        print(json.dumps({"scan_status": scan["status"], "selected": scan["selected"]}, indent=2))
    report = run_initialization(execute=not args.plan_only, config=config, model=model, data=data, mapping=mapping)
    write_report(report)
    print(json.dumps({"success": report["success"], "state": report["state"]}, indent=2))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
