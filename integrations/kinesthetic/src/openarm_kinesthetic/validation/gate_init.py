"""Simulation initialization acceptance gate: SIM_INITIALIZATION_ACCEPTANCE.

Prompt section 19 requires an explicit gate that must pass in full before any real
hardware motion is attempted. The tests below are the A1-A19 list from section 18,
plus the handoff regression and the dataset-isolation proof.

Hardware is never touched: everything runs against the compiled MuJoCo scene, the
offline gripper backend and a temporary recorder root.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from ..config import load_yaml
from ..control.gripper import GripperController, GripperState
from ..paths import workspace_path
from ..planning.kinematics import joint_limits
from ..recording.transaction import EpisodeRecorder
from ..scene.model import load_model, reset_task_ready
from ..teaching.gamepad import GamepadMapper
from ..teaching.virtual_handle import VirtualHandle
from ..initialization.execute import execute_initialization_trajectory
from ..initialization.ik import IK_FAILED, _candidate_offsets, solve_teaching_ready_ik
from ..initialization.pipeline import (
    _ik_seeds,
    build_scene,
    current_start_q,
    load_initialization_config,
    reference_pose_from_poses,
    run_initialization,
)
from ..initialization.planner import plan_initialization_path
from ..initialization.table_geometry import derive_table_center_x, robot_front_reference_x, sim_table_geometry
from ..initialization.teaching_ready import teaching_ready_target
from .. import initialization as initialization_package

PASS_STATEMENT = "SIM_INITIALIZATION_ACCEPTANCE = PASS"

# Where the initialization chain lives, and what would prove it can record an episode.
INITIALIZATION_PACKAGE_DIRECTORY = Path(initialization_package.__file__).resolve().parent
RECORDING_MARKERS = ("EpisodeRecorder", "recording.transaction")


def recording_capable_module_names(directory: Path) -> list[str]:
    """Initialization modules that mention the recorder, if any.

    A14 claims the initialization chain never records an episode. It cannot be proved by
    watching a catalog: `run_initialization` is never handed a recorder, so the catalog is
    empty whether or not the chain *could* write one, and the old check reduced to
    ``before_run["success"]``. Reading the package source is the falsifiable version, and it
    is the technique the plan-only execution test already uses to prove that module has no
    trajectory action client. Scanning the whole package rather than a list of names means a
    new module is covered the moment it is added.
    """
    offenders: list[str] = []
    root = Path(directory)
    for module_path in sorted(root.rglob("*.py")):
        text = module_path.read_text(encoding="utf-8")
        if any(marker in text for marker in RECORDING_MARKERS):
            offenders.append(module_path.relative_to(root).as_posix())
    return offenders


def _scene_table_geometry(model, data) -> tuple[np.ndarray, np.ndarray]:
    """Table centre XY and half-extents read back from the compiled scene."""
    import mujoco

    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    if geom < 0:
        raise KeyError("the scene has no table_top_collision geom")
    size = np.asarray(model.geom_size[geom], dtype=float)
    pos = np.asarray(data.geom_xpos[geom], dtype=float)
    return pos[:2], size[:2]


def validate_gate_init() -> dict:
    tests: dict[str, bool] = {}
    details: dict[str, object] = {}
    config = load_initialization_config()
    model, data, mapping = build_scene(config)
    geometry = sim_table_geometry(model, data, config)
    reference_q = reference_pose_from_poses(config)
    scene_center_xy, scene_half = _scene_table_geometry(model, data)

    # --- A1: the 0.40 x 0.60 table is loaded with the prior orientation ---------
    tests["A1_table_geometry_loaded"] = bool(
        np.allclose(geometry.size, [0.40, 0.60], atol=1e-9)
        and np.allclose(scene_half, [0.20, 0.30], atol=1e-9)
        and np.allclose(scene_center_xy, geometry.center[:2], atol=1e-6)
    )
    details["A1"] = {
        "prior_size_xy_m": geometry.size.tolist(),
        "scene_half_extent_m": scene_half.tolist(),
        "scene_center_xy_m": scene_center_xy.tolist(),
        "prior_center_xy_m": geometry.center[:2].tolist(),
    }

    # --- A2: the world X is derived, and matches the compiled scene -------------
    front_x = robot_front_reference_x(model, data, "openarm_body_link0_collision")
    gap = float(config["table"]["near_edge_gap_from_robot_front_m"]["value"])
    expected_center_x = derive_table_center_x(front_x, gap, 0.40)
    tests["A2_table_position_derived"] = bool(
        abs(geometry.center[0] - expected_center_x) < 1e-9
        and abs(scene_center_xy[0] - expected_center_x) < 2e-3
        and geometry.provenance["center.x"] == "derived_from_prior"
        and geometry.provenance["robot_front_reference.x"] == "model_derived"
    )
    details["A2"] = {
        "robot_front_reference_x_m": front_x,
        "near_edge_gap_m": gap,
        "table_near_edge_x_m": front_x + gap,
        "derived_center_x_m": expected_center_x,
        "historical_configured_center_x_m": 0.43,
        "delta_from_historical_m": expected_center_x - 0.43,
    }

    # --- A3/A4: teaching-ready is above the table and inside the footprint ------
    target = teaching_ready_target(model, data, mapping, geometry, config, reference_q)
    tests["A3_teaching_ready_above_table"] = bool(
        abs((target.position_world_m[2] - geometry.top_z) - 0.28) < 1e-9
        and geometry.contains_xy(target.position_world_m[:2])
        and target.orientation_source == "fk_reference"
    )
    tests["A4_teaching_ready_does_not_penetrate_table"] = bool(target.position_world_m[2] > geometry.top_z + 0.25)
    details["A3"] = target.as_dict()

    # --- A5: collision-aware IK succeeds ---------------------------------------
    ik = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference_q), config)
    strict_ik = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference_q), config)
    tests["A5_collision_aware_ik_succeeds"] = bool(strict_ik.success)
    details["A5"] = strict_ik.as_dict()
    tests["A5_uses_configured_offset_first"] = bool(ik.success and not ik.used_fallback and ik.seed_index == 0)

    # --- A6: an out-of-reach nominal target engages the bounded fallback --------
    fallback_config = copy.deepcopy(config)
    fallback_config["teaching_ready"]["offset_x_m"] = {"value": -0.02, "source": "configured_prior"}
    fallback_config["teaching_ready"]["offset_y_m"] = {"value": -0.16, "source": "configured_prior"}
    fallback_config["teaching_ready"]["fallback_search"]["maximum_candidates"] = 32
    fallback_target = teaching_ready_target(
        model, data, mapping, geometry, fallback_config, reference_q
    )
    fallback_ik = solve_teaching_ready_ik(
        model, data, mapping, fallback_target, _ik_seeds(reference_q), fallback_config
    )
    tests["A6_unreachable_nominal_target_uses_fallback"] = bool(fallback_ik.success and fallback_ik.used_fallback)
    details["A6"] = fallback_ik.as_dict()

    # --- A7: an entirely unreachable target reports the distinct error code ----
    unreachable_config = copy.deepcopy(config)
    unreachable_config["teaching_ready"]["offset_x_m"] = {"value": 0.18, "source": "configured_prior"}
    unreachable_config["teaching_ready"]["fallback_search"]["enabled"] = False
    unreachable_target = teaching_ready_target(
        model, data, mapping, geometry, unreachable_config, reference_q
    )
    unreachable_ik = solve_teaching_ready_ik(
        model, data, mapping, unreachable_target, _ik_seeds(reference_q), unreachable_config
    )
    fallback_bound = len(_candidate_offsets(config)) * len(_ik_seeds(reference_q))
    tests["A7_unreachable_target_reports_ik_failed"] = bool(
        not unreachable_ik.success
        and unreachable_ik.failure == IK_FAILED
        and unreachable_ik.attempts <= fallback_bound
    )
    details["A7"] = {
        "failure": unreachable_ik.failure,
        "attempts": unreachable_ik.attempts,
        "attempt_bound": fallback_bound,
        "candidate_offsets": len(_candidate_offsets(config)),
    }

    # --- A8/A9: the planner starts from the current simulated state ------------
    detour_start = np.asarray(load_yaml("poses.yaml")["motor_zero"]["arm"], dtype=float)
    current_before = current_start_q(mapping, data).copy()
    plan = plan_initialization_path(model, data, mapping, detour_start, strict_ik.q, config, geometry=geometry)
    plan_from_current = plan_initialization_path(
        model, data, mapping, current_before, strict_ik.q, config, geometry=geometry
    )
    tests["A8_planner_start_is_current_sim_state"] = bool(
        np.allclose(plan.start_q, detour_start)
        and np.allclose(plan.waypoints[0], detour_start)
        and np.allclose(plan.waypoints[-1], strict_ik.q)
        and np.allclose(plan_from_current.start_q, current_before)
    )
    lower, upper = joint_limits(model, mapping, 0.0)
    waypoints = np.asarray(plan.waypoints)
    tests["A9_planner_returns_legal_trajectory"] = bool(
        plan.success
        and np.all(waypoints >= lower - 1e-9)
        and np.all(waypoints <= upper + 1e-9)
        and plan.maximum_velocity_rad_s
        <= float(config["planning"]["time_scaling"]["maximum_joint_velocity_rad_s"]) + 1e-6
        and plan.maximum_acceleration_rad_s2
        <= float(config["planning"]["time_scaling"]["maximum_joint_acceleration_rad_s2"]) + 1e-6
    )
    details["A8_A9"] = plan.as_dict()

    # --- A10-A12 + A13: execution is collision free and lands in tolerance -----
    execution_data = None
    execution = None
    if plan.success:
        import mujoco

        execution_data = mujoco.MjData(model)
        execution_mapping = reset_task_ready(model, execution_data)
        execution_data.qpos[execution_mapping.right_qpos] = detour_start
        execution_data.qvel[:] = 0.0
        mujoco.mj_forward(model, execution_data)
        execution = execute_initialization_trajectory(
            model, execution_data, execution_mapping, plan.waypoints, strict_ik.q, target, geometry, config
        )
    seen = set(execution.contact_labels_seen) if execution else {"no_execution"}
    tests["A10_no_table_collision_in_execution"] = "robot_table_collision" not in seen
    tests["A11_no_pedestal_collision_in_execution"] = "arm_pedestal_collision" not in seen
    tests["A12_no_left_arm_collision_in_execution"] = "inter_arm_collision" not in seen
    tests["A13_final_pose_within_tolerance"] = bool(execution and execution.success)
    if execution:
        details["A10_A13"] = execution.as_dict()

    # --- A15: after handoff the initialization reference is no longer in charge -
    planner_released = False
    if execution_data is not None:
        import mujoco

        # Enable the same compliant manual-teaching path the session uses, then check
        # the arm follows the operator handle away from the initialization goal.
        handle = VirtualHandle(
            execution_mapping.grasp_site,
            right_qpos=execution_mapping.right_qpos,
            right_dofs=execution_mapping.right_dof,
            preferred_q=strict_ik.q,
        )
        passive_gain = model.actuator_gainprm[execution_mapping.right_actuators].copy()
        model.actuator_gainprm[execution_mapping.right_actuators] = 0.0
        achieved_before = execution_data.site_xpos[execution_mapping.grasp_site].copy()
        handle.engage_at_current(execution_data)
        handle.set_target(achieved_before + np.asarray([0.0, 0.03, 0.0]))
        for _ in range(400):
            execution_data.qfrc_applied[:] = 0.0
            handle.apply(model, execution_data)
            mujoco.mj_step(model, execution_data)
        displacement = float(np.linalg.norm(execution_data.site_xpos[execution_mapping.grasp_site] - achieved_before))
        still_on_goal = float(
            np.max(np.abs(execution_data.qpos[execution_mapping.right_qpos] - strict_ik.q))
        )
        handle.release(execution_data)
        model.actuator_gainprm[execution_mapping.right_actuators] = passive_gain
        planner_released = bool(displacement > 0.005 and still_on_goal > 1e-3)
        details["A15"] = {"handle_displacement_m": displacement, "joint_departure_rad": still_on_goal}
    tests["A15_planner_released_after_handoff"] = planner_released

    # --- A16: the operator mouse/handle path still works -----------------------
    mujoco_model = load_model(str(config["initialization"]["scene"]))
    handle_data = __import__("mujoco").MjData(mujoco_model)
    handle_mapping = reset_task_ready(mujoco_model, handle_data)
    handle = VirtualHandle(handle_mapping.grasp_site)
    handle.engage_at_current(handle_data)
    start_position = handle_data.site_xpos[handle_mapping.grasp_site].copy()
    handle.set_target(start_position + np.asarray([0.02, 0.0, 0.0]))
    wrench = None
    for _ in range(300):
        handle_data.qfrc_applied[:] = 0.0
        wrench = handle.apply(mujoco_model, handle_data)
        __import__("mujoco").mj_step(mujoco_model, handle_data)
    displacement = float(np.linalg.norm(handle_data.site_xpos[handle_mapping.grasp_site] - start_position))
    handle.release(handle_data)
    tests["A16_mouse_handle_still_works"] = bool(displacement > 0.005 and wrench is not None)
    details["A16"] = {"handle_displacement_m": displacement}

    # --- A17: the gamepad path still works, including the deadman --------------
    # GamepadMapper consumes the raw Linux joystick ABI integers (+-32767), not
    # normalised floats; `_trigger` maps the LT axis to 0..1 as 0.5*(raw/32767+1).
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    axis_range = 32767.0
    axes = np.zeros(8, dtype=float)
    buttons = np.zeros(16, dtype=bool)
    axes[5] = axis_range  # LT deadman fully pressed
    axes[0] = 0.8 * axis_range  # left stick X
    command = mapper.update(axes, buttons)
    released = mapper.update(np.zeros(8, dtype=float), buttons)
    tests["A17_gamepad_still_works"] = bool(
        command.active
        and command.deadman_pressed
        and np.linalg.norm(command.linear_velocity_world_m_s) > 0.0
        and not released.active
        and np.allclose(released.linear_velocity_world_m_s, 0.0)
    )
    details["A17"] = {
        "active": command.active,
        "deadman_pressed": command.deadman_pressed,
        "linear_velocity_world_m_s": command.linear_velocity_world_m_s.tolist(),
        "released_active": released.active,
        "axis_units": "raw linux joystick ABI",
    }

    # --- A18: the shared gripper state machine still works --------------------
    gripper = GripperController(backend="mujoco")
    event = gripper.key_down("C", 0.0)
    states = [gripper.update(-0.35)]
    states.append(gripper.update(-0.10, bilateral_contact=True))
    tests["A18_gripper_still_works"] = bool(
        event.accepted
        and GripperState.CLOSING in states
        and GripperState.HOLDING in states
        and gripper.key_up("C") is None
    )
    details["A18"] = {"event": event.to_dict(), "states": [state.value for state in states]}

    # --- A19 + A14: transactions unaffected, initialization never recorded -----
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        recorder = EpisodeRecorder(root)
        # Run the initialization chain at the same time as a recorder exists, so the catalog
        # below is the state of the world while the chain ran, not a claim about the chain.
        before_run = run_initialization(execute=False, config=config)
        catalog_after_init = recorder.write_catalog()
        recorder.begin({"source": "human_virtual_handle", "include_initialization_trajectory": False})
        recorder.append("sidecar", 0.0, {"task_phase": "STABILIZE"})
        recorder.append("sidecar", 0.01, {"task_phase": "PREGRASP"})
        committed = recorder.finish_success({"full_sequence_success": True, "task_success": True})
        recorder.begin({"source": "human_virtual_handle"})
        recorder.append("sidecar", 0.0, {"task_phase": "PREGRASP"})
        failed = recorder.finish_failure("cube slipped")
        recorder.begin({"source": "human_virtual_handle"})
        recorder.append("sidecar", 0.0, {"task_phase": "APPROACH"})
        recorder.cancel()
        catalog = recorder.write_catalog()
        committed_metadata = json.loads((committed / "metadata.json").read_text(encoding="utf-8"))
        committed_samples = (committed / "sidecar.jsonl").read_text(encoding="utf-8").strip().splitlines()
        tests["A19_transactions_unaffected"] = bool(
            catalog["episodes"] == [committed.name]
            and catalog["failed"] == [failed.name]
            and not catalog["residual_transactions"]
            and committed_metadata["terminal_label"] == "S"
            and len(committed_samples) == 2
        )
        details["A19"] = {
            "catalog": catalog,
            "committed_terminal_label": committed_metadata["terminal_label"],
            "committed_sidecar_samples": len(committed_samples),
        }
        recording_capable = recording_capable_module_names(INITIALIZATION_PACKAGE_DIRECTORY)
        tests["A14_initialization_not_recorded"] = bool(
            not recording_capable and before_run["success"]
        )
        details["A14"] = {
            "catalog_after_initialization_only": catalog_after_init,
            "initialization_state": before_run["state"]["state"],
            "recording_capable_modules": recording_capable,
            "scanned_package": str(INITIALIZATION_PACKAGE_DIRECTORY),
            "include_initialization_trajectory": config["recording"]["include_initialization_trajectory"],
        }

    passed = sum(1 for value in tests.values() if value)
    return {
        "schema_version": 1,
        "gate": "SIM_INITIALIZATION_ACCEPTANCE",
        "status": "PASS" if passed == len(tests) else "FAIL",
        "passed": passed,
        "total": len(tests),
        "tests": tests,
        "details": details,
        "table_geometry": geometry.as_dict(),
        "teaching_ready_tcp": target.as_dict(),
        "statement": PASS_STATEMENT if passed == len(tests) else "SIM_INITIALIZATION_ACCEPTANCE = FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=workspace_path("results", "initialization", "gate_init_report.json"))
    args = parser.parse_args()
    report = validate_gate_init()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, value in report["tests"].items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    print(f"\n{report['statement']}  ({report['passed']}/{report['total']})")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
