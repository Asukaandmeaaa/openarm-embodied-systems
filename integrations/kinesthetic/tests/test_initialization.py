"""Initialization chain: prior table geometry, teaching-ready IK, sampling planner."""

import copy
import json

import mujoco
import numpy as np
import pytest

from openarm_kinesthetic.config import ConfigError, load_yaml, validate_named
from openarm_kinesthetic.initialization.collision import segment_valid
from openarm_kinesthetic.initialization.execute import quintic_blend, reference_at
from openarm_kinesthetic.initialization.ik import IK_FAILED, fallback_offsets, solve_teaching_ready_ik
from openarm_kinesthetic.initialization.pipeline import (
    _ik_seeds,
    build_scene,
    load_initialization_config,
    reference_pose_from_poses,
    run_initialization,
)
from openarm_kinesthetic.initialization.planner import START_INVALID, plan_initialization_path
from openarm_kinesthetic.initialization.states import (
    ERROR_CATEGORIES,
    REAL_STATES,
    SIM_STATES,
    InitializationStateMachine,
    StateTransitionError,
    category_of,
)
from openarm_kinesthetic.initialization.table_geometry import (
    SOURCE_LIVE_DEPTH,
    SOURCE_SIM_GROUND_TRUTH,
    TableGeometry,
    TableGeometryError,
    derive_table_center_x,
    real_table_geometry,
    robot_front_reference_x,
    sim_table_geometry,
)
from openarm_kinesthetic.initialization.teaching_ready import teaching_ready_target


@pytest.fixture(scope="module")
def scene():
    config = load_initialization_config()
    model, data, mapping = build_scene(config)
    return config, model, data, mapping


def test_prior_table_is_depth_along_x_and_width_along_y(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    assert geometry.size.tolist() == [0.40, 0.60]
    assert geometry.frame == "world"
    # The scene must agree with the prior, otherwise the prior is fiction.
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    assert np.allclose(model.geom_size[geom][:2], [0.20, 0.30])
    assert np.allclose(data.geom_xpos[geom][:2], geometry.center[:2], atol=1e-6)
    assert geometry.top_z == pytest.approx(0.40)


def test_table_center_x_is_derived_from_the_model_not_hardcoded(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    front = robot_front_reference_x(model, data, "openarm_body_link0_collision")
    # The pedestal collision mesh spans x in [-0.155, +0.095]; the front is its max.
    assert front == pytest.approx(0.095, abs=1e-4)
    assert geometry.center[0] == pytest.approx(derive_table_center_x(front, 0.17, 0.40), abs=1e-9)
    assert geometry.center[0] != pytest.approx(0.43, abs=1e-4)
    assert geometry.provenance["center.x"] == "derived_from_prior"
    assert geometry.provenance["robot_front_reference.x"] == "model_derived"


def test_table_geometry_requires_per_field_provenance():
    with pytest.raises(TableGeometryError, match="provenance"):
        TableGeometry(
            frame="world",
            center=np.zeros(3),
            size=np.ones(2),
            yaw=0.0,
            thickness_m=0.01,
            provenance={},
        )
    with pytest.raises(TableGeometryError, match="unknown provenance"):
        TableGeometry(
            frame="world",
            center=np.zeros(3),
            size=np.ones(2),
            yaw=0.0,
            thickness_m=0.01,
            provenance={
                "center.x": "derived_from_prior",
                "center.y": "configured_prior",
                "center.z": "maybe",
                "size.x": "user_reported_measurement",
                "size.y": "user_reported_measurement",
                "yaw": "configured_prior",
                "near_edge_gap": "user_measured_approximate",
                "robot_front_reference.x": "model_derived",
            },
        )


def test_real_provider_needs_depth_and_a_model_derived_front(scene):
    config, _model, _data, _mapping = scene
    with pytest.raises(TableGeometryError, match="live depth"):
        real_table_geometry(config, None, robot_front_reference_x_m=0.095)
    with pytest.raises(TableGeometryError, match="robot front reference"):
        real_table_geometry(config, 0.40, robot_front_reference_x_m=None)
    geometry = real_table_geometry(config, 0.42, robot_front_reference_x_m=0.095)
    assert geometry.provenance["center.z"] == SOURCE_LIVE_DEPTH
    assert geometry.provenance["center.x"] == "derived_from_prior"
    sim = sim_table_geometry(*scene[1:3], config)
    assert sim.provenance["center.z"] == SOURCE_SIM_GROUND_TRUTH
    assert geometry.top_z == pytest.approx(0.42)


def test_table_relative_conversions_round_trip(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    point = geometry.center[:2] + np.asarray([0.05, -0.11])
    assert np.allclose(geometry.to_world_xy(geometry.to_table_xy(point)), point, atol=1e-12)
    assert geometry.contains_xy(point)
    assert not geometry.contains_xy(geometry.center[:2] + np.asarray([0.0, 0.31]))


def test_teaching_ready_is_table_relative_with_fk_orientation(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    target = teaching_ready_target(model, data, mapping, geometry, config, reference_pose_from_poses(config))
    offset = config["teaching_ready"]
    expected = geometry.center[:2] + np.asarray(
        [offset["offset_x_m"]["value"], offset["offset_y_m"]["value"]]
    )
    assert np.allclose(target.position_world_m[:2], expected)
    assert target.position_world_m[2] == pytest.approx(geometry.top_z + offset["height_above_table_m"]["value"])
    assert target.orientation_source == "fk_reference"
    assert geometry.contains_xy(target.position_world_m[:2])
    assert np.allclose(target.orientation_matrix @ target.orientation_matrix.T, np.eye(3), atol=1e-9)


def test_teaching_ready_rejects_an_invented_orientation_source(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    broken = copy.deepcopy(config)
    broken["teaching_ready"]["orientation"]["source"] = "euler_guess"
    with pytest.raises(ValueError, match="fk_reference"):
        teaching_ready_target(model, data, mapping, geometry, broken, reference_pose_from_poses(config))


def test_teaching_ready_rejects_offsets_off_the_table(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    broken = copy.deepcopy(config)
    broken["teaching_ready"]["offset_y_m"] = {"value": 0.45, "source": "configured_prior"}
    with pytest.raises(ValueError, match="footprint"):
        teaching_ready_target(model, data, mapping, geometry, broken, reference_pose_from_poses(config))


def test_configured_teaching_ready_target_is_reachable_without_fallback(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    target = teaching_ready_target(model, data, mapping, geometry, config, reference_pose_from_poses(config))
    result = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference_pose_from_poses(config)), config)
    assert result.success
    assert not result.used_fallback
    assert result.position_error_m <= config["teaching_ready"]["ik"]["position_tolerance_m"]
    assert result.orientation_error_rad <= config["teaching_ready"]["ik"]["orientation_tolerance_rad"]


def test_fallback_search_is_deterministic_and_bounded():
    offsets = fallback_offsets(0.03, 0.09, 64)
    assert [offset.tolist() for offset in offsets] == [offset.tolist() for offset in fallback_offsets(0.03, 0.09, 64)]
    assert len(offsets) == 29  # disk of radius 3 cells at 0.03 m
    assert all(float(np.linalg.norm(offset)) <= 0.09 + 1e-9 for offset in offsets)
    # Nearest-first ordering: the origin, then a single step.
    assert np.allclose(offsets[0], [0.0, 0.0])
    assert float(np.linalg.norm(offsets[1])) == pytest.approx(0.03)
    assert len(fallback_offsets(0.03, 0.09, 4)) == 4


def test_unreachable_target_reports_the_distinct_ik_error(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    broken = copy.deepcopy(config)
    broken["teaching_ready"]["offset_x_m"] = {"value": 0.18, "source": "configured_prior"}
    broken["teaching_ready"]["fallback_search"]["enabled"] = False
    target = teaching_ready_target(model, data, mapping, geometry, broken, reference_pose_from_poses(config))
    result = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference_pose_from_poses(config)), broken)
    assert not result.success
    assert result.failure == IK_FAILED
    assert result.attempts == len(_ik_seeds(reference_pose_from_poses(config)))
    assert category_of(result.failure) == "ik"


def test_planner_returns_a_collision_free_path_between_the_requested_endpoints(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    reference = reference_pose_from_poses(config)
    target = teaching_ready_target(model, data, mapping, geometry, config, reference)
    ik = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference), config)
    start = np.asarray([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=float)
    plan = plan_initialization_path(model, data, mapping, start, ik.q, config, geometry=geometry)
    assert plan.success
    assert np.allclose(plan.waypoints[0], start)
    assert np.allclose(plan.waypoints[-1], ik.q)
    assert plan.straight_line_length_rad > 0.0
    # The straight line is blocked here, so a genuine detour is required. A
    # np.linspace "planner" would either collide or report zero deviation.
    straight_ok, _ = segment_valid(model, data, mapping, start, ik.q, step_rad=0.02, margin_rad=0.02)
    assert not straight_ok
    assert plan.maximum_deviation_from_straight_rad > 0.05
    assert plan.maximum_velocity_rad_s <= config["planning"]["time_scaling"]["maximum_joint_velocity_rad_s"] + 1e-9
    assert plan.smoothed


def test_planner_is_deterministic_for_a_fixed_seed(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    reference = reference_pose_from_poses(config)
    target = teaching_ready_target(model, data, mapping, geometry, config, reference)
    ik = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference), config)
    start = np.asarray([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=float)
    first = plan_initialization_path(model, data, mapping, start, ik.q, config, geometry=geometry)
    second = plan_initialization_path(model, data, mapping, start, ik.q, config, geometry=geometry)
    assert [node.tolist() for node in first.waypoints] == [node.tolist() for node in second.waypoints]


def test_planner_refuses_a_start_outside_hard_joint_limits(scene):
    config, model, data, mapping = scene
    geometry = sim_table_geometry(model, data, config)
    reference = reference_pose_from_poses(config)
    target = teaching_ready_target(model, data, mapping, geometry, config, reference)
    ik = solve_teaching_ready_ik(model, data, mapping, target, _ik_seeds(reference), config)
    invalid = np.zeros(7)
    invalid[3] = -0.10
    plan = plan_initialization_path(model, data, mapping, invalid, ik.q, config, geometry=geometry)
    assert not plan.success
    assert plan.failure == START_INVALID
    assert category_of(plan.failure) == "planning"


def test_quintic_blend_has_zero_endpoint_velocity_and_acceleration():
    assert quintic_blend(0.0) == pytest.approx(0.0)
    assert quintic_blend(1.0) == pytest.approx(1.0)
    assert quintic_blend(0.5) == pytest.approx(0.5)
    samples = np.linspace(0.0, 1.0, 201)
    values = np.asarray([quintic_blend(value) for value in samples])
    assert np.all(np.diff(values) >= -1e-12)  # monotonic
    # q(a) = 10a^3 - 15a^4 + 6a^5  =>  q'(a) = 30a^2(1-a)^2, q''(a) = 60a(1-a)(1-2a).
    derivative = 30.0 * samples**2 * (1.0 - samples) ** 2
    second = 60.0 * samples * (1.0 - samples) * (1.0 - 2.0 * samples)
    assert derivative[0] == pytest.approx(0.0, abs=1e-15)
    assert derivative[-1] == pytest.approx(0.0, abs=1e-15)
    assert second[0] == pytest.approx(0.0, abs=1e-15)
    assert second[-1] == pytest.approx(0.0, abs=1e-15)
    assert derivative.max() == pytest.approx(1.875)  # peak velocity of the blend
    assert second.max() == pytest.approx(5.7735, rel=1e-3)  # peak acceleration of the blend


def test_reference_at_endpoints_matches_waypoints():
    waypoints = [np.zeros(7), np.ones(7), np.full(7, 2.0)]
    durations = [1.0, 2.0]
    assert np.allclose(reference_at(waypoints, durations, 0.0), waypoints[0])
    assert np.allclose(reference_at(waypoints, durations, 1.0), waypoints[1])
    assert np.allclose(reference_at(waypoints, durations, 3.0), waypoints[2])
    assert np.allclose(reference_at(waypoints, durations, 99.0), waypoints[2])


def test_pipeline_reaches_manual_teaching_and_never_records(scene):
    config, model, data, mapping = scene
    current_before = data.qpos[mapping.right_qpos].copy()
    report = run_initialization(execute=False, config=config, model=model, data=data, mapping=mapping)
    assert report["success"]
    assert report["state"]["state"] == "SIM_MANUAL_TEACHING"
    assert report["state"]["error"] is None
    assert report["start_state_source"] == "current_joint_state"
    np.testing.assert_allclose(report["start_q"], current_before)
    assert report["planner"]["planner"] == "rrt_connect"
    assert report["table"]["provenance"]["center.x"] == "derived_from_prior"
    # Setup motion is never a demonstration.
    assert config["recording"]["include_initialization_trajectory"] is False


def test_default_pipeline_build_starts_at_physical_motor_zero():
    config = load_initialization_config()
    model, data, mapping = build_scene(config)
    np.testing.assert_allclose(data.qpos[mapping.right_qpos], np.zeros(7))
    report = run_initialization(execute=False, config=config, model=model, data=data, mapping=mapping)
    np.testing.assert_allclose(report["start_q"], np.zeros(7))


def test_state_machine_blocks_illegal_and_unconfirmed_hardware_steps():
    machine = InitializationStateMachine()
    machine.advance("SIM_SCENE_READY")
    machine.advance("SIM_TABLE_READY")
    with pytest.raises(StateTransitionError, match="illegal"):
        machine.advance("SIM_PLAN_READY")
    machine.advance("SIM_COMPUTE_TEACH_READY")
    machine.advance("SIM_IK_READY")
    machine.advance("SIM_PLAN_READY")

    real = InitializationStateMachine()
    real.advance("REAL_WAIT_CAMERA")
    real.advance("REAL_ESTIMATE_TABLE_HEIGHT")
    real.advance("REAL_TABLE_STABLE")
    real.advance("REAL_TABLE_PREVIEW")
    real.advance("REAL_TABLE_COMMITTED")
    real.advance("REAL_CAPTURE_START_STATE")
    real.advance("REAL_COMPUTE_TEACH_READY")
    real.advance("REAL_IK_READY")
    real.advance("REAL_PLAN_READY")
    # Execution is reachable only through the plan-only / confirmation gate.
    real.advance("REAL_WAIT_EXECUTION_CONFIRM")
    with pytest.raises(StateTransitionError, match="EXECUTION_NOT_ENABLED"):
        real.advance("REAL_INITIALIZATION_EXECUTING")
    with pytest.raises(StateTransitionError, match="EXECUTION_CONFIRMATION_REQUIRED"):
        real.advance("REAL_INITIALIZATION_EXECUTING", execution_enabled=True)
    real.advance("REAL_INITIALIZATION_EXECUTING", execution_enabled=True, operator_confirmed=True)
    real.advance("REAL_INITIALIZATION_REACHED")
    real.advance("REAL_TEACH_READY_RIGID")
    real.advance("REAL_MANUAL_TEACHING")


def test_every_error_code_is_distinguishable_and_states_are_separated():
    categories = {category_of(code) for code in ERROR_CATEGORIES}
    assert categories == {
        "simulation_model",
        "vision",
        "table_placement",
        "ik",
        "planning",
        "simulation_execution",
        "hardware_execution",
        "handoff",
    }
    # Perception, IK, planning, execution and handoff must not share a state name.
    assert not set(SIM_STATES) & set(REAL_STATES)
    assert len(set(SIM_STATES)) == len(SIM_STATES)
    assert len(set(REAL_STATES)) == len(REAL_STATES)
    assert "REAL_TEACH_READY_RIGID" in REAL_STATES
    assert "REAL_INITIALIZATION_REACHED" in REAL_STATES


def test_initialization_config_passes_project_validation():
    validate_named("initialization.yaml", load_yaml("initialization.yaml"))


def test_initialization_config_rejects_recording_or_execution_regressions():
    base = load_yaml("initialization.yaml")
    recording = copy.deepcopy(base)
    recording["recording"]["include_initialization_trajectory"] = True
    with pytest.raises(ConfigError, match="never be included"):
        validate_named("initialization.yaml", recording)

    execution = copy.deepcopy(base)
    execution["real_execution"]["enabled"] = True
    with pytest.raises(ConfigError, match="disabled by default"):
        validate_named("initialization.yaml", execution)

    planner = copy.deepcopy(base)
    planner["planning"]["planner"] = "joint_interpolation"
    with pytest.raises(ConfigError, match="sampling planner"):
        validate_named("initialization.yaml", planner)

    hardcoded = copy.deepcopy(base)
    hardcoded["table"]["robot_front_reference"]["x_m"] = 0.095
    with pytest.raises(ConfigError, match="derived at runtime"):
        validate_named("initialization.yaml", hardcoded)

    wrong_start = copy.deepcopy(base)
    wrong_start["simulation"]["initial_pose"] = "right_task_ready_initial_candidate"
    with pytest.raises(ConfigError, match="vertical-down motor_zero"):
        validate_named("initialization.yaml", wrong_start)


def test_gate_report_is_json_serialisable(scene):
    config, model, data, mapping = scene
    report = run_initialization(execute=False, config=config, model=model, data=data, mapping=mapping)
    assert json.loads(json.dumps(report))["success"] is True
