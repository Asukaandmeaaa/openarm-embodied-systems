"""Tests for the simulation-only Teaching Ready SE(3) pose search.

The assertions here map one-to-one onto the properties the search has to have before it
is worth porting to ROS 2:

* the orientation families are built from the compiled tool/table frames and are
  genuinely different from each other;
* different families land on clearly different J4 configurations, instead of every
  candidate sitting near the same value;
* the collision and joint-limit filters actually reject configurations;
* duplicate IK solutions are collapsed;
* the gravity-torque metric is finite, matches MuJoCo's own bias force, and is
  reproducible;
* Top-K planning and the Top-N shortlist are deterministic;
* the legacy ``legacy_pi2`` baseline still runs and reproduces the recorded offset-scan
  selection;
* nothing in this package touches real ROS 2, CAN, MoveIt, or hardware modules.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openarm_kinesthetic.config import ConfigError, load_yaml, validate_ready_search
from openarm_kinesthetic.initialization.collision import configuration_valid
from openarm_kinesthetic.ready_search.config import load_ready_search_config, quick_overrides, refinement_offsets
from openarm_kinesthetic.ready_search.metrics import (
    J4_JOINT_NAME,
    gravity_torque_metrics,
    j4_index,
    static_gravity_torque,
    trajectory_travel_metrics,
    trajectory_gravity_torque,
)
from openarm_kinesthetic.ready_search.report import select_top_n
from openarm_kinesthetic.ready_search.scoring import static_score
from openarm_kinesthetic.ready_search.search import (
    Candidate,
    TargetPosition,
    TeachingReadySearch,
    deduplicate,
    run_ready_search,
    verify_determinism,
)
from openarm_kinesthetic.ready_search.tool_frame import (
    TOOL_SITE,
    minimal_rotation,
    orientation_candidates,
    orientation_for_approach,
)
from openarm_kinesthetic.scene.model import RIGHT_JOINTS, name_id

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "openarm_kinesthetic" / "ready_search"

# A configuration recorded from a deterministic joint grid scan: it drives the arm into
# the table. Fixed constant rather than a search, so the collision filter test cannot
# drift with the rest of the search.
COLLIDING_Q = np.asarray([0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0])
# The TCP pose the earlier offset scan selected, from
# results/initialization/teaching_ready_offset_scan.json.
RECORDED_SCAN_Q = np.asarray(
    [-0.5401738286285871, 2.883191363272719, 0.16171039120593625, 1.3605797661370707,
     1.2818286107877366, 0.10064439487498901, 1.2776404823927299]
)


def test_trajectory_travel_is_cumulative_and_matches_real_metric_semantics() -> None:
    path = [
        np.zeros(7),
        np.asarray([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    metrics = trajectory_travel_metrics(path)

    assert metrics["joint_path_length_rad"] == pytest.approx(3.0)
    assert metrics["maximum_joint_travel_rad"] == pytest.approx(2.0)
    assert metrics["j4_travel_rad"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Fixtures: each search runs once for the whole module.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quick_config():
    return load_ready_search_config(quick_overrides())


@pytest.fixture(scope="module")
def quick_search(quick_config):
    return TeachingReadySearch(quick_config)


@pytest.fixture(scope="module")
def full_search():
    """The full family sweep (all four oblique angles, all heading offsets)."""
    return TeachingReadySearch(load_ready_search_config())


@pytest.fixture(scope="module")
def quick_report(quick_search):
    return quick_search.run()


@pytest.fixture(scope="module")
def full_report():
    return run_ready_search()


# --------------------------------------------------------------------------------------
# Orientation families
# --------------------------------------------------------------------------------------


def test_j4_index_is_derived_from_the_project_ordering() -> None:
    assert RIGHT_JOINTS[j4_index()] == J4_JOINT_NAME
    assert j4_index() == 3
    assert RIGHT_JOINTS[3] == "openarm_right_joint4"


def test_tool_frame_is_derived_from_the_compiled_model(full_search) -> None:
    frame = full_search.frame
    # The approach axis is read from model.site_pos / model.site_quat, not written down.
    assert float(np.linalg.norm(frame.approach_axis_tool)) == pytest.approx(1.0)
    assert frame.approach_axis_is_model_axis
    assert frame.tool_length_m == pytest.approx(0.138, abs=0.005)
    # Orthonormal tool frame.
    basis = np.column_stack((frame.approach_axis_tool, frame.lateral_axis_tool, frame.third_axis_tool))
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    # Table frame derived from the compiled table geom and cross-checked against the prior.
    assert np.allclose(frame.table_normal_world, [0.0, 0.0, 1.0], atol=1e-9)
    assert frame.table_scene_normal_error_deg < 2.0
    assert frame.table_scene_forward_error_deg < 2.0
    assert np.allclose(frame.table_lateral_world, np.cross(frame.table_normal_world, frame.table_forward_world))


def test_orientation_families_are_distinct(full_search, full_report) -> None:
    frame = full_search.frame
    candidates = full_search.orientations
    by_label = {candidate.label: candidate for candidate in candidates}

    legacy = by_label["legacy_pi2"]
    assert legacy.tilt_below_horizontal_deg == pytest.approx(89.3, abs=2.0)
    assert "forward_kinematics" in legacy.source

    horizontal = by_label["side_horizontal_0"]
    assert horizontal.tilt_below_horizontal_deg == pytest.approx(0.0, abs=1e-6)
    assert abs(horizontal.heading_deg - 90.0) < 1.0
    # "Horizontal" really is horizontal: the approach axis lies in the table plane.
    assert abs(float(horizontal.approach_axis_world @ frame.table_normal_world)) < 1e-9

    tilts = {
        label: round(candidate.tilt_below_horizontal_deg, 6)
        for label, candidate in by_label.items()
        if label.startswith("oblique")
    }
    assert set(tilts.values()) == {30.0, 40.0, 50.0, 60.0, 70.0}
    assert {
        "oblique_30_h-15",
        "oblique_30",
        "oblique_30_h+15",
        "oblique_70_h-15",
        "oblique_70",
        "oblique_70_h+15",
    } <= set(tilts)

    # Every family points somewhere clearly different from the others.
    def axis(candidate) -> np.ndarray:
        return np.asarray(candidate.approach_axis_world, dtype=float)

    assert float(np.degrees(np.arccos(abs(axis(horizontal) @ axis(legacy))))) > 60.0
    for label in ("oblique_30", "oblique_40", "oblique_50", "oblique_60"):
        assert float(np.degrees(np.arccos(axis(horizontal) @ axis(by_label[label])))) >= 29.0
        assert float(np.degrees(np.arccos(abs(axis(by_label[label]) @ axis(legacy))))) > 25.0
    assert float(np.degrees(np.arccos(axis(by_label["oblique_30"]) @ axis(by_label["oblique_60"])))) > 25.0

    # The report publishes the same set, so the numbers a reader sees are these ones.
    published = {item["label"] for item in full_report["orientation_candidates"]}
    assert {"legacy_pi2", "side_horizontal_0", "oblique_30", "oblique_40", "oblique_50", "oblique_60"} <= published


def test_search_anchor_uses_the_operator_marked_table_point(full_search, full_report) -> None:
    assert full_search.config.position_anchor_source == "operator_marked_table_point"
    assert np.allclose(full_search.anchor_m, [0.465, -0.24, 0.68], atol=2e-6)
    assert np.allclose(full_search.legacy_anchor_m, [0.315, -0.19, 0.68], atol=2e-6)
    assert full_report["anchor"]["source"] == "operator_marked_table_point"
    assert np.allclose(
        full_report["anchor"]["movement_from_legacy_m"],
        [0.15, -0.05, 0.0],
        atol=2e-6,
    )
    assert "j4_reference_diagnostic" not in full_report
    assert all(
        np.allclose(
            item["tcp_pose"]["position_world_m"],
            full_search.anchor_m,
            atol=2e-6,
        )
        for item in full_report["top_n"]
    )


def test_generated_orientation_reproduces_the_requested_approach_axis(full_search) -> None:
    frame = full_search.frame
    for candidate in full_search.orientations:
        matrix = np.asarray(candidate.orientation_matrix, dtype=float)
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0)
        # R @ tool_approach_axis is exactly the requested world direction.
        achieved = frame.approach_axis_world(matrix)
        assert np.allclose(achieved, candidate.approach_axis_world, atol=1e-9)
        # ... and its tilt below the table plane reproduces the configured angle.
        expected = -math.sin(math.radians(candidate.tilt_below_horizontal_deg))
        assert float(achieved @ frame.table_normal_world) == pytest.approx(expected, abs=1e-9)


def test_orientation_roll_is_resolved_towards_the_reference(full_search) -> None:
    frame = full_search.frame
    reference = full_search.reference_matrix
    axis = frame.approach_axis_tool
    orientation, approach = orientation_for_approach(
        frame, reference, tilt_below_horizontal_deg=40.0, heading_deg=0.0, roll_deg=0.0
    )
    assert np.allclose(orientation @ axis, approach, atol=1e-9)
    # The roll-free orientation is the smallest possible rotation away from the reference.
    minimal = minimal_rotation(reference @ axis, approach) @ reference
    assert np.allclose(orientation, minimal, atol=1e-9)


def test_orientation_candidates_are_deterministic(full_search) -> None:
    again = orientation_candidates(full_search.frame, full_search.reference_matrix, full_search.config.families)
    assert [item.label for item in again] == [item.label for item in full_search.orientations]
    for first, second in zip(again, full_search.orientations):
        assert np.allclose(first.orientation_matrix, second.orientation_matrix, atol=0.0)


# --------------------------------------------------------------------------------------
# Filters and duplicate handling
# --------------------------------------------------------------------------------------


def test_collision_filter_rejects_a_table_contact(quick_search) -> None:
    valid, failures = configuration_valid(
        quick_search.model, quick_search.data, quick_search.mapping, COLLIDING_Q, margin_rad=0.02
    )
    assert not valid
    assert failures == {"robot_table_collision"}


def test_joint_limit_filter_rejects_out_of_range_configurations(quick_search) -> None:
    valid, failures = configuration_valid(
        quick_search.model, quick_search.data, quick_search.mapping, np.full(7, 10.0), margin_rad=0.02
    )
    assert not valid
    assert failures == {"joint_limit_margin"}


def test_every_published_candidate_passed_the_filters(quick_report, quick_config) -> None:
    candidates = quick_report["candidates"]
    assert candidates
    for candidate in candidates:
        metrics = candidate["static_metrics"]
        assert metrics["minimum_joint_margin_normalized"] >= quick_config.filters["minimum_joint_margin_normalized"]
        assert metrics["j4_margin_normalized"] >= quick_config.filters["minimum_j4_margin_normalized"]
        assert metrics["minimum_clearance_m"] >= -quick_config.acceptance["penetration_tolerance_m"]
        assert np.all(metrics["q_rad"] >= metrics["joint_lower_rad"])
        assert np.all(metrics["q_rad"] <= metrics["joint_upper_rad"])
    # At least one generated solve is rejected in the reduced profile, so the accepted
    # candidate assertions above are exercised alongside the rejection path.  Which
    # particular filter fires is solver/version dependent at this single fixed TCP.
    assert sum(quick_report["filter_counts"].values()) > 0


def test_duplicate_solutions_are_collapsed(full_report) -> None:
    first = _synthetic_candidate("a", np.zeros(7), margin=0.2, j4_margin=0.3, torque_j4=-2.0)
    near_duplicate = _synthetic_candidate("b", np.full(7, 0.01), margin=0.1, j4_margin=0.1, torque_j4=-4.0)
    distinct = _synthetic_candidate("c", np.full(7, 1.0), margin=0.05, j4_margin=0.05, torque_j4=-1.0)

    kept, dropped = deduplicate([first, near_duplicate, distinct], threshold_rad=0.1)
    assert dropped == 1
    assert {candidate.candidate_id for candidate in kept} == {"a", "c"}
    # The better-placed solution survives, not merely the first one seen.
    assert first.duplicate_count == 1
    assert near_duplicate.duplicate_of == "a"

    # The real search exercises the same path: on the full grid, several IK seeds reach
    # the same branch for the same TCP pose and are collapsed.
    assert full_report["counts"]["duplicates_collapsed"] > 0


def test_no_two_published_candidates_share_a_configuration(quick_report) -> None:
    candidates = quick_report["candidates"]
    threshold = quick_report["search_parameters"]["dedup_max_joint_distance_rad"]
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            difference = float(np.max(np.abs(np.asarray(first["q_rad"]) - np.asarray(second["q_rad"]))))
            assert difference >= threshold


def test_clearance_metric_ignores_visual_geometry(quick_search) -> None:
    pairs = quick_search.clearance
    visual_geoms = {
        geom for geom in range(quick_search.model.ngeom) if int(quick_search.model.geom_group[geom]) == 2
    }
    involved = set()
    for group in ("task_obstacles", "structure", "self"):
        for first, second in pairs[group]:
            involved.update((first, second))
    assert involved
    assert not (involved & visual_geoms)
    # The arm's structural mount inside the pedestal is excluded by name, not silently.
    excluded = {item["pair"] for item in pairs["excluded_pairs"]}
    assert "base_link_right_collision_00|openarm_body_link0_collision" in excluded


# --------------------------------------------------------------------------------------
# Gravity torque
# --------------------------------------------------------------------------------------


def test_static_gravity_torque_matches_the_project_convention(quick_search) -> None:
    """The value must be MuJoCo's own zero-velocity bias force, not a hand-rolled model."""
    import mujoco

    search = quick_search
    for q in (search.start_q, search.reference_q, COLLIDING_Q):
        torque = static_gravity_torque(search.model, search.data, search.mapping, q)
        search.data.qpos[search.mapping.right_qpos] = np.asarray(q, dtype=float)
        search.data.qvel[:] = 0.0
        mujoco.mj_forward(search.model, search.data)
        assert np.allclose(torque, search.data.qfrc_bias[search.mapping.right_dof], atol=1e-9)
        assert np.all(np.isfinite(torque))
        # Every right-arm joint is forcelimited at 40 N*m or below.
        assert float(np.max(np.abs(torque))) < 40.0


def test_gravity_torque_metrics_name_j4_and_are_finite(quick_search) -> None:
    # Measured at the pose the arm actually attempted on 2026-09-17, not at the FK
    # reference: J4 gravity torque is not monotonic in the J4 angle, which is exactly
    # why "the angle is bad" and "the joint is loaded" have to be measured separately.
    torque = static_gravity_torque(quick_search.model, quick_search.data, quick_search.mapping, RECORDED_SCAN_Q)
    metrics = gravity_torque_metrics(torque)
    assert metrics["gravity_torque_j4_nm"] == pytest.approx(float(torque[j4_index()]))
    assert metrics["absolute_gravity_torque_j4_nm"] == pytest.approx(abs(float(torque[j4_index()])))
    assert metrics["maximum_absolute_gravity_torque_nm"] == pytest.approx(float(np.max(np.abs(torque))))
    # At that pose the shoulder carries the whole arm, but J4 dominates every other
    # distal joint by an order of magnitude and carries a large share of the total load.
    # That is the measured reason a J4 red light there is at least as likely to be a load
    # problem as a joint-angle problem.
    assert metrics["maximum_absolute_gravity_torque_joint"] == "openarm_right_joint1"
    assert abs(float(torque[j4_index()])) > 10.0 * max(float(np.max(np.abs(torque[4:]))), 1e-9)
    assert abs(float(torque[j4_index()])) > 0.25 * float(np.max(np.abs(torque)))


def test_gravity_torque_is_reproducible(quick_search) -> None:
    q = quick_search.reference_q
    first = static_gravity_torque(quick_search.model, quick_search.data, quick_search.mapping, q)
    second = static_gravity_torque(quick_search.model, quick_search.data, quick_search.mapping, q)
    assert np.array_equal(first, second)


def test_trajectory_gravity_torque_samples_the_whole_path(quick_search, quick_report) -> None:
    planned = [item for item in quick_report["planned_candidates"] if item["trajectory"]["success"]]
    assert planned
    sample_count = quick_report["search_parameters"]["metrics"]["trajectory_sample_count"]
    for item in planned:
        trajectory = item["trajectory"]
        series = np.asarray(trajectory["trajectory_j4_gravity_torque_nm"], dtype=float)
        assert series.size == sample_count
        assert np.all(np.isfinite(series))
        reported = float(trajectory["maximum_absolute_j4_gravity_torque_nm"])
        assert reported == pytest.approx(float(np.max(np.abs(series))))
        # The path maximum can exceed the endpoint alone, which is the reason it exists.
        assert reported >= abs(float(trajectory["endpoint_j4_gravity_torque_nm"])) - 1e-9
        assert float(trajectory["mean_absolute_j4_gravity_torque_nm"]) <= reported + 1e-9
        assert trajectory["worst_sample_q_rad"] is not None

    # Directly re-measure one trajectory and compare with the report.
    item = planned[0]
    waypoints = _waypoints_for(quick_search, item)
    repeated = trajectory_gravity_torque(
        quick_search.model, quick_search.data, quick_search.mapping, waypoints, sample_count=sample_count
    )
    assert repeated["maximum_absolute_j4_gravity_torque_nm"] == pytest.approx(
        float(item["trajectory"]["maximum_absolute_j4_gravity_torque_nm"])
    )


def _waypoints_for(quick_search, item: dict) -> list[np.ndarray]:
    """Re-plan the trajectory of a reported candidate, to re-measure it independently."""
    from openarm_kinesthetic.initialization.planner import plan_initialization_path

    planning = dict(quick_search.config.initialization.get("planning", {}))
    planning["maximum_iterations"] = int(quick_search.config.raw["planning"]["maximum_iterations"])
    result = plan_initialization_path(
        quick_search.model,
        quick_search.data,
        quick_search.mapping,
        quick_search.start_q,
        np.asarray(item["q_rad"], dtype=float),
        {**quick_search.config.initialization, "planning": planning},
        geometry=quick_search.geometry,
    )
    assert result.success
    return result.waypoints


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def test_reference_distance_weight_is_zero_in_this_round(quick_config, quick_report) -> None:
    assert quick_config.static_weights["legacy_reference_distance_normalized"] == 0.0
    for candidate in quick_report["candidates"]:
        component = candidate["static_score"]["components"]["legacy_reference_distance_normalized"]
        assert component["weight"] == 0.0
        assert component["contribution"] == 0.0


def test_static_score_is_the_sum_of_its_contributions(quick_report) -> None:
    for candidate in quick_report["candidates"]:
        score = candidate["static_score"]
        total = sum(component["contribution"] for component in score["components"].values())
        # The report is rounded to `round_floats_to` (6) digits per entry, so the sum of
        # seven rounded contributions can drift by up to ~7 * 5e-7 from the rounded total.
        assert score["score"] == pytest.approx(total, abs=1e-5)
        for component in score["components"].values():
            assert 0.0 <= component["normalized"] <= 1.0


def test_scoring_rewards_less_j4_load() -> None:
    references = {
        "manipulability": {"value": 0.02},
        "collision_clearance_m": {"value": 0.1},
        "start_joint_distance_rad": {"value": 3.0},
        "legacy_reference_distance_rad": {"value": 0.5},
        "gravity_torque_j4_nm": {"value": 3.5},
        "gravity_torque_max_all_nm": {"value": 7.0},
    }
    weights = {
        "joint_margin_normalized": 0.2,
        "j4_margin_normalized": 0.25,
        "manipulability_normalized": 0.15,
        "collision_clearance_normalized": 0.15,
        "start_joint_distance_normalized": -0.05,
        "gravity_torque_normalized": -0.2,
        "legacy_reference_distance_normalized": 0.0,
    }
    base = {
        "minimum_joint_margin_normalized": 0.2,
        "j4_margin_normalized": 0.3,
        "yoshikawa_manipulability": 0.02,
        "minimum_clearance_m": 0.1,
        "minimum_task_obstacles_clearance_m": 0.1,
        "minimum_structure_clearance_m": 0.1,
        "minimum_self_clearance_m": 0.1,
        "start_joint_distance_rad": 1.5,
        "legacy_reference_distance_rad": 0.2,
        "gravity_torque_j4_nm": -2.0,
        "absolute_gravity_torque_j4_nm": 2.0,
        "maximum_absolute_gravity_torque_nm": 5.0,
    }
    loaded = {**base, "gravity_torque_j4_nm": -4.0, "absolute_gravity_torque_j4_nm": 4.0}
    light = static_score(base, weights, references)
    heavy = static_score(loaded, weights, references)
    assert light["score"] > heavy["score"]
    # Moving the joint margin the other way flips the ordering, which proves the margin
    # term is not just an offset.
    generous = static_score({**base, "j4_margin_normalized": 0.8}, weights, references)
    assert generous["score"] > light["score"]


def test_legacy_baseline_references_are_run_relative(quick_report) -> None:
    references = quick_report["resolved_references"]["static"]
    for key in ("gravity_torque_j4_nm", "gravity_torque_max_all_nm", "manipulability"):
        entry = references[key]
        assert entry["configured"]["mode"] == "legacy_baseline"
        assert entry["resolved_from"] in {"legacy_baseline", "legacy_baseline_fallback"}
    baseline = quick_report["legacy_baseline"]
    measured = abs(float(baseline["static_gravity_torque_j4_nm"]))
    assert references["gravity_torque_j4_nm"]["value"] == pytest.approx(measured, rel=1e-9)


# --------------------------------------------------------------------------------------
# Baseline, determinism, shortlist
# --------------------------------------------------------------------------------------


def test_legacy_pi2_baseline_still_runs_and_matches_the_recorded_scan(quick_report, full_report) -> None:
    baseline = full_report["legacy_baseline"]
    assert baseline is not None
    assert baseline["candidate_id"].startswith("legacy_pi2")
    assert np.allclose(baseline["q_rad"], RECORDED_SCAN_Q, atol=1e-6)
    assert baseline["j4_rad"] == pytest.approx(float(RECORDED_SCAN_Q[j4_index()]), abs=1e-9)
    assert baseline["static_absolute_gravity_torque_j4_nm"] > 0.0
    assert baseline["trajectory"]["success"]
    assert quick_report["legacy_baseline"] is not None


def test_distinct_joint_branches_have_unique_candidate_ids(full_report) -> None:
    ids = [candidate["candidate_id"] for candidate in full_report["candidates"]]
    assert len(ids) == len(set(ids))
    # Only distinct configurations are ever planned, which is what the duplicate
    # collapse is for.
    threshold = full_report["search_parameters"]["dedup_max_joint_distance_rad"]
    planned = full_report["planned_candidates"]
    for index, first in enumerate(planned):
        for second in planned[index + 1 :]:
            difference = float(np.max(np.abs(np.asarray(first["q_rad"]) - np.asarray(second["q_rad"]))))
            assert difference >= threshold


def test_different_families_produce_clearly_different_j4(full_report) -> None:
    by_family: dict[str, list[float]] = {}
    for candidate in full_report["candidates"]:
        by_family.setdefault(candidate["family"], []).append(float(candidate["static_metrics"]["j4_rad"]))
    assert {"legacy_pi2", "side_horizontal", "oblique"} <= set(by_family)
    medians = {family: float(np.median(values)) for family, values in by_family.items()}
    assert max(medians.values()) - min(medians.values()) >= 0.10
    spread = max(max(values) for values in by_family.values()) - min(min(values) for values in by_family.values())
    assert spread >= full_report["acceptance"]["minimum_j4_spread_rad"]
    assert full_report["acceptance"]["checks"]["families_produce_distinct_j4"] is True
    # Not every candidate sits at the same J4, which is the failure mode this search
    # exists to rule out.
    all_j4 = [value for values in by_family.values() for value in values]
    assert len({round(value, 2) for value in all_j4}) >= 5


def test_full_profile_meets_every_acceptance_check(full_report) -> None:
    acceptance = full_report["acceptance"]
    assert acceptance["status"] == "PASS", acceptance["failed_checks"]
    assert acceptance["failed_checks"] == []
    assert acceptance["checks"]["legacy_baseline_available"] is True
    assert acceptance["checks"]["every_enabled_family_has_valid_candidates"] is True
    assert acceptance["checks"]["all_candidates_collision_free"] is True
    assert acceptance["checks"]["gravity_improvement_over_legacy_found"] is True
    assert acceptance["checks"]["top_n_paths_planned_collision_free"] is True
    assert acceptance["checks"]["top_n_paths_within_time_scaling_limits"] is True
    assert acceptance["checks"]["no_j4_hard_limit_defined"] is True
    improvement = acceptance["gravity_improvement"]
    assert improvement["qualifying_candidate_count"] >= 1
    assert improvement["candidate"]["j4_torque_reduction_ratio"] > 0.0
    assert full_report["counts"]["candidates_planned"] >= full_report["search_parameters"]["planning"]["top_k"]


def test_oblique_produces_valid_candidates_at_multiple_angles(full_report) -> None:
    labels = {
        candidate["label"].split("_r")[0]
        for candidate in full_report["candidates"]
        if candidate["family"] == "oblique"
    }
    assert {"oblique_30", "oblique_40", "oblique_50", "oblique_60"} <= labels


def test_top_n_is_deterministic(quick_config) -> None:
    determinism = verify_determinism(quick_config, runs=2)
    assert determinism["top_n_identical"] is True
    assert determinism["scores_identical"] is True
    assert determinism["deterministic"] is True
    assert len(determinism["signatures"][0]) == quick_config.top_n


def test_search_is_repeatable_on_one_instance(quick_search) -> None:
    first = quick_search.run()
    second = quick_search.run()
    assert [item["candidate_id"] for item in first["top_n"]] == [
        item["candidate_id"] for item in second["top_n"]
    ]
    assert first["filter_counts"] == second["filter_counts"]
    assert first["counts"] == second["counts"]


def test_shortlist_is_diverse_and_excludes_the_legacy_family(quick_report, quick_config) -> None:
    top_n = quick_report["top_n"]
    assert len(top_n) == quick_config.top_n
    for item in top_n:
        assert item["family"] != "legacy_pi2"
    shortlist = quick_report["shortlist"]
    if not shortlist["diversity_relaxed"]:
        axes = [np.asarray(item["tcp_pose"]["approach_axis_world"], dtype=float) for item in top_n]
        for index, first in enumerate(axes):
            for second in axes[index + 1 :]:
                angle = math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0))))
                assert angle >= shortlist["minimum_approach_axis_angle_deg"] - 1e-6
    # Everything the shortlist skipped is still published with its metrics.
    ranking_ids = {item["candidate_id"] for item in quick_report["ranking"]}
    for candidate_id in shortlist["omitted_as_too_similar"]:
        assert candidate_id in ranking_ids


def test_select_top_n_falls_back_when_diversity_cannot_be_met(quick_config) -> None:
    def candidate(index: int) -> Candidate:
        return _synthetic_candidate(f"c{index}", np.zeros(7), margin=0.2, j4_margin=0.2, torque_j4=-1.0,
                                    axis=[0.0, 0.0, -1.0])

    ranked = []
    for index in range(4):
        item = candidate(index)
        item.trajectory_score = {"score": -float(index)}
        ranked.append(item)
    picked, omitted, relaxed = select_top_n(quick_config, ranked, None)
    assert relaxed is True
    assert len(picked) == quick_config.top_n
    # Nothing is silently dropped: every ranked candidate is either picked or listed.
    assert len(picked) + len(omitted) == len(ranked)
    assert {item.candidate_id for item in picked} | {item.candidate_id for item in omitted} == {
        item.candidate_id for item in ranked
    }


def test_report_publishes_every_required_top_n_field(quick_report) -> None:
    required = {
        "family",
        "label",
        "tcp_pose",
        "q_rad",
        "j4_rad",
        "joint_margin_normalized",
        "joint_margin_rad",
        "j4_margin_rad",
        "j4_margin_normalized",
        "manipulability",
        "minimum_clearance_m",
        "static_gravity_torque_j4_nm",
        "static_gravity_torque_max_all_joints_nm",
        "trajectory_maximum_absolute_j4_gravity_torque_nm",
        "trajectory_score",
    }
    for item in quick_report["top_n"]:
        assert required <= set(item)
        assert len(item["q_rad"]) == 7
        assert item["tcp_pose"]["position_world_m"]
        assert item["trajectory_planned"] is True


def test_report_is_json_serializable_and_has_relative_acceptance(quick_report) -> None:
    text = json.dumps(quick_report)
    reloaded = json.loads(text)
    assert reloaded["scope"]["mode"] == "simulation_only"
    assert reloaded["scope"]["hardware_access"] == "forbidden"
    assert reloaded["scope"]["executes_motion"] is False
    assert reloaded["scope"]["j4_hard_limit_defined"] is False
    acceptance = reloaded["acceptance"]
    assert "relative" in acceptance["policy"]
    # The real observation is recorded as an observation, with its policy attached.
    observation = reloaded["known_bad_observations"][0]
    assert observation["j4_target_rad"] == pytest.approx(1.8984)
    assert observation["j4_final_rad"] == pytest.approx(1.4120)
    assert "observation" in observation["policy"].lower()
    # No acceptance key defines an absolute J4 bound.
    for key in acceptance:
        assert "j4_hard_limit" not in key


def test_planning_queue_is_bounded(quick_report, quick_config) -> None:
    planned = quick_report["counts"]["candidates_planned"]
    # baseline + one per orientation label + the top_k by static score, all distinct.
    assert planned <= 1 + len(quick_report["orientation_candidates"]) + quick_config.top_k
    assert planned >= quick_config.top_k


# --------------------------------------------------------------------------------------
# Configuration contract and hardware isolation
# --------------------------------------------------------------------------------------


def test_ready_search_config_is_validated_by_the_project_loader() -> None:
    raw = load_yaml("ready_search.yaml")
    validate_ready_search(raw)
    assert raw["scope"]["mode"] == "simulation_only"
    assert raw["orientations"]["tilt_convention"] == "angle_below_horizontal_deg"


def test_config_rejects_j4_limits_and_weakened_scope() -> None:
    raw = load_yaml("ready_search.yaml")

    doctored = json.loads(json.dumps(raw))
    doctored["acceptance"]["j4_hard_limit_rad"] = 1.0
    with pytest.raises(ConfigError, match="J4 hard limit"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    doctored["acceptance"]["j4_maximum_torque_threshold_nm"] = 3.0
    with pytest.raises(ConfigError, match="J4 hard limit"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    doctored["scope"]["hardware_access"] = "allowed"
    with pytest.raises(ConfigError, match="hardware_access"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    doctored["scope"]["mode"] = "real"
    with pytest.raises(ConfigError, match="simulation_only"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    doctored["orientations"]["tilt_convention"] = "world_rpy"
    with pytest.raises(ConfigError, match="tilt_convention"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    doctored["scoring"]["static"]["weights"]["legacy_reference_distance_normalized"] = 0.1
    with pytest.raises(ConfigError, match="legacy_reference_distance_normalized"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    for family in doctored["orientations"]["families"]:
        if family["name"] == "legacy_pi2":
            family["enabled"] = False
    with pytest.raises(ConfigError, match="legacy_pi2"):
        validate_ready_search(doctored)

    doctored = json.loads(json.dumps(raw))
    doctored["known_bad_observations"][0]["policy"] = "J4 must stay below this value"
    with pytest.raises(ConfigError, match="policy"):
        validate_ready_search(doctored)


def test_refinement_offsets_are_ordered_and_include_the_center() -> None:
    offsets = refinement_offsets(np.zeros(3), 0.02, 0.02)
    assert np.allclose(offsets[0], 0.0)
    assert len(offsets) == 27
    norms = np.abs(offsets).sum(axis=1)
    assert np.all(np.diff(norms) >= 0.0)
    assert np.array_equal(offsets, refinement_offsets(np.zeros(3), 0.02, 0.02))
    # The window is respected and a wider window produces more offsets.
    wider = refinement_offsets(np.zeros(3), 0.04, 0.02)
    assert len(wider) == 125
    assert float(np.max(np.abs(wider))) <= 0.04 + 1e-9


FORBIDDEN_MODULE_PREFIXES = (
    "rclpy",
    "moveit",
    "moveit_msgs",
    "rcl_interfaces",
    "ros2",
    "can",
    "canopen",
    "serial",
    "pyrealsense2",
    "openarm_hardware",
    "openarm_kinesthetic_recorder",
)
FORBIDDEN_DOTTED_SUFFIXES = (
    "initialization.execute",
    "teaching.session",
    "real.operator_protocol",
    "initialization.pipeline.execute_initialization_trajectory",
)


def test_ready_search_package_never_touches_hardware_or_execution_modules() -> None:
    imported: set[str] = set()
    source_text: list[str] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        source_text.append(text)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = "." * node.level
                imported.add(f"{level}{module}")
                imported.update(f"{module}.{alias.name}" for alias in node.names)
    for name in imported:
        assert not name.startswith(FORBIDDEN_MODULE_PREFIXES), f"forbidden import: {name}"
        assert not name.endswith(FORBIDDEN_DOTTED_SUFFIXES), f"forbidden import: {name}"
    joined = "\n".join(source_text)
    for forbidden in ("execute_initialization_trajectory", "execute_trajectory", "subprocess", "os.system"):
        assert forbidden not in joined, f"forbidden reference: {forbidden}"


def test_importing_the_package_pulls_in_no_ros_modules() -> None:
    """Checked in a fresh interpreter: the rest of the suite imports the ROS 2 packages."""
    probe = (
        "import json, sys\n"
        "import openarm_kinesthetic.ready_search as module\n"
        "forbidden = " + repr(FORBIDDEN_MODULE_PREFIXES) + "\n"
        "found = sorted(name for name in sys.modules if name.startswith(forbidden))\n"
        "print(json.dumps({'forbidden': found, 'exported': len(module.__all__)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["forbidden"] == []
    assert payload["exported"] > 0


def test_cli_runs_quick_profile_without_writing(capsys) -> None:
    from openarm_kinesthetic.ready_search.__main__ import main

    main(["--quick", "--no-write"])
    output = capsys.readouterr().out
    assert "TEACHING READY POSE SEARCH" in output
    assert "acceptance PASS" in output
    assert "FAMILY COMPARISON" in output


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _synthetic_candidate(
    candidate_id: str,
    q: np.ndarray,
    *,
    margin: float,
    j4_margin: float,
    torque_j4: float,
    axis: list[float] | None = None,
) -> Candidate:
    metrics = {
        "minimum_joint_margin_normalized": margin,
        "j4_margin_normalized": j4_margin,
        "yoshikawa_manipulability": 0.02,
        "minimum_clearance_m": 0.05,
        "minimum_task_obstacles_clearance_m": 0.05,
        "minimum_structure_clearance_m": 0.05,
        "minimum_self_clearance_m": 0.05,
        "start_joint_distance_rad": 1.0,
        "legacy_reference_distance_rad": 0.0,
        "gravity_torque_j4_nm": torque_j4,
        "absolute_gravity_torque_j4_nm": abs(torque_j4),
        "maximum_absolute_gravity_torque_nm": abs(torque_j4),
    }
    return Candidate(
        candidate_id=candidate_id,
        family="synthetic",
        label=candidate_id,
        pass_name="test",
        orientation={"approach_axis_world": list(axis or [0.0, 0.0, -1.0])},
        target=TargetPosition(
            offset_m=(0.0, 0.0, 0.0), world_m=(0.3, 0.0, 0.7), table_xy_m=(0.0, 0.0), height_above_table_m=0.28
        ),
        seed_index=0,
        q=np.asarray(q, dtype=float),
        ik={"position_error_m": 0.0, "orientation_error_rad": 0.0, "iterations": 1},
        static_metrics=metrics,
    )


def test_apply_candidate_places_the_marker_on_the_tcp(quick_search) -> None:
    """The visual comparison is only meaningful if the marker really is the tool frame."""
    import mujoco

    from openarm_kinesthetic.ready_search.tool_frame import derive_approach_axis
    from openarm_kinesthetic.ready_search.visualize import MARKER_GEOM_GROUP, apply_candidate, scene_option

    search = quick_search
    q = RECORDED_SCAN_Q
    body = name_id(search.model, mujoco.mjtObj.mjOBJ_BODY, "right_handle_target")
    mocap = int(search.model.body_mocapid[body])
    site = name_id(search.model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)

    apply_candidate(search.model, search.data, search.mapping, q)
    assert np.allclose(search.data.mocap_pos[mocap], search.data.site_xpos[site], atol=1e-12)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(search.data.site_xmat[site], dtype=float).reshape(9))
    assert np.allclose(search.data.mocap_quat[mocap], quaternion, atol=1e-12)

    # With an offset the marker moves along the tool approach axis, not sideways.
    offset = 0.06
    apply_candidate(search.model, search.data, search.mapping, q, marker_offset_along_approach_m=offset)
    axis, _, _ = derive_approach_axis(search.model, site)
    expected = np.asarray(
        [search.data.site_xpos[site][i] for i in range(3)]
    ) + offset * (np.asarray(search.data.site_xmat[site], dtype=float).reshape(3, 3) @ axis)
    assert np.allclose(search.data.mocap_pos[mocap], expected, atol=1e-12)

    option = scene_option()
    assert option.geomgroup[MARKER_GEOM_GROUP] == 1
    # The groups the scene normally renders stay enabled.
    assert list(option.geomgroup[:3]) == [1, 1, 1]


def test_candidate_switching_keeps_the_fixed_tcp_and_changes_orientation(quick_search, quick_report) -> None:
    """The marker is the true fixed TCP; candidates differ through its frame axes."""
    import mujoco

    from openarm_kinesthetic.ready_search.visualize import apply_candidate

    search = quick_search
    body = name_id(search.model, mujoco.mjtObj.mjOBJ_BODY, "right_handle_target")
    mocap = int(search.model.body_mocapid[body])
    poses = []
    for item in quick_report["top_n"]:
        apply_candidate(search.model, search.data, search.mapping, item["q_rad"])
        poses.append(
            (
                np.asarray(search.data.mocap_pos[mocap], dtype=float).copy(),
                np.asarray(search.data.mocap_quat[mocap], dtype=float).copy(),
            )
        )
    assert len(poses) >= 2
    for index, (first_position, first_quaternion) in enumerate(poses):
        for second_position, second_quaternion in poses[index + 1 :]:
            assert float(np.linalg.norm(first_position - second_position)) < 0.005
            assert abs(float(first_quaternion @ second_quaternion)) < 0.999
