"""Report assembly, acceptance checks and the human-readable comparison table.

The acceptance block is deliberately relative. It answers questions like "did every
enabled family produce a valid candidate", "do the families land on clearly different
J4 values", and "is there a candidate that is both better placed in J4 and less
loaded in J4 than the configuration that failed on hardware". It never answers "is J4
below 1.0 rad", because no measurement supports that number.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import ReadySearchConfig
from .search import (
    Candidate,
    reference_orientation_summary,
    top_n_signature,
)
from .tool_frame import ToolFrame


def select_top_n(
    config: ReadySearchConfig,
    ranked: Sequence[Candidate],
    baseline: Candidate | None = None,
) -> tuple[list[Candidate], list[Candidate], bool]:
    """Pick the published shortlist, preferring genuinely different tool directions.

    Returns ``(top_n, omitted_as_similar, diversity_relaxed)``. The shortlist exists so
    a human can compare candidates by eye and by hand, so three near-copies of the same
    pose are worth less than three poses that differ in where the tool points. If the
    diversity rule cannot fill the shortlist, the remaining slots are filled by score
    and ``diversity_relaxed`` is set rather than silently returning a shorter list.
    """
    size = int(config.top_n)
    diversity = config.report.get("diversity", {})
    excluded_families = set(diversity.get("exclude_families", ()) or ())
    pool = [candidate for candidate in ranked if candidate.family not in excluded_families]
    if not bool(diversity.get("enabled", False)) or size >= len(pool):
        return list(pool[:size]), [], False

    minimum_angle = float(diversity["minimum_approach_axis_angle_deg"])
    picked: list[Candidate] = []
    omitted: list[Candidate] = []
    for candidate in pool:
        if len(picked) >= size:
            break
        if all(_axis_angle_deg(candidate, chosen) >= minimum_angle for chosen in picked):
            picked.append(candidate)
        else:
            omitted.append(candidate)
    relaxed = False
    if len(picked) < size:
        relaxed = True
        for candidate in omitted:
            if len(picked) >= size:
                break
            picked.append(candidate)
        picked_ids = {candidate.candidate_id for candidate in picked}
        omitted = [candidate for candidate in omitted if candidate.candidate_id not in picked_ids]
    picked.sort(key=lambda item: (-item.trajectory_score["score"], item.candidate_id))
    return picked, omitted, relaxed


def _axis_angle_deg(first: Candidate, second: Candidate) -> float:
    first_axis = np.asarray(first.orientation["approach_axis_world"], dtype=float)
    second_axis = np.asarray(second.orientation["approach_axis_world"], dtype=float)
    cosine = float(np.clip(first_axis @ second_axis, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def build_report(
    *,
    config: ReadySearchConfig,
    search,
    geometry,
    frame: ToolFrame,
    candidates: Sequence[Candidate],
    baseline: Candidate | None,
    static_references: Mapping[str, Any],
    trajectory_references: Mapping[str, Any],
    duplicate_count: int,
    plan_counts: Mapping[str, int],
    refine_details: Sequence[Mapping[str, Any]],
    refine_seed_ids: Iterable[str],
) -> dict[str, Any]:
    digits = int(config.report["round_floats_to"])
    planned = [candidate for candidate in candidates if candidate.trajectory is not None]
    ranked = sorted(
        (candidate for candidate in planned if candidate.trajectory_score is not None),
        key=lambda item: (-item.trajectory_score["score"], item.candidate_id),
    )
    top_n, omitted_as_similar, diversity_relaxed = select_top_n(config, ranked, baseline)
    for rank, candidate in enumerate(top_n, start=1):
        candidate.reference_role = f"top_{rank}"
    ranking_size = min(int(config.report.get("ranking_size", config.top_n)), len(ranked))

    acceptance = acceptance_block(config, candidates, baseline, top_n)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "teaching_ready_pose_search",
        "chain": "teaching_ready_se3_pose_search",
        "scope": {
            **{key: value for key, value in config.raw["scope"].items()},
            "executes_motion": False,
            "imports_ros_or_hardware_modules": False,
            "j4_hard_limit_defined": False,
            "note": (
                "Simulation decision support only. The output is a ranked list of candidate "
                "poses for a later, separately designed ROS 2 / MoveIt port; nothing here "
                "commands an arm."
            ),
        },
        "scene": {
            "path": config.scene,
            "source": "ready_search.yaml `scene`, else initialization.yaml `initialization.scene`",
        },
        "table": _table_summary(geometry),
        "anchor": anchor_summary(search),
        "tool_frame": frame.as_dict(),
        "reference_orientation": reference_orientation_summary(frame, search.reference_matrix),
        "clearance_pairs": {
            "task_obstacle_pair_count": len(search.clearance["task_obstacles"]),
            "structure_pair_count": len(search.clearance["structure"]),
            "self_pair_count": len(search.clearance["self"]),
            "right_collision_geom_count": len(search.clearance["right_geom_ids"]),
            "task_obstacle_geom_count": len(search.clearance["task_obstacle_geom_ids"]),
            "structure_geom_count": len(search.clearance["structure_geom_ids"]),
            "scored_group": str(search.clearance["scored_group"]),
            "excluded_pairs": list(search.clearance["excluded_pairs"]),
            "exclusion_rule": str(search.clearance["exclusion_rule"]),
            "reference_contact_tolerance_m": float(search.clearance["reference_contact_tolerance_m"]),
            "note": (
                "Task obstacles are box geoms, so their distance is exact and they are what the score "
                "consumes. Structure and self pairs are mesh-against-mesh: mj_geomDistance works on "
                "convex hulls, so their values are conservative lower bounds that can read zero without "
                "contact. Real contact is enforced by the collision filter instead."
            ),
        },
        "start_q_rad": np.asarray(search.start_q, dtype=float).tolist(),
        "start_state_source": "initialization.yaml simulation.initial_pose (physical vertical-down motor_zero)",
        "families": [family.as_dict() for family in config.families],
        "orientation_candidates": [item.as_dict() for item in search.orientations],
        "search_parameters": {
            "coarse_offsets_m": np.asarray(config.coarse_offsets_m, dtype=float).tolist(),
            "refine": {
                "enabled": config.refine_enabled,
                "seed_cells": config.refine_seed_cells,
                "window_m": config.refine_window_m,
                "step_m": config.refine_step_m,
                "roll_offsets_deg": list(config.refine_roll_offsets_deg),
            },
            "footprint_margin_m": config.footprint_margin_m,
            "height_band_above_table_m": [
                config.minimum_height_above_table_m,
                config.maximum_height_above_table_m,
            ],
            "ik": dict(config.ik),
            "ik_seeds": [np.asarray(seed, dtype=float).tolist() for seed in search.seeds],
            "filters": dict(config.filters),
            "metrics": dict(config.metrics),
            "planning": {
                "top_k": config.top_k,
                "maximum_iterations": int(config.raw["planning"]["maximum_iterations"]),
                "planner": "rrc_connect (project's existing collision-checked implementation)",
            },
            "dedup_max_joint_distance_rad": float(config.ik["dedup_max_joint_distance_rad"]),
        },
        "scoring": {
            "static_weights": dict(config.static_weights),
            "trajectory_weights": dict(config.trajectory_weights),
            "trajectory_endpoint_weight": config.trajectory_endpoint_weight,
        },
        "resolved_references": {
            "static": static_references,
            "trajectory": trajectory_references,
            "note": (
                "A `legacy_baseline` reference is the value measured on the legacy baseline in this "
                "same run, so every score is a relative statement about this station rather than an "
                "absolute threshold."
            ),
        },
        "legacy_baseline": baseline_summary(baseline),
        "counts": {
            "candidates_valid": len(candidates),
            "candidates_planned": len(planned),
            "plans_succeeded": int(plan_counts.get("planned", 0)),
            "plans_failed": int(plan_counts.get("failed", 0)),
            "duplicates_collapsed": int(duplicate_count),
            "orientation_candidate_count": len(search.orientations),
            "position_candidate_count": len(config.coarse_offsets_m),
            "refined_cells": len(list(refine_details)),
        },
        "filter_counts": dict(sorted(search.state.filter_counts.items())),
        "rejected_targets": list(search.state.rejected_targets),
        "refinement": list(refine_details),
        "refinement_seed_candidate_ids": sorted(refine_seed_ids),
        "candidates": [candidate.as_dict(digits=digits) for candidate in candidates],
        "planned_candidates": [
            candidate.as_dict(digits=digits) for candidate in sorted(planned, key=lambda item: item.candidate_id)
        ],
        "ranking": [candidate.summary(digits=digits) for candidate in ranked[:ranking_size]],
        "top_n": [candidate.summary(digits=digits) for candidate in top_n],
        "orientation_comparison": orientation_comparison(planned),
        "shortlist": {
            "rule": (
                "best trajectory score first, then the best remaining candidate whose tool approach axis "
                "differs from every already-picked candidate by at least the configured angle"
            ),
            "minimum_approach_axis_angle_deg": float(
                config.report.get("diversity", {}).get("minimum_approach_axis_angle_deg", 0.0)
            ),
            "diversity_enabled": bool(config.report.get("diversity", {}).get("enabled", False)),
            "excluded_families": sorted(config.report.get("diversity", {}).get("exclude_families", ()) or ()),
            "diversity_relaxed": bool(diversity_relaxed),
            "omitted_as_too_similar": [candidate.candidate_id for candidate in omitted_as_similar],
            "note": (
                "The shortlist is what a human compares in the viewer; `ranking` is the pure score order "
                "and contains the candidates omitted here only for being too similar."
            ),
        },
        "family_comparison": family_comparison(config, candidates, baseline),
        "acceptance": acceptance,
        "known_bad_observations": [dict(item) for item in config.known_bad_observations],
        "determinism": None,
        "notes": [
            "J4 gravity torque is the zero-velocity inverse-dynamics torque of the compiled "
            "openarm_right_joint4, in N*m, the same convention the hardware passive/gravity-"
            "compensation path uses.",
            "Candidate scores are reproducible from the stored raw metrics and weights; no metric "
            "is recomputed from a rounded value.",
        ],
    }
    return report


def orientation_comparison(planned: Sequence[Candidate]) -> dict[str, Any]:
    """One row per planned configuration: the direct answer to "which tilt is better".

    Every row is a candidate that was actually planned from the measured start state, so
    the J4 columns compare like with like: static gravity torque at the endpoint, and
    the worst gravity torque anywhere along the initialization path.
    """
    rows: list[dict[str, Any]] = []
    for candidate in sorted(planned, key=lambda item: (item.label, item.candidate_id)):
        trajectory = candidate.trajectory or {}
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "family": candidate.family,
                "reference_role": candidate.reference_role,
                "tilt_below_horizontal_deg": candidate.orientation["tilt_below_horizontal_deg"],
                "heading_deg": candidate.orientation["heading_deg"],
                "roll_deg": candidate.orientation["roll_deg"],
                "tcp_position_world_m": list(candidate.target.world_m),
                "j4_rad": float(candidate.static_metrics["j4_rad"]),
                "j4_margin_normalized": float(candidate.static_metrics["j4_margin_normalized"]),
                "minimum_joint_margin_normalized": float(
                    candidate.static_metrics["minimum_joint_margin_normalized"]
                ),
                "manipulability": float(candidate.static_metrics["yoshikawa_manipulability"]),
                "minimum_clearance_m": float(candidate.static_metrics["minimum_clearance_m"]),
                "minimum_task_obstacles_clearance_m": float(
                    candidate.static_metrics["minimum_task_obstacles_clearance_m"]
                ),
                "static_gravity_torque_j4_nm": float(candidate.static_metrics["gravity_torque_j4_nm"]),
                "static_gravity_torque_max_all_joints_nm": float(
                    candidate.static_metrics["maximum_absolute_gravity_torque_nm"]
                ),
                "planned": bool(trajectory.get("success")),
                "trajectory_joint_path_length_rad": trajectory.get("joint_path_length_rad"),
                "trajectory_j4_travel_rad": trajectory.get("j4_travel_rad"),
                "trajectory_maximum_joint_travel_rad": trajectory.get("maximum_joint_travel_rad"),
                "trajectory_duration_s": trajectory.get("duration_s"),
                "trajectory_maximum_velocity_rad_s": trajectory.get("maximum_velocity_rad_s"),
                "trajectory_maximum_acceleration_rad_s2": trajectory.get("maximum_acceleration_rad_s2"),
                "trajectory_minimum_clearance_m": trajectory.get("minimum_clearance_m"),
                "trajectory_minimum_task_obstacles_clearance_m": trajectory.get(
                    "minimum_task_obstacles_clearance_m"
                ),
                "trajectory_maximum_absolute_j4_gravity_torque_nm": trajectory.get(
                    "maximum_absolute_j4_gravity_torque_nm"
                ),
                "trajectory_mean_absolute_j4_gravity_torque_nm": trajectory.get(
                    "mean_absolute_j4_gravity_torque_nm"
                ),
                "trajectory_endpoint_j4_gravity_torque_nm": trajectory.get("endpoint_j4_gravity_torque_nm"),
                "static_score": None if candidate.static_score is None else float(candidate.static_score["score"]),
                "trajectory_score": None
                if candidate.trajectory_score is None
                else float(candidate.trajectory_score["score"]),
            }
        )
    return {
        "rows": rows,
        "note": (
            "`trajectory_maximum_absolute_j4_gravity_torque_nm` is the largest |tau_J4| anywhere along "
            "the path, which catches a candidate whose endpoint is light but whose mid-path is not."
        ),
    }


def render_orientation_table(report: Mapping[str, Any]) -> str:
    rows = report.get("orientation_comparison", {}).get("rows", [])
    header = (
        f"{'label':30} {'tilt':>6} {'head':>6} {'roll':>6} {'J4 rad':>8} {'tauJ4':>8} "
        f"{'traj|tauJ4|':>12} {'traj len':>9} {'traj score':>11}"
    )
    lines = ["PLANNED CONFIGURATION COMPARISON (every configuration that was planned)", header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row['label']):30} {row['tilt_below_horizontal_deg']:>6.1f} {row['heading_deg']:>6.1f} "
            f"{row['roll_deg']:>6.1f} {row['j4_rad']:>8.4f} {_fmt(row['static_gravity_torque_j4_nm']):>8} "
            f"{_fmt(row['trajectory_maximum_absolute_j4_gravity_torque_nm']):>12} "
            f"{_fmt(row['trajectory_joint_path_length_rad']):>9} {_fmt(row['trajectory_score']):>11}"
        )
    return "\n".join(lines)


def _table_summary(geometry) -> dict[str, Any]:
    return {
        "center_xyz_m": np.asarray(geometry.center, dtype=float).tolist(),
        "size_xy_m": np.asarray(geometry.size, dtype=float).tolist(),
        "top_z_m": float(geometry.top_z),
        "yaw_rad": float(geometry.yaw),
        "provenance": dict(sorted(geometry.provenance.items())),
    }


def anchor_summary(search) -> dict[str, Any]:
    offset_x, offset_y, height = search.config.table_anchor_offsets()
    return {
        "source": search.config.position_anchor_source,
        "legacy_offset_x_m": float(offset_x),
        "legacy_offset_y_m": float(offset_y),
        "height_above_table_m": float(height),
        "world_m": np.asarray(search.anchor_m, dtype=float).tolist(),
        "legacy_world_m": np.asarray(search.legacy_anchor_m, dtype=float).tolist(),
        "movement_from_legacy_m": (
            np.asarray(search.anchor_m, dtype=float)
            - np.asarray(search.legacy_anchor_m, dtype=float)
        ).tolist(),
        "note": (
            "New candidates search around world_m. legacy_world_m is retained only "
            "for the previous fixed-pose comparison baseline."
        ),
    }


def baseline_summary(baseline: Candidate | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    metrics = baseline.static_metrics
    payload = {
        "selection_rule": (
            "legacy_pi2 family at zero anchor offset, largest normalized joint-limit margin: the rule "
            "the existing offset scan uses, so this is the pose the real arm actually attempted"
        ),
        "candidate_id": baseline.candidate_id,
        "q_rad": baseline.q.tolist(),
        "j4_rad": float(metrics["j4_rad"]),
        "j4_margin_rad": float(metrics["j4_margin_rad"]),
        "j4_margin_normalized": float(metrics["j4_margin_normalized"]),
        "minimum_joint_margin_normalized": float(metrics["minimum_joint_margin_normalized"]),
        "manipulability": float(metrics["yoshikawa_manipulability"]),
        "minimum_clearance_m": float(metrics["minimum_clearance_m"]),
        "minimum_task_obstacles_clearance_m": float(metrics["minimum_task_obstacles_clearance_m"]),
        "static_gravity_torque_j4_nm": float(metrics["gravity_torque_j4_nm"]),
        "static_absolute_gravity_torque_j4_nm": float(metrics["absolute_gravity_torque_j4_nm"]),
        "static_gravity_torque_max_all_joints_nm": float(metrics["maximum_absolute_gravity_torque_nm"]),
        "static_score": None if baseline.static_score is None else float(baseline.static_score["score"]),
        "trajectory": baseline.trajectory,
        "trajectory_score": None
        if baseline.trajectory_score is None
        else float(baseline.trajectory_score["score"]),
        "role": "relative comparison baseline only; not a limit",
    }
    return payload


def family_comparison(
    config: ReadySearchConfig,
    candidates: Sequence[Candidate],
    baseline: Candidate | None,
) -> dict[str, Any]:
    """The table that answers "which family is better placed AND less loaded in J4"."""
    rows: list[dict[str, Any]] = []
    for family in config.enabled_families():
        members = [
            candidate
            for candidate in candidates
            if candidate.family == family.name and candidate.static_score is not None
        ]
        if not members:
            rows.append({"family": family.name, "valid_candidates": 0, "best_candidate": None})
            continue
        best = min(
            members,
            key=lambda item: (
                -float(item.static_score["score"]),
                abs(float(item.static_metrics["gravity_torque_j4_nm"])),
                item.candidate_id,
            ),
        )
        best_torque = min(members, key=lambda item: abs(float(item.static_metrics["gravity_torque_j4_nm"])))
        metrics = best.static_metrics
        torque_metrics = best_torque.static_metrics
        rows.append(
            {
                "family": family.name,
                "valid_candidates": len(members),
                "j4_rad_min": float(min(item.static_metrics["j4_rad"] for item in members)),
                "j4_rad_max": float(max(item.static_metrics["j4_rad"] for item in members)),
                "best_candidate": {
                    "candidate_id": best.candidate_id,
                    "label": best.label,
                    "tcp_position_world_m": list(best.target.world_m),
                    "q_rad": best.q.tolist(),
                    "j4_rad": float(metrics["j4_rad"]),
                    "j4_margin_normalized": float(metrics["j4_margin_normalized"]),
                    "minimum_joint_margin_normalized": float(metrics["minimum_joint_margin_normalized"]),
                    "manipulability": float(metrics["yoshikawa_manipulability"]),
                    "minimum_clearance_m": float(metrics["minimum_clearance_m"]),
                    "static_gravity_torque_j4_nm": float(metrics["gravity_torque_j4_nm"]),
                    "static_score": float(best.static_score["score"]),
                    "trajectory_maximum_absolute_j4_gravity_torque_nm": None
                    if best.trajectory is None
                    else best.trajectory.get("maximum_absolute_j4_gravity_torque_nm"),
                    "trajectory_score": None
                    if best.trajectory_score is None
                    else float(best.trajectory_score["score"]),
                },
                "lowest_static_j4_torque_candidate": {
                    "candidate_id": best_torque.candidate_id,
                    "label": best_torque.label,
                    "j4_rad": float(torque_metrics["j4_rad"]),
                    "static_gravity_torque_j4_nm": float(torque_metrics["gravity_torque_j4_nm"]),
                },
            }
        )

    legacy_torque = None
    if baseline is not None:
        legacy_torque = abs(float(baseline.static_metrics["gravity_torque_j4_nm"]))
    return {
        "rows": rows,
        "legacy_baseline_absolute_static_j4_torque_nm": legacy_torque,
        "interpretation": (
            "`j4_rad` spread across families shows the orientation family - not the position grid - "
            "is what decides the J4 configuration. Torque columns are MuJoCo gravity torque, the "
            "quantity the real arm cannot report separately from joint error."
        ),
    }


def acceptance_block(
    config: ReadySearchConfig,
    candidates: Sequence[Candidate],
    baseline: Candidate | None,
    top_n: Sequence[Candidate],
) -> dict[str, Any]:
    acceptance = config.acceptance
    minimum_per_family = int(acceptance["minimum_valid_candidates_per_enabled_family"])
    per_family = {
        family.name: sum(1 for candidate in candidates if candidate.family == family.name)
        for family in config.enabled_families()
    }
    families_satisfied = all(count >= minimum_per_family for count in per_family.values()) and bool(per_family)

    j4_values = [float(candidate.static_metrics["j4_rad"]) for candidate in candidates]
    spread = (max(j4_values) - min(j4_values)) if j4_values else 0.0
    spread_satisfied = bool(j4_values) and spread >= float(acceptance["minimum_j4_spread_rad"])

    penetration_tolerance = float(acceptance["penetration_tolerance_m"])
    worst_clearance = (
        min(float(candidate.static_metrics["minimum_clearance_m"]) for candidate in candidates)
        if candidates
        else None
    )
    collision_free = worst_clearance is not None and worst_clearance >= -penetration_tolerance

    improvement = gravity_improvement(config, candidates, baseline)
    improvement_satisfied = improvement["candidate"] is not None

    trajectory_clearances = [
        float(candidate.trajectory["minimum_clearance_m"])
        for candidate in top_n
        if candidate.trajectory is not None and candidate.trajectory.get("minimum_clearance_m") is not None
    ]
    top_paths_planned = all(
        candidate.trajectory is not None and candidate.trajectory.get("success") for candidate in top_n
    )
    top_paths_collision_free = bool(top_n) and top_paths_planned and all(
        value >= -penetration_tolerance for value in trajectory_clearances
    )
    within_limits = bool(top_n) and all(
        candidate.trajectory is not None and bool(candidate.trajectory.get("within_time_scaling_limits"))
        for candidate in top_n
    )

    checks = {
        "legacy_baseline_available": baseline is not None,
        "every_enabled_family_has_valid_candidates": families_satisfied,
        "families_produce_distinct_j4": spread_satisfied,
        "all_candidates_collision_free": bool(collision_free),
        "gravity_improvement_over_legacy_found": improvement_satisfied if bool(
            acceptance["require_gravity_improvement_over_legacy"]
        ) else True,
        "top_n_paths_planned_collision_free": top_paths_collision_free
        if bool(acceptance["require_top3_collision_free_paths"])
        else True,
        "top_n_paths_within_time_scaling_limits": within_limits
        if bool(acceptance["require_top3_within_time_scaling_limits"])
        else True,
        "no_j4_hard_limit_defined": True,
        "hardware_untouched": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "per_family_valid_candidates": per_family,
        "minimum_valid_candidates_per_enabled_family": minimum_per_family,
        "j4_spread_rad": spread,
        "minimum_j4_spread_rad": float(acceptance["minimum_j4_spread_rad"]),
        "j4_rad_range": [min(j4_values), max(j4_values)] if j4_values else None,
        "minimum_clearance_across_candidates_m": worst_clearance,
        "penetration_tolerance_m": penetration_tolerance,
        "gravity_improvement": improvement,
        "top_n_trajectory_minimum_clearance_m": trajectory_clearances,
        "checks": checks,
        "failed_checks": failed,
        "status": "PASS" if not failed else "FAIL",
        "determinism_check": "not_run_in_this_report (see verify_determinism)",
        "policy": (
            "Every check above is relative: a family versus another family, or a candidate versus the "
            "legacy baseline in the same run. No absolute J4 or torque bound is defined or implied."
        ),
    }


def gravity_improvement(
    config: ReadySearchConfig,
    candidates: Sequence[Candidate],
    baseline: Candidate | None,
) -> dict[str, Any]:
    """Best candidate that is both better placed and less loaded in J4 than the baseline.

    "Clearly better placed" is checked as "no worse by more than the configured
    tolerance". Requiring a strict margin *increase* would be the wrong test here: on
    hardware the legacy pose did not violate the J4 limit, it tripped a protection
    event, so the actionable improvement is the load, with the joint placement held at
    least as good.
    """
    acceptance = config.acceptance
    margin_regression = float(acceptance["maximum_j4_margin_regression_normalized"])
    torque_ratio = float(acceptance["minimum_j4_torque_reduction_ratio"])
    if baseline is None:
        return {
            "candidate": None,
            "reason": "no legacy baseline candidate was produced, so no relative comparison is possible",
            "maximum_j4_margin_regression_normalized": margin_regression,
            "minimum_j4_torque_reduction_ratio": torque_ratio,
        }
    baseline_margin = float(baseline.static_metrics["j4_margin_normalized"])
    baseline_torque = abs(float(baseline.static_metrics["gravity_torque_j4_nm"]))
    torque_ceiling = baseline_torque * (1.0 - torque_ratio)
    qualifying: list[Candidate] = []
    for candidate in candidates:
        if candidate.family == baseline.family:
            continue
        if candidate.static_score is None:
            continue
        margin = float(candidate.static_metrics["j4_margin_normalized"])
        torque = abs(float(candidate.static_metrics["gravity_torque_j4_nm"]))
        if margin >= baseline_margin - margin_regression and torque <= torque_ceiling:
            qualifying.append(candidate)
    chosen = (
        min(
            qualifying,
            key=lambda item: (
                -float(item.static_score["score"]),
                item.candidate_id,
            ),
        )
        if qualifying
        else None
    )
    payload: dict[str, Any] = {
        "maximum_j4_margin_regression_normalized": margin_regression,
        "minimum_j4_torque_reduction_ratio": torque_ratio,
        "baseline_j4_margin_normalized": baseline_margin,
        "baseline_absolute_static_j4_torque_nm": baseline_torque,
        "required_j4_torque_ceiling_nm": torque_ceiling,
        "qualifying_candidate_count": len(qualifying),
        "candidate": None,
        "reason": None,
    }
    if chosen is None:
        payload["reason"] = (
            "no non-legacy candidate cut J4 gravity torque by the required ratio while holding the "
            "J4 limit margin within the allowed regression"
        )
        return payload
    chosen_torque = abs(float(chosen.static_metrics["gravity_torque_j4_nm"]))
    payload["candidate"] = {
        "candidate_id": chosen.candidate_id,
        "family": chosen.family,
        "label": chosen.label,
        "q_rad": chosen.q.tolist(),
        "j4_rad": float(chosen.static_metrics["j4_rad"]),
        "j4_margin_normalized": float(chosen.static_metrics["j4_margin_normalized"]),
        "j4_margin_normalized_change": float(chosen.static_metrics["j4_margin_normalized"]) - baseline_margin,
        "absolute_static_j4_torque_nm": chosen_torque,
        "j4_torque_reduction_ratio": 1.0 - chosen_torque / max(baseline_torque, 1e-12),
        "static_score": float(chosen.static_score["score"]),
        "trajectory_maximum_absolute_j4_gravity_torque_nm": None
        if chosen.trajectory is None
        else chosen.trajectory.get("maximum_absolute_j4_gravity_torque_nm"),
    }
    return payload


# --------------------------------------------------------------------------------------
# Human-readable views
# --------------------------------------------------------------------------------------

_COLUMNS = (
    ("rank", 4),
    ("label", 22),
    ("J4 rad", 8),
    ("J4 marg", 8),
    ("jmarg", 7),
    ("manip", 8),
    ("tauJ4 Nm", 9),
    ("traj|tauJ4|", 12),
    ("traj len", 9),
    ("traj score", 11),
)


def render_comparison_table(report: Mapping[str, Any]) -> str:
    """Plain-text Top-N comparison, with the legacy baseline as the last row."""
    rows: list[tuple[str, ...]] = []
    for rank, item in enumerate(report.get("top_n", []), start=1):
        rows.append(
            (
                str(rank),
                str(item.get("label")),
                _fmt(item.get("j4_rad")),
                _fmt(item.get("j4_margin_normalized")),
                _fmt(item.get("joint_margin_normalized")),
                _fmt(item.get("manipulability")),
                _fmt(item.get("static_gravity_torque_j4_nm")),
                _fmt(item.get("trajectory_maximum_absolute_j4_gravity_torque_nm")),
                _fmt(item.get("trajectory_joint_path_length_rad")),
                _fmt(item.get("trajectory_score")),
            )
        )
    baseline = report.get("legacy_baseline")
    if baseline:
        rows.append(
            (
                "base",
                "legacy_pi2",
                _fmt(baseline.get("j4_rad")),
                _fmt(baseline.get("j4_margin_normalized")),
                _fmt(baseline.get("minimum_joint_margin_normalized")),
                _fmt(baseline.get("manipulability")),
                _fmt(baseline.get("static_gravity_torque_j4_nm")),
                _fmt((baseline.get("trajectory") or {}).get("maximum_absolute_j4_gravity_torque_nm")),
                _fmt((baseline.get("trajectory") or {}).get("joint_path_length_rad")),
                _fmt(baseline.get("trajectory_score")),
            )
        )
    header = " ".join(name.ljust(width) for name, width in _COLUMNS)
    rule = "-" * len(header)
    lines = ["TEACHING READY POSE SEARCH - Top candidates vs the legacy baseline", header, rule]
    for row in rows:
        lines.append(" ".join(value.ljust(width) for value, (_, width) in zip(row, _COLUMNS)))
    acceptance = report.get("acceptance", {})
    lines.append(rule)
    lines.append(f"acceptance: {acceptance.get('status')}  failed_checks={acceptance.get('failed_checks')}")
    lines.append(
        "columns: J4 marg / jmarg / manip are normalized; tauJ4 Nm is static gravity torque; "
        "traj|tauJ4| is the maximum along the planned path"
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def render_family_table(report: Mapping[str, Any]) -> str:
    comparison = report.get("family_comparison", {})
    lines = ["FAMILY COMPARISON", f"{'family':16} {'n':>4} {'J4 min':>8} {'J4 max':>8} {'tauJ4 Nm':>9} {'traj|tauJ4|':>12}"]
    for row in comparison.get("rows", []):
        best = row.get("best_candidate")
        if best is None:
            lines.append(f"{row['family']:16} {row['valid_candidates']:>4} {'-':>8} {'-':>8} {'-':>9} {'-':>12}")
            continue
        lines.append(
            f"{row['family']:16} {row['valid_candidates']:>4} {row['j4_rad_min']:>8.4f} {row['j4_rad_max']:>8.4f} "
            f"{best['static_gravity_torque_j4_nm']:>9.4f} "
            f"{_fmt(best['trajectory_maximum_absolute_j4_gravity_torque_nm']):>12}"
        )
    return "\n".join(lines)


def top_n_identity(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    return top_n_signature(report)
