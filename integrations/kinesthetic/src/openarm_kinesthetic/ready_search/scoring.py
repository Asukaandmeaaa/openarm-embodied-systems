"""Two-level scoring for Teaching Ready candidates.

Level 1 scores a static configuration. Level 2 scores a whole initialization
trajectory and adds the endpoint's level-1 score, because a pose is only useful if the
arm can get there.

Three design choices are deliberate:

* Every term is normalized into ``[0, 1]`` before its weight is applied, so the weights
  are readable numbers rather than unit conversions, and the raw values stay in the
  report. A reader can re-weight the terms offline without re-running the search.
* Penalty terms carry a negative weight. The sign is part of the term's meaning rather
  than of the weight, so a weight can be zeroed without flipping anything.
* Reference scales default to the legacy baseline *measured in the same run*. The point
  of this search is "which candidate loads J4 less than the configuration that failed
  on hardware", not "which candidate is below an absolute number".
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import MODE_CONFIGURED, MODE_LEGACY_BASELINE, Reference

STATIC_TERMS = (
    "joint_margin_normalized",
    "j4_margin_normalized",
    "manipulability_normalized",
    "collision_clearance_normalized",
    "start_joint_distance_normalized",
    "legacy_reference_distance_normalized",
    "gravity_torque_normalized",
)

TRAJECTORY_TERMS = (
    "joint_path_length_normalized",
    "maximum_joint_travel_normalized",
    "j4_travel_normalized",
    "maximum_absolute_j4_gravity_torque_normalized",
    "maximum_velocity_normalized",
    "maximum_acceleration_normalized",
    "path_clearance_deficit_normalized",
)

# How a raw trajectory metric becomes a penalty in [0, 1].
SCALED = "scaled"        # normalized = clamp01(raw / reference)
DEFICIT = "deficit"      # normalized = clamp01(1 - raw / reference)


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def resolve_references(
    references: Mapping[str, Reference],
    legacy_metrics: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Resolve every reference scale, recording where each value came from."""
    resolved: dict[str, Any] = {}
    for key, reference in references.items():
        value, origin, measured = reference.resolve(legacy_metrics)
        resolved[key] = {
            "value": float(value),
            "resolved_from": origin,
            "legacy_baseline_measured_value": measured,
            "configured": reference.as_dict(),
        }
    return resolved


def _scale(references: Mapping[str, Any], key: str) -> float:
    return float(references[key]["value"])


def static_score(
    metrics: Mapping[str, Any],
    weights: Mapping[str, float],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    """Level-1 score of one static configuration."""
    j4_torque = abs(float(metrics["absolute_gravity_torque_j4_nm"]))
    all_torque = abs(float(metrics["maximum_absolute_gravity_torque_nm"]))
    j4_normalized = clamp01(j4_torque / _scale(references, "gravity_torque_j4_nm"))
    all_normalized = clamp01(all_torque / _scale(references, "gravity_torque_max_all_nm"))

    components: dict[str, dict[str, Any]] = {
        "joint_margin_normalized": {
            "raw": float(metrics["minimum_joint_margin_normalized"]),
            "normalized": clamp01(float(metrics["minimum_joint_margin_normalized"])),
        },
        "j4_margin_normalized": {
            "raw": float(metrics["j4_margin_normalized"]),
            "normalized": clamp01(float(metrics["j4_margin_normalized"])),
        },
        "manipulability_normalized": {
            "raw": float(metrics["yoshikawa_manipulability"]),
            "normalized": clamp01(float(metrics["yoshikawa_manipulability"]) / _scale(references, "manipulability")),
        },
        "collision_clearance_normalized": {
            # Scored on the exact table/box/cube clearance. See metrics.clearance_pairs
            # for why the structure and self groups are diagnostics rather than scores.
            "raw": float(metrics["minimum_task_obstacles_clearance_m"]),
            "raw_structure_clearance_m": float(metrics["minimum_structure_clearance_m"]),
            "raw_self_clearance_m": float(metrics["minimum_self_clearance_m"]),
            "normalized": clamp01(
                float(metrics["minimum_task_obstacles_clearance_m"]) / _scale(references, "collision_clearance_m")
            ),
        },
        "start_joint_distance_normalized": {
            "raw": float(metrics["start_joint_distance_rad"]),
            "normalized": clamp01(
                float(metrics["start_joint_distance_rad"]) / _scale(references, "start_joint_distance_rad")
            ),
        },
        "legacy_reference_distance_normalized": {
            "raw": float(metrics["legacy_reference_distance_rad"]),
            "normalized": clamp01(
                float(metrics["legacy_reference_distance_rad"]) / _scale(references, "legacy_reference_distance_rad")
            ),
        },
        "gravity_torque_normalized": {
            "raw": max(j4_torque, all_torque),
            "normalized": max(j4_normalized, all_normalized),
            "detail": {
                "absolute_j4_nm": j4_torque,
                "absolute_j4_normalized": j4_normalized,
                "maximum_absolute_all_joints_nm": all_torque,
                "maximum_absolute_all_joints_normalized": all_normalized,
                "signed_j4_nm": float(metrics["gravity_torque_j4_nm"]),
            },
        },
    }

    score = 0.0
    for term, component in components.items():
        weight = float(weights.get(term, 0.0))
        component["weight"] = weight
        component["contribution"] = weight * float(component["normalized"])
        score += component["contribution"]
    return {
        "score": float(score),
        "components": components,
        "weights": {term: float(weights.get(term, 0.0)) for term in STATIC_TERMS},
    }


def trajectory_score(
    *,
    endpoint_static_score: float,
    trajectory_metrics: Mapping[str, Any],
    endpoint_weight: float,
    weights: Mapping[str, float],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    """Level-2 score: endpoint quality plus the cost of getting there."""
    raw_terms = {
        "joint_path_length_normalized": (
            "joint_path_length_rad",
            "joint_path_length_rad",
            SCALED,
        ),
        "maximum_joint_travel_normalized": (
            "maximum_joint_travel_rad",
            "maximum_joint_travel_rad",
            SCALED,
        ),
        "j4_travel_normalized": ("j4_travel_rad", "j4_travel_rad", SCALED),
        "maximum_absolute_j4_gravity_torque_normalized": (
            "maximum_absolute_j4_gravity_torque_nm",
            "maximum_absolute_j4_gravity_torque_nm",
            SCALED,
        ),
        "maximum_velocity_normalized": ("maximum_velocity_rad_s", "maximum_joint_velocity_rad_s", SCALED),
        "maximum_acceleration_normalized": (
            "maximum_acceleration_rad_s2",
            "maximum_joint_acceleration_rad_s2",
            SCALED,
        ),
        "path_clearance_deficit_normalized": (
            "minimum_task_obstacles_clearance_m",
            "minimum_task_obstacles_clearance_m",
            DEFICIT,
        ),
    }
    components: dict[str, dict[str, Any]] = {}
    penalty = 0.0
    for term, (metric_key, reference_key, mode) in raw_terms.items():
        raw = float(trajectory_metrics.get(metric_key, 0.0))
        weight = float(weights.get(term, 0.0))
        scale = _scale(references, reference_key)
        normalized = clamp01(1.0 - raw / scale) if mode == DEFICIT else clamp01(raw / scale)
        contribution = -weight * normalized
        components[term] = {
            "raw": raw,
            "mode": mode,
            "normalized": normalized,
            "weight": weight,
            "contribution": contribution,
        }
        penalty += contribution
    endpoint_contribution = float(endpoint_weight) * float(endpoint_static_score)
    components["endpoint_static_score"] = {
        "raw": float(endpoint_static_score),
        "normalized": float(endpoint_static_score),
        "weight": float(endpoint_weight),
        "contribution": endpoint_contribution,
    }
    return {
        "score": float(endpoint_contribution + penalty),
        "endpoint_static_score": float(endpoint_static_score),
        "endpoint_weight": float(endpoint_weight),
        "penalty_total": float(penalty),
        "components": components,
        "weights": {term: float(weights.get(term, 0.0)) for term in TRAJECTORY_TERMS},
    }


def static_metric_values_for_reference_resolution(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Flat mapping of the raw static metrics a reference scale can be resolved from."""
    return {
        "yoshikawa_manipulability": float(metrics["yoshikawa_manipulability"]),
        "absolute_static_gravity_torque_j4_nm": float(metrics["absolute_gravity_torque_j4_nm"]),
        "maximum_absolute_static_gravity_torque_nm": float(metrics["maximum_absolute_gravity_torque_nm"]),
        "j4_margin_normalized": float(metrics["j4_margin_normalized"]),
        "minimum_joint_margin_normalized": float(metrics["minimum_joint_margin_normalized"]),
    }


def reference_modes_are_legacy_first(references: Mapping[str, Reference]) -> bool:
    """Diagnostic used by the report: are references run-relative rather than invented?"""
    return any(reference.mode == MODE_LEGACY_BASELINE for reference in references.values()) and all(
        reference.mode in {MODE_CONFIGURED, MODE_LEGACY_BASELINE} for reference in references.values()
    )


def weight_vector(weights: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(weights.get(term, 0.0)) for term in STATIC_TERMS], dtype=float)
