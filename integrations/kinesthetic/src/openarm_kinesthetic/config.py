"""YAML loading and safety-oriented configuration validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import config_path


class ConfigError(ValueError):
    """Raised when a configuration violates a project contract."""


def load_yaml(path_or_name: str | Path) -> dict[str, Any]:
    path = Path(path_or_name)
    if not path.is_absolute() and len(path.parts) == 1:
        path = config_path(str(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    reject_non_finite(data, str(path))
    return data


def reject_non_finite(value: Any, label: str = "config") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{label}: non-finite number is forbidden")
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_non_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_non_finite(item, f"{label}[{index}]")


def require_vector(value: Any, length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{label}: expected length {length}")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise ConfigError(f"{label}: all values must be numeric")
    reject_non_finite(value, label)


def _require_frames(config: Mapping[str, Any], required: tuple[str, ...]) -> None:
    frames = config.get("frames")
    if not isinstance(frames, Mapping):
        raise ConfigError("frames: required mapping is missing")
    for name in required:
        if not isinstance(frames.get(name), str) or not frames[name].strip():
            raise ConfigError(f"frames.{name}: required coordinate frame is missing")


def validate_scene(config: Mapping[str, Any]) -> None:
    _require_frames(config, ("world", "robot_base", "table_top", "black_box", "cube"))
    kind = config.get("kind")
    if kind not in {"nominal", "real"}:
        raise ConfigError("scene.kind must be nominal or real")
    if kind == "nominal" and config.get("provisional") is not True:
        raise ConfigError("nominal scene must remain provisional")
    if kind == "real" and config.get("ready") is True:
        missing = []
        for dotted in config.get("readiness_requirements", []):
            if dotted.startswith(("cameras.", "poses.")):
                missing.append(dotted)
                continue
            node: Any = config
            for part in dotted.split("."):
                node = node.get(part) if isinstance(node, Mapping) else None
            if not isinstance(node, Mapping) or node.get("value") is None or node.get("status") != "measured":
                missing.append(dotted)
        if missing:
            raise ConfigError("real scene cannot be ready; unmet calibrated fields: " + ", ".join(missing))
    if kind == "real" and config.get("fallback_for_real_allowed") is not False:
        raise ConfigError("real scene must forbid fallback to nominal geometry")


def validate_poses(config: Mapping[str, Any]) -> None:
    if not isinstance(config.get("coordinate_frame"), str):
        raise ConfigError("poses.coordinate_frame is required")
    require_vector(config.get("motor_zero", {}).get("arm"), 7, "motor_zero.arm")
    require_vector_fields = ("left_park", "right_task_ready")
    for name in require_vector_fields:
        require_vector(config.get(name, {}).get("candidate_arm"), 7, f"{name}.candidate_arm")
    if config.get("motor_zero", {}).get("dataset_role") != "none":
        raise ConfigError("q_motor_zero must not have a dataset role")
    if config.get("episode_home") == "motor_zero":
        raise ConfigError("q_motor_zero must not be ACT episode Home")
    if config.get("episode_home") != "right_task_ready":
        raise ConfigError("ACT episode Home must be right_task_ready")


def validate_task_contract(config: Mapping[str, Any]) -> None:
    _require_frames(config, ("world", "ceiling_camera", "wrist_right_camera"))
    dims = config.get("dimensions", {})
    if dims.get("arm") != 7 or dims.get("arm_with_gripper") != 8 or dims.get("state_action") != 16:
        raise ConfigError("task dimensions must be exactly 7D/8D/16D")
    ordering = config.get("ordering", {})
    if not isinstance(ordering.get("state_action"), list) or len(ordering["state_action"]) != 16:
        raise ConfigError("ordering.state_action must contain 16 fields")
    if len(ordering.get("right_arm_with_gripper", [])) != 8 or len(ordering.get("left_arm_with_gripper", [])) != 8:
        raise ConfigError("each arm-with-gripper ordering must contain 8 fields")
    for feature in ("observation.state", "action"):
        if config.get("features", {}).get(feature, {}).get("shape") != [16]:
            raise ConfigError(f"features.{feature}.shape must be [16]")
    if config.get("rate_hz") != 30:
        raise ConfigError("ACT contract rate must be 30 Hz")
    if config.get("episode", {}).get("home_pose") == "motor_zero":
        raise ConfigError("q_motor_zero must not be ACT episode Home")


def validate_gamepad_control(config: Mapping[str, Any]) -> None:
    axes = config.get("mapping", {}).get("axes", {})
    buttons = config.get("mapping", {}).get("buttons", {})
    required_axes = {"left_x", "left_y", "right_x", "right_y", "lt", "rt", "dpad_x", "dpad_y"}
    required_buttons = {
        "gripper_open_a",
        "record_toggle_x",
        "hand_drag_lb",
        "hand_drag_gripper_toggle_l3",
        "gripper_close_y",
        "stop_rb_r1",
    }
    if set(axes) != required_axes or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in axes.values()):
        raise ConfigError("gamepad mapping must define nonnegative indices for all eight required axes")
    if len(set(axes.values())) != len(axes):
        raise ConfigError("gamepad axis indices must be unique")
    if set(buttons) != required_buttons or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in buttons.values()):
        raise ConfigError(
            "gamepad mapping must define nonnegative A/X/LB/L3/Y/RB button indices"
        )
    if len(set(buttons.values())) != len(buttons):
        raise ConfigError("gamepad button indices must be unique")
    control = config.get("control", {})
    bounded = {
        "trigger_press_threshold": (0.05, 0.95),
        "stick_deadzone": (0.0, 0.5),
        "response_exponent": (0.5, 4.0),
        "maximum_translation_velocity_m_s": (0.001, 0.5),
        "maximum_vertical_velocity_m_s": (0.001, 0.5),
        "maximum_angular_velocity_rad_s": (0.01, 3.0),
        "maximum_target_position_error_m": (0.005, 0.2),
    }
    for name, (lower, upper) in bounded.items():
        value = control.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not lower <= value <= upper:
            raise ConfigError(f"gamepad control.{name} must be in [{lower}, {upper}]")
    if control.get("activation") != "hold_lt_deadman":
        raise ConfigError("gamepad activation must remain hold_lt_deadman")
    safety = config.get("safety", {})
    if safety.get("robot_hardware_access") != "forbidden" or safety.get("stop_is_physical_emergency_stop") is not False:
        raise ConfigError("simulation gamepad must forbid hardware access and physical-E-stop claims")


def validate_initialization(config: Mapping[str, Any]) -> None:
    """Guard the initialization chain's hard invariants.

    These are not style checks: each one corresponds to a requirement that, if it
    silently regressed, would either put setup motion into the training set or let
    unverified hardware execution start.
    """
    from .initialization.table_geometry import ALLOWED_SOURCES

    if config.get("initialization", {}).get("strategy") != "prior_table_teaching_ready":
        raise ConfigError("initialization.strategy must be prior_table_teaching_ready")

    table = config.get("table")
    if not isinstance(table, Mapping):
        raise ConfigError("table: required mapping is missing")
    for key in ("size_x_m", "size_y_m", "thickness_m", "near_edge_gap_from_robot_front_m", "center_y_m", "top_z_m", "yaw_rad"):
        node = table.get(key)
        if not isinstance(node, Mapping) or "value" not in node or node.get("source") not in ALLOWED_SOURCES:
            raise ConfigError(f"table.{key} needs a value and a known provenance source")
    reference = table.get("robot_front_reference")
    if not isinstance(reference, Mapping) or reference.get("source") != "model_derived":
        raise ConfigError("table.robot_front_reference.source must be model_derived")
    if reference.get("x_m") is not None:
        raise ConfigError("table.robot_front_reference.x_m must be derived at runtime, not hardcoded")

    teaching = config.get("teaching_ready")
    if not isinstance(teaching, Mapping):
        raise ConfigError("teaching_ready: required mapping is missing")
    if teaching.get("orientation", {}).get("source") != "fk_reference":
        raise ConfigError("teaching_ready.orientation.source must be fk_reference, not an invented Euler triple")
    for key in ("offset_x_m", "offset_y_m", "height_above_table_m"):
        node = teaching.get(key)
        if not isinstance(node, Mapping) or node.get("source") not in ALLOWED_SOURCES:
            raise ConfigError(f"teaching_ready.{key} needs a known provenance source")
    if float(teaching["height_above_table_m"]["value"]) <= 0.0:
        raise ConfigError("teaching_ready.height_above_table_m must be positive")

    fallback = teaching.get("fallback_search", {})
    if fallback.get("enabled"):
        step = float(fallback["step_m"])
        radius = float(fallback["max_radius_m"])
        if not 0.0 < step <= radius <= 0.20:
            raise ConfigError("fallback_search must stay a small bounded neighbourhood")
        if int(fallback["maximum_candidates"]) < 1:
            raise ConfigError("fallback_search.maximum_candidates must be at least 1")

    planning = config.get("planning", {})
    if planning.get("start_state_source") != "current_joint_state":
        raise ConfigError("planning.start_state_source must be current_joint_state")
    if planning.get("planner") != "rrt_connect":
        raise ConfigError("planning.planner must be the sampling planner, not joint interpolation")
    scaling = planning.get("time_scaling", {})
    if float(scaling.get("maximum_joint_velocity_rad_s", 0)) <= 0 or float(
        scaling.get("maximum_joint_acceleration_rad_s2", 0)
    ) <= 0:
        raise ConfigError("planning.time_scaling limits must be positive")

    simulation = config.get("simulation", {})
    if simulation.get("geometry_source") != "ground_truth":
        raise ConfigError("simulation.geometry_source must be ground_truth")
    if simulation.get("initial_pose") != "motor_zero":
        raise ConfigError("simulation.initial_pose must match the physical vertical-down motor_zero")
    real = config.get("real", {})
    if (
        real.get("xy_source") != "live_depth_registered_with_station_prior"
        or real.get("size_source") != "configured_prior"
        or real.get("z_source") != "live_depth_estimated"
    ):
        raise ConfigError(
            "real table geometry must use registered live-depth XY/yaw, "
            "configured-prior size, and live-depth Z"
        )

    if config.get("recording", {}).get("include_initialization_trajectory") is not False:
        raise ConfigError("the initialization trajectory must never be included in training data")
    if config.get("real_execution", {}).get("enabled") is not False:
        raise ConfigError("real execution must be disabled by default in this phase")
    if config.get("real_execution", {}).get("require_operator_confirmation") is not True:
        raise ConfigError("real execution must require explicit operator confirmation")


# Vocabulary that would turn a single real observation into an invented engineering
# limit. The Teaching Ready search compares candidates *relatively* against the legacy
# baseline, so a J4 bound expressed with this vocabulary is a contract violation
# rather than a tuning choice.
_J4_BOUND_TOKENS = ("limit", "threshold", "bound", "ceiling", "max_allowed", "min_allowed")


def _reject_j4_absolute_bounds(node: Any, path: str) -> None:
    if not isinstance(node, Mapping):
        return
    for key, value in node.items():
        lowered = str(key).lower()
        if ("j4" in lowered or "joint4" in lowered) and any(token in lowered for token in _J4_BOUND_TOKENS):
            raise ConfigError(
                f"{path}.{key}: a J4 hard limit/threshold is forbidden; the search compares candidates "
                "relative to the legacy baseline measured in the same run"
            )
        _reject_j4_absolute_bounds(value, f"{path}.{key}")


def validate_ready_search(config: Mapping[str, Any]) -> None:
    """Guard the Teaching Ready pose search's structural invariants.

    Each check corresponds to a way the search could quietly stop being trustworthy:
    reaching hardware, inventing a world RPY orientation, dropping the baseline it is
    judged against, or promoting one real failure into a J4 limit.
    """
    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise ConfigError("scope: required mapping is missing")
    if scope.get("mode") != "simulation_only":
        raise ConfigError("scope.mode must be simulation_only")
    if scope.get("hardware_access") != "forbidden":
        raise ConfigError("scope.hardware_access must be forbidden")
    if scope.get("real_execution_code") != "untouched":
        raise ConfigError("scope.real_execution_code must be untouched")

    scene = config.get("scene")
    if scene is not None and (not isinstance(scene, str) or not scene.startswith("sim/")):
        raise ConfigError("scene must be null or a path under sim/")

    orientations = config.get("orientations")
    if not isinstance(orientations, Mapping):
        raise ConfigError("orientations: required mapping is missing")
    if orientations.get("tilt_convention") != "angle_below_horizontal_deg":
        raise ConfigError("orientations.tilt_convention must be angle_below_horizontal_deg")
    if orientations.get("roll_resolution") != "minimal_rotation_from_reference":
        raise ConfigError("orientations.roll_resolution must be minimal_rotation_from_reference")

    families = orientations.get("families")
    if not isinstance(families, list) or not families:
        raise ConfigError("orientations.families must be a non-empty list")
    names: list[str] = []
    reference_families: list[str] = []
    tilted_families: list[str] = []
    for index, family in enumerate(families):
        if not isinstance(family, Mapping):
            raise ConfigError(f"orientations.families[{index}] must be a mapping")
        name = family.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"orientations.families[{index}].name is required")
        if name in names:
            raise ConfigError(f"orientations.families: duplicate family name {name!r}")
        names.append(name)
        kind = family.get("kind")
        if kind not in {"reference_orientation", "tilted_approach"}:
            raise ConfigError(
                f"orientations.families[{index}].kind must be reference_orientation or tilted_approach"
            )
        if not bool(family.get("enabled", True)):
            continue
        if kind == "reference_orientation":
            reference_families.append(name)
            continue
        tilted_families.append(name)
        tilts = family.get("tilt_below_horizontal_deg")
        if not isinstance(tilts, list) or not tilts:
            raise ConfigError(f"orientations.families[{index}].tilt_below_horizontal_deg must be a non-empty list")
        for tilt in tilts:
            if not isinstance(tilt, (int, float)) or isinstance(tilt, bool) or not 0.0 <= float(tilt) <= 90.0:
                raise ConfigError(
                    f"orientations.families[{index}].tilt_below_horizontal_deg must lie in [0, 90] degrees"
                )
        heading = family.get("base_heading_deg")
        if not isinstance(heading, (int, float)) or isinstance(heading, bool):
            raise ConfigError(f"orientations.families[{index}].base_heading_deg must be numeric")
        for key in ("heading_offsets_deg", "roll_offsets_deg"):
            values = family.get(key)
            if not isinstance(values, list) or not values:
                raise ConfigError(f"orientations.families[{index}].{key} must be a non-empty list")
            for value in values:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ConfigError(f"orientations.families[{index}].{key} values must be numeric")
    if len(reference_families) != 1 or reference_families[0] != "legacy_pi2":
        raise ConfigError(
            "exactly one enabled reference_orientation family named legacy_pi2 is required as the comparison baseline"
        )
    if not tilted_families:
        raise ConfigError("at least one enabled tilted_approach family is required")

    search = config.get("search")
    if not isinstance(search, Mapping):
        raise ConfigError("search: required mapping is missing")
    position = search.get("position")
    if not isinstance(position, Mapping):
        raise ConfigError("search.position: required mapping is missing")
    if position.get("anchor_source") not in {
        "initialization_yaml_teaching_ready_anchor",
        "table_geometric_centerline",
        "operator_marked_table_point",
    }:
        raise ConfigError(
            "search.position.anchor_source must be "
            "initialization_yaml_teaching_ready_anchor, table_geometric_centerline, "
            "or operator_marked_table_point"
        )
    if position.get("anchor_source") == "operator_marked_table_point":
        for key, maximum in (
            ("anchor_table_offset_x_m", 0.20),
            ("anchor_table_offset_y_m", 0.30),
        ):
            value = position.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or abs(float(value)) > maximum
            ):
                raise ConfigError(
                    f"search.position.{key} must be finite and within +/-{maximum:.2f} m"
                )
    coarse = position.get("coarse")
    if not isinstance(coarse, Mapping):
        raise ConfigError("search.position.coarse: required mapping is missing")
    for key in ("x_offsets_m", "y_offsets_m", "z_offsets_m"):
        values = coarse.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"search.position.coarse.{key} must be a non-empty list")
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"search.position.coarse.{key} values must be numeric")
            if abs(float(value)) > 0.10:
                raise ConfigError(
                    f"search.position.coarse.{key} must stay within 0.10 m of the Teaching Ready anchor: "
                    "this is an anchor neighbourhood search, not a workspace optimizer"
                )
    margin = position.get("footprint_margin_m")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or float(margin) < 0.0:
        raise ConfigError("search.position.footprint_margin_m must be a non-negative number")
    minimum_height = position.get("minimum_height_above_table_m")
    maximum_height = position.get("maximum_height_above_table_m")
    if not isinstance(minimum_height, (int, float)) or not isinstance(maximum_height, (int, float)):
        raise ConfigError("search.position height bounds must be numeric")
    if not 0.0 < float(minimum_height) < float(maximum_height):
        raise ConfigError("search.position height bounds must satisfy 0 < minimum < maximum")

    ik = config.get("ik")
    if not isinstance(ik, Mapping):
        raise ConfigError("ik: required mapping is missing")
    for key in ("position_tolerance_m", "orientation_tolerance_rad", "joint_margin_rad"):
        value = ik.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
            raise ConfigError(f"ik.{key} must be a positive number")
    if int(ik.get("iterations", 0)) < 1:
        raise ConfigError("ik.iterations must be at least 1")
    if float(ik.get("dedup_max_joint_distance_rad", 0.0)) <= 0.0:
        raise ConfigError("ik.dedup_max_joint_distance_rad must be positive")

    filters = config.get("filters")
    if not isinstance(filters, Mapping):
        raise ConfigError("filters: required mapping is missing")
    for key in ("minimum_joint_margin_normalized", "minimum_j4_margin_normalized"):
        value = filters.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) < 1.0:
            raise ConfigError(f"filters.{key} must be a normalized margin in [0, 1)")
    metrics_block = config.get("metrics")
    if not isinstance(metrics_block, Mapping):
        raise ConfigError("metrics: required mapping is missing")
    if float(metrics_block.get("clearance_cap_m", 0.0)) <= 0.0:
        raise ConfigError("metrics.clearance_cap_m must be positive")
    if int(metrics_block.get("trajectory_sample_count", 0)) < 2:
        raise ConfigError("metrics.trajectory_sample_count must be at least 2")

    scoring = config.get("scoring")
    if not isinstance(scoring, Mapping):
        raise ConfigError("scoring: required mapping is missing")
    for level in ("static", "trajectory"):
        block = scoring.get(level)
        if not isinstance(block, Mapping):
            raise ConfigError(f"scoring.{level}: required mapping is missing")
        weights = block.get("weights")
        references = block.get("references")
        if not isinstance(weights, Mapping) or not weights:
            raise ConfigError(f"scoring.{level}.weights must be a non-empty mapping")
        if not isinstance(references, Mapping) or not references:
            raise ConfigError(f"scoring.{level}.references must be a non-empty mapping")
        for key, value in weights.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ConfigError(f"scoring.{level}.weights.{key} must be a finite number")
        if all(float(value) == 0.0 for value in weights.values()):
            raise ConfigError(f"scoring.{level}.weights must not be all zero")
        for key, node in references.items():
            if not isinstance(node, Mapping) or node.get("mode") not in {"configured", "legacy_baseline"}:
                raise ConfigError(f"scoring.{level}.references.{key}.mode must be configured or legacy_baseline")
            if node["mode"] == "configured":
                value = node.get("value")
                if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
                    raise ConfigError(f"scoring.{level}.references.{key}.value must be a positive number")
            else:
                if not isinstance(node.get("metric"), str) or not node["metric"].strip():
                    raise ConfigError(f"scoring.{level}.references.{key}.metric is required for legacy_baseline")
                fallback = node.get("fallback_value")
                if not isinstance(fallback, (int, float)) or isinstance(fallback, bool) or float(fallback) <= 0.0:
                    raise ConfigError(f"scoring.{level}.references.{key}.fallback_value must be positive")

    # Round-1 rule: proximity to the previous pose must not steer the search back to
    # the configuration that failed on hardware.
    legacy_weight = scoring["static"]["weights"].get("legacy_reference_distance_normalized")
    if legacy_weight is None or float(legacy_weight) != 0.0:
        raise ConfigError(
            "scoring.static.weights.legacy_reference_distance_normalized must be 0.0 in this round: "
            "a proximity term would pull the search back towards the previous fixed orientation"
        )
    if not isinstance(scoring["trajectory"].get("endpoint_weight"), (int, float)):
        raise ConfigError("scoring.trajectory.endpoint_weight must be numeric")

    planning = config.get("planning")
    report = config.get("report")
    if not isinstance(planning, Mapping) or not isinstance(report, Mapping):
        raise ConfigError("planning and report blocks are required")
    top_k = int(planning.get("top_k", 0))
    top_n = int(report.get("top_n", 0))
    if top_k < 1 or top_n < 1:
        raise ConfigError("planning.top_k and report.top_n must be at least 1")
    if top_k < top_n:
        raise ConfigError("planning.top_k must be at least report.top_n")
    if int(report.get("ranking_size", top_n)) < top_n:
        raise ConfigError("report.ranking_size must be at least report.top_n")
    diversity = report.get("diversity", {})
    if not isinstance(diversity, Mapping):
        raise ConfigError("report.diversity must be a mapping")
    if bool(diversity.get("enabled", False)):
        angle = diversity.get("minimum_approach_axis_angle_deg")
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not 0.0 < float(angle) < 90.0:
            raise ConfigError(
                "report.diversity.minimum_approach_axis_angle_deg must lie in (0, 90) degrees when enabled"
            )

    acceptance = config.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ConfigError("acceptance: required mapping is missing")
    if int(acceptance.get("minimum_valid_candidates_per_enabled_family", 0)) < 1:
        raise ConfigError("acceptance.minimum_valid_candidates_per_enabled_family must be at least 1")
    if float(acceptance.get("minimum_j4_spread_rad", 0.0)) <= 0.0:
        raise ConfigError("acceptance.minimum_j4_spread_rad must be positive")
    for key in ("maximum_j4_margin_regression_normalized", "minimum_j4_torque_reduction_ratio"):
        value = acceptance.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) < 1.0:
            raise ConfigError(f"acceptance.{key} must be a relative comparison margin in [0, 1)")
    if float(acceptance.get("penetration_tolerance_m", -1.0)) < 0.0:
        raise ConfigError("acceptance.penetration_tolerance_m must be a non-negative depth tolerance")

    observations = config.get("known_bad_observations")
    if not isinstance(observations, list) or not observations:
        raise ConfigError("known_bad_observations must be a non-empty list")
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ConfigError(f"known_bad_observations[{index}] must be a mapping")
        if not isinstance(observation.get("source"), str) or not observation["source"].strip():
            raise ConfigError(f"known_bad_observations[{index}].source is required")
        policy = observation.get("policy")
        if not isinstance(policy, str) or "observation" not in policy.lower():
            raise ConfigError(
                f"known_bad_observations[{index}].policy must state that the record is an observation, "
                "not a derived limit"
            )

    _reject_j4_absolute_bounds(config, "ready_search")


def _valuenode(value: Any, label: str, *, default: float | None = None) -> float:
    """Read either a bare number or a `{value, source}` provenance node."""
    if value is None and default is not None:
        return float(default)
    if isinstance(value, Mapping):
        if "value" not in value or "source" not in value:
            raise ConfigError(f"{label}: expected a `value` and a `source`")
        if value["source"] not in {*ALLOWED_TRAJECTORY_SOURCES, _SOURCE_REUSED_PREFIX}:
            if not str(value["source"]).startswith(_SOURCE_REUSED_PREFIX):
                raise ConfigError(f"{label}: unknown provenance source {value['source']!r}")
        value = value["value"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{label}: expected a number")
    return float(value)


# Provenance labels accepted in ready_trajectory.yaml. `reused_from_*` names the artifact
# the number was taken from rather than claiming it was measured for this purpose.
_SOURCE_REUSED_PREFIX = "reused_from_"
ALLOWED_TRAJECTORY_SOURCES = {"configured_prior"}


def validate_ready_trajectory(config: Mapping[str, Any]) -> None:
    """Guard the validated-trajectory store, the replay gate and the freeze policy.

    Each check corresponds to a way daily replay could quietly become unsafe: a rejected
    replay silently re-planned, a frozen profile built from several different
    trajectories, or a moved camera accepted because its hash was simply missing.
    """
    mode = config.get("mode")
    if not isinstance(mode, Mapping):
        raise ConfigError("mode: required mapping is missing")
    default_mode = mode.get("default")
    allowed_modes = {"search", "acceptance_replay", "validated_replay"}
    if default_mode not in allowed_modes:
        raise ConfigError(
            "mode.default must be search, acceptance_replay, or validated_replay"
        )
    if set(mode.get("allowed", ())) != allowed_modes:
        raise ConfigError(
            "mode.allowed must be exactly [search, acceptance_replay, validated_replay]"
        )
    if mode.get("on_replay_rejection") != "require_search":
        raise ConfigError(
            "mode.on_replay_rejection must be require_search: a rejected replay must be refused, "
            "never silently re-planned"
        )

    store = config.get("store")
    if not isinstance(store, Mapping):
        raise ConfigError("store: required mapping is missing")
    root = store.get("root")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError("store.root must be a non-empty path")
    files = store.get("files")
    required_files = {
        "metadata",
        "ready_pose",
        "trajectory",
        "planning_report",
        "scene_signature",
        "acceptance_report",
    }
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise ConfigError("store.files must name exactly: " + ", ".join(sorted(required_files)))
    if any(not isinstance(value, str) or not value.strip() for value in files.values()):
        raise ConfigError("store.files values must be non-empty file names")

    signature = config.get("scene_signature")
    if not isinstance(signature, Mapping):
        raise ConfigError("scene_signature: required mapping is missing")
    for key in (
        "anchor_position_tolerance_m",
        "table_center_position_tolerance_m",
        "table_yaw_tolerance_deg",
        "plane_offset_tolerance_m",
        "normal_angle_tolerance_deg",
        "guard_size_tolerance_m",
    ):
        if _valuenode(signature.get(key), f"scene_signature.{key}") <= 0.0:
            raise ConfigError(f"scene_signature.{key} must be positive")
    hash_fields = signature.get("hash_fields")
    if not isinstance(hash_fields, list) or not hash_fields or any(
        not isinstance(item, str) or not item.strip() for item in hash_fields
    ):
        raise ConfigError("scene_signature.hash_fields must be a non-empty list of names")
    if signature.get("hash_mismatch_policy") not in {"reject", "warn"}:
        raise ConfigError("scene_signature.hash_mismatch_policy must be reject or warn")
    if signature.get("missing_hash_policy") not in {"reject", "warn"}:
        raise ConfigError("scene_signature.missing_hash_policy must be reject or warn")
    compared = signature.get("compared_fields")
    if not isinstance(compared, list) or not compared:
        raise ConfigError("scene_signature.compared_fields must be a non-empty list")
    required_compared = {
        "anchor_position_world_m",
        "table_center_world_m",
        "table_forward_world",
        "plane_normal_world",
        "plane_offset_m",
        "guard_size_xy_m",
        "selection_mode",
        "table_xy_source",
        "table_z_source",
    }
    if set(compared) != required_compared or len(compared) != len(required_compared):
        raise ConfigError(
            "scene_signature.compared_fields must exactly match the replay comparison contract"
        )

    gate = config.get("replay_gate")
    if not isinstance(gate, Mapping):
        raise ConfigError("replay_gate: required mapping is missing")
    if _valuenode(gate.get("maximum_joint_state_age_s"), "replay_gate.maximum_joint_state_age_s") <= 0.0:
        raise ConfigError("replay_gate.maximum_joint_state_age_s must be positive")
    if _valuenode(gate.get("start_position_tolerance_rad"), "replay_gate.start_position_tolerance_rad") <= 0.0:
        raise ConfigError("replay_gate.start_position_tolerance_rad must be positive")
    controllers = gate.get("required_controllers")
    if not isinstance(controllers, list) or not controllers:
        raise ConfigError("replay_gate.required_controllers must be a non-empty list")
    for key in (
        "require_arm_rigid",
        "require_profile_frozen",
        "require_profile_verified",
        "require_scene_signature_match",
        "require_start_position_match",
    ):
        if not isinstance(gate.get(key), bool):
            raise ConfigError(f"replay_gate.{key} must be a boolean")
    if gate.get("require_profile_verified") is not True:
        raise ConfigError(
            "replay_gate.require_profile_verified must be true: stored-file integrity and "
            "the current commissioning path envelope are non-optional replay gates"
        )
    if gate.get("allow_start_trim") is not False:
        raise ConfigError(
            "replay_gate.allow_start_trim must be false: a start that does not match is a different "
            "trajectory, not a trajectory to trim"
        )

    freeze = config.get("freeze_policy")
    if not isinstance(freeze, Mapping):
        raise ConfigError("freeze_policy: required mapping is missing")
    passes = int(freeze.get("required_consecutive_passes", 0))
    low = int(freeze.get("minimum_low_speed_passes", -1))
    production = int(freeze.get("minimum_production_scaling_passes", -1))
    if passes < 1:
        raise ConfigError("freeze_policy.required_consecutive_passes must be at least 1")
    if not 0 <= low <= passes:
        raise ConfigError("freeze_policy.minimum_low_speed_passes must lie in [0, required_consecutive_passes]")
    if not 0 <= production <= passes:
        raise ConfigError(
            "freeze_policy.minimum_production_scaling_passes must lie in [0, required_consecutive_passes]"
        )
    if low + production > passes:
        raise ConfigError(
            "freeze_policy: minimum_low_speed_passes + minimum_production_scaling_passes cannot "
            "exceed required_consecutive_passes"
        )
    scaling = freeze.get("production_scaling")
    if not isinstance(scaling, Mapping):
        raise ConfigError("freeze_policy.production_scaling: required mapping is missing")
    for key in ("velocity", "acceleration"):
        value = scaling.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 < float(value) <= 0.5:
            raise ConfigError(f"freeze_policy.production_scaling.{key} must lie in (0, 0.5]")
    low_speed = _valuenode(freeze.get("low_speed_scaling_at_most"), "freeze_policy.low_speed_scaling_at_most")
    if not 0.0 < low_speed <= float(scaling["velocity"]):
        raise ConfigError("freeze_policy.low_speed_scaling_at_most must lie in (0, production velocity scaling]")
    for key in (
        "require_identical_trajectory",
        "require_zero_failures_in_streak",
        "fail_on_j4_red_light",
        "fail_on_operator_anomaly",
    ):
        if freeze.get(key) is not True:
            raise ConfigError(f"freeze_policy.{key} must be true")
    for key in (
        "maximum_final_joint_error_rad",
        "maximum_final_tcp_error_m",
        "maximum_final_orientation_error_rad",
    ):
        if _valuenode(freeze.get(key), f"freeze_policy.{key}") <= 0.0:
            raise ConfigError(f"freeze_policy.{key} must be positive")

    observations = config.get("known_bad_observations")
    if not isinstance(observations, list) or not observations:
        raise ConfigError("known_bad_observations must be a non-empty list")
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or not isinstance(observation.get("source"), str):
            raise ConfigError(f"known_bad_observations[{index}] needs a source")
        policy = observation.get("policy")
        if not isinstance(policy, str) or "observation" not in policy.lower():
            raise ConfigError(
                f"known_bad_observations[{index}].policy must state that the record is an observation"
            )

    _reject_j4_absolute_bounds(config, "ready_trajectory")


def validate_real_ready_search(config: Mapping[str, Any]) -> None:
    """Guard the real-station Teaching Ready SE(3) search.

    Each check corresponds to a way the real search could stop being trustworthy: a legacy
    pose that quietly becomes the target, a baseline asserted instead of measured, a world
    coordinate copied from simulation, or one real failure promoted into a physical J4
    limit that then gets quoted as motor data.
    """
    from .real.ready_candidates import (
        IMPROVEMENT_GATE_REFERENCE_BASELINE_ONLY_WHEN_ABOVE_CEILING,
        SIMULATION_PRIOR_FAMILIES,
        TRAJECTORY_SELECTION_KEYS,
    )

    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise ConfigError("scope: required mapping is missing")
    if scope.get("mode") != "real_commissioning":
        raise ConfigError("scope.mode must be real_commissioning")
    if scope.get("default_action") != "plan_only":
        raise ConfigError(
            "scope.default_action must be plan_only: a configuration file must not be able "
            "to authorize real motion on its own"
        )
    if scope.get("execution_authorization") != "cli_flag_allow_real_execution":
        raise ConfigError(
            "scope.execution_authorization must be cli_flag_allow_real_execution; real motion "
            "needs the executor's explicit flag, not a config value"
        )
    if scope.get("joint_values_source") != "live_moveit_ik_on_registered_table":
        raise ConfigError(
            "scope.joint_values_source must be live_moveit_ik_on_registered_table: the real "
            "station solves its own IK and never reuses simulation joint values"
        )
    if scope.get("simulation_prior_role") != "search_order_only":
        raise ConfigError(
            "scope.simulation_prior_role must be search_order_only: the simulation prior "
            "orders the search, it does not choose the result"
        )

    search = config.get("real_ready_search")
    if not isinstance(search, Mapping):
        raise ConfigError("real_ready_search: required mapping is missing")

    families = search.get("families")
    if not isinstance(families, list) or not families:
        raise ConfigError("real_ready_search.families must be a non-empty list")
    for family in families:
        if family not in SIMULATION_PRIOR_FAMILIES:
            raise ConfigError(
                f"real_ready_search.families contains unknown family {family!r}; "
                f"known: {list(SIMULATION_PRIOR_FAMILIES)}"
            )
    if len(set(families)) != len(families):
        raise ConfigError("real_ready_search.families must not repeat a family")

    prior = search.get("simulation_prior")
    if prior not in families:
        raise ConfigError(
            "real_ready_search.simulation_prior must be one of real_ready_search.families"
        )

    for key in ("tilt_window_deg", "tilt_step_deg"):
        value = search.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
            raise ConfigError(f"real_ready_search.{key} must be positive")
    if float(search["tilt_window_deg"]) > 30.0:
        raise ConfigError(
            "real_ready_search.tilt_window_deg must stay within 30 degrees of the simulation "
            "family: this is a local correction, not a new sweep"
        )
    if float(search["tilt_step_deg"]) > float(search["tilt_window_deg"]):
        raise ConfigError("real_ready_search.tilt_step_deg must not exceed tilt_window_deg")

    explicit_tilts = search.get("tilt_offsets_deg", {})
    if not isinstance(explicit_tilts, Mapping):
        raise ConfigError("real_ready_search.tilt_offsets_deg must be a mapping")
    for family, tilts in explicit_tilts.items():
        if family not in families:
            raise ConfigError(
                f"real_ready_search.tilt_offsets_deg names unknown family {family!r}"
            )
        if not isinstance(tilts, list) or not tilts:
            raise ConfigError(
                f"real_ready_search.tilt_offsets_deg.{family} must be a non-empty list"
            )
        for tilt in tilts:
            if (
                not isinstance(tilt, (int, float))
                or isinstance(tilt, bool)
                or not -90.0 <= float(tilt) <= 90.0
            ):
                raise ConfigError(
                    f"real_ready_search.tilt_offsets_deg.{family} must lie in [-90, 90] degrees"
                )

    headings = search.get("heading_offsets_deg")
    if not isinstance(headings, Mapping) or not headings:
        raise ConfigError("real_ready_search.heading_offsets_deg must be a non-empty mapping")
    for family, values in headings.items():
        if family not in families:
            raise ConfigError(
                f"real_ready_search.heading_offsets_deg names unknown family {family!r}"
            )
        if not isinstance(values, list) or not values:
            raise ConfigError(
                f"real_ready_search.heading_offsets_deg.{family} must be a non-empty list"
            )
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(
                    f"real_ready_search.heading_offsets_deg.{family} values must be numeric"
                )
            if abs(float(value)) > 45.0:
                raise ConfigError(
                    f"real_ready_search.heading_offsets_deg.{family} must stay within 45 degrees "
                    "of the registered table forward axis"
                )
    if set(headings) != set(families):
        raise ConfigError(
            "real_ready_search.heading_offsets_deg must name every family in "
            "real_ready_search.families, so no family's heading sweep is left implied"
        )

    rolls = search.get("roll_offsets_deg")
    if not isinstance(rolls, list) or not rolls:
        raise ConfigError("real_ready_search.roll_offsets_deg must be a non-empty list")
    for value in rolls:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or abs(float(value)) > 45.0:
            raise ConfigError("real_ready_search.roll_offsets_deg must stay within 45 degrees")

    offsets = search.get("position_offsets_m")
    if not isinstance(offsets, Mapping):
        raise ConfigError("real_ready_search.position_offsets_m: required mapping is missing")
    magnitude = search.get("maximum_offset_magnitude_m")
    if (
        not isinstance(magnitude, (int, float))
        or isinstance(magnitude, bool)
        or not 0.0 < float(magnitude) <= 0.20
    ):
        raise ConfigError(
            "real_ready_search.maximum_offset_magnitude_m must lie in (0, 0.20] m: this is an "
            "anchor neighbourhood search, not a workspace optimizer"
        )
    magnitude = float(magnitude)
    for axis in ("forward", "lateral", "normal"):
        values = offsets.get(axis)
        if not isinstance(values, list) or not values:
            raise ConfigError(
                f"real_ready_search.position_offsets_m.{axis} must be a non-empty list"
            )
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(
                    f"real_ready_search.position_offsets_m.{axis} values must be numeric"
                )
            if abs(float(value)) > magnitude:
                raise ConfigError(
                    f"real_ready_search.position_offsets_m.{axis} exceeds "
                    f"maximum_offset_magnitude_m ({magnitude} m)"
                )
        if axis == "normal" and any(float(value) < 0.0 for value in values):
            raise ConfigError(
                "real_ready_search.position_offsets_m.normal must not be negative: a negative "
                "normal offset would aim the search below the registered clearance plane"
            )
    angle = search.get("minimum_axis_angle_deg")
    if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not 0.0 < float(angle) < 90.0:
        raise ConfigError(
            "real_ready_search.minimum_axis_angle_deg must lie in (0, 90) degrees"
        )
    maximum = search.get("candidate_max")
    index = search.get("candidate_index")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ConfigError("real_ready_search.candidate_max must be at least 1")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < maximum:
        raise ConfigError(
            "real_ready_search.candidate_index must lie in [0, candidate_max)"
        )

    selection = config.get("trajectory_selection")
    if not isinstance(selection, Mapping):
        raise ConfigError("trajectory_selection: required mapping is missing")
    limit = selection.get("planning_candidate_limit")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= maximum:
        raise ConfigError(
            "trajectory_selection.planning_candidate_limit must lie in "
            "[1, real_ready_search.candidate_max]"
        )
    order = selection.get("order")
    if not isinstance(order, list) or not order:
        raise ConfigError("trajectory_selection.order must be a non-empty list")
    for entry in order:
        if not isinstance(entry, str) or entry.lstrip("-") not in TRAJECTORY_SELECTION_KEYS:
            raise ConfigError(
                f"trajectory_selection.order entry {entry!r} is not a planned-path metric; "
                f"known: {list(TRAJECTORY_SELECTION_KEYS)}"
            )
    if len({entry.lstrip("-") for entry in order}) != len(order):
        raise ConfigError("trajectory_selection.order must not repeat a metric")

    guards = config.get("commissioning_guards")
    if not isinstance(guards, Mapping):
        raise ConfigError("commissioning_guards: required mapping is missing")
    basis = guards.get("basis")
    if not isinstance(basis, str) or "commissioning" not in basis.lower():
        raise ConfigError(
            "commissioning_guards.basis must name this as a commissioning risk gate, so no "
            "reader can mistake it for a machine specification"
        )
    for key in (
        "min_j4_improvement_rad",
        "max_target_j4_rad",
        "max_path_absolute_j4_rad",
    ):
        value = guards.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ConfigError(f"commissioning_guards.{key} must be positive")
    gate_reference = guards.get("improvement_gate_reference")
    if gate_reference != IMPROVEMENT_GATE_REFERENCE_BASELINE_ONLY_WHEN_ABOVE_CEILING:
        raise ConfigError(
            "commissioning_guards.improvement_gate_reference must be "
            f"{IMPROVEMENT_GATE_REFERENCE_BASELINE_ONLY_WHEN_ABOVE_CEILING}: the J4 "
            "improvement is a gate only while the same-run legacy baseline is itself inside "
            "the failed region. A baseline already below max_target_j4_rad is not the pose "
            "that failed, and at a fixed TCP the elbow angle follows the position, so a "
            "floor measured against it rejects every orientation in the region while looking "
            "like a safety decision"
        )
    if guards.get("baseline_source") != "legacy_pose_ik_solved_in_the_same_run":
        raise ConfigError(
            "commissioning_guards.baseline_source must be "
            "legacy_pose_ik_solved_in_the_same_run: a stored J4 number is an observation from "
            "one afternoon, and the table registration, robot state and scene all move"
        )
    if guards.get("legacy_family") != "legacy_reference":
        raise ConfigError("commissioning_guards.legacy_family must be legacy_reference")
    if guards.get("legacy_is_comparison_only") is not True:
        raise ConfigError(
            "commissioning_guards.legacy_is_comparison_only must be true: the legacy pose is "
            "the configuration that kept failing and must never be the execution target"
        )
    on_no_candidate = guards.get("on_no_admissible_candidate")
    if not isinstance(on_no_candidate, str) or not on_no_candidate.startswith("fail_closed"):
        raise ConfigError(
            "commissioning_guards.on_no_admissible_candidate must start with fail_closed: "
            "having no admissible candidate means no motion, never a fallback to the old pose"
        )
    # A guard that starts quoting motor or thermal vocabulary has stopped being a
    # commissioning observation and started being an invented hardware limit.
    forbidden_vocabulary = ("motor", "thermal", "current", "manufacturer", "physical")
    for key in guards:
        lowered = str(key).lower()
        if any(token in lowered for token in forbidden_vocabulary):
            raise ConfigError(
                f"commissioning_guards.{key}: a J4 guard must not be named after a hardware "
                "limit; this gate comes from repeated real tracking failure, not from a datasheet"
            )

    observations = config.get("known_bad_observations")
    if not isinstance(observations, list) or not observations:
        raise ConfigError("known_bad_observations must be a non-empty list")
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ConfigError(f"known_bad_observations[{index}] must be a mapping")
        if not isinstance(observation.get("source"), str) or not observation["source"].strip():
            raise ConfigError(f"known_bad_observations[{index}].source is required")
        policy = observation.get("policy")
        if not isinstance(policy, str) or "observation" not in policy.lower():
            raise ConfigError(
                f"known_bad_observations[{index}].policy must state that the record is an "
                "observation, not a derived limit"
            )

    _reject_j4_absolute_bounds(config, "ready_search_real")


def validate_named(name: str, config: Mapping[str, Any]) -> None:
    reject_non_finite(config, name)
    if name.startswith("scene_"):
        validate_scene(config)
    elif name == "poses.yaml":
        validate_poses(config)
    elif name == "task_contract.yaml":
        validate_task_contract(config)
    elif name == "gamepad_control.yaml":
        validate_gamepad_control(config)
    elif name == "initialization.yaml":
        validate_initialization(config)
    elif name == "ready_search.yaml":
        validate_ready_search(config)
    elif name == "ready_search_real.yaml":
        validate_real_ready_search(config)
    elif name == "ready_trajectory.yaml":
        validate_ready_trajectory(config)


def has_structured_validator(name: str) -> bool:
    return name.startswith("scene_") or name in {
        "poses.yaml",
        "task_contract.yaml",
        "gamepad_control.yaml",
        "initialization.yaml",
        "ready_search.yaml",
        "ready_search_real.yaml",
        "ready_trajectory.yaml",
    }


def validate_project_configs() -> list[str]:
    names = [
        "scene_nominal.yaml",
        "scene_real_v1.yaml",
        "scene_real_table_v1.yaml",
        "poses.yaml",
        "cameras.yaml",
        "passive_dynamics.yaml",
        "keyboard_control.yaml",
        "gamepad_control.yaml",
        "task_contract.yaml",
        "quality_thresholds.yaml",
        "layout_scan.yaml",
        "recording.yaml",
        "replay.yaml",
        "act_smoke.yaml",
        "real_safety.yaml",
        "initialization.yaml",
        "ready_search.yaml",
        "ready_search_real.yaml",
        "ready_trajectory.yaml",
    ]
    for name in names:
        validate_named(name, load_yaml(name))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    names = validate_project_configs()
    structured = sum(has_structured_validator(name) for name in names)
    print(
        f"PASS: checked {len(names)} project configurations "
        f"({structured} structured, {len(names) - structured} finite-value-only)"
    )


if __name__ == "__main__":
    main()
