"""The Teaching Ready SE(3) pose search itself.

Chain, in order:

    anchor neighbourhood (XYZ) x orientation families
      -> multi-seed collision-aware IK
      -> joint-limit and collision filters
      -> duplicate-solution collapse
      -> static pose scoring
      -> coarse-to-refine second pass
      -> Top-K distinct configurations
      -> initialization trajectory planning (the project's existing RRT-Connect)
      -> trajectory scoring with gravity torque sampled along the path
      -> Top-N report

Two things this module deliberately does not do. It does not execute anything, and it
does not invent a J4 limit. The only J4 numbers it produces are measurements, ranked
against the legacy baseline measured in the same run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..initialization.collision import configuration_valid
from ..initialization.pipeline import _ik_seeds, reference_pose_from_poses, simulation_start_q
from ..initialization.planner import plan_initialization_path
from ..initialization.table_geometry import TableGeometry, sim_table_geometry
from ..initialization.teaching_ready import (
    reference_orientation,
    rotation_error_rad,
    rotation_to_quaternion_xyzw,
)
from ..planning.kinematics import solve_site_pose
from ..scene.model import ModelMap, load_model, reset_task_ready
from .config import ReadySearchConfig, refinement_offsets
from .metrics import (
    J4_JOINT_NAME,
    clearance_pairs,
    configuration_difference,
    gravity_torque_metrics,
    joint_distance,
    joint_metrics,
    minimum_clearance,
    static_gravity_torque,
    tcp_height_above_table,
    trajectory_gravity_torque,
    trajectory_travel_metrics,
    yoshikawa_manipulability,
)
from .scoring import (
    resolve_references,
    static_metric_values_for_reference_resolution,
    static_score,
    trajectory_score,
)
from .tool_frame import (
    OrientationCandidate,
    ToolFrame,
    orientation_candidates,
    tool_frame,
    with_roll_offset,
)

FILTER_IK_NONCONVERGENT = "ik_nonconvergence"
FILTER_POSITION_TOLERANCE = "position_tolerance"
FILTER_ORIENTATION_TOLERANCE = "orientation_tolerance"
FILTER_JOINT_LIMIT_MARGIN = "joint_limit_margin"
FILTER_JOINT_MARGIN_BELOW_MINIMUM = "joint_margin_below_minimum"
FILTER_J4_MARGIN_BELOW_MINIMUM = "j4_margin_below_minimum"
FILTER_MANIPULABILITY_BELOW_MINIMUM = "manipulability_below_minimum"
FILTER_COLLISION = "collision"
FILTER_FOOTPRINT = "outside_table_footprint"
FILTER_HEIGHT = "height_above_table_out_of_range"


def _round(value: float, digits: int) -> float:
    return float(round(float(value), int(digits)))


def _round_nested(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return _round(value, digits)
    if isinstance(value, Mapping):
        return {key: _round_nested(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_nested(item, digits) for item in value]
    return value


@dataclass
class TargetPosition:
    """One admissible TCP position from the anchor neighbourhood."""

    offset_m: tuple[float, float, float]
    world_m: tuple[float, float, float]
    table_xy_m: tuple[float, float]
    height_above_table_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset_from_anchor_m": list(self.offset_m),
            "world_m": list(self.world_m),
            "table_xy_m": list(self.table_xy_m),
            "height_above_table_m": float(self.height_above_table_m),
        }


@dataclass
class Candidate:
    """One collision-free configuration reaching one admissible TCP pose."""

    candidate_id: str
    family: str
    label: str
    pass_name: str
    orientation: dict[str, Any]
    target: TargetPosition
    seed_index: int
    q: np.ndarray
    ik: dict[str, Any]
    static_metrics: dict[str, Any]
    static_score: dict[str, Any] | None = None
    trajectory: dict[str, Any] | None = None
    trajectory_score: dict[str, Any] | None = None
    duplicate_of: str | None = None
    duplicate_count: int = 0
    same_configuration_as: str | None = None
    reference_role: str | None = None

    @property
    def score(self) -> float:
        if self.trajectory_score is not None:
            return float(self.trajectory_score["score"])
        if self.static_score is not None:
            return float(self.static_score["score"])
        return float("-inf")

    def as_dict(self, *, digits: int | None = None, include_metrics: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "label": self.label,
            "pass": self.pass_name,
            "reference_role": self.reference_role,
            "orientation": self.orientation,
            "target": self.target.as_dict(),
            "q_rad": self.q.tolist(),
            "seed_index": int(self.seed_index),
            "ik": self.ik,
            "static_score_value": None if self.static_score is None else float(self.static_score["score"]),
            "trajectory_score_value": None if self.trajectory_score is None else float(self.trajectory_score["score"]),
            "duplicate_of": self.duplicate_of,
            "duplicate_count": int(self.duplicate_count),
            "same_configuration_as": self.same_configuration_as,
        }
        if include_metrics:
            payload["static_metrics"] = self.static_metrics
            payload["static_score"] = self.static_score
            payload["trajectory"] = self.trajectory
            payload["trajectory_score"] = self.trajectory_score
        return _round_nested(payload, digits) if digits is not None else payload

    def summary(self, *, digits: int | None = None) -> dict[str, Any]:
        """The compact record the Top-N list publishes."""
        metrics = self.static_metrics
        payload = {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "label": self.label,
            "reference_role": self.reference_role,
            "tcp_pose": {
                "position_world_m": list(self.target.world_m),
                "orientation_xyzw": self.orientation["orientation_xyzw"],
                "approach_axis_world": self.orientation["approach_axis_world"],
                "tilt_below_horizontal_deg": self.orientation["tilt_below_horizontal_deg"],
                "heading_deg": self.orientation["heading_deg"],
                "roll_deg": self.orientation["roll_deg"],
            },
            "q_rad": self.q.tolist(),
            "j4_rad": float(metrics["j4_rad"]),
            "joint_margin_normalized": float(metrics["minimum_joint_margin_normalized"]),
            "joint_margin_rad": float(metrics["minimum_joint_margin_rad"]),
            "j4_margin_rad": float(metrics["j4_margin_rad"]),
            "j4_margin_normalized": float(metrics["j4_margin_normalized"]),
            "manipulability": float(metrics["yoshikawa_manipulability"]),
            "minimum_clearance_m": float(metrics["minimum_clearance_m"]),
            "minimum_task_obstacles_clearance_m": float(metrics["minimum_task_obstacles_clearance_m"]),
            "minimum_structure_clearance_m": float(metrics["minimum_structure_clearance_m"]),
            "minimum_self_clearance_m": float(metrics["minimum_self_clearance_m"]),
            "start_joint_distance_rad": float(metrics["start_joint_distance_rad"]),
            "static_gravity_torque_j4_nm": float(metrics["gravity_torque_j4_nm"]),
            "static_absolute_gravity_torque_j4_nm": float(metrics["absolute_gravity_torque_j4_nm"]),
            "static_gravity_torque_max_all_joints_nm": float(metrics["maximum_absolute_gravity_torque_nm"]),
            "static_score": None if self.static_score is None else float(self.static_score["score"]),
            "trajectory_score": None if self.trajectory_score is None else float(self.trajectory_score["score"]),
        }
        if self.trajectory is not None:
            payload.update(
                {
                    "trajectory_planned": bool(self.trajectory["success"]),
                    "trajectory_failure": self.trajectory.get("failure"),
                    "trajectory_joint_path_length_rad": self.trajectory.get("joint_path_length_rad"),
                    "trajectory_maximum_joint_travel_rad": self.trajectory.get("maximum_joint_travel_rad"),
                    "trajectory_j4_travel_rad": self.trajectory.get("j4_travel_rad"),
                    "trajectory_duration_s": self.trajectory.get("duration_s"),
                    "trajectory_maximum_velocity_rad_s": self.trajectory.get("maximum_velocity_rad_s"),
                    "trajectory_maximum_acceleration_rad_s2": self.trajectory.get("maximum_acceleration_rad_s2"),
                    "trajectory_maximum_absolute_j4_gravity_torque_nm": self.trajectory.get(
                        "maximum_absolute_j4_gravity_torque_nm"
                    ),
                    "trajectory_mean_absolute_j4_gravity_torque_nm": self.trajectory.get(
                        "mean_absolute_j4_gravity_torque_nm"
                    ),
                    "trajectory_endpoint_j4_gravity_torque_nm": self.trajectory.get(
                        "endpoint_j4_gravity_torque_nm"
                    ),
                    "trajectory_minimum_clearance_m": self.trajectory.get("minimum_clearance_m"),
                    "trajectory_minimum_task_obstacles_clearance_m": self.trajectory.get(
                        "minimum_task_obstacles_clearance_m"
                    ),
                }
            )
        return _round_nested(payload, digits) if digits is not None else payload


# --------------------------------------------------------------------------------------
# Duplicate handling
# --------------------------------------------------------------------------------------


def _raw_preference(candidate: Candidate) -> tuple[float, float, float, float, str]:
    """Reference-free ordering used to choose which duplicate to keep.

    Deliberately independent of the scoring weights: collapsing duplicates must not
    depend on a reference scale that is itself resolved later from the baseline.
    """
    metrics = candidate.static_metrics
    return (
        -float(metrics["minimum_joint_margin_normalized"]),
        -float(metrics["j4_margin_normalized"]),
        -float(metrics["yoshikawa_manipulability"]),
        abs(float(metrics["gravity_torque_j4_nm"])),
        candidate.candidate_id,
    )


def deduplicate(
    candidates: Sequence[Candidate],
    *,
    threshold_rad: float,
    group_key: Callable[[Candidate], Any] | None = None,
) -> tuple[list[Candidate], int]:
    """Collapse solutions that describe the same physical configuration.

    IK is seeded several times per pose on purpose, so the same joint branch is
    routinely found more than once. Keeping every copy would both inflate the candidate
    counts and let one pose occupy several Top-K slots.
    """
    groups: dict[Any, list[Candidate]] = {}
    for candidate in candidates:
        key = group_key(candidate) if group_key is not None else None
        groups.setdefault(key, []).append(candidate)

    kept: list[Candidate] = []
    dropped = 0
    for _, group in groups.items():
        ordered = sorted(group, key=_raw_preference)
        survivors: list[Candidate] = []
        for candidate in ordered:
            duplicate_of = None
            for survivor in survivors:
                if configuration_difference(candidate.q, survivor.q) < float(threshold_rad):
                    duplicate_of = survivor.candidate_id
                    break
            if duplicate_of is None:
                survivors.append(candidate)
            else:
                candidate.duplicate_of = duplicate_of
                dropped += 1
        kept.extend(survivors)
    kept.sort(key=lambda item: item.candidate_id)
    for survivor in kept:
        survivor.duplicate_count = sum(
            1 for item in candidates if item.duplicate_of == survivor.candidate_id
        )
    return kept, dropped


# --------------------------------------------------------------------------------------
# Search driver
# --------------------------------------------------------------------------------------


@dataclass
class SearchState:
    filter_counts: dict[str, int] = field(default_factory=dict)
    rejected_targets: list[dict[str, Any]] = field(default_factory=list)

    def count(self, reason: str, amount: int = 1) -> None:
        self.filter_counts[reason] = self.filter_counts.get(reason, 0) + int(amount)

    def count_collisions(self, labels: Iterable[str]) -> None:
        for label in labels or ("unspecified",):
            self.count(f"{FILTER_COLLISION}:{label}")

    def reject_target(self, reason: str, offset: Sequence[float] | np.ndarray, detail: Mapping[str, Any]) -> None:
        """Record a rejected TCP position so a rejection is evidence, not silence."""
        self.count(reason)
        self.rejected_targets.append(
            {"reason": reason, "offset_from_anchor_m": [float(value) for value in offset], **dict(detail)}
        )


class TeachingReadySearch:
    """Search over oriented TCP poses for the initialization handoff."""

    def __init__(
        self,
        config: ReadySearchConfig,
        *,
        model=None,
        data=None,
        mapping: ModelMap | None = None,
    ):
        self.config = config
        if model is None or data is None or mapping is None:
            model, data, mapping = build_search_scene(config)
        self.model = model
        self.data = data
        self.mapping = mapping
        self.geometry: TableGeometry = sim_table_geometry(model, data, config.initialization)
        self.frame: ToolFrame = tool_frame(model, data, self.geometry)
        self.reference_q = reference_pose_from_poses(config.initialization)
        self.reference_matrix = reference_orientation(model, data, mapping, self.reference_q)
        self.start_q = simulation_start_q(config.initialization)
        self.seeds = [np.asarray(seed, dtype=float) for seed in _ik_seeds(self.reference_q)]
        self.clearance = clearance_pairs(model, data, mapping, self.reference_q)
        self.state = SearchState()

        offset_x, offset_y, height = config.table_anchor_offsets()
        self.legacy_anchor_m = np.asarray(
            [self.geometry.center[0] + offset_x, self.geometry.center[1] + offset_y, self.geometry.top_z + height],
            dtype=float,
        )
        if config.position_anchor_source == "table_geometric_centerline":
            self.anchor_m = np.asarray(
                [self.geometry.center[0], self.geometry.center[1], self.geometry.top_z + height],
                dtype=float,
            )
        elif config.position_anchor_source == "operator_marked_table_point":
            anchor_x, anchor_y = config.position_anchor_table_offsets()
            planar = (
                anchor_x * self.frame.table_forward_world
                + anchor_y * self.frame.table_lateral_world
            )
            self.anchor_m = np.asarray(
                [self.geometry.center[0], self.geometry.center[1], self.geometry.top_z + height],
                dtype=float,
            ) + planar
        else:
            self.anchor_m = self.legacy_anchor_m.copy()
        self.orientations = orientation_candidates(self.frame, self.reference_matrix, config.families)

    # -- targets ---------------------------------------------------------------------

    def admissible_targets(self, offsets: np.ndarray, base: np.ndarray | None = None) -> list[TargetPosition]:
        """Keep only offsets whose TCP stays inside the table footprint and height band.

        ``base`` shifts the whole neighbourhood, which the refinement pass uses to
        re-search around the cells the coarse pass liked.
        """
        origin = self.anchor_m if base is None else np.asarray(base, dtype=float)
        margin = self.config.footprint_margin_m
        minimum_height = self.config.minimum_height_above_table_m
        maximum_height = self.config.maximum_height_above_table_m
        half = self.geometry.half_extent
        targets: list[TargetPosition] = []
        for offset in np.atleast_2d(np.asarray(offsets, dtype=float)):
            world = origin + offset
            table_xy = self.geometry.to_table_xy(world[:2])
            # A small absolute slack keeps an exactly-on-the-boundary value (the anchor
            # itself sits on the configured margin) from flipping on floating point.
            inside = bool(np.all(np.abs(table_xy) + margin <= half + 1e-9))
            if not inside:
                self.state.reject_target(
                    FILTER_FOOTPRINT,
                    offset,
                    {
                        "world_m": [float(value) for value in world],
                        "table_xy_m": [float(table_xy[0]), float(table_xy[1])],
                        "half_extent_m": [float(half[0]), float(half[1])],
                        "required_margin_m": float(margin),
                    },
                )
                continue
            height = float(world[2] - self.geometry.top_z)
            if not minimum_height <= height <= maximum_height:
                self.state.reject_target(
                    FILTER_HEIGHT,
                    offset,
                    {
                        "world_m": [float(value) for value in world],
                        "height_above_table_m": height,
                        "allowed_height_m": [float(minimum_height), float(maximum_height)],
                    },
                )
                continue
            targets.append(
                TargetPosition(
                    offset_m=tuple(float(value) for value in offset),
                    world_m=tuple(float(value) for value in world),
                    table_xy_m=(float(table_xy[0]), float(table_xy[1])),
                    height_above_table_m=height,
                )
            )
        return targets

    # -- evaluation ------------------------------------------------------------------

    def _solve(self, orientation: OrientationCandidate, target: TargetPosition) -> list[Candidate]:
        ik_block = self.config.ik
        position_tolerance = float(ik_block["position_tolerance_m"])
        orientation_tolerance = float(ik_block["orientation_tolerance_rad"])
        iterations = int(ik_block["iterations"])
        margin_rad = float(ik_block["joint_margin_rad"])
        forbid_cube = bool(self.config.filters.get("forbid_cube_contact", True))
        goal = np.asarray(target.world_m, dtype=float)
        produced: list[Candidate] = []

        for seed_index, seed in enumerate(self.seeds):
            ik = solve_site_pose(
                self.model,
                self.data,
                self.mapping,
                goal,
                orientation.orientation_matrix,
                seed,
                tolerance_m=position_tolerance,
                orientation_tolerance_rad=orientation_tolerance,
                iterations=iterations,
                margin_rad=margin_rad,
            )
            if not ik.success:
                self.state.count(FILTER_IK_NONCONVERGENT)
                continue
            position_error, orientation_error = self._measured_errors(goal, orientation.orientation_matrix)
            if position_error > position_tolerance:
                self.state.count(FILTER_POSITION_TOLERANCE)
                continue
            if orientation_error > orientation_tolerance:
                self.state.count(FILTER_ORIENTATION_TOLERANCE)
                continue
            valid, failures = configuration_valid(
                self.model,
                self.data,
                self.mapping,
                ik.q,
                margin_rad=margin_rad,
                forbid_cube_contact=forbid_cube,
            )
            if not valid:
                if "joint_limit_margin" in failures:
                    self.state.count(FILTER_JOINT_LIMIT_MARGIN)
                else:
                    self.state.count_collisions(sorted(failures))
                continue
            metrics = self._static_metrics(ik.q)
            if metrics["minimum_joint_margin_normalized"] < float(self.config.filters["minimum_joint_margin_normalized"]):
                self.state.count(FILTER_JOINT_MARGIN_BELOW_MINIMUM)
                continue
            if metrics["j4_margin_normalized"] < float(self.config.filters["minimum_j4_margin_normalized"]):
                self.state.count(FILTER_J4_MARGIN_BELOW_MINIMUM)
                continue
            if metrics["yoshikawa_manipulability"] < float(self.config.filters.get("minimum_manipulability", 0.0)):
                self.state.count(FILTER_MANIPULABILITY_BELOW_MINIMUM)
                continue
            produced.append(
                Candidate(
                    candidate_id=self._candidate_id(orientation.label, target, "", ik.q),
                    family=orientation.family,
                    label=orientation.label,
                    pass_name="",
                    orientation=orientation.as_dict(),
                    target=target,
                    seed_index=seed_index,
                    q=np.asarray(ik.q, dtype=float).copy(),
                    ik={
                        "position_error_m": float(position_error),
                        "orientation_error_rad": float(orientation_error),
                        "iterations": int(ik.iterations),
                    },
                    static_metrics=metrics,
                )
            )
        return produced

    def _measured_errors(self, goal: np.ndarray, orientation: np.ndarray) -> tuple[float, float]:
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        position_error = float(np.linalg.norm(goal - self.data.site_xpos[self.mapping.grasp_site]))
        orientation_error = rotation_error_rad(
            np.asarray(self.data.site_xmat[self.mapping.grasp_site], dtype=float).reshape(3, 3), orientation
        )
        return position_error, orientation_error

    def _candidate_id(
        self, label: str, target: TargetPosition, pass_name: str, q: Sequence[float]
    ) -> str:
        offsets = "".join(f"{value:+.3f}" for value in target.offset_m)
        position = f"{target.world_m[0]:.4f},{target.world_m[1]:.4f},{target.world_m[2]:.4f}"
        configuration = ",".join(f"{float(value):+.5f}" for value in q)
        return f"{label}|{pass_name}|{offsets}|{position}|q={configuration}"

    def _static_metrics(self, q: np.ndarray) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metrics.update(joint_metrics(self.model, self.mapping, q))
        metrics.update(yoshikawa_manipulability(self.model, self.data, self.mapping, q))
        metrics.update(
            minimum_clearance(
                self.model,
                self.data,
                self.mapping,
                q,
                self.clearance,
                cap_m=float(self.config.metrics["clearance_cap_m"]),
            )
        )
        torque = static_gravity_torque(self.model, self.data, self.mapping, q)
        metrics.update(gravity_torque_metrics(torque))
        metrics["start_joint_distance_rad"] = joint_distance(q, self.start_q)
        metrics["legacy_reference_distance_rad"] = joint_distance(q, self.reference_q)
        metrics["tcp_height_above_table_m"] = tcp_height_above_table(
            self.model, self.data, self.mapping, q, self.geometry.top_z
        )
        return metrics

    def evaluate(
        self,
        orientations: Sequence[OrientationCandidate],
        targets: Sequence[TargetPosition],
        *,
        pass_name: str,
    ) -> list[Candidate]:
        """Solve every orientation/position pair and keep the physically valid solutions."""
        produced: list[Candidate] = []
        threshold = float(self.config.ik["dedup_max_joint_distance_rad"])
        for orientation in orientations:
            for target in targets:
                solved = self._solve(orientation, target)
                if not solved:
                    continue
                kept, _ = deduplicate(
                    solved,
                    threshold_rad=threshold,
                    group_key=lambda candidate: (candidate.label, candidate.target.world_m),
                )
                for candidate in kept:
                    candidate.pass_name = pass_name
                    candidate.candidate_id = self._candidate_id(
                        orientation.label, target, pass_name, candidate.q
                    )
                produced.extend(kept)
        return produced

    # -- baseline and scoring --------------------------------------------------------

    def legacy_baseline(self, candidates: Sequence[Candidate]) -> Candidate | None:
        """The configuration the previous fixed chain would have selected.

        The existing offset scan selects the anchor offset and then the collision-free
        IK branch with the largest normalized joint-limit margin. Reproducing that rule
        here is what makes "better than the legacy baseline" a statement about the pose
        the real arm actually failed to reach.
        """
        legacy = [candidate for candidate in candidates if candidate.family == "legacy_pi2"]
        if not legacy:
            return None
        anchor = [
            candidate
            for candidate in legacy
            if all(abs(value) < 1e-9 for value in candidate.target.offset_m)
        ]
        pool = anchor or legacy
        return min(pool, key=_raw_preference)

    def score_static(self, candidates: Sequence[Candidate], baseline: Candidate | None) -> dict[str, Any]:
        legacy_metrics = (
            None if baseline is None else static_metric_values_for_reference_resolution(baseline.static_metrics)
        )
        references = resolve_references(self.config.static_references, legacy_metrics)
        for candidate in candidates:
            candidate.static_score = static_score(candidate.static_metrics, self.config.static_weights, references)
        return references

    def plan(self, candidate: Candidate) -> dict[str, Any]:
        """Plan and measure the initialization trajectory into ``candidate``."""
        from ..initialization.planner import PlannerResult

        planning = dict(self.config.initialization.get("planning", {}))
        planning["maximum_iterations"] = int(self.config.raw["planning"]["maximum_iterations"])
        planning_config = {**self.config.initialization, "planning": planning}
        result: PlannerResult = plan_initialization_path(
            self.model,
            self.data,
            self.mapping,
            self.start_q,
            candidate.q,
            planning_config,
            geometry=self.geometry,
        )
        metrics: dict[str, Any] = {
            "success": bool(result.success),
            "failure": result.failure,
            "collision": list(result.collision),
            "iterations": int(result.iterations),
            "smoothed": bool(result.smoothed),
            "waypoint_count": len(result.waypoints),
            "straight_line_length_rad": float(result.straight_line_length_rad),
            "maximum_deviation_from_straight_rad": float(result.maximum_deviation_from_straight_rad),
            "joint_path_length_rad": float(result.joint_path_length_rad),
            "duration_s": float(result.duration_s),
            "maximum_velocity_rad_s": float(result.maximum_velocity_rad_s),
            "maximum_acceleration_rad_s2": float(result.maximum_acceleration_rad_s2),
            "minimum_joint_margin_normalized": float(result.minimum_joint_margin_normalized),
            "minimum_tcp_table_clearance_m": float(result.minimum_tcp_table_clearance_m),
            "planner": "rrt_connect",
        }
        travel = trajectory_travel_metrics(result.waypoints)
        metrics.update(travel)
        if result.success and result.waypoints:
            samples = self._trajectory_samples(result.waypoints)
            torque = trajectory_gravity_torque(
                self.model,
                self.data,
                self.mapping,
                result.waypoints,
                sample_count=int(self.config.metrics["trajectory_sample_count"]),
            )
            metrics.update(torque)
            clearance = self._trajectory_clearance(samples)
            metrics.update(clearance)
        else:
            metrics.update(
                {
                    "maximum_absolute_j4_gravity_torque_nm": None,
                    "mean_absolute_j4_gravity_torque_nm": None,
                    "endpoint_j4_gravity_torque_nm": None,
                    "maximum_absolute_gravity_torque_nm": None,
                    "minimum_clearance_m": None,
                }
            )
        velocity_limit = float(self.config.initialization["planning"]["time_scaling"]["maximum_joint_velocity_rad_s"])
        acceleration_limit = float(
            self.config.initialization["planning"]["time_scaling"]["maximum_joint_acceleration_rad_s2"]
        )
        metrics["velocity_limit_rad_s"] = velocity_limit
        metrics["acceleration_limit_rad_s2"] = acceleration_limit
        metrics["within_time_scaling_limits"] = bool(
            metrics["maximum_velocity_rad_s"] <= velocity_limit + 1e-9
            and metrics["maximum_acceleration_rad_s2"] <= acceleration_limit + 1e-9
        )
        return metrics

    def _trajectory_samples(self, waypoints: Sequence[np.ndarray]) -> list[np.ndarray]:
        from .metrics import _resample_path

        return _resample_path([np.asarray(point, dtype=float) for point in waypoints], int(self.config.metrics["trajectory_sample_count"]))

    def _trajectory_clearance(self, samples: Sequence[np.ndarray]) -> dict[str, Any]:
        """Smallest clearance along the path, kept apart for obstacles and for self.

        The obstacle value is the safety statement; the overall minimum is the
        no-penetration statement the acceptance check uses.
        """
        cap = float(self.config.metrics["clearance_cap_m"])
        worst = cap
        worst_obstacle = cap
        worst_pair: str | None = None
        for q in samples:
            measured = minimum_clearance(self.model, self.data, self.mapping, q, self.clearance, cap_m=cap)
            if measured["minimum_clearance_m"] < worst:
                worst = float(measured["minimum_clearance_m"])
                worst_pair = measured["minimum_clearance_pair"]
            worst_obstacle = min(worst_obstacle, float(measured["minimum_task_obstacles_clearance_m"]))
        return {
            "minimum_clearance_m": float(worst),
            "minimum_task_obstacles_clearance_m": float(worst_obstacle),
            "minimum_clearance_pair": worst_pair,
        }

    def score_trajectory(self, candidates: Sequence[Candidate], baseline: Candidate | None) -> dict[str, Any]:
        legacy_metrics: dict[str, float] | None = None
        if baseline is not None and baseline.trajectory is not None and baseline.trajectory.get("success"):
            legacy_metrics = {
                key: float(value)
                for key, value in baseline.trajectory.items()
                if isinstance(value, (int, float)) and value is not None
            }
        references = resolve_references(self.config.trajectory_references, legacy_metrics)
        for candidate in candidates:
            if candidate.trajectory is None or not candidate.trajectory.get("success"):
                candidate.trajectory_score = None
                continue
            if candidate.static_score is None:  # pragma: no cover - defensive
                continue
            candidate.trajectory_score = trajectory_score(
                endpoint_static_score=float(candidate.static_score["score"]),
                trajectory_metrics=candidate.trajectory,
                endpoint_weight=self.config.trajectory_endpoint_weight,
                weights=self.config.trajectory_weights,
                references=references,
            )
        return references

    # -- refinement and planning queue -----------------------------------------------

    def refine_seeds(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Best distinct orientation/position cells that the refine pass re-searches."""
        scored = [
            candidate
            for candidate in candidates
            if candidate.static_score is not None and candidate.family != "legacy_pi2"
        ]
        ordered = sorted(scored, key=lambda item: (-item.static_score["score"], item.candidate_id))
        cells: list[Candidate] = []
        seen: set[tuple[str, int, int]] = set()
        for candidate in ordered:
            key = (
                candidate.label,
                int(round(candidate.target.world_m[0] * 100.0)),
                int(round(candidate.target.world_m[1] * 100.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            cells.append(candidate)
            if len(cells) >= self.config.refine_seed_cells:
                break
        return cells

    def planning_queue(self, candidates: Sequence[Candidate], baseline: Candidate | None) -> list[Candidate]:
        """Configurations that get a full initialization plan.

        Three groups are planned, because planning is cheap (about 50 ms) and each group
        answers a different question:

        * the legacy baseline, whose trajectory is the reference scale every other
          trajectory is compared against - it is planned even when it does not rank;
        * the best candidate of every orientation label, so the comparison spans the
          whole family sweep (0/30/40/50/60/90 degrees and the heading offsets) instead
          of collapsing onto whichever label scored best statically;
        * the ``top_k`` highest static scores, which is the ranking the reconfigured
          weights would produce.
        """
        queue: list[Candidate] = []
        queued: list[tuple[np.ndarray, Candidate]] = []
        threshold = float(self.config.ik["dedup_max_joint_distance_rad"])

        def push(candidate: Candidate | None, role: str) -> None:
            if candidate is None or candidate.trajectory is not None:
                return
            for existing_q, existing in queued:
                if existing is candidate:
                    return
                if configuration_difference(candidate.q, existing_q) < threshold:
                    candidate.same_configuration_as = existing.candidate_id
                    return
            candidate.reference_role = role
            queued.append((candidate.q, candidate))
            queue.append(candidate)

        push(baseline, "legacy_baseline")
        ranked = sorted(
            (item for item in candidates if item.static_score is not None),
            key=lambda item: (-item.static_score["score"], item.candidate_id),
        )
        by_label: dict[str, Candidate] = {}
        by_family: dict[str, Candidate] = {}
        for candidate in ranked:
            by_label.setdefault(candidate.label, candidate)
            by_family.setdefault(candidate.family, candidate)
        for candidate in by_label.values():
            push(candidate, "orientation_label_best")
        for family in self.config.enabled_families():
            push(by_family.get(family.name), f"family_best:{family.name}")
        added = 0
        for candidate in ranked:
            if added >= self.config.top_k:
                break
            before = len(queue)
            push(candidate, "static_top_k")
            if len(queue) > before:
                added += 1
        return queue

    # -- top level -------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        from .report import build_report

        # Re-runnable: counters and rejection records belong to one run, and the
        # determinism check compares two independent runs of the same instance.
        self.state = SearchState()

        legacy_orientations = [
            item for item in self.orientations if item.family == "legacy_pi2"
        ]
        search_orientations = [
            item for item in self.orientations if item.family != "legacy_pi2"
        ]
        legacy_offset = self.legacy_anchor_m - self.anchor_m
        baseline_targets = self.admissible_targets(
            np.asarray([legacy_offset], dtype=float)
        )
        baseline_candidates = self.evaluate(
            legacy_orientations, baseline_targets, pass_name="baseline"
        )
        coarse_targets = self.admissible_targets(self.config.coarse_offsets_m)
        coarse = baseline_candidates + self.evaluate(
            search_orientations, coarse_targets, pass_name="coarse"
        )
        baseline = self.legacy_baseline(coarse)
        static_references = self.score_static(coarse, baseline)

        refine_candidates: list[Candidate] = []
        refine_details: list[dict[str, Any]] = []
        if self.config.refine_enabled and self.config.refine_seed_cells > 0:
            for seed in self.refine_seeds(coarse):
                offsets = refinement_offsets(
                    np.zeros(3), self.config.refine_window_m, self.config.refine_step_m
                )
                targets = self.admissible_targets(offsets, base=np.asarray(seed.target.world_m, dtype=float))
                if not targets:
                    continue
                seed_orientation = next(
                    (item for item in self.orientations if item.label == seed.label), None
                )
                if seed_orientation is None:
                    continue
                orientations = [
                    seed_orientation
                    if offset == 0.0
                    else with_roll_offset(self.frame, self.reference_matrix, seed_orientation, offset)
                    for offset in self.config.refine_roll_offsets_deg
                ]
                produced = self.evaluate(orientations, targets, pass_name="refine")
                refine_candidates.extend(produced)
                refine_details.append(
                    {
                        "seed_candidate_id": seed.candidate_id,
                        "seed_family": seed.family,
                        "seed_label": seed.label,
                        "seed_static_score": float(seed.static_score["score"]),
                        "world_m": list(seed.target.world_m),
                        "window_m": self.config.refine_window_m,
                        "step_m": self.config.refine_step_m,
                        "roll_offsets_deg": list(self.config.refine_roll_offsets_deg),
                        "candidate_count": len(produced),
                    }
                )

        all_candidates, duplicates = deduplicate(
            list(coarse) + list(refine_candidates),
            threshold_rad=float(self.config.ik["dedup_max_joint_distance_rad"]),
        )
        baseline = self.legacy_baseline(all_candidates)
        static_references = self.score_static(all_candidates, baseline)
        refine_seed_ids = {detail["seed_candidate_id"] for detail in refine_details}

        plan_counts = {"planned": 0, "failed": 0}
        for candidate in self.planning_queue(all_candidates, baseline):
            candidate.trajectory = self.plan(candidate)
            if candidate.trajectory["success"]:
                plan_counts["planned"] += 1
            else:
                plan_counts["failed"] += 1
        trajectory_references = self.score_trajectory(all_candidates, baseline)

        return build_report(
            config=self.config,
            search=self,
            geometry=self.geometry,
            frame=self.frame,
            candidates=all_candidates,
            baseline=baseline,
            static_references=static_references,
            trajectory_references=trajectory_references,
            duplicate_count=duplicates,
            plan_counts=plan_counts,
            refine_details=refine_details,
            refine_seed_ids=refine_seed_ids,
        )


def build_search_scene(config: ReadySearchConfig):
    """Compile the configured scene and install the configured start state."""
    import mujoco

    model = load_model(config.scene)
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    start = simulation_start_q(config.initialization)
    data.qpos[mapping.right_qpos] = start
    data.qvel[:] = 0.0
    data.ctrl[mapping.right_actuators] = start
    mujoco.mj_forward(model, data)
    return model, data, mapping


def run_ready_search(
    config: ReadySearchConfig | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the search and return the report dictionary."""
    from .config import load_ready_search_config

    resolved = config if config is not None else load_ready_search_config(overrides)
    return TeachingReadySearch(resolved).run()


def top_n_signature(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Deterministic identity of the Top-N list, used to prove reproducibility."""
    return [(item["label"], item["candidate_id"]) for item in report.get("top_n", [])]


def verify_determinism(config: ReadySearchConfig | None = None, *, runs: int = 2) -> dict[str, Any]:
    """Run the search repeatedly and compare the Top-N identity and scores."""
    reports = [run_ready_search(config) for _ in range(max(2, int(runs)))]
    signatures = [top_n_signature(report) for report in reports]
    scores = [[float(item["trajectory_score"]) for item in report.get("top_n", [])] for report in reports]
    identical = all(signature == signatures[0] for signature in signatures)
    scores_identical = all(
        len(item) == len(scores[0]) and all(abs(a - b) < 1e-9 for a, b in zip(item, scores[0])) for item in scores
    )
    return {
        "runs": len(reports),
        "signatures": [[list(entry) for entry in signature] for signature in signatures],
        "scores": scores,
        "top_n_identical": bool(identical),
        "scores_identical": bool(scores_identical),
        "deterministic": bool(identical and scores_identical),
    }


def reference_orientation_summary(frame: ToolFrame, matrix: np.ndarray) -> dict[str, Any]:
    """The legacy reference orientation, described in the same terms as the families."""
    return {
        "orientation_matrix": np.asarray(matrix, dtype=float).tolist(),
        "orientation_xyzw": rotation_to_quaternion_xyzw(matrix),
        "approach_axis_world": frame.approach_axis_world(matrix).tolist(),
        "tilt_below_horizontal_deg": frame.tilt_below_horizontal_deg(matrix),
        "heading_deg": frame.heading_deg(matrix),
        "joint_name": J4_JOINT_NAME,
        "note": (
            "Forward kinematics of the configured reference pose. Its approach axis is close to "
            "straight down, so the legacy pose is the 90 degree end of the same tilt scale the new "
            "families sample."
        ),
    }
