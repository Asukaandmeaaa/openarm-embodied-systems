"""Traditional sampling-based planning from the measured current state to teaching-ready.

This is deliberately *not* ``np.linspace(start_q, goal_q)``. Straight joint
interpolation between the current pose and the teaching-ready pose is collision
checked and rejected on this station (the arm sweeps through the pedestal / box
region), which is exactly why a sampling planner is required.

The planner is a bidirectional RRT-Connect over the 7 right-arm joints with:
  * joint-limit margin enforced per sample,
  * collision checking per extension and per connecting segment,
  * optional shortcut smoothing that only ever replaces a path with a
    collision-checked shorter one,
  * zero-velocity quintic blend time scaling, matching the project's existing
    conservative convention, so velocity/acceleration limits are reported rather
    than assumed.

The start state is always the measured/current configuration passed in by the
caller. Nothing here assumes mechanical zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .collision import configuration_valid, segment_valid, tcp_table_clearance
from ..planning.kinematics import joint_limits, normalized_joint_margin

PLAN_FAILED = "SIM_INITIALIZATION_PLAN_FAILED"
START_INVALID = "START_STATE_CHANGED"


@dataclass
class PlannerResult:
    success: bool
    start_q: np.ndarray
    goal_q: np.ndarray
    waypoints: list[np.ndarray] = field(default_factory=list)
    iterations: int = 0
    failure: str | None = None
    collision: tuple[str, ...] = ()
    joint_path_length_rad: float = 0.0
    straight_line_length_rad: float = 0.0
    maximum_deviation_from_straight_rad: float = 0.0
    minimum_joint_margin_normalized: float = 0.0
    minimum_tcp_table_clearance_m: float = 0.0
    duration_s: float = 0.0
    maximum_velocity_rad_s: float = 0.0
    maximum_acceleration_rad_s2: float = 0.0
    smoothed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "start_q": self.start_q.tolist(),
            "goal_q": self.goal_q.tolist(),
            "waypoints": [waypoint.tolist() for waypoint in self.waypoints],
            "waypoint_count": len(self.waypoints),
            "iterations": int(self.iterations),
            "failure": self.failure,
            "collision": list(self.collision),
            "joint_path_length_rad": float(self.joint_path_length_rad),
            "straight_line_length_rad": float(self.straight_line_length_rad),
            "maximum_deviation_from_straight_rad": float(self.maximum_deviation_from_straight_rad),
            "minimum_joint_margin_normalized": float(self.minimum_joint_margin_normalized),
            "minimum_tcp_table_clearance_m": float(self.minimum_tcp_table_clearance_m),
            "duration_s": float(self.duration_s),
            "maximum_velocity_rad_s": float(self.maximum_velocity_rad_s),
            "maximum_acceleration_rad_s2": float(self.maximum_acceleration_rad_s2),
            "smoothed": self.smoothed,
            "planner": "rrt_connect",
        }


def _nearest(tree: list[np.ndarray], target: np.ndarray) -> int:
    distances = [float(np.linalg.norm(node - target)) for node in tree]
    return int(np.argmin(distances))


def _chain(tree: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
    chain: list[np.ndarray] = []
    while index >= 0:
        chain.append(tree[index])
        index = parents[index]
    return list(reversed(chain))


def plan_initialization_path(
    model,
    data,
    mapping,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    config: Mapping[str, Any],
    *,
    geometry=None,
) -> PlannerResult:
    """RRT-Connect from ``start_q`` to ``goal_q`` for the right arm."""
    block = config.get("planning", {})
    seed = int(block.get("seed", 0))
    max_iterations = int(block.get("maximum_iterations", 6000))
    step_rad = float(block.get("extension_step_rad", 0.20))
    goal_bias = float(block.get("goal_bias", 0.10))
    check_step = float(block.get("collision_check_step_rad", 0.02))
    forbid_cube = bool(block.get("forbid_cube_contact", True))
    margin_rad = float(config.get("teaching_ready", {}).get("ik", {}).get("joint_margin_rad", 0.02))
    start = np.asarray(start_q, dtype=float).copy()
    goal = np.asarray(goal_q, dtype=float).copy()
    straight = float(np.linalg.norm(goal - start))
    blank = PlannerResult(success=False, start_q=start, goal_q=goal, straight_line_length_rad=straight)

    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        blank.failure = START_INVALID
        return blank
    if straight < 1e-9:
        blank.success = True
        blank.waypoints = [start.copy(), goal.copy()]
        return blank

    # A measured/current state may legitimately sit on a hard joint boundary
    # (the physical vertical-down zero has J4=0). Validate that state against
    # hard limits and collisions, then require the configured margin everywhere
    # after the path has left the start.
    ok, found = configuration_valid(model, data, mapping, start, margin_rad=0.0, forbid_cube_contact=forbid_cube)
    if not ok:
        blank.failure = START_INVALID
        blank.collision = tuple(sorted(found))
        return blank
    ok, found = configuration_valid(model, data, mapping, goal, margin_rad=margin_rad, forbid_cube_contact=forbid_cube)
    if not ok:
        blank.failure = START_INVALID
        blank.collision = tuple(sorted(found))
        return blank

    lower, upper = joint_limits(model, mapping, margin_rad)

    def segment_ok(a: np.ndarray, b: np.ndarray) -> bool:
        segment_margin = 0.0 if np.allclose(a, start, atol=1e-10) or np.allclose(b, start, atol=1e-10) else margin_rad
        good, _ = segment_valid(
            model,
            data,
            mapping,
            a,
            b,
            step_rad=check_step,
            margin_rad=segment_margin,
            forbid_cube_contact=forbid_cube,
        )
        return good

    rng = np.random.default_rng(seed)
    # Two trees kept in a dict with an explicit `active` flag rather than by swapping
    # list references. Swapping made it impossible to tell which tree was rooted at
    # the start, and a connection found after an odd number of swaps produced a
    # reversed goal->start path.
    trees: dict[int, tuple[list[np.ndarray], list[int]]] = {
        0: ([start.copy()], [-1]),
        1: ([goal.copy()], [-1]),
    }
    active = 0

    def extend(tree: list[np.ndarray], parents: list[int], target: np.ndarray) -> tuple[int, bool, bool]:
        """One step towards ``target``.

        Returns ``(index, progressed, reached)``. ``progressed`` distinguishes a real
        new node from a rejected extension, which the greedy connect loop needs in
        order to keep stepping instead of giving up after the first partial step.
        """
        near = _nearest(tree, target)
        node = tree[near]
        delta = target - node
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return near, False, True
        candidate = np.clip(node + delta / distance * min(step_rad, distance), lower, upper)
        ok, _ = configuration_valid(
            model, data, mapping, candidate, margin_rad=margin_rad, forbid_cube_contact=forbid_cube
        )
        if not ok or not segment_ok(node, candidate):
            return near, False, False
        tree.append(candidate)
        parents.append(near)
        return len(tree) - 1, True, distance <= step_rad + 1e-9

    def connect(tree: list[np.ndarray], parents: list[int], target: np.ndarray) -> int:
        """Greedily extend towards ``target``; index of the reaching node, else -1."""
        while True:
            index, progressed, reached = extend(tree, parents, target)
            if reached:
                return index
            if not progressed:
                return -1

    iteration = 0
    for iteration in range(1, max_iterations + 1):
        sample = goal.copy() if rng.random() < goal_bias else np.clip(rng.uniform(lower, upper), lower, upper)
        active_tree, active_parents = trees[active]
        other_tree, other_parents = trees[1 - active]
        index_a, progressed, _ = extend(active_tree, active_parents, sample)
        if progressed:
            index_b = connect(other_tree, other_parents, active_tree[index_a])
            if index_b >= 0:
                chain_active = _chain(active_tree, active_parents, index_a)
                chain_other = _chain(other_tree, other_parents, index_b)
                if active == 0:
                    path = chain_active + list(reversed(chain_other))[1:]
                else:
                    # tree 1 is rooted at the start, so it supplies the prefix.
                    path = chain_other + list(reversed(chain_active))[1:]
                return _finalize(model, data, mapping, path, start, goal, straight, iteration, config, geometry)
        active = 1 - active

    blank.failure = PLAN_FAILED
    blank.iterations = iteration
    return blank


def _finalize(
    model,
    data,
    mapping,
    path: list[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    straight: float,
    iterations: int,
    config: Mapping[str, Any],
    geometry,
) -> PlannerResult:
    """Smooth, then time-scale and measure the found path."""
    block = config.get("planning", {})
    margin_rad = float(config.get("teaching_ready", {}).get("ik", {}).get("joint_margin_rad", 0.02))
    check_step = float(block.get("collision_check_step_rad", 0.02))
    forbid_cube = bool(block.get("forbid_cube_contact", True))
    smoothing_attempts = int(block.get("smoothing_attempts", 240))
    rng = np.random.default_rng(int(block.get("seed", 0)) + 1)

    def segment_ok(a: np.ndarray, b: np.ndarray) -> bool:
        segment_margin = 0.0 if np.allclose(a, start, atol=1e-10) or np.allclose(b, start, atol=1e-10) else margin_rad
        good, _ = segment_valid(
            model, data, mapping, a, b, step_rad=check_step, margin_rad=segment_margin, forbid_cube_contact=forbid_cube
        )
        return good

    smoothed = [node.copy() for node in path]
    touched = False
    for _ in range(smoothing_attempts):
        if len(smoothed) < 3:
            break
        i, j = sorted(rng.integers(0, len(smoothed), size=2).tolist())
        if j - i < 2:
            continue
        # Only accept a shortcut that is shorter in joint space AND collision free,
        # so smoothing can never invent an invalid path.
        before = float(np.linalg.norm(smoothed[i + 1] - smoothed[i])) + float(np.linalg.norm(smoothed[j] - smoothed[j - 1]))
        after = float(np.linalg.norm(smoothed[j] - smoothed[i]))
        if after >= before:
            continue
        if not segment_ok(smoothed[i], smoothed[j]):
            continue
        smoothed = smoothed[: i + 1] + smoothed[j:]
        touched = True

    result = PlannerResult(success=True, start_q=start.copy(), goal_q=goal.copy(), iterations=int(iterations))
    result.straight_line_length_rad = straight
    result.waypoints = [node.copy() for node in smoothed]
    result.smoothed = touched

    # Time scaling: quintic zero-velocity blend per segment, the same conservative
    # convention the project already uses for initialization path comparison.
    velocity_limit = float(block.get("time_scaling", {}).get("maximum_joint_velocity_rad_s", 0.45))
    acceleration_limit = float(block.get("time_scaling", {}).get("maximum_joint_acceleration_rad_s2", 0.60))
    duration = 0.0
    max_velocity = 0.0
    max_acceleration = 0.0
    for a, b in zip(smoothed, smoothed[1:]):
        largest = float(np.max(np.abs(b - a)))
        if largest <= 0.0:
            continue
        segment_duration = max(1.0e-6, 1.875 * largest / velocity_limit, math.sqrt(5.8 * largest / acceleration_limit))
        duration += segment_duration
        max_velocity = max(max_velocity, 1.875 * largest / segment_duration)
        max_acceleration = max(max_acceleration, 5.8 * largest / (segment_duration * segment_duration))
    result.duration_s = duration
    result.maximum_velocity_rad_s = max_velocity
    result.maximum_acceleration_rad_s2 = max_acceleration

    # Measurements over the densified path.
    table_top_z = float(geometry.top_z) if geometry is not None else None
    minimum_clearance = float("inf")
    minimum_margin = float("inf")
    length = 0.0
    for a, b in zip(smoothed, smoothed[1:]):
        length += float(np.linalg.norm(b - a))
        count = max(2, int(math.ceil(float(np.max(np.abs(b - a))) / check_step)) + 1)
        for alpha in np.linspace(0.0, 1.0, count):
            q = a + alpha * (b - a)
            minimum_margin = min(minimum_margin, normalized_joint_margin(model, mapping, q))
            if table_top_z is not None:
                minimum_clearance = min(minimum_clearance, tcp_table_clearance(model, data, mapping, q, table_top_z))
    result.joint_path_length_rad = length
    result.minimum_joint_margin_normalized = minimum_margin if math.isfinite(minimum_margin) else 0.0
    result.minimum_tcp_table_clearance_m = minimum_clearance if math.isfinite(minimum_clearance) else 0.0

    # How far the planned path departs from naive joint interpolation, expressed in
    # radians. A value near zero would mean the planner effectively returned
    # np.linspace, which is what this module exists to avoid.
    deviation = 0.0
    for node in smoothed:
        alpha = 0.0 if straight <= 0 else float(np.clip(np.dot(node - start, goal - start) / (straight**2), 0.0, 1.0))
        deviation = max(deviation, float(np.linalg.norm(node - (start + alpha * (goal - start)))))
    result.maximum_deviation_from_straight_rad = deviation
    return result
