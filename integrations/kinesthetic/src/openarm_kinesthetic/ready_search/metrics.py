"""Per-configuration measurements for the Teaching Ready search.

The measurement this module exists for is gravity torque. The real arm can report a
J4 red light but not *why* J4 was unhappy: a bad joint angle and a good joint angle
carrying a large gravitational moment look identical from the outside. MuJoCo can
separate them, so every candidate records

    gravity_torque_j4                  (the joint that faulted)
    gravity_torque_max_all_joints      (whether some other joint is now the problem)

plus the same quantity sampled along a whole initialization trajectory, because a
pose whose endpoint is lightly loaded can still pass through a heavily loaded one.

The torque convention is deliberately the project's existing one: zero-velocity
inverse dynamics in joint space, which is exactly what
``openarm_kinesthetic.control.passive`` uses for hardware gravity feedforward. A
number here therefore means the same thing as a number the passive bridge would
compute, and the two implementations are cross-checked against each other in the
tests.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..scene.model import RIGHT_JOINTS, ModelMap, name_id
from ..planning.kinematics import joint_limits

# The joint that faulted on hardware on 2026-09-17. Resolved by name out of the
# project's own ordering so a change to RIGHT_JOINTS cannot silently make "J4" mean a
# different physical joint.
J4_JOINT_NAME = "openarm_right_joint4"

# Box-shaped task obstacles. Their distance is exact, which is why they are the group the
# score consumes: "is the TCP clear of the table" is a question about these.
TASK_OBSTACLE_GEOM_PREFIXES = (
    "table_top_collision",
    "black_box_",
    "blue_cube_collision",
)
SCORED_CLEARANCE_GROUP = "task_obstacles"
LEFT_BODY_PREFIX = "openarm_left_"
RIGHT_BODY_PREFIX = "openarm_right_"


def j4_index() -> int:
    """Index of J4 inside the project's right-arm ordering."""
    try:
        return list(RIGHT_JOINTS).index(J4_JOINT_NAME)
    except ValueError as exc:  # pragma: no cover - model contract change
        raise KeyError(f"{J4_JOINT_NAME} is missing from RIGHT_JOINTS") from exc


# --------------------------------------------------------------------------------------
# Kinematic quantities
# --------------------------------------------------------------------------------------


def joint_metrics(model, mapping: ModelMap, q: np.ndarray) -> dict[str, Any]:
    """Joint-limit margins, with J4 called out explicitly."""
    candidate = np.asarray(q, dtype=float)
    lower, upper = joint_limits(model, mapping)
    span = np.maximum(upper - lower, 1e-12)
    normalized = np.minimum(candidate - lower, upper - candidate) / span
    index = j4_index()
    return {
        "joint_names": list(RIGHT_JOINTS),
        "q_rad": candidate.tolist(),
        "joint_lower_rad": lower.tolist(),
        "joint_upper_rad": upper.tolist(),
        "joint_margin_rad": np.minimum(candidate - lower, upper - candidate).tolist(),
        "joint_margin_normalized": normalized.tolist(),
        "minimum_joint_margin_normalized": float(np.min(normalized)),
        "minimum_joint_margin_rad": float(np.min(np.minimum(candidate - lower, upper - candidate))),
        "j4_joint_name": J4_JOINT_NAME,
        "j4_index": index,
        "j4_rad": float(candidate[index]),
        "j4_lower_rad": float(lower[index]),
        "j4_upper_rad": float(upper[index]),
        "j4_margin_rad": float(min(candidate[index] - lower[index], upper[index] - candidate[index])),
        "j4_margin_normalized": float(normalized[index]),
    }


def yoshikawa_manipulability(model, data, mapping: ModelMap, q: np.ndarray) -> dict[str, float]:
    """Yoshikawa manipulability of the 6D grasp-site Jacobian, plus its smallest singular value.

    The smallest singular value is reported because it is the quantity that degenerates
    near a wrist singularity; Yoshikawa's measure is the one the score consumes.
    """
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, mapping.grasp_site)
    jacobian = np.vstack((jacp[:, mapping.right_dof], jacr[:, mapping.right_dof]))
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    determinant = float(np.prod(singular_values))
    return {
        "yoshikawa_manipulability": float(math.sqrt(max(0.0, determinant))),
        "minimum_singular_value": float(np.min(singular_values)),
    }


def _body_name(model, body_id: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""


def _geom_name(model, geom_id: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""


def _collidable(model, geom_id: int) -> bool:
    """True when the geom participates in collision at all.

    Visual-only geoms carry ``contype = conaffinity = 0`` and routinely overlap their
    neighbours (the arm's base cover sits inside the pedestal mesh), so including them
    would report a large constant penetration for every configuration.
    """
    return (int(model.geom_contype[geom_id]) | int(model.geom_conaffinity[geom_id])) != 0


def _pair_collidable(model, first: int, second: int) -> bool:
    """MuJoCo's own contype/conaffinity filter, replicated for a single pair."""
    return bool(
        (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
        or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
    )


# A pair within this distance at the reviewed reference pose counts as already in
# contact. The value matches the project's existing penetration policy
# (`contact_failures` and `initialization_contacts` both use dist >= -0.001 m).
REFERENCE_CONTACT_TOLERANCE_M = 1e-3


def clearance_pairs(
    model,
    data=None,
    mapping: ModelMap | None = None,
    reference_q: np.ndarray | None = None,
    *,
    reference_contact_tolerance_m: float = REFERENCE_CONTACT_TOLERANCE_M,
) -> dict[str, Any]:
    """Right-arm geom pairs whose clearance is measured.

    Two groups are kept apart because they mean different things to an operator: an
    ``obstacle`` contact is a table/box/pedestal/left-arm/cube impact, whereas a
    ``self`` contact is the right arm folding into itself.

    Three groups are built, because ``mj_geomDistance`` is only informative for some of
    them. It works on convex hulls, and a convex hull is a *lower bound* on the true
    distance - so it never overstates clearance - but for a concave obstacle such as the
    pedestal shell the hull can contain the arm without the meshes touching, and the
    reported distance collapses to zero and stops being informative. The groups are
    therefore:

    * ``task_obstacles``: the table, the fixed box and the cube. The obstacle side is a
      box, but the robot side can still be a mesh convex hull.  The result is therefore a
      conservative lower bound: a positive value certifies at least that much separation,
      while zero can mean tangency or a convex-hull artifact and must fail closed.
    * ``structure``: the pedestal shell and the other arm. Mesh against mesh, so the
      distance is a conservative lower bound that can read zero without contact; kept as
      a diagnostic, while real contact is enforced by MuJoCo's contact list.
    * ``self``: the right arm against itself, which for mechanically adjacent links is a
      few millimetres by construction and so is a diagnostic too.

    Pairs are excluded for four reasons, each of which would otherwise make the metric
    meaningless:

    * visual-only geoms (``contype = conaffinity = 0``), which overlap by design;
    * pairs the model's own contype/conaffinity masks exclude;
    * same-body and parent-child pairs, which MuJoCo's narrowphase skips;
    * pairs that already read as touching or interpenetrating in the reviewed reference
      pose, within ``reference_contact_tolerance_m``. Such a pair is a structural mount
      or a hull artifact, it reads the same in every configuration, and so it carries no
      information about a candidate pose. The excluded pairs are named, with their
      measured distance, in the returned mapping rather than being dropped silently.
    """
    import mujoco

    right_geoms: list[int] = []
    task_obstacle_geoms: list[int] = []
    structure_geoms: list[int] = []
    left_geoms: list[int] = []
    for geom in range(model.ngeom):
        if not _collidable(model, geom):
            continue
        body_name = _body_name(model, int(model.geom_bodyid[geom]))
        geom_name = _geom_name(model, geom)
        if body_name.startswith(RIGHT_BODY_PREFIX):
            right_geoms.append(geom)
        if geom_name.startswith(TASK_OBSTACLE_GEOM_PREFIXES):
            task_obstacle_geoms.append(geom)
        if geom_name.startswith(("openarm_body_link0_collision",)) or body_name.startswith(LEFT_BODY_PREFIX):
            structure_geoms.append(geom)
        if body_name.startswith(LEFT_BODY_PREFIX):
            left_geoms.append(geom)

    def pairs_with(others: list[int]) -> list[tuple[int, int]]:
        return [
            (right, other)
            for right in right_geoms
            for other in others
            if _pair_collidable(model, right, other)
        ]

    groups: dict[str, list[tuple[int, int]]] = {
        "task_obstacles": pairs_with(task_obstacle_geoms),
        "structure": pairs_with(structure_geoms),
    }
    self_pairs: list[tuple[int, int]] = []
    for index, first in enumerate(right_geoms):
        for second in right_geoms[index + 1 :]:
            body_a = int(model.geom_bodyid[first])
            body_b = int(model.geom_bodyid[second])
            if body_a == body_b:
                continue
            if int(model.body_parentid[body_a]) == body_b or int(model.body_parentid[body_b]) == body_a:
                continue
            if _pair_collidable(model, first, second):
                self_pairs.append((first, second))
    groups["self"] = self_pairs

    excluded: list[dict[str, Any]] = []
    if data is not None and mapping is not None and reference_q is not None:
        data.qpos[mapping.right_qpos] = np.asarray(reference_q, dtype=float)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        fromto = np.zeros(6)
        tolerance = float(reference_contact_tolerance_m)
        for group, pairs in groups.items():
            kept: list[tuple[int, int]] = []
            for first, second in pairs:
                distance = float(mujoco.mj_geomDistance(model, data, first, second, 0.05, fromto))
                if distance <= tolerance:
                    excluded.append(
                        {
                            "group": group,
                            "pair": f"{_geom_name(model, first)}|{_geom_name(model, second)}",
                            "distance_at_reference_m": distance,
                        }
                    )
                    continue
                kept.append((first, second))
            groups[group] = kept

    return {
        "task_obstacles": groups["task_obstacles"],
        "structure": groups["structure"],
        "self": groups["self"],
        # `obstacle` is kept as the union so callers that only want "anything that is not
        # the right arm itself" do not have to know about the split.
        "obstacle": groups["task_obstacles"] + groups["structure"],
        "right_geom_ids": right_geoms,
        "task_obstacle_geom_ids": task_obstacle_geoms,
        "structure_geom_ids": structure_geoms,
        "left_geom_ids": left_geoms,
        "excluded_pairs": excluded,
        "exclusion_rule": (
            "pairs that already read as touching or interpenetrating at the reviewed reference pose "
            "(within reference_contact_tolerance_m) are structural mounts or convex-hull artifacts and "
            "are excluded, because their distance is identical in every configuration; they are listed "
            "in `excluded_pairs` with the distance that disqualified them"
        ),
        "reference_contact_tolerance_m": float(reference_contact_tolerance_m),
        "reference_pose_source": "poses.yaml right_task_ready (the project's validated reference pose)",
        "scored_group": SCORED_CLEARANCE_GROUP,
    }


def minimum_clearance(
    model,
    data,
    mapping: ModelMap,
    q: np.ndarray,
    pairs: Mapping[str, list[tuple[int, int]]],
    *,
    cap_m: float,
) -> dict[str, Any]:
    """Signed distances from the right arm to each obstacle group.

    ``mj_geomDistance`` is capped at ``cap_m``, so a value equal to the cap only means
    "at least this far"; the cap doubles as the scoring reference scale precisely so
    that no invented "safe distance" threshold is needed. ``minimum_clearance_m`` is the
    overall minimum and is a conservative lower bound (see :func:`clearance_pairs`);
    ``minimum_task_obstacle_clearance_m`` is the conservative table/box/cube lower bound
    and is the value the score consumes.  A zero from a mesh convex hull is not proof of
    physical contact and this function must not be used as a penetration gate; callers
    must use MuJoCo's actual contact list via ``configuration_valid`` for that decision.
    """
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6)

    result: dict[str, Any] = {
        "minimum_clearance_m": float(cap_m),
        "minimum_clearance_group": None,
        "minimum_clearance_pair": None,
        "clearance_cap_m": float(cap_m),
        "clearance_saturated": True,
        "acceptance_role": "diagnostic_conservative_lower_bound_not_penetration_gate",
    }
    per_group: dict[str, tuple[float, str | None]] = {}
    for group in ("task_obstacles", "structure", "self"):
        worst = float(cap_m)
        worst_pair: str | None = None
        for first, second in pairs[group]:
            distance = float(mujoco.mj_geomDistance(model, data, first, second, float(cap_m), fromto))
            if distance < worst:
                worst = distance
                worst_pair = f"{_geom_name(model, first)}|{_geom_name(model, second)}"
        per_group[group] = (worst, worst_pair)
        result[f"minimum_{group}_clearance_m"] = worst
    # The binding group is reported only when something is actually inside the cap;
    # otherwise "nothing is close" is the honest answer and no pair is named.
    for group in ("task_obstacles", "structure", "self"):
        worst, worst_pair = per_group[group]
        if worst < result["minimum_clearance_m"]:
            result["minimum_clearance_m"] = worst
            result["minimum_clearance_group"] = group
            result["minimum_clearance_pair"] = worst_pair
    result["clearance_saturated"] = bool(result["minimum_clearance_m"] >= float(cap_m) - 1e-12)
    result["scored_clearance_m"] = float(result["minimum_task_obstacles_clearance_m"])
    result["scored_clearance_group"] = SCORED_CLEARANCE_GROUP
    return result


def tcp_height_above_table(model, data, mapping: ModelMap, q: np.ndarray, table_top_z: float) -> float:
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(data.site_xpos[mapping.grasp_site, 2] - float(table_top_z))


# --------------------------------------------------------------------------------------
# Gravity torque
# --------------------------------------------------------------------------------------


def static_gravity_torque(model, data, mapping: ModelMap, q: np.ndarray) -> np.ndarray:
    """Gravity torque per right-arm joint, in N*m, at zero velocity.

    ``mj_rne`` with ``flg_acc=0`` at ``qvel=0`` is the gravity term alone: it contains
    the Coriolis/centrifugal part only through ``qvel``, and passive spring/damper
    forces are excluded. The value is validated against ``qfrc_bias`` in the tests, so
    this is MuJoCo's own bias force rather than a hand-rolled model.
    """
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    stacked = np.zeros(model.nv, dtype=float)
    mujoco.mj_rne(model, data, 0, stacked)
    return np.asarray(stacked[mapping.right_dof], dtype=float).copy()


def gravity_torque_metrics(torque: Sequence[float]) -> dict[str, Any]:
    """Summarize one gravity-torque vector, naming J4 explicitly."""
    values = np.asarray(torque, dtype=float)
    index = j4_index()
    absolute = np.abs(values)
    worst = int(np.argmax(absolute))
    return {
        "gravity_torque_nm": values.tolist(),
        "absolute_gravity_torque_nm": absolute.tolist(),
        "gravity_torque_j4_nm": float(values[index]),
        "absolute_gravity_torque_j4_nm": float(absolute[index]),
        "maximum_absolute_gravity_torque_nm": float(absolute[worst]),
        "maximum_absolute_gravity_torque_joint": list(RIGHT_JOINTS)[worst],
        "maximum_absolute_gravity_torque_joint_index": worst,
        "j4_is_most_loaded_joint": bool(worst == index),
    }


def trajectory_gravity_torque(
    model,
    data,
    mapping: ModelMap,
    waypoints: Iterable[np.ndarray],
    *,
    sample_count: int,
) -> dict[str, Any]:
    """Gravity torque sampled along a whole initialization trajectory.

    Sampling along the path is what catches "the endpoint is fine but the arm passes
    through a configuration that loads J4 hard", which a single endpoint number cannot
    show.
    """
    path = [np.asarray(point, dtype=float) for point in waypoints]
    samples = _resample_path(path, sample_count)
    torques = np.asarray([static_gravity_torque(model, data, mapping, q) for q in samples], dtype=float)
    index = j4_index()
    j4 = torques[:, index] if torques.size else np.zeros(0, dtype=float)
    absolute_all = np.abs(torques) if torques.size else np.zeros((0, 1), dtype=float)
    worst_sample = int(np.argmax(np.abs(j4))) if j4.size else -1
    worst_joint = int(np.argmax(absolute_all.max(axis=0))) if torques.size else -1
    return {
        "sample_count": int(len(samples)),
        "maximum_absolute_j4_gravity_torque_nm": float(np.max(np.abs(j4))) if j4.size else 0.0,
        "mean_absolute_j4_gravity_torque_nm": float(np.mean(np.abs(j4))) if j4.size else 0.0,
        "endpoint_j4_gravity_torque_nm": float(j4[-1]) if j4.size else 0.0,
        "maximum_absolute_gravity_torque_nm": float(np.max(np.abs(torques))) if torques.size else 0.0,
        "maximum_absolute_gravity_torque_joint": list(RIGHT_JOINTS)[worst_joint] if worst_joint >= 0 else None,
        "trajectory_j4_gravity_torque_nm": j4.tolist(),
        "worst_sample_index": worst_sample,
        "worst_sample_q_rad": samples[worst_sample].tolist() if worst_sample >= 0 else None,
    }


def _resample_path(path: list[np.ndarray], sample_count: int) -> list[np.ndarray]:
    """Uniformly resample a joint-space polyline by arc length, endpoints included."""
    if not path:
        return []
    if len(path) == 1:
        return [path[0].copy()]
    steps = np.asarray(path, dtype=float)
    deltas = np.linalg.norm(np.diff(steps, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(deltas)]
    total = float(cumulative[-1])
    if total <= 1e-12:
        return [path[0].copy(), path[-1].copy()]
    targets = np.linspace(0.0, total, max(2, int(sample_count)))
    resampled: list[np.ndarray] = []
    for target in targets:
        upper = int(np.searchsorted(cumulative, target, side="right"))
        upper = min(max(upper, 1), len(steps) - 1)
        lower = upper - 1
        span = cumulative[upper] - cumulative[lower]
        alpha = 0.0 if span <= 1e-12 else float((target - cumulative[lower]) / span)
        resampled.append(steps[lower] + alpha * (steps[upper] - steps[lower]))
    return resampled


def trajectory_travel_metrics(path: Iterable[np.ndarray]) -> dict[str, float]:
    """Cumulative L1 joint travel, matching the real MoveIt trajectory metric."""
    steps = np.asarray([np.asarray(point, dtype=float) for point in path], dtype=float)
    if steps.ndim != 2 or steps.shape[0] < 2:
        return {"joint_path_length_rad": 0.0, "maximum_joint_travel_rad": 0.0, "j4_travel_rad": 0.0}
    index = j4_index()
    per_step = np.abs(np.diff(steps, axis=0))
    per_joint = np.sum(per_step, axis=0)
    return {
        "joint_path_length_rad": float(np.sum(per_step)),
        "maximum_joint_travel_rad": float(np.max(per_joint)),
        "maximum_joint_travel_joint": list(RIGHT_JOINTS)[int(np.argmax(per_joint))],
        "j4_travel_rad": float(per_joint[index]),
    }


def joint_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def configuration_difference(first: Sequence[float], second: Sequence[float]) -> float:
    """Max-abs joint difference: the distance two IK solutions have to be within to be
    treated as the same physical configuration."""
    return float(np.max(np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))))


def grasp_site_id(model) -> int:
    import mujoco

    return name_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
