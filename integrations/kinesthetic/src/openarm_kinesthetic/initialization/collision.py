"""Collision validity for the initialization motion.

Initialization is setup, not demonstration: the arm must reach the teaching-ready
pose without touching the table, the black box, the pedestal, the other arm, or the
cube. A touched cube at this stage would silently become a false grasp record later,
so the cube test is strict here even though the task evaluator treats finger/cube
contact as a grasp.
"""

from __future__ import annotations

import numpy as np

from ..planning.kinematics import contact_failures, joint_limits

# Contact labels this module can report. Every label names a robot side and an
# obstacle so a failure report says which pair collided.
ROBOT_LABELS = ("right_", "left_")
CUBE_GEOM = "blue_cube_collision"
PENETRATION_TOLERANCE_M = 1e-3


def _arm_side(geom_name: str) -> str | None:
    if "_right_" in geom_name:
        return "right"
    if "_left_" in geom_name:
        return "left"
    return None


def _pair_failure_labels(
    names: tuple[str, str], *, forbid_cube_contact: bool
) -> set[str]:
    """Classify one genuinely penetrating pair for initialization.

    MuJoCo already suppresses same-body and parent-child contacts.  A remaining
    same-arm contact is therefore a real non-adjacent self collision, except for
    contact wholly inside the coupled gripper fingers, which is allowed by the
    existing collision policy.
    """

    first, second = names
    pair = " ".join(names)
    sides = (_arm_side(first), _arm_side(second))
    robot_involved = any(side is not None for side in sides)
    failures: set[str] = set()
    if "table_top_collision" in pair and robot_involved:
        failures.add("robot_table_collision")
    if "black_box_" in pair and robot_involved:
        failures.add("robot_black_box_collision")
    if "openarm_body_link0_collision" in pair and robot_involved:
        failures.add("arm_pedestal_collision")
    if set(sides) == {"left", "right"}:
        failures.add("inter_arm_collision")
    if sides[0] is not None and sides[0] == sides[1]:
        if not ("finger" in first and "finger" in second):
            failures.add("robot_self_collision")
    if forbid_cube_contact and CUBE_GEOM in pair and robot_involved:
        failures.add("robot_cube_collision")
    return failures


def penetrating_contact_records(
    model,
    data,
    *,
    forbid_cube_contact: bool = True,
    tolerance_m: float = PENETRATION_TOLERANCE_M,
) -> list[dict]:
    """Return task-relevant MuJoCo penetrations with pair and depth evidence."""

    import mujoco

    records: list[dict] = []
    for contact in data.contact[: data.ncon]:
        distance = float(contact.dist)
        if distance >= -float(tolerance_m):
            continue
        names = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            )
            or "",
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            )
            or "",
        )
        labels = _pair_failure_labels(
            names, forbid_cube_contact=forbid_cube_contact
        )
        if labels:
            records.append(
                {
                    "pair": f"{names[0]}|{names[1]}",
                    "distance_m": distance,
                    "penetration_depth_m": -distance,
                    "failure_labels": sorted(labels),
                }
            )
    return records


def initialization_contacts(model, data, *, forbid_cube_contact: bool = True) -> set[str]:
    """Collision labels that invalidate an initialization configuration or segment."""
    failures = set(contact_failures(model, data))
    for record in penetrating_contact_records(
        model, data, forbid_cube_contact=forbid_cube_contact
    ):
        failures.update(record["failure_labels"])
    return failures


def joint_margin_ok(model, mapping, q: np.ndarray, margin_rad: float) -> bool:
    lower, upper = joint_limits(model, mapping, margin_rad)
    return bool(np.all(q >= lower) and np.all(q <= upper))


def configuration_valid(
    model,
    data,
    mapping,
    q: np.ndarray,
    *,
    margin_rad: float,
    forbid_cube_contact: bool = True,
) -> tuple[bool, set[str]]:
    """True when the right arm may occupy ``q``; also returns the failure labels."""
    import mujoco

    candidate = np.asarray(q, dtype=float)
    if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
        return False, {"non_finite_configuration"}
    if not joint_margin_ok(model, mapping, candidate, margin_rad):
        return False, {"joint_limit_margin"}
    data.qpos[mapping.right_qpos] = candidate
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    failures = initialization_contacts(
        model, data, forbid_cube_contact=forbid_cube_contact
    )
    return not failures, failures


def segment_valid(
    model,
    data,
    mapping,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    step_rad: float,
    margin_rad: float,
    forbid_cube_contact: bool = True,
) -> tuple[bool, set[str]]:
    """Collision check a straight joint-space segment at ``step_rad`` resolution."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    count = max(2, int(np.ceil(float(np.max(np.abs(goal - start))) / step_rad)) + 1)
    failures: set[str] = set()
    for alpha in np.linspace(0.0, 1.0, count):
        ok, found = configuration_valid(
            model,
            data,
            mapping,
            start + alpha * (goal - start),
            margin_rad=margin_rad,
            forbid_cube_contact=forbid_cube_contact,
        )
        if not ok:
            failures.update(found)
            # A single invalid sample is enough; stop early to keep planning cheap.
            break
    return not failures, failures


def tcp_table_clearance(model, data, mapping, q: np.ndarray, table_top_z: float) -> float:
    """Signed TCP height above the table top for one configuration."""
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(data.site_xpos[mapping.grasp_site, 2] - table_top_z)
