"""Small dependency-free MuJoCo IK and collision helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene.model import ModelMap


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    position_error_m: float
    iterations: int
    failure: str | None = None


def joint_limits(model, mapping: ModelMap, margin_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [int(model.dof_jntid[d]) for d in mapping.right_dof]
    lower = np.asarray([model.jnt_range[j, 0] for j in joint_ids]) + margin_rad
    upper = np.asarray([model.jnt_range[j, 1] for j in joint_ids]) - margin_rad
    return lower, upper


def normalized_joint_margin(model, mapping: ModelMap, q: np.ndarray) -> float:
    lower, upper = joint_limits(model, mapping)
    span = np.maximum(upper - lower, 1e-12)
    return float(np.min(np.minimum(q - lower, upper - q) / span))


def solve_site_position(
    model,
    data,
    mapping: ModelMap,
    target_xyz: np.ndarray,
    seed_q: np.ndarray,
    *,
    tolerance_m: float = 0.004,
    iterations: int = 220,
    margin_rad: float = 0.02,
) -> IKResult:
    import mujoco

    target = np.asarray(target_xyz, dtype=float)
    q = np.asarray(seed_q, dtype=float).copy()
    if target.shape != (3,) or q.shape != (7,) or not np.all(np.isfinite(np.r_[target, q])):
        raise ValueError("target must be 3D and seed must be finite 7D")
    lower, upper = joint_limits(model, mapping, margin_rad)
    q = np.clip(q, lower, upper)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    error_norm = float("inf")
    for iteration in range(1, iterations + 1):
        data.qpos[mapping.right_qpos] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[mapping.grasp_site]
        error_norm = float(np.linalg.norm(error))
        if error_norm <= tolerance_m:
            return IKResult(True, q.copy(), error_norm, iteration)
        mujoco.mj_jacSite(model, data, jacp, jacr, mapping.grasp_site)
        jac = jacp[:, mapping.right_dof]
        damping = 2e-3 + 2e-2 * min(1.0, error_norm / 0.10)
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(3), error)
        norm = float(np.linalg.norm(delta))
        if norm > 0.10:
            delta *= 0.10 / norm
        candidate = np.clip(q + delta, lower, upper)
        if np.linalg.norm(candidate - q) < 1e-9:
            return IKResult(False, q.copy(), error_norm, iteration, "joint_limit_stall")
        q = candidate
    return IKResult(False, q.copy(), error_norm, iterations, "ik_nonconvergence")


def solve_site_pose(
    model,
    data,
    mapping: ModelMap,
    target_xyz: np.ndarray,
    target_xmat: np.ndarray,
    seed_q: np.ndarray,
    *,
    tolerance_m: float = 0.004,
    orientation_tolerance_rad: float = 0.04,
    iterations: int = 300,
    margin_rad: float = 0.02,
) -> IKResult:
    """Damped least-squares 6D site IK with a fixed, explicit orientation."""
    import mujoco

    target = np.asarray(target_xyz, dtype=float)
    target_rotation = np.asarray(target_xmat, dtype=float).reshape(3, 3)
    q = np.asarray(seed_q, dtype=float).copy()
    lower, upper = joint_limits(model, mapping, margin_rad)
    q = np.clip(q, lower, upper)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    position_error = float("inf")
    for iteration in range(1, iterations + 1):
        data.qpos[mapping.right_qpos] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        position = target - data.site_xpos[mapping.grasp_site]
        current_rotation = data.site_xmat[mapping.grasp_site].reshape(3, 3)
        rotation = 0.5 * sum(
            np.cross(current_rotation[:, axis], target_rotation[:, axis]) for axis in range(3)
        )
        position_error = float(np.linalg.norm(position))
        rotation_error = float(np.linalg.norm(rotation))
        if position_error <= tolerance_m and rotation_error <= orientation_tolerance_rad:
            return IKResult(True, q.copy(), position_error, iteration)
        mujoco.mj_jacSite(model, data, jacp, jacr, mapping.grasp_site)
        jac = np.vstack((jacp[:, mapping.right_dof], jacr[:, mapping.right_dof]))
        error = np.r_[position, rotation]
        damping = 5e-3
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(6), error)
        norm = float(np.linalg.norm(delta))
        if norm > 0.10:
            delta *= 0.10 / norm
        candidate = np.clip(q + delta, lower, upper)
        if np.linalg.norm(candidate - q) < 1e-9:
            return IKResult(False, q.copy(), position_error, iteration, "joint_limit_stall")
        q = candidate
    return IKResult(False, q.copy(), position_error, iterations, "ik_pose_nonconvergence")


def contact_failures(model, data) -> set[str]:
    """Classify task-relevant contacts; normal cube support and finger self-contact are allowed."""
    import mujoco

    failures: set[str] = set()
    for contact in data.contact[: data.ncon]:
        # Match the preserved expert's numerical penetration policy: MuJoCo
        # keeps near-zero constraint contacts that are not meaningful impacts.
        if float(contact.dist) >= -0.001:
            continue
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        ]
        pair = " ".join(names)
        if "table_top_collision" in pair and ("right_" in pair or "left_" in pair):
            failures.add("robot_table_collision")
        if "black_box_" in pair and ("right_" in pair or "left_" in pair):
            failures.add("robot_black_box_collision")
        if any("left" in name for name in names) and any("right" in name for name in names):
            failures.add("inter_arm_collision")
        if "openarm_body_link0_collision" in pair and ("right_" in pair or "left_" in pair):
            failures.add("arm_pedestal_collision")
    return failures


def check_joint_path(
    model,
    data,
    mapping: ModelMap,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    step_rad: float = 0.02,
) -> tuple[set[str], float, float]:
    import mujoco

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    count = max(2, int(np.ceil(np.max(np.abs(goal - start)) / step_rad)) + 1)
    failures: set[str] = set()
    min_table_clearance = float("inf")
    min_margin = float("inf")
    table_top_z = 0.40
    for alpha in np.linspace(0.0, 1.0, count):
        q = start + alpha * (goal - start)
        data.qpos[mapping.right_qpos] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        failures.update(contact_failures(model, data))
        min_table_clearance = min(min_table_clearance, float(data.site_xpos[mapping.grasp_site, 2] - table_top_z))
        min_margin = min(min_margin, normalized_joint_margin(model, mapping, q))
    return failures, min_table_clearance, min_margin
