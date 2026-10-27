"""Collision-aware teaching-ready IK with a small, bounded fallback search.

The desired target is attempted first. Only if that fails - either because IK does
not converge or because the solution collides - is a small neighbourhood searched.
The search is intentionally tiny (a disk of ``max_radius_m`` at ``step_m``
resolution, plus one step either side in Z) because a full-table reachability
optimizer is explicitly out of scope.

Every candidate must satisfy joint-limit margin, collision constraints and the
position *and* orientation tolerances; a position-only solution is not accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..planning.kinematics import solve_site_pose
from .collision import configuration_valid
from .teaching_ready import TeachingReadyTarget, rotation_error_rad

IK_FAILED = "SIM_TEACH_READY_IK_FAILED"


@dataclass(frozen=True)
class TeachingReadyIKResult:
    success: bool
    q: np.ndarray | None
    position_error_m: float
    orientation_error_rad: float
    attempts: int
    used_fallback: bool
    offset_m: tuple[float, float, float]
    seed_index: int
    collision: tuple[str, ...]
    failure: str | None
    nonconvergent_attempts: int
    collision_attempts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "q_teaching_ready": self.q.tolist() if self.q is not None else None,
            "position_error_m": float(self.position_error_m),
            "orientation_error_rad": float(self.orientation_error_rad),
            "attempts": int(self.attempts),
            "used_fallback": self.used_fallback,
            "fallback_offset_m": [float(value) for value in self.offset_m],
            "seed_index": int(self.seed_index),
            "collision": list(self.collision),
            "failure": self.failure,
            "nonconvergent_attempts": int(self.nonconvergent_attempts),
            "collision_attempts": int(self.collision_attempts),
        }


def fallback_offsets(step_m: float, max_radius_m: float, maximum_candidates: int) -> list[np.ndarray]:
    """Deterministic in-plane offsets, nearest first.

    Determinism matters: the same table prior and the same seed must always produce
    the same teaching-ready joint solution, otherwise nothing downstream is
    reproducible.
    """
    if step_m <= 0.0 or max_radius_m <= 0.0:
        return []
    steps = int(math.floor(max_radius_m / step_m + 1e-9))
    candidates: list[tuple[float, float, float]] = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            dx, dy = i * step_m, j * step_m
            radius = math.hypot(dx, dy)
            if radius > max_radius_m + 1e-9:
                continue
            candidates.append((radius, dx, dy))
    candidates.sort(key=lambda item: (round(item[0], 9), round(item[1], 9), round(item[2], 9)))
    return [np.asarray([dx, dy], dtype=float) for _, dx, dy in candidates[:maximum_candidates]]


def _candidate_offsets(config: Mapping[str, Any]) -> list[np.ndarray]:
    block = config.get("teaching_ready", {}).get("fallback_search", {})
    if not block.get("enabled", False):
        return [np.zeros(3)]
    step = float(block["step_m"])
    radius = float(block["max_radius_m"])
    maximum = int(block["maximum_candidates"])
    planar = fallback_offsets(step, radius, maximum)
    offsets: list[np.ndarray] = [np.zeros(3)]
    # The exact desired target is always attempted first, so drop the duplicate.
    offsets.extend([np.r_[value, 0.0] for value in planar if not np.allclose(value, 0.0)])
    for dz in (step, -step):
        offsets.extend([np.r_[value, dz] for value in planar])
    return offsets[:maximum]


def solve_teaching_ready_ik(
    model,
    data,
    mapping,
    target: TeachingReadyTarget,
    ik_seeds: np.ndarray | list[np.ndarray],
    config: Mapping[str, Any],
) -> TeachingReadyIKResult:
    """Solve collision-aware IK for the teaching-ready target.

    ``ik_seeds`` is an ordered seed list, reference pose first. The IK seed is
    deliberately *not* the current arm configuration: the real acceptance path seeds
    IK from the reference pose while taking the planner start from the measured
    current state, and mixing those two roles makes IK fail whenever the arm happens
    to be parked far from the target branch.
    """
    ik_block = config.get("teaching_ready", {}).get("ik", {})
    position_tolerance = float(ik_block.get("position_tolerance_m", 0.004))
    orientation_tolerance = float(ik_block.get("orientation_tolerance_rad", 0.04))
    iterations = int(ik_block.get("iterations", 300))
    margin_rad = float(ik_block.get("joint_margin_rad", 0.02))
    forbid_cube = bool(config.get("planning", {}).get("forbid_cube_contact", True))

    seeds = [np.asarray(seed, dtype=float) for seed in np.atleast_2d(ik_seeds)]

    best_position_error = float("inf")
    best_orientation_error = float("inf")
    collision_labels: set[str] = set()
    nonconvergent = 0
    collisions = 0
    attempts = 0

    for offset_index, offset in enumerate(_candidate_offsets(config)):
        goal = np.asarray(target.position_world_m, dtype=float) + np.asarray(offset, dtype=float)
        for seed_index, seed in enumerate(seeds):
            attempts += 1
            ik = solve_site_pose(
                model,
                data,
                mapping,
                goal,
                target.orientation_matrix,
                seed,
                tolerance_m=position_tolerance,
                orientation_tolerance_rad=orientation_tolerance,
                iterations=iterations,
                margin_rad=margin_rad,
            )
            if not ik.success:
                nonconvergent += 1
                best_position_error = min(best_position_error, float(ik.position_error_m))
                continue
            # solve_site_pose reports position error only, so re-measure both errors
            # and the collision state at the returned configuration.
            import mujoco

            data.qpos[mapping.right_qpos] = ik.q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            position_error = float(np.linalg.norm(goal - data.site_xpos[mapping.grasp_site]))
            orientation_error = rotation_error_rad(
                np.asarray(data.site_xmat[mapping.grasp_site], dtype=float).reshape(3, 3),
                target.orientation_matrix,
            )
            best_position_error = min(best_position_error, position_error)
            best_orientation_error = min(best_orientation_error, orientation_error)
            ok, found = configuration_valid(
                model, data, mapping, ik.q, margin_rad=margin_rad, forbid_cube_contact=forbid_cube
            )
            if not ok:
                collisions += 1
                collision_labels.update(found)
                continue
            return TeachingReadyIKResult(
                success=True,
                q=ik.q.copy(),
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                attempts=attempts,
                used_fallback=offset_index > 0,
                offset_m=(float(offset[0]), float(offset[1]), float(offset[2])),
                seed_index=seed_index,
                collision=(),
                failure=None,
                nonconvergent_attempts=nonconvergent,
                collision_attempts=collisions,
            )

    return TeachingReadyIKResult(
        success=False,
        q=None,
        position_error_m=best_position_error if math.isfinite(best_position_error) else float("nan"),
        orientation_error_rad=best_orientation_error if math.isfinite(best_orientation_error) else float("nan"),
        attempts=attempts,
        used_fallback=False,
        offset_m=(0.0, 0.0, 0.0),
        seed_index=-1,
        collision=tuple(sorted(collision_labels)),
        failure=IK_FAILED,
        nonconvergent_attempts=nonconvergent,
        collision_attempts=collisions,
    )
