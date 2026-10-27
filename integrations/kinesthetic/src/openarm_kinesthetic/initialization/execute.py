"""Automatic execution of the initialization trajectory in simulation only.

Prompt section 14 permits automatic execution *in MuJoCo* while still requiring the
stages to stay distinguishable: PLAN -> VALIDATE -> SIM EXECUTE. Nothing in this
module can command hardware; the real backend is a separate, disabled path.

The arm is driven through the ordinary position actuators (``data.ctrl``), never by
writing ``qpos``. That keeps the initialization trajectory subject to the same
dynamics, limits and contact behaviour as any other simulation motion, so a
trajectory that only "works" as a kinematic interpolation cannot pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .collision import initialization_contacts
from ..planning.kinematics import normalized_joint_margin
from .states import ERROR_CATEGORIES
from .teaching_ready import TeachingReadyTarget, rotation_error_rad

EXECUTION_FAILED = "SIM_INITIALIZATION_EXECUTION_FAILED"


def quintic_blend(alpha: float) -> float:
    """Zero-velocity, zero-acceleration blend, matching the project's own convention."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def _segment_durations(waypoints: list[np.ndarray], block: Mapping[str, Any]) -> list[float]:
    velocity_limit = float(block.get("maximum_joint_velocity_rad_s", 0.45))
    acceleration_limit = float(block.get("maximum_joint_acceleration_rad_s2", 0.60))
    durations: list[float] = []
    for a, b in zip(waypoints, waypoints[1:]):
        largest = float(np.max(np.abs(b - a)))
        if largest <= 0.0:
            durations.append(1.0e-6)
            continue
        durations.append(
            max(1.0e-6, 1.875 * largest / velocity_limit, float(np.sqrt(5.8 * largest / acceleration_limit)))
        )
    return durations


def reference_at(waypoints: list[np.ndarray], durations: list[float], time_s: float) -> np.ndarray:
    """Piecewise quintic reference configuration at ``time_s``."""
    if len(waypoints) == 1:
        return waypoints[0].copy()
    elapsed = 0.0
    for index, duration in enumerate(durations):
        if time_s <= elapsed + duration or index == len(durations) - 1:
            alpha = 0.0 if duration <= 0 else (time_s - elapsed) / duration
            return waypoints[index] + quintic_blend(alpha) * (waypoints[index + 1] - waypoints[index])
        elapsed += duration
    return waypoints[-1].copy()


@dataclass
class ExecutionResult:
    success: bool
    failure: str | None
    steps: int
    duration_s: float
    final_position_error_m: float
    final_orientation_error_rad: float
    final_joint_error_rad: float
    minimum_tcp_table_clearance_m: float
    minimum_joint_margin_normalized: float
    forbidden_contacts: tuple[str, ...]
    contact_labels_seen: tuple[str, ...]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    joint_tolerance_rad: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure": self.failure,
            "failure_category": ERROR_CATEGORIES.get(self.failure) if self.failure else None,
            "steps": int(self.steps),
            "duration_s": float(self.duration_s),
            "final_tcp_position_error_m": float(self.final_position_error_m),
            "final_tcp_orientation_error_rad": float(self.final_orientation_error_rad),
            "final_joint_error_rad": float(self.final_joint_error_rad),
            "minimum_tcp_table_clearance_m": float(self.minimum_tcp_table_clearance_m),
            "minimum_joint_margin_normalized": float(self.minimum_joint_margin_normalized),
            "forbidden_contacts": list(self.forbidden_contacts),
            "contact_labels_seen": list(self.contact_labels_seen),
            "tolerances": {
                "position_m": float(self.position_tolerance_m),
                "orientation_rad": float(self.orientation_tolerance_rad),
                "joint_rad": float(self.joint_tolerance_rad),
            },
        }


def gravity_feedforward(model, data, mapping, scratch) -> np.ndarray:
    """Zero-velocity inverse-dynamics gravity torque for the seven right-arm joints.

    The same mj_rne path the project already uses for hardware passive mode, but here
    the position actuators stay enabled. This mirrors the real launch configuration
    (`gravity_compensation:=true`), so the simulated initialization is executed with
    the control chain the station will actually use rather than a softer idealised one.
    """
    import mujoco

    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    scratch.qacc[:] = 0.0
    mujoco.mj_forward(model, scratch)
    total = np.zeros(model.nv, dtype=float)
    mujoco.mj_rne(model, scratch, 0, total)
    return total[mapping.right_dof]


def execute_initialization_trajectory(
    model,
    data,
    mapping,
    waypoints: list[np.ndarray],
    goal_q: np.ndarray,
    target: TeachingReadyTarget,
    geometry,
    config: Mapping[str, Any],
) -> ExecutionResult:
    """Drive the planned trajectory through the position actuators in MuJoCo."""
    import mujoco

    block = config.get("planning", {}).get("time_scaling", {})
    execution = config.get("execution", {})
    settle_s = float(execution.get("settle_s", 0.5))
    position_tolerance = float(execution.get("final_position_tolerance_m", 0.010))
    orientation_tolerance = float(execution.get("final_orientation_tolerance_rad", 0.050))
    joint_tolerance = float(execution.get("final_joint_tolerance_rad", 0.030))
    paths = [np.asarray(node, dtype=float) for node in waypoints]
    durations = _segment_durations(paths, block)
    total = float(sum(durations)) + settle_s
    dt = float(model.opt.timestep)
    steps = int(np.ceil(total / dt))
    # Hold the left arm at its park target and leave the gripper commanded value
    # untouched: initialization moves the right arm only.
    left_ctrl = np.asarray(data.ctrl[mapping.left_actuators], dtype=float).copy()

    forbidden: set[str] = set()
    seen: set[str] = set()
    minimum_clearance = float("inf")
    minimum_margin = float("inf")
    scratch = mujoco.MjData(model)

    for step in range(steps):
        time_s = step * dt
        reference = reference_at(paths, durations, min(time_s, float(sum(durations))))
        data.ctrl[mapping.right_actuators] = reference
        data.ctrl[mapping.left_actuators] = left_ctrl
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[mapping.right_dof] = gravity_feedforward(model, data, mapping, scratch)
        mujoco.mj_step(model, data)
        labels = initialization_contacts(model, data)
        seen.update(labels)
        forbidden.update(labels)
        minimum_clearance = min(
            minimum_clearance, float(data.site_xpos[mapping.grasp_site, 2] - geometry.top_z)
        )
        minimum_margin = min(
            minimum_margin, normalized_joint_margin(model, mapping, data.qpos[mapping.right_qpos])
        )

    final_position_error = float(np.linalg.norm(target.position_world_m - data.site_xpos[mapping.grasp_site]))
    final_orientation_error = rotation_error_rad(
        np.asarray(data.site_xmat[mapping.grasp_site], dtype=float).reshape(3, 3), target.orientation_matrix
    )
    final_joint_error = float(np.max(np.abs(data.qpos[mapping.right_qpos] - np.asarray(goal_q, dtype=float))))

    tolerances_met = (
        final_position_error <= position_tolerance
        and final_orientation_error <= orientation_tolerance
        and final_joint_error <= joint_tolerance
    )
    failure = None
    if forbidden:
        failure = EXECUTION_FAILED
    elif not tolerances_met:
        failure = EXECUTION_FAILED

    return ExecutionResult(
        success=failure is None,
        failure=failure,
        steps=steps,
        duration_s=total,
        final_position_error_m=final_position_error,
        final_orientation_error_rad=final_orientation_error,
        final_joint_error_rad=final_joint_error,
        minimum_tcp_table_clearance_m=minimum_clearance if np.isfinite(minimum_clearance) else 0.0,
        minimum_joint_margin_normalized=minimum_margin if np.isfinite(minimum_margin) else 0.0,
        forbidden_contacts=tuple(sorted(forbidden)),
        contact_labels_seen=tuple(sorted(seen)),
        position_tolerance_m=position_tolerance,
        orientation_tolerance_rad=orientation_tolerance,
        joint_tolerance_rad=joint_tolerance,
    )
