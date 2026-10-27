"""Compliant 6D virtual teaching handle implemented as external wrench."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandleWrench:
    force_n: np.ndarray
    torque_nm: np.ndarray


class VirtualHandle:
    def __init__(
        self, site_id: int, *, translational_stiffness: float = 500.0,
        translational_damping: float = 35.0, rotational_stiffness: float = 8.0,
        rotational_damping: float = 0.8, max_force: float = 35.0, max_torque: float = 4.0,
        right_qpos=None, right_dofs=None, preferred_q=None,
        nullspace_stiffness: float = 3.0, nullspace_damping: float = 0.35,
        max_nullspace_torque: float = 3.0, nullspace_damping_ratio: float = 0.02,
    ):
        self.site_id = int(site_id)
        self.kp = float(translational_stiffness); self.kd = float(translational_damping)
        self.kr = float(rotational_stiffness); self.dr = float(rotational_damping)
        self.max_force = float(max_force); self.max_torque = float(max_torque)
        posture_arguments = (right_qpos, right_dofs, preferred_q)
        if any(value is not None for value in posture_arguments) and not all(value is not None for value in posture_arguments):
            raise ValueError("right_qpos, right_dofs, and preferred_q must be provided together")
        self.right_qpos = None if right_qpos is None else np.asarray(right_qpos, dtype=int).copy()
        self.right_dofs = None if right_dofs is None else np.asarray(right_dofs, dtype=int).copy()
        self.preferred_q = None if preferred_q is None else np.asarray(preferred_q, dtype=float).copy()
        if self.preferred_q is not None and (
            self.right_qpos.shape != self.right_dofs.shape
            or self.preferred_q.shape != self.right_qpos.shape
            or not np.all(np.isfinite(self.preferred_q))
        ):
            raise ValueError("posture arrays must be finite and have matching shapes")
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.nullspace_damping = float(nullspace_damping)
        self.max_nullspace_torque = float(max_nullspace_torque)
        self.nullspace_damping_ratio = float(nullspace_damping_ratio)
        self.active = False
        self.target_position = np.zeros(3); self.target_rotation = np.eye(3)
        self.last_nullspace_torque = np.zeros(0 if self.right_dofs is None else len(self.right_dofs))

    def engage_at_current(self, data) -> None:
        self.target_position = data.site_xpos[self.site_id].copy()
        self.target_rotation = data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.active = True

    def set_target(self, position, rotation=None) -> None:
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("handle target position must be finite 3D")
        self.target_position = position.copy()
        if rotation is not None:
            matrix = np.asarray(rotation, dtype=float)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError("handle target rotation must be finite 3x3")
            self.target_rotation = matrix.copy()
        self.active = True

    @staticmethod
    def _clamp(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector * (maximum / norm) if norm > maximum else vector

    def apply(self, model, data) -> HandleWrench:
        import mujoco

        if not self.active:
            return HandleWrench(np.zeros(3), np.zeros(3))
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, self.site_id)
        linear_velocity = jacp @ data.qvel
        angular_velocity = jacr @ data.qvel
        current_rotation = data.site_xmat[self.site_id].reshape(3, 3)
        rotation_error = 0.5 * sum(
            np.cross(current_rotation[:, axis], self.target_rotation[:, axis]) for axis in range(3)
        )
        force = self._clamp(self.kp * (self.target_position - data.site_xpos[self.site_id]) - self.kd * linear_velocity, self.max_force)
        torque = self._clamp(self.kr * rotation_error - self.dr * angular_velocity, self.max_torque)
        data.qfrc_applied[:] += jacp.T @ force + jacr.T @ torque
        if self.preferred_q is not None:
            jacobian = np.vstack((jacp[:, self.right_dofs], jacr[:, self.right_dofs]))
            regularized = jacobian @ jacobian.T + self.nullspace_damping_ratio**2 * np.eye(6)
            projector = np.eye(len(self.right_dofs)) - jacobian.T @ np.linalg.solve(regularized, jacobian)
            posture_torque = (
                self.nullspace_stiffness * (self.preferred_q - data.qpos[self.right_qpos])
                - self.nullspace_damping * data.qvel[self.right_dofs]
            )
            self.last_nullspace_torque = self._clamp(projector @ posture_torque, self.max_nullspace_torque)
            data.qfrc_applied[self.right_dofs] += self.last_nullspace_torque
        return HandleWrench(force, torque)

    def release(self, data=None) -> None:
        self.active = False
        self.last_nullspace_torque[:] = 0.0
        if data is not None:
            data.qfrc_applied[:] = 0.0
