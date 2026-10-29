"""Joint/actuator mapping for the copied OpenArm MJCF.

Reference: official/openarm_mujoco/src/openarm_mujoco/v2/joint_resolver.py
(Apache-2.0, Enactic, Inc.), independently ported.  The only behavioural
change is the explicit left-finger sign: the current Dora simulation and the
physical ACT dataset use a negative gripper value for the parked left finger,
while the MJCF left finger qpos/ctrl range is positive.
"""
from __future__ import annotations

import mujoco
import numpy as np


class SimMapper:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._qpos = {}
        self._dof = {}
        self._ctrl = {}
        self._finger2 = {}
        for side in ("right", "left"):
            self._qpos[side] = np.array(
                [model.joint(f"openarm_{side}_joint{i}").qposadr[0] for i in range(1, 8)],
                dtype=np.intp,
            )
            self._dof[side] = np.array(
                [model.joint(f"openarm_{side}_joint{i}").dofadr[0] for i in range(1, 8)],
                dtype=np.intp,
            )
            self._ctrl[side] = np.array(
                [model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, 8)]
                + [model.actuator(f"{side}_finger1_ctrl").id],
                dtype=np.intp,
            )
            self._finger2[side] = int(
                model.joint(f"openarm_{side}_finger_joint2").qposadr[0]
            )

    @staticmethod
    def _gripper_sign(side: str) -> float:
        return 1.0 if side == "right" else -1.0

    def set_qpos(self, qpos: np.ndarray, driver: np.ndarray, side: str) -> None:
        driver = np.asarray(driver, dtype=float)
        if driver.shape != (8,):
            raise ValueError("driver position must be 8 values")
        sign = self._gripper_sign(side)
        qpos[self._qpos[side]] = driver[:7]
        qpos[self._finger2[side]] = driver[7] * sign
        finger1 = int(self.model.joint(f"openarm_{side}_finger_joint1").qposadr[0])
        qpos[finger1] = driver[7] * sign

    def set_ctrl(self, ctrl: np.ndarray, driver: np.ndarray, side: str) -> None:
        driver = np.asarray(driver, dtype=float)
        if driver.shape != (8,):
            raise ValueError("driver position must be 8 values")
        sign = self._gripper_sign(side)
        ctrl[self._ctrl[side]] = np.r_[driver[:7], driver[7] * sign]

    def get_driver(self, qpos: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        sign = self._gripper_sign(side)
        finger1 = self.model.joint(f"openarm_{side}_finger_joint1").qposadr[0]
        q = np.r_[qpos[self._qpos[side]], float(qpos[finger1]) * sign]
        qd = np.r_[np.zeros(7), 0.0]  # only positions are consumed by the expert
        return q, qd
