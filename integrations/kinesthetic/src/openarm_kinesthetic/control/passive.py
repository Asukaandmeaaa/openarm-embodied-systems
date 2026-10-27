"""Runtime right-arm passive mode; vendor MJCF remains immutable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene.model import ModelMap


@dataclass
class PassiveSnapshot:
    gain: np.ndarray
    bias: np.ndarray
    gravcomp: np.ndarray


class RightArmPassiveMode:
    """Disable only the seven right position servos and compensate link gravity."""

    def __init__(self, model, mapping: ModelMap, gravity_compensation: float = 1.0):
        self.model = model
        self.mapping = mapping
        self.gravity_compensation = float(gravity_compensation)
        self._snapshot: PassiveSnapshot | None = None
        self._gravity_scale = float(gravity_compensation)
        self._inverse_data = None

    def enable(self) -> None:
        import mujoco

        if self._snapshot is not None:
            return
        actuators = self.mapping.right_actuators
        right_bodies = []
        for body in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if name.startswith("openarm_right_"):
                right_bodies.append(body)
        self._snapshot = PassiveSnapshot(
            self.model.actuator_gainprm[actuators].copy(),
            self.model.actuator_biasprm[actuators].copy(),
            self.model.body_gravcomp[right_bodies].copy(),
        )
        self._right_bodies = np.asarray(right_bodies, dtype=int)
        self.model.actuator_gainprm[actuators] = 0.0
        self.model.actuator_biasprm[actuators] = 0.0
        self.model.body_gravcomp[self._right_bodies] = self.gravity_compensation
        # Runtime body_gravcomp is retained as an auditable model flag, while
        # the explicit inverse-dynamics term below is the tested force path.
        import mujoco
        self._inverse_data = mujoco.MjData(self.model)

    def set_gravity_compensation(self, value: float) -> None:
        if self._snapshot is None:
            raise RuntimeError("passive mode is not enabled")
        self._gravity_scale = float(value)
        self.model.body_gravcomp[self._right_bodies] = float(value)

    def apply_gravity_compensation(self, data) -> np.ndarray:
        """Apply zero-velocity inverse-dynamics gravity only to right-arm DOFs."""
        import mujoco

        if self._snapshot is None or self._inverse_data is None:
            raise RuntimeError("passive mode is not enabled")
        scratch = self._inverse_data
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        scratch.qacc[:] = 0.0
        mujoco.mj_forward(self.model, scratch)
        result = np.zeros(self.model.nv, dtype=float)
        mujoco.mj_rne(self.model, scratch, 0, result)
        compensation = self._gravity_scale * result[self.mapping.right_dof]
        data.qfrc_applied[self.mapping.right_dof] += compensation
        return compensation.copy()

    def restore_position_control(self) -> None:
        if self._snapshot is None:
            return
        self.model.actuator_gainprm[self.mapping.right_actuators] = self._snapshot.gain
        self.model.actuator_biasprm[self.mapping.right_actuators] = self._snapshot.bias
        self.model.body_gravcomp[self._right_bodies] = self._snapshot.gravcomp
        self._snapshot = None
        self._inverse_data = None

    @property
    def enabled(self) -> bool:
        return self._snapshot is not None
