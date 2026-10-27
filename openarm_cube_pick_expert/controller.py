"""Position-actuator adapter shared by expert and policy replay."""

from __future__ import annotations

import numpy as np

from .guard import ActionGuard, GuardResult


class PositionController:
    def __init__(self, mapper, data, guard: ActionGuard, state_fn):
        self.mapper, self.data, self.guard, self.state_fn = mapper, data, guard, state_fn

    def reset(self) -> None:
        self.guard.reset()

    def command(self, target, now_s: float) -> GuardResult:
        if not self.guard.submit(np.asarray(target, dtype=np.float32), now_s):
            return self.guard.step(self.state_fn(), now_s)
        result = self.guard.step(self.state_fn(), now_s)
        self.mapper.set_ctrl(self.data.ctrl, result.action[:8], "right")
        self.mapper.set_ctrl(self.data.ctrl, result.action[8:], "left")
        return result
