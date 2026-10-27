"""Explicit expert state machine with legal-transition enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

from .expert import ExpertPhase


_NEXT = {
    ExpertPhase.RESET: {ExpertPhase.PLAN, ExpertPhase.FAIL},
    ExpertPhase.PLAN: {ExpertPhase.MOVE_PREGRASP, ExpertPhase.FAIL},
    ExpertPhase.MOVE_PREGRASP: {ExpertPhase.APPROACH, ExpertPhase.FAIL},
    ExpertPhase.APPROACH: {ExpertPhase.CLOSE, ExpertPhase.FAIL},
    ExpertPhase.CLOSE: {ExpertPhase.VERIFY_GRASP, ExpertPhase.FAIL},
    ExpertPhase.VERIFY_GRASP: {ExpertPhase.LIFT, ExpertPhase.FAIL},
    ExpertPhase.LIFT: {ExpertPhase.HOLD, ExpertPhase.FAIL},
    ExpertPhase.HOLD: {ExpertPhase.TRANSPORT, ExpertPhase.FAIL},
    ExpertPhase.TRANSPORT: {ExpertPhase.PLACE, ExpertPhase.FAIL},
    ExpertPhase.PLACE: {ExpertPhase.RELEASE, ExpertPhase.FAIL},
    ExpertPhase.RELEASE: {ExpertPhase.RETREAT, ExpertPhase.FAIL},
    ExpertPhase.RETREAT: {ExpertPhase.RETURN_HOME, ExpertPhase.FAIL},
    ExpertPhase.RETURN_HOME: {ExpertPhase.SUCCESS, ExpertPhase.FAIL},
    ExpertPhase.SUCCESS: set(),
    ExpertPhase.FAIL: set(),
}


@dataclass
class ExpertFSM:
    phase: ExpertPhase = ExpertPhase.RESET
    transitions: list[dict] = field(default_factory=list)

    def reset(self, now_s: float = 0.0) -> None:
        self.phase = ExpertPhase.RESET
        self.transitions = [{"phase": self.phase.value, "time_s": float(now_s), "reason": "reset"}]

    def transition(self, phase: ExpertPhase, now_s: float, reason: str) -> None:
        if phase not in _NEXT[self.phase]:
            raise RuntimeError(f"illegal expert transition: {self.phase.value} -> {phase.value}")
        self.phase = phase
        self.transitions.append({"phase": phase.value, "time_s": float(now_s), "reason": reason})
