"""Shared discrete keyboard/gripper state machine for simulation and future ROS 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class GripperState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    HOLDING = "HOLDING"
    OPENING = "OPENING"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class KeyboardEvent:
    raw_timestamp_s: float
    key: str
    requested_target: str | None
    effective_target: float | None
    state_before: str
    state_after: str
    accepted: bool
    rejection_reason: str | None
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GripperController:
    OPEN_TARGET = -0.35
    CLOSE_TARGET = -0.10

    def __init__(self, backend: str = "mujoco"):
        self.backend = backend
        self.state = GripperState.OPEN
        self.effective_target = self.OPEN_TARGET
        self.paused = False
        self.stop_latched = False
        self._pressed: set[str] = set()
        self.events: list[KeyboardEvent] = []

    @staticmethod
    def normalize(key: str) -> str:
        value = key.strip()
        if value.lower() == "space": return "Space"
        if value.lower() in {"esc", "escape"}: return "Esc"
        return value.upper()

    def key_up(self, key: str) -> None:
        self._pressed.discard(self.normalize(key))

    def key_down(self, key: str, timestamp_s: float) -> KeyboardEvent:
        key = self.normalize(key); before = self.state
        requested = None; accepted = True; reason = None
        if key in self._pressed:
            accepted = False; reason = "key_repeat_ignored"
        else:
            self._pressed.add(key)
        bindings = {"O": "open", "C": "close", "R": "record_start", "S": "finish_success", "F": "finish_failure", "X": "cancel", "Space": "pause_hold", "Esc": "software_stop_latch"}
        if key not in bindings:
            accepted = False; reason = reason or "unbound_key"
        requested = bindings.get(key)
        if accepted and self.stop_latched and key != "Esc":
            accepted = False; reason = "software_stop_latched"
        if accepted:
            if key == "O":
                self.effective_target = self.OPEN_TARGET; self.state = GripperState.OPENING
            elif key == "C":
                self.effective_target = self.CLOSE_TARGET; self.state = GripperState.CLOSING
            elif key == "Space":
                self.paused = not self.paused
            elif key == "Esc":
                self.stop_latched = True; self.state = GripperState.STOPPED
        event = KeyboardEvent(float(timestamp_s), key, requested, self.effective_target, before.value, self.state.value, accepted, reason, self.backend)
        self.events.append(event)
        return event

    def update(self, measured_position: float, *, bilateral_contact: bool = False, fault: str | None = None) -> GripperState:
        if fault is not None:
            self.state = GripperState.FAULT
        elif self.stop_latched:
            self.state = GripperState.STOPPED
        elif self.state == GripperState.CLOSING and (bilateral_contact or abs(measured_position - self.CLOSE_TARGET) <= 0.03):
            self.state = GripperState.HOLDING
        elif self.state == GripperState.OPENING and abs(measured_position - self.OPEN_TARGET) <= 0.03:
            self.state = GripperState.OPEN
        return self.state

    def reset_stop(self, *, explicit_operator_action: bool) -> None:
        if not explicit_operator_action:
            raise PermissionError("stop reset requires explicit operator action")
        self.stop_latched = False; self.state = GripperState.OPEN
