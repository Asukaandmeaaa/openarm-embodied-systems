"""Validated operator-to-bridge packets and a receive-side watchdog."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OperatorPacket:
    schema_version: int
    session_id: str
    sequence: int
    sent_monotonic_s: float
    connected: bool
    deadman_pressed: bool
    active: bool
    stop_latched: bool
    open_sequence: int
    close_sequence: int
    arm_motion_source: str = "idle"

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported operator packet schema")
        if not self.session_id or len(self.session_id) > 128:
            raise ValueError("invalid operator session_id")
        if self.sequence < 0 or self.open_sequence < 0 or self.close_sequence < 0:
            raise ValueError("operator counters must be nonnegative")
        if not math.isfinite(self.sent_monotonic_s):
            raise ValueError("operator timestamp must be finite")
        for name in ("connected", "deadman_pressed", "active", "stop_latched"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"operator {name} must be bool")
        if self.active and (not self.connected or not self.deadman_pressed or self.stop_latched):
            raise ValueError("active packet violates deadman/stop invariants")
        if self.arm_motion_source not in (
            "idle",
            "cartesian_jog",
            "physical_kinesthetic",
        ):
            raise ValueError("unsupported real arm motion source")
        if self.active and self.arm_motion_source == "idle":
            raise ValueError("active packet must select one arm motion source")
        if not self.active and self.arm_motion_source != "idle":
            raise ValueError("inactive packet must use the idle arm motion source")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "OperatorPacket":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("operator packet must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise ValueError(f"operator packet fields mismatch: missing={missing}, extra={extra}")
        packet = cls(**raw)
        packet.validate()
        return packet


@dataclass(frozen=True)
class OperatorDecision:
    request_passive: bool
    request_open: bool = False
    request_close: bool = False
    command_fresh: bool = False
    armed: bool = False
    stop_reason: str | None = None
    packet_accepted: bool = False
    control_mode: str = "idle"


class OperatorWatchdog:
    """Fail-rigid gate for a reliable but non-real-time ROS command stream."""

    def __init__(self, timeout_s: float = 0.25):
        self._lock = threading.RLock()
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        self.timeout_s = float(timeout_s)
        self.session_id: str | None = None
        self.last_sequence = -1
        self.last_receive_monotonic_s: float | None = None
        self.open_sequence = 0
        self.close_sequence = 0
        self.armed = False
        self.request_passive = False
        self.control_mode = "idle"
        self.stop_reason: str | None = "no_operator_packet"

    def ingest(self, packet: OperatorPacket, receive_monotonic_s: float | None = None) -> OperatorDecision:
        # Clock sampling and state access must be atomic with decision(). A ROS
        # timer can otherwise compare an earlier clock sample to a packet that
        # another callback just received, falsely treating negative age as loss.
        with self._lock:
            received = time.monotonic() if receive_monotonic_s is None else receive_monotonic_s
            return self._ingest(packet, received)

    def _ingest(self, packet: OperatorPacket, receive_monotonic_s: float) -> OperatorDecision:
        packet.validate()
        if not math.isfinite(receive_monotonic_s):
            raise ValueError("receive timestamp must be finite")

        new_session = packet.session_id != self.session_id
        if new_session:
            self.session_id = packet.session_id
            self.last_sequence = -1
            self.open_sequence = packet.open_sequence
            self.close_sequence = packet.close_sequence
            self.armed = False
            self.request_passive = False
            self.control_mode = "idle"
            self.stop_reason = "new_operator_session_requires_deadman_release"

        if packet.sequence <= self.last_sequence:
            return self.decision(receive_monotonic_s)

        self.last_sequence = packet.sequence
        self.last_receive_monotonic_s = float(receive_monotonic_s)

        request_open = False
        request_close = False
        if packet.open_sequence > self.open_sequence:
            request_open = True
        if packet.close_sequence > self.close_sequence:
            request_close = True
        self.open_sequence = max(self.open_sequence, packet.open_sequence)
        self.close_sequence = max(self.close_sequence, packet.close_sequence)

        if not packet.connected:
            self.request_passive = False
            self.control_mode = "idle"
            self.armed = False
            self.stop_reason = "gamepad_disconnected"
            return self.decision(receive_monotonic_s, packet_accepted=True)
        if packet.stop_latched:
            self.request_passive = False
            self.control_mode = "idle"
            # The gamepad mapper keeps stop latched until LT has been released.
            # Remember that release so the deliberate next LT press may resume.
            self.armed = not packet.deadman_pressed
            self.stop_reason = "operator_stop_latched"
            return self.decision(receive_monotonic_s, packet_accepted=True)

        if not packet.deadman_pressed and not packet.active:
            self.armed = True
            self.request_passive = False
            self.control_mode = "idle"
            self.stop_reason = None
        elif self.armed and packet.active:
            self.control_mode = packet.arm_motion_source
            self.request_passive = self.control_mode == "physical_kinesthetic"
            self.stop_reason = None
        else:
            self.request_passive = False
            self.control_mode = "idle"
            self.stop_reason = "deadman_release_required"

        if self.stop_reason is not None:
            request_open = False
            request_close = False
        return OperatorDecision(
            request_passive=self.request_passive,
            request_open=request_open,
            request_close=request_close,
            command_fresh=True,
            armed=self.armed,
            stop_reason=self.stop_reason,
            packet_accepted=True,
            control_mode=self.control_mode,
        )

    def decision(
        self, now_monotonic_s: float | None = None, *, packet_accepted: bool = False
    ) -> OperatorDecision:
        with self._lock:
            now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
            return self._decision(now, packet_accepted=packet_accepted)

    def _decision(
        self, now_monotonic_s: float, *, packet_accepted: bool = False
    ) -> OperatorDecision:
        fresh = (
            self.last_receive_monotonic_s is not None
            and math.isfinite(now_monotonic_s)
            and 0.0 <= now_monotonic_s - self.last_receive_monotonic_s <= self.timeout_s
        )
        if not fresh:
            self.request_passive = False
            self.control_mode = "idle"
            self.armed = False
            self.stop_reason = "operator_watchdog_timeout"
        return OperatorDecision(
            request_passive=self.request_passive and fresh,
            command_fresh=fresh,
            armed=self.armed,
            stop_reason=self.stop_reason,
            packet_accepted=packet_accepted,
            control_mode=self.control_mode if fresh else "idle",
        )
