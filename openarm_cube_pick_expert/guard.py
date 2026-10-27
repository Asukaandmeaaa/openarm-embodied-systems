"""Reusable 16-axis position-action safety gateway."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GuardResult:
    action: np.ndarray
    disposition: str
    reasons: tuple[str, ...]


class ActionGuard:
    def __init__(
        self,
        lower,
        upper,
        max_velocity,
        max_acceleration,
        max_step,
        watchdog_timeout_s,
        limit_policy="clip",
        target_response_time_s=0.4,
    ):
        arrays = [np.asarray(v, dtype=np.float32) for v in (lower, upper, max_velocity, max_acceleration, max_step)]
        if any(v.shape != (16,) for v in arrays):
            raise ValueError("all action guard arrays must have shape (16,)")
        self.lower, self.upper, self.max_velocity, self.max_acceleration, self.max_step = arrays
        self.watchdog_timeout_s = float(watchdog_timeout_s)
        self.target_response_time_s = float(target_response_time_s)
        if self.target_response_time_s <= 0:
            raise ValueError("target_response_time_s must be positive")
        self.limit_policy = limit_policy
        self.stats = Counter()
        self.reset()

    def reset(self) -> None:
        self.stats.clear()
        self.target = None
        self.target_time = None
        self.previous_action = None
        self.previous_velocity = np.zeros(16, dtype=np.float32)
        self.previous_time = None

    def submit(self, target, now: float) -> bool:
        value = np.asarray(target)
        if value.shape != (16,):
            self.stats["reject_shape"] += 1
            return False
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            self.stats["reject_nonfinite"] += 1
            return False
        value = value.astype(np.float32)
        outside = np.any((value < self.lower) | (value > self.upper))
        if outside and self.limit_policy == "reject":
            self.stats["reject_ctrlrange"] += 1
            return False
        if outside:
            value = np.clip(value, self.lower, self.upper)
            self.stats["clip_ctrlrange"] += 1
        self.target, self.target_time = value, float(now)
        return True

    def step(self, current, now: float) -> GuardResult:
        current = np.asarray(current, dtype=np.float32)
        if current.shape != (16,) or not np.isfinite(current).all():
            raise ValueError("current state must be finite float32[16]")
        now = float(now)
        if self.target is None or self.target_time is None or now - self.target_time > self.watchdog_timeout_s:
            self.previous_action = current.copy()
            self.previous_velocity.fill(0)
            self.previous_time = now
            self.stats["hold_watchdog"] += 1
            return GuardResult(current.copy(), "hold", ("watchdog",))
        dt = max(1e-6, now - self.previous_time) if self.previous_time is not None else 0.001
        base = self.previous_action if self.previous_action is not None else current
        # A target error divided by the 1 kHz integration dt is effectively a
        # very high-gain position controller. With acceleration limiting it
        # repeatedly overshoots a fixed target and creates a visible limit
        # cycle. A finite response horizon makes the requested velocity decay
        # as the command approaches the target, so fixed targets converge
        # without reversing while retaining velocity/acceleration limits.
        desired_velocity = np.clip(
            (self.target - base) / self.target_response_time_s,
            -self.max_velocity,
            self.max_velocity,
        )
        velocity = np.clip(desired_velocity, self.previous_velocity - self.max_acceleration * dt, self.previous_velocity + self.max_acceleration * dt)
        delta = np.clip(velocity * dt, -self.max_step, self.max_step)
        action = np.clip(base + delta, self.lower, self.upper).astype(np.float32)
        reasons = []
        if np.any(np.abs(desired_velocity) >= self.max_velocity - 1e-7): reasons.append("velocity")
        if np.any(np.abs(velocity - self.previous_velocity) >= self.max_acceleration * dt - 1e-7): reasons.append("acceleration")
        if np.any(np.abs(delta) >= self.max_step - 1e-7): reasons.append("step")
        for reason in reasons: self.stats[f"clip_{reason}"] += 1
        self.previous_velocity = ((action - base) / dt).astype(np.float32)
        self.previous_action, self.previous_time = action, now
        return GuardResult(action, "clip" if reasons else "pass", tuple(reasons))
