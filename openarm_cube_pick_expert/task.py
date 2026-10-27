"""PickCube task evaluation, independent from planning and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HoldFrame:
    time_s: float
    lift_m: float
    bilateral_contact: bool
    tcp_distance_m: float
    relative_drift_m: float
    cube_vertical_velocity_m_s: float
    valid: bool


@dataclass
class PickCubeEvaluator:
    reset_z: float
    min_lift_m: float = 0.05
    required_hold_s: float = 1.0
    max_tcp_distance_m: float = 0.075
    max_relative_drift_m: float = 0.025
    max_vertical_velocity_m_s: float = 0.35
    hold_start_s: float | None = None
    max_lift_m: float = 0.0
    frames: list[HoldFrame] = field(default_factory=list)

    @property
    def hold_duration_s(self) -> float:
        if self.hold_start_s is None or not self.frames:
            return 0.0
        return self.frames[-1].time_s - self.hold_start_s

    def update(
        self,
        *,
        now_s: float,
        cube_z: float,
        bilateral_contact: bool,
        tcp_distance_m: float,
        relative_drift_m: float,
        cube_vertical_velocity_m_s: float,
    ) -> bool:
        lift = float(cube_z - self.reset_z)
        self.max_lift_m = max(self.max_lift_m, lift)
        # Contact is mandatory. Distance/drift are association and drop/flick guards.
        valid = (
            lift + 1e-9 >= self.min_lift_m
            and bilateral_contact
            and tcp_distance_m <= self.max_tcp_distance_m
            and relative_drift_m <= self.max_relative_drift_m
            and abs(cube_vertical_velocity_m_s) <= self.max_vertical_velocity_m_s
        )
        frame = HoldFrame(
            float(now_s), lift, bool(bilateral_contact), float(tcp_distance_m),
            float(relative_drift_m), float(cube_vertical_velocity_m_s), valid,
        )
        self.frames.append(frame)
        if valid:
            if self.hold_start_s is None:
                self.hold_start_s = float(now_s)
        else:
            self.hold_start_s = None
        return self.hold_duration_s + 1e-9 >= self.required_hold_s

    def evidence(self) -> dict:
        valid = [f for f in self.frames if f.valid]
        return {
            "max_lift_height_m": self.max_lift_m,
            "hold_duration_s": self.hold_duration_s,
            "hold_window_start_s": self.hold_start_s,
            "hold_window_end_s": valid[-1].time_s if valid and self.hold_start_s is not None else None,
            "hold_frames": [f.__dict__ for f in self.frames],
        }
