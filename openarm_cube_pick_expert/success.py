"""Contact-aware grasp success accumulator."""

from dataclasses import dataclass


@dataclass
class SuccessTracker:
    reset_z: float
    min_lift_m: float = 0.05
    required_hold_s: float = 1.0
    max_tcp_distance_m: float = 0.075
    hold_start: float | None = None
    max_lift: float = 0.0
    hold_duration: float = 0.0

    def update(self, now: float, cube_z: float, tcp_distance: float, gripper_contact: bool, cube_vertical_velocity: float) -> bool:
        lift = cube_z - self.reset_z
        self.max_lift = max(self.max_lift, lift)
        associated = gripper_contact or tcp_distance <= self.max_tcp_distance_m
        # Reject ballistic/flicked cubes: a qualifying hold must be associated
        # with the gripper and nearly settled vertically.
        valid = lift >= self.min_lift_m and associated and abs(cube_vertical_velocity) < 0.35
        if valid:
            if self.hold_start is None: self.hold_start = float(now)
            self.hold_duration = float(now) - self.hold_start
        else:
            self.hold_start = None
            self.hold_duration = 0.0
        return self.hold_duration >= self.required_hold_s
