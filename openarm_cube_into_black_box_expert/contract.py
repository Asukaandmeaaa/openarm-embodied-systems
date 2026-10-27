"""Protocol-level contract for the fixed black-box task.

This module deliberately has no MuJoCo or dataset-writer dependency.  The
collector and replay validator can both use it, which keeps success criteria
identical during collection and audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


TASK_PROMPT = "Put the blue cube inside the black box, then return both arms home."
CONTRACT_VERSION = "openarm_cube_into_black_box_v3"

# These values describe the official demo.xml box, not a randomized target.
BLACK_BOX_CENTER_XY_M = (0.40, 0.0)
BLACK_BOX_FLOOR_Z_M = 1.01
BLACK_BOX_CUBE_CENTER_Z_M = 1.035
BLACK_BOX_SAFE_HALF_RANGE_M = (0.035, 0.035)

CUBE_CENTER_XY_M = (0.33, -0.28)
CUBE_RANDOM_HALF_RANGE_M = (0.05, 0.04)
CUBE_YAW_HALF_RANGE_RAD = 0.2617993877991494


class Phase(str, Enum):
    RESET = "RESET/STABILIZE"
    GRASP = "GRASP"
    LIFT_HOLD = "LIFT_HOLD"
    PLACE_STABLE = "PLACE_STABLE"
    RELEASE = "RELEASE"
    RETURN_HOME = "RETURN_HOME"
    HOME_STABLE = "HOME_STABLE"
    STATIC_TAIL = "STATIC_TAIL"
    SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class TaskContract:
    prompt: str = TASK_PROMPT
    version: str = CONTRACT_VERSION
    hold_s: float = 2.0
    placement_stable_s: float = 0.5
    home_stable_s: float = 1.0
    static_tail_s: float = 3.0
    min_lift_m: float = 0.05
    max_home_error_rad: float = 0.03
    max_box_center_z_error_m: float = 0.025
    max_static_joint_delta_rad: float = 0.005

    def validate_prompt(self, prompt: str) -> None:
        if prompt != self.prompt:
            raise ValueError(f"task prompt must be exact: {self.prompt!r}")

    def box_contains(self, cube_xyz: tuple[float, float, float] | list[float]) -> bool:
        x, y, z = map(float, cube_xyz)
        return (
            abs(x - BLACK_BOX_CENTER_XY_M[0]) <= BLACK_BOX_SAFE_HALF_RANGE_M[0]
            and abs(y - BLACK_BOX_CENTER_XY_M[1]) <= BLACK_BOX_SAFE_HALF_RANGE_M[1]
            and abs(z - BLACK_BOX_CUBE_CENTER_Z_M) <= self.max_box_center_z_error_m
        )

    def cube_reset_is_valid(self, cube_xyz: tuple[float, float, float], yaw_rad: float) -> bool:
        x, y, z = map(float, cube_xyz)
        return (
            abs(x - CUBE_CENTER_XY_M[0]) <= CUBE_RANDOM_HALF_RANGE_M[0] + 1e-9
            and abs(y - CUBE_CENTER_XY_M[1]) <= CUBE_RANDOM_HALF_RANGE_M[1] + 1e-9
            and abs(z - 1.05) <= 1e-6
            and abs(float(yaw_rad)) <= CUBE_YAW_HALF_RANGE_RAD + 1e-9
        )

    def metadata_template(self) -> dict[str, Any]:
        return {
            "task": self.prompt,
            "task_contract": self.version,
            "object": {
                "name": "blue_cube",
                "source_body_name": "orange_cube",
                "color": "blue",
                "xy_center_m": list(CUBE_CENTER_XY_M),
                "xy_random_half_range_m": list(CUBE_RANDOM_HALF_RANGE_M),
                "yaw_randomization_deg": 15.0,
            },
            "target": {
                "name": "black_box",
                "center_xy_m": list(BLACK_BOX_CENTER_XY_M),
                "safe_half_range_m": list(BLACK_BOX_SAFE_HALF_RANGE_M),
                "cube_center_z_m": BLACK_BOX_CUBE_CENTER_Z_M,
                "fixed": True,
                "visible": True,
            },
            "success_gates": {
                "min_lift_m": self.min_lift_m,
                "hold_s": self.hold_s,
                "placement_stable_s": self.placement_stable_s,
                "home_stable_s": self.home_stable_s,
                "static_tail_s": self.static_tail_s,
                "max_home_error_rad": self.max_home_error_rad,
                "one_grasp_only": True,
            },
            "runtime": {
                "active_arm": "right",
                "parked_arm": "left",
                "parked_arm_motion": False,
                "parked_arm_collision": False,
                "parked_arm_visible": False,
            },
        }


@dataclass
class _Window:
    started_at: float | None = None
    ended_at: float | None = None

    def update(self, now_s: float, valid: bool, required_s: float) -> bool:
        if not valid:
            self.started_at = None
            self.ended_at = None
            return False
        if self.started_at is None:
            self.started_at = float(now_s)
        self.ended_at = float(now_s)
        return self.ended_at - self.started_at + 1e-9 >= required_s

    @property
    def duration_s(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return self.ended_at - self.started_at


@dataclass
class CubeIntoBlackBoxEvaluator:
    """Streaming evaluator for collector and strict replay.

    Call the update methods at the simulation clock rate. A caller must keep
    the gate order; out-of-order calls are rejected so a late frame cannot
    accidentally manufacture a successful episode.
    """

    contract: TaskContract = field(default_factory=TaskContract)
    reset_cube_z_m: float = 1.05
    phase: Phase = Phase.RESET
    max_lift_m: float = 0.0
    grasp_count: int = 0
    hold_window: _Window = field(default_factory=_Window)
    placement_window: _Window = field(default_factory=_Window)
    home_window: _Window = field(default_factory=_Window)
    tail_window: _Window = field(default_factory=_Window)
    placement_seen: bool = False
    released: bool = False
    home_reached: bool = False
    evidence_log: list[dict[str, Any]] = field(default_factory=list)

    def _event(self, name: str, now_s: float, **details: Any) -> None:
        self.evidence_log.append({"event": name, "time_s": float(now_s), **details})

    def note_grasp(self, now_s: float, bilateral_contact: bool) -> None:
        if self.phase not in (Phase.RESET, Phase.GRASP):
            raise ValueError(f"grasp occurred after task progressed to {self.phase}")
        if bilateral_contact:
            self.grasp_count += 1
            if self.grasp_count > 1:
                raise ValueError("task permits exactly one grasp")
            self.phase = Phase.GRASP
            self._event("grasp", now_s)

    def update_lift_hold(
        self,
        *,
        now_s: float,
        cube_z_m: float,
        bilateral_contact: bool,
        relative_drift_m: float,
        cube_vertical_velocity_m_s: float,
    ) -> bool:
        if self.phase not in (Phase.GRASP, Phase.LIFT_HOLD):
            raise ValueError(f"lift/hold update is invalid in phase {self.phase}")
        lift = float(cube_z_m) - self.reset_cube_z_m
        self.max_lift_m = max(self.max_lift_m, lift)
        valid = (
            lift >= self.contract.min_lift_m
            and bilateral_contact
            and float(relative_drift_m) <= 0.025
            and abs(float(cube_vertical_velocity_m_s)) <= 0.35
        )
        self.phase = Phase.LIFT_HOLD
        complete = self.hold_window.update(float(now_s), valid, self.contract.hold_s)
        self._event("lift_hold_frame", now_s, lift_m=lift, valid=valid, complete=complete)
        return complete

    def update_placement(self, *, now_s: float, cube_xyz: tuple[float, float, float], inside: bool | None = None) -> bool:
        if self.phase not in (Phase.LIFT_HOLD, Phase.PLACE_STABLE):
            raise ValueError(f"placement update is invalid in phase {self.phase}")
        if self.max_lift_m + 1e-9 < self.contract.min_lift_m:
            raise ValueError("placement cannot precede the required lift")
        if self.hold_window.duration_s + 1e-9 < self.contract.hold_s:
            raise ValueError("placement cannot precede the required stable hold")
        contained = self.contract.box_contains(cube_xyz) if inside is None else bool(inside)
        complete = self.placement_window.update(float(now_s), contained, self.contract.placement_stable_s)
        self.phase = Phase.PLACE_STABLE
        self.placement_seen = self.placement_seen or complete
        self._event("placement_frame", now_s, inside=contained, complete=complete)
        return complete

    def note_release(self, now_s: float, both_fingers_detached: bool) -> None:
        if self.phase != Phase.PLACE_STABLE or not self.placement_seen:
            raise ValueError("release requires stable in-box placement")
        if not both_fingers_detached:
            raise ValueError("release requires both fingers detached")
        self.released = True
        self.phase = Phase.RELEASE
        self._event("release", now_s)

    def update_home(self, *, now_s: float, max_joint_error_rad: float) -> bool:
        if self.phase not in (Phase.RELEASE, Phase.RETURN_HOME):
            raise ValueError(f"home update is invalid in phase {self.phase}")
        if not self.released:
            raise ValueError("home return cannot precede release")
        valid = float(max_joint_error_rad) <= self.contract.max_home_error_rad
        complete = self.home_window.update(float(now_s), valid, self.contract.home_stable_s)
        self.phase = Phase.HOME_STABLE if complete else Phase.RETURN_HOME
        self.home_reached = self.home_reached or complete
        self._event("home_frame", now_s, error_rad=float(max_joint_error_rad), valid=valid, complete=complete)
        return complete

    def update_static_tail(self, *, now_s: float, max_joint_delta_rad: float) -> bool:
        if self.phase != Phase.HOME_STABLE or not self.home_reached:
            raise ValueError("static tail requires stable home")
        valid = float(max_joint_delta_rad) <= self.contract.max_static_joint_delta_rad
        complete = self.tail_window.update(float(now_s), valid, self.contract.static_tail_s)
        if complete:
            self.phase = Phase.SUCCESS
        self._event("static_tail_frame", now_s, delta_rad=float(max_joint_delta_rad), valid=valid, complete=complete)
        return complete

    @property
    def success(self) -> bool:
        return self.phase == Phase.SUCCESS

    def evidence(self) -> dict[str, Any]:
        return {
            "task_contract": self.contract.version,
            "task_prompt": self.contract.prompt,
            "phase": self.phase.value,
            "success": self.success,
            "grasp_count": self.grasp_count,
            "max_lift_height_m": self.max_lift_m,
            "hold_duration_s": self.hold_window.duration_s,
            "placement_stable_duration_s": self.placement_window.duration_s,
            "home_stable_duration_s": self.home_window.duration_s,
            "static_tail_duration_s": self.tail_window.duration_s,
            "events": list(self.evidence_log),
        }
