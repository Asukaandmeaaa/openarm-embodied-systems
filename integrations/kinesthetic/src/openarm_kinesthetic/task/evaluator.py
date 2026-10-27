"""Single streaming success evaluator for teaching, collection, and replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..config import load_yaml


@dataclass(frozen=True)
class EvaluationSample:
    time_s: float
    cube_xyz: tuple[float, float, float]
    cube_linear_velocity: tuple[float, float, float]
    finger_contacts: tuple[bool, bool]
    right_joints: tuple[float, ...]
    relative_grasp_drift_m: float = 0.0
    forbidden_collision: bool = False


@dataclass(frozen=True)
class TaskResult:
    task_success: bool
    full_sequence_success: bool
    phase: str
    grasp_count: int
    max_lift_m: float
    bilateral_hold_s: float
    placement_stable_s: float
    ready_stable_s: float
    static_tail_s: float
    failures: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]


@dataclass
class _ContinuousWindow:
    start: float | None = None
    duration: float = 0.0

    def update(self, now: float, valid: bool) -> float:
        if not valid:
            self.start = None
            self.duration = 0.0
        else:
            if self.start is None:
                self.start = now
            self.duration = max(0.0, now - self.start)
        return self.duration


@dataclass
class TaskEvaluator:
    """Order-sensitive implementation of the authoritative task contract."""

    reset_cube_z_m: float
    ready_joints: np.ndarray
    box_center_xy_m: tuple[float, float]
    box_safe_half_range_m: tuple[float, float]
    box_cube_center_z_m: float
    box_z_tolerance_m: float = 0.015
    phase: str = "WAITING_FOR_GRASP"
    grasp_count: int = 0
    max_lift_m: float = 0.0
    task_success: bool = False
    full_sequence_success: bool = False
    _previous_bilateral: bool = False
    _lost_contact_since: float | None = None
    _hold_complete: bool = False
    _placement_complete: bool = False
    _home_complete_at: float | None = None
    _release_noted: bool = False
    _last_time: float | None = None
    _hold: _ContinuousWindow = field(default_factory=_ContinuousWindow)
    _placement: _ContinuousWindow = field(default_factory=_ContinuousWindow)
    _ready: _ContinuousWindow = field(default_factory=_ContinuousWindow)
    _tail: _ContinuousWindow = field(default_factory=_ContinuousWindow)
    failures: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ready_joints = np.asarray(self.ready_joints, dtype=float)
        if self.ready_joints.shape != (7,) or not np.all(np.isfinite(self.ready_joints)):
            raise ValueError("ready_joints must be finite 7D")
        cfg = load_yaml("task_contract.yaml")["success"]
        self.minimum_lift_m = float(cfg["minimum_lift_m"])
        self.required_hold_s = float(cfg["bilateral_contact_hold_s"])
        self.required_placement_s = float(cfg["placement_stable_s"])
        self.ready_tolerance_rad = float(cfg["maximum_ready_joint_error_rad"])
        self.required_ready_s = float(cfg["ready_stable_s"])
        self.required_tail_s = float(cfg["static_tail_s"])

    @classmethod
    def from_model(cls, model, data, mapping, ready_joints: np.ndarray) -> "TaskEvaluator":
        """Build placement bounds from the safe-region site in the loaded scene."""
        import mujoco

        site_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "black_box_safe_region")
        )
        if site_id < 0:
            raise ValueError("scene is missing the black_box_safe_region site")
        size = np.asarray(model.site_size[site_id], dtype=float)
        center = np.asarray(data.site_xpos[site_id], dtype=float)
        if size.shape != (3,) or center.shape != (3,) or not np.all(
            np.isfinite(np.r_[size, center])
        ):
            raise ValueError("black_box_safe_region must have finite 3D geometry")
        return cls(
            reset_cube_z_m=float(data.qpos[mapping.cube_qpos + 2]),
            ready_joints=np.asarray(ready_joints, dtype=float),
            box_center_xy_m=(float(center[0]), float(center[1])),
            box_safe_half_range_m=(float(size[0]), float(size[1])),
            box_cube_center_z_m=float(center[2]),
            box_z_tolerance_m=float(size[2] + 0.005),
        )

    def _inside_box(self, xyz: tuple[float, float, float]) -> bool:
        return (
            abs(xyz[0] - self.box_center_xy_m[0]) <= self.box_safe_half_range_m[0]
            and abs(xyz[1] - self.box_center_xy_m[1]) <= self.box_safe_half_range_m[1]
            and abs(xyz[2] - self.box_cube_center_z_m) <= self.box_z_tolerance_m
        )

    def update(self, sample: EvaluationSample) -> TaskResult:
        now = float(sample.time_s)
        values = np.asarray((*sample.cube_xyz, *sample.cube_linear_velocity, *sample.right_joints), dtype=float)
        if not np.all(np.isfinite(values)):
            self.failures.append("non_finite_state")
            return self.result()
        if len(sample.right_joints) != 7:
            self.failures.append("wrong_joint_order_or_dimension")
            return self.result()
        if self._last_time is not None and now <= self._last_time:
            self.failures.append("non_monotonic_time")
            return self.result()
        self._last_time = now
        if sample.forbidden_collision and "forbidden_collision" not in self.failures:
            self.failures.append("forbidden_collision")

        bilateral = bool(sample.finger_contacts[0] and sample.finger_contacts[1])
        if bilateral and not self._previous_bilateral and not self._placement_complete:
            is_effective = self.grasp_count == 0 or (
                self._lost_contact_since is not None and now - self._lost_contact_since >= 0.10
            )
            if is_effective:
                self.grasp_count += 1
                self.evidence.append({"event": "effective_grasp", "time_s": now, "count": self.grasp_count})
            self._lost_contact_since = None
            if self.grasp_count > 1 and "more_than_one_effective_grasp" not in self.failures:
                self.failures.append("more_than_one_effective_grasp")
        elif not bilateral and self._previous_bilateral:
            self._lost_contact_since = now
        self._previous_bilateral = bilateral

        lift = float(sample.cube_xyz[2]) - self.reset_cube_z_m
        self.max_lift_m = max(self.max_lift_m, lift)
        hold_valid = (
            self.grasp_count == 1
            and bilateral
            and lift >= self.minimum_lift_m
            and sample.relative_grasp_drift_m <= 0.025
            and float(np.linalg.norm(sample.cube_linear_velocity)) <= 0.35
        )
        if not self._hold_complete and self._hold.update(now, hold_valid) >= self.required_hold_s:
            self._hold_complete = True
            self.phase = "LIFT_HOLD_COMPLETE"
            self.evidence.append({"event": "lift_hold_complete", "time_s": now, "lift_m": lift})

        speed = float(np.linalg.norm(sample.cube_linear_velocity))
        place_valid = self._hold_complete and self._inside_box(sample.cube_xyz) and speed <= 0.05
        if not self._placement_complete and self._placement.update(now, place_valid) >= self.required_placement_s:
            self._placement_complete = True
            self.phase = "PLACEMENT_STABLE"
            self.evidence.append({"event": "placement_stable", "time_s": now})

        if self._placement_complete and not bilateral and not any(sample.finger_contacts) and not self._release_noted:
            self._release_noted = True
            self.task_success = self.grasp_count == 1 and not self.failures
            self.phase = "TASK_SUCCESS" if self.task_success else "TASK_FAILED"
            self.evidence.append({"event": "both_fingers_detached", "time_s": now})

        q = np.asarray(sample.right_joints, dtype=float)
        ready_error = float(np.max(np.abs(q - self.ready_joints)))
        ready_valid = self.task_success and ready_error <= self.ready_tolerance_rad
        if self._ready.update(now, ready_valid) >= self.required_ready_s and self._home_complete_at is None:
            self._home_complete_at = now
            self.phase = "READY_STABLE"
            self.evidence.append({"event": "ready_stable", "time_s": now, "max_error_rad": ready_error})

        tail_valid = self._home_complete_at is not None and ready_valid
        if self._tail.update(now, tail_valid) >= self.required_tail_s:
            self.full_sequence_success = self.task_success and not self.failures
            self.phase = "FULL_SEQUENCE_SUCCESS" if self.full_sequence_success else "SEQUENCE_FAILED"
        return self.result()

    def result(self) -> TaskResult:
        return TaskResult(
            task_success=self.task_success,
            full_sequence_success=self.full_sequence_success,
            phase=self.phase,
            grasp_count=self.grasp_count,
            max_lift_m=self.max_lift_m,
            bilateral_hold_s=self._hold.duration,
            placement_stable_s=self._placement.duration,
            ready_stable_s=self._ready.duration,
            static_tail_s=self._tail.duration,
            failures=tuple(dict.fromkeys(self.failures)),
            evidence=tuple(self.evidence),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.result())
