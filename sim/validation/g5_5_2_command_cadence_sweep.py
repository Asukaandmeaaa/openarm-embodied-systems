#!/usr/bin/env python3
"""G5.5.2 controlled command-cadence sweep.

This experiment keeps the source expert trajectory, initial state, physical
duration, 250 Hz executor clock, 1000 Hz physics clock, Hermite algorithm and
ActionGuard fixed.  Only the knot cadence given to the official
HermiteUpsampler changes.

Two chunk contracts are supported:

* ``act``: fixed 3.3 s chunks (100 knots at 30 Hz), matching the ACT horizon.
* ``full``: one full-duration chunk, reproducing the look-ahead structure used
  by the successful G5.5.1 60 Hz counterfactual.

No ACT model is trained or evaluated here.  Every knot is timestamp-based
linear resampling of the same recorded 30 Hz expert action stream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[2]
OPENARM_MUJOCO_SRC = ROOT / "official/openarm_mujoco/src"
EXECUTOR_SRC = (
    ROOT
    / "official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src"
)
sys.path[:0] = [str(ROOT), str(OPENARM_MUJOCO_SRC), str(EXECUTOR_SRC)]

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal
import types
import yaml

from openarm_cube_pick_expert.config import sha256
from openarm_cube_pick_expert.guard import ActionGuard
from openarm_cube_pick_expert.task import PickCubeEvaluator
from sim.validation.g5_5_1_action_representation_debug import (
    finger_contacts,
    get_state_vector,
    initialize_physics,
    load_expert_trajectory,
    stabilize_robot,
)


SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
PARAMS = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
DEFAULT_DATASET = (
    ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
)
DEFAULT_LEROBOT_DATASET = (
    ROOT / "data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen"
)
OFFICIAL_EXECUTOR = (
    ROOT
    / "official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/"
    "src/dora_openarm_actions_executor/main.py"
)

PHYSICS_HZ = 1000.0
CONTROL_HZ = 250.0
DEFAULT_CADENCES = (30.0, 40.0, 50.0, 60.0, 90.0, 120.0)
DEFAULT_DURATION_S = 33.0
ACT_HORIZON_S = 3.3
JOINT_NAMES = (
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5",
    "right_j6", "right_j7", "right_gripper", "left_j1", "left_j2",
    "left_j3", "left_j4", "left_j5", "left_j6", "left_j7",
    "left_gripper",
)


def load_official_executor_classes():
    """Load the exact classes used by the official Dora executor."""
    sys.modules.setdefault("dora", types.ModuleType("dora"))
    from dora_openarm_actions_executor.main import BiquadLowpass, HermiteUpsampler

    return HermiteUpsampler, BiquadLowpass


def _integer_intervals(duration_s: float, hz: float, label: str) -> int:
    intervals = int(round(duration_s * hz))
    if not math.isclose(intervals / hz, duration_s, abs_tol=1e-10):
        raise ValueError(
            f"{label}: duration {duration_s}s is not an integer number of "
            f"intervals at {hz}Hz"
        )
    return intervals


def relative_source_time(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert recorded timestamps to a strictly increasing relative clock."""
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps_ns.ndim != 1 or len(timestamps_ns) < 2:
        raise ValueError("source timestamps must be a 1D array with >=2 entries")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    return (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9


def load_lerobot_action_source(
    dataset: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load one episode's exact action labels from a local LeRobot v3 dataset."""
    tables = [
        pq.read_table(
            path, columns=["action", "timestamp", "episode_index", "frame_index"]
        )
        for path in sorted((dataset / "data").rglob("*.parquet"))
    ]
    if not tables:
        raise FileNotFoundError(f"no LeRobot parquet files below {dataset / 'data'}")
    table = pa.concat_tables(tables)
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    selected = np.flatnonzero(episodes == int(episode_index))
    if not len(selected):
        raise ValueError(f"LeRobot episode {episode_index} is absent")
    frames = np.asarray(table["frame_index"], dtype=np.int64)[selected]
    selected = selected[np.argsort(frames)]
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)[selected]
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[selected]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("LeRobot action timestamps are not strictly increasing")
    return timestamps - timestamps[0], actions


def interpolate_source(
    source_time_s: np.ndarray,
    source_actions: np.ndarray,
    query_time_s: np.ndarray,
) -> np.ndarray:
    """Timestamp-based linear interpolation without extrapolation."""
    source_time_s = np.asarray(source_time_s, dtype=np.float64)
    source_actions = np.asarray(source_actions, dtype=np.float32)
    query_time_s = np.asarray(query_time_s, dtype=np.float64)
    if source_actions.ndim != 2 or source_actions.shape[1] != 16:
        raise ValueError("source actions must have shape [N,16]")
    if len(source_time_s) != len(source_actions):
        raise ValueError("source timestamps/actions length mismatch")
    tolerance = 1e-9
    if query_time_s[0] < source_time_s[0] - tolerance:
        raise ValueError("query begins before source trajectory")
    if query_time_s[-1] > source_time_s[-1] + tolerance:
        raise ValueError("query ends after source trajectory")
    output = np.empty((len(query_time_s), 16), dtype=np.float32)
    for axis in range(16):
        output[:, axis] = np.interp(
            query_time_s, source_time_s, source_actions[:, axis]
        )
    return output


@dataclass(frozen=True)
class ExecutorTrajectory:
    cadence_hz: float
    chunk_horizon_s: float
    duration_s: float
    control_time_s: np.ndarray
    source_reference: np.ndarray
    hermite_targets: np.ndarray
    knot_count: int
    chunk_count: int


def build_executor_trajectory(
    source_time_s: np.ndarray,
    source_actions: np.ndarray,
    *,
    cadence_hz: float,
    duration_s: float,
    chunk_horizon_s: float,
    control_hz: float = CONTROL_HZ,
) -> ExecutorTrajectory:
    """Build an exact-duration trajectory using the official Hermite class.

    Chunk endpoints are not emitted twice.  The returned control timestamps are
    the global half-open grid ``[0, duration_s)`` at exactly ``control_hz``.
    """
    cadence_hz = float(cadence_hz)
    duration_s = float(duration_s)
    chunk_horizon_s = float(chunk_horizon_s)
    if cadence_hz <= 0 or duration_s <= 0 or chunk_horizon_s <= 0:
        raise ValueError("cadence, duration and chunk horizon must be positive")

    chunk_count = _integer_intervals(
        duration_s, 1.0 / chunk_horizon_s, "chunk contract"
    )
    knot_intervals = _integer_intervals(
        chunk_horizon_s, cadence_hz, "Hermite chunk"
    )
    control_intervals = _integer_intervals(
        chunk_horizon_s, control_hz, "executor chunk"
    )
    total_control_intervals = _integer_intervals(
        duration_s, control_hz, "executor duration"
    )

    HermiteUpsampler, _ = load_official_executor_classes()
    upsampler = HermiteUpsampler(
        chunk_hz=cadence_hz, horizon_sec=chunk_horizon_s
    )

    target_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    total_knots = 0
    local_knot_time = np.arange(knot_intervals + 1, dtype=np.float64) / cadence_hz
    # Half-open grid prevents duplicated endpoints and clock drift across chunks.
    local_control_time = (
        np.arange(control_intervals, dtype=np.float64) / control_hz
    )

    for chunk_index in range(chunk_count):
        chunk_start = chunk_index * chunk_horizon_s
        knot_time = chunk_start + local_knot_time
        knots = interpolate_source(source_time_s, source_actions, knot_time)
        target_parts.append(upsampler.upsample(knots, local_control_time))
        time_parts.append(chunk_start + local_control_time)
        total_knots += len(knots)

    control_time = np.concatenate(time_parts)
    targets = np.concatenate(target_parts)
    if len(control_time) != total_control_intervals:
        raise AssertionError("executor grid length drifted from contract")
    expected_time = np.arange(total_control_intervals, dtype=np.float64) / control_hz
    if not np.allclose(control_time, expected_time, atol=1e-12, rtol=0):
        raise AssertionError("executor grid is not a uniform global 250Hz clock")
    reference = interpolate_source(source_time_s, source_actions, control_time)

    return ExecutorTrajectory(
        cadence_hz=cadence_hz,
        chunk_horizon_s=chunk_horizon_s,
        duration_s=duration_s,
        control_time_s=control_time,
        source_reference=reference,
        hermite_targets=targets,
        knot_count=total_knots,
        chunk_count=chunk_count,
    )


def error_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """Absolute-error summary over aligned [N,16] trajectories."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 16:
        raise ValueError("error inputs must have matching [N,16] shapes")
    error = np.abs(a - b)
    return {
        "mean_rad": float(np.mean(error)),
        "p95_rad": float(np.percentile(error, 95)),
        "max_rad": float(np.max(error)),
        "per_axis_mean_rad": np.mean(error, axis=0).tolist(),
        "per_axis_max_rad": np.max(error, axis=0).tolist(),
    }


def pre_guard_dynamics(
    targets: np.ndarray,
    *,
    control_hz: float,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_step: np.ndarray,
) -> dict:
    """Measure 250Hz Hermite dynamics before ActionGuard."""
    dt = 1.0 / float(control_hz)
    velocity = np.diff(targets, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    step = np.diff(targets, axis=0)

    def metric(values: np.ndarray, limits: np.ndarray) -> dict:
        mask = np.abs(values) > limits[None, :] + 1e-7
        return {
            "axis_event_count": int(np.count_nonzero(mask)),
            "axis_event_ratio": float(np.mean(mask)),
            "step_any_ratio": float(np.mean(np.any(mask, axis=1))),
            "per_axis_event_ratio": np.mean(mask, axis=0).tolist(),
            "per_axis_max_abs": np.max(np.abs(values), axis=0).tolist(),
        }

    return {
        "velocity": metric(velocity, max_velocity),
        "acceleration": metric(acceleration, max_acceleration),
        "step": metric(step, max_step),
    }


def signed_phase_lag_ms(
    reference: np.ndarray,
    output: np.ndarray,
    *,
    sample_hz: float,
    max_lag_s: float = 0.5,
) -> dict:
    """Per-axis lag; positive means output lags the reference.

    Velocity signals are used so static joint offsets do not dominate the
    correlation.  Axes with negligible excitation return ``None``.
    """
    reference_velocity = np.diff(reference, axis=0) * sample_hz
    output_velocity = np.diff(output, axis=0) * sample_hz
    max_lag = int(round(max_lag_s * sample_hz))
    per_axis: list[float | None] = []
    for axis in range(16):
        ref = reference_velocity[:, axis]
        out = output_velocity[:, axis]
        if np.std(ref) < 1e-5 or np.std(out) < 1e-5:
            per_axis.append(None)
            continue
        ref = ref - np.mean(ref)
        out = out - np.mean(out)
        correlation = signal.correlate(out, ref, mode="full", method="fft")
        lags = signal.correlation_lags(len(out), len(ref), mode="full")
        allowed = np.abs(lags) <= max_lag
        lag_samples = int(lags[allowed][np.argmax(correlation[allowed])])
        per_axis.append(lag_samples / sample_hz * 1000.0)
    excited = [value for value in per_axis if value is not None]
    return {
        "sign_convention": "positive means output lags reference",
        "per_axis_ms": per_axis,
        "median_excited_axes_ms": (
            float(np.median(excited)) if excited else None
        ),
    }


@dataclass
class GuardDiagnostics:
    """Exact ActionGuard activity at physics-step, axis and command levels."""

    physics_steps: int = 0
    commands: int = 0
    active_step_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in ("velocity", "acceleration", "step")}
    )
    intervention_step_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in ("velocity", "acceleration", "step")}
    )
    active_axis_counts: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            name: np.zeros(16, dtype=np.int64)
            for name in ("velocity", "acceleration", "step")
        }
    )
    intervention_axis_counts: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            name: np.zeros(16, dtype=np.int64)
            for name in ("velocity", "acceleration", "step")
        }
    )
    command_any_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in ("velocity", "acceleration", "step")}
    )
    _command_any: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in ("velocity", "acceleration", "step")}
    )

    def begin_command(self) -> None:
        self.commands += 1
        self._command_any = {
            name: False for name in ("velocity", "acceleration", "step")
        }

    def observe(
        self,
        active_masks: dict[str, np.ndarray],
        intervention_masks: dict[str, np.ndarray],
    ) -> None:
        self.physics_steps += 1
        for name in ("velocity", "acceleration", "step"):
            active = np.asarray(active_masks[name], dtype=bool)
            intervention = np.asarray(intervention_masks[name], dtype=bool)
            self.active_axis_counts[name] += active.astype(np.int64)
            self.intervention_axis_counts[name] += intervention.astype(np.int64)
            if np.any(active):
                self.active_step_counts[name] += 1
                self._command_any[name] = True
            if np.any(intervention):
                self.intervention_step_counts[name] += 1

    def end_command(self) -> None:
        for name, active in self._command_any.items():
            if active:
                self.command_any_counts[name] += 1

    def as_dict(self) -> dict:
        result: dict[str, object] = {
            "physics_steps": self.physics_steps,
            "commands_250hz": self.commands,
            "definitions": {
                "active": "the limiter is at its configured boundary",
                "intervention": "the unconstrained value strictly exceeds the boundary",
                "physics_step_ratio": "steps with any affected axis / 1000Hz physics steps",
                "axis_ratio": "affected (physics step, axis) pairs / (physics steps * 16)",
                "command_ratio": "250Hz commands with any affected 1000Hz substep / commands",
            },
        }
        for name in ("velocity", "acceleration", "step"):
            result[name] = {
                "active_physics_step_count": self.active_step_counts[name],
                "active_physics_step_ratio": self.active_step_counts[name] / self.physics_steps,
                "active_axis_event_count": int(np.sum(self.active_axis_counts[name])),
                "active_axis_event_ratio": float(
                    np.sum(self.active_axis_counts[name]) / (self.physics_steps * 16)
                ),
                "active_per_axis_ratio": (
                    self.active_axis_counts[name] / self.physics_steps
                ).tolist(),
                "intervention_physics_step_count": self.intervention_step_counts[name],
                "intervention_physics_step_ratio": (
                    self.intervention_step_counts[name] / self.physics_steps
                ),
                "intervention_axis_event_count": int(
                    np.sum(self.intervention_axis_counts[name])
                ),
                "intervention_axis_event_ratio": float(
                    np.sum(self.intervention_axis_counts[name])
                    / (self.physics_steps * 16)
                ),
                "intervention_per_axis_ratio": (
                    self.intervention_axis_counts[name] / self.physics_steps
                ).tolist(),
                "command_any_count": self.command_any_counts[name],
                "command_any_ratio": self.command_any_counts[name] / self.commands,
            }
        return result


def predict_guard_masks(
    guard: ActionGuard,
    current: np.ndarray,
    now: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Reproduce the production guard math without mutating it."""
    if guard.target is None or guard.target_time is None:
        raise RuntimeError("guard target is not initialized")
    if now - guard.target_time > guard.watchdog_timeout_s:
        raise RuntimeError("unexpected watchdog hold during cadence experiment")
    dt = max(1e-6, now - guard.previous_time) if guard.previous_time is not None else 0.001
    base = guard.previous_action if guard.previous_action is not None else current
    raw_velocity = (guard.target - base) / dt
    desired_velocity = np.clip(raw_velocity, -guard.max_velocity, guard.max_velocity)
    lower_acc = guard.previous_velocity - guard.max_acceleration * dt
    upper_acc = guard.previous_velocity + guard.max_acceleration * dt
    velocity = np.clip(desired_velocity, lower_acc, upper_acc)
    raw_delta = velocity * dt

    active = {
        "velocity": np.abs(desired_velocity) >= guard.max_velocity - 1e-7,
        "acceleration": (
            np.abs(velocity - guard.previous_velocity)
            >= guard.max_acceleration * dt - 1e-7
        ),
        "step": np.abs(np.clip(raw_delta, -guard.max_step, guard.max_step)) >= guard.max_step - 1e-7,
    }
    intervention = {
        "velocity": np.abs(raw_velocity) > guard.max_velocity + 1e-7,
        "acceleration": (desired_velocity < lower_acc - 1e-7)
        | (desired_velocity > upper_acc + 1e-7),
        "step": np.abs(raw_delta) > guard.max_step + 1e-7,
    }
    return active, intervention


@dataclass
class TaskDiagnostics:
    success: bool = False
    first_success_s: float | None = None
    max_lift_m: float = 0.0
    max_hold_s: float = 0.0
    left_contact_steps: int = 0
    right_contact_steps: int = 0
    dual_contact_steps: int = 0
    first_dual_contact_s: float | None = None


def summarize_task(task: TaskDiagnostics, duration_s: float) -> dict:
    return {
        "success": task.success,
        "failure_reason": None if task.success else "lift_hold_condition_not_met",
        "max_lift_m": task.max_lift_m,
        "max_hold_s": task.max_hold_s,
        "left_contact_physics_steps": task.left_contact_steps,
        "right_contact_physics_steps": task.right_contact_steps,
        "dual_contact_physics_steps": task.dual_contact_steps,
        "first_dual_contact_s": task.first_dual_contact_s,
        "first_success_s": task.first_success_s,
        "executed_duration_s": duration_s,
    }


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_one(
    *,
    scene: Path,
    cfg: dict,
    expert_data: dict,
    trajectory: ExecutorTrajectory,
    use_lowpass: bool,
    control_mode: str,
    terminal_hold_s: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Execute one controlled cadence condition in MuJoCo."""
    if control_mode not in {"guard", "direct"}:
        raise ValueError(f"unsupported control mode: {control_mode}")
    model, data, mapper, addresses = initialize_physics(
        scene, int(expert_data["entry"]["seed"]), expert_data["entry"]
    )
    if not np.isclose(model.opt.timestep, 1.0 / PHYSICS_HZ):
        raise ValueError(f"expected 1000Hz physics, got dt={model.opt.timestep}")

    initial_qpos = np.asarray(expert_data["entry"]["initial_qpos"], dtype=np.float64)
    mapper.set_qpos(data.qpos, initial_qpos[:8], "right")
    mapper.set_qpos(data.qpos, initial_qpos[8:], "left")
    data.qvel.fill(0)
    mujoco.mj_forward(model, data)

    lower = np.r_[model.actuator_ctrlrange[9:17, 0], model.actuator_ctrlrange[1:9, 0]]
    upper = np.r_[model.actuator_ctrlrange[9:17, 1], model.actuator_ctrlrange[1:9, 1]]
    guard_cfg = cfg["guard"]
    guard = ActionGuard(
        lower,
        upper,
        np.asarray(guard_cfg["max_velocity"], dtype=np.float32),
        np.asarray(guard_cfg["max_acceleration"], dtype=np.float32),
        np.asarray(guard_cfg["max_step"], dtype=np.float32),
        guard_cfg["watchdog_timeout_s"],
        guard_cfg["limit_policy"],
    )
    stabilize_robot(model, data, mapper, guard, cfg)
    stabilize_end_state = get_state_vector(mapper, data.qpos).copy()
    # Preserve the dynamic state established by stabilization.  Production
    # deployment and the official-equivalence validators continue with the
    # same previous_action/previous_velocity; resetting here changes the
    # trajectory even when Hermite targets are bitwise equivalent.  The
    # experiment uses an independent GuardDiagnostics accumulator, so no
    # production-stat reset is needed.

    terminal_hold_intervals = _integer_intervals(
        terminal_hold_s, CONTROL_HZ, "terminal hold"
    )
    raw_targets = trajectory.hermite_targets.copy()
    if terminal_hold_intervals:
        raw_targets = np.concatenate(
            [
                raw_targets,
                np.repeat(
                    raw_targets[-1][None, :], terminal_hold_intervals, axis=0
                ),
            ]
        )
    targets = raw_targets
    if use_lowpass:
        _, BiquadLowpass = load_official_executor_classes()
        lowpass = BiquadLowpass(fs=CONTROL_HZ, fc=15.0)
        lowpass.reset_state(targets[0])
        targets = np.asarray([lowpass.step(target) for target in targets])

    reset_z = float(data.qpos[addresses["cube_qadr"] + 2])
    evaluator = PickCubeEvaluator(
        reset_z,
        cfg["hold_height_m"],
        cfg["hold_s"],
        cfg["tcp_distance_m"],
        cfg.get("max_relative_drift_m", 0.025),
    )
    tcp_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame"
    )
    task = TaskDiagnostics()
    reference_relative = None
    run_start_s = float(data.time)

    guard_diagnostics = GuardDiagnostics() if control_mode == "guard" else None
    guarded_actions: list[np.ndarray] = []
    actual_states: list[np.ndarray] = []
    physics_times: list[float] = []
    target_at_physics: list[np.ndarray] = []
    physics_steps_per_control = _integer_intervals(
        1.0 / CONTROL_HZ, PHYSICS_HZ, "physics/control ratio"
    )

    for target in targets:
        if control_mode == "guard":
            if not guard.submit(target, float(data.time)):
                raise RuntimeError("ActionGuard rejected a finite expert-derived target")
            assert guard_diagnostics is not None
            guard_diagnostics.begin_command()
        else:
            # This is the official MuJoCo actuator path: the executor emits a
            # position target and dora-openarm-mujoco writes it to data.ctrl.
            mapper.set_ctrl(data.ctrl, target[:8], "right")
            mapper.set_ctrl(data.ctrl, target[8:], "left")
        for _ in range(physics_steps_per_control):
            if control_mode == "guard":
                current = get_state_vector(mapper, data.qpos)
                active, intervention = predict_guard_masks(
                    guard, current, float(data.time)
                )
                guarded = guard.step(current, float(data.time))
                for reason in ("velocity", "acceleration", "step"):
                    if (reason in guarded.reasons) != bool(np.any(active[reason])):
                        raise AssertionError(
                            f"diagnostic mask disagrees with guard: {reason}"
                        )
                assert guard_diagnostics is not None
                guard_diagnostics.observe(active, intervention)
                applied_action = guarded.action
                mapper.set_ctrl(data.ctrl, applied_action[:8], "right")
                mapper.set_ctrl(data.ctrl, applied_action[8:], "left")
            else:
                applied_action = target
            mujoco.mj_step(model, data)

            state_after = get_state_vector(mapper, data.qpos)
            guarded_actions.append(applied_action.copy())
            actual_states.append(state_after.copy())
            target_at_physics.append(target.copy())
            physics_times.append(float(data.time - run_start_s))

            contacts = finger_contacts(model, data, addresses["cube"])
            inner = "openarm_right_ee_inner_finger" in contacts
            outer = "openarm_right_ee_outer_finger" in contacts
            task.left_contact_steps += int(inner)
            task.right_contact_steps += int(outer)
            bilateral = inner and outer
            if bilateral:
                task.dual_contact_steps += 1
                if task.first_dual_contact_s is None:
                    task.first_dual_contact_s = float(data.time - run_start_s)

            relative = data.xpos[addresses["cube"]] - data.site_xpos[tcp_site]
            if bilateral and reference_relative is None:
                reference_relative = relative.copy()
            drift = (
                float(np.linalg.norm(relative - reference_relative))
                if reference_relative is not None
                else float("inf")
            )
            cube_z = float(data.qpos[addresses["cube_qadr"] + 2])
            task.max_lift_m = max(task.max_lift_m, cube_z - reset_z)
            success_now = evaluator.update(
                now_s=float(data.time - run_start_s),
                cube_z=cube_z,
                bilateral_contact=bilateral,
                tcp_distance_m=float(np.linalg.norm(relative)),
                relative_drift_m=drift,
                cube_vertical_velocity_m_s=float(
                    data.qvel[addresses["cube_dadr"] + 2]
                ),
            )
            task.max_hold_s = max(task.max_hold_s, evaluator.hold_duration_s)
            if success_now and not task.success:
                task.success = True
                task.first_success_s = float(data.time - run_start_s)
        if control_mode == "guard":
            assert guard_diagnostics is not None
            guard_diagnostics.end_command()

    executed_duration = float(data.time - run_start_s)
    total_duration_s = trajectory.duration_s + terminal_hold_s
    expected_physics_steps = int(round(total_duration_s * PHYSICS_HZ))
    actual_physics_steps = len(physics_times)
    if actual_physics_steps != expected_physics_steps:
        raise AssertionError("physics-step count violated fixed-duration contract")
    if not math.isclose(executed_duration, total_duration_s, abs_tol=2e-9):
        raise AssertionError(
            f"executed duration drifted: {executed_duration} vs {total_duration_s}"
        )

    guarded_array = np.asarray(guarded_actions)
    state_array = np.asarray(actual_states)
    physics_target_array = np.asarray(target_at_physics)
    result = {
        "cadence_hz": trajectory.cadence_hz,
        "chunk_horizon_s": trajectory.chunk_horizon_s,
        "chunk_count": trajectory.chunk_count,
        "knot_count_including_chunk_endpoints": trajectory.knot_count,
        "use_lowpass": use_lowpass,
        "control_mode": control_mode,
        "task": summarize_task(task, executed_duration),
        "time_contract": {
            "source_duration_s": float(
                relative_source_time(expert_data["action_timestamps_ns"])[-1]
            ),
            "requested_duration_s": trajectory.duration_s,
            "terminal_hold_s": terminal_hold_s,
            "total_requested_duration_s": trajectory.duration_s + terminal_hold_s,
            "executor_control_hz": CONTROL_HZ,
            "executor_commands": len(targets),
            "physics_hz": PHYSICS_HZ,
            "physics_steps": actual_physics_steps,
            "first_control_time_s": float(trajectory.control_time_s[0]),
            "last_control_time_s": float(trajectory.control_time_s[-1]),
        },
        "pre_guard": {
            "source_to_hermite_error": error_summary(
                trajectory.source_reference, trajectory.hermite_targets
            ),
            "source_to_executor_target_error": error_summary(
                trajectory.source_reference,
                targets[: len(trajectory.source_reference)],
            ),
            "source_to_hermite_phase_lag": signed_phase_lag_ms(
                trajectory.source_reference,
                trajectory.hermite_targets,
                sample_hz=CONTROL_HZ,
            ),
            "dynamics": pre_guard_dynamics(
                targets,
                control_hz=CONTROL_HZ,
                max_velocity=guard.max_velocity,
                max_acceleration=guard.max_acceleration,
                max_step=guard.max_step,
            ),
        },
        "guard": (
            guard_diagnostics.as_dict() if guard_diagnostics is not None else None
        ),
        "tracking": {
            "executor_target_to_guard_action": error_summary(
                physics_target_array, guarded_array
            ),
            "executor_target_to_actual_state": error_summary(
                physics_target_array, state_array
            ),
            "guard_action_to_actual_state": error_summary(
                guarded_array, state_array
            ),
        },
        "initialization": {
            "stabilize_end_state": stabilize_end_state.tolist(),
            "first_source_action": expert_data["actions"][0].tolist(),
            "max_error_to_first_source_action_rad": float(
                np.max(np.abs(stabilize_end_state - expert_data["actions"][0]))
            ),
        },
    }
    trajectories = {
        "control_time_s": trajectory.control_time_s,
        "execution_control_time_s": (
            np.arange(len(targets), dtype=np.float64) / CONTROL_HZ
        ),
        "source_reference_250hz": trajectory.source_reference,
        "hermite_250hz": trajectory.hermite_targets,
        "executor_target_250hz": targets,
        "physics_time_s": np.asarray(physics_times),
        "executor_target_1000hz": physics_target_array,
        "guard_action_1000hz": guarded_array,
        "actual_state_1000hz": state_array,
    }
    return result, trajectories


def parse_csv_floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one number")
    return result


def parse_chunk_modes(value: str, duration_s: float) -> list[tuple[str, float]]:
    modes: list[tuple[str, float]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item == "full":
            modes.append(("full", duration_s))
        elif item == "act":
            modes.append(("act", ACT_HORIZON_S))
        elif item:
            horizon = float(item)
            modes.append((f"horizon_{horizon:g}s", horizon))
    if not modes:
        raise argparse.ArgumentTypeError("expected full, act, or numeric horizons")
    return modes


def parse_control_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(modes) - {"guard", "direct"})
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid control modes: {invalid}")
    if not modes:
        raise argparse.ArgumentTypeError("expected guard and/or direct")
    return modes


def _summary_table(runs: Iterable[dict]) -> list[dict]:
    table = []
    for run in runs:
        guard = run["guard"]
        row = {
                "condition": run["condition"],
                "cadence_hz": run["cadence_hz"],
                "chunk_mode": run["chunk_mode"],
                "control_mode": run["control_mode"],
                "success": run["task"]["success"],
                "max_lift_m": run["task"]["max_lift_m"],
                "max_hold_s": run["task"]["max_hold_s"],
                "dual_contact_steps": run["task"]["dual_contact_physics_steps"],
                "pre_guard_acceleration_axis_ratio": run["pre_guard"]["dynamics"]["acceleration"]["axis_event_ratio"],
                "target_state_error_p95_rad": run["tracking"]["executor_target_to_actual_state"]["p95_rad"],
            }
        if guard is not None:
            row.update(
                {
                    "guard_velocity_step_ratio": guard["velocity"]["active_physics_step_ratio"],
                    "guard_velocity_axis_ratio": guard["velocity"]["active_axis_event_ratio"],
                    "guard_acceleration_step_ratio": guard["acceleration"]["active_physics_step_ratio"],
                    "guard_acceleration_axis_ratio": guard["acceleration"]["active_axis_event_ratio"],
                }
            )
        else:
            row.update(
                {
                    "guard_velocity_step_ratio": None,
                    "guard_velocity_axis_ratio": None,
                    "guard_acceleration_step_ratio": None,
                    "guard_acceleration_axis_ratio": None,
                }
            )
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="G5.5.2 command-cadence sweep")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--source-actions",
        choices=("raw", "lerobot"),
        default="raw",
        help="raw tests recorded safe ctrl targets; lerobot tests actual ACT labels",
    )
    parser.add_argument(
        "--lerobot-dataset", type=Path, default=DEFAULT_LEROBOT_DATASET
    )
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument(
        "--cadences",
        type=parse_csv_floats,
        default=list(DEFAULT_CADENCES),
        help="comma-separated command knot rates",
    )
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--terminal-hold-s",
        type=float,
        default=0.0,
        help="hold the final raw target after the fixed trajectory duration",
    )
    parser.add_argument(
        "--chunk-modes",
        default="full,act",
        help="comma-separated: full, act, or numeric horizon seconds",
    )
    parser.add_argument("--use-lowpass", action="store_true")
    parser.add_argument(
        "--control-modes",
        type=parse_control_modes,
        default=["guard"],
        help="comma-separated: guard,direct; direct matches official MuJoCo ctrl path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/g5_5_2_command_cadence_sweep",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.params.read_text())
    expert_data = load_expert_trajectory(args.dataset, args.seed)
    if expert_data is None:
        raise SystemExit(f"no expert episode for seed {args.seed}")
    if args.source_actions == "lerobot":
        source_time, source_actions = load_lerobot_action_source(
            args.lerobot_dataset, int(expert_data["entry"]["id"])
        )
        expert_data = dict(expert_data)
        expert_data["actions"] = source_actions
        expert_data["action_timestamps_ns"] = np.rint(
            source_time * 1e9
        ).astype(np.int64)
    else:
        source_time = relative_source_time(expert_data["action_timestamps_ns"])
        source_actions = expert_data["actions"]
    if args.duration_s > source_time[-1] + 1e-9:
        raise SystemExit(
            f"duration {args.duration_s}s exceeds source duration {source_time[-1]}s"
        )
    modes = parse_chunk_modes(args.chunk_modes, args.duration_s)

    seed_dir = args.output_dir / f"seed_{args.seed}_{args.source_actions}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for mode_name, horizon_s in modes:
        for cadence_hz in args.cadences:
            for control_mode in args.control_modes:
                condition = (
                    f"{mode_name}_{cadence_hz:g}hz_{control_mode}"
                    f"{'_lowpass' if args.use_lowpass else ''}"
                )
                print(f"\n{'=' * 80}\n{condition}\n{'=' * 80}", flush=True)
                trajectory = build_executor_trajectory(
                    source_time,
                    source_actions,
                    cadence_hz=cadence_hz,
                    duration_s=args.duration_s,
                    chunk_horizon_s=horizon_s,
                )
                result, trajectories = run_one(
                    scene=args.scene,
                    cfg=cfg,
                    expert_data=expert_data,
                    trajectory=trajectory,
                    use_lowpass=args.use_lowpass,
                    control_mode=control_mode,
                    terminal_hold_s=args.terminal_hold_s,
                )
                result["condition"] = condition
                result["chunk_mode"] = mode_name
                result_path = seed_dir / f"{condition}.json"
                trajectory_path = seed_dir / f"{condition}.npz"
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                np.savez_compressed(trajectory_path, **trajectories)
                runs.append(result)
                task = result["task"]
                print(
                    f"{condition}: success={task['success']} "
                    f"lift={task['max_lift_m']:.5f}m "
                    f"hold={task['max_hold_s']:.3f}s "
                    f"dual={task['dual_contact_physics_steps']}",
                    flush=True,
                )

    source_hash = _array_sha256(
        expert_data["action_timestamps_ns"], expert_data["actions"]
    )
    summary = {
        "experiment": "G5.5.2 controlled command-cadence sweep",
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "source_actions": args.source_actions,
        "lerobot_dataset": (
            str(args.lerobot_dataset.resolve())
            if args.source_actions == "lerobot"
            else None
        ),
        "episode_id": expert_data["episode_id"],
        "source_action_sha256": source_hash,
        "source_sample_count": len(expert_data["actions"]),
        "source_mean_hz": float(1.0 / np.mean(np.diff(source_time))),
        "source_duration_s": float(source_time[-1]),
        "evaluated_duration_s": args.duration_s,
        "terminal_hold_s": args.terminal_hold_s,
        "cadences_hz": args.cadences,
        "chunk_modes": [name for name, _ in modes],
        "use_lowpass": args.use_lowpass,
        "control_modes": args.control_modes,
        "physics_hz": PHYSICS_HZ,
        "control_hz": CONTROL_HZ,
        "scene_sha256": sha256(args.scene),
        "params_sha256": sha256(args.params),
        "official_executor_source": str(OFFICIAL_EXECUTOR.relative_to(ROOT)),
        "official_executor_sha256": sha256(OFFICIAL_EXECUTOR),
        "runs": _summary_table(runs),
    }
    summary_path = seed_dir / (
        "summary_lowpass.json" if args.use_lowpass else "summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved summary: {summary_path}")
    for row in summary["runs"]:
        guard_acc = row["guard_acceleration_axis_ratio"]
        guard_text = "n/a" if guard_acc is None else f"{guard_acc:.2%}"
        print(
            f"{row['condition']:20s} success={str(row['success']):5s} "
            f"lift={row['max_lift_m']:.5f} hold={row['max_hold_s']:.3f} "
            f"guard_acc_axis={guard_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
