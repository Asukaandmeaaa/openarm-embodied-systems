#!/usr/bin/env python3
"""G5.5 Executor Semantic Equivalence Gate.

Four controlled experiments to isolate ACT failure root cause:

E0: Expert strict replay (ground truth)
E1: Expert → 30Hz → Hermite → Guard
E2: Expert → 30Hz → Hermite → Lowpass → Guard (KEY COUNTERFACTUAL)
E3: ACT → 30Hz → Hermite → Lowpass → Guard

All experiments use identical scene, seed, initial state, and cube pose.
Goal: Determine if executor preserves expert task semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
from typing import Literal

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
import torch
import types
import yaml
from scipy import signal

from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.config import sha256
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.guard import ActionGuard
from openarm_cube_pick_expert.task import PickCubeEvaluator


SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
PARAMS = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
DEFAULT_DATASET = ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
EXECUTOR_IMPL = EXECUTOR_SRC / "dora_openarm_actions_executor/main.py"

# ACT deployment parameters
ACT_CHUNK_HZ = 30.0  # 1e9 / 33_333_333
EXECUTOR_CONTROL_HZ = 250.0
LOWPASS_CUTOFF_HZ = 15.0
LOWPASS_Q = 0.707


def load_official_executor_classes():
    """Import HermiteUpsampler and BiquadLowpass from official executor."""
    sys.modules.setdefault("dora", types.ModuleType("dora"))
    from dora_openarm_actions_executor.main import BiquadLowpass, HermiteUpsampler
    return HermiteUpsampler, BiquadLowpass


def load_expert_trajectory(dataset: Path, seed: int) -> dict | None:
    """Load expert episode metadata and trajectories for the given seed."""
    metadata_path = dataset / "metadata.yaml"
    if not metadata_path.exists():
        return None

    metadata = yaml.safe_load(metadata_path.read_text())
    episode_entry = None
    for entry in metadata.get("episodes", []):
        if int(entry.get("seed", -1)) == seed:
            episode_entry = entry
            break

    if episode_entry is None:
        return None

    episode_id = str(episode_entry["id"])
    episode_dir = dataset / "episodes" / episode_id

    # Load action stream
    action_timestamps = None
    action_parts = []
    for side in ("right", "left"):
        path = episode_dir / "action" / "arms" / side / "state.parquet"
        if not path.exists():
            return None
        table = pq.read_table(path, columns=["timestamp", "qpos"])
        side_timestamps = table["timestamp"].cast(pa.int64()).to_numpy()
        values = np.asarray(table["qpos"].to_pylist(), dtype=np.float32)
        if action_timestamps is None:
            action_timestamps = side_timestamps
        action_parts.append(values)

    actions = np.concatenate(action_parts, axis=1)

    return {
        "entry": episode_entry,
        "episode_id": episode_id,
        "action_timestamps_ns": action_timestamps,
        "actions": actions,
    }


@dataclass
class PhysicsState:
    """Complete physics state snapshot."""
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    cube_pose: np.ndarray  # [x, y, z, qw, qx, qy, qz]


@dataclass
class TaskMetrics:
    """Task-level outcome metrics."""
    success: bool
    failure_reason: str | None

    # Contact metrics
    left_finger_contact_steps: int = 0
    right_finger_contact_steps: int = 0
    dual_finger_contact_steps: int = 0
    first_left_contact_s: float | None = None
    first_right_contact_s: float | None = None
    first_dual_contact_s: float | None = None

    # Lift metrics
    initial_cube_height_m: float = 0.0
    max_cube_height_m: float = 0.0
    max_cube_lift_m: float = 0.0

    # Hold metrics
    hold_duration_s: float = 0.0

    # Episode metrics
    physics_steps: int = 0
    episode_duration_s: float = 0.0


@dataclass
class StageMetrics:
    """Metrics for one processing stage."""
    name: str

    # Deviation metrics
    mean_deviation: float = 0.0
    max_deviation: float = 0.0
    p95_deviation: float = 0.0
    per_axis_max_deviation: list[float] = field(default_factory=list)

    # Phase lag metrics (milliseconds)
    phase_lag_ms: float | None = None
    per_axis_lag_ms: list[float | None] = field(default_factory=list)

    # Guard clamp metrics
    velocity_clamps: int = 0
    acceleration_clamps: int = 0
    step_clamps: int = 0
    ctrlrange_clamps: int = 0
    watchdog_holds: int = 0

    # Clamp ratios
    velocity_clamp_ratio: float = 0.0
    acceleration_clamp_ratio: float = 0.0
    step_clamp_ratio: float = 0.0

    # Per-axis clamp counts
    per_axis_velocity_clamps: list[int] = field(default_factory=list)
    per_axis_acceleration_clamps: list[int] = field(default_factory=list)
    per_axis_step_clamps: list[int] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Complete result for one experiment."""
    experiment: Literal["E0", "E1", "E2", "E3"]
    description: str

    task: TaskMetrics
    stages: dict[str, StageMetrics] = field(default_factory=dict)

    # Trajectory recordings
    trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    timestamps: dict[str, np.ndarray] = field(default_factory=dict)

    # Final state
    final_state: PhysicsState | None = None


def compute_phase_lag(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    dt: float,
    min_energy_threshold: float = 0.01
) -> float | None:
    """Compute phase lag in milliseconds between two signals using cross-correlation.

    Returns None if signal energy is too low (insufficient excitation).
    """
    # Check signal energy
    energy_a = np.var(signal_a)
    energy_b = np.var(signal_b)

    if energy_a < min_energy_threshold or energy_b < min_energy_threshold:
        return None

    # Compute cross-correlation
    correlation = signal.correlate(signal_a, signal_b, mode='same')
    lags = signal.correlation_lags(len(signal_a), len(signal_b), mode='same')

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    lag_samples = lags[peak_idx]
    lag_ms = lag_samples * dt * 1000.0

    return float(lag_ms)


def compute_deviation_metrics(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    dt: float
) -> dict:
    """Compute deviation metrics between two trajectories."""
    # Align to common length
    min_len = min(len(trajectory_a), len(trajectory_b))
    trajectory_a = trajectory_a[:min_len]
    trajectory_b = trajectory_b[:min_len]

    diff = np.abs(trajectory_a - trajectory_b)

    # Overall metrics
    metrics = {
        "mean": float(diff.mean()),
        "max": float(diff.max()),
        "p95": float(np.quantile(diff, 0.95)),
        "per_axis_max": diff.max(axis=0).tolist(),
    }

    # Phase lag per axis
    per_axis_lag = []
    for axis_idx in range(trajectory_a.shape[1]):
        lag = compute_phase_lag(
            trajectory_a[:, axis_idx],
            trajectory_b[:, axis_idx],
            dt
        )
        per_axis_lag.append(lag)

    metrics["per_axis_lag_ms"] = per_axis_lag

    # Representative lag (median of valid lags)
    valid_lags = [lag for lag in per_axis_lag if lag is not None]
    if valid_lags:
        metrics["representative_lag_ms"] = float(np.median(valid_lags))
    else:
        metrics["representative_lag_ms"] = None

    return metrics


def finger_contacts(model, data, cube_body: int) -> set[str]:
    """Get set of finger bodies in contact with cube."""
    names: set[str] = set()
    expected = {
        "openarm_right_ee_inner_finger",
        "openarm_right_ee_outer_finger",
    }
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        if cube_body not in bodies or len(bodies) != 2:
            continue
        other = (bodies - {cube_body}).pop()
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if name in expected:
            names.add(name)
    return names


def initialize_physics(
    scene: Path,
    seed: int,
    expert_entry: dict | None = None
) -> tuple[mujoco.MjModel, mujoco.MjData, JointResolver, dict]:
    """Initialize MuJoCo physics with deterministic initial conditions."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError(f"expected 1000 Hz physics, got dt={model.opt.timestep}")

    data = mujoco.MjData(model)
    mapper = JointResolver(model)

    # Reset to home keyframe
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)

    # Get cube body and joint addresses
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
    cube_joint = int(model.body_jntadr[cube])
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    cube_dadr = int(model.jnt_dofadr[cube_joint])

    # Set cube pose (use expert metadata if available, otherwise deterministic seed-based)
    if expert_entry and "objects" in expert_entry:
        cube_pose = np.asarray(expert_entry["objects"]["orange_cube"]["pose"], dtype=np.float64)
        data.qpos[cube_qadr:cube_qadr + 7] = cube_pose
    else:
        rng = np.random.default_rng(seed)
        data.qpos[cube_qadr:cube_qadr + 3] = [
            0.33 + rng.uniform(-0.02, 0.02),
            -0.28 + rng.uniform(-0.02, 0.02),
            1.05,
        ]
        yaw = rng.uniform(-math.radians(15), math.radians(15))
        data.qpos[cube_qadr + 3:cube_qadr + 7] = [
            math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
        ]

    data.qvel.fill(0)
    mujoco.mj_forward(model, data)

    addresses = {
        "cube": cube,
        "cube_qadr": cube_qadr,
        "cube_dadr": cube_dadr,
    }

    return model, data, mapper, addresses


def get_state_vector(mapper: JointResolver, qpos: np.ndarray) -> np.ndarray:
    """Extract 16D state vector (right8 + left8) from qpos."""
    right, right_gripper = mapper.get_driver(qpos, "right")
    left, left_gripper = mapper.get_driver(qpos, "left")
    return combine_action(
        np.r_[right, right_gripper], np.r_[left, left_gripper]
    )


def convert_expert_to_30hz(
    timestamps_ns: np.ndarray,
    actions: np.ndarray,
    target_hz: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert expert trajectory to ACT-compatible 30Hz representation.

    Uses timestamp-based interpolation, not naive frame sampling.
    Returns (new_timestamps_ns, new_actions).
    """
    start_ns = timestamps_ns[0]
    end_ns = timestamps_ns[-1]
    duration_s = (end_ns - start_ns) / 1e9

    # Generate uniform 30Hz timestamps
    interval_ns = int(1e9 / target_hz)
    new_timestamps_ns = np.arange(start_ns, end_ns + interval_ns, interval_ns, dtype=np.int64)

    # Interpolate actions at new timestamps
    new_actions = np.zeros((len(new_timestamps_ns), actions.shape[1]), dtype=np.float32)
    for axis_idx in range(actions.shape[1]):
        new_actions[:, axis_idx] = np.interp(
            new_timestamps_ns,
            timestamps_ns,
            actions[:, axis_idx]
        )

    return new_timestamps_ns, new_actions


def run_e0_expert_strict(
    scene: Path,
    dataset: Path,
    seed: int,
    cfg: dict,
    max_seconds: float = 45.0
) -> ExperimentResult:
    """E0: Expert strict replay (ground truth baseline)."""
    print(f"\n{'='*80}")
    print("E0: Expert Strict Replay (Ground Truth)")
    print(f"{'='*80}\n")

    # Load expert trajectory
    expert_data = load_expert_trajectory(dataset, seed)
    if expert_data is None:
        raise ValueError(f"Cannot load expert trajectory for seed {seed}")

    entry = expert_data["entry"]
    episode_id = expert_data["episode_id"]
    episode_dir = dataset / "episodes" / episode_id

    # Load observation stream for tracking error validation
    obs_timestamps = None
    obs_parts = []
    for side in ("right", "left"):
        path = episode_dir / "obs" / "arms" / side / "state.parquet"
        table = pq.read_table(path, columns=["timestamp", "qpos"])
        side_timestamps = table["timestamp"].cast(pa.int64()).to_numpy()
        values = np.asarray(table["qpos"].to_pylist(), dtype=np.float32)
        if obs_timestamps is None:
            obs_timestamps = side_timestamps
        obs_parts.append(values)

    observations = np.concatenate(obs_parts, axis=1)
    action_timestamps = expert_data["action_timestamps_ns"]
    actions = expert_data["actions"]

    if not np.array_equal(action_timestamps, obs_timestamps):
        raise ValueError("action and observation timestamps differ")

    # Initialize physics
    model, data, mapper, addresses = initialize_physics(scene, seed, entry)

    # Set initial robot state from expert metadata
    if "initial_qpos" in entry:
        initial_qpos = np.asarray(entry["initial_qpos"], dtype=np.float64)
        mapper.set_qpos(data.qpos, initial_qpos[:8], "right")
        mapper.set_qpos(data.qpos, initial_qpos[8:], "left")

    data.qvel.fill(0)
    phase_start_s = float(entry["phase_timestamps"]["MOVE_PREGRASP"])
    data.time = phase_start_s
    mujoco.mj_forward(model, data)

    # Initialize task evaluator
    cube_qadr = addresses["cube_qadr"]
    cube_dadr = addresses["cube_dadr"]
    reset_z = float(data.qpos[cube_qadr + 2])
    evaluator = PickCubeEvaluator(
        reset_z,
        cfg["hold_height_m"],
        cfg["hold_s"],
        cfg["tcp_distance_m"],
        cfg.get("max_relative_drift_m", 0.025),
    )

    # Task metrics tracking
    task = TaskMetrics(success=False, failure_reason=None)
    task.initial_cube_height_m = reset_z

    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
    reference_relative = None

    tracking_errors = []
    previous_timestamp_ns = int(round(phase_start_s * 1e9))

    # Replay loop
    for timestamp_ns, action, recorded_obs in zip(action_timestamps, actions, observations, strict=True):
        timestamp_ns = int(timestamp_ns)
        interval_ns = timestamp_ns - previous_timestamp_ns
        requested_steps = interval_ns / (model.opt.timestep * 1e9)
        steps = int(round(requested_steps))

        if steps < 1:
            continue

        # Apply recorded safe target directly to ctrl (no ActionGuard re-application)
        mapper.set_ctrl(data.ctrl, action[:8], "right")
        mapper.set_ctrl(data.ctrl, action[8:], "left")

        for _ in range(steps):
            mujoco.mj_step(model, data)
            task.physics_steps += 1

            # Update task metrics
            cube = addresses["cube"]
            contacts = finger_contacts(model, data, cube)
            relative = data.xpos[cube] - data.site_xpos[tcp_site]

            if "openarm_right_ee_inner_finger" in contacts:
                task.left_finger_contact_steps += 1
                if task.first_left_contact_s is None:
                    task.first_left_contact_s = float(data.time)

            if "openarm_right_ee_outer_finger" in contacts:
                task.right_finger_contact_steps += 1
                if task.first_right_contact_s is None:
                    task.first_right_contact_s = float(data.time)

            bilateral = len(contacts) == 2
            if bilateral:
                task.dual_finger_contact_steps += 1
                if task.first_dual_contact_s is None:
                    task.first_dual_contact_s = float(data.time)
                if reference_relative is None:
                    reference_relative = relative.copy()

            drift = (
                float(np.linalg.norm(relative - reference_relative))
                if reference_relative is not None
                else float("inf")
            )

            cube_z = float(data.qpos[cube_qadr + 2])
            task.max_cube_height_m = max(task.max_cube_height_m, cube_z)
            task.max_cube_lift_m = max(task.max_cube_lift_m, cube_z - reset_z)

            task.success = evaluator.update(
                now_s=float(data.time),
                cube_z=cube_z,
                bilateral_contact=bilateral,
                tcp_distance_m=float(np.linalg.norm(relative)),
                relative_drift_m=drift,
                cube_vertical_velocity_m_s=float(data.qvel[cube_dadr + 2]),
            ) or task.success

        # Track replay accuracy
        current_state = get_state_vector(mapper, data.qpos)
        tracking_errors.append(float(np.max(np.abs(current_state - recorded_obs))))
        previous_timestamp_ns = timestamp_ns

    task.episode_duration_s = float(data.time - phase_start_s)

    # Get evaluator evidence
    evidence = evaluator.evidence()
    task.hold_duration_s = evidence.get("hold_duration_s", 0.0)

    if not task.success:
        task.failure_reason = "lift_hold_condition_not_met"

    # Tracking error metrics
    errors = np.asarray(tracking_errors)
    tracking_metric = StageMetrics(
        name="strict_replay_tracking",
        mean_deviation=float(errors.mean()),
        max_deviation=float(errors.max()),
        p95_deviation=float(np.quantile(errors, 0.95)),
    )

    result = ExperimentResult(
        experiment="E0",
        description="Expert strict replay (ground truth)",
        task=task,
        stages={"tracking": tracking_metric},
        final_state=PhysicsState(
            time=float(data.time),
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(),
            cube_pose=data.qpos[cube_qadr:cube_qadr + 7].copy(),
        ),
    )

    print(f"E0 Result: {'SUCCESS' if task.success else 'FAIL'}")
    print(f"  Max lift: {task.max_cube_lift_m:.4f} m")
    print(f"  Hold duration: {task.hold_duration_s:.2f} s")
    print(f"  Dual contact steps: {task.dual_finger_contact_steps}")
    print(f"  Tracking error (p95): {tracking_metric.p95_deviation:.6f} rad")

    return result


def stabilize_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: JointResolver,
    guard: ActionGuard,
    cfg: dict,
) -> None:
    """Run stabilization phase matching expert collection."""
    stabilize_target = get_state_vector(mapper, data.qpos)
    stabilize_target[7] = cfg["pregrasp_right_gripper_rad"]

    stabilize_steps = int(round(cfg["stabilize_s"] / model.opt.timestep))
    for _ in range(stabilize_steps):
        guard.submit(stabilize_target, float(data.time))
        guarded = guard.step(get_state_vector(mapper, data.qpos), float(data.time))
        mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
        mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
        mujoco.mj_step(model, data)


# Marker for continuation
def run_e1_expert_hermite(
    scene: Path,
    dataset: Path,
    seed: int,
    cfg: dict,
    max_seconds: float = 45.0
) -> ExperimentResult:
    """E1: Expert → 30Hz → HermiteUpsampler → ActionGuard."""
    print(f"\n{'='*80}")
    print("E1: Expert + Hermite Upsampler Only")
    print(f"{'='*80}\n")

    # Load expert trajectory
    expert_data = load_expert_trajectory(dataset, seed)
    if expert_data is None:
        raise ValueError(f"Cannot load expert trajectory for seed {seed}")

    entry = expert_data["entry"]

    # Convert to 30Hz ACT-compatible representation
    expert_30hz_ts, expert_30hz_actions = convert_expert_to_30hz(
        expert_data["action_timestamps_ns"],
        expert_data["actions"],
        target_hz=ACT_CHUNK_HZ
    )

    print(f"Expert trajectory converted: {len(expert_data['actions'])} frames → {len(expert_30hz_actions)} @ 30Hz")

    # Initialize physics
    model, data, mapper, addresses = initialize_physics(scene, seed, entry)

    if "initial_qpos" in entry:
        initial_qpos = np.asarray(entry["initial_qpos"], dtype=np.float64)
        mapper.set_qpos(data.qpos, initial_qpos[:8], "right")
        mapper.set_qpos(data.qpos, initial_qpos[8:], "left")

    data.qvel.fill(0)
    mujoco.mj_forward(model, data)

    # Setup ActionGuard
    lower = np.r_[
        model.actuator_ctrlrange[9:17, 0], model.actuator_ctrlrange[1:9, 0]
    ]
    upper = np.r_[
        model.actuator_ctrlrange[9:17, 1], model.actuator_ctrlrange[1:9, 1]
    ]
    guard_cfg = cfg["guard"]
    guard = ActionGuard(
        lower, upper,
        np.asarray(guard_cfg["max_velocity"], dtype=np.float32),
        np.asarray(guard_cfg["max_acceleration"], dtype=np.float32),
        np.asarray(guard_cfg["max_step"], dtype=np.float32),
        guard_cfg["watchdog_timeout_s"],
        guard_cfg["limit_policy"],
    )

    # Stabilization
    stabilize_robot(model, data, mapper, guard, cfg)
    guard_stats_baseline = Counter(guard.stats)

    # Setup upsampler
    HermiteUpsampler, _ = load_official_executor_classes()
    chunk_hz = ACT_CHUNK_HZ
    horizon_s = (len(expert_30hz_actions) - 1) / chunk_hz if len(expert_30hz_actions) > 1 else 1.0
    upsampler = HermiteUpsampler(chunk_hz=chunk_hz, horizon_sec=horizon_s)
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    executor_times = np.arange(0.0, horizon_s + 1e-9, executor_dt)

    # Task tracking
    cube_qadr = addresses["cube_qadr"]
    cube_dadr = addresses["cube_dadr"]
    reset_z = float(data.qpos[cube_qadr + 2])
    evaluator = PickCubeEvaluator(reset_z, cfg["hold_height_m"], cfg["hold_s"],
                                   cfg["tcp_distance_m"], cfg.get("max_relative_drift_m", 0.025))

    task = TaskMetrics(success=False, failure_reason=None, initial_cube_height_m=reset_z)
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
    reference_relative = None

    # Trajectory recording
    raw_30hz = []
    upsampled_250hz = []
    guarded_250hz = []

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + max_seconds

    # Process 30Hz chunks through upsampler
    chunk_size = int(horizon_s * chunk_hz) + 1
    for chunk_start_idx in range(0, len(expert_30hz_actions), chunk_size):
        if data.time >= deadline_s:
            break

        chunk = expert_30hz_actions[chunk_start_idx:chunk_start_idx + chunk_size]
        if len(chunk) < 2:
            break

        raw_30hz.append(chunk.copy())

        # Upsample to 250Hz
        current_horizon_s = (len(chunk) - 1) / chunk_hz
        current_times = np.arange(0.0, current_horizon_s + 1e-9, executor_dt)
        upsampled_chunk = upsampler.upsample(chunk, current_times)

        for target_250hz in upsampled_chunk:
            upsampled_250hz.append(target_250hz.copy())
            guard.submit(target_250hz, float(data.time))

            # Execute at 1000Hz physics rate
            for _ in range(int(round(executor_dt / model.opt.timestep))):
                guarded = guard.step(get_state_vector(mapper, data.qpos), float(data.time))
                guarded_250hz.append(guarded.action.copy())

                mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
                mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
                mujoco.mj_step(model, data)
                task.physics_steps += 1

                # Task evaluation
                cube = addresses["cube"]
                contacts = finger_contacts(model, data, cube)
                relative = data.xpos[cube] - data.site_xpos[tcp_site]

                if "openarm_right_ee_inner_finger" in contacts:
                    task.left_finger_contact_steps += 1
                    if task.first_left_contact_s is None:
                        task.first_left_contact_s = float(data.time)

                if "openarm_right_ee_outer_finger" in contacts:
                    task.right_finger_contact_steps += 1
                    if task.first_right_contact_s is None:
                        task.first_right_contact_s = float(data.time)

                bilateral = len(contacts) == 2
                if bilateral:
                    task.dual_finger_contact_steps += 1
                    if task.first_dual_contact_s is None:
                        task.first_dual_contact_s = float(data.time)
                    if reference_relative is None:
                        reference_relative = relative.copy()

                drift = (
                    float(np.linalg.norm(relative - reference_relative))
                    if reference_relative is not None else float("inf")
                )

                cube_z = float(data.qpos[cube_qadr + 2])
                task.max_cube_height_m = max(task.max_cube_height_m, cube_z)
                task.max_cube_lift_m = max(task.max_cube_lift_m, cube_z - reset_z)

                task.success = evaluator.update(
                    now_s=float(data.time), cube_z=cube_z, bilateral_contact=bilateral,
                    tcp_distance_m=float(np.linalg.norm(relative)), relative_drift_m=drift,
                    cube_vertical_velocity_m_s=float(data.qvel[cube_dadr + 2]),
                ) or task.success

                if data.time >= deadline_s or task.success:
                    break

            if data.time >= deadline_s or task.success:
                break

    task.episode_duration_s = float(data.time - rollout_start_s)
    evidence = evaluator.evidence()
    task.hold_duration_s = evidence.get("hold_duration_s", 0.0)
    if not task.success:
        task.failure_reason = "lift_hold_condition_not_met"

    # Guard statistics
    guard_delta = {k: guard.stats.get(k, 0) - guard_stats_baseline.get(k, 0)
                   for k in set(guard.stats) | set(guard_stats_baseline)}

    guard_metric = StageMetrics(name="hermite_guard")
    guard_metric.velocity_clamps = guard_delta.get("clip_velocity", 0)
    guard_metric.acceleration_clamps = guard_delta.get("clip_acceleration", 0)
    guard_metric.step_clamps = guard_delta.get("clip_step", 0)
    guard_metric.ctrlrange_clamps = guard_delta.get("clip_ctrlrange", 0)
    guard_metric.watchdog_holds = guard_delta.get("hold_watchdog", 0)

    total_control_steps = len(upsampled_250hz)
    if total_control_steps > 0:
        guard_metric.velocity_clamp_ratio = guard_metric.velocity_clamps / total_control_steps
        guard_metric.acceleration_clamp_ratio = guard_metric.acceleration_clamps / total_control_steps
        guard_metric.step_clamp_ratio = guard_metric.step_clamps / total_control_steps

    result = ExperimentResult(
        experiment="E1",
        description="Expert → 30Hz → Hermite → Guard",
        task=task,
        stages={"guard": guard_metric},
        trajectories={
            "raw_30hz": np.concatenate(raw_30hz) if raw_30hz else np.array([]),
            "upsampled_250hz": np.array(upsampled_250hz),
            "guarded_250hz": np.array(guarded_250hz),
        },
        final_state=PhysicsState(
            time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(), cube_pose=data.qpos[cube_qadr:cube_qadr + 7].copy(),
        ),
    )

    print(f"E1 Result: {'SUCCESS' if task.success else 'FAIL'}")
    print(f"  Max lift: {task.max_cube_lift_m:.4f} m")
    print(f"  Hold duration: {task.hold_duration_s:.2f} s")
    print(f"  Dual contact steps: {task.dual_finger_contact_steps}")
    print(f"  Guard velocity clamps: {guard_metric.velocity_clamps} ({guard_metric.velocity_clamp_ratio:.2%})")
    print(f"  Guard acceleration clamps: {guard_metric.acceleration_clamps} ({guard_metric.acceleration_clamp_ratio:.2%})")

    return result


def run_e2_expert_full_executor(
    scene: Path,
    dataset: Path,
    seed: int,
    cfg: dict,
    max_seconds: float = 45.0
) -> ExperimentResult:
    """E2: Expert → 30Hz → Hermite → Lowpass → Guard (KEY COUNTERFACTUAL)."""
    print(f"\n{'='*80}")
    print("E2: Expert + Full ACT Deployment Executor (KEY COUNTERFACTUAL)")
    print(f"{'='*80}\n")

    # Load expert trajectory
    expert_data = load_expert_trajectory(dataset, seed)
    if expert_data is None:
        raise ValueError(f"Cannot load expert trajectory for seed {seed}")

    entry = expert_data["entry"]

    # Convert to 30Hz
    expert_30hz_ts, expert_30hz_actions = convert_expert_to_30hz(
        expert_data["action_timestamps_ns"],
        expert_data["actions"],
        target_hz=ACT_CHUNK_HZ
    )

    print(f"Expert trajectory converted: {len(expert_data['actions'])} frames → {len(expert_30hz_actions)} @ 30Hz")

    # Initialize physics
    model, data, mapper, addresses = initialize_physics(scene, seed, entry)

    if "initial_qpos" in entry:
        initial_qpos = np.asarray(entry["initial_qpos"], dtype=np.float64)
        mapper.set_qpos(data.qpos, initial_qpos[:8], "right")
        mapper.set_qpos(data.qpos, initial_qpos[8:], "left")

    data.qvel.fill(0)
    mujoco.mj_forward(model, data)

    # Setup ActionGuard
    lower = np.r_[
        model.actuator_ctrlrange[9:17, 0], model.actuator_ctrlrange[1:9, 0]
    ]
    upper = np.r_[
        model.actuator_ctrlrange[9:17, 1], model.actuator_ctrlrange[1:9, 1]
    ]
    guard_cfg = cfg["guard"]
    guard = ActionGuard(
        lower, upper,
        np.asarray(guard_cfg["max_velocity"], dtype=np.float32),
        np.asarray(guard_cfg["max_acceleration"], dtype=np.float32),
        np.asarray(guard_cfg["max_step"], dtype=np.float32),
        guard_cfg["watchdog_timeout_s"],
        guard_cfg["limit_policy"],
    )

    # Stabilization
    stabilize_robot(model, data, mapper, guard, cfg)
    guard_stats_baseline = Counter(guard.stats)

    # Setup upsampler and lowpass
    HermiteUpsampler, BiquadLowpass = load_official_executor_classes()
    chunk_hz = ACT_CHUNK_HZ
    horizon_s = (len(expert_30hz_actions) - 1) / chunk_hz if len(expert_30hz_actions) > 1 else 1.0
    upsampler = HermiteUpsampler(chunk_hz=chunk_hz, horizon_sec=horizon_s)
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    executor_times = np.arange(0.0, horizon_s + 1e-9, executor_dt)

    lowpass = BiquadLowpass(fs=EXECUTOR_CONTROL_HZ, fc=LOWPASS_CUTOFF_HZ, Q=LOWPASS_Q)

    # Task tracking
    cube_qadr = addresses["cube_qadr"]
    cube_dadr = addresses["cube_dadr"]
    reset_z = float(data.qpos[cube_qadr + 2])
    evaluator = PickCubeEvaluator(reset_z, cfg["hold_height_m"], cfg["hold_s"],
                                   cfg["tcp_distance_m"], cfg.get("max_relative_drift_m", 0.025))

    task = TaskMetrics(success=False, failure_reason=None, initial_cube_height_m=reset_z)
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
    reference_relative = None

    # Trajectory recording
    raw_30hz = []
    upsampled_250hz = []
    lowpassed_250hz = []
    guarded_250hz = []

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + max_seconds

    # Process chunks
    chunk_size = int(horizon_s * chunk_hz) + 1
    chunk_idx = 0
    for chunk_start_idx in range(0, len(expert_30hz_actions), chunk_size):
        if data.time >= deadline_s:
            break

        chunk = expert_30hz_actions[chunk_start_idx:chunk_start_idx + chunk_size]
        if len(chunk) < 2:
            break

        raw_30hz.append(chunk.copy())

        # Upsample to 250Hz
        current_horizon_s = (len(chunk) - 1) / chunk_hz
        current_times = np.arange(0.0, current_horizon_s + 1e-9, executor_dt)
        upsampled_chunk = upsampler.upsample(chunk, current_times)

        # Initialize lowpass on first chunk
        if chunk_idx == 0:
            lowpass.reset_state(upsampled_chunk[0])
        chunk_idx += 1

        for target_250hz in upsampled_chunk:
            upsampled_250hz.append(target_250hz.copy())

            # Apply lowpass filter
            filtered_250hz = lowpass.step(target_250hz)
            lowpassed_250hz.append(filtered_250hz.copy())

            guard.submit(filtered_250hz, float(data.time))

            # Execute at 1000Hz physics rate
            for _ in range(int(round(executor_dt / model.opt.timestep))):
                guarded = guard.step(get_state_vector(mapper, data.qpos), float(data.time))
                guarded_250hz.append(guarded.action.copy())

                mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
                mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
                mujoco.mj_step(model, data)
                task.physics_steps += 1

                # Task evaluation
                cube = addresses["cube"]
                contacts = finger_contacts(model, data, cube)
                relative = data.xpos[cube] - data.site_xpos[tcp_site]

                if "openarm_right_ee_inner_finger" in contacts:
                    task.left_finger_contact_steps += 1
                    if task.first_left_contact_s is None:
                        task.first_left_contact_s = float(data.time)

                if "openarm_right_ee_outer_finger" in contacts:
                    task.right_finger_contact_steps += 1
                    if task.first_right_contact_s is None:
                        task.first_right_contact_s = float(data.time)

                bilateral = len(contacts) == 2
                if bilateral:
                    task.dual_finger_contact_steps += 1
                    if task.first_dual_contact_s is None:
                        task.first_dual_contact_s = float(data.time)
                    if reference_relative is None:
                        reference_relative = relative.copy()

                drift = (
                    float(np.linalg.norm(relative - reference_relative))
                    if reference_relative is not None else float("inf")
                )

                cube_z = float(data.qpos[cube_qadr + 2])
                task.max_cube_height_m = max(task.max_cube_height_m, cube_z)
                task.max_cube_lift_m = max(task.max_cube_lift_m, cube_z - reset_z)

                task.success = evaluator.update(
                    now_s=float(data.time), cube_z=cube_z, bilateral_contact=bilateral,
                    tcp_distance_m=float(np.linalg.norm(relative)), relative_drift_m=drift,
                    cube_vertical_velocity_m_s=float(data.qvel[cube_dadr + 2]),
                ) or task.success

                if data.time >= deadline_s or task.success:
                    break

            if data.time >= deadline_s or task.success:
                break

    task.episode_duration_s = float(data.time - rollout_start_s)
    evidence = evaluator.evidence()
    task.hold_duration_s = evidence.get("hold_duration_s", 0.0)
    if not task.success:
        task.failure_reason = "lift_hold_condition_not_met"

    # Compute stage metrics
    upsampled_arr = np.array(upsampled_250hz)
    lowpassed_arr = np.array(lowpassed_250hz)
    guarded_arr = np.array(guarded_250hz)

    # Hermite deviation
    hermite_metrics = compute_deviation_metrics(
        np.repeat(np.concatenate(raw_30hz) if raw_30hz else np.array([]),
                  len(upsampled_arr) // len(np.concatenate(raw_30hz)) if raw_30hz and len(np.concatenate(raw_30hz)) > 0 else 1,
                  axis=0)[:len(upsampled_arr)],
        upsampled_arr,
        executor_dt
    ) if len(upsampled_arr) > 0 else {}

    # Lowpass deviation
    lowpass_metrics = compute_deviation_metrics(
        upsampled_arr, lowpassed_arr, executor_dt
    ) if len(lowpassed_arr) > 0 else {}

    # Guard deviation
    guard_metrics_data = compute_deviation_metrics(
        lowpassed_arr, guarded_arr, executor_dt
    ) if len(guarded_arr) > 0 else {}

    # Guard statistics
    guard_delta = {k: guard.stats.get(k, 0) - guard_stats_baseline.get(k, 0)
                   for k in set(guard.stats) | set(guard_stats_baseline)}

    hermite_stage = StageMetrics(
        name="hermite_upsampler",
        mean_deviation=hermite_metrics.get("mean", 0.0),
        max_deviation=hermite_metrics.get("max", 0.0),
        p95_deviation=hermite_metrics.get("p95", 0.0),
        phase_lag_ms=hermite_metrics.get("representative_lag_ms"),
        per_axis_max_deviation=hermite_metrics.get("per_axis_max", []),
        per_axis_lag_ms=hermite_metrics.get("per_axis_lag_ms", []),
    )

    lowpass_stage = StageMetrics(
        name="biquad_lowpass",
        mean_deviation=lowpass_metrics.get("mean", 0.0),
        max_deviation=lowpass_metrics.get("max", 0.0),
        p95_deviation=lowpass_metrics.get("p95", 0.0),
        phase_lag_ms=lowpass_metrics.get("representative_lag_ms"),
        per_axis_max_deviation=lowpass_metrics.get("per_axis_max", []),
        per_axis_lag_ms=lowpass_metrics.get("per_axis_lag_ms", []),
    )

    guard_stage = StageMetrics(
        name="action_guard",
        mean_deviation=guard_metrics_data.get("mean", 0.0),
        max_deviation=guard_metrics_data.get("max", 0.0),
        p95_deviation=guard_metrics_data.get("p95", 0.0),
        per_axis_max_deviation=guard_metrics_data.get("per_axis_max", []),
        velocity_clamps=guard_delta.get("clip_velocity", 0),
        acceleration_clamps=guard_delta.get("clip_acceleration", 0),
        step_clamps=guard_delta.get("clip_step", 0),
        ctrlrange_clamps=guard_delta.get("clip_ctrlrange", 0),
        watchdog_holds=guard_delta.get("hold_watchdog", 0),
    )

    total_control_steps = len(lowpassed_250hz)
    if total_control_steps > 0:
        guard_stage.velocity_clamp_ratio = guard_stage.velocity_clamps / total_control_steps
        guard_stage.acceleration_clamp_ratio = guard_stage.acceleration_clamps / total_control_steps
        guard_stage.step_clamp_ratio = guard_stage.step_clamps / total_control_steps

    result = ExperimentResult(
        experiment="E2",
        description="Expert → 30Hz → Hermite → Lowpass → Guard (full ACT executor)",
        task=task,
        stages={
            "hermite": hermite_stage,
            "lowpass": lowpass_stage,
            "guard": guard_stage,
        },
        trajectories={
            "raw_30hz": np.concatenate(raw_30hz) if raw_30hz else np.array([]),
            "upsampled_250hz": upsampled_arr,
            "lowpassed_250hz": lowpassed_arr,
            "guarded_250hz": guarded_arr,
        },
        final_state=PhysicsState(
            time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(), cube_pose=data.qpos[cube_qadr:cube_qadr + 7].copy(),
        ),
    )

    print(f"E2 Result: {'SUCCESS' if task.success else 'FAIL'}")
    print(f"  Max lift: {task.max_cube_lift_m:.4f} m")
    print(f"  Hold duration: {task.hold_duration_s:.2f} s")
    print(f"  Dual contact steps: {task.dual_finger_contact_steps}")
    print(f"  Hermite deviation (mean): {hermite_stage.mean_deviation:.6f} rad")
    print(f"  Lowpass deviation (mean): {lowpass_stage.mean_deviation:.6f} rad, lag: {lowpass_stage.phase_lag_ms} ms")
    print(f"  Guard deviation (mean): {guard_stage.mean_deviation:.6f} rad")
    print(f"  Guard clamps: vel={guard_stage.velocity_clamps}, acc={guard_stage.acceleration_clamps}, step={guard_stage.step_clamps}")

    return result


def run_e3_act_deployment(
    scene: Path,
    checkpoint: Path,
    seed: int,
    cfg: dict,
    device: str = "cuda",
    max_seconds: float = 45.0
) -> ExperimentResult:
    """E3: ACT checkpoint → 30Hz → Hermite → Lowpass → Guard."""
    print(f"\n{'='*80}")
    print("E3: ACT Full Deployment")
    print(f"{'='*80}\n")

    # Load ACT policy
    from lerobot.policies import get_policy_class, make_pre_post_processors

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable, using CPU")

    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()

    print(f"ACT policy loaded: chunk_size={policy.config.chunk_size}")

    # Initialize physics (use deterministic seed-based reset, no expert metadata)
    model, data, mapper, addresses = initialize_physics(scene, seed, expert_entry=None)

    # Setup ActionGuard
    lower = np.r_[
        model.actuator_ctrlrange[9:17, 0], model.actuator_ctrlrange[1:9, 0]
    ]
    upper = np.r_[
        model.actuator_ctrlrange[9:17, 1], model.actuator_ctrlrange[1:9, 1]
    ]
    guard_cfg = cfg["guard"]
    guard = ActionGuard(
        lower, upper,
        np.asarray(guard_cfg["max_velocity"], dtype=np.float32),
        np.asarray(guard_cfg["max_acceleration"], dtype=np.float32),
        np.asarray(guard_cfg["max_step"], dtype=np.float32),
        guard_cfg["watchdog_timeout_s"],
        guard_cfg["limit_policy"],
    )

    # Stabilization
    stabilize_robot(model, data, mapper, guard, cfg)
    guard_stats_baseline = Counter(guard.stats)

    # Setup executor
    HermiteUpsampler, BiquadLowpass = load_official_executor_classes()
    chunk_hz = ACT_CHUNK_HZ
    horizon_s = (policy.config.chunk_size - 1) / chunk_hz
    upsampler = HermiteUpsampler(chunk_hz=chunk_hz, horizon_sec=horizon_s)
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    executor_times = np.arange(0.0, horizon_s + 1e-9, executor_dt)
    lowpass = BiquadLowpass(fs=EXECUTOR_CONTROL_HZ, fc=LOWPASS_CUTOFF_HZ, Q=LOWPASS_Q)

    # Setup rendering for ACT observations
    renderers = {
        name: mujoco.Renderer(model, height=600, width=960)
        for name in ("ceiling", "wrist_right")
    }

    # Task tracking
    cube_qadr = addresses["cube_qadr"]
    cube_dadr = addresses["cube_dadr"]
    reset_z = float(data.qpos[cube_qadr + 2])
    evaluator = PickCubeEvaluator(reset_z, cfg["hold_height_m"], cfg["hold_s"],
                                   cfg["tcp_distance_m"], cfg.get("max_relative_drift_m", 0.025))

    task = TaskMetrics(success=False, failure_reason=None, initial_cube_height_m=reset_z)
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
    reference_relative = None

    # Trajectory recording
    raw_chunks = []
    upsampled_250hz = []
    lowpassed_250hz = []
    guarded_250hz = []

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + max_seconds
    chunk_idx = 0

    try:
        while data.time < deadline_s and not task.success:
            # Get ACT observation
            batch = {"observation.state": torch.from_numpy(get_state_vector(mapper, data.qpos)).unsqueeze(0)}
            for key, camera in (
                ("ceiling", "camera_ceiling"),
                ("wrist_right", "camera_wrist_right"),
            ):
                renderers[key].update_scene(data, camera=camera)
                rgb = renderers[key].render().copy()
                batch[f"observation.images.{key}"] = (
                    torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
                )

            # Get ACT prediction
            with torch.inference_mode():
                chunk = postprocessor(
                    policy.predict_action_chunk(preprocessor(batch))
                ).numpy()[0]

            if chunk.shape != (policy.config.chunk_size, 16):
                print(f"Invalid chunk shape: {chunk.shape}, stopping")
                break

            raw_chunks.append(chunk.copy())

            # Upsample to 250Hz
            upsampled_chunk = upsampler.upsample(chunk, executor_times)

            # Initialize lowpass on first chunk
            if chunk_idx == 0:
                lowpass.reset_state(upsampled_chunk[0])
            chunk_idx += 1

            for target_250hz in upsampled_chunk:
                upsampled_250hz.append(target_250hz.copy())

                # Apply lowpass
                filtered_250hz = lowpass.step(target_250hz)
                lowpassed_250hz.append(filtered_250hz.copy())

                guard.submit(filtered_250hz, float(data.time))

                # Execute at 1000Hz
                for _ in range(int(round(executor_dt / model.opt.timestep))):
                    guarded = guard.step(get_state_vector(mapper, data.qpos), float(data.time))
                    guarded_250hz.append(guarded.action.copy())

                    mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
                    mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
                    mujoco.mj_step(model, data)
                    task.physics_steps += 1

                    # Task evaluation
                    cube = addresses["cube"]
                    contacts = finger_contacts(model, data, cube)
                    relative = data.xpos[cube] - data.site_xpos[tcp_site]

                    if "openarm_right_ee_inner_finger" in contacts:
                        task.left_finger_contact_steps += 1
                        if task.first_left_contact_s is None:
                            task.first_left_contact_s = float(data.time)

                    if "openarm_right_ee_outer_finger" in contacts:
                        task.right_finger_contact_steps += 1
                        if task.first_right_contact_s is None:
                            task.first_right_contact_s = float(data.time)

                    bilateral = len(contacts) == 2
                    if bilateral:
                        task.dual_finger_contact_steps += 1
                        if task.first_dual_contact_s is None:
                            task.first_dual_contact_s = float(data.time)
                        if reference_relative is None:
                            reference_relative = relative.copy()

                    drift = (
                        float(np.linalg.norm(relative - reference_relative))
                        if reference_relative is not None else float("inf")
                    )

                    cube_z = float(data.qpos[cube_qadr + 2])
                    task.max_cube_height_m = max(task.max_cube_height_m, cube_z)
                    task.max_cube_lift_m = max(task.max_cube_lift_m, cube_z - reset_z)

                    task.success = evaluator.update(
                        now_s=float(data.time), cube_z=cube_z, bilateral_contact=bilateral,
                        tcp_distance_m=float(np.linalg.norm(relative)), relative_drift_m=drift,
                        cube_vertical_velocity_m_s=float(data.qvel[cube_dadr + 2]),
                    ) or task.success

                    if data.time >= deadline_s or task.success:
                        break

                if data.time >= deadline_s or task.success:
                    break

    finally:
        for renderer in renderers.values():
            renderer.close()

    task.episode_duration_s = float(data.time - rollout_start_s)
    evidence = evaluator.evidence()
    task.hold_duration_s = evidence.get("hold_duration_s", 0.0)
    if not task.success:
        task.failure_reason = "lift_hold_condition_not_met"

    # Guard statistics
    guard_delta = {k: guard.stats.get(k, 0) - guard_stats_baseline.get(k, 0)
                   for k in set(guard.stats) | set(guard_stats_baseline)}

    guard_stage = StageMetrics(
        name="action_guard",
        velocity_clamps=guard_delta.get("clip_velocity", 0),
        acceleration_clamps=guard_delta.get("clip_acceleration", 0),
        step_clamps=guard_delta.get("clip_step", 0),
        ctrlrange_clamps=guard_delta.get("clip_ctrlrange", 0),
        watchdog_holds=guard_delta.get("hold_watchdog", 0),
    )

    total_control_steps = len(lowpassed_250hz)
    if total_control_steps > 0:
        guard_stage.velocity_clamp_ratio = guard_stage.velocity_clamps / total_control_steps
        guard_stage.acceleration_clamp_ratio = guard_stage.acceleration_clamps / total_control_steps
        guard_stage.step_clamp_ratio = guard_stage.step_clamps / total_control_steps

    result = ExperimentResult(
        experiment="E3",
        description="ACT → 30Hz → Hermite → Lowpass → Guard",
        task=task,
        stages={"guard": guard_stage},
        trajectories={
            "raw_30hz": np.concatenate(raw_chunks, axis=0) if raw_chunks else np.array([]),
            "upsampled_250hz": np.array(upsampled_250hz),
            "lowpassed_250hz": np.array(lowpassed_250hz),
            "guarded_250hz": np.array(guarded_250hz),
        },
        final_state=PhysicsState(
            time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(), cube_pose=data.qpos[cube_qadr:cube_qadr + 7].copy(),
        ),
    )

    print(f"E3 Result: {'SUCCESS' if task.success else 'FAIL'}")
    print(f"  Max lift: {task.max_cube_lift_m:.4f} m")
    print(f"  Hold duration: {task.hold_duration_s:.2f} s")
    print(f"  Dual contact steps: {task.dual_finger_contact_steps}")
    print(f"  Chunks generated: {len(raw_chunks)}")
    print(f"  Guard clamps: vel={guard_stage.velocity_clamps}, acc={guard_stage.acceleration_clamps}, step={guard_stage.step_clamps}")

    return result


def serialize_result(result: ExperimentResult) -> dict:
    """Convert ExperimentResult to JSON-serializable dict."""
    task_dict = {
        "success": result.task.success,
        "failure_reason": result.task.failure_reason,
        "left_finger_contact_steps": result.task.left_finger_contact_steps,
        "right_finger_contact_steps": result.task.right_finger_contact_steps,
        "dual_finger_contact_steps": result.task.dual_finger_contact_steps,
        "first_left_contact_s": result.task.first_left_contact_s,
        "first_right_contact_s": result.task.first_right_contact_s,
        "first_dual_contact_s": result.task.first_dual_contact_s,
        "initial_cube_height_m": result.task.initial_cube_height_m,
        "max_cube_height_m": result.task.max_cube_height_m,
        "max_cube_lift_m": result.task.max_cube_lift_m,
        "hold_duration_s": result.task.hold_duration_s,
        "physics_steps": result.task.physics_steps,
        "episode_duration_s": result.task.episode_duration_s,
    }

    stages_dict = {}
    for stage_name, stage in result.stages.items():
        stages_dict[stage_name] = {
            "name": stage.name,
            "mean_deviation": stage.mean_deviation,
            "max_deviation": stage.max_deviation,
            "p95_deviation": stage.p95_deviation,
            "per_axis_max_deviation": stage.per_axis_max_deviation,
            "phase_lag_ms": stage.phase_lag_ms,
            "per_axis_lag_ms": stage.per_axis_lag_ms,
            "velocity_clamps": stage.velocity_clamps,
            "acceleration_clamps": stage.acceleration_clamps,
            "step_clamps": stage.step_clamps,
            "ctrlrange_clamps": stage.ctrlrange_clamps,
            "watchdog_holds": stage.watchdog_holds,
            "velocity_clamp_ratio": stage.velocity_clamp_ratio,
            "acceleration_clamp_ratio": stage.acceleration_clamp_ratio,
            "step_clamp_ratio": stage.step_clamp_ratio,
            "per_axis_velocity_clamps": stage.per_axis_velocity_clamps,
            "per_axis_acceleration_clamps": stage.per_axis_acceleration_clamps,
            "per_axis_step_clamps": stage.per_axis_step_clamps,
        }

    final_state_dict = None
    if result.final_state:
        final_state_dict = {
            "time": result.final_state.time,
            "cube_pose": result.final_state.cube_pose.tolist(),
        }

    return {
        "experiment": result.experiment,
        "description": result.description,
        "task": task_dict,
        "stages": stages_dict,
        "final_state": final_state_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G5.5 Executor Semantic Equivalence Gate"
    )
    parser.add_argument("--experiment", choices=["E0", "E1", "E2", "E3", "all"],
                       default="all", help="Which experiment to run")
    parser.add_argument("--seed", type=int, default=20260902,
                       help="Fixed seed for deterministic reset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                       help="Expert dataset path")
    parser.add_argument("--checkpoint", type=Path,
                       default=ROOT / "outputs/act_openarm_cube_pick_smoke_g5_500/checkpoints/last",
                       help="ACT checkpoint path (for E3)")
    parser.add_argument("--scene", type=Path, default=SCENE,
                       help="MuJoCo scene XML")
    parser.add_argument("--params", type=Path, default=PARAMS,
                       help="Expert parameters YAML")
    parser.add_argument("--device", default="cuda",
                       help="Device for ACT policy (E3)")
    parser.add_argument("--max-seconds", type=float, default=45.0,
                       help="Maximum episode duration")
    parser.add_argument("--output-dir", type=Path,
                       default=ROOT / "results/g5_5_executor_equivalence",
                       help="Output directory for results")
    parser.add_argument("--smoke", action="store_true",
                       help="Run 10s smoke test instead of full episode")
    args = parser.parse_args()

    if args.smoke:
        args.max_seconds = 10.0
        print("\n[SMOKE TEST MODE: 10 second duration]\n")

    # Load config
    cfg = yaml.safe_load(args.params.read_text())

    # Create output directory
    timestamp = Path(__file__).stat().st_mtime  # Use script mtime as pseudo-timestamp
    output_subdir = args.output_dir / f"seed_{args.seed}_{'smoke' if args.smoke else 'full'}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"G5.5 Executor Semantic Equivalence Gate")
    print(f"=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Scene: {args.scene}")
    print(f"Dataset: {args.dataset}")
    print(f"Max duration: {args.max_seconds}s")
    print(f"Output: {output_subdir}")
    print(f"=" * 80)

    results = {}

    # Run experiments
    if args.experiment in ["E0", "all"]:
        try:
            results["E0"] = run_e0_expert_strict(
                args.scene, args.dataset, args.seed, cfg, args.max_seconds
            )
        except Exception as e:
            print(f"\nE0 FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results["E0"] = None

    if args.experiment in ["E1", "all"]:
        try:
            results["E1"] = run_e1_expert_hermite(
                args.scene, args.dataset, args.seed, cfg, args.max_seconds
            )
        except Exception as e:
            print(f"\nE1 FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results["E1"] = None

    if args.experiment in ["E2", "all"]:
        try:
            results["E2"] = run_e2_expert_full_executor(
                args.scene, args.dataset, args.seed, cfg, args.max_seconds
            )
        except Exception as e:
            print(f"\nE2 FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results["E2"] = None

    if args.experiment in ["E3", "all"]:
        try:
            results["E3"] = run_e3_act_deployment(
                args.scene, args.checkpoint, args.seed, cfg, args.device, args.max_seconds
            )
        except Exception as e:
            print(f"\nE3 FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results["E3"] = None

    # Save individual results
    for exp_name, result in results.items():
        if result is not None:
            result_path = output_subdir / f"{exp_name.lower()}_result.json"
            result_dict = serialize_result(result)
            result_path.write_text(json.dumps(result_dict, indent=2) + "\n")
            print(f"\nSaved {exp_name} result: {result_path}")

            # Save trajectories
            traj_path = output_subdir / f"{exp_name.lower()}_trajectories.npz"
            np.savez_compressed(traj_path, **result.trajectories)
            print(f"Saved {exp_name} trajectories: {traj_path}")

    # Generate summary
    summary = {
        "seed": args.seed,
        "scene_sha256": sha256(args.scene),
        "params_sha256": sha256(args.params),
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint.exists() else None,
        "max_seconds": args.max_seconds,
        "smoke_test": args.smoke,
        "experiments": {},
    }

    for exp_name, result in results.items():
        if result is not None:
            summary["experiments"][exp_name] = {
                "success": result.task.success,
                "max_lift_m": result.task.max_cube_lift_m,
                "hold_duration_s": result.task.hold_duration_s,
                "dual_contact_steps": result.task.dual_finger_contact_steps,
                "episode_duration_s": result.task.episode_duration_s,
            }
        else:
            summary["experiments"][exp_name] = {"error": "experiment_failed"}

    # Determine gate status
    e0_pass = results.get("E0") and results["E0"].task.success
    e2_pass = results.get("E2") and results["E2"].task.success
    e3_pass = results.get("E3") and results["E3"].task.success

    gate_status = "UNKNOWN"
    gate_case = None
    recommendation = "Insufficient data to determine next step"

    if e0_pass is False:
        gate_status = "BLOCKED"
        gate_case = "E"
        recommendation = "E0 baseline failed. Restore expert replay before proceeding."
    elif e0_pass and e2_pass and e3_pass:
        gate_status = "PASS"
        gate_case = "D"
        recommendation = "All experiments passed. ACT + executor functional. Proceed to multi-seed evaluation."
    elif e0_pass and e2_pass and not e3_pass:
        gate_status = "PASS_WITH_FINDING"
        gate_case = "C"
        recommendation = "Executor preserves expert semantics. ACT policy is the bottleneck. Proceed to ACT baseline training (50 episodes)."
    elif e0_pass and not e2_pass:
        gate_status = "FAIL"
        gate_case = "B"
        recommendation = "Executor breaks expert trajectory. Do NOT proceed to ACT training. Debug executor semantic issues first."

    summary["gate"] = {
        "status": gate_status,
        "case": gate_case,
        "e0_pass": e0_pass,
        "e2_pass": e2_pass,
        "e3_pass": e3_pass,
        "recommendation": recommendation,
    }

    # Save summary
    summary_path = output_subdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved summary: {summary_path}")

    # Print final summary
    print(f"\n{'='*80}")
    print("G5.5 GATE SUMMARY")
    print(f"{'='*80}")
    print(f"E0 (Expert Strict):          {'PASS' if e0_pass else 'FAIL'}")
    if results.get("E0"):
        print(f"  Lift: {results['E0'].task.max_cube_lift_m:.4f}m, Hold: {results['E0'].task.hold_duration_s:.2f}s")

    if results.get("E1"):
        e1 = results["E1"]
        print(f"E1 (Expert + Hermite):       {'PASS' if e1.task.success else 'FAIL'}")
        print(f"  Lift: {e1.task.max_cube_lift_m:.4f}m, Hold: {e1.task.hold_duration_s:.2f}s")

    if results.get("E2"):
        e2 = results["E2"]
        print(f"E2 (Expert + Full Executor): {'PASS' if e2_pass else 'FAIL'}")
        print(f"  Lift: {e2.task.max_cube_lift_m:.4f}m, Hold: {e2.task.hold_duration_s:.2f}s")
        if "lowpass" in e2.stages:
            print(f"  Lowpass deviation: {e2.stages['lowpass'].mean_deviation:.6f} rad, lag: {e2.stages['lowpass'].phase_lag_ms} ms")

    if results.get("E3"):
        e3 = results["E3"]
        print(f"E3 (ACT Deployment):         {'PASS' if e3_pass else 'FAIL'}")
        print(f"  Lift: {e3.task.max_cube_lift_m:.4f}m, Hold: {e3.task.hold_duration_s:.2f}s")

    print(f"\nGATE STATUS: {gate_status} (Case {gate_case})")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

