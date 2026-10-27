#!/usr/bin/env python3
"""G5.5.1 Action Representation Equivalence Debug.

Three controlled experiments to isolate the root cause of semantic break:

E1-ZOH: Expert → 30Hz → ZeroOrderHold → Guard
E1-60Hz: Expert → 60Hz → Hermite → Guard
Hermite-Analysis: Offline safety analysis of Hermite output

Goal: Determine WHERE expert trajectory loses semantic during ACT-compatible conversion.
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
import types
import yaml

from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.config import sha256
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.guard import ActionGuard
from openarm_cube_pick_expert.task import PickCubeEvaluator


SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
PARAMS = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
DEFAULT_DATASET = ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"

# ACT deployment parameters
ACT_CHUNK_HZ = 30.0
EXECUTOR_CONTROL_HZ = 250.0


def load_official_executor_classes():
    """Import HermiteUpsampler from official executor."""
    sys.modules.setdefault("dora", types.ModuleType("dora"))
    from dora_openarm_actions_executor.main import HermiteUpsampler
    return HermiteUpsampler


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
    cube_pose: np.ndarray


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
class GuardMetrics:
    """ActionGuard statistics with proper ratio definitions."""
    # Total counts
    velocity_clamps: int = 0
    acceleration_clamps: int = 0
    step_clamps: int = 0
    ctrlrange_clamps: int = 0
    watchdog_holds: int = 0

    # Step-level ratios (clamped_steps / total_steps)
    step_clamp_ratio: float = 0.0

    # Axis-level ratios (clamped_axis_events / total_axis_samples)
    velocity_clamp_ratio: float = 0.0
    acceleration_clamp_ratio: float = 0.0

    # Per-joint distribution
    per_joint_velocity_clamps: list[int] = field(default_factory=list)
    per_joint_acceleration_clamps: list[int] = field(default_factory=list)
    per_joint_step_clamps: list[int] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Complete result for one experiment."""
    experiment: str
    description: str

    task: TaskMetrics
    guard: GuardMetrics | None = None

    # Trajectory recordings
    trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    timestamps: dict[str, np.ndarray] = field(default_factory=dict)

    # Final state
    final_state: PhysicsState | None = None


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

    # Set cube pose
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


def convert_expert_to_target_hz(
    timestamps_ns: np.ndarray,
    actions: np.ndarray,
    target_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert expert trajectory to target Hz representation.

    Uses timestamp-based interpolation.
    Returns (new_timestamps_ns, new_actions).
    """
    start_ns = timestamps_ns[0]
    end_ns = timestamps_ns[-1]

    # Generate uniform target Hz timestamps
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


def analyze_expert_frequency(timestamps_ns: np.ndarray) -> dict:
    """Analyze original expert sampling frequency."""
    dts = np.diff(timestamps_ns) / 1e9  # Convert to seconds
    mean_dt = float(np.mean(dts))
    std_dt = float(np.std(dts))
    original_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0

    return {
        "original_hz": original_hz,
        "mean_dt_s": mean_dt,
        "std_dt_s": std_dt,
        "min_dt_s": float(np.min(dts)),
        "max_dt_s": float(np.max(dts)),
    }


def run_e1_zoh(
    scene: Path,
    dataset: Path,
    seed: int,
    cfg: dict,
    max_seconds: float = 45.0
) -> ExperimentResult:
    """E1-ZOH: Expert → 30Hz → Zero Order Hold → Guard.

    Tests whether Hermite interpolation itself is the problem.
    Uses simple ZOH (step function) instead of Hermite.
    """
    print(f"\n{'='*80}")
    print("E1-ZOH: Expert → 30Hz → ZeroOrderHold → Guard")
    print(f"{'='*80}\n")

    # Load expert trajectory
    expert_data = load_expert_trajectory(dataset, seed)
    if expert_data is None:
        raise ValueError(f"Cannot load expert trajectory for seed {seed}")

    entry = expert_data["entry"]

    # Analyze original expert frequency
    freq_analysis = analyze_expert_frequency(expert_data["action_timestamps_ns"])
    print(f"Expert original frequency: {freq_analysis['original_hz']:.2f} Hz")
    print(f"  Mean dt: {freq_analysis['mean_dt_s']:.6f} s")
    print(f"  Std dt: {freq_analysis['std_dt_s']:.6f} s")

    # Convert to 30Hz
    expert_30hz_ts, expert_30hz_actions = convert_expert_to_target_hz(
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
    zoh_250hz = []
    guarded_250hz = []

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + max_seconds

    # ZOH upsampling parameters
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    steps_per_30hz = int(round((1.0 / ACT_CHUNK_HZ) / executor_dt))

    print(f"ZOH upsampling: {steps_per_30hz} steps per 30Hz frame (250Hz / 30Hz = {EXECUTOR_CONTROL_HZ / ACT_CHUNK_HZ:.2f})")

    # Process 30Hz actions with ZOH
    for action_30hz in expert_30hz_actions:
        if data.time >= deadline_s or task.success:
            break

        raw_30hz.append(action_30hz.copy())

        # ZOH: hold constant for entire 30Hz period
        for _ in range(steps_per_30hz):
            zoh_250hz.append(action_30hz.copy())
            guard.submit(action_30hz, float(data.time))

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

    total_control_steps = len(zoh_250hz)
    total_axis_samples = total_control_steps * 16  # 16 joints

    guard_metric = GuardMetrics(
        velocity_clamps=guard_delta.get("clip_velocity", 0),
        acceleration_clamps=guard_delta.get("clip_acceleration", 0),
        step_clamps=guard_delta.get("clip_step", 0),
        ctrlrange_clamps=guard_delta.get("clip_ctrlrange", 0),
        watchdog_holds=guard_delta.get("hold_watchdog", 0),
    )

    if total_control_steps > 0:
        guard_metric.step_clamp_ratio = guard_metric.step_clamps / total_control_steps
    if total_axis_samples > 0:
        guard_metric.velocity_clamp_ratio = guard_metric.velocity_clamps / total_axis_samples
        guard_metric.acceleration_clamp_ratio = guard_metric.acceleration_clamps / total_axis_samples

    result = ExperimentResult(
        experiment="E1-ZOH",
        description="Expert → 30Hz → ZeroOrderHold → Guard",
        task=task,
        guard=guard_metric,
        trajectories={
            "raw_30hz": np.array(raw_30hz),
            "zoh_250hz": np.array(zoh_250hz),
            "guarded_250hz": np.array(guarded_250hz),
        },
        final_state=PhysicsState(
            time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(), cube_pose=data.qpos[cube_qadr:cube_qadr + 7].copy(),
        ),
    )

    print(f"\nE1-ZOH Result: {'SUCCESS' if task.success else 'FAIL'}")
    print(f"  Max lift: {task.max_cube_lift_m:.4f} m")
    print(f"  Hold duration: {task.hold_duration_s:.2f} s")
    print(f"  Dual contact steps: {task.dual_finger_contact_steps}")
    print(f"  Guard step clamp ratio: {guard_metric.step_clamp_ratio:.2%}")
    print(f"  Guard velocity clamp ratio: {guard_metric.velocity_clamp_ratio:.2%}")
    print(f"  Guard acceleration clamp ratio: {guard_metric.acceleration_clamp_ratio:.2%}")

    return result


def run_e1_60hz_hermite(
    scene: Path,
    dataset: Path,
    seed: int,
    cfg: dict,
    max_seconds: float = 45.0
) -> ExperimentResult:
    """E1-60Hz: Expert → 60Hz → Hermite → Guard.

    Tests whether 30Hz representation loses information.
    Uses 60Hz instead of 30Hz to see if higher frequency helps.
    """
    print(f"\n{'='*80}")
    print("E1-60Hz: Expert → 60Hz → Hermite → Guard")
    print(f"{'='*80}\n")

    # Load expert trajectory
    expert_data = load_expert_trajectory(dataset, seed)
    if expert_data is None:
        raise ValueError(f"Cannot load expert trajectory for seed {seed}")

    entry = expert_data["entry"]

    # Analyze original expert frequency
    freq_analysis = analyze_expert_frequency(expert_data["action_timestamps_ns"])
    print(f"Expert original frequency: {freq_analysis['original_hz']:.2f} Hz")

    # Convert to 60Hz (2x ACT frequency)
    target_hz = 60.0
    expert_60hz_ts, expert_60hz_actions = convert_expert_to_target_hz(
        expert_data["action_timestamps_ns"],
        expert_data["actions"],
        target_hz=target_hz
    )

    print(f"Expert trajectory converted: {len(expert_data['actions'])} frames → {len(expert_60hz_actions)} @ 60Hz")

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

    # Setup Hermite upsampler
    HermiteUpsampler = load_official_executor_classes()
    chunk_hz = target_hz
    horizon_s = (len(expert_60hz_actions) - 1) / chunk_hz if len(expert_60hz_actions) > 1 else 1.0
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
    raw_60hz = []
    hermite_250hz = []
    guarded_250hz = []

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + max_seconds

    # Process 60Hz chunks through Hermite upsampler
    chunk_size = int(horizon_s * chunk_hz) + 1
    for chunk_start_idx in range(0, len(expert_60hz_actions), chunk_size):
        if data.time >= deadline_s:
            break

        chunk = expert_60hz_actions[chunk_start_idx:chunk_start_idx + chunk_size]
        if len(chunk) < 2:
            break

        raw_60hz.append(chunk.copy())

        # Upsample to 250Hz using Hermite
        current_horizon_s = (len(chunk) - 1) / chunk_hz
        current_times = np.arange(0.0, current_horizon_s + 1e-9, executor_dt)
        upsampled_chunk = upsampler.upsample(chunk, current_times)

        for target_250hz in upsampled_chunk:
            hermite_250hz.append(target_250hz.copy())
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

    total_control_steps = len(hermite_250hz)
    total_axis_samples = total_control_steps * 16

    guard_metric = GuardMetrics(
        velocity_clamps=guard_delta.get("clip_velocity", 0),
        acceleration_clamps=guard_delta.get("clip_acceleration", 0),
        step_clamps=guard_delta.get("clip_step", 0),
        ctrlrange_clamps=guard_delta.get("clip_ctrlrange", 0),
        watchdog_holds=guard_delta.get("hold_watchdog", 0),
    )

    if total_control_steps > 0:
        guard_metric.step_clamp_ratio = guard_metric.step_clamps / total_control_steps
    if total_axis_samples > 0:
        guard_metric.velocity_clamp_ratio = guard_metric.velocity_clamps / total_axis_samples
        guard_metric.acceleration_clamp_ratio = guard_metric.acceleration_clamps / total_axis_samples

    result = ExperimentResult(
        experiment="E1-60Hz",
        description="Expert → 60Hz → Hermite → Guard",
        task=task,
        guard=guard_metric,
        trajectories={
            "raw_60hz": np.concatenate(raw_60hz) if raw_60hz else np.array([]),
            "hermite_250hz": np.array(hermite_250hz),
            "guarded_250hz": np.array(guarded_250hz),
        },
        final_state=PhysicsState(
            time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(), cube_pose=data.qpos[cube_qadr:cube_qadr + 7].copy(),
        ),
    )

    print(f"\nE1-60Hz Result: {'SUCCESS' if task.success else 'FAIL'}")
    print(f"  Max lift: {task.max_cube_lift_m:.4f} m")
    print(f"  Hold duration: {task.hold_duration_s:.2f} s")
    print(f"  Dual contact steps: {task.dual_finger_contact_steps}")
    print(f"  Guard step clamp ratio: {guard_metric.step_clamp_ratio:.2%}")
    print(f"  Guard velocity clamp ratio: {guard_metric.velocity_clamp_ratio:.2%}")
    print(f"  Guard acceleration clamp ratio: {guard_metric.acceleration_clamp_ratio:.2%}")

    return result


def analyze_hermite_offline(
    dataset: Path,
    seed: int,
    cfg: dict,
) -> dict:
    """Hermite Offline Safety Analysis.

    Analyzes Hermite output WITHOUT entering MuJoCo simulation.
    Compares Hermite velocity/acceleration against Guard limits.
    """
    print(f"\n{'='*80}")
    print("Hermite Offline Safety Analysis")
    print(f"{'='*80}\n")

    # Load expert trajectory
    expert_data = load_expert_trajectory(dataset, seed)
    if expert_data is None:
        raise ValueError(f"Cannot load expert trajectory for seed {seed}")

    # Analyze original frequency
    freq_analysis = analyze_expert_frequency(expert_data["action_timestamps_ns"])
    print(f"Expert original frequency: {freq_analysis['original_hz']:.2f} Hz")

    # Convert to 30Hz
    expert_30hz_ts, expert_30hz_actions = convert_expert_to_target_hz(
        expert_data["action_timestamps_ns"],
        expert_data["actions"],
        target_hz=ACT_CHUNK_HZ
    )

    print(f"Expert trajectory converted: {len(expert_data['actions'])} frames → {len(expert_30hz_actions)} @ 30Hz")

    # Setup Hermite upsampler
    HermiteUpsampler = load_official_executor_classes()
    chunk_hz = ACT_CHUNK_HZ
    horizon_s = (len(expert_30hz_actions) - 1) / chunk_hz if len(expert_30hz_actions) > 1 else 1.0
    upsampler = HermiteUpsampler(chunk_hz=chunk_hz, horizon_sec=horizon_s)
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    executor_times = np.arange(0.0, horizon_s + 1e-9, executor_dt)

    # Process through Hermite
    hermite_output = []
    chunk_size = int(horizon_s * chunk_hz) + 1
    for chunk_start_idx in range(0, len(expert_30hz_actions), chunk_size):
        chunk = expert_30hz_actions[chunk_start_idx:chunk_start_idx + chunk_size]
        if len(chunk) < 2:
            break

        current_horizon_s = (len(chunk) - 1) / chunk_hz
        current_times = np.arange(0.0, current_horizon_s + 1e-9, executor_dt)
        upsampled_chunk = upsampler.upsample(chunk, current_times)
        hermite_output.extend(upsampled_chunk)

    hermite_trajectory = np.array(hermite_output)
    print(f"Hermite output: {len(hermite_trajectory)} samples @ 250Hz")

    # Compute velocity (rad/s)
    velocity = np.diff(hermite_trajectory, axis=0) / executor_dt
    velocity = np.vstack([velocity, velocity[-1:]])  # Pad to match length

    # Compute acceleration (rad/s²)
    acceleration = np.diff(velocity, axis=0) / executor_dt
    acceleration = np.vstack([acceleration, acceleration[-1:]])

    # Load Guard limits
    guard_cfg = cfg["guard"]
    max_velocity = np.asarray(guard_cfg["max_velocity"], dtype=np.float32)
    max_acceleration = np.asarray(guard_cfg["max_acceleration"], dtype=np.float32)
    max_step = np.asarray(guard_cfg["max_step"], dtype=np.float32)

    # Compute step size
    step = np.diff(hermite_trajectory, axis=0)
    step = np.vstack([step, step[-1:]])

    # Per-axis analysis
    joint_names = [
        "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7", "right_gripper",
        "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7", "left_gripper",
    ]

    per_axis_analysis = []
    for axis_idx in range(16):
        axis_data = {
            "joint": joint_names[axis_idx],
            "position_range": [float(hermite_trajectory[:, axis_idx].min()), float(hermite_trajectory[:, axis_idx].max())],
            "velocity": {
                "max": float(np.abs(velocity[:, axis_idx]).max()),
                "mean": float(np.abs(velocity[:, axis_idx]).mean()),
                "limit": float(max_velocity[axis_idx]),
                "violations": int(np.sum(np.abs(velocity[:, axis_idx]) > max_velocity[axis_idx])),
                "violation_ratio": float(np.mean(np.abs(velocity[:, axis_idx]) > max_velocity[axis_idx])),
            },
            "acceleration": {
                "max": float(np.abs(acceleration[:, axis_idx]).max()),
                "mean": float(np.abs(acceleration[:, axis_idx]).mean()),
                "limit": float(max_acceleration[axis_idx]),
                "violations": int(np.sum(np.abs(acceleration[:, axis_idx]) > max_acceleration[axis_idx])),
                "violation_ratio": float(np.mean(np.abs(acceleration[:, axis_idx]) > max_acceleration[axis_idx])),
            },
            "step": {
                "max": float(np.abs(step[:, axis_idx]).max()),
                "mean": float(np.abs(step[:, axis_idx]).mean()),
                "limit": float(max_step[axis_idx]),
                "violations": int(np.sum(np.abs(step[:, axis_idx]) > max_step[axis_idx])),
                "violation_ratio": float(np.mean(np.abs(step[:, axis_idx]) > max_step[axis_idx])),
            },
        }
        per_axis_analysis.append(axis_data)

    # Overall statistics
    total_samples = len(hermite_trajectory)
    total_velocity_violations = sum(a["velocity"]["violations"] for a in per_axis_analysis)
    total_acceleration_violations = sum(a["acceleration"]["violations"] for a in per_axis_analysis)
    total_step_violations = sum(a["step"]["violations"] for a in per_axis_analysis)

    overall_stats = {
        "total_samples": total_samples,
        "total_axis_samples": total_samples * 16,
        "velocity": {
            "total_violations": total_velocity_violations,
            "violation_ratio": total_velocity_violations / (total_samples * 16),
        },
        "acceleration": {
            "total_violations": total_acceleration_violations,
            "violation_ratio": total_acceleration_violations / (total_samples * 16),
        },
        "step": {
            "total_violations": total_step_violations,
            "violation_ratio": total_step_violations / (total_samples * 16),
        },
    }

    result = {
        "frequency_analysis": freq_analysis,
        "overall_statistics": overall_stats,
        "per_axis_analysis": per_axis_analysis,
    }

    # Print summary
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Velocity violations: {total_velocity_violations} / {total_samples * 16} ({overall_stats['velocity']['violation_ratio']:.2%})")
    print(f"  Acceleration violations: {total_acceleration_violations} / {total_samples * 16} ({overall_stats['acceleration']['violation_ratio']:.2%})")
    print(f"  Step violations: {total_step_violations} / {total_samples * 16} ({overall_stats['step']['violation_ratio']:.2%})")

    print(f"\nTop 5 joints with velocity violations:")
    sorted_by_vel = sorted(per_axis_analysis, key=lambda x: x["velocity"]["violations"], reverse=True)
    for i, axis in enumerate(sorted_by_vel[:5]):
        print(f"  {i+1}. {axis['joint']}: {axis['velocity']['violations']} violations ({axis['velocity']['violation_ratio']:.2%}), "
              f"max={axis['velocity']['max']:.3f} vs limit={axis['velocity']['limit']:.3f}")

    print(f"\nTop 5 joints with acceleration violations:")
    sorted_by_acc = sorted(per_axis_analysis, key=lambda x: x["acceleration"]["violations"], reverse=True)
    for i, axis in enumerate(sorted_by_acc[:5]):
        print(f"  {i+1}. {axis['joint']}: {axis['acceleration']['violations']} violations ({axis['acceleration']['violation_ratio']:.2%}), "
              f"max={axis['acceleration']['max']:.3f} vs limit={axis['acceleration']['limit']:.3f}")

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

    guard_dict = None
    if result.guard:
        guard_dict = {
            "velocity_clamps": result.guard.velocity_clamps,
            "acceleration_clamps": result.guard.acceleration_clamps,
            "step_clamps": result.guard.step_clamps,
            "ctrlrange_clamps": result.guard.ctrlrange_clamps,
            "watchdog_holds": result.guard.watchdog_holds,
            "step_clamp_ratio": result.guard.step_clamp_ratio,
            "velocity_clamp_ratio": result.guard.velocity_clamp_ratio,
            "acceleration_clamp_ratio": result.guard.acceleration_clamp_ratio,
            "per_joint_velocity_clamps": result.guard.per_joint_velocity_clamps,
            "per_joint_acceleration_clamps": result.guard.per_joint_acceleration_clamps,
            "per_joint_step_clamps": result.guard.per_joint_step_clamps,
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
        "guard": guard_dict,
        "final_state": final_state_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G5.5.1 Action Representation Equivalence Debug"
    )
    parser.add_argument("--experiment", choices=["E1-ZOH", "E1-60Hz", "hermite-analysis", "all"],
                       default="all", help="Which experiment to run")
    parser.add_argument("--seed", type=int, default=20260902,
                       help="Fixed seed for deterministic reset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                       help="Expert dataset path")
    parser.add_argument("--scene", type=Path, default=SCENE,
                       help="MuJoCo scene XML")
    parser.add_argument("--params", type=Path, default=PARAMS,
                       help="Expert parameters YAML")
    parser.add_argument("--max-seconds", type=float, default=45.0,
                       help="Maximum episode duration")
    parser.add_argument("--output-dir", type=Path,
                       default=ROOT / "results/g5_5_1_action_representation_debug",
                       help="Output directory for results")
    args = parser.parse_args()

    # Load config
    cfg = yaml.safe_load(args.params.read_text())

    # Create output directory
    output_subdir = args.output_dir / f"seed_{args.seed}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"G5.5.1 Action Representation Equivalence Debug")
    print(f"=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Scene: {args.scene}")
    print(f"Dataset: {args.dataset}")
    print(f"Max duration: {args.max_seconds}s")
    print(f"Output: {output_subdir}")
    print(f"=" * 80)

    results = {}

    # Run experiments
    if args.experiment in ["E1-ZOH", "all"]:
        try:
            results["E1-ZOH"] = run_e1_zoh(
                args.scene, args.dataset, args.seed, cfg, args.max_seconds
            )
        except Exception as e:
            print(f"\nE1-ZOH FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results["E1-ZOH"] = None

    if args.experiment in ["E1-60Hz", "all"]:
        try:
            results["E1-60Hz"] = run_e1_60hz_hermite(
                args.scene, args.dataset, args.seed, cfg, args.max_seconds
            )
        except Exception as e:
            print(f"\nE1-60Hz FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results["E1-60Hz"] = None

    if args.experiment in ["hermite-analysis", "all"]:
        try:
            hermite_analysis = analyze_hermite_offline(
                args.dataset, args.seed, cfg
            )
            # Save hermite analysis
            analysis_path = output_subdir / "hermite_safety_analysis.json"
            analysis_path.write_text(json.dumps(hermite_analysis, indent=2) + "\n")
            print(f"\nSaved Hermite analysis: {analysis_path}")
        except Exception as e:
            print(f"\nHermite analysis FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    # Save individual results
    for exp_name, result in results.items():
        if result is not None:
            result_path = output_subdir / f"{exp_name.lower().replace('-', '_')}_result.json"
            result_dict = serialize_result(result)
            result_path.write_text(json.dumps(result_dict, indent=2) + "\n")
            print(f"\nSaved {exp_name} result: {result_path}")

            # Save trajectories
            traj_path = output_subdir / f"{exp_name.lower().replace('-', '_')}_trajectories.npz"
            np.savez_compressed(traj_path, **result.trajectories)
            print(f"Saved {exp_name} trajectories: {traj_path}")

    # Generate summary
    summary = {
        "seed": args.seed,
        "scene_sha256": sha256(args.scene),
        "params_sha256": sha256(args.params),
        "dataset": str(args.dataset.resolve()),
        "max_seconds": args.max_seconds,
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
                "guard_step_clamp_ratio": result.guard.step_clamp_ratio if result.guard else None,
                "guard_velocity_clamp_ratio": result.guard.velocity_clamp_ratio if result.guard else None,
                "guard_acceleration_clamp_ratio": result.guard.acceleration_clamp_ratio if result.guard else None,
            }
        else:
            summary["experiments"][exp_name] = {"error": "experiment_failed"}

    # Save summary
    summary_path = output_subdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved summary: {summary_path}")

    # Print final summary
    print(f"\n{'='*80}")
    print("G5.5.1 SUMMARY")
    print(f"{'='*80}")

    for exp_name, result in results.items():
        if result:
            print(f"{exp_name}: {'SUCCESS' if result.task.success else 'FAIL'}")
            print(f"  Lift: {result.task.max_cube_lift_m:.4f}m, Hold: {result.task.hold_duration_s:.2f}s")
            if result.guard:
                print(f"  Guard step clamp: {result.guard.step_clamp_ratio:.2%}")
                print(f"  Guard velocity clamp: {result.guard.velocity_clamp_ratio:.2%}")
                print(f"  Guard acceleration clamp: {result.guard.acceleration_clamp_ratio:.2%}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

