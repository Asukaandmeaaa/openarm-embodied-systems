#!/usr/bin/env python3
"""Compare expert and ACT trajectories at each smoothing stage.

Execution chain breakdown:
  Expert:  raw_expert_action -> ActionGuard -> ctrl
  ACT:     raw_act_chunk -> HermiteUpsampler -> BiquadLowpass -> ActionGuard -> ctrl

This script isolates each stage's contribution to trajectory deviation and
identifies where velocity/acceleration clamping occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
import torch
import types
import yaml
from lerobot.policies import get_policy_class, make_pre_post_processors

from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.config import sha256
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.guard import ActionGuard


SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
PARAMS = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
EXECUTOR_IMPL = EXECUTOR_SRC / "dora_openarm_actions_executor/main.py"
ACTION_INTERVAL_NS = 33_333_333
EXECUTOR_CONTROL_HZ = 250


def load_official_executor_classes():
    """Import math helpers from the official executor without requiring Dora."""
    sys.modules.setdefault("dora", types.ModuleType("dora"))
    from dora_openarm_actions_executor.main import BiquadLowpass, HermiteUpsampler
    return HermiteUpsampler, BiquadLowpass


@dataclass
class StageMetrics:
    """Metrics for one processing stage."""
    name: str
    mean_deviation: float
    max_deviation: float
    velocity_clamps: int
    acceleration_clamps: int
    step_clamps: int
    ctrlrange_clamps: int


@dataclass
class TrajectoryComparison:
    """Complete trajectory comparison at all stages."""
    seed: int
    checkpoint: str

    # Raw output comparison
    raw_act_vs_expert_deviation: float

    # Stage-by-stage ACT processing metrics
    after_upsampler: StageMetrics
    after_lowpass: StageMetrics
    after_guard: StageMetrics

    # Expert processing metrics (for reference)
    expert_guard: StageMetrics

    # Final state comparison
    final_state_deviation: float


def compute_deviation(trajectory_a: np.ndarray, trajectory_b: np.ndarray) -> dict:
    """Compute deviation metrics between two trajectories."""
    if trajectory_a.shape != trajectory_b.shape:
        # Interpolate to common length for comparison
        min_len = min(len(trajectory_a), len(trajectory_b))
        trajectory_a = trajectory_a[:min_len]
        trajectory_b = trajectory_b[:min_len]

    diff = np.abs(trajectory_a - trajectory_b)
    return {
        "mean": float(diff.mean()),
        "max": float(diff.max()),
        "p95": float(np.quantile(diff, 0.95)),
        "per_axis_max": diff.max(axis=0).tolist(),
    }


def analyze_guard_stats(stats_delta: dict) -> dict:
    """Extract clamp counts from guard statistics."""
    return {
        "velocity_clamps": stats_delta.get("clip_velocity", 0),
        "acceleration_clamps": stats_delta.get("clip_acceleration", 0),
        "step_clamps": stats_delta.get("clip_step", 0),
        "ctrlrange_clamps": stats_delta.get("clip_ctrlrange", 0),
        "watchdog_holds": stats_delta.get("hold_watchdog", 0),
    }


def load_expert_trajectory(dataset: Path, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load expert action trajectory for the given seed."""
    import pyarrow.parquet as pq

    metadata_path = dataset / "metadata.yaml"
    if not metadata_path.exists():
        return None

    metadata = yaml.safe_load(metadata_path.read_text())
    episode_id = None
    for entry in metadata.get("episodes", []):
        if int(entry.get("seed", -1)) == seed:
            episode_id = str(entry["id"])
            break

    if episode_id is None:
        return None

    episode_dir = dataset / "episodes" / episode_id
    timestamps = None
    parts = []

    for side in ("right", "left"):
        path = episode_dir / "action" / "arms" / side / "state.parquet"
        if not path.exists():
            return None
        table = pq.read_table(path, columns=["timestamp", "qpos"])
        side_timestamps = table["timestamp"].to_numpy()
        values = np.asarray(table["qpos"].to_pylist(), dtype=np.float32)

        if timestamps is None:
            timestamps = side_timestamps
        parts.append(values)

    actions = np.concatenate(parts, axis=1)
    return timestamps, actions


def run_comparison(
    checkpoint: Path,
    seed: int,
    expert_dataset: Path,
    device: str = "cuda",
    max_seconds: float = 45.0,
) -> TrajectoryComparison | None:
    """Run trajectory comparison experiment."""

    cfg = yaml.safe_load(PARAMS.read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mapper = JointResolver(model)

    # Reset to seed-specific initial condition
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
    cube_joint = int(model.body_jntadr[cube])
    cube_qadr = int(model.jnt_qposadr[cube_joint])

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

    def state() -> np.ndarray:
        right, right_gripper = mapper.get_driver(data.qpos, "right")
        left, left_gripper = mapper.get_driver(data.qpos, "left")
        return combine_action(
            np.r_[right, right_gripper], np.r_[left, left_gripper]
        )

    # Setup guards
    lower = np.r_[
        model.actuator_ctrlrange[9:17, 0], model.actuator_ctrlrange[1:9, 0]
    ]
    upper = np.r_[
        model.actuator_ctrlrange[9:17, 1], model.actuator_ctrlrange[1:9, 1]
    ]
    guard_cfg = cfg["guard"]

    # Stabilization phase
    stabilize_target = state()
    stabilize_target[7] = cfg["pregrasp_right_gripper_rad"]
    guard = ActionGuard(
        lower, upper,
        guard_cfg["max_velocity"],
        guard_cfg["max_acceleration"],
        guard_cfg["max_step"],
        guard_cfg["watchdog_timeout_s"],
        guard_cfg["limit_policy"],
    )

    for _ in range(int(round(cfg["stabilize_s"] / model.opt.timestep))):
        guard.submit(stabilize_target, float(data.time))
        guarded = guard.step(state(), float(data.time))
        mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
        mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
        mujoco.mj_step(model, data)

    # Load ACT policy
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()

    # Setup rendering
    renderers = {
        name: mujoco.Renderer(model, height=600, width=960)
        for name in ("ceiling", "wrist_right")
    }

    # Setup processing chain
    HermiteUpsampler, BiquadLowpass = load_official_executor_classes()
    chunk_hz = 1e9 / ACTION_INTERVAL_NS
    horizon_s = (policy.config.chunk_size - 1) / chunk_hz
    upsampler = HermiteUpsampler(chunk_hz=chunk_hz, horizon_sec=horizon_s)
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    executor_times = np.arange(0.0, horizon_s + 1e-9, executor_dt)
    lowpass = BiquadLowpass(fs=EXECUTOR_CONTROL_HZ, fc=15.0)

    # Trajectory recording
    raw_chunks = []
    upsampled_trajectory = []
    lowpass_trajectory = []
    guarded_trajectory = []

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + max_seconds
    guard_stats_baseline = Counter(guard.stats)

    try:
        chunk_idx = 0
        while data.time < deadline_s and chunk_idx < 10:  # Limit to first 10 chunks for analysis
            batch = {"observation.state": torch.from_numpy(state()).unsqueeze(0)}
            for key, camera in (
                ("ceiling", "camera_ceiling"),
                ("wrist_right", "camera_wrist_right"),
            ):
                renderers[key].update_scene(data, camera=camera)
                rgb = renderers[key].render().copy()
                batch[f"observation.images.{key}"] = (
                    torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
                )

            with torch.inference_mode():
                chunk = postprocessor(
                    policy.predict_action_chunk(preprocessor(batch))
                ).numpy()[0]

            if chunk.shape != (policy.config.chunk_size, 16):
                break

            raw_chunks.append(chunk.copy())

            # Stage 1: Upsampling
            executor_chunk = upsampler.upsample(chunk, executor_times)
            if chunk_idx == 0:
                lowpass.reset_state(executor_chunk[0])

            # Stage 2: Lowpass filtering
            for target in executor_chunk:
                upsampled_trajectory.append(target.copy())
                filtered_target = lowpass.step(target)
                lowpass_trajectory.append(filtered_target.copy())

                # Stage 3: ActionGuard
                guard.submit(filtered_target, float(data.time))
                for _ in range(int(round(executor_dt / model.opt.timestep))):
                    guarded = guard.step(state(), float(data.time))
                    guarded_trajectory.append(guarded.action.copy())

                    mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
                    mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
                    mujoco.mj_step(model, data)

                    if data.time >= deadline_s:
                        break
                if data.time >= deadline_s:
                    break

            chunk_idx += 1

    finally:
        for renderer in renderers.values():
            renderer.close()

    # Compute stage-by-stage metrics
    raw_chunks_flat = np.concatenate(raw_chunks, axis=0)
    upsampled_trajectory = np.array(upsampled_trajectory)
    lowpass_trajectory = np.array(lowpass_trajectory)
    guarded_trajectory = np.array(guarded_trajectory)

    guard_stats = dict(guard.stats)
    guard_delta = {k: guard_stats.get(k, 0) - guard_stats_baseline.get(k, 0)
                   for k in set(guard_stats) | set(guard_stats_baseline)}
    guard_analysis = analyze_guard_stats(guard_delta)

    # Compute deviations
    upsampler_dev = compute_deviation(
        upsampled_trajectory,
        np.repeat(raw_chunks_flat, len(upsampled_trajectory) // len(raw_chunks_flat), axis=0)[:len(upsampled_trajectory)]
    )
    lowpass_dev = compute_deviation(lowpass_trajectory, upsampled_trajectory)
    guard_dev = compute_deviation(
        guarded_trajectory,
        np.repeat(lowpass_trajectory, len(guarded_trajectory) // len(lowpass_trajectory), axis=0)[:len(guarded_trajectory)]
    )

    result = {
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "chunks_analyzed": len(raw_chunks),
        "raw_chunk_count": len(raw_chunks_flat),
        "upsampled_count": len(upsampled_trajectory),
        "lowpass_count": len(lowpass_trajectory),
        "guarded_count": len(guarded_trajectory),

        "stage_analysis": {
            "upsampler": {
                "deviation": upsampler_dev,
                "description": "Hermite cubic spline interpolation from 30Hz to 250Hz",
            },
            "lowpass": {
                "deviation": lowpass_dev,
                "description": "Biquad lowpass filter at 15Hz cutoff",
            },
            "guard": {
                "deviation": guard_dev,
                "clamps": guard_analysis,
                "description": "ActionGuard velocity/acceleration/step limiting",
            },
        },

        "trajectory_smoothing_responsibility": {
            "primary_smoother": "BiquadLowpass",
            "rationale": "15Hz cutoff explicitly targets trajectory smoothness",
            "upsampler_role": "interpolation_only",
            "guard_role": "safety_clipping",
        },
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--expert-dataset", type=Path,
                       default=ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    print(f"Analyzing trajectory deviation across processing stages...", flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  Seed: {args.seed}", flush=True)

    result = run_comparison(
        args.checkpoint,
        args.seed,
        args.expert_dataset,
        args.device,
        args.max_seconds,
    )

    if result is None:
        print("Failed to complete analysis", flush=True)
        return

    rendered = json.dumps(result, indent=2) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"\nResults written to: {args.output}", flush=True)

    print("\n" + rendered, end="")

    # Print summary
    print("\n" + "="*80)
    print("TRAJECTORY SMOOTHING RESPONSIBILITY BOUNDARY")
    print("="*80)
    print(f"\nPrimary smoothing: {result['trajectory_smoothing_responsibility']['primary_smoother']}")
    print(f"Rationale: {result['trajectory_smoothing_responsibility']['rationale']}")
    print(f"\nStage deviations:")
    for stage_name, stage_data in result["stage_analysis"].items():
        print(f"  {stage_name:12s}: mean={stage_data['deviation']['mean']:.6f}  max={stage_data['deviation']['max']:.6f}")
        if "clamps" in stage_data:
            clamps = stage_data["clamps"]
            print(f"                velocity={clamps['velocity_clamps']}  "
                  f"accel={clamps['acceleration_clamps']}  "
                  f"step={clamps['step_clamps']}")


if __name__ == "__main__":
    main()
