#!/usr/bin/env python3
"""Strictly replay recorded safe position targets through MuJoCo actuators.

The recorder stores the exact guarded ``data.ctrl`` target together with the
post-physics observation.  Therefore action ``i`` is applied over the interval
``(timestamp[i-1], timestamp[i]]`` and is compared with observation ``i``.
Recorded actions must not be passed through ActionGuard a second time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "official/openarm_mujoco/src")]

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm.auto import tqdm

from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.config import sha256
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.task import PickCubeEvaluator


DEFAULT_DATASET = ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
DEFAULT_SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"


def load_stream(episode_dir: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Load synchronized right8+left8 qpos and integer nanosecond timestamps."""
    timestamps = None
    parts = []
    for side in ("right", "left"):
        path = episode_dir / kind / "arms" / side / "state.parquet"
        table = pq.read_table(path, columns=["timestamp", "qpos"])
        side_timestamps = table["timestamp"].cast(pa.int64()).to_numpy()
        values = np.asarray(table["qpos"].to_pylist(), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError(f"{path}: expected [N,8], got {values.shape}")
        if timestamps is None:
            timestamps = side_timestamps
        elif not np.array_equal(timestamps, side_timestamps):
            raise ValueError(f"{kind} right/left timestamps differ")
        parts.append(values)
    assert timestamps is not None
    return timestamps, np.concatenate(parts, axis=1)


def _finger_contacts(model, data, cube_body: int) -> set[str]:
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


def replay_episode(
    dataset: Path,
    scene: Path,
    entry: dict,
    *,
    max_tracking_error_rad: float = 0.05,
    task_contract: str | None = None,
) -> dict:
    episode_id = str(entry["id"])
    episode_dir = dataset / "episodes" / episode_id
    action_ts, actions = load_stream(episode_dir, "action")
    observation_ts, observations = load_stream(episode_dir, "obs")
    if not np.array_equal(action_ts, observation_ts):
        raise ValueError("action and observation timestamps differ")
    if len(actions) < 2 or np.any(np.diff(action_ts) <= 0):
        raise ValueError("episode timestamps must be strictly increasing")

    model = mujoco.MjModel.from_xml_path(str(scene))
    black_box_task = task_contract == "openarm_cube_into_black_box_v3"
    if black_box_task:
        from openarm_cube_into_black_box_expert.scene import move_black_box_to_middle

        move_black_box_to_middle(model)
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError(f"strict replay requires 1000 Hz physics, got dt={model.opt.timestep}")
    data = mujoco.MjData(model)
    mapper = JointResolver(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
    cube_joint = int(model.body_jntadr[cube])
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    cube_dadr = int(model.jnt_dofadr[cube_joint])
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")

    initial = np.asarray(entry["initial_qpos"], dtype=np.float64)
    cube_pose = np.asarray(entry["objects"]["orange_cube"]["pose"], dtype=np.float64)
    if initial.shape != (16,) or cube_pose.shape != (7,):
        raise ValueError("metadata initial_qpos/cube pose has an invalid shape")
    mapper.set_qpos(data.qpos, initial[:8], "right")
    mapper.set_qpos(data.qpos, initial[8:], "left")
    data.qpos[cube_qadr:cube_qadr + 7] = cube_pose
    data.qvel.fill(0)
    phase_start_s = float(entry["phase_timestamps"]["MOVE_PREGRASP"])
    data.time = phase_start_s
    mujoco.mj_forward(model, data)

    def state() -> np.ndarray:
        right, right_gripper = mapper.get_driver(data.qpos, "right")
        left, left_gripper = mapper.get_driver(data.qpos, "left")
        return combine_action(
            np.r_[right, right_gripper], np.r_[left, left_gripper]
        )

    full_flow = entry.get("success_reason") == "pick_hold_place_release_return_home"
    evaluator = PickCubeEvaluator(
        float(cube_pose[2]), required_hold_s=2.0 if full_flow else 1.0
    )
    reference_relative = None
    success = False
    tracking_errors = []
    physics_steps = 0
    previous_timestamp_ns = int(round(phase_start_s * 1e9))
    timestamp_error_ns = []

    def step_and_evaluate() -> None:
        nonlocal physics_steps, reference_relative, success
        mujoco.mj_step(model, data)
        physics_steps += 1
        relative = data.xpos[cube] - data.site_xpos[tcp_site]
        bilateral = len(_finger_contacts(model, data, cube)) == 2
        if bilateral and reference_relative is None:
            reference_relative = relative.copy()
        drift = (
            float(np.linalg.norm(relative - reference_relative))
            if reference_relative is not None
            else float("inf")
        )
        # Once the required hold has completed, preserve its evidence while
        # replay continues through PLACE/RELEASE/RETURN_HOME.
        if not success:
            success = evaluator.update(
                now_s=float(data.time),
                cube_z=float(data.qpos[cube_qadr + 2]),
                bilateral_contact=bilateral,
                tcp_distance_m=float(np.linalg.norm(relative)),
                relative_drift_m=drift,
                cube_vertical_velocity_m_s=float(data.qvel[cube_dadr + 2]),
            )

    for timestamp_ns, action, recorded_observation in zip(
        action_ts, actions, observations, strict=True
    ):
        timestamp_ns = int(timestamp_ns)
        interval_ns = timestamp_ns - previous_timestamp_ns
        requested_steps = interval_ns / (model.opt.timestep * 1e9)
        steps = int(round(requested_steps))
        if steps < 1 or abs(requested_steps - steps) > 1e-3:
            raise ValueError(
                f"timestamp interval {interval_ns} ns is off the 1000 Hz physics grid"
            )

        # The stored action is already the safe target actually sent to ctrl.
        mapper.set_ctrl(data.ctrl, action[:8], "right")
        mapper.set_ctrl(data.ctrl, action[8:], "left")
        for _ in range(steps):
            step_and_evaluate()

        timestamp_error_ns.append(abs(round(data.time * 1e9) - timestamp_ns))
        tracking_errors.append(float(np.max(np.abs(state() - recorded_observation))))
        previous_timestamp_ns = timestamp_ns

    # Sampling stops on success and may end up to one 30 Hz period before the
    # terminal FSM timestamp. Hold the final recorded safe ctrl through that
    # unsampled tail so replay covers the complete physical episode.
    terminal_s = float(
        entry["phase_timestamps"].get(
            "SUCCESS", entry["phase_timestamps"].get("FAIL", data.time)
        )
    )
    terminal_steps = max(0, int(round((terminal_s - data.time) / model.opt.timestep)))
    for _ in range(terminal_steps):
        step_and_evaluate()

    errors = np.asarray(tracking_errors, dtype=np.float64)
    evidence = evaluator.evidence()
    if evidence["hold_duration_s"] + 1e-9 >= evaluator.required_hold_s:
        evidence["hold_duration_s"] = max(
            evidence["hold_duration_s"], evaluator.required_hold_s
        )
    valid_hold_frames = [frame for frame in evidence.pop("hold_frames") if frame["valid"]]
    report = {
        "episode": int(episode_id),
        "seed": int(entry["seed"]),
        "success": bool(success),
        "frames_replayed": int(len(actions)),
        "physics_steps": int(physics_steps),
        "terminal_hold_steps": int(terminal_steps),
        "physics_hz": int(round(1.0 / model.opt.timestep)),
        "action_alignment": "action[i] drives (timestamp[i-1], timestamp[i]]; observation[i] is post-physics",
        "action_path": "recorded_safe_position_target -> data.ctrl -> mj_step (no second ActionGuard)",
        "scene_sha256": sha256(scene),
        "expected_scene_sha256": entry.get("scene_sha256"),
        "timestamp_max_error_ns": int(max(timestamp_error_ns, default=0)),
        "tracking_error_rad": {
            "mean": float(errors.mean()),
            "p95": float(np.quantile(errors, 0.95)),
            "max": float(errors.max()),
            "threshold": float(max_tracking_error_rad),
        },
        **evidence,
        "valid_hold_frame_count": len(valid_hold_frames),
        "valid_hold_first": valid_hold_frames[0] if valid_hold_frames else None,
        "valid_hold_last": valid_hold_frames[-1] if valid_hold_frames else None,
        "final_state": state().tolist(),
        "final_cube_pose_wxyz": data.qpos[cube_qadr:cube_qadr + 7].tolist(),
    }
    if full_flow:
        target_xy = np.asarray(entry["place_target_xy_m"], dtype=np.float64)
        final_cube = data.qpos[cube_qadr:cube_qadr + 7].copy()
        final_state = state()
        report["place_error_m"] = float(np.linalg.norm(final_cube[:2] - target_xy))
        if black_box_task:
            from openarm_cube_into_black_box_expert.contract import (
                BLACK_BOX_CUBE_CENTER_Z_M,
                TaskContract,
            )

            report["place_height_error_m"] = abs(
                float(final_cube[2] - BLACK_BOX_CUBE_CENTER_Z_M)
            )
            placement_valid = TaskContract().box_contains(final_cube[:3])
            report["inside_black_box"] = bool(placement_valid)
        else:
            report["place_height_error_m"] = abs(float(final_cube[2] - cube_pose[2]))
            placement_valid = report["place_error_m"] <= 0.035 and report["place_height_error_m"] <= 0.025
        report["return_home_error_rad"] = float(np.max(np.abs(final_state[:7] - initial[:7])))
        report["released"] = len(_finger_contacts(model, data, cube)) == 0
        report["full_flow_valid"] = bool(
            placement_valid
            and report["return_home_error_rad"] <= 0.03
            and report["released"]
        )
    else:
        report["full_flow_valid"] = True
    report["valid"] = bool(
        report["success"]
        and report["max_lift_height_m"] + 1e-9 >= evaluator.min_lift_m
        and report["hold_duration_s"] + 1e-9 >= evaluator.required_hold_s
        and report["tracking_error_rad"]["p95"] <= max_tracking_error_rad
        and report["scene_sha256"] == report["expected_scene_sha256"]
        and report["full_flow_valid"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--episode", default="0", help="episode id, or 'all'")
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata = yaml.safe_load((args.dataset / "metadata.yaml").read_text())
    entries = metadata.get("episodes", [])
    if args.episode != "all":
        entries = [entry for entry in entries if str(entry["id"]) == args.episode]
    if not entries:
        raise SystemExit(f"no matching episode: {args.episode}")

    reports = [
        replay_episode(
            args.dataset,
            args.scene,
            entry,
            max_tracking_error_rad=args.max_tracking_error_rad,
            task_contract=metadata.get("task_contract"),
        )
        for entry in tqdm(
            entries,
            desc="Strict replay",
            unit="episode",
            dynamic_ncols=True,
        )
    ]
    output = {
        "dataset": str(args.dataset.resolve()),
        "episodes_requested": len(entries),
        "episodes_valid": sum(report["valid"] for report in reports),
        "valid": all(report["valid"] for report in reports),
        "episodes": reports,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if output["valid"] else 2)


if __name__ == "__main__":
    main()
