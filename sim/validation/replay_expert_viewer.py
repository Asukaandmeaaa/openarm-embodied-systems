#!/usr/bin/env python3
"""Visually replay recorded OpenArm expert actions in the original MuJoCo scene."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from openarm_mujoco.v2 import JointResolver


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
DEFAULT_SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
DEFAULT_EPISODES = (0, 1, 2, 3)


def load_actions(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = None
    sides = []
    for side in ("right", "left"):
        path = episode_dir / "action" / "arms" / side / "state.parquet"
        table = pq.read_table(path, columns=["timestamp", "qpos"])
        side_timestamps = table["timestamp"].cast(pa.int64()).to_numpy()
        actions = np.asarray(table["qpos"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"{path}: expected [N,8], got {actions.shape}")
        if timestamps is None:
            timestamps = side_timestamps
        elif not np.array_equal(timestamps, side_timestamps):
            raise ValueError(f"episode action timestamps differ between {side} and the other arm")
        sides.append(actions)
    assert timestamps is not None
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("action timestamps must be strictly increasing")
    return timestamps, np.concatenate(sides, axis=1)


def parse_episode_ids(value: str) -> list[int]:
    try:
        episode_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episodes must be comma-separated integers") from exc
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise argparse.ArgumentTypeError("episodes must be a non-empty list without duplicates")
    return episode_ids


def configure_camera(model: mujoco.MjModel, viewer, camera: str) -> None:
    if viewer is None or camera == "free":
        return
    if camera == "both":
        camera = "camera_ceiling"
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"camera not found in model: {camera}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def set_wrist_picture_in_picture(viewer, episode_dir: Path, timestamp_ns: int) -> None:
    """Overlay the synchronized recorded wrist RGB frame in the MuJoCo window."""
    if viewer is None:
        return
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--camera both requires opencv-python in the active environment") from exc
    path = episode_dir / "cameras" / "wrist_right" / f"{timestamp_ns}.jpeg"
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"missing recorded camera frame: {path}")
    viewport = viewer.viewport
    if viewport is None:
        return
    width = min(420, max(240, int(viewport.width * 0.32)))
    height = int(round(width * frame.shape[0] / frame.shape[1]))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 3)
    cv2.putText(frame, "WRIST RIGHT", (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2, cv2.LINE_AA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    margin = 14
    pip_viewport = mujoco.MjrRect(
        viewport.left + margin,
        viewport.bottom + viewport.height - height - margin,
        width,
        height,
    )
    viewer.set_images((pip_viewport, rgb))


def reset_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: JointResolver,
    entry: dict,
) -> int:
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    initial = np.asarray(entry["initial_qpos"], dtype=np.float64)
    cube_pose = np.asarray(entry["objects"]["orange_cube"]["pose"], dtype=np.float64)
    if initial.shape != (16,) or cube_pose.shape != (7,):
        raise ValueError("episode metadata has invalid initial_qpos or cube pose")
    mapper.set_qpos(data.qpos, initial[:8], "right")
    mapper.set_qpos(data.qpos, initial[8:], "left")

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
    if entry["objects"]["orange_cube"].get("color") == "blue":
        cube_geoms = np.flatnonzero(model.geom_bodyid == cube)
        model.geom_rgba[cube_geoms] = np.array([0.04, 0.18, 0.90, 1.0])
    cube_joint = int(model.body_jntadr[cube])
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    data.qpos[cube_qadr : cube_qadr + 7] = cube_pose
    data.qvel.fill(0)
    data.time = float(entry["phase_timestamps"]["MOVE_PREGRASP"])
    mujoco.mj_forward(model, data)
    return cube_qadr


def replay_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: JointResolver,
    viewer,
    dataset: Path,
    entry: dict,
    speed: float,
    sync_hz: float,
) -> bool:
    episode_id = str(entry["id"])
    episode_dir = dataset / "episodes" / episode_id
    timestamps, actions = load_actions(episode_dir)
    cube_qadr = reset_episode(model, data, mapper, entry)
    configure_camera(model, viewer, replay_episode.camera)

    phase_events = sorted(
        (float(timestamp), name)
        for name, timestamp in entry.get("phase_timestamps", {}).items()
        if float(timestamp) >= data.time
    )
    next_phase = 0
    previous_timestamp_ns = int(round(data.time * 1e9))
    wall_start = time.monotonic()
    simulation_start = float(data.time)
    next_sync = float(data.time)

    print(
        f"episode={episode_id} seed={entry.get('seed')} frames={len(actions)} "
        f"cube_xy=({data.qpos[cube_qadr]:.3f}, {data.qpos[cube_qadr + 1]:.3f})"
    )
    for timestamp_ns, action in zip(timestamps, actions, strict=True):
        if viewer is not None and not viewer.is_running():
            return False
        timestamp_ns = int(timestamp_ns)
        requested_steps = (timestamp_ns - previous_timestamp_ns) / (model.opt.timestep * 1e9)
        steps = int(round(requested_steps))
        if steps < 1 or abs(requested_steps - steps) > 1e-3:
            raise ValueError(f"episode {episode_id}: timestamp interval is off the physics grid")
        mapper.set_ctrl(data.ctrl, action[:8], "right")
        mapper.set_ctrl(data.ctrl, action[8:], "left")
        for _ in range(steps):
            mujoco.mj_step(model, data)
            while next_phase < len(phase_events) and data.time >= phase_events[next_phase][0]:
                print(f"  t={data.time - simulation_start:6.2f}s phase={phase_events[next_phase][1]}")
                next_phase += 1
            if viewer is not None and data.time >= next_sync:
                viewer.sync()
                target_wall = wall_start + (data.time - simulation_start) / speed
                remaining = target_wall - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                next_sync += 1.0 / sync_hz
        if replay_episode.camera == "both":
            set_wrist_picture_in_picture(viewer, episode_dir, timestamp_ns)
        previous_timestamp_ns = timestamp_ns

    if viewer is not None:
        viewer.sync()
    lift = float(data.qpos[cube_qadr + 2] - np.asarray(entry["objects"]["orange_cube"]["pose"])[2])
    print(f"  completed: final cube lift={lift:.3f} m")
    return viewer is None or viewer.is_running()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--episodes",
        type=parse_episode_ids,
        default=list(DEFAULT_EPISODES),
        help="comma-separated episode IDs (default: 0,1,2,3; strict-replay-valid pilot episodes)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--pause", type=float, default=1.5, help="wall-clock pause between episodes")
    parser.add_argument("--sync-hz", type=float, default=60.0, help="viewer refresh frequency")
    parser.add_argument(
        "--camera",
        choices=("free", "camera_ceiling", "camera_wrist_right", "both"),
        default="free",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.speed <= 0 or args.pause < 0 or args.sync_hz <= 0:
        parser.error("speed and sync-hz must be positive; pause must be non-negative")

    metadata = yaml.safe_load((args.dataset / "metadata.yaml").read_text(encoding="utf-8"))
    entries_by_id = {int(entry["id"]): entry for entry in metadata.get("episodes", [])}
    missing = [episode_id for episode_id in args.episodes if episode_id not in entries_by_id]
    if missing:
        parser.error(f"episodes not present in dataset metadata: {missing}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError(f"expected 1000 Hz physics, got timestep={model.opt.timestep}")
    data = mujoco.MjData(model)
    mapper = JointResolver(model)
    replay_episode.camera = args.camera
    if args.camera == "both" and args.headless:
        parser.error("--camera both requires a desktop window (do not combine it with --headless)")
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    with context as viewer:
        while True:
            for episode_id in args.episodes:
                if not replay_episode(
                    model,
                    data,
                    mapper,
                    viewer,
                    args.dataset,
                    entries_by_id[episode_id],
                    args.speed,
                    args.sync_hz,
                ):
                    return
                if viewer is not None and args.pause:
                    deadline = time.monotonic() + args.pause
                    while viewer.is_running() and time.monotonic() < deadline:
                        viewer.sync()
                        time.sleep(min(1.0 / args.sync_hz, deadline - time.monotonic()))
            if not args.loop:
                break


if __name__ == "__main__":
    main()
