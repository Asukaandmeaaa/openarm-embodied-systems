#!/usr/bin/env python3
"""Validate OpenArm cube-pick pilot episodes before LeRobot conversion."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image


PROMPTS = {
    "Pick up the orange cube with the right arm.",
    "Pick up the blue cube, hold it for 2 seconds, place it nearby, and return the right arm home.",
    "Put the blue cube inside the black box, then return both arms home.",
}
CAMERAS = ("ceiling", "wrist_right")


def read_stream(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["timestamp", "qpos"])
    timestamps = table["timestamp"].cast(pa.int64()).to_numpy()
    values = np.asarray(table["qpos"].to_pylist(), dtype=np.float32)
    return timestamps, values


def check_stream(errors: list[str], label: str, path: Path) -> np.ndarray:
    if not path.exists():
        errors.append(f"{label}: missing {path}")
        return np.empty((0, 8), dtype=np.float32)
    timestamps, values = read_stream(path)
    if values.ndim != 2 or values.shape[1] != 8:
        errors.append(f"{label}: expected [N,8], got {values.shape}")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        errors.append(f"{label}: timestamps are not strictly increasing")
    if not np.isfinite(values).all():
        errors.append(f"{label}: contains NaN/Inf")
    return values


def check_camera(errors: list[str], episode: Path, name: str) -> int:
    directory = episode / "cameras" / name
    files = sorted(
        [*directory.glob("*.jpeg"), *directory.glob("*.jpg")],
        key=lambda path: int(path.stem),
    )
    if not files:
        errors.append(f"camera {name}: no JPEG frames")
        return 0
    timestamps = [int(path.stem) for path in files]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        errors.append(f"camera {name}: timestamps are not strictly increasing")
    for path in (files[0], files[len(files) // 2], files[-1]):
        with Image.open(io.BytesIO(path.read_bytes())) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != (600, 960, 3):
            errors.append(f"camera {name}: {path.name} has shape {rgb.shape}")
        if float(rgb.std()) < 1.0:
            errors.append(f"camera {name}: {path.name} is near-constant/black")
    return len(files)


def validate_episode(root: Path, entry: dict) -> dict:
    episode = root / "episodes" / str(entry["id"])
    errors: list[str] = []
    streams = {}
    for kind in ("action", "obs"):
        for side in ("right", "left"):
            label = f"{kind}/{side}"
            streams[label] = check_stream(
                errors, label, episode / kind / "arms" / side / "state.parquet"
            )

    right_action = streams["action/right"]
    right_obs = streams["obs/right"]
    if len(right_action) and float(np.ptp(right_action, axis=0).max()) < 1e-3:
        errors.append("right action is effectively constant")
    if len(right_obs) and float(np.ptp(right_obs, axis=0).max()) < 1e-3:
        errors.append("right observation does not move")
    if len(right_action):
        if np.any(right_action[:, 7] < -0.7855) or np.any(right_action[:, 7] > 1e-4):
            errors.append("right gripper is outside [-0.7854, 0]")
    left_action = streams["action/left"]
    if len(left_action):
        if np.any(left_action[:, 7] < -1e-4) or np.any(left_action[:, 7] > 0.7855):
            errors.append("left gripper is outside [0, 0.7854]")

    camera_frames = {name: check_camera(errors, episode, name) for name in CAMERAS}
    extra_cameras = sorted(
        path.name for path in (episode / "cameras").glob("*") if path.name not in CAMERAS
    )
    if extra_cameras:
        errors.append(f"unexpected cameras: {extra_cameras}")
    if entry.get("seed") is None or "orange_cube" not in entry.get("objects", {}):
        errors.append("missing deterministic seed/orange_cube initial pose metadata")
    return {
        "id": str(entry["id"]),
        "success": bool(entry.get("success")),
        "valid": not errors,
        "errors": errors,
        "samples": {key: len(value) for key, value in streams.items()},
        "camera_frames": camera_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-success", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    metadata_path = args.dataset / "metadata.yaml"
    if not metadata_path.exists():
        report = {
            "dataset": str(args.dataset.resolve()),
            "valid_success_episodes": 0,
            "required_success_episodes": args.min_success,
            "valid": False,
            "errors": [f"missing dataset metadata: {metadata_path}"],
            "episodes": [],
        }
        output = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output, encoding="utf-8")
        print(output, end="")
        raise SystemExit(1)
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    errors = []
    tasks = metadata.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("prompt") not in PROMPTS:
        errors.append("dataset task prompt is not a supported expert contract")
    camera_keys = set(
        metadata.get("equipment", {})
        .get("perceptions", {})
        .get("cameras", {})
    )
    if camera_keys != set(CAMERAS):
        errors.append(f"metadata cameras must be exactly {CAMERAS}, got {sorted(camera_keys)}")

    episode_reports = [validate_episode(args.dataset, e) for e in metadata.get("episodes", [])]
    valid_success = sum(e["valid"] and e["success"] for e in episode_reports)
    if valid_success < args.min_success:
        errors.append(f"valid successful pilots {valid_success} < required {args.min_success}")
    report = {
        "dataset": str(args.dataset.resolve()),
        "valid_success_episodes": valid_success,
        "required_success_episodes": args.min_success,
        "valid": not errors and all(e["valid"] for e in episode_reports if e["success"]),
        "errors": errors,
        "episodes": episode_reports,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
