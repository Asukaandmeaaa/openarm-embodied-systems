#!/usr/bin/env python3
"""Fully traverse a local LeRobot v3 OpenArm dataset and validate its contract."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


DEFAULT_PROMPT = "Pick up the orange cube with the right arm."
CAMERAS = ("observation.images.ceiling", "observation.images.wrist_right")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repo-id", default="local/openarm_cube_pick_expert_pilot_v3_g4_frozen")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="expected task prompt for every frame",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=1,
        help="minimum number of episodes required for this validation gate",
    )
    parser.add_argument(
        "--state-normalization",
        choices=("identity", "mean_std"),
        default="mean_std",
        help="normalization planned for ACT training",
    )
    parser.add_argument(
        "--low-variance-std",
        type=float,
        default=1e-4,
        help="state-axis standard deviation below which mean/std is unsafe",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.min_episodes < 1:
        parser.error("--min-episodes must be at least 1")
    if args.low_variance_std <= 0:
        parser.error("--low-variance-std must be positive")

    dataset = LeRobotDataset(args.repo_id, root=args.dataset)
    errors: list[str] = []
    warnings: list[str] = []
    episode_counts: Counter[int] = Counter()
    previous: dict[int, tuple[int, float]] = {}
    action_min = np.full(16, np.inf)
    action_max = np.full(16, -np.inf)
    state_min = np.full(16, np.inf)
    state_max = np.full(16, -np.inf)
    state_sum = np.zeros(16, dtype=np.float64)
    state_square_sum = np.zeros(16, dtype=np.float64)
    state_count = 0
    image_std_min = {key: math_inf for key in CAMERAS}

    for index in tqdm(range(len(dataset)), desc="Validating frames", unit="frame"):
        sample = dataset[index]
        action = sample["action"].numpy()
        state = sample["observation.state"].numpy()
        episode = int(sample["episode_index"])
        frame = int(sample["frame_index"])
        timestamp = float(sample["timestamp"])
        task = sample["task"]
        if action.shape != (16,) or state.shape != (16,):
            errors.append(f"frame {index}: invalid state/action shapes")
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            errors.append(f"frame {index}: state/action contains NaN/Inf")
        if task != args.prompt:
            errors.append(f"frame {index}: task mismatch {task!r}")
        if episode in previous:
            old_frame, old_timestamp = previous[episode]
            if frame != old_frame + 1 or timestamp <= old_timestamp:
                errors.append(f"frame {index}: non-monotonic episode frame/timestamp")
        elif frame != 0:
            errors.append(f"episode {episode}: first frame is {frame}, expected 0")
        previous[episode] = (frame, timestamp)
        episode_counts[episode] += 1
        action_min = np.minimum(action_min, action)
        action_max = np.maximum(action_max, action)
        state_min = np.minimum(state_min, state)
        state_max = np.maximum(state_max, state)
        state64 = state.astype(np.float64)
        state_sum += state64
        state_square_sum += state64 * state64
        state_count += 1
        for key in CAMERAS:
            image = sample[key].numpy()
            if image.shape != (3, 600, 960) or not np.isfinite(image).all():
                errors.append(f"frame {index}: invalid {key}")
            image_std_min[key] = min(image_std_min[key], float(image.std()))

    if len(episode_counts) != dataset.num_episodes:
        errors.append("traversed episode count does not match dataset metadata")
    if len(episode_counts) < args.min_episodes:
        errors.append(
            f"episodes {len(episode_counts)} < required {args.min_episodes}; "
            "the dataset is only suitable for smoke/memorization testing"
        )
    if float((action_max - action_min).max()) < 1e-3:
        errors.append("action is effectively constant")
    if float((state_max - state_min).max()) < 1e-3:
        errors.append("state is effectively constant")
    if action_min[7] < -0.7855 or action_max[7] > 1e-4:
        errors.append("right gripper action violates [-0.7854, 0]")
    if action_min[15] < -1e-4 or action_max[15] > 0.7855:
        errors.append("left gripper action violates [0, 0.7854]")
    for key, minimum in image_std_min.items():
        if minimum < 1e-3:
            errors.append(f"{key} contains a near-constant decoded frame")

    action_ptp = action_max - action_min
    state_ptp = state_max - state_min
    state_mean = state_sum / max(1, state_count)
    state_variance = np.maximum(
        state_square_sum / max(1, state_count) - state_mean * state_mean,
        0.0,
    )
    state_std = np.sqrt(state_variance)
    low_variance_state_axes = np.flatnonzero(state_std < args.low_variance_std)
    constant_action_axes = np.flatnonzero(action_ptp < 1e-8)
    if len(low_variance_state_axes):
        message = (
            "near-constant state axes can become extreme out-of-distribution "
            "inputs under mean/std normalization"
        )
        if args.state_normalization == "mean_std":
            errors.append(
                f"{message}: axes={low_variance_state_axes.tolist()}; "
                "use --policy.normalization_mapping with STATE=IDENTITY"
            )
        else:
            warnings.append(
                f"{message}: axes={low_variance_state_axes.tolist()}; "
                "STATE=IDENTITY avoids division by these standard deviations"
            )
    if len(constant_action_axes):
        warnings.append(
            "constant action axes should be held outside the learned policy or "
            "explicitly audited in the action normalizer"
        )

    report = {
        "dataset": str(args.dataset.resolve()),
        "repo_id": args.repo_id,
        "lerobot_codebase_version": dataset.meta.info.codebase_version,
        "frames_traversed": len(dataset),
        "episodes_traversed": dict(sorted(episode_counts.items())),
        "camera_keys": list(CAMERAS),
        "image_std_min": image_std_min,
        "action_ptp_max": float(action_ptp.max()),
        "state_ptp_max": float(state_ptp.max()),
        "action_ptp_per_axis": action_ptp.tolist(),
        "state_ptp_per_axis": state_ptp.tolist(),
        "state_std_per_axis": state_std.tolist(),
        "low_variance_state_std_threshold_rad": args.low_variance_std,
        "low_variance_state_axes": low_variance_state_axes.tolist(),
        "requested_state_normalization": args.state_normalization,
        "recommended_state_normalization": "identity" if len(low_variance_state_axes) else args.state_normalization,
        "required_min_episodes": args.min_episodes,
        "constant_action_axes": constant_action_axes.tolist(),
        "warnings": warnings,
        "errors": errors[:100],
        "valid": not errors,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if report["valid"] else 2)


math_inf = float("inf")


if __name__ == "__main__":
    main()
