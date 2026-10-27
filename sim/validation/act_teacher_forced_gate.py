#!/usr/bin/env python3
"""Measure an ACT checkpoint against exact LeRobot action chunks.

This is a diagnostic gate, not a rollout.  It feeds observations recorded in
the dataset to the policy and compares the resulting physical-unit action
chunk with the corresponding unnormalized labels.  It separates supervised
fit from closed-loop covariate shift and executor effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/openarm_cube_pick")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    policy = get_policy_class("act").from_pretrained(args.checkpoint).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    fps = 30
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset,
        episodes=[args.episode],
        delta_timestamps={"action": [i / fps for i in range(policy.config.chunk_size)]},
        video_backend="torchcodec",
        return_uint8=True,
    )

    episode_positions = [
        i
        for i, value in enumerate(dataset.hf_dataset["episode_index"])
        if int(value) == args.episode
    ]
    selected = episode_positions[:: args.stride]
    if episode_positions and episode_positions[-1] not in selected:
        selected.append(episode_positions[-1])

    valid_abs_errors: list[np.ndarray] = []
    first_abs_errors: list[np.ndarray] = []
    samples: list[dict] = []
    for index in selected:
        item = dataset[index]
        batch: dict[str, torch.Tensor] = {
            "observation.state": item["observation.state"].unsqueeze(0)
        }
        for key in policy.config.image_features:
            image = item[key]
            if image.dtype == torch.uint8:
                image = image.float().div(255.0)
            batch[key] = image.unsqueeze(0)

        policy.reset()
        with torch.inference_mode():
            predicted = postprocessor(
                policy.predict_action_chunk(preprocessor(batch))
            ).numpy()[0]
        target = item["action"].numpy()
        valid = ~item["action_is_pad"].numpy().astype(bool)
        error = np.abs(predicted - target)
        valid_abs_errors.append(error[valid])
        first_abs_errors.append(error[0])
        samples.append(
            {
                "dataset_index": index,
                "frame_index": int(item["frame_index"]),
                "valid_action_steps": int(valid.sum()),
                "valid_mae_rad": float(error[valid].mean()),
                "valid_max_error_rad": float(error[valid].max()),
                "first_action_mae_rad": float(error[0].mean()),
                "predicted_chunk_axis_range_max_rad": float(
                    np.ptp(predicted, axis=0).max()
                ),
                "target_chunk_axis_range_max_rad": float(np.ptp(target[valid], axis=0).max()),
            }
        )

    all_valid = np.concatenate(valid_abs_errors, axis=0)
    all_first = np.stack(first_abs_errors)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_model_sha256": sha256(args.checkpoint / "model.safetensors"),
        "dataset": str(args.dataset.resolve()),
        "episode": args.episode,
        "fps": fps,
        "chunk_size": policy.config.chunk_size,
        "sample_stride": args.stride,
        "sample_count": len(samples),
        "valid_action_element_count": int(all_valid.size),
        "metrics": {
            "chunk_mae_rad": float(all_valid.mean()),
            "chunk_p95_abs_error_rad": float(np.quantile(all_valid, 0.95)),
            "chunk_max_abs_error_rad": float(all_valid.max()),
            "first_action_mae_rad": float(all_first.mean()),
            "first_action_max_abs_error_rad": float(all_first.max()),
        },
        "samples": samples,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
