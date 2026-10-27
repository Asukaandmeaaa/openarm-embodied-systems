#!/usr/bin/env python3
"""Offline ACT checkpoint smoke test; never opens ROS, CAN, or camera devices."""

import argparse
import json
import time
from pathlib import Path

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {args.checkpoint}")

    started = time.perf_counter()
    policy = ACTPolicy.from_pretrained(str(args.checkpoint)).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    load_s = time.perf_counter() - started

    features = policy.config.input_features
    expected_images = {
        key: tuple(features[key].shape)
        for key in ("observation.images.ceiling", "observation.images.wrist_right")
    }
    for key, shape in expected_images.items():
        if len(shape) != 3 or shape[0] != 3 or any(int(v) <= 0 for v in shape):
            raise RuntimeError(f"{key}: invalid CHW image shape {shape}")
    if len(set(expected_images.values())) != 1:
        raise RuntimeError(f"camera feature shapes differ: {expected_images}")
    if tuple(features["observation.state"].shape) != (16,):
        raise RuntimeError("checkpoint does not declare a 16D state")
    if tuple(policy.config.output_features["action"].shape) != (16,):
        raise RuntimeError("checkpoint does not declare a 16D action")

    observation = {
        "observation.state": torch.zeros(16, dtype=torch.float32),
        "observation.images.ceiling": torch.zeros(*expected_images["observation.images.ceiling"]),
        "observation.images.wrist_right": torch.zeros(*expected_images["observation.images.wrist_right"]),
    }
    device = policy.config.device
    if device == "cuda":
        torch.cuda.synchronize()
    with torch.inference_mode():
        started = time.perf_counter()
        processed = preprocessor(observation)
        policy.reset()
        raw = policy.predict_action_chunk(processed)
        if device == "cuda":
            torch.cuda.synchronize()
        predicted_s = time.perf_counter() - started
        n_batch, chunk_size, action_dim = raw.shape
        actions = postprocessor(raw.reshape(n_batch * chunk_size, action_dim))
        actions = actions.reshape(n_batch, chunk_size, action_dim)
        if device == "cuda":
            torch.cuda.synchronize()

    if tuple(actions.shape) != (1, 50, 16):
        raise RuntimeError(f"expected (1, 50, 16), got {tuple(actions.shape)}")
    if not bool(torch.isfinite(actions).all()):
        raise RuntimeError("checkpoint produced non-finite actions")

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "device": str(device),
                "load_s": round(load_s, 3),
                "predict_post_s": round(predicted_s, 3),
                "camera_features": expected_images,
                "state_shape": [16],
                "action_shape": [50, 16],
                "finite": True,
                "network": "disabled by HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
