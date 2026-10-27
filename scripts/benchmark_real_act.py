#!/usr/bin/env python3
"""Benchmark the real ACT checkpoint without ROS or robot commands."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA PyTorch is required for this Jetson GPU benchmark")
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory missing: {args.checkpoint}")

    load_start = time.perf_counter()
    policy = ACTPolicy.from_pretrained(str(args.checkpoint)).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start
    observation = {
        "observation.state": torch.zeros(16, dtype=torch.float32),
        "observation.images.ceiling": torch.rand(3, 360, 640),
        "observation.images.wrist_right": torch.rand(3, 360, 640),
    }
    durations_ms = []
    with torch.inference_mode():
        for index in range(args.warmup + args.iterations):
            policy.reset()  # Force a new ACT inference for each timed sample.
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = post(policy.select_action(pre(observation)))
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            if action.numel() != 16 or not torch.isfinite(action).all():
                raise RuntimeError("ACT returned an invalid 16-dimensional action")
            if index >= args.warmup:
                durations_ms.append(elapsed_ms)

    ordered = sorted(durations_ms)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "load_s": round(load_s, 3),
        "iterations": len(durations_ms),
        "mean_ms": round(statistics.mean(durations_ms), 2),
        "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 2),
        "max_ms": round(max(durations_ms), 2),
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "scope": "Synthetic observation; model and processors only, no camera or ROS overhead",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
