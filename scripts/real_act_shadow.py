#!/usr/bin/env python3
"""Run ACT on live snapshots and log predictions; never command the robot."""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def image_tensor(path: Path, shape) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    _, height, width = shape
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", type=Path, default=Path("/live"))
    parser.add_argument("--output", type=Path, default=Path("/live/predictions.jsonl"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--chunk-actions", type=int, default=1,
                        help="Log this many queued ACT actions from each fresh observation (1-10)")
    parser.add_argument("--max-age", type=float, default=0.5)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.samples < 1 or not 1 <= args.chunk_actions <= 10 or args.max_age <= 0 or args.idle_timeout <= 0:
        parser.error("samples, max-age, and idle-timeout must be positive; chunk-actions must be 1-10")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    policy = ACTPolicy.from_pretrained(str(args.checkpoint)).eval()
    image_shapes = {key: tuple(policy.config.input_features[key].shape) for key in
                    ('observation.images.ceiling', 'observation.images.wrist_right')}
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    print("ACT_SHADOW_READY; read-only inference, no ROS or control connection", flush=True)
    last_sequence = -1
    count = 0
    last_success = time.monotonic()
    with args.output.open("a") as log, torch.inference_mode():
        while count < args.samples:
            if time.monotonic() - last_success > args.idle_timeout:
                raise TimeoutError("no fresh successful inference within idle timeout; check ROS streams")
            manifest = args.input / "latest.json"
            if not manifest.exists():
                time.sleep(0.05)
                continue
            try:
                sample = json.loads(manifest.read_text())
                sequence = int(sample["sequence"])
                if sequence <= last_sequence:
                    time.sleep(0.02)
                    continue
                last_sequence = sequence
                age = time.monotonic() - float(sample["monotonic_s"])
                if age < 0 or age > args.max_age:
                    print(f"SKIP stale snapshot {sequence}: {age:.3f}s", flush=True)
                    continue
                state = sample["state"]
                if len(state) != 16 or not all(math.isfinite(float(x)) for x in state):
                    raise ValueError("invalid 16-dimensional joint state")
                observation = {
                    "observation.state": torch.tensor(state, dtype=torch.float32),
                    "observation.images.ceiling": image_tensor(args.input / sample["images"]["ceiling"], image_shapes['observation.images.ceiling']),
                    "observation.images.wrist_right": image_tensor(args.input / sample["images"]["wrist_right"], image_shapes['observation.images.wrist_right']),
                }
                policy.reset()  # Every shadow prediction uses the latest observation.
                torch.cuda.synchronize()
                start = time.perf_counter()
                processed = pre(observation)
                actions = [post(policy.select_action(processed)).cpu().reshape(-1)
                           for _ in range(args.chunk_actions)]
                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - start) * 1000
                if any(action.shape != (16,) or not torch.isfinite(action).all() for action in actions):
                    raise ValueError("invalid ACT action")
                predicted = actions[0].tolist()
                record = {
                    "sequence": sequence,
                    "observation_age_ms": round(age * 1000, 1),
                    "source_ages_ms": {k: round(v * 1000, 1) for k, v in sample["ages_s"].items()},
                    "inference_ms": round(inference_ms, 1),
                    "right_state_rad": state[:8],
                    "right_action_rad": predicted[:8],
                    "right_delta_rad": [round(a - s, 5) for a, s in zip(predicted[:8], state[:8])],
                    "left_action_rad": predicted[8:],
                }
                if args.chunk_actions > 1:
                    chunk = [action.tolist()[:8] for action in actions]
                    record["right_chunk_rad"] = chunk
                    record["maximum_chunk_step_rad"] = [
                        round(max(abs(b[j] - a[j]) for a, b in zip(chunk, chunk[1:])), 5)
                        for j in range(8)
                    ]
                log.write(json.dumps(record) + "\n")
                log.flush()
                count += 1
                last_success = time.monotonic()
                print(f"sample={count} seq={sequence} age={age*1000:.0f}ms infer={inference_ms:.1f}ms right_delta={record['right_delta_rad']}", flush=True)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                print(f"SKIP snapshot: {error}", flush=True)
                time.sleep(0.05)
    print(f"ACT_SHADOW_COMPLETE samples={count} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
