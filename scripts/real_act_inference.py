#!/usr/bin/env python3
"""Run ACT on live snapshots and publish atomic action chunks.

This process is inference-only.  It never imports ROS and never commands the
robot.  A ROS executor on the Nano host can consume ``latest_action.json``
from the same shared directory.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


ACTION_ORDER = [
    *[f"openarm_right_joint{i}" for i in range(1, 8)],
    "openarm_right_finger_joint1",
    *[f"openarm_left_joint{i}" for i in range(1, 8)],
    "openarm_left_finger_joint1",
]


def image_tensor(path: Path, shape) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    _, height, width = shape
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)


def write_atomic(path: Path, payload: dict) -> None:
    """Publish a complete JSON document in one rename operation."""
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temp, path)


def finite_vector(values, size: int, label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"invalid {label}: expected {size} finite values")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", type=Path, default=Path("/live"))
    parser.add_argument("--output", type=Path, default=Path("/live/latest_action.json"))
    parser.add_argument(
        "--chunk-actions",
        type=int,
        default=10,
        help="number of sequential ACT actions to publish per fresh observation (1-10)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="number of chunks to publish; 0 runs until stopped",
    )
    parser.add_argument("--max-age", type=float, default=0.5)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not 1 <= args.chunk_actions <= 10:
        parser.error("chunk-actions must be between 1 and 10")
    if args.samples < 0 or args.max_age <= 0 or args.idle_timeout <= 0:
        parser.error("samples must be non-negative; max-age and idle-timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = ACTPolicy.from_pretrained(str(args.checkpoint)).eval()
    image_shapes = {key: tuple(policy.config.input_features[key].shape) for key in
                    ('observation.images.ceiling', 'observation.images.wrist_right')}
    if policy.config.temporal_ensemble_coeff is not None:
        raise ValueError("this action-chunk publisher requires temporal ensembling to be disabled")
    if args.chunk_actions > policy.config.n_action_steps:
        raise ValueError(
            f"chunk-actions={args.chunk_actions} exceeds model n_action_steps={policy.config.n_action_steps}"
        )
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    # Keep ACT's action queue alive across observations.  Resetting for every
    # snapshot would turn this into independent one-step shadow predictions.
    policy.reset()
    print(
        "ACT_INFERENCE_READY; action chunks only, no ROS or control connection "
        f"chunk_actions={args.chunk_actions} output={args.output}",
        flush=True,
    )

    last_sequence = -1
    published = 0
    last_success = time.monotonic()
    last_published = float("-inf")
    # Match the official executor horizon: ten samples at 30 Hz cover the
    # interval between the first and last sample, (10 - 1) / 30 seconds.
    chunk_period_s = max(0.0, (args.chunk_actions - 1) / 30.0)
    # Never expose an action file left behind by an earlier process/session.
    args.output.unlink(missing_ok=True)
    with torch.inference_mode():
        while args.samples == 0 or published < args.samples:
            if time.monotonic() - last_success > args.idle_timeout:
                raise TimeoutError(
                    "no fresh successful inference within idle timeout; check observation bridge"
                )
            manifest = args.input / "latest.json"
            if not manifest.exists():
                time.sleep(0.05)
                continue
            # A chunk represents 10 / 30 seconds of targets by default.
            # The observation bridge currently runs at 5 Hz, so skip interim
            # snapshots instead of publishing overlapping action chunks.
            if time.monotonic() - last_published < chunk_period_s:
                time.sleep(0.02)
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
                if sample["joint_order"] != ACTION_ORDER:
                    raise ValueError("observation joint_order differs from model action order")
                state = finite_vector(sample["state"], 16, "state")
                observation = {
                    "observation.state": torch.tensor(state, dtype=torch.float32),
                    "observation.images.ceiling": image_tensor(
                        args.input / sample["images"]["ceiling"],
                        image_shapes['observation.images.ceiling'],
                    ),
                    "observation.images.wrist_right": image_tensor(
                        args.input / sample["images"]["wrist_right"],
                        image_shapes['observation.images.wrist_right'],
                    ),
                }
                torch.cuda.synchronize()
                started = time.perf_counter()
                processed = pre(observation)
                # Match the official OpenArm policy server: infer one complete
                # ACT chunk and postprocess that chunk as one batch. Repeated
                # select_action calls exercise LeRobot's action queue and are
                # a different control contract.
                predicted = policy.predict_action_chunk(processed)
                if predicted.ndim != 3 or predicted.shape[0] != 1:
                    raise ValueError(
                        f"unexpected ACT chunk shape: {tuple(predicted.shape)}"
                    )
                chunk_length = int(predicted.shape[1])
                action_dim = int(predicted.shape[2])
                postprocessed = post(
                    predicted.reshape(chunk_length, action_dim)
                ).reshape(1, chunk_length, action_dim)
                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - started) * 1000.0
                action_vectors = [
                    finite_vector(action.tolist(), 16, "ACT action")
                    for action in postprocessed[0][: args.chunk_actions]
                ]
                if len(action_vectors) != args.chunk_actions:
                    raise ValueError(
                        f"policy returned {len(action_vectors)} actions, "
                        f"requested {args.chunk_actions}"
                    )
                published_at = time.monotonic()
                payload = {
                    "schema": "openarm.act_action_chunk.v1",
                    "published_monotonic_s": published_at,
                    "valid_until_monotonic_s": published_at + 0.5,
                    "action_sequence": published + 1,
                    "source_observation_sequence": sequence,
                    "source_observation_monotonic_s": float(sample["monotonic_s"]),
                    "source_observation_age_ms": round(age * 1000.0, 1),
                    "inference_ms": round(inference_ms, 1),
                    "inference_api": "predict_action_chunk",
                    "chunk_size": len(action_vectors),
                    "action_hz": 30.0,
                    "action_order": ACTION_ORDER,
                    "actions": action_vectors,
                    "right_actions": [action[:8] for action in action_vectors],
                }
                write_atomic(args.output, payload)
                published += 1
                last_published = time.monotonic()
                last_success = last_published
                print(
                    f"chunk={published} seq={sequence} age={age * 1000:.0f}ms "
                    f"infer={inference_ms:.1f}ms actions={len(action_vectors)}",
                    flush=True,
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                print(f"SKIP snapshot: {error}", flush=True)
                time.sleep(0.05)
    print(f"ACT_INFERENCE_COMPLETE chunks={published} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
