#!/usr/bin/env python3
"""Summarize ACT shadow inference timing and prediction stability."""

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def describe(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("log contains no records")
    if any(len(row["right_action_rad"]) != 8 or len(row["right_state_rad"]) != 8 for row in rows):
        parser.error("expected eight right arm and gripper positions per record")
    result = {
        "samples": len(rows),
        "sequence_span": [rows[0]["sequence"], rows[-1]["sequence"]],
        "inference_ms": describe([row["inference_ms"] for row in rows]),
        "observation_age_ms": describe([row["observation_age_ms"] for row in rows]),
        "source_age_ms": {
            key: describe([row["source_ages_ms"][key] for row in rows])
            for key in ("state", "ceiling", "wrist_right")
        },
        "maximum_absolute_right_delta_rad": [
            round(max(abs(row["right_action_rad"][joint] - row["right_state_rad"][joint]) for row in rows), 5)
            for joint in range(8)
        ],
        "maximum_successive_right_action_jump_rad": [
            round(max(
                (abs(b["right_action_rad"][joint] - a["right_action_rad"][joint]) for a, b in zip(rows, rows[1:])),
                default=0.0,
            ), 5)
            for joint in range(8)
        ],
    }
    if all("right_chunk_rad" in row for row in rows):
        result["chunk_actions"] = len(rows[0]["right_chunk_rad"])
        result["maximum_chunk_step_rad"] = [
            round(max(row["maximum_chunk_step_rad"][joint] for row in rows), 5)
            for joint in range(8)
        ]
        result["maximum_chunk_implied_velocity_rad_s"] = [
            round(value * 30.0, 3) for value in result["maximum_chunk_step_rad"]
        ]
        result["maximum_chunk_from_measured_rad"] = [
            round(max(abs(action[joint] - row["right_state_rad"][joint])
                      for row in rows for action in row["right_chunk_rad"]), 5)
            for joint in range(8)
        ]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
