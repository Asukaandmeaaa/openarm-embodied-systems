#!/usr/bin/env python3
"""Repeat one deterministic expert seed under the current frozen sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_cube_pick_expert.collect import Collector
from openarm_cube_pick_expert.config import PARAMS, SCENE, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scratch = args.output.parent / ".g1_fixed_seed_scratch"
    collector = Collector(scratch, args.seed, False)
    attempts = []
    try:
        for repetition in range(args.repetitions):
            # index=0 deliberately keeps episode_seed == base_seed.
            success, details = collector.run_episode(0, record=False)
            attempts.append({
                "repetition": repetition,
                "seed": details["seed"],
                "success": bool(success),
                "success_reason": details.get("success_reason"),
                "fail_reason": details.get("fail_reason"),
                "max_lift_height_m": details.get("max_lift_height_m", 0.0),
                "hold_duration_s": details.get("hold_duration_s", 0.0),
                "hold_window_start_s": details.get("hold_window_start_s"),
                "hold_window_end_s": details.get("hold_window_end_s"),
                "selected_candidate": details.get("selected_candidate"),
                "action_guard": details.get("action_guard", {}),
            })
    finally:
        collector.close()

    valid = all(
        item["success"]
        and item["seed"] == args.seed
        and item["max_lift_height_m"] + 1e-9 >= 0.05
        and item["hold_duration_s"] + 1e-9 >= 1.0
        for item in attempts
    )
    report = {
        "seed": args.seed,
        "repetitions": args.repetitions,
        "successes": sum(item["success"] for item in attempts),
        "valid": valid,
        "scene_sha256": sha256(SCENE),
        "params_sha256": sha256(PARAMS),
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
