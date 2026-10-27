#!/usr/bin/env python3
"""Summarize act_executor_diagnostics.jsonl without ROS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _position_span(rows: list[list[float]]) -> list[float] | None:
    if not rows:
        return None
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        return None
    return np.round(np.ptp(values, axis=0), 5).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [event["monotonic_s"] for event in events if event.get("event") == "start_requested"]
    start = starts[-1] if starts else None
    stops = [event["monotonic_s"] for event in events if event.get("event") == "cleanup_requested" and (start is None or event["monotonic_s"] >= start)]
    stop = stops[0] if stops else None
    active = [
        event for event in events
        if (start is None or event.get("monotonic_s", 0) >= start)
        and (stop is None or event.get("monotonic_s", 0) <= stop)
    ]
    received = [event for event in active if event.get("event") == "chunk_received"]
    prepared = [event for event in active if event.get("event") == "chunk_prepared"]
    sent = [event for event in active if event.get("event") in ("goal_send_requested", "trajectory_published")]
    feedback = [event for event in active if event.get("event") == "joint_feedback"]
    states = [event for event in active if event.get("event") == "controller_state"]
    intervals = [float(event["chunk_interval_s"]) for event in received if event.get("chunk_interval_s") is not None]
    holds = [float(event["estimated_previous_goal_hold_s"]) for event in prepared if event.get("estimated_previous_goal_hold_s") is not None]
    errors = {}
    for event in states:
        value = event.get("max_reference_feedback_error")
        if value is not None:
            errors.setdefault(event.get("controller", "unknown"), []).append(float(value))
    jumps = []
    raw_jumps = [
        float(event["raw_first_point_jump_max_rad"])
        for event in received
        if event.get("raw_first_point_jump_max_rad") is not None
    ]
    processed_jumps = [
        float(event["blended_first_point_jump_max_rad"])
        for event in prepared
        if event.get("blended_first_point_jump_max_rad") is not None
    ]
    previous = None
    for event in prepared:
        points = event.get("blended_right_actions")
        if previous is not None and points:
            jumps.append(float(np.max(np.abs(np.asarray(points[0]) - np.asarray(previous[-1])))))
        if points:
            previous = points
    report = {
        "event_count": len(events),
        "active_duration_s": round(stop - start, 3) if start is not None and stop is not None else None,
        "chunk_count": len(received),
        "sent_goal_count": len(sent),
        "chunk_interval_min_s": min(intervals) if intervals else None,
        "chunk_interval_max_s": max(intervals) if intervals else None,
        "estimated_hold_max_s": max(holds) if holds else None,
        "cross_chunk_first_point_jump_max_rad": max(jumps) if jumps else None,
        "raw_cross_chunk_first_point_jump_max_rad": max(raw_jumps) if raw_jumps else None,
        "blended_cross_chunk_first_point_jump_max_rad": max(processed_jumps) if processed_jumps else None,
        "controller_reference_feedback_max_error_rad": {
            key: max(values) for key, values in errors.items()
        },
        "goal_cancel_results": [
            {key: event.get(key) for key in ("role", "confirmed", "error")}
            for event in events if event.get("event") == "goal_cancel_result"
        ],
        "goal_result_status_counts": {
            role: {
                str(status): sum(
                    event.get("role") == role and event.get("status") == status
                    for event in active if event.get("event") == "goal_result"
                )
                for status in sorted({
                    event.get("status") for event in active
                    if event.get("event") == "goal_result" and event.get("role") == role
                }, key=lambda value: (value is None, value))
            }
            for role in ("arm", "gripper")
        },
        "right_goal_target_span_rad": _position_span([
            point for event in sent for point in event.get("blended_right_actions", [])
        ]),
        "controller_reference_span_rad": {
            role: _position_span([
                event["reference_positions"] for event in states
                if event.get("controller") == role and event.get("reference_positions")
            ])
            for role in ("arm", "gripper")
        },
        "controller_feedback_span_rad": {
            role: _position_span([
                event["feedback_positions"] for event in states
                if event.get("controller") == role and event.get("feedback_positions")
            ])
            for role in ("arm", "gripper")
        },
        "joint_feedback_span_rad": _position_span([
            event["feedback_positions"] for event in feedback
            if event.get("feedback_positions")
        ]),
        "interpolated_filtered_target_span_rad": _position_span([
            point
            for event in sent
            for point in event.get("interpolated_filtered_right_actions", [])
        ]),
        "first_sent_goal": (
            {
                "action_sequence": sent[0].get("action_sequence"),
                "first_right_action": sent[0]["blended_right_actions"][0],
                "last_right_action": sent[0]["blended_right_actions"][-1],
            } if sent else None
        ),
        "last_sent_goal": (
            {
                "action_sequence": sent[-1].get("action_sequence"),
                "first_right_action": sent[-1]["blended_right_actions"][0],
                "last_right_action": sent[-1]["blended_right_actions"][-1],
            } if sent else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
