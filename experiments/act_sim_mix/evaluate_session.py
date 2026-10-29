"""Offline-only diagnostics from an existing Dora session. Never imports robot drivers.

MIT q_ref/q_feedback pairs are read from the same recorded dispatch. Driver log
timestamps are post-dispatch observations, not CAN acknowledgements or fresh
encoder timestamps. Missing evidence is reported, never imputed as zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import digest
from .metrics import difference_metrics, distribution


def records(path):
    if not path.exists():
        return
    with path.open() as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid session record {path}:{number}") from exc


def vector(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all():
        raise ValueError("Expected finite 8-joint logged vector")
    return a


def positive_elapsed(end, start):
    if end is None or start is None:
        return None
    if int(end) < int(start):
        raise ValueError("Non-monotonic session timing evidence")
    return (int(end) - int(start)) / 1e6


def summarize_session(session: Path) -> dict:
    session = session.resolve()
    if not session.is_dir():
        raise FileNotFoundError(session)
    files = [session / name for name in ("policy_client.jsonl", "executor.jsonl", "feedforward_right.jsonl")]
    raw, inference_ms = {}, []
    for r in records(files[0]):
        if r.get("event") != "raw_policy_chunk":
            continue
        md = r["metadata"]
        cid = str(md["chunk_id"])
        if cid in raw:
            raise ValueError("Duplicate raw chunk ID")
        positions = r["positions"]
        steps = md.get("n_action_steps")
        raw[cid] = {"first": vector(positions[0][:8]), "metadata": md,
                    "last_committed": vector(positions[steps - 1][:8])
                    if type(steps) is int and 0 < steps <= len(positions) else None}
        elapsed = positive_elapsed(md.get("replied_monotonic_ns"), md.get("requested_monotonic_ns"))
        if elapsed is not None:
            inference_ms.append(elapsed)
    executor_ms, first_executor = [], {}
    for r in records(files[1]):
        if r.get("event") != "executor_output" or r.get("output") != "move_position_right":
            continue
        md = r.get("metadata", {})
        cid = str(md.get("chunk_id"))
        if cid not in first_executor:
            first_executor[cid] = r["monotonic_ns"]
            elapsed = positive_elapsed(r["monotonic_ns"], md.get("observation_monotonic_ns"))
            if elapsed is not None:
                executor_ms.append(elapsed)
    errors, held, active_count, missing_reference = [], 0, 0, 0
    driver_ms, boundaries = [], []
    previous_id, previous_qref, previous_committed = None, None, None
    for r in records(files[2]):
        if r.get("event") != "mit_command":
            continue
        md = r.get("reference") or {}
        if r.get("held"):
            held += 1
            continue
        if md.get("chunk_id") is None:
            missing_reference += 1
            continue
        cid = str(md["chunk_id"])
        qref, feedback = vector(r["q_ref"]), vector(r["q_feedback"])
        active_count += 1
        errors.append(feedback - qref)
        if cid != previous_id:
            elapsed = positive_elapsed(r["monotonic_ns"], md.get("observation_monotonic_ns"))
            if elapsed is not None:
                driver_ms.append(elapsed)
            prediction = raw.get(cid)
            boundary = {"chunk_id": cid, "execution_mode": md.get("execution_mode"),
                        "raw_prediction_available": prediction is not None,
                        "actual_reference_to_feedback": difference_metrics((qref - feedback)[None]),
                        "actual_reference_to_previous_reference": difference_metrics((qref - previous_qref)[None])
                            if previous_qref is not None else None,
                        "raw_first_to_feedback": difference_metrics((prediction["first"] - feedback)[None])
                            if prediction else None,
                        "raw_first_to_previous_committed_last": difference_metrics((prediction["first"] - previous_committed)[None])
                            if prediction and previous_committed is not None else None,
                        "observation_to_driver_log_ms": elapsed}
            boundaries.append(boundary)
            previous_id = cid
            previous_committed = prediction["last_committed"] if prediction else None
        previous_qref = qref
    jumps = [b["actual_reference_to_previous_reference"]["arm_max_axis_error_deg"]["max"]
             for b in boundaries if b["actual_reference_to_previous_reference"]]
    return {"schema": "act_session_metrics_v1", "session": str(session),
            "input_sha256": {p.name: digest(p) for p in files if p.exists()},
            "missing_files": [p.name for p in files if not p.exists()],
            "raw_predictions": len(raw), "active_driver_samples": active_count,
            "held_driver_samples_excluded": held, "driver_samples_without_chunk_excluded": missing_reference,
            "tracking": difference_metrics(np.asarray(errors).reshape(-1, 8)),
            "timing_ms": {"inference_round_trip": distribution(inference_ms),
                          "inference_round_trip_missing_chunks": len(raw) - len(inference_ms),
                          "observation_to_executor_output": distribution(executor_ms),
                          "observation_to_driver_log": distribution(driver_ms)},
            "filtered_boundary_max_arm_deg": distribution(jumps), "boundaries": boundaries,
            "timing_note": "All elapsed times use recorded monotonic timestamps. Driver log is after dispatch; not CAN ACK latency. q_feedback is the cached feedback logged with that command; its independent sensor age is unavailable.",
            "boundary_note": "Raw h0 may differ from the reference actually sent. Fixed-prefix tail comparison requires recorded n_action_steps; official_blend has no fixed committed tail.",
            "hardware_accessed": False, "physical_success_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve it and choose a new report path")
    report = summarize_session(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("raw_predictions", "active_driver_samples", "missing_files", "tracking", "timing_ms")}, indent=2))


if __name__ == "__main__":
    main()
