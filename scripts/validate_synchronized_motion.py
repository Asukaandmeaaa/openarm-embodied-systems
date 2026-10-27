#!/usr/bin/env python3
"""Offline regression of production Guard backends on frozen recorded targets.

Measured right feedback is idealized as the last proposal, not taken from the
old dynamics. Left context and action ages/tick timing remain from logs. No
hardware, ROS, CAN, image inference, or new closed-loop model evaluation.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.act_trial.guard import Guard
from scripts.act_trial.motion_generator import contract, digest
from scripts.act_trial.preflight import DEFAULT_BUNDLE
from scripts.experiment_act_coordinated import rows_for
from simulate import Scene, model_hashes


def simulate(trial, bundle, mode, trace_path=None):
    rows = rows_for(trial)
    first = next(i for i, r in enumerate(rows) if r["stage"] == "active")
    begin = rows[first]["now"]
    prev = rows[first-1]
    initial = np.array(prev["measured"])
    initial[:8] = prev["dispatched_proposal"]
    scene = Scene(bundle, position_only=True)
    g = Guard(initial, bundle, now=begin-.02, initial_command8=initial[:8],
              scene=scene, profile="task", motion_generator=mode)
    decision = g.tick(None, initial, begin-.01, .001)
    if decision["stage"] != "holding":
        raise ValueError("invalid synthetic starting state")
    g.start(begin-.001)
    frames = [(r["now"], np.array(r["clamped"]), np.array(r["measured"])[8:],
               r["action_age_s"]) for r in rows[first:]]
    # A live bridge streams records; keeping thousands of historical nested
    # dictionaries alive makes generation-2 GC timings an offline artifact.
    # Streaming mode removes that retention, NOT garbage collection itself.
    if trace_path is not None:
        del rows
        gc.collect()  # setup only, outside motion/measurement; GC stays enabled
    trace_stream = trace_path.open("x") if trace_path is not None else None
    records = []
    collections = []
    pending_gc = {}
    def gc_timing(phase, info):
        if phase == "start":
            pending_gc.update(time=time.perf_counter(), tick=len(records), generation=info["generation"])
        elif pending_gc:
            collections.append({"tick": pending_gc["tick"], "generation": pending_gc["generation"],
                                "wall_s": time.perf_counter()-pending_gc["time"]})
            pending_gc.clear()
    gc.callbacks.append(gc_timing)
    try:
        for now, target, left, age in frames:
            measured = np.r_[g.command, left]
            d = g.tick(target, measured, now, .001, age)
            if trace_stream is not None:
                trace_stream.write(json.dumps(d)+"\n")
                records.append({
                    "stage": d["stage"], "errors": d["errors"],
                    "clamped": d["clamped"], "dispatched_proposal": d["dispatched_proposal"],
                    "proposal_velocity_rad_s": d["proposal_velocity_rad_s"],
                    "timing": d["timing"], "collision_checked_samples": d["collision_checked_samples"],
                })
            else:
                records.append(d)
            if d["stage"] != "active":
                break
    finally:
        gc.callbacks.remove(gc_timing)
        if trace_stream is not None:
            trace_stream.close()
    errors = [float(np.sqrt(np.mean((np.array(d["dispatched_proposal"])[:7]
                                    -np.array(d["clamped"])[:7])**2)))
              for d in records if d.get("clamped") is not None]
    wall = [d["timing"]["wall_s"] for d in records]
    cpu = [d["timing"]["thread_cpu_s"] for d in records]
    return {
        "mode": mode, "input_ticks": len(frames), "processed_ticks": len(records),
        "completed_input": len(records) == len(frames) and records[-1]["stage"] == "active",
        "terminal_errors": records[-1]["errors"],
        "median_target_rms_rad": float(np.median(errors)),
        "maximum_command_speed_rad_s": float(max(np.max(np.abs(d["proposal_velocity_rad_s"])) for d in records)),
        "wall_s": {str(p): float(np.percentile(wall, p)) for p in (50, 95, 99, 100)},
        "thread_cpu_s": {str(p): float(np.percentile(cpu, p)) for p in (50, 95, 99, 100)},
        "gc_collections": collections,
        "guard_wall_over_100ms_count": sum(t > .1 for t in wall),
        "collision_checked_samples": sum(d["collision_checked_samples"] for d in records),
        "trace": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streaming", action="store_true",
                        help="reduce offline retention; keep normal automatic garbage collection enabled")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    audit_path = ROOT/"results/act_failure_investigation_20261020/full_audit.json"
    audit = json.loads(audit_path.read_text())
    report = {
        "hardware_io": False, "gpu_inference": False, "physical_authorization": False,
        "motion_contract": contract("synchronized_v1"),
        "sources": {str(p): digest(p) for p in (Path(__file__), DEFAULT_BUNDLE, audit_path)},
        "geometry_hashes": model_hashes(ROOT.parent/"openarm_kinesthetic_ws"),
        "streaming_benchmark": args.streaming,
        "limitations": [
            "Current task mode disables MuJoCo collision checks; completion is not a geometric collision check.",
            "Frozen old executor targets; no prediction changes caused by different images/feedback.",
            "Ideal one-tick right command tracking; no physical dynamic simulation.",
            "Original policy chunk scheduling/old-tail blend is intentionally unchanged.",
            "Existing stop remains last-reference hold; reserve is diagnostic only, not executed or used to reject commands.",
            "Zero-penetration sampled geometry is not positive-clearance physical safety proof.",
            "Timing observed under host load is not a real-time worst-case guarantee.",
        ],
        "trials": [],
    }
    for trial in audit["trials"]:
        if trial["reason"] != "collision in right command proposal sweep":
            continue
        item = {"trial": trial["trial"], "recorded_fault": trial["reason"],
                "bridge_sha256": digest(Path(trial["trial"])/"bridge.jsonl"),
                "checkpoint": trial.get("checkpoint")}
        for mode in ("legacy", "synchronized_v1"):
            print(mode, trial["trial"], flush=True)
            name = f"trial_{len(report['trials'])+1}_{mode}."+("jsonl" if args.streaming else "json")
            result = simulate(Path(trial["trial"]), bundle, mode,
                              args.output/name if args.streaming else None)
            trace = result.pop("trace")
            if not args.streaming:
                (args.output/name).write_text(json.dumps(trace))
            result["trace_file"] = name
            item[mode] = result
            print(json.dumps({k: v for k, v in result.items() if k != "gc_collections"}), flush=True)
        report["trials"].append(item)
        (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print("REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__":
    main()
