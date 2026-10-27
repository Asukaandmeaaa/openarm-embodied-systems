#!/usr/bin/env python3
"""Offline-only coordinated execution experiments. Never publishes commands.

1. Successful expert arm paths: old independent shaping versus uniformly
   retimed C2 reference. Gripper is fixed OPEN in both, not marker replay.
2. Six logged collision attempts: moving-state local splice probes and sequential
   target-stream counterfactual. Recorded targets are frozen, not new closed-loop
   predictions; actual measured motion cannot be inferred from this experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.act_trial.offline_coordinated import (
    bounds, extrema_bounds, fit_and_slow, search_splice,
)
from scripts.act_trial.guard import Guard
from scripts.act_trial.preflight import DEFAULT_BUNDLE
from simulate import Scene, model_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_for(trial):
    rows = []
    for line in (trial/"bridge.jsonl").open():
        r = json.loads(line)
        if r.get("kind") == "guard":
            rows.append(r["decision"])
    return rows


def make_guard(first, bundle, scene):
    return Guard(first["measured"], bundle, now=first["now"], scene=scene,
                 initial_command8=first.get("tracking_reference_rad"), profile="task")


def collision_callback(scene, left):
    return lambda q: scene.put(np.r_[q, left])


def search(guard, q, v, a, target, age, collision, bound_method):
    return search_splice(
        q, v, a, target, lower=guard.envelope_lower, upper=guard.envelope_upper,
        speed=guard.speed, acceleration=np.full(8, guard.ACCELERATION),
        remaining_age=guard.ACTION_AGE-float(age), collision=collision,
        bound_method=bound_method)


def describe(r):
    return {k: getattr(r, k) for k in
            ("kind", "fraction", "duration", "tested", "bound_passed", "geometry_rejected")}


def fault_probes(trial, bundle, scene, bound_method):
    rows = rows_for(trial)
    guard = make_guard(rows[0], bundle, scene)
    fault_ix = next(i for i, r in enumerate(rows)
                    if "collision in right command proposal sweep" in r.get("errors", []))
    fault_time = rows[fault_ix]["now"]
    probes = []
    for offset in (1., .5, .25, 0.):
        i = max(2, max(j for j in range(fault_ix+1) if rows[j]["now"] <= fault_time-offset))
        current, prev, prior = rows[i], rows[i-1], rows[i-2]
        q = np.array(prev["dispatched_proposal"])
        v = np.array(prev["proposal_velocity_rad_s"])
        dt = prev.get("slew_period_s") or min(prev["now"]-prior["now"], .01)
        a = (v-np.array(prior["proposal_velocity_rad_s"]))/dt
        target = np.array(current["clamped"])
        hit = collision_callback(scene, np.array(current["measured"])[8:])
        guard.command, guard.velocity = q.copy(), v.copy()
        legacy, _, _, _ = guard._propose(target, current["now"]-prev["now"])
        r = search(guard, q, v, a, target, current["action_age_s"], hit, bound_method)
        probes.append({
            "seconds_before_fault": float(fault_time-current["now"]),
            "initial_q": q.tolist(), "initial_v": v.tolist(), "initial_a": a.tolist(),
            "target": target.tolist(), "action_age_s": current["action_age_s"],
            "velocity_only_min_stop_time_s": float(np.max(np.abs(v))/guard.ACCELERATION),
            "remaining_observation_age_s": guard.ACTION_AGE-current["action_age_s"],
            "legacy_next_candidate_contacts": hit(legacy),
            "initial_command_contacts": hit(q), "search": describe(r),
            "accepted": r.poly is not None,
            "terminal_q": None if r.poly is None else r.poly(r.duration).tolist(),
            "terminal_velocity": None if r.poly is None else r.poly(r.duration, 1).tolist(),
        })
    return probes


def shadow_stream(trial, bundle, scene, bound_method):
    """Replay continuous command p/v/a, replanning at <=10 Hz, fixed left pose.

    A previously accepted trajectory may finish under its ORIGINAL expiry.
    A new observation never renews that deadline. No solution at rest means hold;
    no solution while moving retains only the previously checked stop trajectory.
    No new feedback or image is synthesized.
    """
    all_rows = rows_for(trial)
    guard = make_guard(all_rows[0], bundle, scene)
    first_ix = next(i for i, r in enumerate(all_rows) if r["stage"] == "active")
    rows = all_rows[first_ix:]
    q = np.array(all_rows[first_ix-1]["dispatched_proposal"])
    v = np.zeros(8)
    a = np.zeros(8)
    left = np.array(rows[0]["measured"])[8:]
    hit = collision_callback(scene, left)
    active = None
    origin = rows[0]["now"]
    last_replan = -float("inf")
    active_origin = origin
    expires = origin
    counts, accepted_fractions = Counter(), []
    trace, events = [], []
    failure = None
    peak_speed = np.zeros(8)
    peak_accel = np.zeros(8)
    total_distance = 0.
    q_previous = q.copy()
    for row in rows:
        now = row["now"]
        if active is not None:
            u = min(now-active_origin, active.duration)
            if now > expires+1e-9 and u < active.duration-1e-9:
                failure = "original observation expired while trajectory still moving"
                break
            q, v, a = (active.poly(u, k) for k in (0, 1, 2))
        total_distance += float(np.linalg.norm(q-q_previous))
        q_previous = q.copy()
        peak_speed = np.maximum(peak_speed, np.abs(v))
        peak_accel = np.maximum(peak_accel, np.abs(a))
        if hit(q):
            failure = "shadow sampled command collision"
            break
        if row.get("clamped") is not None and now-last_replan >= .1-1e-9:
            target = np.array(row["clamped"])
            r = search(guard, q, v, a, target, row["action_age_s"], hit, bound_method)
            counts["replans"] += 1
            counts["candidate_tests"] += r.tested
            counts["bounds_passed"] += r.bound_passed
            counts["geometry_rejected"] += r.geometry_rejected
            if r.poly is not None:
                active, active_origin = r, now
                expires = now+guard.ACTION_AGE-row["action_age_s"]
                counts[r.kind] += 1
                accepted_fractions.append(r.fraction)
                b = (bounds if bound_method == "bernstein" else extrema_bounds)(r.poly)
                peak_speed = np.maximum(peak_speed, b["speed"])
                peak_accel = np.maximum(peak_accel, b["acceleration"])
            else:
                counts["no_new_plan"] += 1
            events.append({"elapsed_s": now-origin, **describe(r),
                           "accepted": r.poly is not None})
            last_replan = now
        trace.append({
            "t": now-origin, "q": q.tolist(), "v": v.tolist(), "a": a.tolist(),
            "target": row.get("clamped"),
        })
    # The final polynomial was checked through its terminal rest state, not just
    # the 100 ms committed prefix. Never execute this trace on hardware.
    errors = [float(np.sqrt(np.mean((np.array(r["q"])[:7]-np.array(r["target"])[:7])**2)))
              for r in trace if r["target"] is not None]
    old = [r for r in rows if r.get("clamped") is not None]
    old_errors = [float(np.sqrt(np.mean((np.array(r["dispatched_proposal"])[:7]
                                         -np.array(r["clamped"])[:7])**2))) for r in old]
    return {
        "duration_s": rows[-1]["now"]-origin, "counts": dict(counts),
        "failure": failure, "completed_frozen_input_stream": failure is None,
        "accepted_fraction_median": float(np.median(accepted_fractions)) if accepted_fractions else None,
        "joint_space_distance_rad": total_distance,
        "shadow_arm_target_rms_median_rad": float(np.median(errors)),
        "legacy_arm_target_rms_median_rad": float(np.median(old_errors)),
        "peak_speed_bound_rad_s": peak_speed.tolist(),
        "peak_acceleration_bound_rad_s2": peak_accel.tolist(),
        "events": events, "trace": trace,
        "limitations": [
            "Frozen recorded executor targets; different motion would change images and subsequent predictions.",
            "Ideal command tracking; no actual/measured feedback counterfactual.",
            "Left arm fixed to initial recorded pose; no moving-context validation.",
            "Static zero-penetration sampling, no positive clearance or dynamics certificate.",
            "Finite candidate search can reject feasible paths; numerical extrema are not interval arithmetic."
            if bound_method == "extrema" else
            "Conservative Bernstein rejection and finite candidate search can reject feasible paths.",
            "Not a real-time performance test and not deployment authorized.",
        ],
    }


def geometry_stats(scene, arms, left, grip):
    count = 0
    first = None
    minimum = 0.
    pairs = {}
    for i, arm in enumerate(arms):
        contacts = scene.put(np.r_[arm, grip, left])
        if contacts:
            count += 1
            minimum = min(minimum, min(c["distance_m"] for c in contacts))
            for c in contacts:
                detail = pairs.setdefault(c["pair"], {"samples": 0, "minimum_distance_m": 0.})
                detail["samples"] += 1
                detail["minimum_distance_m"] = min(detail["minimum_distance_m"], c["distance_m"])
            if first is None:
                first = {"sample": i, "contacts": contacts}
    return {"sample_count": len(arms), "colliding_samples": count,
            "most_negative_contact_m": minimum, "first_collision": first, "pairs": pairs}


def expert_experiment(entry, bundle, scene):
    file = Path(entry["trajectory"])
    if digest(file) != entry["sha256"]:
        raise ValueError("playlist trajectory hash mismatch")
    record = json.loads(file.read_text())
    t = np.array([r["time_from_start_s"] for r in record["samples"]])
    q = np.array([r["positions_rad"][:7] for r in record["samples"]])
    first = {"measured": bundle["measured_end_rad"], "now": 0.}
    guard = make_guard(first, bundle, scene)
    speed, acceleration = guard.speed[:7], np.full(7, guard.ACCELERATION)
    original, slowed, scale = fit_and_slow(t, q, speed, acceleration)
    # Sufficient spatial resolution for fitted path independent of dilation.
    dt = min(.005, .002/max(np.max(bounds(original)["speed"]), 1e-12))
    dense_t = np.linspace(0, t[-1], max(2, int(np.ceil(t[-1]/dt))+1))
    dense_q = original(dense_t)
    linear = np.column_stack([np.interp(dense_t, t, q[:, j]) for j in range(7)])
    left = np.array(bundle["measured_end_rad"])[8:]
    # Fixed open gripper in BOTH arms-only counterfactuals. Marker commands not
    # reconstructed; this is not a reproduction of the physical expert replay.
    grip = -.35
    guard.command = np.r_[q[0], grip]
    guard.velocity[:] = 0
    guard.speed[-1] = 0
    clock = np.linspace(0, t[-1], int(np.ceil(t[-1]/.01))+1)
    shaped, rates, accels = [q[0]], [], []
    old_v = guard.velocity.copy()
    for before, now in zip(clock[:-1], clock[1:]):
        target = np.r_[[np.interp(now, t, q[:, j]) for j in range(7)], grip]
        candidate, velocity, _, used_dt = guard._propose(target, now-before)
        shaped.append(candidate[:7].copy())
        rates.append(velocity[:7].copy())
        accels.append((velocity[:7]-old_v[:7])/used_dt)
        guard.command, guard.velocity = candidate, velocity
        old_v = velocity.copy()
    shaped = np.array(shaped)
    distances = cKDTree(linear).query(shaped)[0]/np.sqrt(7)
    target_at_clock = np.column_stack([np.interp(clock, t, q[:, j]) for j in range(7)])
    b = bounds(slowed)
    dense_clock = np.linspace(0, t[-1], int(np.ceil(t[-1]/.005))+1)
    dense_shaped = np.column_stack([np.interp(dense_clock, clock, shaped[:, j])
                                   for j in range(7)])
    # Sampling a scaled trajectory at corresponding phase gives exactly same
    # geometry as original fit. No need to waste samples in very slow segments.
    return {
        "label": entry["label"], "source": str(file), "sha256": digest(file),
        "arm_only_fixed_open_gripper_rad": grip,
        "original_duration_s": float(t[-1]), "retimed_duration_s": float(slowed.x[-1]),
        "time_scale": scale,
        "speed_limit_rad_s": speed.tolist(), "acceleration_limit_rad_s2": acceleration.tolist(),
        "canonical_limits_rad": np.c_[guard.lower[:7], guard.upper[:7]].tolist(),
        "retimed_position_bounds_inside_canonical_limits": bool(
            np.all(b["lower"] >= guard.lower[:7]) and np.all(b["upper"] <= guard.upper[:7])),
        "retimed_peak_speed_bound_rad_s": b["speed"].tolist(),
        "retimed_peak_acceleration_bound_rad_s2": b["acceleration"].tolist(),
        "retimed_knot_error_max_rad": float(np.max(np.abs(slowed(t*scale)-q))),
        "fit_vs_recorded_linear_max_joint_difference_rad": float(np.max(np.abs(dense_q-linear))),
        "legacy_shaping_nearest_dense_reference_rms_rad": {
            "median": float(np.median(distances)), "p95": float(np.percentile(distances, 95)),
            "max": float(np.max(distances)),
        },
        "legacy_same_timestamp_rms_error_p95_rad": float(np.percentile(
            np.sqrt(np.mean((shaped-target_at_clock)**2, axis=1)), 95)),
        "legacy_peak_command_speed_rad_s": np.max(np.abs(rates), axis=0).tolist(),
        "legacy_peak_command_acceleration_rad_s2": np.max(np.abs(accels), axis=0).tolist(),
        "original_linear_geometry": geometry_stats(scene, linear, left, grip),
        "retimed_fit_geometry": geometry_stats(scene, dense_q, left, grip),
        "legacy_shaped_geometry": geometry_stats(scene, dense_shaped, left, grip),
        "limitations": [
            "Arms-only: fixed fully open gripper, no semantic gripper markers or object model.",
            "Legacy _propose evaluated without fault latch to measure whole-path distortion; not commands.",
            "Uniform dilation preserves fitted curve, NOT original piecewise-linear interpolation.",
            "No denoising: raw recording noise can make uniform slowdown excessively long.",
            "Geometry samples are not proof of continuous clearance or physical tracking.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=ROOT/"results/act_failure_investigation_20261020/full_audit.json")
    p.add_argument("--playlist", type=Path, default=ROOT.parent/"openarm_kinesthetic_ws/configs/replay_compare3_20261020.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bound-method", choices=("bernstein", "extrema"), default="bernstein")
    p.add_argument("--skip-experts", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    scene = Scene(bundle, position_only=True)
    report = {
        "hardware_io": False, "gpu_inference": False, "deployment_modified": False,
        "physical_execution_authorized": False,
        "method": "Offline C2 quintic finite-search prototype, NOT TOPPRA/Ruckig.",
        "bound_method": args.bound_method,
        "input_hashes": {str(f): digest(f) for f in (args.audit, args.playlist, DEFAULT_BUNDLE)},
        "source_hashes": {str(f): digest(f) for f in
                          (Path(__file__), ROOT/"scripts/act_trial/offline_coordinated.py",
                           ROOT/"scripts/act_trial/guard.py", ROOT/"scripts/home_replay/simulate.py")},
        "geometry_hashes": model_hashes(ROOT.parent/"openarm_kinesthetic_ws"),
        "trials": [], "experts": [],
    }
    for trial in json.loads(args.audit.read_text())["trials"]:
        if trial["reason"] != "collision in right command proposal sweep":
            continue
        path = Path(trial["trial"])
        print("PROBES", path, flush=True)
        probes = fault_probes(path, bundle, scene, args.bound_method)
        print("SHADOW", path, flush=True)
        shadow = shadow_stream(path, bundle, scene, args.bound_method)
        output = args.output/f"shadow_{len(report['trials'])+1}.json"
        output.write_text(json.dumps(shadow, indent=2))
        row = {"trial": str(path), "bridge_sha256": digest(path/"bridge.jsonl"),
               "probes": probes, "shadow": {k: v for k, v in shadow.items() if k not in ("trace", "events")},
               "shadow_detail": str(output)}
        report["trials"].append(row)
        print(json.dumps(row["shadow"]), flush=True)
    for entry in ([] if args.skip_experts else json.loads(args.playlist.read_text())["entries"]):
        print("EXPERT", entry["label"], flush=True)
        result = expert_experiment(entry, bundle, scene)
        report["experts"].append(result)
        print(json.dumps(result), flush=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print("REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__":
    main()
