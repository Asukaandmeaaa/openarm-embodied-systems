"""Read-only, streaming ACT diagnostics in physical units; never a training loss."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import FPS, digest
from .paths import recorded_path

RECOVERY_TYPES = ("closed_approach", "lateral_offset", "bad_height", "recoverable_drop", "pre_release_hold")
PHASES = ("reach", "pregrasp_open", "approach", "close", "carry", "release", "return", "other")
SIM_PHASES = {
    "STABILIZE": "reach", "PREGRASP": "pregrasp_open", "RECOVERY_OPEN": "pregrasp_open",
    "RECOVERY_PREGRASP": "pregrasp_open", "APPROACH": "approach", "CLOSE": "close",
    "VERIFY": "close", "LIFT": "carry", "HOLD": "carry", "TRANSPORT": "carry",
    "PLACE_ABOVE_BOX": "carry", "RELEASE": "release", "SETTLE": "release",
    "RETURN_HOME": "return", "TAIL": "return",
}
REAL_PHASES = {"reach": "reach", "pregrasp_open": "pregrasp_open", "approach": "approach",
               "close": "close", "carry": "carry", "open": "release", "return": "return",
               "transit_close": "return", "return_closed": "return"}


def distribution(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size and not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic values")
    return {"count": int(a.size), "mean": float(a.mean()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "p95": float(np.percentile(a, 95)) if a.size else None,
            "max": float(a.max()) if a.size else None}


def difference_metrics(delta) -> dict:
    a = np.asarray(delta, dtype=float)
    if a.ndim != 2 or a.shape[1] < 8 or not np.isfinite(a).all():
        raise ValueError("Expected finite N x 8 joint differences")
    arm = np.abs(a[:, :7]) * 180 / np.pi
    return {"samples": len(a), "arm_abs_error_deg": distribution(arm),
            "arm_max_axis_error_deg": distribution(arm.max(axis=1) if len(a) else []),
            "arm_mae_deg_by_joint": arm.mean(axis=0).tolist() if len(a) else None,
            "gripper_abs_error_rad": distribution(np.abs(a[:, 7]))}


def phase_labels(ds, episode: dict) -> tuple[np.ndarray, dict]:
    entry = ds.manifest["episodes"][episode["number"]]
    n = episode["length"]
    if entry.get("collection_root"):
        raw = recorded_path(entry["collection_root"]) / entry["path"] / "numeric.npz"
        actual = digest(raw)
        if entry.get("sha256_numeric") and actual != entry["sha256_numeric"]:
            raise ValueError(f"Phase source hash mismatch: {raw}")
        start = int(entry["converted_start_frame"])
        with np.load(raw) as z:
            phases = z["phase"][start:start + n]
            valid = z["valid"][start:start + n]
            if len(phases) != n or not valid.all():
                raise ValueError("Phase source does not match expert-valid suffix")
        labels = np.asarray([SIM_PHASES.get(str(p), "other") for p in phases])
        return labels, {"path": str(raw), "sha256": actual, "start_frame": start}
    source = entry.get("source_episode_path")
    if not source:
        raise ValueError("No phase provenance for episode")
    path = recorded_path(source) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labels = np.full(n, "other", dtype="U16")
    times = np.arange(n) / FPS
    timeline = sorted((float(e["timestamp_s"]) - entry["start_timestamp_s"], e["detail"]["name"])
                      for e in events if e.get("phase") == "timeline_event")
    for t, name in timeline:
        labels[times >= t] = REAL_PHASES.get(name, "other")
    return labels, {"path": str(path), "sha256": digest(path), "start_timestamp_s": entry["start_timestamp_s"]}


def event_metrics(predicted_gripper, reference_gripper, labels) -> dict:
    """h0 predictions vs reference transitions. Missing transitions stay in the denominator."""
    pred, ref = np.asarray(predicted_gripper), np.asarray(reference_gripper)
    n = len(ref)
    out = {}
    for phase in ("pregrasp_open", "close", "release"):
        mask = np.asarray(labels) == phase
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
        events = []
        no_transition = 0
        for start, end in zip(starts, ends):
            g0, g1 = float(ref[max(0, start - 1)]), float(ref[end - 1])
            if abs(g1 - g0) < 0.05 or (g1 > g0) != (phase == "close"):
                no_transition += 1
                continue
            # Limit search to this event, with one second of early-warning context.
            lo, hi = max(0, start - FPS), min(n, end + 15)
            def crossing(values, fraction, first, last):
                flags = (values[first:last] - g0) / (g1 - g0) >= fraction
                hits = np.flatnonzero(flags[:-2] & flags[1:-1] & flags[2:])
                return int(first + hits[0]) if len(hits) else None
            fractions = {}
            for fraction in (0.1, 0.5, 0.9):
                truth = crossing(ref, fraction, start, min(n, end + 2))
                hit = crossing(pred, fraction, lo, hi)
                fractions[str(fraction)] = {"reference_frame": truth, "predicted_frame": hit,
                    "missing": hit is None, "delay_s": (hit - truth) / FPS if hit is not None and truth is not None else None}
            events.append({"start_frame": int(start), "end_frame_exclusive": int(end), "crossings": fractions})
        out[phase] = {"eligible_events": len(events), "runs_without_reference_transition": no_transition,
                      "events": events, "by_fraction": {}}
        for fraction in ("0.1", "0.5", "0.9"):
            rows = [e["crossings"][fraction] for e in events]
            eligible = [e for e in rows if e["reference_frame"] is not None]
            missing = sum(e["missing"] for e in eligible)
            out[phase]["by_fraction"][fraction] = {
                "reference_events": len(eligible), "missing_events": missing,
                "missing_fraction": missing / len(eligible) if eligible else None,
                "delay_s": distribution([e["delay_s"] for e in eligible if e["delay_s"] is not None])}
    expected = (np.asarray(labels) == "pregrasp_open") & (ref <= -0.35)
    wrong = expected & (pred > -0.10)
    out["failure_to_open"] = {"expected_open_frames": int(expected.sum()),
        "predicted_near_closed_frames": int(wrong.sum()),
        "fraction": float(wrong.sum() / expected.sum()) if expected.any() else None,
        "reference_open_threshold_rad": -0.35, "prediction_near_closed_threshold_rad": -0.10}
    return out


class PredictionMetrics:
    """Frame-weighted target-time-aligned phase/domain errors, no padded labels."""
    def __init__(self, ds, horizon: int):
        self.ds, self.horizon = ds, horizon
        self.horizons = [h for h in (0, 10, 25, 49, 99) if h < horizon]
        self.labels, self.provenance, self.pred_grip, self.seen = {}, {}, {}, {}
        self.episodes = {e["number"]: e for e in ds.episodes}
        self.sums = {}
        for e in ds.episodes:
            number = e["number"]
            self.labels[number], self.provenance[number] = phase_labels(ds, e)
            self.pred_grip[number] = np.full(e["length"], np.nan)
            self.seen[number] = np.zeros(e["length"], dtype=bool)

    def update(self, predictions, episode_numbers, frames):
        pred = np.asarray(predictions)
        if pred.shape != (len(frames), self.horizon, 16) or not np.isfinite(pred).all():
            raise ValueError("Invalid predictions for diagnostics")
        for p, number, frame in zip(pred, episode_numbers, frames):
            number, frame = int(number), int(frame)
            e = self.episodes[number]
            if self.seen[number][frame]:
                raise ValueError("Duplicate diagnostic observation")
            self.seen[number][frame] = True
            self.pred_grip[number][frame] = p[0, 7]
            for h in self.horizons:
                target = frame + h
                if target >= e["length"]:
                    continue
                delta = np.abs(p[h].astype(float) - e["a"][target])
                buckets = [("overall", "all"), ("phase", self.labels[number][target]),
                           ("domain", e["domain"])]
                if e.get("recovery_type"):
                    buckets.append(("recovery_type", e["recovery_type"]))
                for group, name in buckets:
                    key = (group, name, str(h))
                    value = self.sums.setdefault(key, np.zeros(3))
                    value += [delta[:7].mean() * 180 / np.pi, delta[7], 1]

    def report(self):
        if any(not seen.all() for seen in self.seen.values()):
            raise ValueError("Diagnostic coverage incomplete")
        report = {"schema": "act_prediction_metrics_v1", "aggregation": "frame-weighted; target-time phase; padded targets excluded",
                  "timing": "h0 predictions on saved observations; not closed-loop execution timing",
                  "phase_sources": self.provenance, "by_phase": {}, "by_domain": {}, "by_recovery_type": {},
                  "overall": {}, "events_by_episode": {}, "event_summary": {}}
        for group, name, h in self.sums:
            arm, grip, count = self.sums[(group, name, h)]
            dest = report["overall"] if group == "overall" else report["by_" + group].setdefault(name, {})
            dest[h] = {"frames": int(count), "arm_mae_deg": float(arm / count), "gripper_mae_rad": float(grip / count)}
        for name in PHASES:
            report["by_phase"].setdefault(name, {})
        for name in RECOVERY_TYPES:
            report["by_recovery_type"].setdefault(name, {})
        for number, e in self.episodes.items():
            report["events_by_episode"][str(number)] = event_metrics(self.pred_grip[number], e["a"][:, 7], self.labels[number])
        for phase in ("pregrasp_open", "close", "release"):
            report["event_summary"][phase] = {}
            for fraction in ("0.1", "0.5", "0.9"):
                entries = [r[phase]["by_fraction"][fraction] for r in report["events_by_episode"].values()]
                total, missing = sum(e["reference_events"] for e in entries), sum(e["missing_events"] for e in entries)
                delays = [event["crossings"][fraction]["delay_s"] for r in report["events_by_episode"].values()
                          for event in r[phase]["events"] if event["crossings"][fraction]["delay_s"] is not None]
                report["event_summary"][phase][fraction] = {"reference_events": total, "missing_events": missing,
                    "missing_fraction": missing / total if total else None, "delay_s": distribution(delays)}
        open_rows = [r["failure_to_open"] for r in report["events_by_episode"].values()]
        expected = sum(r["expected_open_frames"] for r in open_rows)
        wrong = sum(r["predicted_near_closed_frames"] for r in open_rows)
        report["failure_to_open"] = {"expected_open_frames": expected, "predicted_near_closed_frames": wrong,
            "fraction": wrong / expected if expected else None,
            "reference_open_threshold_rad": -0.35, "prediction_near_closed_threshold_rad": -0.10}
        return report


def recovery_success(rows: list[dict]) -> dict:
    out = {}
    for kind in RECOVERY_TYPES:
        subset = [r for r in rows if r["scene"].get("recovery_type") == kind]
        failures = {}
        for r in subset:
            if not r["success"]:
                reason = r.get("failure_type") or "unknown"
                failures[reason] = failures.get(reason, 0) + 1
        success = sum(bool(r["success"]) for r in subset)
        out[kind] = {"scenes": len(subset), "families": len({r['scene']['group_id'] for r in subset}),
                     "success": success, "success_rate": success / len(subset) if subset else None,
                     "failure_types": failures}
    return out


def validation_brief(row: dict) -> dict:
    metrics = row.get("diagnostics", {})
    return {"loss": row.get("loss"), "h0": metrics.get("overall", {}).get("0"),
            "by_phase_h0": {name: horizons.get("0") for name, horizons in metrics.get("by_phase", {}).items()},
            "failure_to_open": metrics.get("failure_to_open"),
            "event_50pct": {phase: fractions.get("0.5") for phase, fractions in metrics.get("event_summary", {}).items()},
            "by_recovery_type_h0": {name: horizons.get("0") for name, horizons in metrics.get("by_recovery_type", {}).items()}}
