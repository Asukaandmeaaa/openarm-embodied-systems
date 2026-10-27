#!/usr/bin/env python3
"""Read-only Dora probes for ACT shadow runs.

``feedback`` only reads already configured CAN feedback.  It never calls a
driver start/stop method, enables/disables motors, or sends a position.
``actions`` audits policy chunks and deliberately has no motion output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


FAULTS = ("bus_off", "error_passive", "error_warning", "tx_overflow", "rx_overflow",
          "ack_error", "tx_timeout", "write_net_down", "write_no_buffer", "write_other")


def _finite(values, n, label):
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size != n or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected {n} finite values")
    return result.astype(np.float32)


class ReadonlyFeedbackProbe:
    """Safety boundary for feedback-only sampling, usable with fake drivers."""

    def __init__(self, drivers):
        if len(drivers) != 2:
            raise ValueError("feedback probe requires right and left drivers")
        self.drivers = drivers
        self.last_poll = -float("inf")

    @staticmethod
    def _fresh_responses(driver, before):
        collection = driver.openarm.get_arm()
        after = [collection.get_link_stats(i).responses for i in range(8)]
        return after, [after[i] > before[i] for i in range(8)]

    def poll(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if now - self.last_poll < 0.05:
            return None
        reports = {}
        positions = {}
        for side, driver in zip(("right", "left"), self.drivers):
            collection = driver.openarm.get_arm()
            before = [collection.get_link_stats(i).responses for i in range(8)]
            started_ns = time.monotonic_ns()
            position = _finite(driver.fetch_position(refresh=True), 8, f"{side} position")
            after, updated = self._fresh_responses(driver, before)
            motors = collection.get_motors()
            enabled = [bool(m.is_enabled()) for m in motors]
            motor_errors = [int(m.get_error_code()) for m in motors]
            _, bus = driver.get_health()
            bus_faults = {name: int(bus.get(name, 0)) for name in FAULTS}
            fresh = all(updated) and not any(enabled) and not any(motor_errors)
            fresh = fresh and bool(bus.get("carrier", True)) and not any(bus_faults.values())
            if not fresh:
                raise RuntimeError(
                    f"{side}: feedback gate failed (updated={updated}, enabled={enabled}, "
                    f"errors={motor_errors}, bus={bus_faults})"
                )
            positions[side] = position
            reports[side] = {
                "responses_before": before, "responses_after": after,
                "updated": updated, "motor_enabled": enabled,
                "motor_error_codes": motor_errors, "bus_faults": bus_faults,
                "source_monotonic_ns": started_ns,
            }
        self.last_poll = now
        return {"arm_right": positions["right"], "arm_left": positions["left"],
                "reports": reports, "source_monotonic_ns": min(
                    reports[s]["source_monotonic_ns"] for s in reports),
                "status": "ready"}


class ActionAudit:
    """Audit complete 50x16 chunks without emitting any action or motion."""

    def __init__(self, max_age_s=0.5, max_skew_s=0.25):
        self.max_age_s, self.max_skew_s = max_age_s, max_skew_s
        self.accepted = self.rejected = 0
        self.ages, self.skews, self.reject_reasons = [], [], []

    def ingest(self, chunk, metadata=None, now=None):
        metadata = metadata or {}
        now = time.monotonic_ns() if now is None else int(now)
        try:
            array = np.asarray(chunk, dtype=np.float64)
            if array.shape != (50, 16) or not np.isfinite(array).all():
                raise ValueError("expected finite action chunk shape 50x16")
            source = metadata.get("source_monotonic_ns", metadata.get("timestamp_ns"))
            if source is not None:
                age = (now - int(source)) / 1e9
                if age < -0.05 or age > self.max_age_s:
                    raise ValueError(f"stale/future source timestamp age={age:.3f}s")
                self.ages.append(age)
            skew = metadata.get("camera_skew_s", metadata.get("timestamp_skew_s"))
            if skew is not None:
                skew = abs(float(skew))
                if skew > self.max_skew_s:
                    raise ValueError(f"camera timestamp skew={skew:.3f}s")
                self.skews.append(skew)
            self.accepted += 1
            return True
        except (TypeError, ValueError, OverflowError) as exc:
            self.rejected += 1
            self.reject_reasons.append(str(exc))
            return False

    def report(self):
        return {"accepted": self.accepted, "rejected": self.rejected,
                "max_age_s": max(self.ages, default=None),
                "max_skew_s": max(self.skews, default=None),
                "reject_reasons": self.reject_reasons,
                "ok": self.accepted > 0 and self.rejected == 0}


def _feedback_main(args):
    import pyarrow as pa
    import dora
    import openarm_driver
    from dora_openarm import feedback_driver
    from dora_openarm.can_runtime import require_paced_can
    require_paced_can()
    config = openarm_driver.Config(args.config)
    probe = ReadonlyFeedbackProbe([
        feedback_driver.SingleArmDriver("right_arm", config),
        feedback_driver.SingleArmDriver("left_arm", config),
    ])
    node = dora.Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "tick":
            continue
        result = probe.poll()
        if result is None:
            continue
        metadata = {"source_monotonic_ns": result["source_monotonic_ns"]}
        node.send_output("arm_right", pa.array(result["arm_right"]), metadata)
        node.send_output("arm_left", pa.array(result["arm_left"]), metadata)
        node.send_output("status", pa.array([result["status"]]), metadata)


def _actions_main(args):
    import dora
    audit = ActionAudit(args.max_age, args.max_skew)
    node = dora.Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "actions":
            continue
        value = event["value"]
        try:
            # Arrow list arrays cannot be converted to a numeric ndarray with
            # ``to_numpy``; materialize the bounded chunk explicitly.
            chunk = np.asarray(value.to_pylist(), dtype=np.float64)
        except AttributeError:
            chunk = value
        audit.ingest(chunk, event.get("metadata", {}))
    report = audit.report()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    feedback = sub.add_parser("feedback")
    feedback.add_argument("--config", type=Path, required=True)
    actions = sub.add_parser("actions")
    actions.add_argument("--report", type=Path, required=True)
    actions.add_argument("--max-age", type=float, default=0.5)
    actions.add_argument("--max-skew", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.mode == "feedback":
        return _feedback_main(args)
    return _actions_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
