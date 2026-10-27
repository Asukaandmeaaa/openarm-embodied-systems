"""Validate the installed ACT driver using feedback reads only, with motors disabled.

Run with the Dora evaluation Python environment. Never calls start/stop, enables,
disables, motor writes, or motion methods. Interfaces must already be configured.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import openarm_can as oa
from openarm_driver import Config
from dora_openarm.feedback_driver import SingleArmDriver
from dora_openarm.can_runtime import require_paced_can

from probe_openarm_can import check_link, snapshot

CONFIG = Path(__file__).resolve().parents[1] / "official/dora-openarm-evaluation/openarm_real_ros_coordinates.yaml"
FAULTS = (
    "bus_off", "error_passive", "error_warning", "tx_overflow", "rx_overflow",
    "ack_error", "tx_timeout", "write_net_down", "write_no_buffer", "write_other",
)


def validate(interface: str, cycles: int, hz: int) -> dict:
    require_paced_can()
    config = Config(CONFIG)
    if config.get_gripper_posforce():
        raise RuntimeError("Read-only validator requires eight MIT axes")
    before = snapshot(interface)
    check_link(before)
    # SingleArmDriver constructor queries feedback but does not enable motors.
    arm = SingleArmDriver("right_arm" if interface == "can0" else "left_arm", config)
    collection = arm.openarm.get_arm()
    # Allow the constructor's last refresh deadline to expire before sampling.
    time.sleep(0.05)
    misses = [0] * 8
    durations = []
    baseline = [collection.get_link_stats(i).responses for i in range(8)]
    report = {"interface": interface, "version": oa.__version__, "cycles": cycles,
              "hz": hz, "before": before, "ok": False}
    try:
        for _ in range(cycles):
            start = time.monotonic()
            prior = [collection.get_link_stats(i).responses for i in range(8)]
            arm.fetch_position(refresh=True)
            durations.append(time.monotonic() - start)
            for i in range(8):
                misses[i] += collection.get_link_stats(i).responses == prior[i]
            if not arm.openarm.is_bus_healthy():
                raise RuntimeError("CAN bus error; stopped polling")
            time.sleep(max(0.0, start + 1 / hz - time.monotonic()))
        motors = collection.get_motors()
        report.update(
            replies=[collection.get_link_stats(i).responses - baseline[i] for i in range(8)],
            cycles_without_reply=misses,
            q_rad=[float(m.get_position()) for m in motors],
            motor_error_codes=[int(m.get_error_code()) for m in motors],
            motor_enabled=[bool(m.is_enabled()) for m in motors],
            read_ms_max=1000 * max(durations),
        )
        bus = arm.openarm.get_bus_status()
        report["bus_faults"] = {name: int(getattr(bus, name).count) for name in FAULTS}
        report["ok"] = (
            not any(misses)
            and not any(report["bus_faults"].values())
            and not any(report["motor_error_codes"])
            and not any(report["motor_enabled"])
        )
    except (OSError, RuntimeError) as error:
        report["error"] = str(error)
    finally:
        report["after"] = snapshot(interface)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", choices=("can0", "can1"))
    parser.add_argument("--cycles", type=int, default=100, choices=range(1, 601))
    parser.add_argument("--hz", type=int, default=20, choices=(10, 20))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.interface, args.cycles, args.hz)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("before", "after")}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
