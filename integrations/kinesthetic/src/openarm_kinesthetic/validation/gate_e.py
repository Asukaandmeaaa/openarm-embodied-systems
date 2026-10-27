"""Gate E strict replay acceptance."""

from __future__ import annotations

import argparse
import json

from ..paths import workspace_path
from ..replay import strict_replay


def validate_gate_e() -> dict:
    replay = strict_replay()
    checks = {
        "task_success": replay["task_success"], "full_sequence_success": replay["full_sequence_success"],
        "normal_actuator_path": replay["pipeline"]["command_target"] == "data.ctrl",
        "no_qpos_write": replay["no_qpos_write_after_reset"], "no_weld": not replay["object_weld_used"],
        "return_ready": replay["return_ready_error_rad"] <= 0.03,
        "box_path_plan_traceable": bool(replay.get("box_path_plan_hash")),
    }
    return {"gate":"E","status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"replay":replay}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();report=validate_gate_e()
    output=workspace_path("results","replay","gate_e_report.json");output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if report["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
