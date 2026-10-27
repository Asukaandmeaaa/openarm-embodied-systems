"""Gate D automatic expert, recorder fault injection, sync, and quality acceptance."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..expert import AutomaticExpert
from ..paths import workspace_path
from ..quality import evaluate_expert_run
from ..recording import EpisodeRecorder, RecorderStateError
from ..recording.synchronization import synchronize_30hz


def validate_gate_d(*, save_artifact: bool = True) -> dict:
    run = AutomaticExpert().run(); quality = evaluate_expert_run(run)
    camera_timestamps = np.arange(run.times[0], run.times[-1] + 1e-9, 1 / 30)
    _, exported_states, exported_actions, _, sync = synchronize_30hz(
        run.times, run.states, run.times, run.actions,
        {"ceiling": camera_timestamps.tolist(), "wrist_right": (camera_timestamps + 0.004).tolist()},
    )
    with tempfile.TemporaryDirectory(prefix="openarm_recorder_fault_") as temporary:
        recorder = EpisodeRecorder(Path(temporary))
        recorder.begin({"source": "automatic_expert"}, "success")
        recorder.append("joint_observation", float(run.times[0]), {"state": run.states[0].tolist()})
        destination = recorder.finish_success(run.result)
        recorder.begin({"source": "fault_injection"}, "failure"); recorder.finish_failure("injected_failure")
        recorder.begin({"source": "fault_injection"}, "cancel"); recorder.cancel()
        residual = recorder.transactions / "crash_residual"; residual.mkdir(); (residual / "metadata.json").write_text('{"transaction_state":"ACTIVE"}\n')
        catalog = recorder.write_catalog()
        rejected = False
        recorder.begin({"source": "contract_rejection"}, "rejected")
        try:
            recorder.finish_success({"full_sequence_success": False})
        except RecorderStateError:
            rejected = True
        recorder.cancel()
        recorder_checks = {
            "atomic_success_commit": destination.parent.name == "episodes",
            "failed_isolated": catalog["failed"] == ["failure"],
            "cancel_removed": "cancel" not in json.dumps(catalog),
            "residual_reported": catalog["residual_transactions"] == [{"episode_id": "crash_residual", "state": "ACTIVE"}],
            "contract_rejection": rejected,
        }

    artifact = workspace_path("results", "expert", f"automatic_expert_{run.seed}.npz")
    quality_path = workspace_path("results", "quality", f"automatic_expert_{run.seed}.json")
    if save_artifact:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, times=run.times, states=run.states, actions=run.actions, phases=run.phases, cube_poses=run.cube_poses, right_qvel=run.right_qvel, commands_1000hz=run.commands_1000hz)
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    checks = {
        "automatic_expert_contract": run.result.full_sequence_success,
        "quality_hard_gate": quality["accepted"],
        "state_action_float32_16d": run.states.dtype == np.float32 and run.actions.dtype == np.float32 and run.states.shape[1] == run.actions.shape[1] == 16,
        "left_action_fixed_park": bool(np.all(run.actions[:, 8:16] == 0.0)),
        "box_path_collision_checked": bool(run.box_path_plan.get("plan_hash")),
        "sync_export_30hz": exported_states.shape == exported_actions.shape and exported_states.shape[1] == 16,
        "camera_skew_below_hard_limit": max(sync.maximum_camera_skew_s.values()) <= 0.05,
        **recorder_checks,
    }
    return {
        "gate": "D", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "automatic_expert": {
            "seed": run.seed, "task_success": run.result.task_success, "full_sequence_success": run.result.full_sequence_success,
            "grasp_count": run.result.grasp_count, "max_lift_m": run.result.max_lift_m,
            "hold_s": run.result.bilateral_hold_s, "placement_stable_s": run.result.placement_stable_s,
            "final_cube_xyz": run.cube_poses[-1, :3].tolist(), "samples_100hz": len(run.times),
            "box_path_plan": run.box_path_plan,
        },
        "sync": asdict(sync), "quality": quality,
        "artifact": str(artifact) if save_artifact else None,
        "manual_kinesthetic_data": "not_fabricated; requires user interaction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    report = validate_gate_d(); output = workspace_path("results", "quality", "gate_d_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
