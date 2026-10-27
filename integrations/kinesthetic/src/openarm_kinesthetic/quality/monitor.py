"""Hard gates and descriptive soft metrics for simulation episodes."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..expert.automatic import ExpertRun


def evaluate_expert_run(run: ExpertRun) -> dict:
    dt = np.diff(run.times)
    velocities = np.diff(run.actions[:, :7], axis=0) / dt[:, None]
    acceleration = np.diff(velocities, axis=0) / dt[1:, None]
    jerk = np.diff(acceleration, axis=0) / dt[2:, None]
    hard_failures = list(run.result.failures)
    if not run.result.full_sequence_success: hard_failures.append("task_contract_failed")
    if run.states.shape[1:] != (16,) or run.actions.shape[1:] != (16,): hard_failures.append("wrong_16d_shape")
    if not np.all(np.isfinite(run.states)) or not np.all(np.isfinite(run.actions)): hard_failures.append("non_finite_state_action")
    return {
        "accepted": not hard_failures, "hard_failures": list(dict.fromkeys(hard_failures)),
        "task_result": {
            "task_success": run.result.task_success, "full_sequence_success": run.result.full_sequence_success,
            "grasp_count": run.result.grasp_count, "max_lift_m": run.result.max_lift_m,
            "hold_s": run.result.bilateral_hold_s, "placement_stable_s": run.result.placement_stable_s,
            "ready_stable_s": run.result.ready_stable_s, "static_tail_s": run.result.static_tail_s,
        },
        "soft_metrics": {
            "joint_velocity_abs_max_rad_s": float(np.max(np.abs(velocities))),
            "joint_acceleration_abs_max_rad_s2": float(np.max(np.abs(acceleration))),
            "joint_jerk_abs_max_rad_s3": float(np.max(np.abs(jerk))),
            "j5_j6_j7_total_variation_rad": np.sum(np.abs(np.diff(run.actions[:, 4:7], axis=0)), axis=0).tolist(),
            "phase_sample_counts_100hz": dict(Counter(run.phases.tolist())),
        },
        "threshold_policy": "unknown soft thresholds are reported, not auto-tuned",
    }
