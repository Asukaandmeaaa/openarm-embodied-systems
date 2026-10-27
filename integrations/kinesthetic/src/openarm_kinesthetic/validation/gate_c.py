"""Gate C numerical acceptance for passive dynamics and compliant teaching."""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..control.gripper import GripperController, GripperState
from ..control.passive import RightArmPassiveMode
from ..scene.model import load_model, reset_task_ready
from ..teaching.virtual_handle import VirtualHandle


def _fresh():
    import mujoco
    model = load_model(); data = mujoco.MjData(model); mapping = reset_task_ready(model, data)
    return mujoco, model, data, mapping


def _step(mujoco, model, data, count: int, callback=None) -> None:
    for _ in range(count):
        data.qfrc_applied[:] = 0.0
        if callback is not None: callback()
        mujoco.mj_step(model, data)


def validate_gate_c() -> dict:
    # Positive: gravcomp plus retained joint damping should preserve the released pose.
    mujoco, model, data, mapping = _fresh(); ready = data.qpos[mapping.right_qpos].copy(); left = data.qpos[mapping.left_qpos].copy()
    passive = RightArmPassiveMode(model, mapping); passive.enable()
    _step(mujoco, model, data, 1000, lambda: passive.apply_gravity_compensation(data))
    positive_joint_drift = float(np.max(np.abs(data.qpos[mapping.right_qpos] - ready)))
    left_drift = float(np.max(np.abs(data.qpos[mapping.left_qpos] - left)))
    right_gain_norm = float(np.linalg.norm(model.actuator_gainprm[mapping.right_actuators]))
    gripper_gain = float(model.actuator_gainprm[mapping.right_gripper_actuator, 0])

    # Negative control proves gravity compensation is what distinguishes the test.
    mujoco_n, model_n, data_n, map_n = _fresh(); initial_z = float(data_n.site_xpos[map_n.grasp_site, 2])
    negative = RightArmPassiveMode(model_n, map_n); negative.enable(); negative.set_gravity_compensation(0.0)
    _step(mujoco_n, model_n, data_n, 1000)
    negative_drop = initial_z - float(data_n.site_xpos[map_n.grasp_site, 2])
    negative_joint_drift = float(np.max(np.abs(data_n.qpos[map_n.right_qpos] - ready)))

    # A released displaced pose must not spring back to the old servo target.
    mujoco_r, model_r, data_r, map_r = _fresh(); passive_r = RightArmPassiveMode(model_r, map_r); passive_r.enable()
    displaced = ready.copy(); displaced[0] += 0.08; displaced[3] -= 0.05
    data_r.qpos[map_r.right_qpos] = displaced; data_r.qvel[:] = 0; mujoco_r.mj_forward(model_r, data_r)
    _step(mujoco_r, model_r, data_r, 500, lambda: passive_r.apply_gravity_compensation(data_r))
    final = data_r.qpos[map_r.right_qpos].copy()
    release_distance_to_displaced = float(np.max(np.abs(final - displaced)))
    release_distance_to_old = float(np.max(np.abs(final - ready)))

    # The handle applies a bounded generalized external wrench and clears on release.
    handle = VirtualHandle(map_r.grasp_site)
    before = data_r.site_xpos[map_r.grasp_site].copy(); handle.engage_at_current(data_r)
    handle.set_target(before + np.asarray([0.025, 0.0, 0.015]))
    maximum_force = 0.0
    def apply_handle():
        nonlocal maximum_force
        passive_r.apply_gravity_compensation(data_r)
        wrench = handle.apply(model_r, data_r); maximum_force = max(maximum_force, float(np.linalg.norm(wrench.force_n)))
    _step(mujoco_r, model_r, data_r, 600, apply_handle)
    handle_displacement = float(np.linalg.norm(data_r.site_xpos[map_r.grasp_site] - before))
    handle.release(data_r); released_force_norm = float(np.linalg.norm(data_r.qfrc_applied))

    # Right gripper remains a live position actuator while arm gains are zero.
    mujoco_g, model_g, data_g, map_g = _fresh(); passive_g = RightArmPassiveMode(model_g, map_g); passive_g.enable()
    gripper_initial = float(data_g.qpos[map_g.right_gripper_qpos]); data_g.ctrl[map_g.right_gripper_actuator] = -0.10
    _step(mujoco_g, model_g, data_g, 800, lambda: passive_g.apply_gravity_compensation(data_g))
    gripper_final = float(data_g.qpos[map_g.right_gripper_qpos])

    # Restore normal position actuators for strict replay.
    passive_g.restore_position_control(); data_g.ctrl[map_g.right_actuators] = ready
    _step(mujoco_g, model_g, data_g, 1500)
    restored_tracking_error = float(np.max(np.abs(data_g.qpos[map_g.right_qpos] - ready)))

    # Keyboard latch/debounce semantics are exercised here and more deeply in unit tests.
    keyboard = GripperController(); close = keyboard.key_down("C", 1.0); repeat = keyboard.key_down("C", 1.01)
    keyboard.update(-0.10); keyboard.key_up("C"); stop = keyboard.key_down("Esc", 2.0)

    checks = {
        "right_position_gain_zero": right_gain_norm == 0.0,
        "gravity_compensated_drift_below_0p03_rad": positive_joint_drift < 0.03,
        "negative_control_drops": negative_drop > 0.02 and negative_joint_drift > positive_joint_drift * 3.0,
        "left_park_held": left_drift < 0.01,
        "right_gripper_actuator_retained": gripper_gain > 0 and abs(gripper_final - gripper_initial) > 0.10,
        "no_return_spring": release_distance_to_old > release_distance_to_displaced,
        "virtual_handle_moves_and_releases": handle_displacement > 0.005 and maximum_force <= handle.max_force + 1e-9 and released_force_norm == 0.0,
        "position_control_restored": restored_tracking_error < 0.03,
        "keyboard_latched_and_debounced": close.accepted and not repeat.accepted and stop.state_after == GripperState.STOPPED.value,
    }
    return {
        "gate": "C", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "metrics": {
            "positive_max_joint_drift_rad_1s": positive_joint_drift, "negative_tcp_drop_m_1s": negative_drop,
            "negative_max_joint_drift_rad_1s": negative_joint_drift, "left_max_joint_drift_rad_1s": left_drift,
            "release_distance_to_displaced_rad": release_distance_to_displaced,
            "release_distance_to_old_target_rad": release_distance_to_old,
            "handle_tcp_displacement_m": handle_displacement, "handle_max_force_n": maximum_force,
            "gripper_initial_rad": gripper_initial, "gripper_final_rad": gripper_final,
            "restored_tracking_max_error_rad": restored_tracking_error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    report = validate_gate_c()
    from ..paths import workspace_path
    output=workspace_path("results","dynamics","gate_c_report.json");output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
