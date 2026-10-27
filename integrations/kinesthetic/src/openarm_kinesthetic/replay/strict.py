"""30 Hz -> official Hermite -> official Biquad -> position ctrl -> 1000 Hz."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..expert import AutomaticExpert, ExpertRun
from ..planning.kinematics import contact_failures
from ..recording.synchronization import synchronize_30hz
from ..scene.model import load_model, reset_task_ready
from ..task import EvaluationSample, TaskEvaluator
from .official_filters import BiquadLowpass, HermiteUpsampler


def _apply_actuator_command(data, mapping, command: np.ndarray) -> bool:
    """Apply one command and prove that command application did not mutate qpos."""
    before = data.qpos.copy()
    data.ctrl[mapping.right_actuators] = command[:7]
    data.ctrl[mapping.right_gripper_actuator] = command[7]
    data.ctrl[mapping.left_actuators] = command[8:15]
    data.ctrl[mapping.left_gripper_actuator] = command[15]
    return bool(np.array_equal(before, data.qpos))


def _cube_weld_used(mujoco, model, cube_body: int) -> bool:
    """Inspect compiled equality constraints instead of trusting a report literal."""
    for index in range(model.neq):
        if int(model.eq_type[index]) != int(mujoco.mjtEq.mjEQ_WELD):
            continue
        if cube_body in {int(model.eq_obj1id[index]), int(model.eq_obj2id[index])}:
            return True
    return False


def _finger_contacts(mujoco, model, data, cube_body: int) -> tuple[bool, bool]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger")]
    found = [False, False]
    for contact in data.contact[:data.ncon]:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        for index, finger in enumerate(ids):
            if cube_body in bodies and finger in bodies: found[index] = True
    return bool(found[0]), bool(found[1])


def strict_replay(run: ExpertRun | None = None, *, input_origin: str = "automatic_expert_regression_not_human_demonstration") -> dict:
    import mujoco

    run = run or AutomaticExpert().run()
    camera_times = np.arange(run.times[0], run.times[-1] + 1e-9, 1/30)
    timeline, _, actions30, _, _ = synchronize_30hz(run.times, run.states, run.times, run.actions, {"ceiling": camera_times, "wrist_right": camera_times})
    horizon = (len(actions30)-1)/30.0; eval_times = np.arange(0.0, horizon + 1e-12, 0.001)
    upsampler = HermiteUpsampler(30, horizon); commands = upsampler.upsample(actions30, eval_times)
    lowpass = BiquadLowpass(1000, 15.0); filtered = np.asarray([lowpass.step(command) for command in commands])

    model = load_model(); data = mujoco.MjData(model); mapping = reset_task_ready(model, data)
    ready = data.qpos[mapping.right_qpos].copy()
    evaluator = TaskEvaluator.from_model(model, data, mapping, ready)
    errors=[]; collision_counts={}; first_contact=None; last_contact=None; reference_relative=None
    command_application_preserved_qpos = True
    for index, command in enumerate(filtered):
        command_application_preserved_qpos &= _apply_actuator_command(
            data, mapping, command
        )
        mujoco.mj_step(model, data)
        contacts = _finger_contacts(mujoco, model, data, mapping.cube_body)
        if all(contacts) and first_contact is None: first_contact = float(data.time)
        if all(contacts): last_contact = float(data.time)
        relative=data.qpos[mapping.cube_qpos:mapping.cube_qpos+3]-data.site_xpos[mapping.grasp_site]
        if all(contacts) and reference_relative is None:reference_relative=relative.copy()
        drift=float(np.linalg.norm(relative-reference_relative)) if reference_relative is not None else 0.0
        failures = contact_failures(model, data)
        for failure in failures: collision_counts[failure] = collision_counts.get(failure, 0) + 1
        evaluator.update(EvaluationSample(
            float(data.time), tuple(data.qpos[mapping.cube_qpos:mapping.cube_qpos+3]),
            tuple(data.qvel[mapping.cube_dof:mapping.cube_dof+3]), contacts,
            tuple(data.qpos[mapping.right_qpos]), relative_grasp_drift_m=drift, forbidden_collision=bool(failures),
        ))
        measured = mapping.dataset_vector(data.qpos)
        errors.append(measured - command)
    errors = np.abs(np.asarray(errors)); names = [f"right_joint{i}" for i in range(1,8)] + ["right_gripper"] + [f"left_joint{i}" for i in range(1,8)] + ["left_gripper"]
    per_joint = {name: {"max": float(np.max(errors[:,i])), "rms": float(np.sqrt(np.mean(errors[:,i]**2))), "p95": float(np.quantile(errors[:,i],.95))} for i,name in enumerate(names)}
    raw_events = {event["event"]: event["time_s"] for event in run.result.evidence}
    replay_events = {event["event"]: event["time_s"] for event in evaluator.result().evidence}
    common = sorted(set(raw_events) & set(replay_events))
    return {
        "pipeline": {
            "resample_hz": 30,
            "upsampler": type(upsampler).__name__,
            "lowpass": type(lowpass).__name__,
            "command_target": "data.ctrl",
            "physics_hz": int(round(1.0 / float(model.opt.timestep))),
        },
        "input_origin": input_origin,
        "task_success": evaluator.result().task_success, "full_sequence_success": evaluator.result().full_sequence_success,
        "task_result": asdict(evaluator.result()), "frames_30hz": len(actions30), "physics_steps": len(filtered),
        "per_joint_tracking_rad": per_joint,
        "tracking_global": {"max": float(np.max(errors)), "rms": float(np.sqrt(np.mean(errors**2))), "p95": float(np.quantile(errors,.95))},
        "gripper_timing": {"first_bilateral_contact_s": first_contact, "last_bilateral_contact_s": last_contact},
        "phase_timing_differences_s": {name: replay_events[name]-raw_events[name] for name in common},
        "box_path_plan_hash": run.box_path_plan["plan_hash"],
        "collision_differences": {"raw_forbidden": list(run.result.failures), "replay_counts": collision_counts},
        "final_cube_pose_wxyz": data.qpos[mapping.cube_qpos:mapping.cube_qpos+7].tolist(),
        "return_ready_error_rad": float(np.max(np.abs(data.qpos[mapping.right_qpos]-ready))),
        "no_qpos_write_after_reset": command_application_preserved_qpos,
        "object_weld_used": _cube_weld_used(mujoco, model, mapping.cube_body),
    }
