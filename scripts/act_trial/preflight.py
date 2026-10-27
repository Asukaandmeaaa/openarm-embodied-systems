"""Offline ACT trial provenance and starting-pose audit; no hardware I/O.

This writes a proposed experiment contract, never a hardware authorization.
Even a supplied recent state file is evidence supplied by the caller, not a
live-state acquisition or proof that the physical scene is unchanged.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from scripts.act_trial.speed_contract import measured_speed_contract
from scripts.act_trial.profiles import execution_profile, collision_checks_enabled
from scripts.act_trial.tracking_contract import tracking_contract
from scripts.act_trial.action_queue import contract as action_queue_contract

ROOT = Path(__file__).resolve().parents[2]
KIN = ROOT.parent / "openarm_kinesthetic_ws"
REPLAY = ROOT / "scripts/home_replay"
sys.path.insert(0, str(REPLAY))
from core import NAMES, digest, validate_bundle, write_json
from simulate import Scene, model_hashes
from validation_contract import ros_limits

DEFAULT_BUNDLE = ROOT / "results/h2_capture_20261014_172126_measured.json"
DEFAULT_CHECKPOINT = ROOT / "outputs/act_real_combined_unreviewed_30000_20261008/checkpoints/030000/pretrained_model"
from scripts.act_trial.closed_contract import CHECKPOINTS
DEFAULT_VISUAL32_CHECKPOINT = CHECKPOINTS[30000]


def trial_settings(profile='commissioning'):
    """Experiment bounds, not replacements for the robot's physical limits."""
    settings = execution_profile(profile)
    gripper_mode = ('ACT_with_visual32_dataset_endpoints' if profile == 'visual32' else
                    'ACT_with_ROS_teaching_endpoints' if settings['command_gripper'] else
                    'hold_existing_controller_reference')
    return {
        "execution_profile": settings,
        "action_schedule": action_queue_contract(profile),
        "collision_checks_enabled": collision_checks_enabled(profile),
        "schema": "openarm_right_act_trial_config_v1",
        "duration_s": settings['duration_s'],
        "arm_envelope_rad": settings['arm_envelope_rad'],
        "max_velocity_rad_s": None if profile == 'visual32' else 0.25,
        "max_acceleration_rad_s2": None if profile == 'visual32' else 0.25,
        "gripper_mode": gripper_mode,
        "gripper_max_motion_rad": (None if profile == 'visual32' else
                                   .25 if settings['command_gripper'] else 0.0),
        "velocity_limit_policy": ('canonical URDF intersect MoveIt, enforced by live guard'
                                  if profile == 'visual32' else 'conservative commissioning cap'),
        "initial_reference_tolerance_rad": 0.03,
        "home_tolerance_rad": 0.03,
        "feedback_max_age_s": 0.10,
        "neighborhood_half_width_rad": 0.07,
        "policy_predictions_modified": False,
        "execution_scope": "right_arm_and_gripper" if settings['command_gripper'] else "right_arm_seven_joints_with_gripper_held",
        "execution_backend": "Dora_ACT_to_ROS_joint_trajectory_controllers",
        "motor_controller": "existing_ROS_gravity_compensated_hardware",
        "allowed_command_topics": ["/right_joint_trajectory_controller/joint_trajectory", "/right_gripper_controller/joint_trajectory"],
        "original_physical_Dora_graph_forbidden": True,
        "left_command_output": False,
        "automatic_home_or_start_alignment": False,
        "hardware_launch": None,
        "physical_execution_authorized": False,
        "constraint_response": ("canonical velocity rejection; analytic rolling q/qd; no extra slew"
                                if profile == 'visual32' else
                                "log_raw_predictions_clip_execution_envelope_and_bound_velocity_acceleration"),
        "gripper_intervention": "retain_raw_prediction_log_all_execution_constraints",
        "threshold_basis": {
            "home_tolerance_rad": ("visual32 uses its separate mechanical-zero contract"
                                   if profile == 'visual32' else
                                   "captured ROS H2 final acceptance and gripper tolerances"),
            "initial_reference_tolerance_rad": "initial/holding admission only; running policy is separate",
            "duration_and_motion_bounds": "deliberate commissioning-trial restrictions; not robot-limit calibration changes",
            "neighborhood_half_width_rad": "non-commissioning: joint-limit corners diagnostic only, MuJoCo collision checks disabled; commissioning: corners block admission",
            "feedback_max_age_s": "deliberate trial freshness requirement, stricter than existing ROS ACT 0.20 s gate",
        },
    }


def finite_start(value):
    if isinstance(value, dict):
        if value.get("joint_names") != NAMES:
            raise ValueError("start joint_names must match the complete recorded 16-joint order")
        value = value.get("positions_rad")
    pose = np.asarray(value, dtype=float)
    if pose.shape != (16,) or not np.all(np.isfinite(pose)):
        raise ValueError("start state must contain exactly 16 finite positions")
    return pose


def inspect_start(pose, home_right, position_limits, scene, settings=None):
    """Check supplied pose and all 128 right-seven-joint box corners.

    The parked left arm and right gripper stay at the supplied values. No
    interpolation, alignment, clipping, or motor command is generated.
    Corner samples do not prove the interior or connecting paths collision-free.
    """
    cfg = trial_settings() if settings is None else settings
    collision_enabled = collision_checks_enabled(cfg['execution_profile']['name'])
    q = finite_start(pose)
    home = np.asarray(home_right, dtype=float)
    limits = np.asarray(position_limits, dtype=float)
    if home.shape != (8,) or limits.shape != (16, 2) or not np.all(np.isfinite(np.r_[home, limits.ravel()])):
        raise ValueError("invalid HOME or ROS position-limit contract")
    errors = np.abs(q[:8] - home)
    right_outside = np.flatnonzero((q[:8] < limits[:8, 0]) | (q[:8] > limits[:8, 1]))
    left_outside = np.flatnonzero((q[8:] < limits[8:, 0]) | (q[8:] > limits[8:, 1])) + 8
    failures = []
    zero_contract = cfg['execution_profile'].get('mechanical_zero_completion')
    if zero_contract is not None:
        lo, hi = zero_contract['closed_gripper_range_rad']
        if np.any(np.abs(q[:7]) > zero_contract['arm_tolerance_rad']):
            failures.append('right_mechanical_zero_mismatch')
        if not lo <= q[7] <= hi:
            failures.append('right_gripper_not_closed')
    elif np.any(errors > cfg["home_tolerance_rad"]):
        failures.append("right_HOME_mismatch")
    if len(right_outside):
        failures.append("right_joint_limit")
    start_hits = scene.put(q) if collision_enabled else []
    if start_hits:
        failures.append("starting_pose_collision")
    corner_collisions, corner_limits = [], []
    for index, signs in enumerate(itertools.product((-1.0, 1.0), repeat=7)):
        sample = q.copy()
        sample[:7] += cfg["neighborhood_half_width_rad"] * np.asarray(signs)
        outside = np.flatnonzero((sample[:7] < limits[:7, 0]) | (sample[:7] > limits[:7, 1]))
        if len(outside):
            corner_limits.append({"corner_index": index, "joint_names": [NAMES[i] for i in outside]})
        hits = scene.put(sample) if collision_enabled else []
        if hits:
            corner_collisions.append({"corner_index": index, "signs": list(signs), "contacts": hits})
    # Non-commissioning modes retain start-pose and joint-limit admission
    # without relying on the legacy MuJoCo task scene.
    corners_block = cfg['execution_profile']['name'] == 'commissioning'
    if corner_collisions and corners_block:
        failures.append("sampled_neighborhood_collision")
    if corner_limits and corners_block:
        failures.append("sampled_neighborhood_joint_limit")
    return {
        "supplied_start_checks_passed": not failures,
        "failures": failures,
        "positions_rad": q.tolist(),
        "HOME_error_rad": errors.tolist(),
        "maximum_HOME_error_rad": float(errors.max()),
        "start_contract": cfg['execution_profile'].get('start_contract', 'configured_HOME'),
        "mechanical_zero_arm_error_rad": np.abs(q[:7]).tolist(),
        "right_limit_violations": [NAMES[i] for i in right_outside],
        "left_limit_diagnostics_nonblocking": [NAMES[i] for i in left_outside],
        "left_role": ("actual_supplied_collision_context_no_commands" if collision_enabled
                      else "measured_only_no_commands"),
        "start_contacts": start_hits,
        "collision_checks_enabled": collision_enabled,
        "neighborhood_corner_count": 128,
        "neighborhood_blocks_start": corners_block,
        "neighborhood_half_width_rad": cfg["neighborhood_half_width_rad"],
        "neighborhood_corner_collisions": corner_collisions,
        "neighborhood_corner_limit_violations": corner_limits,
        "neighborhood_scope": "all 128 sign corners of right J1-J7; left and right gripper fixed",
        "continuous_collision_proof": False,
        "limitations": [
            "MuJoCo collision checks disabled for this non-commissioning profile." if not collision_enabled
            else "MuJoCo collision checks enabled for commissioning.",
            "Corners and center are discrete geometric samples; interior and paths are not certified.",
            "Static geometry does not certify gravity tracking, communication, stopping distance, or braking.",
            "A caller-supplied state file does not establish fresh live hardware feedback or unchanged physical obstacles.",
        ],
    }


def audit_checkpoint(path, profile='commissioning'):
    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    inputs = config.get("input_features", {})
    image_shape = [3, 480, 640] if profile == 'visual32' else [3, 360, 640]
    expected_shapes = {"observation.state": [16], "observation.images.ceiling": image_shape,
                       "observation.images.wrist_right": image_shape}
    if (config.get("type") != "act" or config.get("n_action_steps") != 10 or config.get("chunk_size") != 50
            or config.get("temporal_ensemble_coeff") is not None
            or any(inputs.get(name, {}).get("shape") != shape for name, shape in expected_shapes.items())
            or config.get("output_features", {}).get("action", {}).get("shape") != [16]):
        raise ValueError("checkpoint differs from the established 16D two-camera ACT contract")
    if profile == 'visual32' and path not in {p.resolve() for p in CHECKPOINTS.values()}:
        raise ValueError('closed-transit trial requires the reviewed 30K or 50K checkpoint; no legacy weights')
    required = ["model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"]
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"checkpoint is missing {name}")
    sources = {str(p): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
    return {"directory": str(path), "config": config, "file_sha256": sources,
            "weights_or_preprocessors_modified": False,
            "action_contract": "predict 50x16 with checkpoint processors; execute prefix 10 at 30 Hz; next inference after completion from a fresh observation; no old-tail blend"}


def source_hashes():
    paths = [Path(__file__), REPLAY / "core.py", REPLAY / "simulate.py", REPLAY / "validation_contract.py",
             ROOT / "scripts/act_trial/speed_contract.py", ROOT / "scripts/act_trial/guard.py",
             ROOT / "scripts/act_trial/profiles.py", ROOT / "scripts/act_trial/ros_bridge.py",
             ROOT / "scripts/act_trial/tracking_contract.py",
             ROOT / "scripts/act_trial/episode_completion.py",
             ROOT / "scripts/act_trial/zero_gripper.py",
             ROOT / "scripts/act_trial/motion_generator.py",
             ROOT / "scripts/act_trial/return_home.py", ROOT / "scripts/act_trial/manual_experiment.py",
             ROOT / "scripts/act_trial/control_runtime.py", ROOT / "scripts/actctl",
             ROOT / "scripts/act_trial/action_queue.py", ROOT / "scripts/act_trial/queued_policy_client.py",
             ROOT / "scripts/act_trial/queued_executor.py", ROOT / "scripts/act_trial/logged_executor.py",
             ROOT / "official/dora-openarm-evaluation/dataflow-act-trial-ros-host.yaml",
             ROOT / "official/dora-openarm-evaluation/dataflow-act-visual32-ros-host.yaml",
             ROOT / "scripts/control_ownership/h2.py",
             ROOT / "scripts/control_ownership/gateway.py", ROOT / "scripts/control_ownership/client.py",
             ROOT / "scripts/real_act_inference.py",
             KIN / "src/openarm_kinesthetic/initialization/collision.py"]
    executor = KIN / "ros2_ws/src/openarm_kinesthetic_recorder/openarm_kinesthetic_recorder"
    paths.extend(executor / name for name in ("act_executor.py", "act_trajectory.py", "act_contract.py"))
    paths.append(ROOT.parent / "openarm_ws/src/openarm_ros2/openarm_hardware/src/openarm_simple_hardware.cpp")
    return {str(p.resolve()): digest(p) for p in paths}


def run(bundle_path=DEFAULT_BUNDLE, checkpoint_path=DEFAULT_CHECKPOINT, start_path=None, *, profile='commissioning'):
    bundle_path = Path(bundle_path).resolve()
    bundle = json.loads(bundle_path.read_text())
    validate_bundle(bundle)
    if bundle.get("phase") != "H2" or bundle.get("source") != "measured" or not bundle.get("h2_execution", {}).get("success"):
        raise ValueError("successful measured ROS H2 capture required")
    # ros_limits verifies the manifest and captured URDF hashes before reading.
    limits, velocities, limit_sources, policy = ros_limits(bundle, None, None, required=True)
    if profile == 'visual32':
        from scripts.act_trial.closed_contract import limits as closed_limits
        limits, velocities, limit_sources, policy = closed_limits()
    capture = Path(bundle["capture_directory"])
    manifest = json.loads((capture / "manifest.json").read_text())
    for name, key in (("home_config.yaml", "home_config"), ("driver_config.yaml", "driver_config")):
        path = capture / "snapshots" / name
        if digest(path) != manifest["files"].get(f"snapshots/{name}"):
            raise ValueError(f"captured {name} hash mismatch")
        if yaml.safe_load(path.read_text()) != bundle[key]:
            raise ValueError(f"bundle {key} differs from sealed snapshot")
        limit_sources[str(path)] = digest(path)
    h2_path = capture / "h2_execution.json"
    if digest(h2_path) != manifest["files"].get("h2_execution.json") or json.loads(h2_path.read_text()) != bundle["h2_execution"]:
        raise ValueError("H2 execution report differs from sealed capture")
    limit_sources[str(h2_path)] = digest(h2_path)
    home = bundle["home_config"]["right_home_joint_positions_rad"] + [bundle["home_config"]["home_gripper"]["open_position_rad"]]
    cfg = trial_settings(profile)
    cfg['running_tracking_contract'] = tracking_contract(bundle,profile)
    cfg['threshold_basis']['running_tracking_contract'] = cfg['running_tracking_contract']['active_stop_basis']
    if (bundle["home_config"]["final_acceptance"]["joint_tolerance_rad"] != cfg["home_tolerance_rad"]
            or bundle["home_config"]["home_gripper"]["position_tolerance_rad"] != cfg["home_tolerance_rad"]):
        raise ValueError("trial HOME tolerance is inconsistent with captured ROS H2 acceptance")
    speed_contract = measured_speed_contract(velocities[:7])
    if cfg['execution_profile']['command_gripper']:
        speed_contract['limits_rad_s'].append(min(.50, float(velocities[7])*speed_contract['effective_planning_scale']))
        speed_contract['command_limits_rad_s'].append(min(.25, speed_contract['source_command_cap_rad_s'], speed_contract['limits_rad_s'][-1]))
    cfg.update(joint_names=NAMES, right_joint_names=NAMES[:8], home_right_rad=home,
               measured_speed_contract=speed_contract,
               position_limits_rad=limits[:8].tolist(), velocity_limits_rad_s=velocities[:8].tolist(),
               effective_trial_velocity_limits_rad_s=speed_contract['command_limits_rad_s'],
               effective_table=bundle["home_config"]["static_collision"],
               effective_table_role='collision_guard' if collision_checks_enabled(profile) else 'record_only_unused_in_selected_ACT_profile',
               supplied_start_positions_rad=None)
    report = {"schema": "openarm_right_act_trial_preflight_v1", "physical_execution_authorized": False,
              "physical_replay_authorized": False, "hardware_io_performed": False,
              "needs_fresh_live_state": True, "needs_current_physical_scene_confirmation": True,
              "config": cfg, "checkpoint": audit_checkpoint(checkpoint_path, profile=profile),
              "limit_policy": policy, "limit_source_sha256": limit_sources,
              "capture_bundle": str(bundle_path), "capture_bundle_sha256": digest(bundle_path),
              "current_source_sha256": source_hashes(), "model_sha256": model_hashes(KIN),
              "startup_policy": "no automatic HOME, no alignment, no original physical Dora graph launch",
              "offline_checks_passed": True, "start_state": None,
              "physical_blockers": ["fresh monitored feedback and current scene required",
                                    "holding and measured stop behavior require attended commissioning",
                                    "offline preflight cannot authorize physical execution"]}
    if start_path is not None:
        path = Path(start_path).resolve()
        pose = finite_start(json.loads(path.read_text()))
        scene = Scene(bundle) if collision_checks_enabled(profile) else None
        report["start_state"] = inspect_start(pose, home, limits, scene, settings=cfg)
        report["start_state_source"] = {"path": str(path), "sha256": digest(path), "live_acquisition_verified": False}
        report["offline_checks_passed"] = report["start_state"]["supplied_start_checks_passed"]
        cfg["supplied_start_positions_rad"] = pose.tolist()
        cfg["gripper_hold_position_rad"] = float(pose[7])
    return report, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--start-state", type=Path, help="JSON 16-vector or {joint_names, positions_rad}; offline evidence only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--profile", choices=['commissioning','task','visual32'], default='commissioning')
    args = parser.parse_args()
    report, cfg = run(args.bundle, args.checkpoint, args.start_state, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    if args.config_output:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.config_output, cfg)
    print(json.dumps({"offline_checks_passed": report["offline_checks_passed"],
                      "needs_fresh_live_state": True, "physical_execution_authorized": False,
                      "report": str(args.output)}))
    return 0 if report["offline_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
