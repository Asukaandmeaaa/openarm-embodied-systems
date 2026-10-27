"""Pinned closed-transit deployment evidence. No ROS, CAN or camera access."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from vision_grasp_baseline.gripper_settings import (
    COLLECTION_POLICY, OPEN_POSITION, TRANSIT_POSITION,
    TRANSIT_MIN_POSITION, TRANSIT_MAX_POSITION,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'outputs/act_closed_v2_37eps_20261023b_50k/checkpoints'
CHECKPOINTS = {step: RUN/f'{step:06d}/pretrained_model' for step in (30000, 50000)}
EPISODE = ROOT/'vision_grasp_baseline/collections/nearby/episodes/episode_033'
MODEL = ROOT/'vision_grasp_baseline/collections/nearby/runs/cycle_033/live_model/robot.urdf'
MODEL_SHA256 = '3d019da6d7cc7365114cb8cbc526ab1803d2d5f91be3fd4dd5063e65bd99fb71'
CEILING_SERIAL = '344322073972'


def conversion_source_hashes():
    paths = ['scripts/act_trial/guard.py', 'scripts/act_trial/motion_generator.py',
             'scripts/act_trial/rolling_actions.py', 'scripts/act_trial/queued_policy_client.py',
             'scripts/act_trial/action_queue.py', 'scripts/act_trial/queued_executor.py',
             'scripts/act_trial/profiles.py', 'scripts/act_trial/speed_contract.py',
             'scripts/act_trial/closed_contract.py', 'scripts/act_trial/closed_replay_audit.py',
             'scripts/act_trial/ros_bridge.py', 'scripts/control_ownership/gateway.py',
             'scripts/act_trial/position_velocity.yaml', 'scripts/act_trial/h1.py',
             'official/dora-openarm-evaluation/dataflow-act-visual32-ros-host.yaml',
             'official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def model_text():
    payload = MODEL.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError('closed-transit collection robot model missing/changed')
    return payload.decode()


def verify_live_model(xml):
    # Ignore serialization whitespace only, never hardware parameters or limits.
    if ET.canonicalize(xml, strip_text=True) != ET.canonicalize(model_text(), strip_text=True):
        raise ValueError('live robot model differs from pinned closed-transit collection model')


def closed_start(measured, command):
    from vision_grasp_baseline.gripper_settings import require_closed_transit
    require_closed_transit(float(measured), float(command))


def execution_contract():
    return dict(version=COLLECTION_POLICY, arm_command_interfaces=['position', 'velocity'],
                execution_mode='position_stream_v1', offline_replay_required=False,
                trajectory_contract='rolling_time_aligned_v1',
                validation_policy='operator requested direct physical trial; replay not performed',
                gripper_command_interfaces=['position'], empty_gripper_target_rad=TRANSIT_POSITION,
                camera_serial=CEILING_SERIAL, robot_model=str(MODEL), robot_model_sha256=MODEL_SHA256,
                hardware_io_performed=False, physical_validation_passed=False)


def limits():
    import numpy as np
    import yaml
    from scripts.home_replay.core import NAMES
    from scripts.home_replay.validation_contract import MOVEIT_LIMITS
    robot = ET.fromstring(model_text())
    planning = yaml.safe_load(MOVEIT_LIMITS.read_text())['joint_limits']
    bounds, speeds = [], []
    for name in NAMES:
        bound = robot.find(f"joint[@name='{name}']/limit")
        lo, hi, speed = (float(bound.get(k)) for k in ('lower', 'upper', 'velocity'))
        extra = planning.get(name, {})
        if extra.get('has_position_limits'):
            lo, hi = max(lo, float(extra.get('min_position', lo))), min(hi, float(extra.get('max_position', hi)))
        if extra.get('has_velocity_limits'):
            speed = min(speed, float(extra['max_velocity']))
        if not np.isfinite([lo, hi, speed]).all() or lo >= hi or speed <= 0:
            raise ValueError('invalid closed-transit limits: '+name)
        bounds.append([lo, hi]); speeds.append(speed)
    return (np.asarray(bounds), np.asarray(speeds),
            {str(MODEL): MODEL_SHA256, str(MOVEIT_LIMITS): hashlib.sha256(MOVEIT_LIMITS.read_bytes()).hexdigest()},
            'pinned_closed_collection_URDF_intersect_current_MoveIt')
