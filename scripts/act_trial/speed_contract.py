"""Offline provenance for a low-speed measured-response trip threshold.

This is an attended commissioning bound, NOT a motor rating or a claim that
ROS teleoperation has validated ACT dynamics. Command slew remains separate.
"""
from pathlib import Path
import hashlib

import numpy as np
import yaml

ROS_TEACHING_CONFIG = (Path(__file__).resolve().parents[3] /
    'openarm_kinesthetic_ws/ros2_ws/src/openarm_kinesthetic_recorder/config/real_kinesthetic.yaml')
ROS_PLANNING_CONFIG = (Path(__file__).resolve().parents[3] /
    'openarm_ws/install/openarm_bimanual_moveit_config/share/'
    'openarm_bimanual_moveit_config/config/openarm_v2.0/joint_limits.yaml')
REVIEWED_CEILING = .50
REVIEWED_COMMAND_CEILING = .25
REVIEWED_PLANNING_SCALE = .10


def measured_speed_contract(canonical_velocity, source=None, planning_source=None):
    path = ROS_TEACHING_CONFIG if source is None else Path(source)
    payload = path.read_bytes()
    configured = float(yaml.safe_load(payload)['right_kinesthetic_bridge']['ros__parameters']
                       ['cartesian_max_joint_velocity_rad_s'])
    planning_path = ROS_PLANNING_CONFIG if planning_source is None else Path(planning_source)
    planning_payload = planning_path.read_bytes()
    scale = float(yaml.safe_load(planning_payload)['default_velocity_scaling_factor'])
    limits = np.asarray(canonical_velocity, dtype=float)
    if (limits.shape != (7,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0)
            or not np.isfinite(configured) or configured <= 0
            or not np.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError('invalid measured-speed source limits')
    scaled = limits * min(scale, REVIEWED_PLANNING_SCALE)
    effective = np.minimum(scaled, REVIEWED_CEILING)
    command = np.minimum(effective, min(configured, REVIEWED_COMMAND_CEILING))
    return {
        'limits_rad_s': effective.tolist(),
        'command_limits_rad_s': command.tolist(),
        'reviewed_ceiling_rad_s': REVIEWED_CEILING,
        'reviewed_command_ceiling_rad_s': REVIEWED_COMMAND_CEILING,
        'source_command_cap_rad_s': configured,
        'source_path': str(path), 'source_sha256': hashlib.sha256(payload).hexdigest(),
        'planning_source_path': str(planning_path),
        'planning_source_sha256': hashlib.sha256(planning_payload).hexdigest(),
        'source_default_velocity_scaling_factor': scale,
        'effective_planning_scale': min(scale, REVIEWED_PLANNING_SCALE),
        'scaled_canonical_limits_rad_s': scaled.tolist(),
        'basis': 'command cap from existing ROS teaching; measured trip cap 0.50 rad/s '
                 'for the next attended trial, intersected with canonical ROS limits '
                 'at no more than the reviewed MoveIt default 10% scaling; '
                 'H2 previously recorded right-arm peaks up to 0.385 rad/s',
        'not_a_motor_rating_or_ACT_dynamic_validation': True,
        'no_debounce_or_speed_filter': True,
    }


def check_measured_speed(velocity7, contract):
    velocity = np.asarray(velocity7, dtype=float)
    limits = np.asarray(contract['limits_rad_s'], dtype=float)
    if (limits.shape not in ((7,), (8,)) or velocity.shape != limits.shape
            or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(limits))
            or np.any(limits <= 0)):
        raise ValueError('invalid measured right-arm velocity or limits')
    exceeded = np.flatnonzero(np.abs(velocity) > limits)
    record = {'velocity_rad_s': velocity.tolist(), 'limits_rad_s': limits.tolist(),
              'exceeded_joints': [int(i)+1 for i in exceeded], 'passed': not len(exceeded)}
    return record
