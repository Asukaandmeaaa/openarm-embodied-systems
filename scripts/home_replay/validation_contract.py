"""Read-only ROS limit provenance and explicit simulation command scope."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from core import NAMES, digest, validate_bundle

MOVEIT_LIMITS = (Path(__file__).resolve().parents[3] / 'openarm_ws/install/'
                 'openarm_bimanual_moveit_config/share/openarm_bimanual_moveit_config/'
                 'config/openarm_v2.0/joint_limits.yaml')


def replay_arrays(bundle, scope):
    if scope not in ('all', 'right'):
        raise ValueError('unsupported replay scope')
    t, q = validate_bundle(bundle)
    active = np.arange(8 if scope == 'right' else 16)
    if scope == 'right':
        measured = np.asarray(bundle.get('measured_positions_rad'), dtype=float)
        if measured.shape != q.shape or not np.all(np.isfinite(measured)):
            raise ValueError('right-only replay requires complete measured left collision context')
        q = q.copy()
        # Observed scene context, never a left-arm actuator output. Do not use
        # left CAN command targets as a substitute for its actual parked pose.
        q[:, 8:] = measured[:, 8:]
    return t, q, active


def ros_limits(bundle, fallback_limits, fallback_velocity, *, required=False):
    """Captured URDF is authoritative; current installed MoveIt only tightens it.

    The MoveIt YAML is supplemental current configuration, NOT claimed as a
    parameter dump from the recording. Both sources are hashed in the report.
    Old synthetic all-arm fixtures can still use the model/driver intersection.
    """
    root = bundle.get('capture_directory')
    if not root:
        if required:
            raise ValueError('right-only validation requires captured ROS URDF provenance')
        return np.array(fallback_limits), np.array(fallback_velocity), {}, 'legacy_model_driver'
    root = Path(root)
    manifest_path = root/'manifest.json'
    if digest(manifest_path) != bundle['capture_manifest_sha256']:
        raise ValueError('capture manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('complete'):
        raise ValueError('ROS limit source capture is incomplete')
    relative = 'snapshots/robot_description.urdf'
    urdf_path = root/relative
    if digest(urdf_path) != manifest['files'].get(relative):
        raise ValueError('captured URDF hash mismatch')
    robot = ET.fromstring(urdf_path.read_text())
    planning = yaml.safe_load(MOVEIT_LIMITS.read_text())['joint_limits']
    limits, velocities = [], []
    for name in NAMES:
        joint = robot.find(f"joint[@name='{name}']")
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError(f'missing/unsupported ROS joint {name}')
        bound = joint.find('limit')
        lo, hi, velocity = (float(bound.get(k)) for k in ('lower', 'upper', 'velocity'))
        extra = planning.get(name, {})
        if extra.get('has_position_limits'):
            lo = max(lo, float(extra.get('min_position', lo)))
            hi = min(hi, float(extra.get('max_position', hi)))
        if extra.get('has_velocity_limits'):
            velocity = min(velocity, float(extra['max_velocity']))
        if not np.all(np.isfinite([lo, hi, velocity])) or lo >= hi or velocity <= 0:
            raise ValueError(f'invalid ROS limits for {name}')
        limits.append([lo, hi]); velocities.append(velocity)
    sources = {str(p): digest(p) for p in (manifest_path, urdf_path, MOVEIT_LIMITS)}
    return np.array(limits), np.array(velocities), sources, 'captured_URDF_intersect_current_installed_MoveIt'


def simulation_output(pose, scope):
    """The right-only output can never carry left actuator targets."""
    if scope == 'right':
        return 'simulated_position_right', np.asarray(pose)[:8]
    if scope == 'all':
        return 'simulated_position', np.asarray(pose)
    raise ValueError('unsupported replay scope')


def aligned_pose(t, q, elapsed):
    """Interpolate one recorded segment without copying 16 entire columns/tick."""
    i = int(np.clip(np.searchsorted(t, elapsed, side='right')-1, 0, len(t)-2))
    alpha = np.clip((elapsed-t[i])/(t[i+1]-t[i]), 0., 1.)
    return q[i] + alpha*(q[i+1]-q[i])
