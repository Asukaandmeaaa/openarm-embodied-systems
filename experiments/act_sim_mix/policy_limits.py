"""Explicit offline comparison with frozen deployed position saturation.

The expert collector keeps its tighter demonstration/planning guards. This
module only affects a policy evaluation that explicitly selects deployment.
No hardware/runtime imports and no changes to upstream interpolation/filter.
"""
import json
from pathlib import Path

import numpy as np

from .sim_expert.world import SimWorld

SNAPSHOT = Path(__file__).resolve().parent / 'config/deployment_command_envelope_20261029.json'


class DeploymentEnvelope:
    def __init__(self):
        self.snapshot = json.loads(SNAPSHOT.read_text())
        if self.snapshot['schema'] != 'act_deployment_command_envelope_v1':
            raise ValueError('Invalid deployment envelope schema')
        self.right = np.asarray(self.snapshot['right_position_limits_rad'], dtype=float)
        if self.right.shape != (8, 2) or not np.isfinite(self.right).all() or np.any(self.right[:, 0] >= self.right[:, 1]):
            raise ValueError('Invalid deployed position endpoints')

    def apply(self, requested):
        command = np.asarray(requested, dtype=float).copy()
        if command.shape != (16,) or not np.isfinite(command).all():
            raise ValueError('Invalid finite 16-dimensional policy command')
        # Mirrors CollectionFeedforward.parameters position saturation. The
        # simulated left side remains its initial feedback hold.
        command[:8] = np.clip(command[:8], self.right[:, 0], self.right[:, 1])
        return command, np.flatnonzero(command[:8] != np.asarray(requested)[:8])


class DeploymentPolicyWorld(SimWorld):
    def __init__(self, seed, cfg, *, jitter=None, envelope):
        self.deployment_envelope = envelope
        super().__init__(seed, cfg, jitter=jitter)

    def validate_command(self, action):
        # Clipping is performed explicitly by execute_block and counted. The
        # physics layer rejects any command that bypasses that stage.
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError('invalid 16-dimensional command')
        bounds = self.deployment_envelope.right
        bad = np.flatnonzero((action[:8] < bounds[:, 0]) | (action[:8] > bounds[:, 1]))
        if len(bad):
            axis = int(bad[0])
            raise RuntimeError(f'deployment endpoint bypassed: right axis {axis + 1}, value={action[axis]}, bounds={bounds[axis].tolist()}')
        return action


def range_diagnostics(prediction, steps, cfg, envelope=None):
    q = np.asarray(prediction)[:steps, :8]
    expert = np.asarray(cfg['position_limits_rad'], dtype=float).copy()
    expert[7] = cfg['gripper_command_range_rad']
    out = {'raw_first_committed_right_min_rad': q.min(axis=0).tolist(),
           'raw_first_committed_right_max_rad': q.max(axis=0).tolist(),
           'expert_outside_counts_by_axis': np.sum((q < expert[:, 0] - 1e-9) | (q > expert[:, 1] + 1e-9), axis=0).tolist()}
    if envelope is not None:
        out['hardware_outside_counts_by_axis'] = np.sum((q < envelope.right[:, 0]) | (q > envelope.right[:, 1]), axis=0).tolist()
    return out
