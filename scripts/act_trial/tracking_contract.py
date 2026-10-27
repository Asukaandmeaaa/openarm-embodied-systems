"""Separate ACT performance diagnostics from admission and stop criteria.

Historical H2 peaks do not establish dynamic ACT tracking fault limits.
The stationary recovery contract is deliberately separate and unchanged.
No function in this module communicates with hardware.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def tracking_contract(bundle, profile='commissioning'):
    if profile not in ('commissioning', 'task', 'visual32'):
        raise ValueError('unknown tracking profile')
    return {
        'schema': 'act_tracking_policy_v3',
        'warning_thresholds_rad': [.03]*8,
        'active_stop_limits_rad': ([None]*8 if profile in ('task', 'visual32') else [.03]*8),
        'holding_stop_limits_rad': [.03]*7 + ([.035] if profile == 'visual32' else [.03]),
        'measured_envelope_margin_rad': [.03]*8,
        'warning_basis': 'display-only commissioning reference; not a dynamic safety or task-success bound',
        'active_stop_basis': ('task arm and gripper tracking are diagnostic only; operator-requested '
                              'removal of experimental gripper tracking stop; no force/contact protection'
                              if profile in ('task', 'visual32') else 'short commissioning criterion'),
        'reference': 'previous guard command versus latest feedback; not time-aligned controller error',
        'home_tolerance_unchanged': True,
        'automatic_threshold_adaptation': False,
        'H2_peaks_used_for_ACT_stop': False,
        'validated_dynamic_arm_tracking_fault_limit': False,
    }


def stationary_recovery_contract(bundle):
    """Existing return-HOME admission only; never use as an ACT running limit."""
    limits = [.03]*8
    root = Path(bundle['capture_directory'])
    manifest_path = root/'manifest.json'
    manifest_payload = manifest_path.read_bytes()
    if hashlib.sha256(manifest_payload).hexdigest() != bundle['capture_manifest_sha256']:
        raise ValueError('tracking source manifest hash mismatch')
    manifest = json.loads(manifest_payload)
    path = root/'h2_execution.json'; payload=path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not manifest.get('complete') or digest != manifest['files'].get('h2_execution.json'):
        raise ValueError('tracking source H2 report hash mismatch')
    report = json.loads(payload)
    if report != bundle['h2_execution'] or report.get('success') is not True:
        raise ValueError('successful sealed H2 report required for tracking contract')
    peaks = np.zeros(7)
    names = [f'openarm_right_joint{i}' for i in range(1,8)]
    for stage in ('stage_a','stage_b'):
        rows = report['execution_tracking_audit'][stage]['per_joint']
        if [r['joint_name'] for r in rows] != names:
            raise ValueError('H2 tracking joint order mismatch')
        values = np.array([r['maximum_absolute_tracking_error_rad'] for r in rows],dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values<0):
            raise ValueError('invalid H2 tracking peaks')
        peaks = np.maximum(peaks,values)
    limits[5] = .04
    if not .03 < peaks[5] <= .04 or np.any(peaks > np.array(limits[:7])):
        raise ValueError('H2 tracking evidence differs from reviewed per-joint contract')
    return {'limits_rad':limits, 'recorded_H2_peak_rad':peaks.tolist(),
            'source_path':str(path), 'source_sha256':digest,
            'basis':'Existing stationary return-HOME admission only: J6 limit0.04 covers sealed H2 peak0.03863; '
                    'all other joints and gripper retain0.03. Not a certified safety bound.',
            'home_tolerance_unchanged':True, 'automatic_threshold_adaptation':False}
