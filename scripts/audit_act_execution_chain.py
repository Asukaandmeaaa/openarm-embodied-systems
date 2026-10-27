#!/usr/bin/env python3
"""Read stored ACT logs only. No ROS/CAN, inference or motion.

Differences are contemporaneous log differences, NOT time-aligned tracking
errors. This audit cannot establish physical clearance or task success.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def differences(rows, first, second):
    values = [np.asarray(r[first])[:8] - np.asarray(r[second])[:8]
              for r in rows if r.get(first) is not None and r.get(second) is not None]
    if not values:
        return None
    return {'count': len(values), 'absolute_p50_rad': np.percentile(np.abs(values), 50, axis=0).tolist(),
            'absolute_p95_rad': np.percentile(np.abs(values), 95, axis=0).tolist()}


def audit(trial):
    guards = []
    for line in (trial/'bridge.jsonl').open():
        record = json.loads(line)
        if record.get('kind') == 'guard':
            guards.append(record)
    active = [r for r in guards if r['decision'].get('stage') == 'active']
    if not active:
        raise ValueError('No active guard records; nothing to compare')
    start, end = active[0]['monotonic_ns'], guards[-1]['monotonic_ns']
    blend_rows, source_hashes = [], []
    for path in trial.glob('executor_*.jsonl'):
        raw = None
        raw_hash = None
        for line in path.open():
            r = json.loads(line)
            if r.get('event') == 'executor_wrapper_started':
                source_hashes.append(r.get('source_sha256'))
            if r.get('event') == 'executor_raw_chunk':
                raw, raw_hash = r['full_positions_rad'], r.get('trial_chunk_sha256')
            if (r.get('event') == 'executor_blended_and_hermite' and raw is not None
                    and r.get('trial_chunk_sha256') == raw_hash
                    and start <= r['monotonic_ns'] <= end):
                blend_rows.append({'raw': raw[0], 'blend': r['blended_positions_rad'][0]})
    decisions = [r['decision'] for r in active]
    return {
        'trial': str(trial.resolve()), 'hardware_io': False,
        'interpretation': 'Contemporaneous differences, not time-aligned tracking error; no physical clearance proof.',
        'active_ticks': len(active), 'active_log_duration_s': (end-start)/1e9,
        'interventions': dict(Counter(i for r in decisions for i in r.get('interventions', []))),
        'raw_to_blended_first_action': differences(blend_rows, 'raw', 'blend'),
        'executor_request_to_guard_command': differences(decisions, 'desired', 'dispatched_proposal'),
        'measured_to_guard_command': differences(decisions, 'measured', 'dispatched_proposal'),
        'faults': [{'errors': r['decision']['errors'],
                    'contacts': r['decision'].get('collision_contacts'),
                    'measured': r['decision'].get('measured')}
                   for r in guards if r['decision'].get('errors')],
        'executor_source_hashes': source_hashes,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.trial), indent=2))
