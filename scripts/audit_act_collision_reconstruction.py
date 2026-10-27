#!/usr/bin/env python3
"""Reconstruct stopped candidate poses from logs; offline MuJoCo geometry only.

This is NOT dynamic simulation and cannot certify safe real-world execution.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.act_trial.guard import Guard
from scripts.act_trial.preflight import DEFAULT_BUNDLE
from simulate import Scene


def reconstruct(trial):
    records = []
    for line in (trial/'bridge.jsonl').open():
        r = json.loads(line)
        if r.get('kind') == 'guard':
            records.append(r)
    ix = next(i for i, r in enumerate(records)
              if 'collision in right command proposal sweep' in r['decision'].get('errors', []))
    prev, fault = records[ix-1]['decision'], records[ix]['decision']
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    scene = Scene(bundle, position_only=True)
    first = records[0]['decision']
    guard = Guard(first['measured'], bundle, now=first['now'], scene=scene,
                  initial_command8=first.get('tracking_reference_rad'), profile='task')
    guard.command = np.array(prev['dispatched_proposal'])
    guard.velocity = np.array(prev['proposal_velocity_rad_s'])
    candidate, velocity, _, _ = guard._propose(np.array(fault['clamped']), fault['now']-prev['now'])
    measured = np.array(fault['measured'])
    raw = None
    key = records[ix]['source_metadata'].get('trial_chunk_sha256')
    for file in trial.glob('executor_*.jsonl'):
        for line in file.open():
            r = json.loads(line)
            if r.get('event') == 'executor_raw_chunk' and r.get('trial_chunk_sha256') == key:
                raw = np.array(r['full_positions_rad'])
    def contacts(right):
        return scene.put(np.r_[right, measured[8:]])
    def sweep(a, b):
        n, hits = guard._sweep(np.r_[a, measured[8:]], np.r_[b, measured[8:]])
        return {'samples': n, 'contacts': hits}
    result = {
        'trial': str(trial), 'candidate': candidate.tolist(), 'candidate_velocity': velocity.tolist(),
        'previous_command': guard.command.tolist(), 'executor_target': fault['clamped'],
        'velocity_opposes_current_target_joints_1based': (np.flatnonzero(
            (np.array(fault['clamped'])-guard.command)*velocity < -1e-10)+1).tolist(),
        'logged_contacts': fault['collision_contacts'],
        'previous_command_matches_fault_hold': bool(np.allclose(guard.command, fault['dispatched_proposal'], atol=1e-12)),
        'candidate_contacts': contacts(candidate),
        'measured_to_candidate': sweep(measured[:8], candidate),
        'command_to_candidate': sweep(guard.command, candidate),
        'executor_desired_contacts': contacts(fault['clamped']),
        'raw_first_contacts': None if raw is None else contacts(raw[0, :8]),
        'raw_horizon_pose_collision_indices': None if raw is None else
            [i for i, a in enumerate(raw) if contacts(a[:8])],
        'measured_to_raw_first': None if raw is None else sweep(measured[:8], raw[0, :8]),
        'warning': 'Static geometry counterfactual only; no authorization to bypass limits or execute raw actions.',
    }
    # Re-evaluate candidate with full forward to cross-check optimized backend.
    scene.position_only = False
    result['full_forward_candidate_contacts'] = contacts(candidate)
    result['full_forward_measured_to_candidate'] = sweep(measured[:8], candidate)
    result['full_forward_command_to_candidate'] = sweep(guard.command, candidate)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trials = json.loads(args.audit.read_text())['trials']
    results = [reconstruct(Path(t['trial'])) for t in trials
               if t['reason'] == 'collision in right command proposal sweep']
    with args.output.open('x') as out:
        json.dump({'hardware_io': False, 'results': results}, out, indent=2)
    print(args.output)
