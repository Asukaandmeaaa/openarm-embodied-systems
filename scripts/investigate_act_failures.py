#!/usr/bin/env python3
"""CPU-only investigation of stored data/logs. No inference or robot connection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/lerobot/act_mixed_20261016'


def quantiles(values):
    x = np.asarray(values)
    return {str(p): np.percentile(x, p, axis=0).tolist() for p in (5, 50, 95)} if len(x) else {}


def sustained_index(mask, count=6, start=0):
    for i in range(start, len(mask)-count+1):
        if np.all(mask[i:i+count]):
            return i
    return None


def data_audit():
    records = []
    missing = []
    for split in ('train', 'validation', 'test'):
        manifest = json.loads((DATA/split/'meta/mixed_manifest.json').read_text())
        rows = [r for r in manifest['episodes'] if r['domain'] != 'simulation']
        table = ds.dataset(DATA/split/'data', format='parquet').to_table(
            columns=['episode_index', 'frame_index', 'observation.state', 'action'],
            filter=ds.field('episode_index').isin([r['episode_index'] for r in rows]))
        episodes = np.asarray(table['episode_index'])
        frame = np.asarray(table['frame_index'])
        states = np.asarray(table['observation.state'].to_pylist())
        actions = np.asarray(table['action'].to_pylist())
        for row in rows:
            ids = np.flatnonzero(episodes == row['episode_index'])
            ids = ids[np.argsort(frame[ids])]
            q, a = states[ids], actions[ids]
            close = sustained_index(a[:, 7] > -.225)
            release = sustained_index(a[:, 7] < -.30, start=(close+15 if close is not None else len(a)))
            onset = sustained_index(np.max(abs(a[:, :7]-a[0, :7]), axis=1) > .03)
            v = np.diff(a[:, :7], axis=0)*30
            # A 0.2-second average rate is less sensitive to zero-order-hold edges.
            v200 = (a[6:, :7]-a[:-6, :7])*5
            rec = {k: row[k] for k in ('domain', 'group', 'episode_id', 'split', 'frame_count', 'path')}
            rec.update(
                initial_state=q[0].tolist(), initial_action=a[0].tolist(),
                onset_s=None if onset is None else onset/30,
                close_s=None if close is None else close/30,
                release_s=None if release is None else release/30,
                close_pose=None if close is None else a[close, :7].tolist(),
                release_pose=None if release is None else a[release, :7].tolist(),
                onset_plus_1s_displacement=None if onset is None else
                    (a[min(onset+30, len(a)-1), :7]-a[onset, :7]).tolist(),
                fraction_frames_any_arm_rate_over_025=float(np.mean(np.any(abs(v) > .25, axis=1))),
                fraction_frames_any_arm_rate200ms_over_025=float(np.mean(np.any(abs(v200) > .25, axis=1))),
                p95_abs_arm_rate200ms=quantiles(abs(v200)).get('95'),
                action_state_gap=quantiles(abs(a[:, :8]-q[:, :8])),
                low_arm_motion_fraction=float(np.mean(np.max(abs(v200), axis=1) < .02)),
                held_gripper_command_median=float(np.median(a[a[:, 7] > -.15, 7])) if np.any(a[:, 7] > -.15) else None,
                held_gripper_feedback_median=float(np.median(q[a[:, 7] > -.15, 7])) if np.any(a[:, 7] > -.15) else None,
                alignment=row.get('audit', {}).get('alignment'),
            )
            if close is None or release is None:
                missing.append(row['episode_id'])
            records.append(rec)
    # Select a single latest capture per SOURCE for diversity analysis.
    grouped = defaultdict(list)
    for r in records:
        grouped[r['group']].append(r)
    representatives = [sorted(v, key=lambda r: (r['domain'] == 'latest_real', r['episode_id']))[-1]
                       for v in grouped.values()]
    pairs = []
    for i, a in enumerate(representatives):
        for b in representatives[i+1:]:
            initial = float(np.sqrt(np.mean((np.array(a['initial_state'][:7])-b['initial_state'][:7])**2)))
            if a['onset_plus_1s_displacement'] is None or b['onset_plus_1s_displacement'] is None:
                continue
            x, y = np.array(a['onset_plus_1s_displacement']), np.array(b['onset_plus_1s_displacement'])
            cosine = float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))) if np.linalg.norm(x)*np.linalg.norm(y) > 1e-8 else None
            pairs.append({'sources': [a['group'], b['group']], 'initial_arm_rms_rad': initial,
                          'onset_direction_cosine': cosine, 'onset_1s_displacement_rms_difference_rad': float(np.sqrt(np.mean((x-y)**2))),
                          'close_pose_rms_difference_rad': None if a['close_pose'] is None or b['close_pose'] is None else
                              float(np.sqrt(np.mean((np.array(a['close_pose'])-b['close_pose'])**2)))})
    near = [p for p in pairs if p['initial_arm_rms_rad'] < .03]
    return {
        'scope': 'All reviewed REAL train/validation/test records; source-deduplicated pose analysis. Images not assumed identical.',
        'counts': dict(Counter(r['domain'] for r in records)), 'unique_source_trajectories': len(grouped),
        'source_multiplicity': dict(Counter(len(v) for v in grouped.values())),
        'missing_close_or_release_threshold_detection': missing,
        'representative_close_pose_quantiles': quantiles([r['close_pose'] for r in representatives if r['close_pose'] is not None]),
        'representative_onset_displacement_quantiles': quantiles([r['onset_plus_1s_displacement'] for r in representatives if r['onset_plus_1s_displacement'] is not None]),
        'near_initial_state_source_pairs': len(near),
        'near_pairs_negative_onset_cosine': sum(p['onset_direction_cosine'] is not None and p['onset_direction_cosine'] < 0 for p in near),
        'near_pair_close_pose_rms_rad': quantiles([p['close_pose_rms_difference_rad'] for p in near if p['close_pose_rms_difference_rad'] is not None]),
        'domain_summaries': {domain: {
            'rate200ms_over025_fraction_per_episode': quantiles([r['fraction_frames_any_arm_rate200ms_over_025'] for r in records if r['domain'] == domain]),
            'onset_seconds': quantiles([r['onset_s'] for r in records if r['domain'] == domain and r['onset_s'] is not None]),
            'low_arm_motion_fraction': quantiles([r['low_arm_motion_fraction'] for r in records if r['domain'] == domain]),
        } for domain in ('latest_real', 'old_real')},
        'episodes': records, 'source_pairs': pairs,
    }


def trial_audit():
    from audit_act_execution_chain import audit
    roots = ['act_manual_80yqsf30', 'act_manual_j6bf6l4e', 'act_manual_c16eqo9h',
             'act_manual_fk7ehk2p', 'act_manual_3_ta1s2j']
    result = []
    for name in roots:
        for path in sorted((ROOT/'results'/name).glob('*/result.json')):
            trial = path.parent
            terminal = json.loads(path.read_text())
            row = {'trial': str(trial), 'reason': terminal.get('reason'), 'started': terminal.get('started')}
            if (trial/'offline_preflight.json').exists():
                pre = json.loads((trial/'offline_preflight.json').read_text())
                row['checkpoint'] = pre['checkpoint']['directory']
                row['checkpoint_hashes'] = pre['checkpoint'].get('file_sha256')
            try:
                row['execution'] = audit(trial)
            except (ValueError, OSError) as exc:
                row['audit_unavailable'] = str(exc)
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {'hardware_io': False, 'gpu_inference': False, 'data': data_audit()}
    print('DATA_AUDIT_COMPLETE', flush=True)
    report['trials'] = trial_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
