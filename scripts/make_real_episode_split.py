#!/usr/bin/env python3
"""Create a deterministic train/validation/test split by source trajectory group.

Each replay of one source trajectory is assigned to exactly one split.  The
output is a small JSON manifest consumed by ``convert_real_episodes_act.py``.
"""
import argparse, json, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--pattern', default='*', help='episode directory glob, e.g. 20261008_*')
    ap.add_argument('--validation-groups', type=int, default=2)
    ap.add_argument('--test-groups', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261008)
    args = ap.parse_args()
    rows = []
    rejected = []
    pending = []
    for p in sorted(args.source.glob(f'{args.pattern}/metadata.json')):
        m = json.loads(p.read_text())
        status = m.get('review_status')
        if status == 'rejected':
            rejected.append(p.parent.name)
            continue
        if status != 'approved' or m.get('operator_marked_success') is not True:
            pending.append(p.parent.name)
            continue
        group = m.get('source_trajectory_trial_id') or m.get('trial_id') or p.parent.name
        rows.append((p.parent.name, str(group)))
    if pending:
        raise SystemExit('episodes still pending review: ' + ', '.join(sorted(pending)))
    groups = sorted({g for _, g in rows})
    if args.validation_groups < 0 or args.test_groups < 0 or args.validation_groups + args.test_groups >= len(groups):
        raise SystemExit(f'invalid group counts for {len(groups)} groups')
    random.Random(args.seed).shuffle(groups)
    test = set(groups[:args.test_groups])
    validation = set(groups[args.test_groups:args.test_groups + args.validation_groups])
    out = {'seed': args.seed, 'group_key': 'source_trajectory_trial_id',
           'rejected_episodes': sorted(rejected),
           'groups': {'train': [], 'validation': sorted(validation), 'test': sorted(test)},
           'train': [], 'validation': [], 'test': []}
    for eid, group in rows:
        role = 'test' if group in test else 'validation' if group in validation else 'train'
        out[role].append(eid)
        if group not in out['groups'][role]: out['groups'][role].append(group)
    for role in ('train','validation','test'):
        out[role].sort()
        out['groups'][role].sort()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
