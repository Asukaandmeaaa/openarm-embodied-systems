#!/usr/bin/env python3
"""Convert reviewed real JSONL episodes to a timestamp-aligned LeRobot dataset.

The converter is deliberately review-gated: replay/audit success is not a
substitute for a human confirmation that the cube was grasped, placed, and
released.  Use ``--allow-pending`` only to run a non-training audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = [name for side in ('right', 'left') for name in
         [*[f'openarm_{side}_joint{i}' for i in range(1, 8)], f'openarm_{side}_finger_joint1']]
CAMERAS = ('ceiling', 'wrist_right')
TASK = 'Pick up the blue cube with the right arm and place it into the black box.'


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def times(rows):
    ts = np.asarray([r['timestamp_s'] for r in rows], dtype=np.float64)
    if len(ts) < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Stream timestamps must be finite and strictly increasing')
    return ts


def joints(rows):
    arrays = []
    for row in rows:
        if len(row['order']) != 16 or set(row['order']) != set(ORDER):
            raise ValueError('Unexpected joint names')
        arrays.append([row['positions'][row['order'].index(n)] for n in ORDER])
    q = np.asarray(arrays, dtype=np.float32)
    if q.shape != (len(rows), 16) or not np.isfinite(q).all():
        raise ValueError('Invalid joint positions')
    return q


def same_float32_target(a, b):
    """Allow one representation ULP, not a physical-motion tolerance."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    ulp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    return bool(np.all(np.abs(a-b) <= ulp))


def prepare(path, fps):
    streams = {k: read(path / f'{k}.jsonl') for k in ('joint_observation', 'demonstration_action')}
    cameras = read(path / 'camera.jsonl')
    streams.update({k: [r for r in cameras if r['key'] == k] for k in CAMERAS})
    ts = {k: times(rows) for k, rows in streams.items()}
    q = {k: joints(streams[k]) for k in ('joint_observation', 'demonstration_action')}
    start, end = max(t[0] for t in ts.values()), min(t[-1] for t in ts.values())
    if end-start < 1:
        raise ValueError(f'No usable overlap: {path}')
    grid = start + np.arange(int(np.floor((end-start)*fps))+1)/fps
    ids = {k: np.searchsorted(t, grid, side='right')-1 for k, t in ts.items()}
    ages = {k: grid-ts[k][ids[k]] for k in ts}
    # A recorded constant controller target may legitimately span the terminal gap.
    action_ts = ts['demonstration_action']
    for i in np.flatnonzero(np.diff(action_ts) > .1):
        if not same_float32_target(q['demonstration_action'][i], q['demonstration_action'][i+1]):
            raise ValueError(f'Unexplained action gap: {path}, sample {i}')
    if ages['joint_observation'].max() > .05:
        raise ValueError(f'Stale joint observations in active interval: {path}')
    for key in CAMERAS:
        if ages[key].max() > .5:
            raise ValueError(f'Camera gap exceeds 0.5s: {path}/{key}')
        for row in streams[key]:
            image_path = (path / row['path']).resolve()
            if not image_path.is_relative_to(path.resolve()) or not image_path.is_file():
                raise ValueError(f'Missing or unsafe image path: {image_path}')
    meta = json.loads((path/'metadata.json').read_text())
    source_group = meta.get('source_trajectory_trial_id') or meta.get('trial_id') or path.name
    report = {
        'source_episode': path.name, 'frame_count': len(grid),
        'source_trajectory_trial_id': source_group,
        'start_timestamp_s': float(start), 'end_timestamp_s': float(grid[-1]),
        'duration_s': float(grid[-1]-start),
        'trimmed_camera_prefix_s': float(start-min(ts[k][0] for k in CAMERAS)),
        'original_review': {k: meta.get(k) for k in ('review_status','operator_marked_success','terminal_label','automatic_task_contract_passed')},
        'action_gap_hold_precision': 'one float32 ULP; original samples unchanged',
        'alignment': {k: {'max_age_ms': float(a.max()*1000), 'p99_age_ms': float(np.quantile(a,.99)*1000), 'frames_age_over_100ms': int((a>.1).sum())} for k,a in ages.items()},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [path/'metadata.json',path/'camera.jsonl',path/'joint_observation.jsonl',path/'demonstration_action.jsonl']},
    }
    return streams, q, ids, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'openarm_kinesthetic_data/collections/episodes_pending_review')
    parser.add_argument('--output', type=Path, default=ROOT/'data/lerobot/openarm_real_fixed13_20260930')
    parser.add_argument('--report', type=Path, default=ROOT/'results/act_real_fixed13/conversion_report.json')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH','HEIGHT'), default=(640,360))
    parser.add_argument('--allow-pending', action='store_true', help='permit pending episodes; never use for training')
    parser.add_argument('--manifest', type=Path, help='JSON split manifest mapping train/validation/test to episode IDs')
    parser.add_argument('--roles', nargs='+', choices=('train', 'validation', 'test'), default=('train', 'validation'),
                        help='manifest roles to export (default: train validation; keep test held out)')
    parser.add_argument('--repo-id', default=None, help='LeRobot repo ID; defaults to local/<output-folder-name>')
    parser.add_argument('--pattern', default='*', help='episode directory glob when no manifest is supplied')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('fps must be positive')
    if args.fps != 30 or tuple(args.image_size) != (640, 360):
        raise ValueError('The frozen ACT contract requires --fps 30 and --image-size 640 360')
    paths = sorted(p.parent for p in args.source.glob(f'{args.pattern}/metadata.json'))
    if not paths:
        raise ValueError(f'No episode metadata found under {args.source}')
    metas = {p.name: json.loads((p/'metadata.json').read_text()) for p in paths}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        selected = [eid for role in args.roles for eid in manifest.get(role, [])]
        unknown = sorted(set(selected) - set(metas))
        if unknown:
            raise ValueError(f'Manifest references unknown episodes: {unknown}')
        if len(selected) != len(set(selected)):
            raise ValueError('Manifest contains duplicate episode IDs')
        paths = [args.source/eid for eid in selected]
        metas = {eid: metas[eid] for eid in selected}
    if not args.allow_pending:
        pending = [eid for eid,m in metas.items() if m.get('review_status') != 'approved' or m.get('operator_marked_success') is not True]
        if pending:
            raise ValueError('Human review required before export; pending/unapproved episodes: ' + ', '.join(pending))
    prepared = [prepare(p,args.fps) for p in paths]
    report = {'source': str(args.source.resolve()), 'output': str(args.output.resolve()),
              'fps': args.fps, 'image_size_wh': list(args.image_size), 'joint_order': ORDER,
              'alignment_rule': 'latest sample at or before grid timestamp; no smoothing or joint interpolation; retain recorded terminal constant target gaps',
              'scope': 'Reviewed episodes selected by the caller; source trajectory groups are preserved for leakage-safe splitting.',
              'task': TASK, 'episodes': [v[3] for v in prepared]}
    for i, ep in enumerate(report['episodes']):
        ep['episode_index'] = i
        print(ep['source_episode'],ep['frame_count'],'frames',ep['alignment'],flush=True)
    report['total_frames'] = sum(ep['frame_count'] for ep in report['episodes'])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if args.audit_only:
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {'observation.state': {'dtype':'float32','shape':(16,),'names':ORDER},
                'action': {'dtype':'float32','shape':(16,),'names':ORDER}}
    features.update({f'observation.images.{k}': {'dtype':'image','shape':(360,640,3),'names':['height','width','channels']} for k in CAMERAS})
    repo_id = args.repo_id or f'local/{args.output.name}'
    ds = LeRobotDataset.create(repo_id=repo_id,root=args.output,fps=args.fps,
                               features=features,robot_type='openarm',use_videos=False,
                               image_writer_threads=4)
    for path,(streams,q,ids,ep) in zip(paths,prepared):
        for i in range(ep['frame_count']):
            frame = {'observation.state':q['joint_observation'][ids['joint_observation'][i]].copy(),
                     'action':q['demonstration_action'][ids['demonstration_action'][i]].copy(), 'task':TASK}
            for key in CAMERAS:
                img_path = path/streams[key][ids[key][i]]['path']
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f'Image cannot decode: {img_path}')
                frame[f'observation.images.{key}'] = cv2.cvtColor(cv2.resize(img,tuple(args.image_size),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            ds.add_frame(frame)
        ds.save_episode()
        print('SAVED',path.name,flush=True)
    ds.finalize()
    (args.output/'meta/real_source_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Completed: {len(paths)} episodes, {report["total_frames"]} frames',flush=True)


if __name__ == '__main__':
    main()
