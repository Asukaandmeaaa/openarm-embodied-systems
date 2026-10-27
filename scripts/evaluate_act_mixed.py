#!/usr/bin/env python3
"""Held-out, teacher-forced ACT chunk errors; no robot or simulator control."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from act_mixed.data import ROOT, DEST, MASS
sys.path.insert(0, str(ROOT))
from scripts.act_trial.preflight import audit_checkpoint


def chunk_errors(pred, target, padding):
    pred, target = np.asarray(pred), np.asarray(target)
    valid = ~np.asarray(padding, dtype=bool)
    if pred.shape != (50, 16) or target.shape != pred.shape or valid.shape != (50,):
        raise ValueError('Expected 50x16 actions and 50 padding flags')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite action')
    errors = np.abs(pred - target)
    return {str(n): errors[:n][valid[:n]].mean(axis=0) for n in (1, 10, 50) if valid[:n].any()}


def summarize(episodes):
    domains = {}
    for domain in MASS:
        groups = defaultdict(list)
        for ep in episodes:
            if ep['domain'] == domain:
                groups[ep['group']].append(ep)
        if not groups:
            raise ValueError(f'No held-out episodes for {domain}')
        metrics = {}
        for horizon in ('1', '10', '50'):
            # Equal source groups, equal recordings within each group.
            error = np.mean([np.mean([r['mae_rad'][horizon] for r in rs], axis=0)
                             for rs in groups.values()], axis=0)
            metrics[horizon] = {'per_joint_mae_rad': error.tolist(),
                               'right_arm_mae_rad': float(error[:7].mean()),
                               'right_gripper_mae_rad': float(error[7])}
        domains[domain] = {'groups': len(groups), 'episodes': sum(map(len, groups.values())),
                           'horizons': metrics}
    return domains


def evaluate(checkpoint, root, manifest, device, samples, shared=None):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    contract = audit_checkpoint(checkpoint)
    cfg = ACTConfig.from_pretrained(str(checkpoint)); cfg.device = device
    policy = ACTPolicy.from_pretrained(str(checkpoint), config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': device}})
    repo_id = f'local/act_mixed_20261016_{root.name}'
    # Reuse identical held-out observations across checkpoints. This avoids
    # decoding the same two camera images five times; model inputs stay exact.
    shared = {} if shared is None else shared
    meta = shared.get('meta')
    if meta is None:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        shared['meta'] = meta
    deltas = resolve_delta_timestamps(cfg, meta)
    contract_key = (str(root.resolve()), deltas)
    if 'contract_key' in shared and shared['contract_key'] != contract_key:
        raise ValueError('Cannot share samples across different datasets or temporal contracts')
    shared['contract_key'] = contract_key
    ds = shared.get('dataset')
    if ds is None:
        ds = LeRobotDataset(repo_id, root=root, delta_timestamps=deltas)
        shared['dataset'] = ds
    cache = shared.setdefault('samples', {})
    if ds.num_episodes != len(manifest['episodes']):
        raise ValueError('Manifest episode count mismatch')
    episodes = []
    with torch.inference_mode():
        for ep in manifest['episodes']:
            index = ep['episode_index']
            start = int(ds.meta.episodes[index]['dataset_from_index'])
            end = int(ds.meta.episodes[index]['dataset_to_index'])
            if end - start != ep['frame_count']:
                raise ValueError('Manifest frame count mismatch')
            estimates = defaultdict(list)
            for i in np.unique(np.linspace(start, end - 1, samples, dtype=int)):
                if int(i) not in cache:
                    cache[int(i)] = ds[int(i)]
                row = cache[int(i)]
                obs = {k: v for k, v in row.items() if k.startswith('observation.')}
                policy.reset()
                # No target action is passed into the CVAE encoder during inference.
                pred = post(policy.predict_action_chunk(pre(obs))).detach().cpu().numpy()[0]
                for horizon, error in chunk_errors(pred, row['action'].numpy(), row['action_is_pad'].numpy()).items():
                    estimates[horizon].append(error)
            episodes.append({'episode_id': ep['episode_id'], 'domain': ep['domain'], 'group': ep['group'],
                             'mae_rad': {k: np.mean(v, axis=0).tolist() for k, v in estimates.items()}})
            print(f'EVAL {checkpoint.parent.name} {index + 1}/{ds.num_episodes}', flush=True)
    return {'checkpoint': str(checkpoint), 'file_sha256': contract['file_sha256'],
            'domains': summarize(episodes), 'episodes': episodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, nargs='+', default=[20000, 30000, 50000])
    ap.add_argument('--output-dir', type=Path, default=ROOT/'outputs/act_mixed_50k_20261016')
    ap.add_argument('--data-root', type=Path, default=DEST)
    ap.add_argument('--split', choices=['validation', 'test'], default='validation')
    ap.add_argument('--samples-per-episode', type=int, default=20)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = ap.parse_args()
    if args.samples_per_episode < 1: ap.error('samples-per-episode must be positive')
    if args.split == 'test' and len(args.steps) != 1:
        ap.error('Use validation to choose a checkpoint; test evaluates one selected checkpoint')
    if not (args.data_root/'READY.json').exists(): ap.error('Complete reviewed export required')
    checkpoints = [args.output_dir/'checkpoints'/f'{s:06d}'/'pretrained_model' for s in args.steps]
    for checkpoint in checkpoints: audit_checkpoint(checkpoint)
    manifest_path = args.data_root/args.split/'meta/mixed_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    result = {'scope': 'Held-out teacher-forced errors in radians, NOT closed-loop task success',
              'split': args.split, 'samples_per_episode': args.samples_per_episode,
              'aggregation': 'uniform query times; valid chunk frames only; equal source groups then recordings',
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'models': []}
    out = ROOT/'results/act_mixed_20261016'/f'{args.split}_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    shared = {}
    for checkpoint in checkpoints:
        result['models'].append(evaluate(checkpoint, args.data_root/args.split, manifest, args.device, args.samples_per_episode, shared))
        out.write_text(json.dumps(result, indent=2)+'\n')
    for model in result['models']:
        print(model['checkpoint'])
        for domain, metrics in model['domains'].items():
            h = metrics['horizons']['1']
            print(f"  {domain}: arm_MAE={h['right_arm_mae_rad']:.6f} gripper_MAE={h['right_gripper_mae_rad']:.6f} rad")
    print(f'Report: {out}')


if __name__ == '__main__': main()
