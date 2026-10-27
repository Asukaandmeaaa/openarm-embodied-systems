#!/usr/bin/env python3
"""Fresh ACT, reviewed real captures only, 70/30 source sampling. No robot I/O."""
import copy
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/lerobot/act_mixed_20261016'
MASS = {'latest_real': .7, 'old_real': .3}


def selection(data=DATA):
    manifests = {s: json.loads((data/s/'meta/mixed_manifest.json').read_text())
                 for s in ('train', 'validation', 'test')}
    selected = {s: [r for r in m['episodes'] if r['domain'] in MASS]
                for s, m in manifests.items()}
    for rows in selected.values():
        if set(r['domain'] for r in rows) != set(MASS):
            raise ValueError('Both reviewed real domains required in every split')
        if any(not r.get('approved') for r in rows):
            raise ValueError('Unapproved capture')
    groups = {s: {r['group'] for r in rows} for s, rows in selected.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if groups[a] & groups[b]:
            raise ValueError('Source group leakage')
    return manifests, selected


def masses(rows, total):
    result = [0.] * total  # Simulation is not selected and has no sampling mass.
    for domain, mass in MASS.items():
        records = [r for r in rows if r['domain'] == domain]
        groups = Counter(r['group'] for r in records)
        for r in records:
            result[r['episode_index']] = mass / len(groups) / groups[r['group']]
    return result


def real_stats(rows, weights, data=DATA):
    """Read only numeric TRAIN columns, never holdouts or image/sim statistics."""
    import numpy as np
    import pyarrow.dataset as ds
    indices = [r['episode_index'] for r in rows]
    table = ds.dataset(data/'train/data', format='parquet').to_table(
        columns=['episode_index', 'action', 'observation.state'],
        filter=ds.field('episode_index').isin(indices))
    ep = np.asarray(table['episode_index'])
    stats = {}
    for key in ('action', 'observation.state'):
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if values.shape != (len(ep), 16) or not np.isfinite(values).all():
            raise ValueError('Invalid physical training vectors')
        mean = np.zeros(16); second = np.zeros(16)
        for row in rows:
            x = values[ep == row['episode_index']]
            if len(x) != row['frame_count']:
                raise ValueError('Manifest and parquet frame counts differ')
            w = weights[row['episode_index']]
            mean += w*x.mean(0); second += w*np.square(x).mean(0)
        stats[key] = {'mean': mean, 'std': np.sqrt(np.maximum(second-mean**2, 0)),
                      'min': values.min(0), 'max': values.max(0), 'count': np.array([len(ep)])}
    return stats


def main():
    audit_steps = 30000
    audit_mode = '--audit' in sys.argv
    if audit_mode:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--audit', action='store_true')
        parser.add_argument('--steps', type=int, choices=[30000, 50000], default=30000)
        audit_steps = parser.parse_args().steps
    manifests, selected = selection()
    weights = masses(selected['train'], len(manifests['train']['episodes']))
    stats = real_stats(selected['train'], weights)
    recipe = {
        'initialization': 'fresh ACT; ImageNet ResNet18 backbone, no ACT checkpoint',
        'steps': audit_steps, 'domain_mass': MASS, 'simulation_selected': False,
        'new_data_scope': 'latest_real captures, including re-recorded legacy trajectories',
        'sampling': 'domain -> uniform source group -> uniform recording -> uniform frame',
        'selected': selected,
        'stats': {k: {n: v.tolist() for n, v in d.items()} for k, d in stats.items()},
        'manifest_sha256': {s: hashlib.sha256((DATA/s/'meta/mixed_manifest.json').read_bytes()).hexdigest()
                            for s in manifests},
    }
    if audit_mode:
        print(json.dumps({k: recipe[k] for k in ('initialization', 'steps', 'domain_mass', 'new_data_scope')} | {
            'counts': {s: dict(Counter(r['domain'] for r in rows)) for s, rows in selected.items()},
            'frames': {s: {d: sum(r['frame_count'] for r in rows if r['domain'] == d) for d in MASS}
                       for s, rows in selected.items()},
            'sampling_mass_verified': {d: sum(weights[r['episode_index']] for r in selected['train'] if r['domain'] == d) for d in MASS},
            'normalization': 'recomputed from real TRAIN only at 70/30',
            'hardware_io': False, 'training_started': False}, indent=2))
        return
    from lerobot.policies.act.configuration_act import ACTConfig  # registry
    from lerobot.scripts import lerobot_train
    from lerobot.datasets.factory import make_dataset
    from act_mixed.sampler import MixedEpisodeSampler

    def datasets(cfg):
        if cfg.resume or cfg.policy.pretrained_path:
            raise ValueError('Fresh training only; no ACT checkpoint or resume override')
        if cfg.steps not in (30000, 50000) or Path(cfg.dataset.root).resolve() != DATA/'train' or cfg.dataset.episodes is not None:
            raise ValueError('Do not override audited recipe')
        recipe['steps'] = cfg.steps
        cfg.dataset.episodes = [r['episode_index'] for r in selected['train']]
        train = make_dataset(cfg)
        train.meta.stats.update(copy.deepcopy(stats))
        val_cfg = copy.deepcopy(cfg)
        val_cfg.dataset.root = DATA/'validation'
        val_cfg.dataset.repo_id = 'local/act_real7030_validation'
        val_cfg.dataset.episodes = [r['episode_index'] for r in selected['validation']]
        validation = make_dataset(val_cfg)
        validation.meta.stats = copy.deepcopy(train.meta.stats)
        MixedEpisodeSampler.episode_masses = weights
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        recipe['wrapper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (out/'real7030_recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
        return train, validation

    lerobot_train.make_train_eval_datasets = datasets
    lerobot_train.EpisodeAwareSampler = MixedEpisodeSampler
    lerobot_train.main()


if __name__ == '__main__':
    main()
