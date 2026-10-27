#!/usr/bin/env python3
"""Run existing LeRobot ACT with explicit grouped holdouts and source sampling.
No installed LeRobot files are modified. Use this entry for initial AND resume.
"""
import copy,json,os,hashlib,logging
from pathlib import Path
from lerobot.policies.act.configuration_act import ACTConfig  # register JSON policy type
from lerobot.scripts import lerobot_train
from lerobot.datasets.factory import make_dataset
from act_mixed.sampler import MixedEpisodeSampler
from act_mixed.data import ROOT,MASS


def datasets(cfg):
    train_root=Path(cfg.dataset.root).resolve();root=train_root.parent
    if train_root.name!='train' or not (root/'READY.json').exists():raise ValueError('Complete reviewed mixed export required (READY.json missing)')
    manifests={s:json.loads((root/s/'meta/mixed_manifest.json').read_text()) for s in ('train','validation','test')}
    groups={s:{r['group'] for r in m['episodes']} for s,m in manifests.items()}
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Source trajectory leakage between splits')
    if manifests['train']['domain_mass']!=MASS:raise ValueError('Sampling recipe differs from expected 60/30/10')
    if cfg.dataset.episodes is not None:raise ValueError('Do not override the prepared train episode selection')
    train=make_dataset(cfg)
    validation_cfg=copy.deepcopy(cfg)
    validation_cfg.dataset.root=root/'validation';validation_cfg.dataset.repo_id='local/act_mixed_20261016_validation'
    validation=make_dataset(validation_cfg)
    rows=manifests['train']['episodes']
    if len(rows)!=train.num_episodes:raise ValueError('Export and mixed manifest counts differ')
    MixedEpisodeSampler.episode_masses=[r['sampling_mass'] for r in rows]
    # Snapshot the custom wrapper/recipe alongside standard LeRobot checkpoints.
    recipe={'sampling':manifests['train'],'manifest_sha256':{s:hashlib.sha256((root/s/'meta/mixed_manifest.json').read_bytes()).hexdigest() for s in manifests},
            'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sampler_sha256':hashlib.sha256((ROOT/'scripts/act_mixed/sampler.py').read_bytes()).hexdigest()}
    out=Path(cfg.output_dir);snapshot=out/'mixed_training_recipe.json'
    if cfg.resume and (not snapshot.exists() or json.loads(snapshot.read_text())!=recipe):raise ValueError('Resume recipe/code differs from original run')
    out.mkdir(parents=True,exist_ok=True);snapshot.write_text(json.dumps(recipe,indent=2)+'\n')
    logging.info('MIXED SAMPLING latest_real=0.60 old_real=0.30 simulation=0.10; grouped holdouts; validation=%s',validation.num_episodes)
    return train,validation


base_logging=lerobot_train.init_logging

def logged(*,accelerator=None,**kwargs):
    p=ROOT/'results/act_mixed_20261016/training.log';p.parent.mkdir(parents=True,exist_ok=True)
    return base_logging(log_file=p,accelerator=accelerator,**kwargs)

lerobot_train.make_train_eval_datasets=datasets
lerobot_train.EpisodeAwareSampler=MixedEpisodeSampler
lerobot_train.init_logging=logged
if __name__=='__main__':lerobot_train.main()
