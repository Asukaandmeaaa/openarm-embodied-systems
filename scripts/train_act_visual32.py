#!/usr/bin/env python3
"""ACT on the frozen 26/6 visual grasp split; no robot control interfaces."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vision_grasp_baseline.prepare_training import DATA


def datasets(cfg):
    # eval_split enables periodic validation in this installed LeRobot version;
    # the explicit exported 26/6 roots below supply it, without another split.
    from lerobot.datasets.factory import make_dataset
    manifest=json.loads((DATA/'split_manifest.json').read_text())
    if Path(cfg.dataset.root).resolve()!=DATA/'train' or cfg.dataset.episodes is not None:
        raise ValueError('This entry requires the complete frozen visual32 train dataset')
    sources={}
    for split in ('train','validation'):
        exported=json.loads((DATA/split/'meta/visual_expert_manifest.json').read_text())
        sources[split]=[row['source_episode_path'] for row in exported['episodes']]
        if sources[split]!=manifest['splits'][split]:raise ValueError('Frozen split/export mismatch')
    if set(sources['train'])&set(sources['validation']):raise ValueError('Train/validation leakage')
    train=make_dataset(cfg)
    val_cfg=copy.deepcopy(cfg);val_cfg.dataset.root=str(DATA/'validation');val_cfg.dataset.repo_id='local/validation'
    val=make_dataset(val_cfg)
    val.meta.stats=copy.deepcopy(train.meta.stats)
    if train.fps!=30 or val.fps!=30:raise ValueError('Expected 30Hz dataset')
    if train.meta.features['action']['shape']!=(16,) and list(train.meta.features['action']['shape'])!=[16]:
        raise ValueError('Expected 16-dimensional native joint target')
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'visual32_recipe.json').write_text(json.dumps(dict(split=manifest,normalization='TRAIN statistics only',
        initialization='resume' if cfg.resume else 'fresh ACT with ImageNet ResNet18',hardware_io=False,
        cameras=list(train.meta.camera_keys),fps=30),ensure_ascii=False,indent=2)+'\n')
    return train,val


def main():
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_train
    from lerobot.datasets.factory import make_train_eval_datasets
    base_logging=lerobot_train.init_logging
    log=Path(os.environ.get('OPENARM_TRAIN_LOG',ROOT/'results/act_training/act_visual32_20k_20261022.log'))
    log.parent.mkdir(parents=True,exist_ok=True)
    def logging(*,accelerator=None,**kwargs):
        return base_logging(log_file=log,accelerator=accelerator,**kwargs)
    lerobot_train.init_logging=logging
    lerobot_train.make_train_eval_datasets=datasets
    lerobot_train.main()


if __name__=='__main__':main()
