#!/usr/bin/env python3
"""LeRobot ACT on an audited frozen closed-transit snapshot; no robot I/O."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vision_grasp_baseline.closed_training import digest
from vision_grasp_baseline.collection_io import ORDER,CAMERAS,write_json
from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY


def validate_recipe(cfg):
    if cfg.max_eval_samples!=0 or cfg.eval_steps<=0:
        raise ValueError('Closed ACT requires full validation: max_eval_samples=0 and eval_steps>0')
    if cfg.dataset.episodes is not None or cfg.dataset.eval_split<=0:
        raise ValueError('Use complete explicit train/validation exports, not another episode subset')
    data=Path(cfg.dataset.root).resolve().parent
    if Path(cfg.dataset.root).resolve()!=data/'train':raise ValueError('Expected frozen train export')
    manifest=json.loads((data/'split_manifest.json').read_text())
    ready=json.loads((data/'READY.json').read_text())
    if ready['split_manifest_sha256']!=digest(data/'split_manifest.json'):
        raise ValueError('Frozen manifest changed since export')
    if manifest['gripper_policy']!=COLLECTION_POLICY or ready['gripper_policy']!=COLLECTION_POLICY:
        raise ValueError('New training entry refuses old gripper datasets')
    groups={};sources={}
    for split in ('train','validation'):
        report=json.loads((data/split/'meta/visual_expert_manifest.json').read_text())
        sources[split]=[r['source_episode_path'] for r in report['episodes']]
        groups[split]={r['source_trajectory_trial_id'] for r in report['episodes']}
        if sources[split]!=manifest['splits'][split] or report['gripper_policy']!=COLLECTION_POLICY:
            raise ValueError('Frozen split/export mismatch')
        for row in report['episodes']:
            if not row['phase_coverage']['all_task_phases_present'] or not row['action_contract_audit']['passed']:
                raise ValueError('Incomplete phase/action provenance audit')
            expected=manifest['source_sha256'][row['source_episode_path']]
            if any(expected[name]!=value for name,value in row['source_sha256'].items()):
                raise ValueError('Export differs from frozen source')
    if set(sources['train'])&set(sources['validation']) or groups['train']&groups['validation']:
        raise ValueError('Train/validation physical trajectory leakage')
    return data,manifest,ready


def datasets(cfg):
    from lerobot.datasets.factory import make_dataset
    from vision_grasp_baseline.selected_column_reader import enable_selected_columns
    import lerobot
    data,manifest,ready=validate_recipe(cfg)
    train=enable_selected_columns(make_dataset(cfg))
    val_cfg=copy.deepcopy(cfg)
    val_cfg.dataset.root=str(data/'validation');val_cfg.dataset.repo_id='local/validation'
    val_cfg.dataset.image_transforms.enable=False
    val=enable_selected_columns(make_dataset(val_cfg))
    val.meta.stats=copy.deepcopy(train.meta.stats)
    for split,ds in (('train',train),('validation',val)):
        if ds.fps!=30 or set(ds.meta.camera_keys)!={f'observation.images.{k}' for k in CAMERAS}:
            raise ValueError('Expected 30Hz top/wrist RGB observations')
        for key in ('observation.state','action'):
            feature=ds.meta.features[key]
            if list(feature['shape'])!=[16] or list(feature['names'])!=ORDER:
                raise ValueError('Expected native 16 joint order: '+key)
        if len(ds)!=ready['frames'][split]:raise ValueError('Export frame count changed')
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'closed_training_recipe.json',dict(hardware_io=False,split=manifest,
        lerobot_version=getattr(lerobot,'__version__','unknown'),
        trainer_sha256=digest(__file__),normalization='TRAIN statistics only; full validation',
        initialization='resume' if cfg.resume else 'fresh ACT + cached ImageNet ResNet18',
        complete_validation_frames=len(val),train_frames=len(train),
        deployment_ready=False,deployment_requirement='Must match position+velocity controller and closed-transit start; not legacy ACT position-only runtime'))
    print(f'CLOSED_ACT_DATASET: train={len(train)} validation={len(val)}; ALL validation frames, complete task; no hardware I/O',flush=True)
    return train,val


def main():
    from lerobot.policies.act.configuration_act import ACTConfig # register config
    from lerobot.scripts import lerobot_train
    base_logging=lerobot_train.init_logging
    log=Path(os.environ.get('OPENARM_TRAIN_LOG',ROOT/'results/act_training/act_closed_v2_20261023a_50k.log'))
    log.parent.mkdir(parents=True,exist_ok=True)
    def logging(*,accelerator=None,**kwargs):
        return base_logging(log_file=log,accelerator=accelerator,**kwargs)
    lerobot_train.init_logging=logging
    lerobot_train.make_train_eval_datasets=datasets
    lerobot_train.main()


if __name__=='__main__':main()
