"""Freeze reviewed closed-transit episodes, audit, and export an ACT snapshot."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

from .closed_training import SOURCE_FILES,digest,select,verify_sources
from .collection_io import write_json
from .gripper_settings import COLLECTION_POLICY

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_NAME='act_closed_v2_20261023a'


def config_for(data,name):
    cfg=json.loads((ROOT/'configs/training/act_visual32_20261022.json').read_text())
    cfg['dataset']['root']=str(data/'train')
    cfg['dataset']['eval_split']=.2 # Enables eval, but explicit frozen roots supply the split.
    cfg.update(output_dir=str(ROOT/'outputs'/f'{name}_50k'),job_name=f'{name}_50k',
               steps=50000,num_workers=2,prefetch_factor=1,persistent_workers=False,
               seed=20261023,eval_steps=5000,max_eval_samples=0,save_freq=5000)
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default=DEFAULT_NAME)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+',a.name):p.error('name must be a simple directory name')
    source=ROOT/'vision_grasp_baseline/collections';data=ROOT/'data/lerobot'/a.name
    manifest_path=data/'split_manifest.json';ready=data/'READY.json'
    data.mkdir(parents=True,exist_ok=True)
    lock=(data/'.prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());verify_sources(manifest)
        print('沿用冻结清单；后续新增采集不会并入本版。',flush=True)
    else:
        manifest=select(source)
        manifest['source_sha256']={path:{name:digest(Path(path)/name) for name in SOURCE_FILES}
                                  for paths in manifest['splits'].values() for path in paths}
        write_json(manifest_path,manifest)
    print(json.dumps(dict(counts=manifest['counts'],dataset=str(data)),ensure_ascii=False),flush=True)
    for split,episodes in manifest['splits'].items():
        episode_list=data/f'{split}_episodes.json';write_json(episode_list,episodes)
        out=data/split;proof=out/'meta/visual_expert_manifest.json'
        if out.exists():
            if not proof.exists():raise ValueError('Incomplete export; do not overwrite: '+str(out))
            report=json.loads(proof.read_text())
            if [r['source_episode_path'] for r in report['episodes']]!=episodes:
                raise ValueError('Frozen split/export mismatch')
            for row in report['episodes']:
                if not row['action_contract_audit']['passed'] or not row['phase_coverage']['all_task_phases_present']:
                    raise ValueError('Missing action/phase audit')
            continue
        cmd=[sys.executable,'-m','vision_grasp_baseline.export_act','--source',str(source),
             '--output',str(out),'--episode-list',str(episode_list),'--gripper-policy',COLLECTION_POLICY,
             '--original-jpeg']
        if a.audit_only:cmd.append('--audit-only')
        subprocess.run(cmd,cwd=ROOT,check=True)
    verify_sources(manifest)
    cfg_path=ROOT/'configs/training'/f'{a.name}_50k.json';cfg=config_for(data,a.name)
    if cfg_path.exists() and json.loads(cfg_path.read_text())!=cfg:raise ValueError('Refusing to replace changed training config')
    write_json(cfg_path,cfg)
    if not a.audit_only:
        write_json(ready,dict(hardware_io=False,split_manifest_sha256=digest(manifest_path),
            config=str(cfg_path),gripper_policy=COLLECTION_POLICY,frames={split:
                json.loads((data/split/'meta/info.json').read_text())['total_frames'] for split in manifest['splits']}))
    print(json.dumps(dict(audit_only=a.audit_only,config=str(cfg_path),ready=ready.exists()),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
