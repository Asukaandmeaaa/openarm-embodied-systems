"""Frozen whole-episode stratified split and native-resolution ACT export."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
DATA=ROOT/'data/lerobot/visual_act32_20261022'


def selection(source):
    rng=random.Random(20261022);splits={'train':[],'validation':[]};counts={}
    for tag,expected,holdout in [('nearby',10,2),('spread',16,3),('rotation',6,1)]:
        paths=[]
        for file in sorted((source/tag/'episodes').glob('episode_*/metadata.json')):
            meta=json.loads(file.read_text())
            if meta.get('review_status')=='approved' and meta.get('terminal_label')=='returned_zero' and not meta.get('diagnostic_only') and all(
                    meta.get(k) is True for k in ('operator_marked_success','automatic_task_contract_passed','data_quality_passed')):
                paths.append(str(file.parent.resolve()))
        if len(paths)!=expected:raise ValueError(f'{tag}: expected frozen {expected} approved episodes, found {len(paths)}')
        rng.shuffle(paths);splits['validation'].extend(paths[:holdout]);splits['train'].extend(paths[holdout:])
        counts[tag]=dict(train=expected-holdout,validation=holdout)
    groups=[]
    for split in ('train','validation'):
        values=[json.loads((Path(p)/'metadata.json').read_text())['source_trajectory_trial_id'] for p in splits[split]]
        if len(values)!=len(set(values)):raise ValueError('Duplicate source trajectory identifiers')
        groups.append(set(values))
    if groups[0]&groups[1]:raise ValueError('Source trajectory leakage between train and validation')
    return dict(seed=20261022,counts=counts,splits=splits,fps=30,image_size_wh=[640,480],hardware_io=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-only',action='store_true');args=p.parse_args()
    source=BASE/'collections';manifest=selection(source);DATA.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'split_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Split manifest changed; use a new dataset name instead of changing a frozen split')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for split,episodes in manifest['splits'].items():
        episode_list=DATA/(split+'_episodes.json');episode_list.write_text(json.dumps(episodes,indent=2)+'\n')
        output=DATA/split
        if output.exists():
            proof=output/'meta/visual_expert_manifest.json'
            if not proof.exists():raise RuntimeError('Incomplete export; inspect and move aside before retry: '+str(output))
            exported=json.loads(proof.read_text())
            if [r['source_episode_path'] for r in exported['episodes']]!=episodes:
                raise ValueError('Existing dataset does not match frozen split')
            for row in exported['episodes']:
                for name,digest in row['source_sha256'].items():
                    if hashlib.sha256((Path(row['source_episode_path'])/name).read_bytes()).hexdigest()!=digest:
                        raise ValueError('Source episode changed since export')
            print('已存在且来源匹配：'+str(output),flush=True);continue
        cmd=[sys.executable,'-m','vision_grasp_baseline.export_act','--source',str(source),
             '--episode-list',str(episode_list),'--output',str(output)]
        if args.audit_only:cmd.append('--audit-only')
        subprocess.run(cmd,cwd=ROOT,check=True)
    print(json.dumps(dict(train=26,validation=6,dataset=str(DATA),audit_only=args.audit_only),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
