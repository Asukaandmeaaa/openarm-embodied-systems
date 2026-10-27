#!/usr/bin/env python3
"""Audit or export reviewed real + successful sim data; never trains a model."""
import argparse,json,os,sys
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from act_mixed.data import ROOT,DEST,MASS,ORDER,CAMERAS,TASK,inventory,assign_splits,episode_masses,prepared


def export_split(rows,dest,split):
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 features={'observation.state':{'dtype':'float32','shape':(16,),'names':ORDER},'action':{'dtype':'float32','shape':(16,),'names':ORDER}}
 features.update({f'observation.images.{k}':{'dtype':'image','shape':(360,640,3),'names':['height','width','channels']} for k in CAMERAS})
 if not rows:raise ValueError(f'No eligible episodes in {split}')
 if dest.exists():raise FileExistsError(dest)
 ds=LeRobotDataset.create(repo_id=f'local/act_mixed_20261016_{split}',root=dest,fps=30,features=features,
                          robot_type='openarm',use_videos=False,image_writer_threads=4)
 masses=episode_masses(rows) if split=='train' else None
 moments=[];manifest=[]
 for ep,row in enumerate(rows):
  state,action,images,audit=prepared(row)
  for i in range(len(state)):
   frame={'observation.state':state[i].copy(),'action':action[i].copy(),'task':TASK}
   for k in CAMERAS:
    image=cv2.imread(str(images[k][i]))
    if image is None:raise ValueError(f'Image decode failed: {images[k][i]}')
    frame[f'observation.images.{k}']=cv2.cvtColor(cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
   ds.add_frame(frame)
  ds.save_episode()
  moments.append((action.astype(np.float64).mean(0),np.square(action.astype(np.float64)).mean(0)))
  manifest.append(dict(row,episode_index=ep,frame_count=len(state),audit=audit,sampling_mass=masses[ep] if masses else None))
  print(f'SAVED {split} {ep+1}/{len(rows)} {row["domain"]} {row["episode_id"]}',flush=True)
 ds.finalize()
 # Match action normalization to the intended training distribution, excluding
 # validation/test. State is IDENTITY and visual stats use ImageNet constants.
 if masses:
  statpath=dest/'meta/stats.json';stats=json.loads(statpath.read_text())
  mean=sum(w*x[0] for w,x in zip(masses,moments));second=sum(w*x[1] for w,x in zip(masses,moments))
  stats['action']['mean']=mean.tolist();stats['action']['std']=np.sqrt(np.maximum(second-mean**2,0)).tolist()
  statpath.write_text(json.dumps(stats,indent=2)+'\n')
 (dest/'meta/mixed_manifest.json').write_text(json.dumps(dict(schema_version=1,split=split,domain_mass=MASS,
   sampling='domain -> uniform source group -> uniform recording -> uniform valid frame',
   initialization='fresh ACT, ImageNet ResNet18 backbone; no previous ACT checkpoint',episodes=manifest),indent=2)+'\n')


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit',action='store_true');ap.add_argument('--export',action='store_true')
 args=ap.parse_args()
 if not (args.audit or args.export):ap.error('choose --audit or --export')
 rows=assign_splits(inventory());reportdir=ROOT/'results/act_mixed_20261016';reportdir.mkdir(parents=True,exist_ok=True)
 report={'counts':dict(Counter(r['domain'] for r in rows)),'review_counts':dict(Counter(f'{r["domain"]}:{r["review"]}' for r in rows)),
         'domain_mass':MASS,'episodes':rows,'quality_errors':[]}
 if args.audit:
  for i,row in enumerate(rows):
   try:
    q,a,imgs,audit=prepared(row);row['audit']=audit
   except (ValueError,KeyError,OSError) as e:report['quality_errors'].append(dict(path=row['path'],error=str(e)))
   if i%25==0:print(f'AUDIT {i+1}/{len(rows)}',flush=True)
 (reportdir/'data_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2),flush=True)
 if args.export:
  unresolved=[r['episode_id'] for r in rows if r['domain']!='simulation' and not r['approved'] and r['review']!='rejected']
  if unresolved:raise ValueError('Human outcome review still pending; use review_real_episodes.py: '+', '.join(unresolved))
  selected=[r for r in rows if r['approved']]
  # Audit every selected episode before creating any export output.
  for r in selected:prepared(r)
  for split in ('train','validation','test'):
   if (DEST/split).exists():raise FileExistsError(f'No overwrite/resume of partial export: {DEST/split}')
  for split in ('train','validation','test'):export_split([r for r in selected if r['split']==split],DEST/split,split)
  (DEST/'READY.json').write_text(json.dumps({'domain_mass':MASS,'splits':{s:sum(r['split']==s for r in selected) for s in ('train','validation','test')}},indent=2)+'\n')

if __name__=='__main__':main()
