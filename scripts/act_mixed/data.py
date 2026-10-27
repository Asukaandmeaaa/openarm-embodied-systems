"""Source inventory, deterministic group split and compatible raw readers."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sys
import numpy as np
import yaml
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from convert_real_episodes_act import prepare as prepare_real,ORDER,CAMERAS,TASK
REAL_ROOTS={
 'latest_real':[ROOT/'openarm_kinesthetic_data/collections_real14_20261016_smooth_v1/episodes_pending_review',ROOT/'openarm_kinesthetic_data/collections_legacy25_20261016_smooth_v1/episodes_pending_review'],
 'old_real':[ROOT/'openarm_kinesthetic_data/collections/episodes_pending_review'],
}
SIM_ROOT=ROOT/'data/sim/openarm_randomized_aux300_20261016'
MASS={'latest_real':.6,'old_real':.3,'simulation':.1}
DEST=ROOT/'data/lerobot/act_mixed_20261016'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
 rows=[]
 for domain,roots in REAL_ROOTS.items():
  for root in roots:
   for meta in sorted(root.glob('*/metadata.json')):
    v=json.loads(meta.read_text());source=v.get('source_trajectory_trial_id') or v.get('trial_id') or meta.parent.name
    rows.append(dict(domain=domain,path=str(meta.parent),episode_id=meta.parent.name,group='real:'+source,
      review=v.get('review_status','unreviewed'),approved=v.get('review_status')=='approved' and v.get('operator_marked_success') is True,
      metadata_sha256=sha(meta)))
 sm=yaml.safe_load((SIM_ROOT/'metadata.yaml').read_text())
 for v in sorted(sm['episodes'],key=lambda x:int(x['id'])):
  rows.append(dict(domain='simulation',path=str(SIM_ROOT/'episodes'/str(v['id'])),episode_id=str(v['id']),
   group='sim:'+str(v['seed']),approved=v.get('success') is True,review='sim_success' if v.get('success') is True else 'rejected',
   seed=v['seed'],randomization=v['randomization'],sim_metadata_sha256=sha(SIM_ROOT/'metadata.yaml')))
 return rows


def assign_splits(rows,seed=1000):
 # Group before filtering reviews so human decisions do not move other groups.
 pools=defaultdict(set)
 for r in rows:
  pool='simulation' if r['domain']=='simulation' else ('new_teaching' if r['group'].startswith('real:20261016_') else 'old_teaching')
  pools[pool].add(r['group'])
 mapping={}
 for pool,groups in pools.items():
  ordered=sorted(groups,key=lambda s:hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
  n=max(1,round(len(ordered)*.1)) if len(ordered)>=3 else 0
  for i,g in enumerate(ordered):mapping[g]='validation' if i<n else 'test' if i<2*n else 'train'
 return [dict(r,split=mapping[r['group']]) for r in rows]


def episode_masses(rows):
 # Domain -> source trajectory group -> recording -> time frame.
 pools=defaultdict(lambda:defaultdict(list))
 for i,r in enumerate(rows):pools[r['domain']][r['group']].append(i)
 if set(pools)!=set(MASS):raise ValueError('Every training domain needs at least one eligible episode')
 masses=[0.]*len(rows)
 for domain,groups in pools.items():
  for members in groups.values():
   for i in members:masses[i]=MASS[domain]/len(groups)/len(members)
 return masses


def prepared(row):
 path=Path(row['path'])
 if row['domain']!='simulation':
  streams,q,ids,report=prepare_real(path,30)
  state=q['joint_observation'][ids['joint_observation']]
  action=q['demonstration_action'][ids['demonstration_action']]
  images={k:[path/streams[k][j]['path'] for j in ids[k]] for k in CAMERAS}
  return state,action,images,report
 parts={};stamps=None;hashes={}
 for kind in ('obs','action'):
  for side in ('right','left'):
   p=path/kind/'arms'/side/'state.parquet';table=pq.read_table(p)
   t=table['timestamp'].cast('int64').to_numpy();q=np.array(table['qpos'].to_pylist(),dtype=np.float32)
   if len(t)<2 or np.any(np.diff(t)<=0) or q.shape!=(len(t),8) or not np.isfinite(q).all():raise ValueError(f'Invalid sim stream: {p}')
   if stamps is not None and not np.array_equal(t,stamps):raise ValueError('Sim state/action clocks differ')
   stamps=t;parts[kind,side]=q;hashes[str(p.relative_to(path))]=sha(p)
 # Samples were scheduled at 30 Hz on a 1 ms simulator clock. Resample
 # causally to exact 30 Hz instead of relabeling unequal timestamps.
 grid=stamps[0]+np.rint(np.arange(int(np.floor((stamps[-1]-stamps[0])*30/1e9))+1)*1e9/30).astype(np.int64)
 ids=np.searchsorted(stamps,grid,side='right')-1
 if np.max(grid-stamps[ids])>50_000_000:raise ValueError(f'Sim state gap: {path}')
 state=np.concatenate([parts['obs','right'],parts['obs','left']],axis=1)[ids]
 action=np.concatenate([parts['action','right'],parts['action','left']],axis=1)[ids]
 images={k:[path/'cameras'/k/f'{int(stamps[i])}.jpeg' for i in ids] for k in CAMERAS}
 if any(not p.is_file() for paths in images.values() for p in paths):raise ValueError(f'Missing sim image: {path}')
 return state,action,images,dict(frame_count=len(grid),duration_s=(len(grid)-1)/30,source_sha256=hashes,
   alignment='causal exact 30 Hz grid from original synchronized simulation timestamps')
