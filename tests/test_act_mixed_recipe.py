import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from act_mixed.data import assign_splits,episode_masses
from act_mixed.sampler import MixedEpisodeSampler


def test_domains_have_requested_total_mass_despite_unequal_replays():
 rows=[dict(domain='latest_real',group='a')]*5+[dict(domain='latest_real',group='b')]+[dict(domain='old_real',group='c')]*2+[dict(domain='simulation',group=str(i)) for i in range(30)]
 mass=episode_masses(rows);actual=defaultdict(float);groups=defaultdict(float)
 for r,w in zip(rows,mass):actual[r['domain']]+=w;groups[r['domain'],r['group']]+=w
 assert dict(actual)==pytest.approx({'latest_real':.6,'old_real':.3,'simulation':.1})
 assert groups['latest_real','a']==pytest.approx(groups['latest_real','b'])


def test_same_source_never_crosses_holdouts():
 rows=[]
 for i in range(40):
  for domain in ['latest_real','old_real']:rows.append(dict(domain=domain,group=f'real:20260929_{i}'))
 split=assign_splits(rows);seen={}
 for r in split:
  if r['group'] in seen:assert r['split']==seen[r['group']]
  seen[r['group']]=r['split']
 assert set(seen.values())=={'train','validation','test'}
 assert split==assign_splits(rows)


def test_sampler_frequency_and_exact_resume():
 sampler=MixedEpisodeSampler([0,10000,30000],[10000,30000,100000],episode_masses=[.6,.3,.1],shuffle=True,seed=1000)
 seq=np.array(list(sampler));fractions=[np.mean(seq<10000),np.mean((seq>=10000)&(seq<30000)),np.mean(seq>=30000)]
 assert fractions==pytest.approx([.6,.3,.1],abs=.006)
 resumed=MixedEpisodeSampler([0,10000,30000],[10000,30000,100000],episode_masses=[.6,.3,.1],shuffle=True,seed=1000)
 resumed.load_state_dict({'epoch':0,'start_index':1234})
 assert np.array_equal(np.array(list(resumed)),seq[1234:])


def test_dropped_frames_and_relative_indices_never_escape_episode():
 sampler=MixedEpisodeSampler([0,10],[10,20],episode_indices_to_use=[1],drop_n_first_frames=1,drop_n_last_frames=2,
   absolute_to_relative_idx={i:i-10 for i in range(10,20)},episode_masses=[.2,.8],seed=1)
 assert set(sampler)<=set(range(1,8))
 with pytest.raises(ValueError):MixedEpisodeSampler([0],[2],drop_n_last_frames=2,episode_masses=[1.])


def test_hold_gap_accepts_only_float32_rounding_noise():
 from convert_real_episodes_act import same_float32_target
 a=np.array([.00171664,.3],dtype=np.float32)
 b=np.nextafter(a,np.full_like(a,np.inf))
 assert same_float32_target(a,b)
 assert not same_float32_target(a,a+1e-5)
 assert not same_float32_target(a,np.nextafter(b,np.full_like(a,np.inf)))


def test_export_reload_chunk_shapes_and_weighted_stats(tmp_path,monkeypatch):
 import cv2,json
 import prepare_act_mixed as export
 from lerobot.policies.act.configuration_act import ACTConfig
 from lerobot.datasets.factory import resolve_delta_timestamps
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 image=tmp_path/'fixture.jpg';cv2.imwrite(str(image),np.zeros((360,640,3),dtype=np.uint8))
 rows=[dict(domain=d,group=d,episode_id=str(i),path=str(tmp_path),approved=True) for i,d in enumerate(['latest_real','old_real','simulation'])]
 def fixture(row):
  q=np.zeros((5,16),dtype=np.float32);a=np.full((5,16),float(row['episode_id']),dtype=np.float32)
  return q,a,{k:[image]*5 for k in ('ceiling','wrist_right')},{'fixture':True}
 monkeypatch.setattr(export,'prepared',fixture)
 root=tmp_path/'train';export.export_split(rows,root,'train')
 stats=json.loads((root/'meta/stats.json').read_text())
 assert stats['action']['mean']==pytest.approx([.5]*16)
 assert stats['action']['std']==pytest.approx([np.sqrt(.45)]*16)
 base=LeRobotDataset('local/act_mixed_20261016_train',root=root)
 config=ACTConfig(device='cpu',chunk_size=50,n_action_steps=10)
 ds=LeRobotDataset('local/act_mixed_20261016_train',root=root,delta_timestamps=resolve_delta_timestamps(config,base.meta))
 sample=ds[0]
 assert sample['action'].shape==(50,16)
 assert sample['action_is_pad'].shape==(50,)
 assert sample['observation.images.ceiling'].shape==(3,360,640)
 assert sample['observation.state'].shape==(16,)


def test_eval_ignores_padded_actions_and_separates_gripper():
 from evaluate_act_mixed import chunk_errors,summarize
 pred=np.zeros((50,16));target=pred.copy();target[:2,:7]=.2;target[:2,7]=.7;target[2:]=999
 padding=np.arange(50)>=2
 metrics=chunk_errors(pred,target,padding)
 assert metrics['50'][:7]==pytest.approx([.2]*7)
 assert metrics['50'][7]==pytest.approx(.7)
 rows=[dict(domain=d,group=d,mae_rad=metrics) for d in ['latest_real','old_real','simulation']]
 assert summarize(rows)['latest_real']['horizons']['50']['right_arm_mae_rad']==pytest.approx(.2)


def test_eval_reuses_identical_samples_between_models(tmp_path,monkeypatch):
 import torch
 from types import SimpleNamespace
 import evaluate_act_mixed as ev
 import lerobot.datasets.lerobot_dataset as ld
 import lerobot.datasets.factory as df
 import lerobot.policies.factory as pf
 from lerobot.policies.act.configuration_act import ACTConfig
 from lerobot.policies.act.modeling_act import ACTPolicy
 reads=[]
 metadata=SimpleNamespace(episodes=[{'dataset_from_index':i*5,'dataset_to_index':(i+1)*5} for i in range(3)])
 class Dataset:
  num_episodes=3;meta=metadata
  def __init__(self,*a,**kw):pass
  def __getitem__(self,i):
   reads.append(i)
   return {'observation.state':torch.zeros(16),'action':torch.zeros(50,16),'action_is_pad':torch.zeros(50,dtype=torch.bool)}
 class Policy:
  def to(self,*a):return self
  def eval(self):return self
  def reset(self):pass
  def predict_action_chunk(self,batch):return torch.zeros(1,50,16)
 monkeypatch.setattr(ev,'audit_checkpoint',lambda p:{'file_sha256':{}})
 monkeypatch.setattr(ACTConfig,'from_pretrained',lambda p:SimpleNamespace(device='cpu'))
 monkeypatch.setattr(ACTPolicy,'from_pretrained',lambda *a,**kw:Policy())
 monkeypatch.setattr(ld,'LeRobotDatasetMetadata',lambda *a,**kw:metadata)
 monkeypatch.setattr(ld,'LeRobotDataset',Dataset)
 monkeypatch.setattr(df,'resolve_delta_timestamps',lambda *a:{'action':[0.]})
 monkeypatch.setattr(pf,'make_pre_post_processors',lambda *a,**kw:(lambda x:x,lambda x:x))
 manifest={'episodes':[dict(domain=d,group=d,episode_id=d,episode_index=i,frame_count=5) for i,d in enumerate(['latest_real','old_real','simulation'])]}
 shared={}
 one=ev.evaluate(tmp_path/'first',tmp_path,manifest,'cpu',1,shared)
 two=ev.evaluate(tmp_path/'second',tmp_path,manifest,'cpu',1,shared)
 assert reads==[0,5,10]
 assert one['domains']==two['domains']
 with pytest.raises(ValueError,match='Cannot share'):
  ev.evaluate(tmp_path/'second',tmp_path/'different',manifest,'cpu',1,shared)
