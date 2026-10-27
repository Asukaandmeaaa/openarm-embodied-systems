import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vision_grasp_baseline.closed_training import select,verify_sources,digest,phase_coverage
from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY,COLLECTION_EVENTS
from vision_grasp_baseline.prepare_closed_training import config_for


def populate(root,count=27):
    files=[]
    for i in range(count):
        path=root/'nearby/episodes'/f'episode_{i:03d}'
        path.mkdir(parents=True)
        f=path/'metadata.json';f.write_text(json.dumps(dict(review_status='approved',terminal_label='returned_zero',
            operator_marked_success=True,automatic_task_contract_passed=True,data_quality_passed=True,
            gripper_policy=COLLECTION_POLICY,source_trajectory_trial_id=f'physical_{i}',
            complete_trajectory_sha256=f'plan_{i}')))
        files.append(f)
    return files


def test_27_whole_episode_split_is_reproducible_and_pending_is_excluded(tmp_path):
    files=populate(tmp_path,28)
    d=json.loads(files[-1].read_text());d['review_status']='pending';files[-1].write_text(json.dumps(d))
    manifest=select(tmp_path)
    assert manifest==select(tmp_path)
    assert manifest['counts']['nearby']==dict(train=22,validation=5)
    assert not set(manifest['splits']['train'])&set(manifest['splits']['validation'])
    assert manifest['excluded']==[str(files[-1].parent)]


@pytest.mark.parametrize('field',['source_trajectory_trial_id','complete_trajectory_sha256'])
def test_same_source_or_replayed_plan_cannot_leak(tmp_path,field):
    files=populate(tmp_path)
    d=json.loads(files[1].read_text());d[field]=json.loads(files[0].read_text())[field]
    files[1].write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Duplicate'):select(tmp_path)


def test_frozen_source_hashes_detect_changes(tmp_path):
    f=tmp_path/'metadata.json';f.write_text('first')
    m=dict(source_sha256={str(tmp_path):{f.name:digest(f)}})
    verify_sources(m);f.write_text('second')
    with pytest.raises(ValueError,match='Frozen source changed'):verify_sources(m)


def test_phase_grid_covers_closed_departure_grasp_release_closed_return(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(dict(ros_time_mapping=dict(offset_ns=-10_000_000_000))))
    events=[dict(phase='continuous_dispatch',detail=dict(start_ros_ns=110_000_000_000,duration_s=10.))]
    events += [dict(phase='timeline_event',detail=dict(name=name,time_s=i)) for i,name in enumerate(COLLECTION_EVENTS)]
    f=tmp_path/'events.jsonl';f.write_text(''.join(json.dumps(row)+'\n' for row in events))
    report=dict(start_timestamp_s=99.,frame_count=361)
    coverage=phase_coverage(tmp_path,report)
    assert coverage['timeline_start_timestamp_s']==100.
    assert sum(coverage['frame_counts'].values())==361
    assert all(coverage['frame_counts'][k]>0 for k in COLLECTION_EVENTS)
    assert coverage['frame_counts']['return_closed']==60
    with pytest.raises(ValueError,match='loses'):phase_coverage(tmp_path,dict(start_timestamp_s=99.,frame_count=255))
    events.pop();f.write_text(''.join(json.dumps(row)+'\n' for row in events))
    with pytest.raises(ValueError,match='events'):phase_coverage(tmp_path,report)


def test_config_uses_full_validation_new_output_and_low_worker_count(tmp_path):
    cfg=config_for(tmp_path,'test_closed')
    assert cfg['max_eval_samples']==0 and cfg['eval_steps']==cfg['save_freq']==5000
    assert cfg['steps']==50000 and cfg['num_workers']==2
    assert cfg['policy']['chunk_size']==50 and cfg['policy']['n_action_steps']==10
    assert cfg['dataset']['root']==str(tmp_path/'train')
    assert cfg['output_dir'].endswith('outputs/test_closed_50k')


def trainer_module():
    source=Path(__file__).parents[2]/'scripts/train_act_closed.py'
    spec=importlib.util.spec_from_file_location('closed_trainer_tests',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('limit',[1,256,512])
def test_trainer_rejects_prefix_validation_even_if_cli_overrides_it(limit):
    with pytest.raises(ValueError,match='full validation'):
        trainer_module().validate_recipe(NS(max_eval_samples=limit,eval_steps=5000))


def test_disabled_validation_is_rejected():
    with pytest.raises(ValueError,match='full validation'):
        trainer_module().validate_recipe(NS(max_eval_samples=0,eval_steps=0))


def test_offline_evaluator_does_not_normalize_float_images_twice():
    torch=pytest.importorskip('torch')
    source=Path(__file__).parents[2]/'scripts/evaluate_act_closed.py'
    spec=importlib.util.spec_from_file_location('closed_eval_tests',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    key='observation.images.ceiling'
    for dtype,value in [(torch.uint8,128),(torch.float32,128/255)]:
        batch={key:torch.full((1,3,2,2),value,dtype=dtype),'observation.state':torch.zeros(1,16),
               'action':torch.ones(1,50,16)}
        result=module.policy_observation(batch,[key])
        assert result[key].float().mean().item()==pytest.approx(128/255)
        assert 'action' not in result and torch.equal(result['observation.state'],batch['observation.state'])


@pytest.mark.parametrize('mutation',[None,'group','hash','phase','policy','manifest'])
def test_training_entry_checks_frozen_exports_without_reading_mutable_collections(tmp_path,mutation):
    data=tmp_path/'frozen';data.mkdir()
    manifest=dict(gripper_policy=COLLECTION_POLICY,splits={},source_sha256={})
    reports={}
    for split in ('train','validation'):
        source=f'/no-live-collection-needed/{split}/episode_000'
        manifest['splits'][split]=[source]
        manifest['source_sha256'][source]={'metadata.json':'sha'}
        reports[split]=dict(gripper_policy=COLLECTION_POLICY,episodes=[dict(source_episode_path=source,
            source_trajectory_trial_id=split,source_sha256={'metadata.json':'sha'},
            phase_coverage=dict(all_task_phases_present=True),action_contract_audit=dict(passed=True))])
        (data/split/'meta').mkdir(parents=True)
    file=data/'split_manifest.json';file.write_text(json.dumps(manifest))
    ready=dict(gripper_policy=COLLECTION_POLICY,split_manifest_sha256=digest(file))
    if mutation=='group':reports['validation']['episodes'][0]['source_trajectory_trial_id']='train'
    if mutation=='hash':reports['validation']['episodes'][0]['source_sha256']['metadata.json']='changed'
    if mutation=='phase':reports['validation']['episodes'][0]['phase_coverage']['all_task_phases_present']=False
    if mutation=='policy':ready['gripper_policy']='legacy_open_transit_v1'
    if mutation=='manifest':file.write_text(file.read_text()+' ')
    (data/'READY.json').write_text(json.dumps(ready))
    for split,report in reports.items():
        (data/split/'meta/visual_expert_manifest.json').write_text(json.dumps(report))
    cfg=NS(max_eval_samples=0,eval_steps=5000,
           dataset=NS(episodes=None,eval_split=.2,root=str(data/'train')))
    if mutation is None:assert trainer_module().validate_recipe(cfg)[0]==data
    else:
        with pytest.raises(ValueError):trainer_module().validate_recipe(cfg)
