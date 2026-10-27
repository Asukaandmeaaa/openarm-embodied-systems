"""Offline only: batch sequencing, provenance, and fail-closed capture."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from openarm_kinesthetic_recorder.capture_preflight import capture_preflight,batch_suffix
from openarm_kinesthetic_recorder.real_episode_recorder import validate_trial_metadata
from openarm_kinesthetic_recorder.episode_capture_button import EpisodeCaptureButton


def test_inline_metadata_does_not_claim_prior_checks():
    payload=dict(batch_id='test',trial_id='20261016_130312_001',position_id='pose_007',
                 cube_forward_m=0.,cube_lateral_m=0.,placement_measurement_method='per_episode_marked_origin',
                 source_trajectory_trial_id='20261016_113607_007',
                 **capture_preflight('inline',False,False))
    result=validate_trial_metadata(payload)
    assert result['capture_validation_mode']=='inline'
    assert result['plan_only_passed'] is False and result['dry_replay_passed'] is False
    assert result['source_trajectory_trial_id']==payload['source_trajectory_trial_id']
    payload.pop('capture_validation_mode')
    with pytest.raises(ValueError):validate_trial_metadata(payload)


@pytest.mark.parametrize('mode,plan,dry',[
    ('unknown',False,False),('inline',True,False),('inline',False,True),
    ('inline','false',False),('prior_verified',True,False),('prior_verified',False,True),
])
def test_invalid_or_false_claims_rejected(mode,plan,dry):
    with pytest.raises(ValueError):capture_preflight(mode,plan,dry)


def test_resume_order_skips_first_only_and_rejects_bad_index():
    paths=[Path(f'{x}.json') for x in ['005','007','008']]
    assert batch_suffix(paths,2)==paths[1:]
    assert batch_suffix(paths,1)==paths
    for n in [0,4,-1,True,'2']:
        with pytest.raises(ValueError):batch_suffix(paths,n)


def test_saved_batch_advances_but_waits_for_new_b_and_updates_position():
    paths=[Path('/tmp/trajectories/20261016_113607_007.json'),Path('/tmp/trajectories/20261016_113639_008.json')]
    logs=[]
    node=SimpleNamespace(stage='SAVING',trajectories=paths,trajectory_index=0,
                         metadata={},position_id_from_source=True,allocate_new_episode_ids=False,
                         auto_finish_future=SimpleNamespace(done=lambda:True,result=lambda:SimpleNamespace(success=True,message='pending')),
                         get_logger=lambda:SimpleNamespace(warn=logs.append,error=logs.append))
    node._episode_trial_id=lambda:EpisodeCaptureButton._episode_trial_id(node)
    node.select_trajectory=lambda i:EpisodeCaptureButton.select_trajectory(node,i)
    EpisodeCaptureButton.on_timer(node)
    assert node.stage=='READY' and node.trajectory_index==1
    assert node.metadata['position_id']=='pose_20261016_113639_008'
    assert node.metadata['source_trajectory_trial_id']=='20261016_113639_008'
    # READY timer cannot initiate recording or motion without a new B event.
    EpisodeCaptureButton.on_timer(node)
    assert node.stage=='READY'
    node.stage='SAVING'
    EpisodeCaptureButton.on_timer(node)
    assert node.stage=='COMPLETE'


def test_failed_player_does_not_advance_or_save():
    logs=[]
    node=SimpleNamespace(stage='REPLAYING',trajectory_index=0,player=SimpleNamespace(poll=lambda:2),
                         get_logger=lambda:SimpleNamespace(warn=logs.append,error=logs.append))
    EpisodeCaptureButton.on_timer(node)
    assert node.stage=='FAILED' and node.trajectory_index==0
