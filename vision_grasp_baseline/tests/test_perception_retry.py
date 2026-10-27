from pathlib import Path

import pytest

from vision_grasp_baseline.perception_retry import StageFailure,capture_stable_scene


def test_new_batch_replaces_rejected_batch_and_keeps_attempt_evidence(tmp_path):
    calls=[];attempts=[]
    def run(stage,*args):
        calls.append((stage,args))
        if stage=='detect' and len(calls)==2:
            raise StageFailure('detect',3,'first.log','unstable cube')
    scene=capture_stable_scene(run,tmp_path,'execution_scene','calibration',attempts)
    assert scene==tmp_path/'execution_scene_retry_1/scene.json'
    assert [stage for stage,args in calls]==['capture','detect','capture','detect']
    assert calls[0][1][-1]==10 and calls[2][1][-1]==20
    assert [a['accepted'] for a in attempts]==[False,True]


def test_persistent_instability_stops_without_returning_a_scene(tmp_path):
    attempts=[];captures=[]
    def run(stage,*args):
        if stage=='capture':captures.append(args[1])
        else:raise StageFailure('detect',3,'bad.log','unstable')
    with pytest.raises(RuntimeError,match='3 fresh image batches'):
        capture_stable_scene(run,tmp_path,'scene','calibration',attempts)
    assert len(set(captures))==3
    assert not any(a['accepted'] for a in attempts)


@pytest.mark.parametrize('stage,code',[('capture',1),('detect',1)])
def test_other_failures_are_not_hidden_by_retry(tmp_path,stage,code):
    calls=[]
    def run(current,*args):
        calls.append(current)
        if current==stage:raise StageFailure(stage,code,'failure.log','camera or code error')
    with pytest.raises(StageFailure):capture_stable_scene(run,tmp_path,'scene','calibration',[])
    assert calls.count('capture')==1
