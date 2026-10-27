import pytest

from vision_grasp_baseline.perception_retry import StageFailure
from vision_grasp_baseline.planning_retry import plan_complete


def test_second_complete_candidate_is_selected_and_first_preserved(tmp_path):
    calls=[];attempts=[]
    def run(stage,*args):
        calls.append((stage,args))
        if len(calls)==1:raise StageFailure(stage,1,'bad.log','no collision-free path')
    result=plan_complete(run,tmp_path,'scene','registration','model',attempts)
    assert result==tmp_path/'complete_plan_retry_1/trajectory.json'
    assert [a['accepted'] for a in attempts]==[False,True]
    assert len(calls)==2 and all(stage=='continuous-plan' for stage,args in calls)
    assert calls[0][1][:-1]==calls[1][1][:-1]


def test_three_invalid_candidates_never_dispatch_motion(tmp_path):
    calls=[];attempts=[]
    def run(stage,*args):
        calls.append(stage)
        raise StageFailure(stage,1,'bad.log','path collision')
    with pytest.raises(RuntimeError,match='3 complete trajectory candidates failed'):
        plan_complete(run,tmp_path,'scene','registration','model',attempts)
    assert calls==['continuous-plan']*3
    assert len(attempts)==3 and not any(a['accepted'] for a in attempts)


def test_interrupt_or_configuration_error_is_not_retried(tmp_path):
    calls=[]
    def run(stage,*args):
        calls.append(stage)
        raise StageFailure(stage,2,'bad.log','bad configuration')
    with pytest.raises(StageFailure):plan_complete(run,tmp_path,'scene','registration','model',[])
    assert len(calls)==1
