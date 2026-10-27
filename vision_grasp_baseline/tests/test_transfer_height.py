import numpy as np
import pytest

from vision_grasp_baseline import continuous_plan as planner


def test_failed_partial_transfer_is_discarded_and_lower_height_restarts_from_lift(monkeypatch):
    start=np.eye(4);start[2,3]=.1
    release=np.eye(4);release[:3,3]=[.1,0,.12]
    calls=[]
    def linear(node,name,q,gripper,pose,attached,**kwargs):
        calls.append((name,list(q),float(pose[2,3])))
        if name=='transfer' and pose[2,3]>.149:raise RuntimeError('no IK')
        return [x+.1 for x in q],dict(name=name)
    monkeypatch.setattr(planner,'plan_linear',linear)
    q,segments,rise,oriented,above,rejected=planner.plan_transfer(None,[0.]*7,start,release,None)
    assert len(rejected)==1 and rejected[0]['extra_clearance_m']==.03
    assert rise[2,3]==pytest.approx(.135)
    assert all(s['transfer_extra_clearance_m']==.015 for s in segments)
    starts=[q for name,q,z in calls if name=='vertical_clearance']
    assert starts==[[0.]*7,[0.]*7]
    assert [s['name'] for s in segments]==['vertical_clearance','transfer','vertical_release']


def test_all_invalid_heights_are_rejected_without_inventing_a_path(monkeypatch):
    start=np.eye(4);start[2,3]=.1
    release=start.copy();release[:3,3]=[.1,0,.12]
    def fail(*args,**kwargs):raise RuntimeError('path collision')
    monkeypatch.setattr(planner,'plan_linear',fail)
    with pytest.raises(RuntimeError,match='No validated transfer height'):
        planner.plan_transfer(None,[0.]*7,start,release,None)
