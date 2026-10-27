import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from vision_grasp_baseline import continuous_plan as planner
from vision_grasp_baseline.hover import box
from vision_grasp_baseline.pickup_clearance import padded_cube,require_pickup_clearance
from vision_grasp_baseline.continuous_timing import assemble
from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY


@pytest.mark.parametrize('margin',[0.,-.01,.029,float('nan'),float('inf')])
def test_clearance_cannot_be_omitted_or_reduced(margin):
    with pytest.raises(ValueError):require_pickup_clearance(dict(pickup_clearance_m=margin))
    with pytest.raises(ValueError):padded_cube(box('cube',[0.]*3,[.04]*3),margin)


def test_padding_preserves_true_object_pose_and_attachment_geometry():
    cube=box('vision_baseline_cube',[.3,-.1,.28],[.04]*3,.5)
    padded=padded_cube(cube)
    assert padded.id==cube.id and padded.primitive_poses==cube.primitive_poses
    assert list(padded.primitives[0].dimensions)==pytest.approx([.1]*3)
    assert list(cube.primitives[0].dimensions)==pytest.approx([.04]*3)
    with pytest.raises(ValueError,match='Old trajectory'):require_pickup_clearance({})


@pytest.mark.parametrize('collision',[False,True])
def test_final_controller_interpolant_checks_inflated_cube_then_restores_nominal(monkeypatch,collision):
    levels=[0.,.2,.3,.5,.4,0.]
    phases=[dict(times=[0.,2.],positions=[[a]*7,[b]*7],velocities=[[0.]*7]*2,
                 accelerations=[[0.]*7]*2) for a,b in zip(levels,levels[1:])]
    plan=assemble(phases,dict(velocity=10.,acceleration=.5,jerk=10.),COLLECTION_POLICY)
    limits={k:[v]*7 for k,v in dict(velocity=100.,acceleration=100.,jerk=100.,lower=-1.,upper=1.).items()}
    grip={k:v for k,v in dict(velocity=100.,acceleration=100.,jerk=100.,lower=-1.,upper=1.).items()}
    plan.update(limits=limits,gripper_limits=grip,pickup_clearance_m=.03)
    scene=dict(table_collision_envelope=dict(size=[1.,1.,1.]),measured=dict(
        bin=dict(center_xy=[.3,0.],support_z=.2,yaw=0.)))
    cube=box('vision_baseline_cube',[.3,-.1,.28],[.04]*3)
    active=[cube];checks=[];restored=[]
    node=NS(precise_scene=lambda *a:(None,cube),attached_at=lambda *a:object(),
            finger_contacts=Mock(),valid_client=object(),
            update_object=lambda obj:active.__setitem__(0,copy.deepcopy(obj)),
            state=lambda q,g,attached=None:(q,g))
    def call(client,request):
        checks.append((float(request.robot_state[0][0]),active[0].primitives[0].dimensions[0]))
        return NS(valid=not collision,contacts=[])
    node.call=call
    monkeypatch.setattr(planner,'read_limits',lambda *a:(limits,grip,{}))
    monkeypatch.setattr(planner,'ChainFK',lambda *a:NS(pose=lambda *a:np.eye(4)))
    monkeypatch.setattr(planner,'scene_attachment',lambda *a,**kw:restored.append(kw))
    if collision:
        with pytest.raises(RuntimeError,match='Interpolated path collision'):
            planner.validate_timeline(node,plan,scene,[0.]*3,'unused')
        assert checks[0][1]==pytest.approx(.1)
        assert restored[-1]['world'] is cube
        return
    result=planner.validate_timeline(node,plan,scene,[0.]*3,'unused')
    assert result['passed'] and result['checked_states']>100
    assert result['pickup_clearance_m']==.03 and result['pickup_clearance_until']=='approach'
    assert checks[0][1]==pytest.approx(.1)
    assert any(width==pytest.approx(.1) and q==pytest.approx(.2) for q,width in checks)
    assert any(width==pytest.approx(.04) and q>.2 for q,width in checks)
    assert restored[-1]['world'] is cube
