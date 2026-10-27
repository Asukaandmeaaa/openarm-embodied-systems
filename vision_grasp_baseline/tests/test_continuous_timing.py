import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.interpolate import BPoly

from vision_grasp_baseline.continuous_timing import assemble,controller_bounds,path_knots
from vision_grasp_baseline.continuous_execute import build_goals,validate_proof
from vision_grasp_baseline.collection_io import finish_interrupted_episode
from unittest.mock import patch
from vision_grasp_baseline.gripper_settings import (COLLECTION_POLICY,COLLECTION_EVENTS,require_current_plan,
                                                   TRANSIT_POSITION,CLOSE_POSITION,TRANSIT_SETTLE_S)


def test_preflight_proof_binds_scene_path_model_and_expiry():
    binding=dict(trajectory_sha256='path',scene_sha256='scene',model_sha256='model')
    proof=dict(success=True,check_only=True,physical_motion=False,validation=dict(passed=True),
               file_binding=binding,finished_wall_ns=100_000_000_000)
    assert validate_proof(proof,binding,101_000_000_000)['passed']
    for now in [99_000_000_000,131_000_000_000]:
        with pytest.raises(ValueError):validate_proof(proof,binding,now)
    for key in binding:
        changed={**binding,key:'changed'}
        with pytest.raises(ValueError):validate_proof(proof,changed,101_000_000_000)
    for key,value in [('success',False),('physical_motion',True),('check_only',False),('validation',dict(passed=False))]:
        with pytest.raises(ValueError):validate_proof({**proof,key:value},binding,101_000_000_000)


def phase(q0,q1):
    return dict(times=[0.,2.],positions=[q0,q1],velocities=[[0.]*7]*2,
                accelerations=[[0.]*7]*2)


def test_complete_timeline_has_only_two_gripper_holds_and_synchronized_goals():
    phases=[phase([0.]*7,[.2]*7),phase([.2]*7,[.4]*7),phase([.4]*7,[0.]*7)]
    plan=assemble(phases,dict(velocity=10.,acceleration=.5,jerk=10.))
    assert [x['name'] for x in plan['events']]==['reach','close','carry','open','return']
    goals=build_goals(plan)
    assert len(goals[0].trajectory.joint_trajectory.points)==len(goals[1].trajectory.points)
    for arm,gripper,row in zip(goals[0].trajectory.joint_trajectory.points,goals[1].trajectory.points,plan['samples']):
        assert arm.time_from_start==gripper.time_from_start
        assert list(arm.positions)+list(gripper.positions)==row['positions']
    rows=plan['samples'];t=np.array([x['time_s'] for x in rows])
    assert np.all(np.diff(t)>0)
    q,v,a=[np.array([x[k] for x in rows]) for k in ('positions','velocities','accelerations')]
    bounds,_,_=controller_bounds(t,q,v,a)
    assert bounds[1][-1]<=.5
    assert bounds[2][-1]<=10.
    for event in plan['events']:
        if event['name'] not in ('close','open'):continue
        i=np.argmin(abs(t-event['time_s']))
        np.testing.assert_array_equal(q[i,:7],q[i+1,:7])
        assert q[i,7]!=q[i+1,7]
    assert q[-1,7]==pytest.approx(-.45)


def test_audit_detects_between_knot_speed_and_position_overshoot():
    t=np.array([0.,1.]);q=np.array([[0.],[0.]]);v=np.array([[4.],[-4.]]);a=np.zeros((2,1))
    bounds,lo,hi=controller_bounds(t,q,v,a)
    poly=BPoly.from_derivatives(t,[[q[0],v[0],a[0]],[q[1],v[1],a[1]]])
    assert hi[0]>1.
    grid=np.linspace(0,1,1001)
    for k in range(1,4):assert np.max(abs(poly(grid,k)))<=bounds[k-1][0]+1e-8
    assert np.min(poly(grid))>=lo[0]-1e-8


def closed_plan():
    points=[0.,.2,.3,.5,.4,0.]
    return assemble([phase([a]*7,[b]*7) for a,b in zip(points,points[1:])],
                    dict(velocity=10.,acceleration=.5,jerk=10.),COLLECTION_POLICY)


def test_closed_transit_four_stationary_gripper_moves_and_synchronized_goals():
    plan=closed_plan();require_current_plan(plan)
    assert tuple(e['name'] for e in plan['events'])==COLLECTION_EVENTS
    assert plan['samples'][0]['positions'][7]==plan['samples'][-1]['positions'][7]==TRANSIT_POSITION
    events={e['name']:e['time_s'] for e in plan['events']}
    assert next(r for r in plan['samples'] if r['time_s']==events['carry'])['positions'][7]==CLOSE_POSITION
    held=[r for r in plan['samples'] if events['transit_close']<=r['time_s']<=events['return_closed']]
    assert held[-1]['time_s']-held[-2]['time_s']==pytest.approx(TRANSIT_SETTLE_S)
    arm,grip=build_goals(plan)
    assert [p.time_from_start for p in arm.trajectory.joint_trajectory.points]==[
        p.time_from_start for p in grip.trajectory.points]
    rows=plan['samples'];t=np.array([r['time_s'] for r in rows])
    assert np.all(np.diff(t)>0)
    q,v,a=[np.asarray([r[k] for r in rows]) for k in ('positions','velocities','accelerations')]
    bounds,_,_=controller_bounds(t,q,v,a)
    assert bounds[0][-1]<=10. and bounds[1][-1]<=.5 and bounds[2][-1]<=10.


@pytest.mark.parametrize('mutation',['legacy','v1','events','end','position','velocity','target'])
def test_rejects_legacy_or_nonstationary_closed_plan(mutation):
    plan=closed_plan()
    if mutation=='legacy':plan.pop('gripper_policy')
    elif mutation=='v1':plan['gripper_policy']='closed_transit_v1'
    elif mutation=='events':plan['events'].pop(1)
    elif mutation=='end':plan['samples'][-1]['positions'][7]=-.45
    else:
        t=plan['events'][1]['time_s']
        row=next(r for r in plan['samples'] if r['time_s']>t)
        if mutation=='position':row['positions'][0]+=.01
        elif mutation=='velocity':row['velocities'][0]=.01
        else:
            t=plan['events'][2]['time_s']
            next(r for r in plan['samples'] if r['time_s']==t)['positions'][7]=-.085
    with pytest.raises(ValueError):require_current_plan(plan)


def test_geometry_refuses_hidden_jump_between_plans():
    def segment(name,a,b):
        return dict(name=name,trajectory=dict(joint_trajectory=dict(points=[dict(positions=[a]*7),dict(positions=[b]*7)])))
    with pytest.raises(ValueError,match='Discontinuous'):
        path_knots([segment('first',0,.2),segment('second',.21,.4)])
    q=path_knots([segment('first',0,.2),segment('second',.2,.4)])
    np.testing.assert_array_equal(q[0],[0.]*7);np.testing.assert_array_equal(q[-1],[.4]*7)


def test_parent_saves_failed_recording_if_executor_is_terminated(tmp_path):
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps(dict(automatic_task_contract_passed=False)))
    def finish(operation,**kwargs):
        assert operation=='finish' and kwargs=={'success':False}
        metadata.write_text(json.dumps(dict(end_timestamp_s=10.,data_quality_passed=True)))
    with patch('vision_grasp_baseline.collection_io.rpc',side_effect=finish) as call:
        finish_interrupted_episode(tmp_path,'tracking guard')
        assert call.call_count==1
    saved=json.loads(metadata.read_text())
    assert saved['data_quality_passed'] and not saved['automatic_task_contract_passed']
    assert saved['terminal_label']=='failed_or_interrupted'
    assert saved['execution_error']=='tracking guard'


@pytest.mark.parametrize('closed',[False,True])
def test_actual_retimer_keeps_endpoint_rest_and_continuous_interior(tmp_path,closed):
    root=Path(__file__).resolve().parents[2]
    interpreter=root/'.venv-trajectory-processing/bin/python'
    def segment(a,b):
        return dict(name='line',trajectory=dict(joint_trajectory=dict(points=[dict(positions=[a]*7),dict(positions=[b]*7)])))
    limits=dict(velocity=[1.]*7,acceleration=[.5]*7,jerk=[5.]*7,lower=[-1.]*7,upper=[1.]*7)
    source=dict(phase_plans=[[segment(0,.1),segment(.1,.2)],[segment(.2,.4)],[segment(.4,0)]],
                limits=limits,gripper_limits=dict(velocity=10.,acceleration=.5,jerk=10.))
    if closed:
        source['phase_plans']=[[segment(0,.1),segment(.1,.2)],[segment(.2,.3)],
                              [segment(.3,.4)],[segment(.4,.3)],[segment(.3,0)]]
        source['gripper_policy']=COLLECTION_POLICY
    inp=tmp_path/'input.json';out=tmp_path/'output.json';inp.write_text(json.dumps(source))
    subprocess.run([str(interpreter),'-m','vision_grasp_baseline.continuous_timing',str(inp),str(out)],cwd=root,check=True)
    result=json.loads(out.read_text());p=result['phases'][0]
    if closed:require_current_plan(result)
    q=np.asarray(p['positions']);v=np.asarray(p['velocities']);a=np.asarray(p['accelerations'])
    np.testing.assert_allclose(q[[0,-1]],[[0.]*7,[.2]*7],atol=1e-10)
    np.testing.assert_allclose(v[[0,-1]],0,atol=1e-10);np.testing.assert_allclose(a[[0,-1]],0,atol=1e-10)
    assert abs(v[np.argmin(abs(q[:,0]-.1)),0])>.01
    for key,limit in [('velocity_bound',1.),('acceleration_bound',.5),('jerk_bound',5.)]:
        assert max(p['audit'][key])<=limit


def test_near_limit_endpoint_does_not_create_artificial_jerk_slowdown(tmp_path):
    root=Path(__file__).resolve().parents[2]
    def segment(a,b):
        return dict(name='line',trajectory=dict(joint_trajectory=dict(points=[dict(positions=[a]*7),dict(positions=[b]*7)])))
    limits=dict(velocity=[1.]*7,acceleration=[.5]*7,jerk=[5.]*7,lower=[-.17453]*7,upper=[1.]*7)
    source=dict(phase_plans=[[segment(0,.1)],[segment(.1,-.16685)],[segment(-.16685,0)]],
                limits=limits,gripper_limits=dict(velocity=10.,acceleration=.5,jerk=10.))
    inp=tmp_path/'source.json';out=tmp_path/'timing.json';inp.write_text(json.dumps(source))
    subprocess.run([str(root/'.venv-trajectory-processing/bin/python'),'-m','vision_grasp_baseline.continuous_timing',str(inp),str(out)],cwd=root,check=True)
    result=json.loads(out.read_text())
    for phase in result['phases']:
        assert phase['audit']['time_scale_for_jerk']<1.5
        assert max(phase['audit']['jerk_bound'])<=5.
        assert np.min(phase['positions'])>=-.17453
        np.testing.assert_allclose(np.asarray(phase['velocities'])[[0,-1]],0.,atol=1e-8)
    np.testing.assert_allclose(result['phases'][1]['positions'][-1],[-.16685]*7,atol=1e-10)
