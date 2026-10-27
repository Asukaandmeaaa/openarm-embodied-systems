from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vision_grasp_baseline.empty_gripper import StableEmptyClosure,ReturnClosureGuard
from vision_grasp_baseline.gripper_settings import require_closed_transit,TRANSIT_POSITION


@pytest.mark.parametrize('actual',[-.1204,-.085,-.0551,-.06,.0001,float('nan')])
def test_screenshot_and_open_residual_are_no_longer_accepted(actual):
    with pytest.raises(ValueError):require_closed_transit(actual)


@pytest.mark.parametrize('actual',[-.055,-.0494,-.02,0.])
def test_operator_confirmed_closed_position_and_explicit_bounds_are_accepted(actual):
    require_closed_transit(actual,TRANSIT_POSITION)


def test_grasp_target_cannot_be_used_as_empty_transit_command():
    with pytest.raises(ValueError):require_closed_transit(TRANSIT_POSITION,-.085)


def test_stability_requires_continuously_fresh_feedback_and_restarts_after_opening():
    gate=StableEmptyClosure()
    assert not gate.update(-.03,10.,10.)
    assert not gate.update(-.03,10.,10.4) # One old sample cannot prove a hold.
    assert not gate.update(-.03,10.5,10.5)
    assert not gate.update(-.12,10.7,10.7)
    assert not gate.update(-.03,10.8,10.8)
    for stamp in (10.85,10.90,10.95,11.0,11.05,11.11):
        ready=gate.update(-.03,stamp,stamp)
    assert ready


def guard():
    return ReturnClosureGuard(dict(events=[dict(name='transit_close',time_s=10.),
                                          dict(name='return_closed',time_s=13.)]))


def test_unclosed_gripper_fails_before_arm_return_time():
    gate=guard()
    gate.check(12.4,-.1204,100.,100.)
    with pytest.raises(RuntimeError,match='before return'):
        gate.check(12.75,-.1204,100.35,100.35)
    assert 12.75<gate.return_time and not gate.confirmed


@pytest.mark.parametrize('actual',[-.03,-.0494])
def test_closed_gripper_can_return_but_reopening_is_rejected(actual):
    gate=guard()
    for i in range(10):gate.check(12.3+i*.05,actual,100.+i*.05,100.+i*.05)
    assert gate.confirmed
    gate.check(12.8,actual,100.5,100.5)
    with pytest.raises(RuntimeError):gate.check(12.85,-.12,100.55,100.55)


def test_unobserved_feedback_gap_does_not_prove_stable_closure():
    gate=StableEmptyClosure()
    assert not gate.update(-.02,100.,100.)
    assert not gate.update(-.02,100.5,100.5)


def test_missing_feedback_history_cannot_pass_late_check():
    with pytest.raises(RuntimeError):guard().check(13.1,-.02,100.,100.)


@pytest.mark.parametrize('actual,passed',[(-.03,True),(-.0494,True),(-.1204,False)])
def test_zero_acceptance_observes_stable_finger_without_sending_commands(monkeypatch,actual,passed):
    import rclpy
    from vision_grasp_baseline import empty_gripper
    from vision_grasp_baseline.grasp import FINGER
    clock=[100.]
    node=SimpleNamespace(passive_state=False,current_positions=[0.]*7,
        full_joints=SimpleNamespace(name=[FINGER],position=[actual]),
        full_joint_state_monotonic_s=100.,execution_table_guard_failure=lambda:'')
    monkeypatch.setattr(empty_gripper.time,'monotonic',lambda:clock[0])
    def spin(*a,**kw):
        clock[0]+=.01
        node.full_joint_state_monotonic_s=clock[0]
    monkeypatch.setattr(rclpy,'spin_once',spin)
    if passed:
        result=empty_gripper.verify_stationary_empty_gripper(node,[0.]*7,timeout=.6)
        assert result['passed'] and result['actual_gripper_rad']==actual
    else:
        with pytest.raises(RuntimeError,match='no collection started'):
            empty_gripper.verify_stationary_empty_gripper(node,[0.]*7,timeout=.6)


def test_return_zero_request_fixes_gripper_without_adding_an_arm_group_finger_goal(monkeypatch):
    from vision_grasp_baseline.continuous_plan import plan_empty_zero
    from vision_grasp_baseline.grasp import FINGER
    from vision_grasp_baseline.hover import NAMES
    from openarm_kinesthetic_recorder import home_runtime
    from openarm_kinesthetic_recorder.home_runtime import HomeRuntimeNode
    captured=[]
    future=SimpleNamespace(result=lambda:SimpleNamespace(motion_plan_response='plan'))
    client=SimpleNamespace(call_async=lambda request:(captured.append(request) or future))
    node=SimpleNamespace(plan_client=client,wait_for_service=Mock(),
                         payload=lambda q:dict(joint_names=NAMES,start_positions_rad=list(q),move_group='right_arm'))
    node.plan=lambda *a:HomeRuntimeNode.plan(node,*a)
    monkeypatch.setattr(home_runtime.rclpy,'spin_until_future_complete',lambda *a,**kw:None)
    assert plan_empty_zero(node,[.1]*7)=='plan'
    request=captured[0].motion_plan_request
    assert list(request.start_state.joint_state.name)==[*NAMES,FINGER]
    assert list(request.start_state.joint_state.position)==[.1]*7+[TRANSIT_POSITION]
    assert [c.joint_name for c in request.goal_constraints[0].joint_constraints]==NAMES


def test_executor_cancels_both_goals_before_unclosed_return(monkeypatch):
    from concurrent.futures import Future
    import numpy as np
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from vision_grasp_baseline import continuous_execute as executor
    from vision_grasp_baseline.continuous_timing import assemble
    from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY
    from vision_grasp_baseline.grasp import FINGER
    from vision_grasp_baseline.hover import NAMES
    levels=[0.,.2,.3,.5,.4,0.]
    phases=[dict(times=[0.,2.],positions=[[a]*7,[b]*7],velocities=[[0.]*7]*2,
                 accelerations=[[0.]*7]*2) for a,b in zip(levels,levels[1:])]
    plan=assemble(phases,dict(velocity=10.,acceleration=.5,jerk=10.),COLLECTION_POLICY)
    times={e['name']:e['time_s'] for e in plan['events']}
    gate_time=times['return_closed']-.25
    moments=iter([*times.values()][:8]+[gate_time-.1,gate_time])
    elapsed=[-1.]
    monkeypatch.setattr(executor.time,'monotonic',lambda:100.+elapsed[0])
    clock=SimpleNamespace(now=lambda:Time(nanoseconds=round((elapsed[0]+1.)*1e9)))
    handles=[];clients=[]
    for _ in range(2):
        handle=SimpleNamespace(accepted=True,cancel_goal_async=Mock(return_value=Future()),
                               get_result_async=lambda:Future())
        handles.append(handle)
        sent=Future();sent.set_result(handle)
        clients.append(SimpleNamespace(wait_for_server=lambda **kw:True,send_goal_async=lambda goal,f=sent:f))
    node=SimpleNamespace(execute_client=clients[0],gripper_client=clients[1],passive_state=False,
        refresh_rigid_state=lambda t:[0.]*7,current_positions=[0.]*7,
        full_joints=JointState(name=[*NAMES,FINGER],position=[0.]*7+[TRANSIT_POSITION]),
        full_joint_state_monotonic_s=99.,current_state_monotonic_s=99.,
        get_clock=lambda:clock,execution_table_guard_failure=lambda:'',chain=SimpleNamespace(pose=lambda q:np.eye(4)))
    def spin(*a,**kw):
        elapsed[0]=next(moments)
        finger=min(plan['samples'],key=lambda r:abs(r['time_s']-elapsed[0]))['positions'][7]
        if elapsed[0]>=gate_time-.1:finger=-.1204
        node.full_joints.position=[0.]*7+[finger]
        node.full_joint_state_monotonic_s=node.current_state_monotonic_s=100.+elapsed[0]
    monkeypatch.setattr(executor.rclpy,'spin_once',spin)
    monkeypatch.setattr(executor.rclpy,'spin_until_future_complete',lambda *a,**kw:None)
    monkeypatch.setattr(executor,'verify_required_controllers',Mock())
    monkeypatch.setattr(executor,'event',Mock())
    with pytest.raises(RuntimeError,match='not stably closed before return'):
        executor.execute(node,plan,executor.build_goals(plan),False)
    assert elapsed[0]<times['return_closed']
    for handle in handles:handle.cancel_goal_async.assert_called_once()
