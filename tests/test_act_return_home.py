"""Offline recovery tests; synthetic ROS messages and collision geometry only."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scripts.act_trial import return_home as rh
from scripts.act_trial import manual_experiment as manual


def fixture(monkeypatch):
    q=np.zeros(16);q[7]=-.35
    b={'home_config':{'home_gripper':{'open_position_rad':-.35}}}
    monkeypatch.setattr(rh,'stationary_recovery_contract',lambda *a:{'limits_rad':[.03]*5+[.04,.03,.03]})
    state={'positions_rad':q.tolist(),'command_positions_rad':q.tolist(),
           'velocity_rad_s':[0.]*16,'motor_status':[1]*16,'max_age_s':.01,'command_max_age_s':.01}
    return q,b,state


def test_stationary_recovery_preserves_strict_start_and_gripper(monkeypatch):
    q,b,s=fixture(monkeypatch)
    s['command_positions_rad'][5]=.031
    assert np.array_equal(rh.check_snapshot(s,b,q),q)
    for key,value in [('positions_rad',[float('nan')]*16),('velocity_rad_s',[.051]*16),
                      ('motor_status',[0]*16),('max_age_s',.11),('command_max_age_s',-.01)]:
        bad=deepcopy(s);bad[key]=value
        with pytest.raises(ValueError):rh.check_snapshot(bad,b)
    for axis,value in [(5,.041),(7,-.30)]:
        bad=deepcopy(s);bad['command_positions_rad'][axis]=value
        with pytest.raises(ValueError):rh.check_snapshot(bad,b)
    changed=q.copy();changed[12]=.011
    with pytest.raises(ValueError,match='start changed'):rh.check_snapshot(s,b,changed)


class Scene:
    def __init__(self,collision=False):self.samples=[];self.collision=collision
    def put(self,q):
        self.samples.append(q.copy())
        return [{'collision':True}] if self.collision else []


def path(q,delta=.02,duration=2):
    p=q[:7].copy();end=p.copy();end[0]+=delta
    def point(pos,t):return NS(positions=pos.tolist(),velocities=[0.]*7,accelerations=[0.]*7,
                               effort=[],time_from_start=NS(sec=t,nanosec=0))
    return NS(joint_trajectory=NS(joint_names=rh.NAMES[:7],points=[point(p,0),point(end,duration)]),
              multi_dof_joint_trajectory=NS(points=[])),end


def test_polynomial_checked_between_knots_and_preserves_context():
    q=np.zeros(16);q[7]=-.35;q[8:]=np.arange(8)*.01
    trajectory,goal=path(q)
    scene=Scene();limits=np.tile([-2.,2.],(16,1))
    report=rh.validate_path(trajectory,q,q,goal,limits,np.ones(16)*5,scene)
    assert report['collision_samples']>400
    for s in scene.samples:np.testing.assert_array_equal(s[7:],q[7:])
    # Collisions, acceleration excess and changed joint order cannot be bypassed.
    with pytest.raises(ValueError,match='collision'):
        rh.validate_path(trajectory,q,q,goal,limits,np.ones(16)*5,Scene(True))
    fast,goal=path(q,delta=.2,duration=1)
    with pytest.raises(ValueError,match='acceleration|speed'):
        rh.validate_path(fast,q,q,goal,limits,np.ones(16)*5,Scene())
    trajectory.joint_trajectory.joint_names.reverse()
    with pytest.raises(ValueError,match='joint order'):
        rh.validate_path(trajectory,q,q,goal,limits,np.ones(16)*5,Scene())


def test_recovery_does_not_run_while_act_processes_exist(monkeypatch,tmp_path):
    monkeypatch.setattr(manual,'observation_processes',lambda _: [42])
    monkeypatch.setattr(manual.subprocess,'run',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='Stop old Dora'):
        manual.recover_home(tmp_path,tmp_path/'act',execute=True)


def test_manual_plan_does_not_acquire_motion_lease(monkeypatch,tmp_path):
    calls=[]
    class Client:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(manual,'Client',Client)
    monkeypatch.setattr(manual,'wait_idle',lambda *a,**k: {})
    monkeypatch.setattr(manual,'observation_processes',lambda _: [])
    monkeypatch.setattr(manual.subprocess,'run',lambda c,**k:calls.append(c))
    manual.recover_home(tmp_path,tmp_path/'act')
    assert len(calls)==1
    assert 'scripts.act_trial.return_home' in calls[0]
    assert '--execute' not in calls[0] and '--owner' not in calls[0]
    assert (tmp_path/'return_home_current.path').exists()


def test_saved_execution_requires_new_attempt_and_exclusive_h2_lease(monkeypatch,tmp_path):
    import json
    calls=[]
    class Client:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(manual,'Client',Client)
    monkeypatch.setattr(manual,'wait_idle',lambda *a,**k: {})
    monkeypatch.setattr(manual,'observation_processes',lambda _: [])
    directory=tmp_path/'return_home_reviewed';directory.mkdir()
    (tmp_path/'return_home_current.path').write_text(str(directory))
    (directory/'plan.json').write_text(json.dumps({'success':True,'physical_motion_requested':False}))
    monkeypatch.setattr(manual,'verify_pose',lambda *a:None)
    def run(c,**kwargs):
        calls.append(c)
        if 'hold-test' in c:
            (directory/'home_hold_test.json').write_text('{"passed":true}')
    monkeypatch.setattr(manual.subprocess,'run',run)
    manual.recover_home(tmp_path,tmp_path/'act',execute=True)
    assert calls[0][:7]==['/usr/bin/python3','-m','scripts.control_ownership.client','run','--owner','h2','--']
    assert '--execute' in calls[0]
    (directory/'execution.json').write_text('{}')
    with pytest.raises(ValueError,match='already attempted'):
        manual.recover_home(tmp_path,tmp_path/'act',execute=True)
    assert len(calls)==2


def test_plan_and_execute_saved_ros_path_with_mocked_hardware(monkeypatch,tmp_path):
    import json
    from pathlib import Path
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from scripts.act_trial import telemetry,ros_bridge
    from scripts.home_replay import simulate
    from openarm_kinesthetic_recorder import home_guarded_executor as ros
    from scripts.act_trial.preflight import DEFAULT_BUNDLE
    b=json.loads(DEFAULT_BUNDLE.read_text());cfg=b['home_config']
    q=np.array(cfg['right_home_joint_positions_rad']+[cfg['home_gripper']['open_position_rad']]+[0.]*8)
    q[0]+=.04
    state={'positions_rad':q.tolist(),'command_positions_rad':q.tolist(),
           'velocity_rad_s':[0.]*16,'motor_status':[1]*16,'max_age_s':.01,'command_max_age_s':.01,
           'command_kp':[0.]*16,'command_kd':[0.]*16}
    plans=[];executions=[]
    class Inputs:
        def __init__(self,**kw):assert kw=={'start_parameter_services':False}
        def snapshot(self,_):return {'actual':np.array(state['positions_rad'])}
        def get(self,_):return None
        def audit(self,_):return {'expected_right_gains':{'kp':[0.]*8,'kd':[0.]*8},
                                  'live_controller_parameters':{'right_joint_trajectory_controller':
                                   {'interpolation_method':'splines','interpolate_from_desired_state':False}}}
        def close(self):pass
    class Telemetry:
        def __init__(self,*a):pass
        def snapshot(self):return deepcopy(state)
        def close(self):pass
    class Node:
        def __init__(self):pass
        def guard_snapshot(self,object_id,*a):assert object_id==cfg['static_collision']['object_id'];return {}
        def refresh_rigid_state(self,*a):return state['positions_rad'][:7]
        def plan(self,payload,goal,*a):
            plans.append(payload);traj=RobotTrajectory();traj.joint_trajectory.joint_names=rh.NAMES[:7]
            for pos,t in ((payload['start_positions_rad'],0),(goal,4)):
                point=JointTrajectoryPoint();point.positions=pos;point.velocities=[0.]*7;point.accelerations=[0.]*7
                point.time_from_start.sec=t;traj.joint_trajectory.points.append(point)
            return NS(error_code=NS(val=1),trajectory=traj)
        def execute_same_trajectory(self,traj,*a):
            executions.append(traj)
            state['positions_rad'][:7]=traj.joint_trajectory.points[-1].positions
            state['command_positions_rad'][:7]=traj.joint_trajectory.points[-1].positions
            return {'action_status':4,'moveit_error_code':1}
        def destroy_node(self):pass
    from contextlib import nullcontext
    monkeypatch.setattr(rh,'validated_token',lambda **kw:'a'*64)
    monkeypatch.setattr(rh,'inject_action_capabilities',lambda _:nullcontext())
    monkeypatch.setattr(ros_bridge,'Inputs',Inputs)
    monkeypatch.setattr(telemetry,'Telemetry',Telemetry)
    monkeypatch.setattr(ros,'HomeInitializationNode',Node)
    monkeypatch.setattr(ros,'verify_static_collision_snapshot',lambda *a:None)
    monkeypatch.setattr(ros,'wait_for_home_settle',lambda *a:{'accepted':True})
    monkeypatch.setattr(simulate,'Scene',lambda _:Scene())
    rh.run(NS(directory=tmp_path,execute=False))
    assert len(plans)==1 and not executions
    saved=json.loads((tmp_path/'plan.json').read_text())
    assert saved['success'] and not saved['physical_motion_requested']
    rh.run(NS(directory=tmp_path,execute=True))
    assert len(plans)==1 and len(executions)==1  # no replan or alternate path
    assert json.loads((tmp_path/'execution.json').read_text())['success']


def test_initialization_failure_is_recorded_without_opening_can(monkeypatch,tmp_path):
    import json
    from scripts.act_trial import ros_bridge,telemetry
    monkeypatch.setattr(rh,'validated_token',lambda **kw:None)
    def fail(**kwargs):raise RuntimeError('synthetic observer initialization failure')
    monkeypatch.setattr(ros_bridge,'Inputs',fail)
    monkeypatch.setattr(telemetry,'Telemetry',lambda *a:pytest.fail('CAN observer must not open'))
    with pytest.raises(RuntimeError,match='observer initialization'):
        rh.run(NS(directory=tmp_path,execute=False))
    report=json.loads((tmp_path/'plan.json').read_text())
    assert report['success'] is False
    assert report['physical_motion_requested'] is False
    assert report['error']=='synthetic observer initialization failure'
