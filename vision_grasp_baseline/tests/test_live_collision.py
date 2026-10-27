from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from moveit_msgs.msg import RobotState

from vision_grasp_baseline.live_collision import LiveCollisionMonitor
from vision_grasp_baseline.continuous_plan import pickup_target_xy
from vision_grasp_baseline.continuous_execute import execute
from vision_grasp_baseline.waypoint_place_test import WaypointNode
from vision_grasp_baseline.grasp_test import TestNode as GraspTestNode


@pytest.fixture
def monitor(tmp_path):
    from vision_grasp_baseline.hover import NAMES
    from vision_grasp_baseline.grasp import FINGER
    model=tmp_path/'model.urdf'
    model.write_text('<robot>'+''.join(f'<joint name="{name}"><limit lower="-1" upper="1"/></joint>' for name in [*NAMES,FINGER])+'</robot>')
    node=SimpleNamespace(finger_contacts=Mock(),telemetry=Mock(),valid_client=Mock(),state=Mock(return_value=RobotState()),
                         execution_table_guard_failure=Mock(return_value=''))
    node.telemetry.snapshot.return_value={'positions_rad':[0.]*16}
    node.valid_client.call_async.side_effect=lambda request:Future()
    result=LiveCollisionMonitor(node,model)
    result.prepared=True # Runtime-only tests start after startup acceptance.
    return result


def test_actual_collision_and_hard_bounds_still_stop(monitor):
    monitor.check(0.)
    monitor.pending.set_result(SimpleNamespace(valid=False,contacts=[]))
    with pytest.raises(RuntimeError,match='collision/model bounds'):monitor.check(.02)
    monitor.node.telemetry.snapshot.return_value={'positions_rad':[1.1]*16}
    with pytest.raises(RuntimeError,match='hard position limits'):monitor.check(.04)


def test_missing_collision_response_fails_closed(monitor):
    monitor.check(0.)
    with pytest.raises(RuntimeError,match='timed out'):monitor.check(.21)


def test_unprepared_monitor_cannot_start_runtime_queries(monitor):
    monitor.prepared=False
    with pytest.raises(RuntimeError,match='must be prepared'):monitor.check(0.)
    monitor.node.valid_client.call_async.assert_not_called()


def test_hard_bounds_are_checked_even_while_response_is_pending(monitor):
    monitor.check(0.)
    monitor.node.telemetry.snapshot.return_value={'positions_rad':[1.1]*16}
    with pytest.raises(RuntimeError,match='hard position limits'):monitor.check(.01)


def test_late_completed_response_does_not_bypass_runtime_deadline(monitor):
    monitor.check(0.)
    monitor.pending.set_result(SimpleNamespace(valid=True))
    with pytest.raises(RuntimeError,match='timed out'):monitor.check(.21)
    assert monitor.runtime_timeouts==1


def test_startup_waits_for_discovery_and_two_fast_replies_after_cold_response(monitor):
    now=[0.];pending=[]
    monitor.node.valid_client.service_is_ready.side_effect=lambda:now[0]>=.4
    def send(request):
        future=Future();delay=.35 if not pending else .01
        pending.append((future,now[0]+delay));return future
    monitor.node.valid_client.call_async.side_effect=send
    def spin():
        now[0]+=.01
        for future,ready in pending:
            if not future.done() and now[0]>=ready:
                future.set_result(SimpleNamespace(valid=True))
    monitor.prepare(timeout=2.,clock=lambda:now[0],spin=spin)
    assert monitor.prepared and monitor.startup_probe_count==3
    assert monitor.startup_max_response_s>=.35
    assert monitor.report()['runtime_timeout_s']==.2
    assert monitor.pending is None


@pytest.mark.parametrize('discoverable',[False,True])
def test_startup_unavailable_or_stalled_service_fails_without_motion(monitor,discoverable):
    now=[0.]
    monitor.node.valid_client.service_is_ready.return_value=discoverable
    def spin():now[0]+=.01
    with pytest.raises(RuntimeError,match='no trajectory dispatched'):
        monitor.prepare(timeout=.5,clock=lambda:now[0],spin=spin)
    assert not monitor.prepared
    if discoverable:
        assert monitor.node.valid_client.call_async.call_count==1
    else:monitor.node.valid_client.call_async.assert_not_called()


def test_startup_collision_is_not_treated_as_a_cold_network_retry(monitor):
    monitor.node.valid_client.service_is_ready.return_value=True
    def send(request):
        result=Future();result.set_result(SimpleNamespace(valid=False,contacts=[]));return result
    monitor.node.valid_client.call_async.side_effect=send
    with pytest.raises(RuntimeError,match='collision/model bounds'):
        monitor.prepare(clock=lambda:0.,spin=lambda:None)
    assert not monitor.prepared


def test_contact_update_time_is_not_mistaken_for_rpc_delay(monitor,monkeypatch):
    now=[0.]
    monkeypatch.setattr('vision_grasp_baseline.live_collision.time.monotonic',lambda:now[0])
    monitor.phase='carry';monitor.support_contact=True;monitor.pickup=np.eye(4)
    tcp=np.eye(4);tcp[2,3]=.1
    monitor.node.chain=SimpleNamespace(pose=lambda q:tcp)
    monitor.node.finger_contacts.side_effect=lambda *a,**kw:now.__setitem__(0,.3)
    monitor.node.telemetry.snapshot.side_effect=[{'positions_rad':[0.]*16},
                                                {'positions_rad':[.05]+[0.]*15},
                                                {'positions_rad':[.05]+[0.]*15}]
    monitor.check()
    assert monitor.sent==.3
    assert monitor.node.state.call_args.args[0][0]==.05
    now[0]=.31;monitor.check() # Still only 10ms since actual request dispatch.


def test_failed_monitor_startup_cannot_begin_recording_or_dispatch_goals(monkeypatch):
    from vision_grasp_baseline.continuous_timing import assemble
    from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY
    levels=[0.,.2,.3,.5,.4,0.]
    phases=[dict(times=[0.,2.],positions=[[a]*7,[b]*7],velocities=[[0.]*7]*2,
                 accelerations=[[0.]*7]*2) for a,b in zip(levels,levels[1:])]
    plan=assemble(phases,dict(velocity=10.,acceleration=.5,jerk=10.),COLLECTION_POLICY)
    node=SimpleNamespace(execute_client=Mock(),gripper_client=Mock(),refresh_rigid_state=Mock())
    monitor=Mock();monitor.prepare.side_effect=RuntimeError('startup unavailable; no trajectory dispatched')
    begin=Mock();mark_dispatch=Mock()
    monkeypatch.setattr('vision_grasp_baseline.continuous_execute.verify_required_controllers',Mock())
    with pytest.raises(RuntimeError,match='startup unavailable'):
        execute(node,plan,[],True,before_dispatch=begin,collision_monitor=monitor,on_dispatch=mark_dispatch)
    monitor.prepare.assert_called_once()
    begin.assert_not_called();mark_dispatch.assert_not_called()
    node.execute_client.send_goal_async.assert_not_called()
    node.gripper_client.send_goal_async.assert_not_called()


def test_attachment_transition_discards_stale_phase_query(monitor):
    monitor.check(0.);pending=monitor.pending
    monitor.transition('return')
    assert pending.cancelled() and monitor.pending is None
    monitor.node.valid_client.remove_pending_request.assert_called_once_with(pending)
    monitor.node.finger_contacts.assert_called_with(False,support_contact=False)


def test_phase_transition_cannot_hide_already_received_collision(monitor):
    monitor.check(0.)
    monitor.pending.set_result(SimpleNamespace(valid=False,contacts=[
        SimpleNamespace(contact_body_1='finger',contact_body_2='cube')]))
    with pytest.raises(RuntimeError,match='finger.*cube'):monitor.transition('grasp')
    assert monitor.phase=='reach'
    assert monitor.report()['failure']['query']['phase']=='reach'
    monitor.node.finger_contacts.assert_not_called()


def test_timeout_report_distinguishes_execution_loop_stall(monitor):
    monitor.check(1.)
    with pytest.raises(RuntimeError,match='check_gap_s=0.318'):monitor.check(1.318)
    report=monitor.report()
    assert report['max_check_gap_s']==pytest.approx(.318)
    assert report['failure']['response_done'] is False
    assert report['failure']['query']['positions_rad']==[0.]*8
    pending=monitor.pending;monitor.close()
    assert pending.cancelled() and monitor.pending is None


def test_delayed_recorder_does_not_starve_collision_polling(monitor):
    import threading
    import time
    from vision_grasp_baseline.runtime_recording import RuntimeRecording
    entered=threading.Event();release=threading.Event()
    def http(operation,**payload):
        if operation=='event':entered.set();assert release.wait(1.)
    io=RuntimeRecording(call=http)
    try:
        io.event('continuous_dispatch');assert entered.wait(1.)
        started=time.monotonic()
        while time.monotonic()-started<.32:
            # Simulates servicing the ROS future on each execution iteration.
            if monitor.pending is not None and not monitor.pending.done():
                monitor.pending.set_result(SimpleNamespace(valid=True))
            monitor.check();io.check();time.sleep(.005)
        assert monitor.checked>=10 and monitor.max_check_gap_s<.2
        assert monitor.runtime_timeouts==0
        release.set();io.close(drain=True)
    finally:release.set();io.close();monitor.close()


def test_cube_contact_is_only_allowed_after_closure_begins(monitor):
    monitor.transition('reach')
    monitor.node.finger_contacts.assert_called_with(False,support_contact=False)
    monitor.transition('grasp')
    monitor.node.finger_contacts.assert_called_with(True,support_contact=False)
    monitor.transition('carry')
    monitor.node.finger_contacts.assert_called_with(True,support_contact=True)
    monitor.transition('return')
    monitor.node.finger_contacts.assert_called_with(False,support_contact=False)
    monitor.check(.01);monitor.pending.set_result(SimpleNamespace(valid=True))
    monitor.check(.02)
    assert monitor.checked==1


def test_tracking_distance_is_diagnostic_in_continuous_mode(monkeypatch):
    monkeypatch.setattr(GraspTestNode,'execution_table_guard_failure',lambda *args:'')
    poses=[np.eye(4),np.eye(4)];poses[1][0,3]=.029
    node=SimpleNamespace(telemetry=Mock(),chain=Mock(),max_tracking_error_m=0.,tracking_abort=False,tracking_error_since=None)
    node.telemetry.snapshot.return_value=dict(positions_rad=[0.]*16,command_positions_rad=[0.]*16)
    node.chain.pose.side_effect=poses
    # Use an actual instance without starting a ROS node so super() resolves.
    instance=WaypointNode.__new__(WaypointNode)
    for key,value in vars(node).items():setattr(instance,key,value)
    assert instance.execution_table_guard_failure()==''
    assert instance.max_tracking_error_m==pytest.approx(.029)


def test_forward_correction_moves_pickup_only():
    center=np.array([.34,-.18,.28]);before=center.copy()
    np.testing.assert_allclose(pickup_target_xy(center,[.002,0.,0.]),[.342,-.18])
    np.testing.assert_array_equal(center,before)
    with pytest.raises(ValueError):pickup_target_xy(center,[0.,0.,.002])


def test_unknown_rigid_mode_cannot_send_motion_goals(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.continuous_execute.verify_required_controllers',Mock())
    from vision_grasp_baseline.continuous_timing import assemble
    from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY
    points=[0.,.2,.3,.5,.4,0.]
    phases=[dict(times=[0.,2.],positions=[[a]*7,[b]*7],velocities=[[0.]*7]*2,
                 accelerations=[[0.]*7]*2) for a,b in zip(points,points[1:])]
    plan=assemble(phases,dict(velocity=10.,acceleration=.5,jerk=10.),COLLECTION_POLICY)
    node=SimpleNamespace(execute_client=Mock(),gripper_client=Mock(),verify_controller_active=Mock(),
                         refresh_rigid_state=Mock(return_value=[0.]*7),passive_state=None)
    with pytest.raises(RuntimeError,match='confirmation required'):execute(node,plan,[],False)
    node.execute_client.send_goal_async.assert_not_called()
    node.gripper_client.send_goal_async.assert_not_called()
