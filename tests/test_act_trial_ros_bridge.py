"""ROS message boundary tests; no ROS initialization, sockets, or hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.act_trial import ros_bridge as bridge


def test_position_point_preserves_guard_target_without_extra_interpolation():
    pytest.importorskip('trajectory_msgs.msg')
    q = np.array([-.13, -.01, 0., 1.37, .01, -.19, -.12])
    msg = bridge.trajectory_for(q)
    assert msg.joint_names == bridge.NAMES[:7]
    assert msg.header.stamp.sec == msg.header.stamp.nanosec == 0
    assert len(msg.points) == 1
    point = msg.points[0]
    assert list(point.positions) == q.tolist()
    assert point.time_from_start.sec == point.time_from_start.nanosec == 0
    assert len(point.velocities) == len(point.accelerations) == len(point.effort) == 0


@pytest.mark.parametrize('q', [np.zeros(8), np.zeros(6), [float('nan')]*7, [float('inf')]*7])
def test_bad_dispatch_rejected(q):
    pytest.importorskip('trajectory_msgs.msg')
    with pytest.raises(ValueError, match='seven finite'):
        bridge.trajectory_for(q)


def test_feedback_reorders_and_rejects_missing_duplicates_or_nonfinite():
    np.testing.assert_equal(bridge.ordered(['b', 'a'], [2, 1], ['a', 'b']), [1, 2])
    for names, values in [(['a', 'a'], [1, 2]), (['a'], [float('nan')]), (['a'], [])]:
        with pytest.raises(ValueError): bridge.ordered(names, values, ['a'])
    with pytest.raises(KeyError): bridge.ordered(['a'], [1], ['b'])


def test_age_accounts_for_source_and_receiver_delay():
    stamp = SimpleNamespace(sec=100, nanosec=0)
    assert bridge.source_age(stamp, 100_020_000_000, 5., 5.03) == pytest.approx(.05)
    with pytest.raises(ValueError): bridge.source_age(stamp, 99_000_000_000, 5., 5.)
    with pytest.raises(ValueError): bridge.source_age(SimpleNamespace(sec=0, nanosec=0), 0, 5., 5.)


def test_controller_and_motor_dispatch_must_belong_to_recent_reference_history():
    q = np.array([-.13, -.01, 0., 1.37, .01, -.19, -.12, -.35])
    quantized = np.floor((q+12.5)/25*65535)*25/65535-12.5
    bridge.check_reference_history(q, quantized, [q], q[7])
    changed = q.copy(); changed[0] += .001
    with pytest.raises(ValueError, match='ROS output'):
        bridge.check_reference_history(changed, quantized, [q], q[7])
    changed = q.copy(); changed[7] += .001
    with pytest.raises(ValueError, match='gripper'):
        bridge.check_reference_history(changed, quantized, [q], q[7])
    changed = quantized.copy(); changed[1] += 25/65535
    with pytest.raises(ValueError, match='CAN command'):
        bridge.check_reference_history(q, changed, [q], q[7])


def test_bridge_has_no_raw_ros_publisher_and_only_requests_right_scoped_authority():
    tree = ast.parse(Path(bridge.__file__).read_text())
    publishers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'create_publisher']
    assert not publishers
    commands = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == 'authority.command']
    assert len(commands) == 1
    assert ast.unparse(commands[0].args[0]) == 'command[:count].tolist()'
    assert 'count = 8 if guard.command_gripper else 7' in Path(bridge.__file__).read_text()
    assert bridge.ARM_TOPIC == '/right_joint_trajectory_controller/joint_trajectory'
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert 'authority.acquire' in calls and 'authority.command' in calls
    assert not any(name.endswith(('.enable', '.disable', '.send_goal_async', '.cancel_goal_async'))
                   for name in calls)


def test_task_gripper_history_allows_separate_ros_updates_but_rejects_competing_target():
    q = np.zeros(8); q[7] = -.35
    following = q.copy(); following[0] = .001; following[7] = -.349
    output = following.copy(); output[7] = q[7]
    encoded = np.floor((output+12.5)/25*65535)*25/65535-12.5
    bridge.check_reference_history(output, encoded, [q,following], None)
    output[7] = -.25
    with pytest.raises(ValueError, match='gripper output'):
        bridge.check_reference_history(output, encoded, [q,following], None)


def controller_inputs(monkeypatch):
    import threading
    from concurrent.futures import Future
    now=[10.]
    monkeypatch.setattr(bridge.time, 'monotonic', lambda: now[0])
    obj=bridge.Inputs.__new__(bridge.Inputs)
    obj.lock=threading.Lock()
    obj.controller_future=None; obj.controllers_at=None
    obj.controllers=None; obj.controller_error=None
    obj.list_type=SimpleNamespace(Request=lambda: object())
    calls=[]
    def request(_):
        future=Future(); calls.append(future); return future
    obj.list_client=SimpleNamespace(service_is_ready=lambda: True, call_async=request)
    def complete(future, inactive=None):
        names=[bridge.ARM_CONTROLLER,bridge.GRIP_CONTROLLER,'left_joint_trajectory_controller']
        future.set_result(SimpleNamespace(controller=[SimpleNamespace(name=n,state='inactive' if n==inactive else 'active') for n in names]))
    return obj,now,calls,complete


def test_controller_poll_continues_while_main_loop_is_initializing(monkeypatch):
    obj,now,calls,complete=controller_inputs(monkeypatch)
    # No query_controllers/main-loop call during a1.5second simulated audit.
    for stamp in (10.,10.3,10.6,10.9,11.2,11.5):
        now[0]=stamp; obj.poll_controllers(); complete(calls[-1])
    assert len(calls)==6
    obj.query_controllers(11.5)
    assert obj.controllers_at == 11.5


def test_pending_service_cannot_be_relabelled_as_fresh(monkeypatch):
    obj,now,calls,complete=controller_inputs(monkeypatch)
    obj.poll_controllers(); complete(calls[-1])
    now[0]=10.3; obj.poll_controllers()
    now[0]=11.1; obj.poll_controllers()
    assert len(calls)==2
    with pytest.raises(ValueError,match='unavailable or stale'): obj.query_controllers(now[0])
    complete(calls[-1])
    obj.query_controllers(now[0])


def test_inactive_and_failed_controller_responses_still_block(monkeypatch):
    obj,now,calls,complete=controller_inputs(monkeypatch)
    obj.poll_controllers(); complete(calls[-1],bridge.ARM_CONTROLLER)
    with pytest.raises(ValueError,match='is not active'): obj.query_controllers(now[0])
    now[0]=10.3; obj.poll_controllers(); calls[-1].set_exception(RuntimeError('service lost'))
    assert obj.controller_error=='service lost'
    with pytest.raises(ValueError,match='unavailable or stale'): obj.query_controllers(now[0])


def test_completed_future_callback_does_not_deadlock_poll(monkeypatch):
    from concurrent.futures import Future
    obj,now,calls,complete=controller_inputs(monkeypatch)
    future=Future(); complete(future)
    obj.list_client.call_async=lambda _: future
    obj.poll_controllers()
    assert obj.controller_future is None
    obj.query_controllers(now[0])


def test_h2_observer_disables_jazzy_type_service_without_changing_normal_observer():
    from rclpy.parameter import Parameter
    options=bridge.input_node_options(False)
    assert options['start_parameter_services'] is False
    assert len(options['parameter_overrides'])==1
    override=options['parameter_overrides'][0]
    assert override.name=='start_type_description_service'
    assert override.type_==Parameter.Type.BOOL and override.value is False
    assert bridge.input_node_options(True)=={'start_parameter_services':True}
