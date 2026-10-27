"""Gateway polling with fake futures and clocks: no ROS, CAN or real process."""
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from scripts.control_ownership import gateway


def polling_gate():
    gate=gateway.Gateway.__new__(gateway.Gateway)
    gate.controllers={};gate.controller_command_interfaces={};gate.controllers_at=0.
    gate.list_future=None;gate.list_sent_at=None;gate.list_next_query_at=0.
    gate.list_service_ready=False;gate.list_requests=gate.list_timeouts=gate.list_failures=0
    gate.list_last_error=gate.list_last_rtt_s=None
    gate.list_type=NS(Request=lambda:object())
    gate.list_client=Mock()
    gate.list_client.service_is_ready.return_value=True
    gate.list_client.call_async.side_effect=lambda _:Future()
    gate.log=Mock()
    return gate


def active_response():
    return NS(controller=[NS(name=name,state='active',claimed_interfaces=[joint+'/position' for joint in gateway.JOINTS[name]])
                          for name in gateway.CONTROLLERS[:3]])


def test_lost_query_is_retired_and_following_query_can_recover():
    gate=polling_gate()
    gate.poll_controllers(100.)
    old=gate.list_future
    gate.poll_controllers(100.75)
    assert old.cancelled() and gate.list_future is None
    gate.list_client.remove_pending_request.assert_called_once_with(old)
    assert not gate.controller_query_status(100.75)['fresh']
    gate.poll_controllers(100.99)
    assert gate.list_requests==1  # bounded backoff, not 50Hz retry spam
    gate.poll_controllers(101.)
    new=gate.list_future
    assert new is not old and gate.list_requests==2
    new.set_result(active_response())
    gate.poll_controllers(101.02)
    assert gate.controllers_at==101.  # request time, not delayed receipt time
    status=gate.controller_query_status(101.02)
    assert status['fresh'] and status['age_s']==pytest.approx(.02)
    assert status['last_error'] is None and status['timeouts']==1


def test_late_completed_reply_never_refreshes_cached_active_state():
    gate=polling_gate()
    gate.poll_controllers(100.)
    old=gate.list_future
    old.set_result(active_response())
    gate.poll_controllers(101.)
    assert gate.controllers_at==0. and gate.controllers=={}
    assert gate.list_timeouts==1 and gate.list_future is None
    gate.list_client.remove_pending_request.assert_called_once_with(old)


def test_persistent_loss_does_not_leave_one_future_stuck_for300seconds():
    gate=polling_gate()
    for cycle in range(301):
        gate.poll_controllers(100.+cycle)
        gate.poll_controllers(100.+cycle+.75)
    assert gate.list_requests==301 and gate.list_timeouts==301
    assert gate.list_future is None
    assert gate.list_client.remove_pending_request.call_count==301
    assert not gate.controller_query_status(401.)['fresh']


def test_server_absent_is_visible_and_never_creates_requests():
    gate=polling_gate();gate.list_client.service_is_ready.return_value=False
    gate.poll_controllers(100.)
    status=gate.controller_query_status(100.)
    assert not status['fresh'] and status['age_s'] is None
    assert status['last_error']=='controller-manager service unavailable'
    gate.list_client.call_async.assert_not_called()


def test_empty_or_exceptional_reply_does_not_refresh_state():
    for error in (None,RuntimeError('response failed')):
        gate=polling_gate();gate.poll_controllers(100.)
        if error is None:gate.list_future.set_result(None)
        else:gate.list_future.set_exception(error)
        gate.poll_controllers(100.01)
        assert gate.list_failures==1 and gate.controllers_at==0.
        assert gate.list_future is None and gate.list_last_error


def test_old_active_cache_remains_explicitly_stale_and_cannot_admit_motion(monkeypatch):
    gate=polling_gate()
    gate.controllers={name:'active' for name in gateway.CONTROLLERS[:3]}
    gate.controllers_at=100.
    monkeypatch.setattr(gateway.time,'monotonic',lambda:101.01)
    assert not gate.controller_query_status()['fresh']
    with pytest.raises(ValueError,match='unavailable/stale'):
        gate.controller_readiness()


def test_future_clock_cache_is_not_fresh():
    gate=polling_gate();gate.controllers_at=101.
    assert not gate.controller_query_status(100.)['fresh']


def test_rpc_send_exception_can_retry_without_claiming_freshness():
    gate=polling_gate()
    gate.list_client.call_async.side_effect=[RuntimeError('send failed'),Future()]
    gate.poll_controllers(100.)
    assert gate.list_future is None and gate.list_sent_at is None and gate.list_failures==1
    assert not gate.controller_query_status(100.)['fresh']
    gate.poll_controllers(100.25)
    assert gate.list_future is not None


def test_stale_cached_active_cannot_accept_a_goal_even_with_an_owned_lease(monkeypatch):
    import sys
    import threading
    import types
    fake=types.ModuleType('rclpy.action');fake.GoalResponse=NS(REJECT='reject')
    monkeypatch.setitem(sys.modules,'rclpy.action',fake)
    monkeypatch.setattr(gateway.time,'monotonic',lambda:102.)
    gate=polling_gate();gate.controllers_at=100.
    gate.controllers={name:'active' for name in gateway.CONTROLLERS}
    gate.mutex=threading.RLock();gate.closing=False;gate.pending=set()
    gate.authority=NS(snapshot=lambda now:dict(state='owned',owner='h2'))
    assert gate.accept(gateway.CONTROLLERS[0],object())=='reject'
