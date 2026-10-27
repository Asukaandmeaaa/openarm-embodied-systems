from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vision_grasp_baseline.controller_readiness import verify_required_controllers


def response(state='active'):
    future=Future();future.set_result(NS(controller=[NS(name='arm',state=state),NS(name='gripper',state='active')]))
    return future


def node_for(client):
    return NS(controller_client=client,destroy_client=Mock(return_value=True),
              create_client=Mock(return_value=client))


def test_lost_response_is_discarded_and_both_controllers_checked_together(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.rclpy.spin_until_future_complete',Mock())
    missing=Future();client=Mock();client.call_async.side_effect=[missing,response()]
    result=verify_required_controllers(node_for(client),('arm','gripper'))
    assert result['attempts']==2 and result['controllers']==dict(arm='active',gripper='active')
    assert missing.cancelled()
    client.remove_pending_request.assert_called_once_with(missing)


def test_real_inactive_controller_fails_without_retry(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.rclpy.spin_until_future_complete',Mock())
    client=Mock();client.call_async.return_value=response('inactive')
    with pytest.raises(RuntimeError,match='not active'):verify_required_controllers(node_for(client),('arm','gripper'))
    assert client.call_async.call_count==1


def test_three_timeouts_cannot_be_accepted(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.rclpy.spin_until_future_complete',Mock())
    client=Mock();client.call_async.side_effect=[Future(),Future(),Future()]
    with pytest.raises(RuntimeError,match='after 3 queries'):verify_required_controllers(node_for(client),('arm','gripper'))
    assert client.call_async.call_count==3 and client.remove_pending_request.call_count==3


def test_retry_recreates_only_readonly_client_and_uses_new_response(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.rclpy.spin_until_future_complete',Mock())
    old=Mock();old.call_async.return_value=Future()
    new=Mock();new.call_async.return_value=response()
    node=node_for(old);node.create_client.return_value=new
    result=verify_required_controllers(node,('arm','gripper'))
    assert result['diagnostics']['client_recreations']==1
    node.destroy_client.assert_called_once_with(old)
    assert node.create_client.call_args.args[1]=='/controller_manager/list_controllers'
    assert node.controller_client is new


def test_overall_deadline_prevents_endless_service_wait(monkeypatch):
    now=[100.]
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.time.monotonic',lambda:now[0])
    client=Mock()
    def wait(*,timeout_sec):now[0]+=timeout_sec;return False
    client.wait_for_service.side_effect=wait
    node=node_for(client)
    with pytest.raises(RuntimeError,match='within 4.0s'):
        verify_required_controllers(node,('arm','gripper'),total_timeout_s=4.)
    assert now[0]==104. and node.controller_readiness_diagnostics['queries']==0
    client.call_async.assert_not_called()


def test_late_completed_response_cannot_pass_preflight(monkeypatch):
    now=[100.]
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.time.monotonic',lambda:now[0])
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.rclpy.spin_until_future_complete',
        lambda *a,**k:now.__setitem__(0,now[0]+3.1))
    client=Mock();client.call_async.side_effect=lambda request:response()
    with pytest.raises(RuntimeError,match='no trajectory dispatched'):
        verify_required_controllers(node_for(client),('arm','gripper'))
    assert client.remove_pending_request.call_count==3


def test_stop_or_motor_feedback_failure_prevents_query(monkeypatch):
    client=Mock();node=node_for(client)
    node.execution_table_guard_failure=lambda:'operator requested stop'
    with pytest.raises(RuntimeError,match='operator requested stop'):
        verify_required_controllers(node,('arm','gripper'))
    client.call_async.assert_not_called()


def test_client_cleanup_failure_stops_instead_of_leaking_pending_queries(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.controller_readiness.rclpy.spin_until_future_complete',Mock())
    client=Mock();client.call_async.return_value=Future()
    node=node_for(client);node.destroy_client.return_value=False
    with pytest.raises(RuntimeError,match='Could not retire'):
        verify_required_controllers(node,('arm','gripper'))
    node.create_client.assert_not_called()
