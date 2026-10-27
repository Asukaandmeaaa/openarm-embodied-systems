"""Adversarial command ownership tests; no ROS initialization or CAN access.

These exercise authorization before dispatch. A downstream command-history
monitor alone cannot satisfy these properties because it detects after a write.
"""
import math
import asyncio
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from scripts.control_ownership.authority import LeaseAuthority
from scripts.control_ownership.gateway import finite_positions
from scripts.control_ownership import gateway


@pytest.fixture
def authority():
    return LeaseAuthority(ttl=0.5)


def test_only_one_owner_can_acquire(authority):
    token = authority.acquire("h2", "connection-a", 10.0)
    assert isinstance(token, str) and len(token) >= 32
    with pytest.raises(ValueError):
        authority.acquire("dora", "connection-b", 10.1)
    with pytest.raises(ValueError):
        authority.acquire("h2", "connection-c", 10.1)
    authority.authorize(token, "connection-a", 10.1, "action")


@pytest.mark.parametrize("positions", [
    None, "0,0,0,0,0,0,0", [], [0.0] * 6, [0.0] * 8,
    [math.nan] * 7, [math.inf] * 7, [-math.inf] * 7, [True] * 7, ["0"] * 7,
])
def test_gateway_rejects_malformed_positions_before_dispatch(positions):
    with pytest.raises(ValueError):
        finite_positions(positions, 7)


def test_gateway_validates_without_changing_numeric_policy_values():
    source = [0, -0.12345678912345678, 0.2, 1.375, 0.01, -0.19, -0.12]
    result = finite_positions(source, 7)
    assert result == source
    assert result is not source


def test_stationary_check_uses_joint_state_velocities_without_changing_jtc_interfaces(monkeypatch):
    monkeypatch.setattr(gateway.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(gateway.time, "time", lambda: 100.0)
    gate = gateway.Gateway.__new__(gateway.Gateway)
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0)),
        name=["b", "a"], velocity=[0.02, -0.01])
    gate.joint_state = (message, 10.0)
    assert gate.measured_velocities(["a", "b"]) == [-0.01, 0.02]


@pytest.mark.parametrize("fault", ["missing", "stale_receipt", "stale_source", "duplicate", "missing_joint", "nonfinite"])
def test_stationary_check_rejects_unreliable_velocity_evidence(monkeypatch, fault):
    monkeypatch.setattr(gateway.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(gateway.time, "time", lambda: 100.0)
    gate = gateway.Gateway.__new__(gateway.Gateway)
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0)),
        name=["a", "b"], velocity=[0.0, 0.0])
    received = 10.0
    if fault == "stale_receipt": received = 9.7
    if fault == "stale_source": message.header.stamp.sec = 99
    if fault == "duplicate": message.name = ["a", "a"]
    if fault == "missing_joint": message.name = ["a", "c"]
    if fault == "nonfinite": message.velocity[0] = math.nan
    gate.joint_state = None if fault == "missing" else (message, received)
    with pytest.raises(ValueError):
        gate.measured_velocities(["a", "b"])


@pytest.mark.parametrize("owner", ["", "moveit", "teleop", "ros", "Dora", None])
def test_only_explicit_owner_modes_exist(authority, owner):
    with pytest.raises(ValueError):
        authority.acquire(owner, "connection-a", 10.0)
    # A rejected request must not consume the vacant lease.
    authority.acquire("dora", "connection-a", 10.0)


@pytest.mark.parametrize("operation", ["command", "renew", "release"])
def test_token_without_same_session_cannot_take_over(authority, operation):
    token = authority.acquire("dora", "connection-a", 10.0)
    with pytest.raises(ValueError):
        if operation == "command":
            authority.authorize(token, "connection-b", 10.1, operation)
        else:
            getattr(authority, operation)(token, "connection-b", 10.1)
    authority.authorize(token, "connection-a", 10.1, "command")


@pytest.mark.parametrize("bad_token", ["", "not-the-issued-token", None])
def test_bad_token_never_authorizes_or_renews(authority, bad_token):
    good = authority.acquire("dora", "connection-a", 10.0)
    for operation in ("command", "renew", "release"):
        with pytest.raises(ValueError):
            if operation == "command":
                authority.authorize(bad_token, "connection-a", 10.1, operation)
            else:
                getattr(authority, operation)(bad_token, "connection-a", 10.1)
    authority.authorize(good, "connection-a", 10.1, "command")


@pytest.mark.parametrize("owner,forbidden", [("dora", "action"), ("h2", "command")])
def test_owner_mode_restricts_command_route(authority, owner, forbidden):
    token = authority.acquire(owner, "connection-a", 10.0)
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", 10.1, forbidden)


@pytest.mark.parametrize("operation", ["", "passive", "enable", "disable", "switch", "gripper", "left", "any"])
def test_unknown_or_out_of_scope_operations_rejected(authority, operation):
    token = authority.acquire("dora", "connection-a", 10.0)
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", 10.1, operation)


def test_renewal_preserves_token_and_extends_only_live_lease(authority):
    token = authority.acquire("dora", "connection-a", 10.0)
    authority.renew(token, "connection-a", 10.4)
    authority.authorize(token, "connection-a", 10.8, "command")
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", 10.901, "command")
    with pytest.raises(ValueError):
        authority.renew(token, "connection-a", 10.902)


@pytest.mark.parametrize("operation", ["command", "renew", "release"])
def test_expired_capability_cannot_be_used(authority, operation):
    token = authority.acquire("dora", "connection-a", 10.0)
    with pytest.raises(ValueError):
        if operation == "command":
            authority.authorize(token, "connection-a", 10.5, operation)
        else:
            getattr(authority, operation)(token, "connection-a", 10.5)


def test_expiry_does_not_automatically_grant_another_source(authority):
    authority.acquire("h2", "connection-a", 10.0)
    with pytest.raises(ValueError):
        authority.acquire("dora", "connection-b", 11.0)


def test_release_revokes_old_commands_and_prevents_undrained_handoff(authority):
    token = authority.acquire("h2", "connection-a", 10.0)
    authority.release(token, "connection-a", 10.1)
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", 10.2, "action")
    with pytest.raises(ValueError):
        authority.acquire("dora", "connection-b", 10.2)


def test_certified_drain_handoff_issues_a_new_capability(authority):
    old_token = authority.acquire("h2", "connection-a", 10.0)
    authority.release(old_token, "connection-a", 10.1)
    authority.drain_completed(10.2)
    new_token = authority.acquire("dora", "connection-b", 10.3)
    assert new_token != old_token
    with pytest.raises(ValueError):
        authority.authorize(old_token, "connection-a", 10.4, "action")
    with pytest.raises(ValueError):
        authority.authorize(old_token, "connection-b", 10.4, "command")
    authority.authorize(new_token, "connection-b", 10.4, "command")


def test_drain_acknowledgement_cannot_revoke_an_active_owner(authority):
    token = authority.acquire("dora", "connection-a", 10.0)
    with pytest.raises(ValueError):
        authority.drain_completed(10.1)
    authority.authorize(token, "connection-a", 10.2, "command")


def test_status_does_not_disclose_the_authorization_token(authority):
    token = authority.acquire("dora", "connection-a", 10.0)
    snapshot = authority.snapshot(10.1)
    assert snapshot["owner"] == "dora"
    assert token not in repr(snapshot)


def test_unrelated_connection_disconnect_cannot_revoke_owner(authority):
    token = authority.acquire("dora", "connection-a", 10.0)
    authority.disconnect("connection-b", 10.1)
    authority.authorize(token, "connection-a", 10.2, "command")


def test_owner_disconnect_revokes_and_requires_certified_drain(authority):
    token = authority.acquire("dora", "connection-a", 10.0)
    authority.disconnect("connection-a", 10.1)
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", 10.2, "command")
    with pytest.raises(ValueError):
        authority.acquire("h2", "connection-b", 10.3)
    authority.drain_completed(10.4)
    authority.acquire("h2", "connection-b", 10.5)


def test_fault_revokes_commands_and_cannot_be_cleared_by_reacquiring(authority):
    token = authority.acquire("dora", "connection-a", 10.0)
    authority.fault("downstream action not yet drained")
    for owner in ("h2", "dora"):
        with pytest.raises(ValueError):
            authority.acquire(owner, "connection-b", 10.1)
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", 10.1, "command")
    with pytest.raises(ValueError):
        authority.renew(token, "connection-a", 10.1)


@pytest.mark.parametrize("now", [math.nan, math.inf, -math.inf])
def test_nonfinite_clock_cannot_create_an_unexpiring_lease(authority, now):
    with pytest.raises(ValueError):
        authority.acquire("dora", "connection-a", now)


@pytest.mark.parametrize("now", [math.nan, math.inf, -math.inf, 9.9])
def test_invalid_or_backwards_clock_cannot_renew_or_authorize(authority, now):
    token = authority.acquire("dora", "connection-a", 10.0)
    with pytest.raises(ValueError):
        authority.authorize(token, "connection-a", now, "command")
    with pytest.raises(ValueError):
        authority.renew(token, "connection-a", now)


def recovery_fixture(monkeypatch):
    """A faulted gate with complete, fresh and already stationary hold evidence."""
    monkeypatch.setattr(gateway.time, "monotonic", lambda: 10.0)
    gate = gateway.Gateway.__new__(gateway.Gateway)
    gate.mutex = threading.RLock()
    gate.closing = False
    gate.authority = LeaseAuthority()
    gate.authority.fault("test recovery")
    gate.pending = set()
    gate.hold_pending = set()
    gate.raw_active = {name: set() for name in gateway.CONTROLLERS}
    gate.required_holds = set(gateway.CONTROLLERS[:3])
    gate.holds = {name: [0.0] * len(gateway.JOINTS[name]) for name in gate.required_holds}
    gate.controllers = {name: "active" for name in gate.required_holds}
    gate.controller_command_interfaces = {name: ['position'] for name in gate.required_holds}
    gate.controllers_at = 10.0
    gate.list_future = None
    gate.list_sent_at = None
    gate.list_service_ready = True
    gate.list_requests = gate.list_timeouts = gate.list_failures = 0
    gate.list_last_error = gate.list_last_rtt_s = None
    gate.drain_at = 8.0
    gate.stationary_since = 9.0
    gate.reference = lambda name: list(gate.holds[name])
    gate.measured_velocities = lambda names: [0.0] * len(names)
    gate.log = lambda *args, **kwargs: None
    return gate


def scoped_dora_fixture(monkeypatch, scope):
    gate = recovery_fixture(monkeypatch)
    gate.authority.reset_fault(10.)
    gate.readiness = lambda: None
    gate.reference = lambda name: [0.]*len(gateway.JOINTS[name])
    result = gate.request('dora-test', {'op':'acquire','owner':'dora','scope':scope})
    gate.sent = []
    gate.publish = lambda controller,q: gate.sent.append((controller,list(q)))
    return gate, result['token']


@pytest.mark.parametrize('scope,count', [('right_arm',7),('right_arm_gripper',8)])
def test_dora_scope_only_publishes_authorized_right_joints(monkeypatch, scope, count):
    gate,token = scoped_dora_fixture(monkeypatch,scope)
    q=[.01]*7+([-.3] if count==8 else [])
    gate.request('dora-test',{'op':'command','token':token,'positions':q})
    assert gate.sent == [(gateway.CONTROLLERS[0],q[:7])]+([(gateway.CONTROLLERS[1],q[7:])] if count==8 else [])
    for invalid in ([0.]*16, [0.]*(8 if count==7 else 7)):
        with pytest.raises(ValueError,match='finite positions'):
            gate.request('dora-test',{'op':'command','token':token,'positions':invalid})
    with pytest.raises(ValueError,match='no live authority'):
        gate.request('competitor',{'op':'command','token':token,'positions':q})


def test_task_release_holds_both_last_targets_not_delayed_controller_state(monkeypatch):
    gate,token = scoped_dora_fixture(monkeypatch,'right_arm_gripper')
    q=[.02]*7+[-.2]
    gate.request('dora-test',{'op':'command','token':token,'positions':q})
    # Pure selector used by the real drain path; controller state is still zero.
    gate.authority.release(token,'dora-test',10.)
    assert gate.drain_reference(gateway.CONTROLLERS[0]) == q[:7]
    assert gate.drain_reference(gateway.CONTROLLERS[1]) == q[7:]
    assert gate.drain_reference(gateway.CONTROLLERS[2]) == [0.]*7


def test_partial_task_publication_faults_and_preserves_attempted_pair(monkeypatch):
    gate,token = scoped_dora_fixture(monkeypatch,'right_arm_gripper')
    q=[.02]*7+[-.2]
    def publish(controller, positions):
        if controller == gateway.CONTROLLERS[1]: raise RuntimeError('DDS publication failed')
        gate.sent.append((controller,positions))
    gate.publish = publish
    drained=[];gate.begin_drain=lambda now:drained.append(now)
    with pytest.raises(ValueError,match='dispatch uncertain'):
        gate.request('dora-test',{'op':'command','token':token,'positions':q})
    assert gate.authority.state=='fault' and drained
    assert gate.drain_reference(gateway.CONTROLLERS[0])==q[:7]
    assert gate.drain_reference(gateway.CONTROLLERS[1])==q[7:]


@pytest.mark.parametrize("missing", gateway.CONTROLLERS[:3])
def test_incomplete_hold_set_cannot_clear_fault_even_when_remaining_axes_are_stationary(monkeypatch, missing):
    gate = recovery_fixture(monkeypatch)
    assert gate.drain_ready() is True
    del gate.holds[missing]
    assert gate.drain_ready() is False
    with pytest.raises(ValueError, match="goals drained and hold verified"):
        gate.request("recovery-session", {"op": "reset-fault"})
    assert gate.authority.state == "fault"


@pytest.mark.parametrize("failed_phase", ["acceptance", "result"])
def test_uncertain_raw_dispatch_is_quarantined_until_terminal_evidence(monkeypatch, failed_phase):
    # Fake ROS result type is sufficient: execute() itself creates no ROS node,
    # CAN socket or hardware plugin. Lost replies must never clear pending.
    fake_package = types.ModuleType("control_msgs")
    fake_action = types.ModuleType("control_msgs.action")
    fake_action.FollowJointTrajectory = SimpleNamespace(
        Result=lambda: SimpleNamespace(error_code=0, error_string=""))
    monkeypatch.setitem(sys.modules, "control_msgs", fake_package)
    monkeypatch.setitem(sys.modules, "control_msgs.action", fake_action)

    class Reply:
        def __init__(self, value=None, error=None):
            self.value, self.error = value, error

        def __await__(self):
            async def resolve():
                if self.error is not None:
                    raise self.error
                return self.value
            return resolve().__await__()

    gate = recovery_fixture(monkeypatch)
    gate.authority.reset_fault(10.0)
    token = gate.authority.acquire("h2", "h2-session", 10.0)
    controller = gateway.CONTROLLERS[0]
    gate.pending = {controller}
    gate.handles = {}
    raw = SimpleNamespace(
        accepted=True, goal_id=SimpleNamespace(uuid=[0] * 16),
        get_result_async=lambda: Reply(error=RuntimeError("terminal reply lost")))
    reply = (Reply(error=RuntimeError("acceptance reply lost"))
             if failed_phase == "acceptance" else Reply(raw))
    gate.action_clients = {controller: SimpleNamespace(
        server_is_ready=lambda: True, send_goal_async=lambda *a, **k: reply)}
    aborted = []
    outer = SimpleNamespace(
        request=SimpleNamespace(trajectory=SimpleNamespace(header=SimpleNamespace(
            frame_id=f"openarm-lease:{token}:{'a' * 32}:"))),
        publish_feedback=lambda _: None, abort=lambda: aborted.append(True))
    result = asyncio.run(gate.execute(controller, outer))
    assert result.error_code == -1 and aborted == [True]
    assert controller in gate.pending
    assert gate.authority.state == "fault"
    if failed_phase == "result":
        assert gate.handles[controller] is raw
    assert gate.drain_ready() is False
    with pytest.raises(ValueError):
        gate.request("recovery-session", {"op": "reset-fault"})


@pytest.mark.parametrize("operation", ["acquire", "renew", "command", "release", "freeze", "reset-fault", "synchronize-hold"])
def test_graceful_shutdown_rejects_all_mutating_requests(monkeypatch, operation):
    gate = recovery_fixture(monkeypatch)
    gate.closing = True
    with pytest.raises(ValueError, match="shutting down"):
        gate.request("late-client", {"op": operation, "owner": "h2"})
