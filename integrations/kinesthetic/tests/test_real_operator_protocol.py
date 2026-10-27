import json
import threading
from types import SimpleNamespace

import pytest

from openarm_kinesthetic.real import OperatorPacket, OperatorWatchdog
from openarm_kinesthetic.real import operator_protocol


def test_live_clock_fresh_packet_after_earlier_timer_snapshot_stays_passive(monkeypatch):
    clock=[10.0]
    monkeypatch.setattr(operator_protocol,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    guard=OperatorWatchdog(.25)
    guard.ingest(packet())
    earlier_timer_snapshot=10.049
    clock[0]=10.050
    drag=packet(sequence=1,active=True,deadman_pressed=True,arm_motion_source='physical_kinesthetic')
    guard.ingest(drag)
    clock[0]=10.051
    # The bridge now asks for a fresh atomic decision instead of passing the
    # obsolete timer snapshot, which would have latched a false timeout.
    assert earlier_timer_snapshot<guard.last_receive_monotonic_s
    assert guard.decision().request_passive
    clock[0]=10.301
    assert guard.decision().stop_reason=='operator_watchdog_timeout'
    clock[0]=10.302
    assert not guard.ingest(packet(sequence=2,active=True,deadman_pressed=True,
                                   arm_motion_source='physical_kinesthetic')).request_passive


def test_live_receive_cannot_be_overtaken_by_timer_clock_read(monkeypatch):
    guard=OperatorWatchdog()
    receiving=threading.Event();release=threading.Event();timer_clock=threading.Event()
    calls=[];decisions=[]
    def clock():
        calls.append(1)
        if len(calls)==1:
            receiving.set()
            assert release.wait(2)
            return 10.0
        timer_clock.set()
        return 10.001
    monkeypatch.setattr(operator_protocol,'time',SimpleNamespace(monotonic=clock))
    receiver=threading.Thread(target=lambda:guard.ingest(packet()))
    timer=threading.Thread(target=lambda:decisions.append(guard.decision()))
    receiver.start();assert receiving.wait(2);timer.start()
    try:
        assert not timer_clock.wait(.05)
    finally:
        release.set();receiver.join(2);timer.join(2)
    assert not receiver.is_alive() and not timer.is_alive()
    assert decisions[0].command_fresh and decisions[0].armed


def packet(**overrides) -> OperatorPacket:
    values = {
        "schema_version": 1,
        "session_id": "session-a",
        "sequence": 0,
        "sent_monotonic_s": 10.0,
        "connected": True,
        "deadman_pressed": False,
        "active": False,
        "stop_latched": False,
        "open_sequence": 0,
        "close_sequence": 0,
        "arm_motion_source": "idle",
    }
    values.update(overrides)
    return OperatorPacket(**values)


def test_packet_json_is_strict_and_round_trips() -> None:
    original = packet()
    assert OperatorPacket.from_json(original.to_json()) == original
    raw = json.loads(original.to_json())
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        OperatorPacket.from_json(json.dumps(raw))


def test_packet_cannot_carry_arm_motion_or_bypass_deadman() -> None:
    with pytest.raises(ValueError, match="unsupported real arm motion source"):
        packet(arm_motion_source="gamepad_cartesian").validate()
    with pytest.raises(ValueError, match="deadman/stop"):
        packet(
            active=True,
            deadman_pressed=False,
            arm_motion_source="cartesian_jog",
        ).validate()


def test_new_session_requires_release_before_passive_enable() -> None:
    gate = OperatorWatchdog(0.25)
    first = gate.ingest(packet(sequence=1, deadman_pressed=True, active=True, arm_motion_source="physical_kinesthetic"), 1.0)
    assert not first.request_passive
    assert first.stop_reason == "deadman_release_required"
    released = gate.ingest(packet(sequence=2), 1.02)
    assert released.armed and not released.request_passive
    enabled = gate.ingest(packet(sequence=3, deadman_pressed=True, active=True, arm_motion_source="physical_kinesthetic"), 1.04)
    assert enabled.request_passive


def test_release_stop_disconnect_and_timeout_all_fail_rigid() -> None:
    gate = OperatorWatchdog(0.25)
    gate.ingest(packet(sequence=1), 1.0)
    assert gate.ingest(packet(sequence=2, deadman_pressed=True, active=True, arm_motion_source="physical_kinesthetic"), 1.01).request_passive
    assert not gate.ingest(packet(sequence=3), 1.02).request_passive
    assert not gate.ingest(packet(sequence=4, connected=False), 1.03).request_passive
    gate.ingest(packet(session_id="session-b", sequence=1), 2.0)
    gate.ingest(packet(session_id="session-b", sequence=2, deadman_pressed=True, active=True, arm_motion_source="physical_kinesthetic"), 2.01)
    assert not gate.decision(2.27).request_passive
    assert gate.stop_reason == "operator_watchdog_timeout"


def test_gripper_edges_are_counted_once_and_blocked_during_stop() -> None:
    gate = OperatorWatchdog(0.25)
    gate.ingest(packet(sequence=1), 1.0)
    opened = gate.ingest(packet(sequence=2, open_sequence=1), 1.01)
    assert opened.request_open and not opened.request_close
    replayed = gate.ingest(packet(sequence=3, open_sequence=1), 1.02)
    assert not replayed.request_open
    stopped = gate.ingest(
        packet(sequence=4, open_sequence=2, stop_latched=True), 1.03
    )
    assert not stopped.request_open


def test_gripper_edge_remains_available_during_physical_drag() -> None:
    gate = OperatorWatchdog(0.25)
    gate.ingest(packet(sequence=0), 1.0)
    decision = gate.ingest(
        packet(
            sequence=1,
            deadman_pressed=True,
            active=True,
            arm_motion_source="physical_kinesthetic",
            open_sequence=1,
        ),
        1.01,
    )
    assert decision.request_passive
    assert decision.request_open


def test_stop_recovery_requires_release_before_next_deadman_press() -> None:
    gate = OperatorWatchdog(0.25)
    gate.ingest(packet(sequence=1), 1.0)
    assert gate.ingest(
        packet(sequence=2, deadman_pressed=True, active=True, arm_motion_source="physical_kinesthetic"), 1.01
    ).request_passive
    stopped = gate.ingest(
        packet(
            sequence=3,
            deadman_pressed=True,
            active=False,
            stop_latched=True,
            arm_motion_source="idle",
        ),
        1.02,
    )
    assert not stopped.request_passive and not stopped.armed
    released = gate.ingest(packet(sequence=4, stop_latched=True), 1.03)
    assert not released.request_passive and released.armed
    resumed = gate.ingest(
        packet(sequence=5, deadman_pressed=True, active=True, arm_motion_source="physical_kinesthetic"), 1.04
    )
    assert resumed.request_passive


def test_lt_servo_and_lt_lb_drag_are_mutually_exclusive() -> None:
    gate = OperatorWatchdog(0.25)
    gate.ingest(packet(sequence=1), 1.0)
    servo = gate.ingest(
        packet(
            sequence=2,
            deadman_pressed=True,
            active=True,
            arm_motion_source="cartesian_jog",
        ),
        1.01,
    )
    assert servo.control_mode == "cartesian_jog"
    assert not servo.request_passive

    drag = gate.ingest(
        packet(
            sequence=3,
            deadman_pressed=True,
            active=True,
            arm_motion_source="physical_kinesthetic",
        ),
        1.02,
    )
    assert drag.control_mode == "physical_kinesthetic"
    assert drag.request_passive


def test_stale_or_reordered_packets_do_not_retrigger() -> None:
    gate = OperatorWatchdog(0.25)
    gate.ingest(packet(sequence=5), 1.0)
    stale = gate.ingest(packet(sequence=4, open_sequence=10), 1.01)
    assert not stale.request_open
    assert not stale.packet_accepted
