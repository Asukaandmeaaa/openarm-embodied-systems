"""Receive-only CAN health tests with fake sockets; no hardware operations."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import struct
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.act_trial import telemetry as module

WALL = 1_800_000_000_000_000_000
MONO = 2_000_000_000


class ReceiveSocket:
    """The fake has no send method, making accidental TX fail immediately."""
    def __init__(self, packet):
        self.packet = packet

    def recvmsg(self, *args):
        return self.packet


class Selector:
    def __init__(self, entries, stop):
        self.entries = entries
        self.stop = stop

    def select(self, timeout):
        self.stop.set()
        return [(SimpleNamespace(fileobj=ReceiveSocket(packet), data=side), None)
                for side, packet in self.entries]


class MemoryLog:
    def __init__(self):
        self.stream = io.StringIO()

    def append(self, row):
        self.stream.write(json.dumps(row)+'\n')

    def check(self):
        pass


def packet(axis, *, status=1, stamp=WALL, overflow=None, flags=0,
           motor_id=None, can_id=None, payload=None, fd=False):
    motor_id = axis+1 if motor_id is None else motor_id
    can_id = 0x11+axis if can_id is None else can_id
    payload = (bytes([status*16+motor_id, 0x80, 0, 0x80, 8, 0, 25, 26])
               if payload is None else payload)
    raw = struct.pack('=IBBBB8s', can_id, len(payload), 0, 0, 0, payload)
    if fd:
        raw += bytes(56)
    ancillary = []
    if stamp is not None:
        ancillary.append((socket.SOL_SOCKET, 35,
                          struct.pack('=qq', stamp//10**9, stamp % 10**9)))
    if overflow is not None:
        ancillary.append((socket.SOL_SOCKET, 40, struct.pack('=I', overflow)))
    return raw, ancillary, flags, None


def observer(monkeypatch, entries=()):
    """Construct only passive state; never call Telemetry's socket constructor."""
    obj = module.Telemetry.__new__(module.Telemetry)
    obj.log = MemoryLog()
    obj.stream = obj.log.stream  # Test-only, synchronous inspection of rows.
    obj.stop_event = threading.Event()
    obj.lock = threading.Lock()
    obj.latest = {}
    obj.latest_command = {}
    obj.error = None
    obj.drops = {}
    obj.count = 0
    obj.selector = Selector(entries, obj.stop_event)
    monkeypatch.setattr(module.time, 'time_ns', lambda: WALL)
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: MONO)
    monkeypatch.setattr(module.time, 'monotonic', lambda: MONO/1e9)
    return obj


def fresh_entries():
    entries = []
    for side in range(2):
        for axis in range(8):
            entries.append((side, packet(axis, can_id=axis+1,
                                         payload=bytes.fromhex('8000800800800800'),
                                         stamp=WALL-10_000_000)))
            entries.append((side, packet(axis, stamp=WALL-10_000_000)))
    return entries


def test_all16_feedback_kernel_age_status_and_order(monkeypatch):
    obj = observer(monkeypatch, fresh_entries())
    obj._run()
    assert obj.error is None
    result = obj.snapshot()
    assert len(result['positions_rad']) == 16
    assert result['motor_status'] == [1]*16
    assert result['source_monotonic_ns'] == [MONO-10_000_000]*16
    assert result['max_age_s'] == pytest.approx(.01)
    assert len(result['command_positions_rad']) == 16
    assert result['command_kp'] == pytest.approx([2048*500/4095]*16)
    assert result['command_kd'] == pytest.approx([2048*5/4095]*16)
    assert result['command_source_monotonic_ns'] == [MONO-10_000_000]*16
    assert result['command_max_age_s'] == pytest.approx(.01)
    assert obj.count == 32
    assert len(obj.stream.getvalue().splitlines()) == 32


def test_missing_axis_cannot_be_hidden_by_fresh_ros_publication(monkeypatch):
    obj = observer(monkeypatch, fresh_entries()[:-1])
    obj._run()
    with pytest.raises(RuntimeError, match=r'missing axes: \[15\]'):
        obj.snapshot()


@pytest.mark.parametrize('status', [0, 8, 9, 10, 11, 12, 13, 14, 15])
def test_disabled_or_fault_axis_blocks_snapshot(monkeypatch, status):
    entries = fresh_entries()
    entries[-1] = (1, packet(7, status=status))
    obj = observer(monkeypatch, entries)
    obj._run()
    with pytest.raises(RuntimeError, match='disabled or reports a fault'):
        obj.snapshot()


def test_axis_id_mismatch_is_latched(monkeypatch):
    obj = observer(monkeypatch, [(0, packet(0, motor_id=2))])
    obj._run()
    assert obj.error == 'CAN response motor ID mismatch'
    assert not obj.latest
    with pytest.raises(RuntimeError, match='motor ID mismatch'):
        obj.snapshot()


@pytest.mark.parametrize('change,error', [
    ({'overflow': 1}, 'receive overflow'),
    ({'can_id': 0x20000001}, 'CAN error frame'),
    ({'flags': socket.MSG_TRUNC}, 'truncated packet'),
    ({'flags': socket.MSG_CTRUNC}, 'truncated packet'),
    ({'flags': socket.MSG_DONTROUTE | socket.MSG_TRUNC}, 'truncated packet'),
    ({'flags': socket.MSG_DONTROUTE | socket.MSG_CTRUNC}, 'truncated packet'),
    ({'stamp': None}, 'invalid CAN timestamp'),
    ({'stamp': WALL-100_000_001}, 'timestamp delayed'),
    ({'stamp': WALL+1}, 'clock discontinuity'),
])
def test_transport_errors_fail_closed(monkeypatch, change, error):
    obj = observer(monkeypatch, [(0, packet(0, **change))])
    obj._run()
    assert obj.error and error in obj.error
    with pytest.raises(RuntimeError, match=error):
        obj.snapshot()


def test_snapshot_checks_source_age_not_receive_age(monkeypatch):
    obj = observer(monkeypatch, fresh_entries())
    obj._run()
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: MONO+91_000_000)
    with pytest.raises(RuntimeError, match='stale CAN feedback'):
        obj.snapshot()


def test_future_monotonic_source_is_rejected(monkeypatch):
    obj = observer(monkeypatch, fresh_entries())
    obj._run()
    decoded, _ = obj.latest[0]
    obj.latest[0] = decoded, MONO+1
    with pytest.raises(RuntimeError, match='stale CAN feedback'):
        obj.snapshot()


def test_commands_and_parameter_replies_do_not_count_as_feedback(monkeypatch):
    entries = [
        (0, packet(0, can_id=1, payload=b'\xff'*7+b'\xfc')),
        (0, packet(0, can_id=1, payload=bytes.fromhex('8000800800800800'))),
        (0, packet(0, payload=bytes([1, 0, 0x33, 0, 0, 0, 0, 0]))),
    ]
    obj = observer(monkeypatch, entries)
    obj._run()
    assert obj.error is None
    assert not obj.latest
    with pytest.raises(RuntimeError, match='missing axes'):
        obj.snapshot()


def test_zero_overflow_counter_is_accepted(monkeypatch):
    entries = fresh_entries()
    entries[1] = (0, packet(0, overflow=0))
    obj = observer(monkeypatch, entries)
    obj._run()
    assert obj.error is None
    assert obj.snapshot()['motor_status'] == [1]*16


def test_feedback_without_all16_actual_commands_is_insufficient(monkeypatch):
    entries = fresh_entries()
    del entries[-2]  # Missing left gripper's actual MIT command.
    obj = observer(monkeypatch, entries)
    obj._run()
    with pytest.raises(RuntimeError, match=r'MIT commands missing axes: \[15\]'):
        obj.snapshot()


@pytest.mark.parametrize('stamp,error', [(MONO-101_000_000, 'stale CAN commands'),
                                         (MONO+1, 'stale CAN commands')])
def test_command_timestamp_freshness_is_independent_of_feedback(monkeypatch, stamp, error):
    obj = observer(monkeypatch, fresh_entries())
    obj._run()
    decoded, _ = obj.latest_command[0]
    obj.latest_command[0] = decoded, stamp
    with pytest.raises(RuntimeError, match=error):
        obj.snapshot()


def test_lifecycle_message_cannot_replace_missing_position_command(monkeypatch):
    entries = fresh_entries()
    entries[0] = (0, packet(0, can_id=1, payload=b'\xff'*7+b'\xfc'))
    obj = observer(monkeypatch, entries)
    obj._run()
    with pytest.raises(RuntimeError, match=r'MIT commands missing axes: \[0\]'):
        obj.snapshot()


def test_command_queue_delay_is_rejected_before_freshness_conversion(monkeypatch):
    entries = [(0, packet(0, can_id=1, payload=bytes.fromhex('8000800800800800'),
                           stamp=WALL-100_000_001))]
    obj = observer(monkeypatch, entries)
    obj._run()
    assert not obj.latest_command
    with pytest.raises(RuntimeError, match='timestamp delayed'):
        obj.snapshot()


@pytest.mark.parametrize('flags', [socket.MSG_DONTROUTE, socket.MSG_CONFIRM,
                                   socket.MSG_DONTROUTE | socket.MSG_CONFIRM])
@pytest.mark.parametrize('fd', [False, True])
def test_local_can_origin_flags_are_preserved_not_treated_as_truncation(monkeypatch, flags, fd):
    entries = fresh_entries()
    entries[0] = (0, packet(0, can_id=1, payload=bytes.fromhex('8000800800800800'),
                            flags=flags, fd=fd))
    obj = observer(monkeypatch, entries)
    obj._run()
    assert obj.error is None
    assert obj.snapshot()['motor_status'] == [1]*16
    raw_log = json.loads(obj.stream.getvalue().splitlines()[0])
    assert raw_log['socket_flags'] == flags
    assert raw_log['frame_bytes'] == (72 if fd else 16)
    assert raw_log['decoded']['kind'] == 'command'


def test_source_contains_no_can_transmit_or_motor_control_calls():
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    forbidden = {'send', 'sendall', 'sendmsg', 'sendto', 'write',
                 'enable_all', 'disable_all', 'set_zero_all', 'mit_control_all',
                 'send_position', 'publish', 'call_async'}
    # Stream.write is the sole write: persistent logging, not a CAN descriptor.
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    forbidden_calls = []
    for node in calls:
        if node.func.attr not in forbidden:
            continue
        if node.func.attr == 'write' and ast.unparse(node.func.value) == 'self.stream':
            continue
        forbidden_calls.append(ast.unparse(node.func))
    assert forbidden_calls == []
    assert any(node.func.attr == 'recvmsg' for node in calls)
    assert 'openarm_can' not in source


def test_live_ros_dm4340_scale_corrects_velocity_only():
    for axis in range(8):
        for kind in ('feedback', 'command'):
            can_id = 0x11+axis if kind == 'feedback' else 1+axis
            payload = (bytes([0x10+axis+1, 0x80, 0, 0xff, 0xf8, 0, 25, 26])
                       if kind == 'feedback' else bytes.fromhex('8000fff800800800'))
            original = module.decode_capture(can_id, payload)
            live = module.decode(can_id, payload)
            assert live['q'] == original['q']
            assert live['torque'] == original['torque']
            if axis in (2, 3):
                assert live['dq'] == pytest.approx(8.)
                assert live['capture_decoder_dq_rad_s'] == 10.
                assert live['velocity_scale_rad_s'] == 8.
            else:
                assert live == original


def test_raw_log_close_drains_every_row_in_order(tmp_path):
    path = tmp_path/'can.jsonl'
    log = module.RawCanLog(path)
    for i in range(100):
        log.append({'index': i})
    log.close()
    assert log.written == 100
    assert [json.loads(line)['index'] for line in path.read_text().splitlines()] == list(range(100))
    log.close()  # Idempotent without reopening/truncating the file.
    with pytest.raises(RuntimeError, match='already closed'):
        log.append({})


def test_slow_disk_does_not_block_can_reception_or_fresh_snapshot(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowStream(io.StringIO):
        def write(self, value):
            entered.set()
            assert release.wait(2)
            return super().write(value)
    monkeypatch.setattr(module.Path, 'open', lambda *a, **k: SlowStream())
    log = module.RawCanLog(tmp_path/'can.jsonl')
    log.append({'disk_write': 'blocked'})
    assert entered.wait(1)
    obj = observer(monkeypatch, fresh_entries())
    obj.log = log
    received = threading.Event()
    def receive():
        obj._run()
        received.set()
    thread = threading.Thread(target=receive)
    thread.start()
    try:
        assert received.wait(.2), 'CAN reception blocked behind diagnostic disk I/O'
        assert obj.snapshot()['motor_status'] == [1]*16
    finally:
        release.set()
        thread.join(1)
        log.close()
    assert log.written == 33


def test_raw_log_queue_overflow_is_latched_and_never_silently_drops(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowStream(io.StringIO):
        def write(self, value):
            entered.set()
            assert release.wait(2)
            return super().write(value)
    monkeypatch.setattr(module.Path, 'open', lambda *a, **k: SlowStream())
    log = module.RawCanLog(tmp_path/'can.jsonl', capacity=1)
    log.append({'first': 1})
    assert entered.wait(1)
    try:
        log.append({'second': 2})
        with pytest.raises(RuntimeError, match='queue overflow'):
            log.append({'third': 3})
        obj = observer(monkeypatch, fresh_entries())
        obj.log = log
        with pytest.raises(RuntimeError, match='queue overflow'):
            obj.snapshot()
    finally:
        release.set()
        with pytest.raises(RuntimeError, match='queue overflow'):
            log.close()
    assert log.written == 2


def test_raw_log_writer_error_blocks_snapshot(monkeypatch, tmp_path):
    class BrokenStream(io.StringIO):
        def write(self, value):
            raise OSError('disk full')
    monkeypatch.setattr(module.Path, 'open', lambda *a, **k: BrokenStream())
    log = module.RawCanLog(tmp_path/'can.jsonl')
    log.append({'index': 1})
    log.thread.join(1)
    assert not log.thread.is_alive()
    obj = observer(monkeypatch)
    obj.log = log
    with pytest.raises(RuntimeError, match='writer failed: disk full'):
        obj.snapshot()
    with pytest.raises(RuntimeError, match='writer failed: disk full'):
        log.close()
