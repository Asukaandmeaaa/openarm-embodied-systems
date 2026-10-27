"""Real subprocess/IPC regressions using only synthetic feedback, never CAN."""
import ast
import ctypes
import os
from pathlib import Path
import socket
import sys
import time

import pytest

from scripts.act_trial import isolated_telemetry as module


def observer(tmp_path, monkeypatch, mode='normal'):
    fixture = Path(__file__).with_name('telemetry_fake_worker.py')
    monkeypatch.setattr(module, 'worker_command',
        lambda output, interfaces, descriptor: [sys.executable, str(fixture), mode, str(descriptor)])
    return module.Telemetry(tmp_path/'raw.jsonl')


def test_parent_gil_block_for250ms_does_not_block_observer_process(tmp_path, monkeypatch):
    telemetry = observer(tmp_path, monkeypatch)
    try:
        before = telemetry.snapshot()
        assert telemetry.worker_pid != os.getpid()
        assert before['fixture_pid'] == telemetry.worker_pid
        assert before['observer_backend'] == 'isolated_receive_only_process'
        assert set(before['fixture_thread_limits'].values()) == {'1'}
        # PyDLL deliberately HOLDS this interpreter's GIL during native sleep.
        # Threads would stop here; the independent observer must keep sampling.
        libc = ctypes.PyDLL(None)
        libc.usleep.argtypes = [ctypes.c_uint]
        libc.usleep.restype = ctypes.c_int
        assert libc.usleep(250_000) == 0
        after = telemetry.snapshot()
        assert after['fixture_count']-before['fixture_count'] >= 50
        assert after['max_age_s'] < .1
        assert after['command_max_age_s'] < .1
    finally:
        telemetry.close()
    assert telemetry.process.poll() == 0


@pytest.mark.parametrize('mode, error', [('stale', 'stale CAN feedback'),
    ('stale_command', 'stale CAN commands'), ('future', 'stale CAN feedback')])
def test_parent_rechecks_original_source_age_after_ipc(tmp_path, monkeypatch, mode, error):
    telemetry = observer(tmp_path, monkeypatch, mode)
    try:
        with pytest.raises(RuntimeError, match=error):
            telemetry.snapshot()
    finally:
        telemetry.close()


def test_transport_timeout_is_latched_and_late_reply_cannot_be_reused(tmp_path, monkeypatch):
    telemetry = observer(tmp_path, monkeypatch, 'delay')
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='IPC failed'):
            telemetry.snapshot()
        assert time.monotonic()-started < .15
        with pytest.raises(RuntimeError, match='IPC failed'):
            telemetry.snapshot()
    finally:
        telemetry.close()
    assert telemetry.process.poll() is not None


def test_worker_crash_fails_closed_without_orphan(tmp_path, monkeypatch):
    telemetry = observer(tmp_path, monkeypatch, 'crash')
    try:
        with pytest.raises(RuntimeError, match='IPC failed'):
            telemetry.snapshot()
    finally:
        telemetry.close()
    assert telemetry.process.poll() == 9


def test_latched_can_error_survives_process_boundary(tmp_path, monkeypatch):
    telemetry = observer(tmp_path, monkeypatch, 'latched_error')
    try:
        with pytest.raises(RuntimeError, match='CAN source timestamp delayed'):
            telemetry.snapshot()
        assert telemetry.error == 'CAN source timestamp delayed: fixture'
        with pytest.raises(RuntimeError, match='CAN source timestamp delayed'):
            telemetry.snapshot()
    finally:
        telemetry.close()


def test_worker_startup_failure_is_reported(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='fixture startup error'):
        observer(tmp_path, monkeypatch, 'startup_error')


def test_parent_disconnect_stops_worker(tmp_path, monkeypatch):
    telemetry = observer(tmp_path, monkeypatch)
    telemetry.channel.close()
    telemetry.process.wait(timeout=2.)
    telemetry.close()
    assert telemetry.process.returncode == 0


def test_cannot_relax_source_age_in_process_facade(tmp_path, monkeypatch):
    telemetry = observer(tmp_path, monkeypatch)
    try:
        for age in (.2, 0., float('inf'), float('nan')):
            with pytest.raises(ValueError, match='source age'):
                telemetry.snapshot(age)
    finally:
        telemetry.close()


def test_collection_nodes_use_isolated_observation_and_worker_has_no_ros_or_control_imports():
    root = Path(__file__).resolve().parents[1]
    for name in ('zero', 'grasp_test', 'continuous_execute', 'place_test', 'waypoint_place_test'):
        tree = ast.parse((root/'vision_grasp_baseline'/f'{name}.py').read_text())
        assert any(isinstance(node, ast.ImportFrom) and node.module == 'scripts.act_trial.isolated_telemetry'
                   for node in ast.walk(tree))
    tree = ast.parse((root/'scripts/act_trial/telemetry_worker.py').read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == ['isolated_telemetry', 'telemetry']
    assert 'rclpy' not in (root/'scripts/act_trial/telemetry_worker.py').read_text()


def test_ipc_rejects_oversized_records_before_json_decode():
    import struct
    first, second = socket.socketpair()
    try:
        first.sendall(struct.pack('!I', module.MAX_RECORD+1))
        with pytest.raises(RuntimeError, match='record size'):
            module.receive_record(second, time.monotonic()+.1)
    finally:
        first.close()
        second.close()
