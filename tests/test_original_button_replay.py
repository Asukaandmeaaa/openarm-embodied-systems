"""Offline tests: never open joystick, ROS, or CAN."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

PATH = Path('/home/lushilin/openarm_kinesthetic_ws/scripts/button_play_recorded_motion.py')
spec = importlib.util.spec_from_file_location('original_button_replay', PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def event(initial=False, value=1, index=1):
    return NS(kind='button', index=index, value=value, initial=initial)


def test_reuses_exact_original_command():
    assert replay.player_command(Path('/data/001.json')) == [
        'ros2', 'run', 'openarm_kinesthetic_recorder', 'recorded_motion_player',
        '--trajectory', '/data/001.json', '--allow-real-execution',
        '--playback-rate', '1.0', '--keyframe-settle-tolerance-rad', '0.04']


def test_no_start_on_initial_event_release_or_other_button():
    assert replay.pressed([event()], 1)
    for e in [event(initial=True), event(value=0), event(index=0)]:
        assert not replay.pressed([e], 1)


@pytest.mark.parametrize('mode', ['complete', 'failed', 'rt_stop', 'disconnect', 'conflict'])
def test_sequence_only_advances_after_success_and_new_press(monkeypatch, mode):
    calls = []
    class Pad:
        connected = True
        axes = []
        buttons = [0]*8
        count = 0
        def poll(self):
            self.count += 1
            if self.count in (2, 5):
                self.buttons[1] = 1
                calls.append('B')
                return [event()]
            if self.count == 4:
                self.buttons[1] = 0
            if self.count == 3 and mode == 'disconnect':
                self.connected = False
            return []
    pad = Pad()
    mapper = NS(button={'episode_capture_b': 1}, update=lambda *a, **k: NS(
        stop_latched=(mode == 'rt_stop' and pad.count == 3),
        deadman_pressed=False, drag_modifier_pressed=False))
    class Child:
        pid = 12345
        count = 0
        returncode = None
        def __init__(self, cmd, **kw):
            assert calls[-1] == 'B'
            assert kw['start_new_session']
            calls.append('player')
        def poll(self):
            self.count += 1
            if self.count >= 2:
                self.returncode = 2 if mode == 'failed' else 0
            return self.returncode
    monkeypatch.setattr(replay.subprocess, 'Popen', Child)
    monkeypatch.setattr(replay.time, 'sleep', lambda _: None)
    monkeypatch.setattr(replay, 'competing_processes', lambda: ['gateway'] if mode == 'conflict' else [])
    monkeypatch.setattr(replay, 'stop_player', lambda c: calls.append('stop') if c else None)
    log = lambda *a, **k: None
    if mode == 'complete':
        replay.play_sequence([Path('1.json'), Path('2.json')], pad, mapper, log)
        assert calls == ['B', 'player', 'B', 'player']
    else:
        with pytest.raises(RuntimeError):
            replay.play_sequence([Path('1.json'), Path('2.json')], pad, mapper, log)
        assert calls.count('player') == (0 if mode == 'conflict' else 1)
        if mode in ('rt_stop', 'disconnect'):
            assert calls[-1] == 'stop'


def test_stop_uses_original_SIGINT_not_kill(monkeypatch):
    calls = []
    monkeypatch.setattr(replay.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    child = NS(pid=123, poll=lambda: None, wait=lambda **kw: calls.append(kw))
    replay.stop_player(child)
    assert calls == [(123, replay.signal.SIGINT), {'timeout': 10}]
