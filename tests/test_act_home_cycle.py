from unittest.mock import MagicMock

import pytest

from scripts.act_trial import manual_experiment as m
from scripts.act_trial.keyboard import dispatch


def setup_cycle(monkeypatch, tmp_path, fault=None):
    client = MagicMock()
    client.request.return_value = {
        'status': {'state': 'fault' if fault else 'idle', 'fault': fault},
        'drain_ready': True, 'pending': [], 'raw_active': {},
        'hold_pending': [], 'synchronized': True,
    }
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(m, 'Client', factory)
    cleanup = MagicMock()
    recover = MagicMock()
    monkeypatch.setattr(m, 'stop_trial_observers', cleanup)
    monkeypatch.setattr(m, 'recover_home', recover)
    monkeypatch.setattr(m, 'wait_idle', MagicMock())
    return client, cleanup, recover


def test_cancel_has_no_side_effect(monkeypatch, tmp_path):
    client, cleanup, recover = setup_cycle(monkeypatch, tmp_path)
    m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'no')
    assert not (tmp_path/'stop.request').exists()
    cleanup.assert_not_called()
    recover.assert_not_called()
    client.request.assert_not_called()


def test_collision_recovery_plans_then_executes(monkeypatch, tmp_path):
    client, cleanup, recover = setup_cycle(
        monkeypatch, tmp_path, 'collision in right command proposal sweep')
    m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'HOME')
    client.request.assert_any_call('reset-fault')
    assert recover.call_count == 2
    assert recover.call_args.kwargs['execute'] is True


def test_other_fault_never_reset_or_move(monkeypatch, tmp_path):
    client, cleanup, recover = setup_cycle(monkeypatch, tmp_path, 'stale state')
    with pytest.raises(ValueError):
        m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'HOME')
    assert all(c.args != ('reset-fault',) for c in client.request.call_args_list)
    recover.assert_not_called()


def test_failed_cleanup_never_moves(monkeypatch, tmp_path):
    client, cleanup, recover = setup_cycle(monkeypatch, tmp_path)
    cleanup.side_effect = ValueError('still running')
    with pytest.raises(ValueError):
        m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'HOME')
    client.request.assert_not_called()
    recover.assert_not_called()


def test_keyboard_shortcut():
    run = MagicMock()
    dispatch('b', run=run)
    run.assert_called_once_with('home')


def test_home_does_not_mark_unused_trial_stopped(monkeypatch, tmp_path):
    setup_cycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'observation_processes', lambda _: [])
    m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'HOME')
    assert not (tmp_path/'stop.request').exists()


def test_home_requests_stop_for_running_observer(monkeypatch, tmp_path):
    setup_cycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'observation_processes', lambda _: [123])
    m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'HOME')
    assert (tmp_path/'stop.request').exists()


def test_home_requests_stop_when_motion_evidence_exists(monkeypatch, tmp_path):
    setup_cycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'observation_processes', lambda _: [])
    (tmp_path/'started.json').touch()
    m.home_cycle(tmp_path, tmp_path, tmp_path, confirm=lambda _: 'HOME')
    assert (tmp_path/'stop.request').exists()


def test_wait_for_transient_drain(monkeypatch, tmp_path):
    client, _, _ = setup_cycle(monkeypatch, tmp_path,
                              'collision in right command proposal sweep')
    ready = client.request.return_value
    client.request.side_effect = [dict(ready, drain_ready=False), ready]
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    assert m.wait_home_recovery_ready(client)['drain_ready']
    assert all(c.args == ('status',) for c in client.request.call_args_list)


def test_drain_timeout_has_diagnostics(monkeypatch, tmp_path):
    client, _, _ = setup_cycle(monkeypatch, tmp_path,
                              'collision in right command proposal sweep')
    client.request.return_value['drain_ready'] = False
    with pytest.raises(ValueError, match='drain_ready'):
        m.wait_home_recovery_ready(client, timeout=0)
    client.request.assert_called_once_with('status')
