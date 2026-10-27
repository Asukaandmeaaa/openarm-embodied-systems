import json
from unittest.mock import Mock
import pytest
from scripts.act_trial import manual_experiment as m


@pytest.fixture
def setup(monkeypatch, tmp_path):
    trial = tmp_path/'act'
    trial.mkdir()
    monkeypatch.setenv('SOCKET', str(tmp_path/'policy.sock'))
    monkeypatch.setattr(m, 'require_home_handoff', Mock())
    monkeypatch.setattr(m, 'observation_processes', lambda _: [])
    monkeypatch.setattr(m, 'profile_from_config', lambda _: {'name': 'task'})
    monkeypatch.setattr(m, 'require_fresh_trial', Mock(return_value={'name': 'task'}))
    (trial/'trial_config.json').write_text('{}')
    rotate = Mock()
    monkeypatch.setattr(m, 'next_trial', rotate)
    return tmp_path, trial, rotate


def test_completed_attempt_rotates_preserving_profile(setup):
    run, trial, rotate = setup
    (trial/'result.json').write_text(json.dumps({'started': True}))
    m.prepare_policy_attempt(run, trial, run)
    rotate.assert_called_once_with(run, trial, profile='task', control=run)


def test_completed_attempt_with_old_source_reaches_repreparation(setup):
    from scripts.act_trial.motion_generator import contract, from_config
    run, trial, rotate = setup
    previous = {'motion_generator': contract('synchronized_v1')}
    previous['motion_generator']['source_sha256'] = {'guard.py': 'previous-version'}
    (trial/'trial_config.json').write_text(json.dumps(previous))
    (trial/'result.json').write_text(json.dumps({'started': True}))
    assert from_config(previous) == 'synchronized_v1'
    m.prepare_policy_attempt(run, trial, run)
    rotate.assert_called_once_with(run, trial, profile='task', control=run)


def test_fresh_attempt_does_not_rotate(setup):
    run, trial, rotate = setup
    m.prepare_policy_attempt(run, trial, run)
    rotate.assert_not_called()


def test_active_process_refused_before_archive(setup, monkeypatch):
    run, trial, rotate = setup
    monkeypatch.setattr(m, 'observation_processes', lambda _: [123])
    with pytest.raises(SystemExit, match='Stop old'):
        m.prepare_policy_attempt(run, trial, run)
    rotate.assert_not_called()


def test_socket_refused_before_archive(setup):
    run, trial, rotate = setup
    (run/'policy.sock').touch()
    with pytest.raises(ValueError, match='socket already'):
        m.prepare_policy_attempt(run, trial, run)
    rotate.assert_not_called()


def test_gateway_fault_propagates_without_bypass(setup):
    run, trial, rotate = setup
    (trial/'result.json').write_text('{"started":true}')
    rotate.side_effect = ValueError('gateway fault')
    with pytest.raises(ValueError, match='gateway fault'):
        m.prepare_policy_attempt(run, trial, run)


def test_shell_prepares_before_opening_log():
    from pathlib import Path
    source = Path('scripts/actctl').read_text()
    assert source.index('manual_experiment prepare-policy') < source.index('tee -a "$ACT_TRIAL_DIR/policy.log"')
    assert 'flock -n 8' in source
