"""No-hardware verification for the direct-CAN startup contract.

The route is measured feedback -> J1-J7 nominal zero -> side-prehome ->
lifted HOME -> J8 open. The tests intentionally keep J8 out of the initial
zero gate and verify that a nonzero startup pose is driven to zero before the
HOME phases begin.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('guarded_home', ROOT / 'nodes/dora-openarm/src/dora_openarm/guarded_home.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Guard = module.Guard


def config():
    return json.loads((ROOT / 'guarded-home-right.json').read_text())


def ready():
    guard = Guard(config(), np.zeros(8), 0)
    now = 0
    while guard.state in ('returning_zero', 'homing') and now < 120:
        previous = guard.command.copy()
        now += .004
        guard.tick(previous, now)
        limits = np.asarray(config()['joint_velocity_limits']) * .004
        assert np.all(np.abs(guard.command - previous) <= limits + 1e-12)
        if guard.phase < 3:
            assert guard.command[-1] == 0  # open only after arm HOME
    assert guard.state == 'home_ready'
    return guard, now


def test_home_route_velocity_gripper_and_start():
    cfg = config()
    # J8 is intentionally not part of H1's mechanical-zero gate.
    guard = Guard(cfg, np.array([0, 0, 0, 0, 0, 0, 0, -0.35]), 0)
    now = 0
    while guard.state in ('returning_zero', 'homing') and now < 120:
        previous = guard.command.copy()
        now += .004
        guard.tick(previous, now)
    assert guard.state == 'home_ready'
    np.testing.assert_allclose(guard.command, config()['home'])
    assert guard.start(guard.command, now)
    assert not guard.start(guard.command, now)  # duplicate does not restart


def test_homing_transition_uses_bounded_time_parameterized_steps():
    """A delayed Dora tick must not issue the ROS phase-boundary jump."""
    guard = Guard(config(), np.zeros(8), 0)
    now = 0.0
    while guard.phase == 0:
        now += 0.01
        guard.tick(guard.command, now)
    assert guard.state == 'homing'
    previous = guard.command.copy()
    # A 20 ms service interval used to produce J7's 0.126 rad max-velocity
    # step.  HOME's own trajectory speed leaves margin under the unchanged
    # 0.120 rad tracking guard.
    now += 0.02
    command = guard.tick(previous, now)
    assert abs(command[6] - previous[6]) < config()['tracking_tolerance']
    assert np.max(np.abs(command - previous)) < config()['tracking_tolerance']


def test_homing_tolerates_two_sample_feedback_delay():
    """Healthy feedback arriving two control ticks late still reaches HOME."""
    guard = Guard(config(), np.zeros(8), 0)
    history = [guard.command.copy(), guard.command.copy()]
    now = 0.0
    while guard.state in ('returning_zero', 'homing') and now < 30.0:
        now += 0.02
        measured = history.pop(0)
        history.append(guard.command.copy())
        command = guard.tick(measured, now)
        assert np.all(np.isfinite(command))
    assert guard.state == 'home_ready'
    np.testing.assert_allclose(guard.command, config()['home'])


def test_homing_tolerates_paced_refresh_interval():
    """The driver's 20 Hz explicit refresh does not create a HOME jump."""
    guard = Guard(config(), np.zeros(8), 0)
    delayed = [guard.command.copy()]
    now = 0.0
    while guard.state in ('returning_zero', 'homing') and now < 30.0:
        now += 0.05  # feedback_driver's explicit refresh interval
        measured = delayed.pop(0)
        command = guard.tick(measured, now)
        delayed.append(command.copy())
    assert guard.state == 'home_ready'


def test_active_command_step_keeps_feedback_tracking_margin():
    """ACT interpolation uses the same delayed-feedback command envelope."""
    guard, now = ready()
    assert guard.start(guard.command, now)
    target = guard.command.copy()
    target[6] += 0.5
    guard.action(target, now)
    previous = guard.command.copy()
    command = guard.tick(previous, now + 0.02, feedback_age=0.06)
    assert np.max(np.abs(command - previous)) < config()['tracking_tolerance']


def test_homing_uses_actual_feedback_age_at_100hz():
    """A 20 Hz CAN refresh must not be treated as a 100 Hz measurement."""
    guard = Guard(config(), np.zeros(8), 0)
    measured = guard.command.copy()
    now = 0.0
    for tick in range(1, 3001):
        now += 0.01
        if tick % 6 == 1:
            measured = guard.command.copy()
        command = guard.tick(measured, now, feedback_age=0.06)
        assert np.all(np.isfinite(command))
        if guard.state == 'home_ready':
            break
    assert guard.state == 'home_ready'


@pytest.mark.parametrize('pose', [np.full(8, np.nan), [0]*7, [0]*7+[float('inf')], [3.0]*8])
def test_invalid_initial_pose(pose):
    with pytest.raises(ValueError):
        Guard(config(), pose, 0)


def test_no_bypass_or_early_actions():
    guard = Guard(config(), np.array([0.5, 0, 0, 0, 0, 0, 0, -0.35]), 0)
    assert not guard.start(np.zeros(8), 0)
    guard.action(np.ones(8), 0)
    expected_zero = np.asarray(config()['zero'], dtype=float)
    expected_zero[-1] = -0.35
    np.testing.assert_allclose(guard.target, expected_zero)
    guard.halt('stop')
    assert not guard.start(np.zeros(8), .01)


def test_nonzero_start_is_driven_to_zero_before_home():
    cfg = config()
    initial = np.array([0.05, -0.04, 0.03, 0.02, -0.02, 0.01, 0.04, -0.20])
    guard = Guard(cfg, initial, 0)
    assert guard.state == 'returning_zero'
    assert np.allclose(guard.target[:-1], cfg['zero'][:-1])
    now = 0.0
    while guard.state in ('returning_zero', 'homing') and now < 120:
        now += 0.004
        guard.tick(guard.command, now)
    assert guard.state == 'home_ready'
    np.testing.assert_allclose(guard.command, cfg['home'], atol=cfg['tolerance'])


def test_reported_nano_pose_enters_bounded_zero_recovery():
    # 2026-10-12 15:10:34: J4 is outside the HOME/ACT lower bound, but inside
    # the separately declared activation envelope.
    initial = [0.1157778, -0.0089647, -0.1253147, -0.1955062,
               0.7047761, -0.1199741, 0.1249332, -0.0562676]
    guard = Guard(config(), initial, 0)
    assert guard.state == 'returning_zero'
    assert guard.recovery_scope
    np.testing.assert_allclose(guard.target[:-1], np.zeros(7))
    for k in range(1, 201):
        previous = guard.command.copy()
        guard.tick(previous, k * .01)
        # Synchronized seven-axis interpolation, with J8 unchanged.
        np.testing.assert_allclose(guard.command[:-1],
                                   np.array(initial[:-1]) * (1 - k / 200), atol=.004)
        assert guard.command[-1] == initial[-1]
        assert guard.phase == 0
    np.testing.assert_allclose(guard.command[:-1], 0, atol=1e-12)
    now = 2.0
    phases = []
    while guard.state != 'home_ready' and now < 10:
        previous_phase = guard.phase
        previous = guard.command.copy()
        now += .01
        guard.tick(previous, now)
        if guard.phase != previous_phase:
            phases.append(guard.phase)
        if guard.phase < 3:
            assert guard.command[-1] == initial[-1]
        if guard.phase > 0:
            assert not guard.recovery_scope
            assert guard.command[3] >= 0
    assert phases == [1, 2, 3, 4]
    assert guard.state == 'home_ready'
    np.testing.assert_allclose(guard.command, config()['home'])
    assert config()['lower'][3] == 0
    assert config()['return_to_zero']['lower'][3] < 0


def test_zero_target_requires_measured_arrival(capsys):
    # A valid nonzero start must move toward zero, not redefine it as zero.
    initial = np.array([0.10, 0, 0, 0, 0, 0, 0, -0.20])
    guard = Guard(config(), initial, 0)
    for i in range(1, 301):
        guard.tick(initial, i * .01)  # motor never follows the zero command
    np.testing.assert_allclose(guard.command[:-1], np.zeros(7), atol=1e-12)
    assert guard.command[-1] == initial[-1]
    assert guard.state == 'returning_zero'
    assert guard.phase == 0
    assert 'HOME verified phase 0' not in capsys.readouterr().out


def test_fault_after_phase_zero_is_visible_without_ui(capsys):
    guard = Guard(config(), np.zeros(8), 0)
    now = 0.0
    while guard.phase == 0 and now < 4:
        now += .01
        guard.tick(np.zeros(8), now)
    assert guard.phase == 1
    assert 'HOME_TARGET: phase=1' in capsys.readouterr().out
    for _ in range(50):
        now += .01
        guard.tick(np.zeros(8), now)  # motor stays near zero after phase 0
    output = capsys.readouterr().out
    assert guard.state == 'home_error'
    assert 'HOME_ERROR: phase=1 state=homing' in output
    assert 'command tracking error: J' in output
    assert output.count('HOME_ERROR:') == 1
    assert 'HOME verified phase 1' not in output
    first_error = guard.error
    command = guard.command.copy()
    guard.tick(np.zeros(8), now + config()['watchdog'] + 1)
    assert guard.error == first_error
    np.testing.assert_array_equal(guard.command, command)
    assert capsys.readouterr().out == ''


def test_hold_only_never_homes():
    initial = np.zeros(8); initial[3] = -.195506
    guard = Guard(config(), initial, 0, True)
    for i in range(100):
        np.testing.assert_array_equal(guard.tick(initial, i * .004), initial)
    assert guard.state == 'hold_only'
    assert not guard.start(np.zeros(8), 1)


def test_watchdog_no_large_catchup():
    guard = Guard(config(), np.zeros(8), 0)
    np.testing.assert_array_equal(guard.tick(np.zeros(8), 1), np.zeros(8))
    assert guard.state == 'home_error'


def test_tracking_fault_and_timeout():
    guard = Guard(config(), np.zeros(8), 0)
    for i in range(1, 1000):
        guard.tick(np.zeros(8), i*.004)
    assert guard.state == 'home_error'
    cfg = config(); cfg['return_to_zero']['timeout'] = .008
    guard = Guard(cfg, np.zeros(8), 0)
    guard.tick(np.zeros(8), .012)
    assert guard.state == 'home_error'


def test_home_drift_blocks_start():
    guard, now = ready()
    measured = guard.command.copy(); measured[0] += .05
    assert not guard.start(measured, now)
    assert guard.state == 'home_error'


def test_policy_jump_and_action_timeout():
    guard, now = ready(); guard.start(guard.command, now)
    with pytest.raises(ValueError):
        guard.action(np.ones(8), now)
    for _ in range(2600):
        now += .004
        guard.tick(guard.command, now)
    assert guard.state == 'home_error'


@pytest.mark.parametrize('key', ['timeout','watchdog'])
def test_bad_configuration(key):
    cfg = config(); cfg[key] = float('nan')
    with pytest.raises(ValueError):
        Guard(cfg, cfg['zero'], 0)


def test_outside_startup_envelope_rejected_with_joint_detail():
    initial = np.zeros(8); initial[3] = -.251
    with pytest.raises(ValueError, match='J4=-0.251000 rad outside'):
        Guard(config(), initial, 0)


def test_recovery_tracking_fault_holds_and_cannot_restart():
    initial = np.zeros(8); initial[3] = -.195506
    guard = Guard(config(), initial, 0)
    for k in range(1, 201):
        guard.tick(initial, k * .01)  # Jammed joint.
    assert guard.state == 'home_error'
    assert guard.phase == 0
    assert 'command tracking error: J4' in guard.error
    held = guard.command.copy()
    guard.tick(initial, 2.01)
    guard.action(np.zeros(8), 2.02)
    assert not guard.start(initial, 2.02)
    np.testing.assert_array_equal(guard.command, held)


def test_zero_feedback_must_enter_normal_bounds_before_home():
    initial = np.zeros(8); initial[3] = -.01
    guard = Guard(config(), initial, 0)
    for k in range(1, 301):
        guard.tick(initial, k * .01)
    assert guard.phase == 0  # Within 0.06, but still below HOME's J4 bound.
    assert guard.recovery_scope
    np.testing.assert_allclose(guard.command, 0, atol=1e-12)
    for k in range(301, 362):
        guard.tick(np.zeros(8), k * .01)
    assert guard.phase == 1
    assert not guard.recovery_scope


def test_recovery_envelope_cannot_be_used_by_act():
    guard, now = ready()
    assert guard.start(guard.command, now)
    action = guard.command.copy(); action[3] = -.1
    with pytest.raises(ValueError, match='ACT target: joint limit exceeded'):
        guard.action(action, now)
    assert not guard.recovery_scope


def test_zero_gate_waits_for_dispatched_endpoint():
    initial = np.zeros(8); initial[3] = -.001
    guard = Guard(config(), initial, 0)
    for k in range(1, 301):
        guard.tick(np.zeros(8), k * .01)
        # A driver rate clamp prevents the proposed zero command reaching CAN.
        guard.command = initial.copy()
    assert guard.state == 'returning_zero'
    assert guard.phase == 0
    assert guard.recovery_scope
    assert guard.settled is None
    guard.command = np.zeros(8)
    for k in range(301, 362):
        guard.tick(np.zeros(8), k * .01)
    assert guard.phase == 1


def test_verify_home_only_requires_measured_home_and_settle():
    cfg = config()
    with pytest.raises(ValueError, match='physical HOME absent'):
        Guard(cfg, np.zeros(8), 0, verify_home_only=True)
    initial = np.asarray(cfg['home']) + .01
    guard = Guard(cfg, initial, 0, verify_home_only=True)
    assert not guard.start(initial, 0)
    assert not guard.recovery_scope
    for k in range(1, 62):
        np.testing.assert_array_equal(guard.tick(initial, k * .01), initial)
    assert guard.state == 'home_ready'
    assert guard.start(initial, .62)


@pytest.mark.parametrize('key,value', [('steps', 0), ('steps', 2.5),
                                     ('period', float('nan')), ('timeout', -1)])
def test_invalid_recovery_configuration(key, value):
    cfg = config(); cfg['return_to_zero'][key] = value
    with pytest.raises(ValueError):
        Guard(cfg, cfg['zero'], 0)
