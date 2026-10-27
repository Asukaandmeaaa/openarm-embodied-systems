"""Offline visual32 deployment contracts; never open ROS, CAN, or cameras."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.act_trial.episode_completion import MechanicalZeroCompletion
from scripts.act_trial import guard as guard_module
from scripts.act_trial import manual_experiment as manual
from scripts.act_trial.preflight import audit_checkpoint, inspect_start, trial_settings
from scripts.act_trial.profiles import execution_profile, profile_from_config


def test_visual32_profile_uses_training_geometry_and_gripper_contract():
    profile = execution_profile('visual32')
    assert profile['start_contract'] == 'mechanical_zero'
    assert profile['duration_s'] is None
    assert profile['gripper_limits_rad'] == [-.45, -.02]
    checkpoint = Path(__file__).resolve().parents[1] / (
        'outputs/act_closed_v2_37eps_20261023b_50k/checkpoints/050000/pretrained_model')
    audited = audit_checkpoint(checkpoint, profile='visual32')
    features = audited['config']['input_features']
    assert features['observation.images.ceiling']['shape'] == [3, 480, 640]
    assert features['observation.images.wrist_right']['shape'] == [3, 480, 640]


def test_visual32_old_sixty_second_session_requires_fresh_prepare():
    old_profile = execution_profile('visual32')
    old_profile['duration_s'] = 60.0
    with pytest.raises(ValueError, match='prepare a new trial'):
        profile_from_config({'execution_profile': old_profile})
    assert profile_from_config({'execution_profile': execution_profile('visual32')})[
        'duration_s'] is None


def test_visual32_start_requires_zero_arms_and_closed_gripper():
    q = np.zeros(16); q[7] = -.049
    limits = np.tile([-3., 3.], (16, 1))
    result = inspect_start(q, np.zeros(8), limits, None, trial_settings('visual32'))
    assert result['supplied_start_checks_passed']
    assert result['start_contract'] == 'mechanical_zero'
    q[2] = .061
    assert inspect_start(q, np.zeros(8), limits, None,
                         trial_settings('visual32'))['failures'] == [
                             'right_mechanical_zero_mismatch']
    q[2] = 0; q[7] = -.25
    assert inspect_start(q, np.zeros(8), limits, None,
                         trial_settings('visual32'))['failures'] == [
                             'right_gripper_not_closed']


def test_completion_cannot_fire_at_start_and_requires_settled_return():
    detector = MechanicalZeroCompletion(
        execution_profile('visual32')['mechanical_zero_completion'])
    zero = np.zeros(16); zero[7] = -.02
    assert not detector.update(zero, np.zeros(16), 10.6, 10.)['complete']
    departed = zero.copy(); departed[0] = .13
    assert detector.update(departed, np.zeros(16), 11., 10.)['departed']
    assert not detector.update(zero, np.full(16, .051), 13., 10.)['complete']
    assert not detector.update(zero, np.zeros(16), 13.1, 10., command8=zero[:8])['complete']
    result = detector.update(zero, np.zeros(16), 13.61, 10., command8=zero[:8])
    assert result['complete'] and result['at_mechanical_zero'] and result['settled']


def test_completion_requires_parked_left_arm_to_remain_at_zero():
    detector = MechanicalZeroCompletion(
        execution_profile('visual32')['mechanical_zero_completion'])
    zero = np.zeros(16); zero[7] = -.02
    departed = zero.copy(); departed[0] = .13
    detector.update(departed, np.zeros(16), 1., 0.)
    left_drift = zero.copy(); left_drift[10] = .061
    result = detector.update(left_drift, np.zeros(16), 4., 0.)
    assert not result['at_mechanical_zero'] and not result['complete']


def test_visual32_guard_accepts_zero_not_legacy_home(monkeypatch):
    limits = np.tile([-2., 2.], (16, 1)); limits[7] = [-.7854, 0]
    monkeypatch.setattr(guard_module, 'ros_limits', lambda *a, **k: (
        limits, np.full(16, 5.), {'synthetic': 'test'}, 'synthetic-test-only'))
    bundle = {'home_config': {
        'right_home_joint_positions_rad': [-.145, -.015, -.006, 1.375, -.003, -.198, -.122],
        'home_gripper': {'open_position_rad': -.35, 'position_tolerance_rad': .03},
        'final_acceptance': {'joint_tolerance_rad': .03},
    }}
    initial = np.zeros(16); initial[7] = -.02
    guard = guard_module.Guard(initial, bundle, now=0., profile='visual32')
    assert guard.start_contract == 'mechanical_zero'
    assert guard.envelope_lower[7] == pytest.approx(-.45)
    assert guard.envelope_upper[7] == pytest.approx(-.02)
    bad = initial.copy(); bad[0] = .061
    with pytest.raises(ValueError, match='mechanical_zero'):
        guard_module.Guard(bad, bundle, now=0., profile='visual32')


def test_zero_pose_includes_both_arms_and_right_gripper():
    bundle = {'home_config': {
        'final_acceptance': {'joint_tolerance_rad': .03},
        'maximum_start_from_zero_error_rad': .06,
        'right_home_joint_positions_rad': [0.] * 7,
        'home_gripper': {'open_position_rad': -.35, 'position_tolerance_rad': .03},
    }}
    q = [0.] * 16; q[7] = -.02
    snapshot = {'positions_rad': q, 'command_positions_rad': q.copy(),
                'velocity_rad_s': [0.] * 16, 'motor_status': [1] * 16,
                'max_age_s': .01, 'command_max_age_s': .01}
    manual.check_pose(snapshot, 'visual32_zero', bundle)
    snapshot['positions_rad'][7] = -.25
    with pytest.raises(ValueError, match='gripper'):
        manual.check_pose(snapshot, 'visual32_zero', bundle)


def test_visual32_graph_is_native_640x480():
    graph = Path(__file__).resolve().parents[1] / (
        'official/dora-openarm-evaluation/dataflow-act-visual32-ros-host.yaml')
    nodes = {node['id']: node for node in yaml.safe_load(graph.read_text())['nodes']}
    for name in ('camera-wrist-right', 'camera-ceiling'):
        assert nodes[name]['env']['IMAGE_WIDTH'] == '640'
        assert nodes[name]['env']['IMAGE_HEIGHT'] == '480'
        assert nodes[name]['env']['CAPTURE_WIDTH'] == '640'
        assert nodes[name]['env']['CAPTURE_HEIGHT'] == '480'


def test_zero_handoff_verifies_synchronizes_reverifies_and_holds(monkeypatch, tmp_path):
    events = []
    state = {'status': {'state': 'idle'}, 'synchronized': False,
             'supports_dora_arm_velocity': True,
             'controller_command_interfaces': {'right_joint_trajectory_controller': ['position', 'velocity']},
             'controllers': {name: 'active' for name in (
                 'right_joint_trajectory_controller', 'left_joint_trajectory_controller',
                 'right_gripper_controller')}}

    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, op):
            events.append(op)
            return state

    monkeypatch.setattr(manual, 'Client', FakeClient)
    monkeypatch.setattr(manual, 'verify_pose', lambda phase, *_: events.append(phase))
    monkeypatch.setattr(manual, 'wait_idle', lambda *a, **k: events.append('verified_idle'))
    monkeypatch.setattr(manual.subprocess, 'run',
                        lambda command, check: events.append('hold_test'))
    manual.zero_handoff(tmp_path)
    assert events == ['status', 'visual32_zero', 'status', 'synchronize-hold',
                      'verified_idle', 'visual32_zero', 'hold_test']


def test_visual32_handoff_marker_is_distinct_from_legacy_home(tmp_path):
    (tmp_path/'home_hold_test.json').write_text('{"passed":true}')
    with pytest.raises(SystemExit, match='ZERO handoff'):
        manual.require_home_handoff(tmp_path, 'visual32')
    (tmp_path/'closed_zero_hold_test.json').write_text('{"passed":true}')
    manual.require_home_handoff(tmp_path, 'visual32')
