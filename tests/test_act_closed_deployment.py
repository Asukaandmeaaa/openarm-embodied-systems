"""Closed-transit regression tests. No device access or ROS initialization."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.act_trial import closed_contract as contract
from scripts.act_trial.episode_completion import MechanicalZeroCompletion
from scripts.act_trial.profiles import execution_profile
from scripts.act_trial.preflight import audit_checkpoint
from scripts.act_trial.h1 import act_node
from scripts.control_ownership import gateway
from test_control_ownership import scoped_dora_fixture


def test_closed_start_does_not_confuse_command_with_contact_feedback():
    contract.closed_start(-.0494, -.02)
    for q, command in [(-.40, -.02), (-.0494, -.085), (float('nan'), -.02), (.01, -.02)]:
        with pytest.raises(ValueError):
            contract.closed_start(q, command)


def test_legacy_zero_marker_and_model_are_not_new_evidence():
    with pytest.raises(ValueError, match='30K or 50K'):
        audit_checkpoint(contract.ROOT/'outputs/act_visual32_20k_20261022/checkpoints/020000/pretrained_model',
                         profile='visual32')
    contract.verify_live_model(contract.model_text())
    with pytest.raises(ValueError):
        contract.verify_live_model(contract.model_text().replace('startup_gripper_closed', 'changed_startup'))


def test_closed_completion_requires_departure_and_command_closure():
    detector = MechanicalZeroCompletion(execution_profile('visual32')['mechanical_zero_completion'])
    q = np.zeros(16); q[7] = -.0494
    command = np.zeros(8); command[7] = -.02
    for t in (1., 5., 10.):
        assert not detector.update(q, np.zeros(16), t, 0., command)['complete']
    q[0] = .13
    detector.update(q, np.zeros(16), 11., 0., command)
    q[0] = 0.
    opened = command.copy(); opened[7] = -.45
    assert not detector.update(q, np.zeros(16), 12., 0., opened)['complete']
    assert not detector.update(q, np.zeros(16), 13., 0., command)['complete']
    assert detector.update(q, np.zeros(16), 13.6, 0., command)['complete']


def test_pv_gateway_requires_seven_finite_velocities_before_any_publish(monkeypatch):
    gate, token = scoped_dora_fixture(monkeypatch, 'right_arm_gripper')
    gate.controller_command_interfaces[gateway.CONTROLLERS[0]] = ['position', 'velocity']
    sent = []
    gate.publish = lambda c, q, velocities=None: sent.append((c, q, velocities))
    request = dict(op='command', token=token, positions=[0.]*7+[-.02])
    for velocity in (None, [0.]*8, [float('nan')]*7, [True]*7):
        with pytest.raises(ValueError):
            gate.request('dora-test', dict(request, velocities=velocity))
        assert sent == []
    velocity = [.1]*7
    gate.request('dora-test', dict(request, velocities=velocity))
    assert sent == [(gateway.CONTROLLERS[0], [0.]*7, velocity),
                    (gateway.CONTROLLERS[1], [-.02], None)]


def test_pv_hold_publish_explicitly_zeros_velocity(monkeypatch):
    class Point:
        def __init__(self): self.velocities = []
    class Trajectory:
        pass
    monkeypatch.setitem(sys.modules, 'trajectory_msgs.msg',
                        SimpleNamespace(JointTrajectory=Trajectory, JointTrajectoryPoint=Point))
    gate = gateway.Gateway.__new__(gateway.Gateway)
    controller = gateway.CONTROLLERS[0]
    gate.controller_command_interfaces = {controller: ['position', 'velocity']}
    sent = []
    gate.publishers = {controller: SimpleNamespace(publish=sent.append)}
    gate.publish(controller, [.2]*7, velocities=[.1]*7)
    gate.publish(controller, [.2]*7)
    assert sent[0].points[0].velocities == [.1]*7
    assert sent[1].points[0].velocities == [0.]*7
    assert sent[1].points[0].positions == [.2]*7


def test_streaming_jtc_config_is_separate_from_collection():
    node = act_node(dict(package='controller_manager', executable='spawner',
                         arguments=['right_joint_trajectory_controller']))
    path = node['arguments'][-1]
    assert path.endswith('scripts/act_trial/position_velocity.yaml')
    params = yaml.safe_load(open(path))['right_joint_trajectory_controller']['ros__parameters']
    assert params['command_interfaces'] == ['position', 'velocity']
    assert params['allow_nonzero_velocity_at_trajectory_end'] is True
    assert params['cmd_timeout'] > params['constraints']['goal_time']


def test_operator_requested_stream_does_not_require_or_claim_replay_validation():
    import inspect
    from scripts.act_trial import manual_experiment, ros_bridge
    settings = contract.execution_contract()
    assert settings['offline_replay_required'] is False
    assert settings['physical_validation_passed'] is False
    assert settings['execution_mode'] == 'position_stream_v1'
    for module in (manual_experiment, ros_bridge):
        assert 'require_conversion_review' not in inspect.getsource(module)


@pytest.fixture
def stream_guard(monkeypatch):
    from scripts.act_trial import guard as module
    bounds = np.tile([-2., 2.], (16, 1)); bounds[7] = [-.7854, 0.]
    source = lambda *a, **k: (bounds.copy(), np.full(16, 2.), {'synthetic': 'test'}, 'synthetic')
    monkeypatch.setattr(module, 'ros_limits', source)
    monkeypatch.setattr(contract, 'limits', source)
    bundle = {'home_config': {
        'right_home_joint_positions_rad': [0.]*7,
        'home_gripper': {'open_position_rad': -.35, 'position_tolerance_rad': .03},
        'final_acceptance': {'joint_tolerance_rad': .03},
    }}
    initial = np.zeros(16); initial[7] = -.02
    guard = module.Guard(initial, bundle, now=0., profile='visual32')
    assert guard.tick(None, initial, .01, .001)['stage'] == 'holding'
    guard.start(.01)
    return guard


def test_stream_preserves_targets_above_old_caps_and_does_not_stop_at_points(stream_guard):
    guard = stream_guard
    measured = guard.initial.copy()
    assert guard.generator is None and guard.ACCELERATION is None
    assert guard.profile()['speed_rad_s'] == [2.]*8
    assert guard.measured_speed_contract['limits_rad_s'] == [2.]*8
    previous = guard.command.copy()
    # 0.5 -> 1.0 rad/s: exceeds both old 0.25 limits, within canonical speed.
    for t, x, v in ((.02, .005, .5), (.03, .015, 1.0), (.04, .025, 1.0)):
        target = guard.initial[:8].copy(); target[0] = x
        velocity = np.zeros(8); velocity[0] = v
        decision = guard.tick(target, measured, t, .001, .02, requested_velocity=velocity)
        assert decision['stage'] == 'active', decision['errors']
        np.testing.assert_array_equal(decision['dispatched_proposal'], target)
        assert decision['proposal_velocity_rad_s'][0] == pytest.approx(v)
        np.testing.assert_allclose(target-previous, np.array(decision['proposal_velocity_rad_s'])*.01)
        assert not {'velocity_limit', 'acceleration_or_braking_limit'} & set(decision['interventions'])
        measured[:8] = target
        previous = target


def test_stream_uses_actual_elapsed_time_not_old_capped_slew_period(stream_guard):
    target = stream_guard.command.copy(); target[0] += .015
    velocity = np.zeros(8); velocity[0] = .5
    decision = stream_guard.tick(target, stream_guard.initial, .04, .001, .02,
                                 requested_velocity=velocity)
    assert decision['stage'] == 'active'
    assert decision['proposal_velocity_rad_s'][0] == pytest.approx(.5)
    assert 'late_tick_no_catchup' not in decision['interventions']


def test_stream_forwards_curve_derivative_instead_of_differentiating_ipc(stream_guard):
    target=stream_guard.command.copy();target[0]+=.005
    velocity=np.zeros(8);velocity[0]=.8
    result=stream_guard.tick(target,stream_guard.initial,.02,.001,.02,
                             requested_velocity=velocity)
    assert result['stage']=='active'
    assert result['proposal_velocity_rad_s'][0]==.8  # not delta-q/.01 = .5


def test_stream_rejects_missing_curve_velocity(stream_guard):
    result=stream_guard.tick(stream_guard.command,stream_guard.initial,.02,.001,.02)
    assert result['stage']=='fault'
    assert 'rolling trajectory velocity' in result['errors'][0]


@pytest.mark.parametrize('failure', ['jump', 'nan', 'stale_feedback', 'stale_action'])
def test_stream_faults_hold_last_position_and_zero_velocity(stream_guard, failure):
    target = stream_guard.command.copy()
    initial_command = target.copy()
    target[0] += .03 if failure == 'jump' else .005
    if failure == 'nan': target[0] = np.nan
    decision = stream_guard.tick(target, stream_guard.initial, .02,
                                .2 if failure == 'stale_feedback' else .001,
                                .6 if failure == 'stale_action' else .02,
                                requested_velocity=np.zeros(8))
    assert decision['stage'] == 'fault'
    np.testing.assert_array_equal(decision['dispatched_proposal'], initial_command)
    assert decision['proposal_velocity_rad_s'] == [0.]*8


def test_old_saved_execution_profile_is_rejected():
    from scripts.act_trial.profiles import profile_from_config
    profile = execution_profile('visual32')
    profile['deployment_contract'].pop('execution_mode')
    with pytest.raises(ValueError, match='prepare a new trial'):
        profile_from_config({'execution_profile': profile})


def test_q_qd_pair_has_continuous_trajectory_consistency():
    from scripts.act_trial.motion_generator import SynchronizedGenerator
    generator = SynchronizedGenerator(np.full(8, .25), .25, np.full(8, -2.), np.full(8, 2.), True)
    q, v, target = np.zeros(8), np.zeros(8), np.full(8, .2)
    for _ in range(20):
        next_q, next_v, _, dt = generator.propose(q, v, target, .01)
        assert np.allclose(next_q-q, (v+next_v)*dt/2, atol=1e-9)
        q, v = next_q, next_v


def test_prepare_defaults_to_30k_stream_without_replay_or_hardware(monkeypatch, tmp_path, capsys):
    from scripts.act_trial import manual_experiment as manual
    monkeypatch.setattr(manual, 'ROOT', tmp_path)
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(manual, 'environment', lambda *a: {})
    selected = []
    def preflight(bundle, checkpoint, state, **kwargs):
        selected.append(checkpoint)
        return {'offline_checks_passed': True}, {}
    monkeypatch.setattr(manual, 'offline_preflight', preflight)
    manual.prepare()
    assert selected == [contract.CHECKPOINTS[30000]]
    report = json.loads(capsys.readouterr().out)
    assert report['motion_generator'] == 'position_stream_v1'
    assert report['conversion_review_required_before_start'] is False
    assert report['hardware_io_performed'] is False
    assert report['physical_validation_passed'] is False
    config_path, = (tmp_path/'results').glob('act_manual_*/act/trial_config.json')
    assert json.loads(config_path.read_text())['motion_generator']['name'] == 'position_stream_v1'


def test_stream_does_not_load_legacy_ruckig(monkeypatch):
    from scripts.act_trial import motion_generator as motion
    monkeypatch.setattr(motion, 'load_ruckig', lambda: pytest.fail('stream must not load Ruckig'))
    assert motion.from_config({'motion_generator': motion.contract(motion.STREAM_MODE)}) == motion.STREAM_MODE


def test_stream_latched_fault_cannot_resume_on_a_valid_target(stream_guard):
    bad = stream_guard.command.copy(); bad[0] = np.nan
    assert stream_guard.tick(bad, stream_guard.initial, .02, .001, .02)['stage'] == 'fault'
    good = stream_guard.command.copy(); good[0] += .005
    stopped = stream_guard.tick(good, stream_guard.initial, .03, .001, .02)
    assert stopped['stage'] == 'fault'
    assert stopped['proposal_velocity_rad_s'] == [0.]*8
    assert stopped['dispatched_proposal'][0] == 0.
