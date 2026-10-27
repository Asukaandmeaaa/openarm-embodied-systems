"""Commissioning guard decisions only; no motor library or physical commands."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.act_trial import guard as module


class Scene:
    def __init__(self, collision=None):
        self.seen = []
        self.collision = collision or (lambda q: False)

    def put(self, q):
        q = np.array(q, dtype=float).copy()
        self.seen.append(q)
        return [{'distance_m': -.001, 'pair': 'synthetic'}] if self.collision(q) else []


@pytest.fixture
def setup(monkeypatch):
    home = np.array([-.145, -.015, -.006, 1.375, -.003, -.198, -.122, -.35])
    initial = np.r_[home, np.zeros(7), -.019]
    limits = np.tile([-2., 2.], (16, 1))
    limits[3] = [-.02, 2.4435]
    limits[5] = [-.7654, .7654]
    limits[6] = [-1.5508, 1.5508]
    limits[7] = [-.7854, 0]
    limits[15] = [0, .7854]  # Observed left gripper lies outside: not commanded.
    monkeypatch.setattr(module, 'ros_limits', lambda *a, **k: (
        limits.copy(), np.full(16, 5.), {'synthetic': 'test'}, 'synthetic-test-only'))
    bundle = {'home_config': {
        'right_home_joint_positions_rad': home[:7].tolist(),
        'home_gripper': {'open_position_rad': float(home[7]), 'position_tolerance_rad': .03},
        'final_acceptance': {'joint_tolerance_rad': .03},
    }}
    return initial, bundle, limits


def make_guard(setup, scene=None):
    initial, bundle, _ = setup
    return module.Guard(initial, bundle, now=0., scene=scene or Scene())


def start_guard(guard):
    decision = guard.tick(None, guard.initial, .01, .01)
    assert decision['stage'] == 'holding'
    guard.start(.01)


@pytest.mark.parametrize('profile,mode', [('task', 'legacy'), ('task', 'synchronized_v1'),
                                         ('visual32', 'position_stream_v1')])
def test_full_policy_runs_past_sixty_seconds_and_still_stops_on_stale_feedback(setup, profile, mode):
    initial,bundle,_ = setup
    initial = initial.copy()
    if profile == 'visual32':
        initial[:7] = 0.
        initial[7] = -.02
    guard = module.Guard(initial,bundle,now=0,scene=Scene(),profile=profile,
                         motion_generator=mode)
    start_guard(guard)
    request = initial[:8].copy(); request[0] += .20; request[7] = -.10
    measured = initial.copy(); max_motion = 0
    for step in range(1,6002):
        if profile == 'visual32':
            request[0] = min(.20, step*.005)
            request[7] = max(-.10, -.02-step*.002)
        velocity = (request-guard.command)/.01 if profile == 'visual32' else None
        decision = guard.tick(request,measured,.01+step*.01,.005,.03,
                              requested_velocity=velocity)
        assert decision['stage'] == 'active', decision['errors']
        measured[:8] = decision['dispatched_proposal']
        assert guard.envelope_lower[7]-1e-10 <= measured[7] <= guard.envelope_upper[7]+1e-10
        np.testing.assert_array_equal(measured[8:],initial[8:])
        max_motion = max(max_motion,abs(measured[0]-initial[0]))
    assert guard.DURATION is None
    assert max_motion > .19 and measured[7] > -.11
    assert decision['desired'] == request.tolist()
    assert not decision['left_command_emitted']
    decision = guard.tick(request,measured,60.04,.20,.03)
    assert decision['stage']=='fault'
    assert decision['errors']


@pytest.mark.parametrize('mode', ['legacy', 'synchronized_v1'])
def test_task_does_not_construct_or_query_collision_scene(setup, monkeypatch, mode):
    initial,bundle,_ = setup
    def unexpected_scene(*args, **kwargs):
        pytest.fail('task must not construct or query the MuJoCo scene')
    monkeypatch.setattr(module, 'Scene', unexpected_scene)
    guard = module.Guard(initial,bundle,now=0,profile='task',motion_generator=mode)
    guard.scene = type('UnusedScene', (), {'put': unexpected_scene})()
    start_guard(guard); measured = initial.copy()
    request = initial[:8].copy(); request[0] += .2
    for step in range(1,201):
        decision = guard.tick(request,measured,.01+step*.01,.005,.03)
        assert decision['stage'] == 'active', decision['errors']
        measured[:8] = decision['dispatched_proposal']
    assert decision['dispatched_proposal'][0] > initial[0]+.08
    assert decision['collision_checks_enabled'] is False
    assert decision['collision_checked_samples'] == 0
    assert decision['timing']['collision_backend'] == 'disabled'
    assert guard.profile()['collision_checks_enabled'] is False


def test_task_stationary_gripper_feedback_warns_without_tracking_stop(setup):
    initial,bundle,_ = setup
    guard = module.Guard(initial,bundle,now=0,scene=Scene(),profile='task')
    start_guard(guard)
    request = initial[:8].copy(); request[7] = 0.0
    for step in range(1,201):
        decision = guard.tick(request,initial,.01+step*.01,.005,.03)
        assert decision['stage'] == 'active', decision['errors']
    assert decision['tracking_warning_joints'] == [module.NAMES[7]]
    assert decision['tracking_stop_joints'] == []
    assert decision['performance_warnings'] and not decision['errors']
    assert decision['clamped'][7] == -.10
    assert initial[7] + .03 < decision['dispatched_proposal'][7] <= -.10
    assert decision['desired'][7] == 0.0


@pytest.mark.parametrize('axis', range(8))
def test_task_tracking_is_monitor_only_without_changing_prediction(setup, axis):
    initial,bundle,_=setup
    guard=module.Guard(initial,bundle,now=0,scene=Scene(),profile='task')
    start_guard(guard)
    measured=initial.copy();measured[axis]+=.030735
    decision=guard.tick(initial[:8],measured,.02,.005,.03)
    assert decision['stage']=='active'
    measured[axis]=initial[axis]+.040200
    decision=guard.tick(initial[:8],measured,.03,.005,.03)
    assert decision['stage']=='active'
    assert decision['tracking_warning_joints']==[module.NAMES[axis]]
    assert decision['tracking_stop_joints']==[]
    assert decision['tracking_stop_limits_rad']==[None]*8
    assert decision['performance_warnings'] and not decision['errors']
    assert decision['collision_checked_samples'] == 0
    np.testing.assert_array_equal(decision['desired'],initial[:8])
    np.testing.assert_array_equal(decision['dispatched_proposal'],initial[:8])
    json.dumps(decision,allow_nan=False)
    bad=initial.copy();bad[5]+=.030735
    with pytest.raises(ValueError,match='HOME acceptance'):
        module.Guard(bad,bundle,now=0,scene=Scene(),profile='task')


def test_recorded_gripper_stop_offset_is_diagnostic_within_existing_limits(setup):
    initial,bundle,_=setup
    guard=module.Guard(initial,bundle,now=0,scene=Scene(),profile='task')
    start_guard(guard)
    # Seed the ongoing task's reference and feedback from the 2026-10-15 stop.
    # Subtracting this offset from the open endpoint would test the separate
    # measured endpoint envelope, not the recorded tracking fault.
    guard.command[7]=-.31119011808485114
    measured=initial.copy(); measured[7]=-.3419928282597091
    request=guard.command.copy(); request[7]=-.2980842590332031
    decision=guard.tick(request,measured,.02,.027350763,.114991812)
    assert decision['stage']=='active', decision['errors']
    assert decision['tracking_error_rad'][7]==pytest.approx(-.030802710174857972)
    assert decision['tracking_warning_joints']==[module.NAMES[7]]
    assert decision['tracking_stop_joints']==[]
    assert decision['collision_checked_samples']==0
    np.testing.assert_array_equal(decision['desired'],request)


@pytest.mark.parametrize('axis', [5, 7])
def test_task_pre_start_tracking_is_still_blocking(setup, axis):
    initial,bundle,_=setup
    guard=module.Guard(initial,bundle,now=0,scene=Scene(),profile='task')
    measured=initial.copy(); measured[axis]+=.031
    decision=guard.tick(None,measured,.01,.005)
    assert decision['stage']=='fault'
    assert decision['tracking_stop_joints']==[module.NAMES[axis]]
    with pytest.raises(ValueError): guard.start(.01)


@pytest.mark.parametrize('fault', ['limit', 'feedback', 'action', 'nan'])
@pytest.mark.parametrize('axis', [5, 7])
def test_task_diagnostic_tracking_cannot_bypass_other_faults(setup, fault, axis):
    initial,bundle,_=setup
    guard=module.Guard(initial,bundle,now=0,scene=Scene(),profile='task')
    start_guard(guard)
    measured=initial.copy(); measured[axis]+=.05
    feedback_age=.005; action_age=.03
    if fault=='limit': measured[0]=3.
    if fault=='feedback': feedback_age=.101
    if fault=='action': action_age=.501
    if fault=='nan': measured[0]=float('nan')
    decision=guard.tick(initial[:8],measured,.02,feedback_age,action_age)
    assert decision['stage']=='fault' and decision['errors']
    np.testing.assert_array_equal(decision['dispatched_proposal'],initial[:8])


def test_holds_without_zero_home_gripper_or_left_commands(setup):
    guard = make_guard(setup)
    request = np.ones(8)
    decision = guard.tick(request, guard.initial, .01, .01)
    np.testing.assert_array_equal(decision['dispatched_proposal'], guard.initial[:8])
    assert decision['stage'] == 'holding'
    assert not decision['left_command_emitted'] and not decision['dispatch_performed']
    assert len(decision['dispatched_proposal']) == 8
    assert decision['measured'][-1] == -.019


def test_start_needs_fresh_checked_hold_and_cannot_restart(setup):
    guard = make_guard(setup)
    with pytest.raises(ValueError, match='holding tick'):
        guard.start(.001)
    guard.tick(None, guard.initial, .01, .01)
    with pytest.raises(ValueError, match='holding tick'):
        guard.start(.2)
    guard.start(.02)
    with pytest.raises(ValueError, match='cannot restart'):
        guard.start(.03)


def test_model_construction_time_is_not_a_control_tick_gap(setup):
    guard = make_guard(setup)
    # No control loop existed before the first measured holding check.
    decision = guard.tick(None, guard.initial, .5, .01)
    assert decision['stage'] == 'holding'
    guard.start(.5)
    decision = guard.tick(guard.initial[:8], guard.initial, .61, .01, .01)
    assert decision['stage'] == 'fault'  # Subsequent gaps are still strict.


@pytest.mark.parametrize('joint,offset', [(0, .0301), (7, .0301)])
def test_home_acceptance_uses_ros_bounds(setup, joint, offset):
    initial, bundle, _ = setup
    initial[joint] += offset
    with pytest.raises(ValueError, match='HOME acceptance'):
        module.Guard(initial, bundle, now=0, scene=Scene())


def test_clips_execution_only_and_logs_gripper_lock(setup):
    guard = make_guard(setup)
    start_guard(guard)
    request = guard.initial[:8] + .4
    original = request.copy()
    decision = guard.tick(request, guard.initial, .02, .01, .02)
    np.testing.assert_array_equal(request, original)
    np.testing.assert_array_equal(decision['desired'], original)
    np.testing.assert_allclose(decision['clamped'][:7], guard.initial[:7]+.04)
    assert decision['clamped'][7] == guard.initial[7]
    assert decision['dispatched_proposal'][7] == guard.initial[7]
    assert 'command_envelope_clipping' in decision['interventions']
    assert 'gripper_hold_existing_reference' in decision['interventions']
    assert 'acceleration_or_braking_limit' in decision['interventions']
    assert decision['stage'] == 'active'


def test_velocity_acceleration_braking_and_reversal_stay_bounded(setup):
    guard = make_guard(setup)
    start_guard(guard)
    previous_command = guard.command.copy()
    previous_velocity = guard.velocity.copy()
    for tick in range(2, 300):
        direction = 1 if tick < 120 else -1
        request = guard.initial[:8] + direction * .5
        measured = np.r_[guard.command, guard.initial[8:]]
        decision = guard.tick(request, measured, tick*.01, .01, .01)
        assert decision['stage'] == 'active', decision['errors']
        delta = guard.command - previous_command
        assert np.all(np.abs(delta) <= guard.speed*.01 + 1e-10)
        assert np.max(np.abs(guard.velocity-previous_velocity)) <= .25*.01 + 1e-10
        assert np.max(np.abs(guard.command[:7]-guard.initial[:7])) <= .04+1e-10
        previous_command = guard.command.copy()
        previous_velocity = guard.velocity.copy()


def test_duration_latches_hold_and_does_not_restart(setup):
    guard = make_guard(setup)
    start_guard(guard)
    for tick in range(2, 302):
        decision = guard.tick(guard.initial[:8]+.5,
                              np.r_[guard.command, guard.initial[8:]],
                              tick*.01, .01, .01)
    assert decision['stage'] == 'complete'
    command = guard.command.copy()
    decision = guard.tick(guard.initial[:8]-.5, np.r_[command, guard.initial[8:]],
                          3.02, .01, .01)
    assert decision['stage'] == 'complete'
    np.testing.assert_array_equal(guard.command, command)
    with pytest.raises(ValueError, match='cannot restart'):
        guard.start(3.03)


@pytest.mark.parametrize('fault', ['feedback', 'action', 'watchdog', 'tracking', 'nan', 'bad_request'])
def test_faults_latch_without_further_motion(setup, fault):
    guard = make_guard(setup)
    start_guard(guard)
    measured = guard.initial.copy()
    request = guard.initial[:8].copy()
    now, feedback_age, action_age = .02, .01, .01
    if fault == 'feedback':
        feedback_age = .101
    elif fault == 'action':
        action_age = .501
    elif fault == 'watchdog':
        now = .111
    elif fault == 'tracking':
        measured[0] += .0301
    elif fault == 'nan':
        measured[2] = float('nan')
    else:
        request[0] = float('nan')
    decision = guard.tick(request, measured, now, feedback_age, action_age)
    assert decision['stage'] == 'fault'
    first_error = decision['errors']
    np.testing.assert_array_equal(guard.command, guard.initial[:8])
    json.dumps(decision, allow_nan=False)
    decision = guard.tick(guard.initial[:8]+.01, guard.initial, now+.01, .01, .01)
    assert decision['stage'] == 'fault'
    assert decision['errors'] == first_error
    np.testing.assert_array_equal(guard.command, guard.initial[:8])


def test_delayed_tick_has_no_catchup(setup):
    guard = make_guard(setup)
    start_guard(guard)
    decision = guard.tick(guard.initial[:8]+.04, guard.initial, .09, .01, .01)
    assert decision['stage'] == 'active'
    assert decision['slew_period_s'] == .01
    assert 'late_tick_no_catchup' in decision['interventions']
    assert max(abs(guard.command-guard.initial[:8])) <= .25*.01**2+1e-10


def test_collision_full_left_context_and_intermediate_sweep(setup):
    # Neither endpoint collides; the intermediate measured left position does.
    scene = Scene(lambda q: .0029 < q[8] < .0031)
    guard = make_guard(setup, scene)
    start_guard(guard)
    measured = guard.initial.copy()
    measured[8] = .0061
    decision = guard.tick(guard.initial[:8], measured, .02, .01, .01)
    # Four equal intervals at <=.002 include .00305 between clear endpoints.
    assert decision['stage'] == 'fault'
    assert 'full16' in decision['errors'][0]
    assert decision['collision_contacts']
    np.testing.assert_array_equal(guard.command, guard.initial[:8])


def test_proposed_collision_rejected_before_command_changes(setup):
    initial, _, _ = setup
    scene = Scene(lambda q: q[0] > initial[0] + .00001)
    guard = make_guard(setup, scene)
    start_guard(guard)
    decision = guard.tick(initial[:8]+.04, initial, .02, .01, .01)
    assert decision['stage'] == 'fault'
    assert 'proposal' in decision['errors'][0]
    np.testing.assert_array_equal(guard.command, initial[:8])


def test_canonical_right_limits_cannot_be_widened_by_envelope(setup):
    initial, bundle, limits = setup
    limits[0, 1] = initial[0] + .005
    guard = make_guard(setup)
    assert guard.envelope_upper[0] == initial[0]+.005
    start_guard(guard)
    measured = initial.copy()
    measured[0] += .006
    decision = guard.tick(initial[:8], measured, .02, .01, .01)
    assert decision['stage'] == 'fault'
    assert 'canonical' in decision['errors'][0]


def test_no_duplicate_smoothing_and_initial_left_limit_is_diagnostic_only(setup):
    guard = make_guard(setup)
    profile = guard.profile()
    assert 'caller blends/filters' in profile['smoothing']
    assert profile['left_role'] == 'measured_collision_context_no_command'
    assert profile['limit_policy'] == 'synthetic-test-only'
    assert profile['canonical_right_limits_rad'][5] == [-.7654, .7654]
    json.dumps(profile, allow_nan=False)


def test_existing_reference_preserves_stable_pd_offset_without_handoff_jump(setup):
    initial, bundle, _ = setup
    reference = initial[:8].copy()
    reference[3] += .024
    reference[7] -= .006
    original = reference.copy()
    scene = Scene()
    guard = module.Guard(initial, bundle, now=0., scene=scene,
                         initial_command8=reference)
    np.testing.assert_array_equal(reference, original)
    np.testing.assert_array_equal(guard.command, reference)
    np.testing.assert_allclose(guard.envelope_lower[:7], initial[:7]-.04)
    np.testing.assert_allclose(guard.envelope_upper[:7], initial[:7]+.04)
    record = guard.tick(None, initial, .01, .01)
    np.testing.assert_array_equal(record['dispatched_proposal'], reference)
    guard.start(.01)
    record = guard.tick(reference, initial, .02, .01, .01)
    assert record['stage'] == 'active'
    np.testing.assert_array_equal(record['dispatched_proposal'], reference)
    np.testing.assert_array_equal(record['proposal_velocity_rad_s'], np.zeros(8))
    record = guard.tick(initial[:8], initial, .03, .01, .01)
    assert record['dispatched_proposal'][7] == reference[7]
    assert 'gripper_hold_existing_reference' in record['interventions']
    profile = guard.profile()
    assert profile['gripper_policy'] == 'hold_existing_controller_reference'
    np.testing.assert_array_equal(profile['initial_controller_reference_rad'], reference)
    np.testing.assert_allclose(profile['initial_reference_minus_measured_rad'],
                               reference-initial[:8])
    assert any(np.allclose(q, np.r_[reference, initial[8:]]) for q in scene.seen)


@pytest.mark.parametrize('kind', ['tracking', 'home', 'finite', 'collision'])
def test_initial_reference_rejected_when_inconsistent(setup, kind):
    initial, bundle, _ = setup
    reference = initial[:8].copy()
    scene = Scene()
    if kind == 'tracking':
        initial[3] -= .02
        reference[3] += .02
        message = 'tracking allowance'
    elif kind == 'home':
        initial[3] += .02
        reference[3] = initial[3] + .02
        message = 'HOME acceptance'
    elif kind == 'finite':
        reference[0] = float('nan')
        message = 'finite'
    else:
        reference[3] += .024
        scene = Scene(lambda q: q[3] > initial[3] + .01)
        message = 'controller reference'
    with pytest.raises(ValueError, match=message):
        module.Guard(initial, bundle, now=0., scene=scene,
                     initial_command8=reference)
