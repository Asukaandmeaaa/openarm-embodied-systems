"""Offline commissioning admission regression; no CAN/ROS processes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.act_trial import manual_experiment as manual
from scripts.control_ownership import secure_launch


def bundle():
    return {'home_config': {'final_acceptance': {'joint_tolerance_rad': .03},
                            'maximum_start_from_zero_error_rad': .06,
                            'right_home_joint_positions_rad': [0., 0., 0., 1.37, 0., -.2, -.12],
                            'home_gripper': {'open_position_rad': -.35, 'position_tolerance_rad': .03}}}


def snapshot():
    q = [0.]*16; q[7] = q[15] = -.35
    return {'positions_rad': q, 'command_positions_rad': q.copy(),
            'velocity_rad_s': [0.]*16, 'motor_status': [1]*16,
            'max_age_s': .01, 'command_max_age_s': .01}


def test_h1_checks_both_arms_without_commanding_grippers():
    value = snapshot()
    manual.check_pose(value, 'h1', bundle())
    for axis in manual.ARM_AXES:
        bad = deepcopy(value); bad['positions_rad'][axis] = .061
        with pytest.raises(ValueError, match='measured pose'):
            manual.check_pose(bad, 'h1', bundle())
    # The previously validated H2 starting envelope is0.06, not HOME's0.03.
    value['positions_rad'][3] = .045
    manual.check_pose(value, 'h1', bundle())


def test_command_feedback_and_freshness_are_independent():
    for key, value in [('command_positions_rad', [.1]*16),
                       ('positions_rad', [float('nan')]*16),
                       ('positions_rad', [0.]*15), ('motor_status', [0]*16),
                       ('max_age_s', .101), ('command_max_age_s', .101),
                       ('max_age_s', -.01), ('velocity_rad_s', [.051]*16)]:
        state = snapshot(); state[key] = value
        with pytest.raises(ValueError): manual.check_pose(state, 'h1', bundle())


def test_home_includes_right_gripper_and_preserves_left_observation():
    state = snapshot()
    home = bundle()['home_config']['right_home_joint_positions_rad'] + [-.35]
    state['positions_rad'][:8] = home; state['command_positions_rad'][:8] = home
    state['positions_rad'][11] = -.045
    manual.check_pose(state, 'home', bundle())
    state['positions_rad'][7] = 0.
    with pytest.raises(ValueError, match='measured pose'):
        manual.check_pose(state, 'home', bundle())


@pytest.mark.parametrize('changes', [{'ready_for_start':False}, {'started':True},
                                    {'stage':'fault'}, {'monotonic_ns':1},
                                    {'monotonic_ns':3_000_000_000}])
def test_stale_or_stopped_start_refused(changes):
    ready = {'ready_for_start':True, 'started':False, 'stage':'holding', 'monotonic_ns':1_900_000_000}
    manual.validate_start(ready, 2_000_000_000)
    ready.update(changes)
    with pytest.raises(ValueError): manual.validate_start(ready, 2_000_000_000)


def test_initial_synchronization_waits_for_ack_but_other_faults_fail(monkeypatch):
    fault = {'status':{'state':'fault', 'reason':'manual_hold_synchronization'}, 'synchronized':False}
    ready = {'status':{'state':'idle'}, 'synchronized':True, 'drain_ready':True,
             'pending':[], 'hold_pending':[], 'raw_active':{'arm':[]}}
    states=iter([fault, ready])
    client=SimpleNamespace(request=lambda op: next(states))
    monkeypatch.setattr(manual.time, 'sleep', lambda _: None)
    assert manual.wait_idle(client, synchronizing=True) == ready
    bad=SimpleNamespace(request=lambda op: {'status':{'state':'fault', 'reason':'hold rejected'}})
    with pytest.raises(ValueError, match='gateway fault'):
        manual.wait_idle(bad, synchronizing=True)


def test_pre_activation_probe_failure_does_not_continue_to_other_arm(monkeypatch, tmp_path):
    from scripts import probe_openarm_can
    calls=[]
    def failed(interface, ids):
        calls.append(interface)
        return {'ok':False, 'error':'ENOBUFS'}
    monkeypatch.setattr(probe_openarm_can, 'probe', failed)
    output=tmp_path/'report.json'
    with pytest.raises(ValueError, match='before enable'):
        secure_launch.verify_motor_responses(output)
    assert calls == ['can0']
    assert '"passed": false' in output.read_text()
    assert '"motor_enable_requested": false' in output.read_text()


def test_pre_activation_checks_every_motor_and_current_queue(monkeypatch, tmp_path):
    from scripts import probe_openarm_can
    calls=[]
    def success(interface, ids):
        assert ids == list(range(1,9)); calls.append(interface)
        return {'ok':True}
    monkeypatch.setattr(probe_openarm_can, 'probe', success)
    monkeypatch.setattr(secure_launch, 'verify_empty_can_tx_queues', lambda: calls.append('queue'))
    secure_launch.verify_motor_responses(tmp_path/'report.json')
    assert calls == ['can0','can1','queue']


def test_h2_never_opens_observer_before_backend_ready(monkeypatch, tmp_path):
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, op): return {'status':{'state':'idle'}, 'controllers':{}}
    monkeypatch.setattr(manual, 'Client', FakeClient)
    monkeypatch.setattr(manual, 'verify_pose', lambda *_: pytest.fail('observer opened before startup completed'))
    with pytest.raises(ValueError, match='wait for H1'):
        manual.h2(tmp_path)


def test_full_h2_stage_orders_verified_start_plan_home_and_handoff(monkeypatch, tmp_path):
    events=[]
    state={'status':{'state':'idle'}, 'synchronized':False,
           'controllers':{c:'active' for c in ('right_joint_trajectory_controller',
                        'left_joint_trajectory_controller','right_gripper_controller')}}
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, op):
            events.append(op)
            return state
    monkeypatch.setattr(manual, 'Client', FakeClient)
    monkeypatch.setattr(manual, 'verify_pose', lambda phase, *_: events.append(phase))
    monkeypatch.setattr(manual, 'wait_idle', lambda *a, **k: events.append('verified_idle'))
    monkeypatch.setenv('H2_CAPTURE_CONFIG', '/test/validated_home.yaml')
    def execute(command, **kwargs):
        assert kwargs == {'check':True}
        if 'run' in command:
            assert command[:8] == ['/usr/bin/python3','-m','scripts.control_ownership.client',
                                   'run','--owner','h2','--','/usr/bin/python3']
            assert '--allow-real-execution' in command
            events.append('original_H2')
        else:
            assert 'hold-test' in command
            events.append('hold_test')
    monkeypatch.setattr(manual.subprocess, 'run', execute)
    manual.h2(tmp_path)
    assert events == ['status','h1','status','synchronize-hold','verified_idle',
                      'original_H2','verified_idle','home','hold_test']


def test_h2_pose_failure_never_dispatches_a_plan(monkeypatch, tmp_path):
    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, op):
            return {'status':{'state':'idle'},'controllers':{c:'active' for c in (
                'right_joint_trajectory_controller','left_joint_trajectory_controller','right_gripper_controller')}}
    monkeypatch.setattr(manual, 'Client', FakeClient)
    def fail(*args): raise ValueError('stale CAN feedback')
    monkeypatch.setattr(manual, 'verify_pose', fail)
    monkeypatch.setattr(manual.subprocess, 'run', lambda *a, **k: pytest.fail('H2 motion dispatched'))
    with pytest.raises(ValueError, match='stale CAN'):
        manual.h2(tmp_path)


def test_missing_handoff_report_explains_stage_order(tmp_path):
    with pytest.raises(SystemExit, match='Wait for terminal 3'):
        manual.require_home_handoff(tmp_path)


@pytest.mark.parametrize('content', ['{', '{}', '{"passed":false}', '{"passed":"true"}', '[]'])
def test_incomplete_or_failed_handoff_is_not_admitted(tmp_path, content):
    (tmp_path/'home_hold_test.json').write_text(content)
    with pytest.raises(SystemExit): manual.require_home_handoff(tmp_path)


def test_completed_handoff_allows_software_stage(tmp_path):
    (tmp_path/'home_hold_test.json').write_text('{"passed":true}')
    manual.require_home_handoff(tmp_path)


def failed_observation(tmp_path):
    import json
    trial=tmp_path/'act'; trial.mkdir()
    (tmp_path/'home_hold_test.json').write_text('{"passed":true}')
    (trial/'result.json').write_text(json.dumps({'started':False,'reason':'controller status stale'}))
    (trial/'bridge.jsonl').write_text('{"kind":"terminal"}\n')
    for name in ('offline_preflight.json','trial_config.json'): (trial/name).write_text('{}')
    return trial


def test_prestart_recovery_archives_all_evidence_without_hardware(monkeypatch,tmp_path):
    trial=failed_observation(tmp_path)
    monkeypatch.setattr(manual,'observation_processes',lambda _: [])
    manual.retry_observation(tmp_path,trial)
    archives=list(tmp_path.glob('act_failed_before_start_*'))
    assert len(archives)==1 and (archives[0]/'result.json').exists()
    assert (archives[0]/'bridge.jsonl').exists()
    assert not (trial/'result.json').exists()
    assert not (trial/'start.request').exists()
    assert (trial/'recovery.json').exists()


@pytest.mark.parametrize('marker',['start.request','started.json'])
def test_start_request_is_never_cleared_as_observation_retry(tmp_path,marker):
    trial=failed_observation(tmp_path); (trial/marker).write_text('{}')
    with pytest.raises(ValueError,match='BEFORE any START'): manual.retry_observation(tmp_path,trial)
    assert (trial/marker).exists()


def test_running_observer_prevents_archiving(monkeypatch,tmp_path):
    trial=failed_observation(tmp_path)
    monkeypatch.setattr(manual,'observation_processes',lambda _: [123])
    with pytest.raises(SystemExit,match='Stop terminal 5'): manual.retry_observation(tmp_path,trial)
    assert (trial/'bridge.jsonl').exists()


def test_dispatched_motion_prevents_observation_retry(tmp_path):
    trial=failed_observation(tmp_path)
    (trial/'bridge.jsonl').write_text('{"kind":"ROS_target_published"}\n')
    with pytest.raises(ValueError,match='motion evidence'): manual.retry_observation(tmp_path,trial)


def test_observation_process_scan_does_not_stop_or_block_ros_backend(tmp_path):
    trial=tmp_path/'act'
    for pid,name in [(10,'ros2_control_node'),(11,'ros_bridge.py'),(12,'logged_policy_server.py')]:
        p=tmp_path/str(pid); p.mkdir()
        (p/'cmdline').write_bytes(name.encode()+b'\0')
        (p/'environ').write_bytes(('ACT_TRIAL_DIR='+str(trial)).encode()+b'\0')
    assert sorted(manual.observation_processes(trial,tmp_path))==[11,12]


def test_gateway_restarts_preserve_logs_and_ownership_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(manual, 'session', lambda: (tmp_path, tmp_path/'act'))
    monkeypatch.setattr(manual, 'check_environment', lambda: None)
    monkeypatch.setattr(manual, 'runtime_root', lambda: tmp_path)
    monkeypatch.setattr(manual.sys, 'argv', ['actctl', 'gateway'])
    monkeypatch.setenv('OPENARM_AUTHORITY_SOCKET', '/test/ownership.sock')
    commands = []
    monkeypatch.setattr(manual.os, 'execv', lambda executable, args: commands.append(args))
    logs = []
    for _ in range(3):
        manual.main()
        command = commands[-1]
        assert command[:3] == ['/usr/bin/python3', '-m', 'scripts.control_ownership.gateway']
        assert command[command.index('--socket')+1] == '/test/ownership.sock'
        assert command[-3:] == ['--ros-args', '--enclave', '/openarm/gateway']
        log = manual.Path(command[command.index('--log')+1])
        assert not log.exists()
        log.write_text('original audit evidence\n')
        logs.append(log)
    assert logs[0] == tmp_path/'gateway.jsonl'
    assert len(set(logs)) == 3
    assert all(log.read_text() == 'original audit evidence\n' for log in logs)


def motion_attempt(monkeypatch, tmp_path, state):
    trial = failed_observation(tmp_path)
    (trial/'result.json').write_text('{"started":true,"reason":"speed"}')
    (trial/'start.request').write_text('{}')
    (trial/'started.json').write_text('{}')
    monkeypatch.setattr(manual, 'observation_processes', lambda _: [])
    class ReadOnlyClient:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, op):
            assert op == 'status', 'preparation must never reset, acquire or command'
            return state
    monkeypatch.setattr(manual, 'Client', ReadOnlyClient)
    monkeypatch.setattr(manual, 'offline_preflight', lambda *a: (
        {'offline_checks_passed':True}, {'new_contract':True}))
    return trial


def test_next_trial_retains_motion_evidence_and_refreshes_contract(monkeypatch, tmp_path):
    state={'status':{'state':'idle'}, 'synchronized':True, 'drain_ready':True,
           'pending':[], 'hold_pending':[], 'raw_active':{'arm':[]}}
    trial = motion_attempt(monkeypatch, tmp_path, state)
    manual.next_trial(tmp_path, trial)
    archive, = tmp_path.glob('act_completed_attempt_*')
    assert (archive/'start.request').exists() and (archive/'started.json').exists()
    assert (archive/'result.json').exists() and (archive/'bridge.jsonl').exists()
    assert not (trial/'start.request').exists()
    assert 'new_contract' in (trial/'trial_config.json').read_text()


def test_prepare_exports_selected_checkpoint_without_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(manual, 'ROOT', tmp_path)
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(manual, 'environment', lambda *a: {})
    checkpoint=tmp_path/'model_20000'
    seen=[]
    def preflight(bundle, model, state, **kwargs):
        seen.append(model)
        return {'offline_checks_passed':True}, {}
    monkeypatch.setattr(manual, 'offline_preflight', preflight)
    manual.prepare(profile='task', checkpoint=checkpoint)
    assert seen == [checkpoint]
    assert f'OPENARM_PRETRAINED_PATH={checkpoint}' in (tmp_path/'results/act_manual_current.env').read_text()


def test_next_trial_keeps_selected_checkpoint(monkeypatch, tmp_path):
    state={'status':{'state':'idle'}, 'synchronized':True, 'drain_ready':True,
           'pending':[], 'hold_pending':[], 'raw_active':{'arm':[]}}
    trial=motion_attempt(monkeypatch,tmp_path,state)
    checkpoint=tmp_path/'model_30000'
    monkeypatch.setenv('OPENARM_PRETRAINED_PATH',str(checkpoint))
    seen=[]
    def preflight(bundle, model, start):
        seen.append(model)
        return {'offline_checks_passed':True}, {}
    monkeypatch.setattr(manual,'offline_preflight',preflight)
    manual.next_trial(tmp_path,trial)
    assert seen == [checkpoint]


def test_next_trial_never_clears_gateway_fault(monkeypatch, tmp_path):
    trial = motion_attempt(monkeypatch, tmp_path, {'status':{'state':'fault','reason':'speed'}})
    with pytest.raises(ValueError, match='gateway fault'): manual.next_trial(tmp_path, trial)
    assert (trial/'start.request').exists()


def test_next_trial_rejects_running_observers(monkeypatch, tmp_path):
    trial = motion_attempt(monkeypatch, tmp_path, {})
    monkeypatch.setattr(manual, 'observation_processes', lambda _: [123])
    with pytest.raises(SystemExit, match='Stop terminal 5'): manual.next_trial(tmp_path, trial)


def test_next_trial_rejects_failed_offline_preflight(monkeypatch, tmp_path):
    state={'status':{'state':'idle'}, 'synchronized':True, 'drain_ready':True,
           'pending':[], 'hold_pending':[], 'raw_active':{'arm':[]}}
    trial = motion_attempt(monkeypatch, tmp_path, state)
    monkeypatch.setattr(manual, 'offline_preflight', lambda *a: ({'offline_checks_passed':False}, {}))
    with pytest.raises(ValueError, match='offline preflight'): manual.next_trial(tmp_path, trial)
    assert (trial/'start.request').exists()


@pytest.mark.parametrize('supports_task', [False,True])
def test_task_preparation_requires_gateway_support_and_explicit_profile(monkeypatch,tmp_path,supports_task):
    from scripts.act_trial.profiles import profile_from_config
    state={'status':{'state':'idle'}, 'synchronized':True, 'drain_ready':True,
           'pending':[], 'hold_pending':[], 'raw_active':{'arm':[]},
           'supported_dora_scopes':['right_arm_gripper'] if supports_task else []}
    trial=motion_attempt(monkeypatch,tmp_path,state)
    def preflight(*args,**kwargs):
        assert kwargs == {'profile':'task'}
        from scripts.act_trial.preflight import trial_settings
        return {'offline_checks_passed':True},trial_settings('task')
    monkeypatch.setattr(manual,'offline_preflight',preflight)
    if not supports_task:
        with pytest.raises(ValueError,match='updated gateway'):
            manual.next_trial(tmp_path,trial,profile='task')
        assert (trial/'start.request').exists()
    else:
        manual.next_trial(tmp_path,trial,profile='task')
        import json
        assert profile_from_config(json.loads((trial/'trial_config.json').read_text()))['duration_s'] is None
        assert not (trial/'start.request').exists()


@pytest.mark.parametrize('stage',['policy','dora'])
def test_used_attempt_rejected_before_spawning_any_software(monkeypatch,tmp_path,stage):
    trial=failed_observation(tmp_path)
    monkeypatch.setattr(manual,'session',lambda:(tmp_path,trial))
    monkeypatch.setattr(manual,'check_environment',lambda:None)
    monkeypatch.setattr(manual,'control_session',lambda:tmp_path)
    monkeypatch.setattr(manual.sys,'argv',['actctl',stage])
    monkeypatch.setattr(manual.os,'execv',lambda *a:pytest.fail('spawned a used experiment'))
    with pytest.raises(SystemExit,match='Experiment directory already used'): manual.main()
    assert (trial/'result.json').exists() and (trial/'bridge.jsonl').exists()


def test_fresh_task_profile_admission_has_no_hardware_operations(tmp_path):
    import json
    from scripts.act_trial.preflight import trial_settings
    config=tmp_path/'trial_config.json'
    config.write_text(json.dumps(trial_settings('task')))
    assert manual.require_fresh_trial(tmp_path)['duration_s'] is None
    (tmp_path/'stop.request').touch()
    with pytest.raises(SystemExit,match='already used'): manual.require_fresh_trial(tmp_path)


def test_missing_profile_refuses_before_software_launch(tmp_path):
    (tmp_path/'trial_config.json').write_text('{}')
    with pytest.raises(SystemExit,match='missing/invalid'): manual.require_fresh_trial(tmp_path)


def test_new_task_session_prepares_manual_stop_profile_without_motor_or_session_calls(monkeypatch,tmp_path):
    import json
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(manual,'ROOT',tmp_path)
    monkeypatch.setattr(manual,'environment',lambda *a,**k:{})
    seen=[]
    def preflight(*args,profile):
        seen.append(profile)
        return {'offline_checks_passed':True},{'execution_profile':{'name':profile},'duration_s':None}
    monkeypatch.setattr(manual,'offline_preflight',preflight)
    monkeypatch.setattr(manual.subprocess,'run',lambda *a,**k:pytest.fail('prepare must not launch any process'))
    monkeypatch.setattr(manual,'session',lambda:pytest.fail('prepare must not reuse old session'))
    monkeypatch.setattr(manual.sys,'argv',['actctl','prepare','--profile','task'])
    manual.main()
    configs=list((tmp_path/'results').glob('act_manual_*/act/trial_config.json'))
    assert len(configs)==1 and seen==['task']
    assert json.loads(configs[0].read_text())['duration_s'] is None
    assert (tmp_path/'results/act_manual_current.env').exists()
