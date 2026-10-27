"""Operator-invoked experiment stages. Import/prepare never operate hardware.

Default prepare is closed-transit 30K; H1 uses collection's closed zero startup.
Dora observes until explicit START. Offline replay is not required. STOP ends
the ACT producer, never disables the ROS motor backend. No automatic HOME.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

from scripts.act_trial.preflight import (
    DEFAULT_BUNDLE, DEFAULT_CHECKPOINT, DEFAULT_VISUAL32_CHECKPOINT,
    run as offline_preflight,
)
from scripts.act_trial.profiles import execution_profile, profile_from_config, collision_checks_enabled
from scripts.act_trial.motion_generator import (
    MODES as MOTION_MODES, attach as attach_motion, from_config as motion_from_config,
    selected_name as selected_motion_name, check_live_host,
)
from scripts.act_trial.control_runtime import control_session, control_operation, runtime_root
from scripts.control_ownership.client import Client
from scripts.control_ownership.security import environment
from scripts.home_replay.core import NAMES, write_json
from scripts.act_trial.action_queue import contract as action_queue_contract

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'official/dora-openarm-evaluation/.venv/bin/python'
GRAPH = ROOT / 'official/dora-openarm-evaluation/dataflow-act-trial-ros-host.yaml'
VISUAL32_GRAPH = ROOT / 'official/dora-openarm-evaluation/dataflow-act-visual32-ros-host.yaml'
ARM_AXES = list(range(7)) + list(range(8, 15))


def check_environment():
    expected = environment(os.environ['OPENARM_KEYSTORE'], 117)
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('secure ROS environment changed; use ./scripts/actctl')


def session():
    run = Path(os.environ['OWNERSHIP_RUN']).resolve()
    trial = Path(os.environ['ACT_TRIAL_DIR']).resolve()
    if trial != run/'act' or not trial.is_dir():
        raise ValueError('ACT trial missing; use ./scripts/actctl prepare --profile visual32')
    check_environment()
    return run, trial


def show_gateway_status():
    try:
        with Client() as client:
            print(json.dumps(client.request('status'), indent=2))
    except (FileNotFoundError, ConnectionRefusedError):
        raise SystemExit('Gateway is not running or its socket is unavailable. '
                         'No controller was started or stopped. Check the H1/gateway terminals.') from None


def startup_report_path(run):
    """A report belongs to one H1 invocation; never overwrite previous evidence."""
    parent = run/'h1_startups'
    parent.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    directory = Path(tempfile.mkdtemp(prefix=stamp+'_', dir=parent))
    return directory/'motor_startup.json'


def prepare(profile='visual32', checkpoint=None, motion_generator=None):
    from scripts.act_trial.motion_generator import STREAM_MODE
    motion_generator = motion_generator or (STREAM_MODE if profile == 'visual32' else 'legacy')
    if profile == 'visual32' and motion_generator != STREAM_MODE:
        raise ValueError('closed-transit ACT requires position_stream_v1')
    default_checkpoint = DEFAULT_VISUAL32_CHECKPOINT if profile == 'visual32' else DEFAULT_CHECKPOINT
    checkpoint = Path(default_checkpoint if checkpoint is None else checkpoint).resolve()
    report, config = offline_preflight(DEFAULT_BUNDLE, checkpoint, None, profile=profile)
    if not report['offline_checks_passed']:
        raise ValueError('offline policy/robot contract failed')
    attach_motion(config, report, motion_generator)
    keys = ROOT/'.control_ownership_keys/physical117'
    environment(keys, 117)  # Never replace keys or weaken security.
    run = Path(tempfile.mkdtemp(prefix='act_manual_', dir=ROOT/'results'))
    trial = run/'act'; trial.mkdir()
    write_json(trial/'offline_preflight.json', report)
    write_json(trial/'trial_config.json', config)
    exports = {'OWNERSHIP_RUN': str(run), 'ACT_TRIAL_DIR': str(trial),
               'OPENARM_KEYSTORE': str(keys),
               'OPENARM_AUTHORITY_SOCKET': f'/run/user/{os.getuid()}/openarm-control.sock',
               'OPENARM_PRETRAINED_PATH': str(checkpoint),
               'SOCKET': f'/run/user/{os.getuid()}/act-{uuid.uuid4().hex[:12]}.sock'}
    lines = ['# Source in each experiment terminal; no motor I/O.',
             f'cd {shlex.quote(str(ROOT))} || return',
             'source scripts/home_replay/workstation_env.sh || return',
             'source .control_ownership_overlay/install/setup.bash || return']
    lines += [f'export {k}={shlex.quote(v)}' for k,v in exports.items()]
    lines += ['security_exports=$(/usr/bin/python3 -m scripts.control_ownership.security env '
              '--keystore "$OPENARM_KEYSTORE" --domain 117) || return',
              'eval "$security_exports"', 'set -o pipefail',
              f'actctl() {{ {shlex.quote(str(PYTHON))} -m scripts.act_trial.manual_experiment "$@"; }}']
    content = '\n'.join(lines)+'\n'
    (run/'environment.sh').write_text(content)
    (ROOT/'results/act_manual_current.env').write_text(content)
    print(json.dumps({'prepared': str(run), 'profile':profile, 'motion_generator':motion_generator,
                      'hardware_io_performed': False,
                      'physical_execution_authorized': False,
                      'conversion_review_required_before_start': False,
                      'physical_validation_passed': False, 'checkpoint': str(checkpoint),
                      'next': ('./scripts/actctl policy after ZERO_AND_HANDOFF_VERIFIED'
                               if profile == 'visual32' else
                               './scripts/actctl policy after HOME_AND_HANDOFF_VERIFIED')}, indent=2))


def check_pose(snapshot, phase, bundle):
    """Use H2's 0.06rad start bound and 0.03rad final HOME acceptance.

    H1 command/feedback proof covers both arms; grippers are not zero targets.
    H2 checks recorded right HOME including gripper; left stays monitored.
    """
    q = np.asarray(snapshot['positions_rad'], dtype=float)
    cmd = np.asarray(snapshot['command_positions_rad'], dtype=float)
    velocity = np.asarray(snapshot['velocity_rad_s'], dtype=float)
    if any(v.shape != (16,) or not np.all(np.isfinite(v)) for v in (q, cmd, velocity)):
        raise ValueError('invalid complete CAN state/command vectors')
    if snapshot['motor_status'] != [1]*16:
        raise ValueError('motor disabled or faulted')
    if any(not 0 <= snapshot[k] <= .1 for k in ('max_age_s', 'command_max_age_s')):
        raise ValueError('stale CAN feedback/commands')
    config = bundle['home_config']
    tolerance = float(config['final_acceptance']['joint_tolerance_rad'])
    if not np.isfinite(tolerance) or not 0 < tolerance <= .03:
        raise ValueError('unexpected ROS H2 acceptance contract')
    if phase == 'h1':
        start_tolerance = float(config['maximum_start_from_zero_error_rad'])
        if not np.isfinite(start_tolerance) or not 0 < start_tolerance <= .06:
            raise ValueError('unexpected ROS H2 starting-pose acceptance contract')
        axes = ARM_AXES; expected = np.zeros(14); tolerances = np.full(14, start_tolerance)
    elif phase == 'visual32_zero':
        start_tolerance = .06
        if (np.any(np.abs(q[ARM_AXES]) > start_tolerance)
                or np.any(np.abs(cmd[ARM_AXES]) > start_tolerance + 25/65535)):
            raise ValueError('visual32 arms are not at mechanical zero')
        from scripts.act_trial.closed_contract import closed_start
        try:
            closed_start(q[7], cmd[7])
        except ValueError as exc:
            raise ValueError('closed-transit gripper start rejected; with empty gripper use '
                             'actctl zero-close-gripper before zero-handoff: '+str(exc)) from exc
        if np.max(np.abs(velocity[ARM_AXES + [7]])) > .05:
            raise ValueError('visual32 start joints have not settled')
        return
    elif phase == 'home':
        axes = list(range(8))
        expected = np.array(config['right_home_joint_positions_rad'] +
                            [config['home_gripper']['open_position_rad']])
        tolerances = np.r_[np.full(7, tolerance),
                           float(config['home_gripper']['position_tolerance_rad'])]
        if not 0 < tolerances[-1] <= .03:
            raise ValueError('unexpected ROS gripper acceptance contract')
    else:
        raise ValueError('unknown pose phase')
    if np.any(np.abs(q[axes]-expected) > tolerances):
        raise ValueError(f'{phase} measured pose not accepted: '+str((q[axes]-expected).tolist()))
    # Command encoding has at most one 16bit quantization step of error.
    if np.any(np.abs(cmd[axes]-expected) > tolerances + 25/65535):
        raise ValueError(f'{phase} actual motor targets do not match expected pose')
    if np.max(np.abs(velocity[ARM_AXES])) > .05:
        raise ValueError('arms have not settled')


def verify_pose(phase, output, timeout=10.):
    from scripts.act_trial.telemetry import Telemetry
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    telemetry = Telemetry(output.with_suffix('.can.jsonl'))
    deadline = time.monotonic()+timeout; since = None; error = 'waiting for CAN feedback'
    try:
        while time.monotonic() < deadline:
            try:
                snapshot = telemetry.snapshot()
                check_pose(snapshot, phase, bundle)
            except (ValueError, RuntimeError) as exc:
                error = str(exc); since = None
            else:
                if since is None: since = time.monotonic()
                if time.monotonic()-since >= .5:
                    write_json(output, {'passed': True, 'phase': phase, 'receive_only': True,
                                        'joint_names': NAMES, 'snapshot': snapshot, 'wall_ns': time.time_ns()})
                    return
            time.sleep(.02)
        raise ValueError(f'{phase} live pose verification failed: {error}; keep ROS holding if stable; no automatic retry')
    except BaseException as exc:
        write_json(output, {'passed': False, 'phase': phase, 'error': str(exc), 'receive_only': True})
        raise
    finally:
        telemetry.close()


def wait_idle(client, timeout=15., synchronizing=False):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        state = client.request('status')
        if (state['status']['state'] == 'idle' and state['synchronized'] and state['drain_ready']
                and not state['pending'] and not state['hold_pending']
                and not any(state['raw_active'].values())):
            return state
        if (state['status']['state'] == 'fault'
                and not (synchronizing and state['status'].get('reason') == 'manual_hold_synchronization')):
            raise ValueError('gateway fault: '+str(state))
        time.sleep(.1)
    raise ValueError('gateway did not reach verified idle; do not reset automatically')


def h2(run):
    trial_config = Path(os.environ.get('ACT_TRIAL_DIR', ''))/'trial_config.json'
    if trial_config.is_file() and profile_from_config(json.loads(trial_config.read_text()))['name'] == 'visual32':
        raise ValueError('visual32 starts at mechanical zero: use actctl zero-handoff, never legacy H2 HOME')
    if (run/'h2_execution.json').exists() or (run/'h1_actual_pose.json').exists():
        raise ValueError('H2 was already attempted for the current ROS/gateway processes; '
                         'do not repeat an in-progress or interrupted motion. '
                         'After HOME_AND_HANDOFF_VERIFIED continue with policy/Dora. '
                         'Changing ACT log folders does not reset H2.')
    # Do not open passive CAN sockets while the hardware driver is still starting:
    # its ownership admission intentionally rejects pre-existing receivers.
    with Client() as client:
        state = client.request('status')
        required = ('right_joint_trajectory_controller', 'left_joint_trajectory_controller',
                    'right_gripper_controller')
        if state['status']['state'] != 'idle' or any(state['controllers'].get(c) != 'active' for c in required):
            raise ValueError('wait for H1 controller activation and gateway idle before H2')
    # Checks directly received motor frames, not cached ROS joint-state timestamps.
    verify_pose('h1', run/'h1_actual_pose.json')
    with Client() as client:
        state = client.request('status')
        if state['status']['state'] != 'idle':
            raise ValueError('gateway must be idle before initial synchronization')
        if not state['synchronized']:
            client.request('synchronize-hold')
        wait_idle(client, synchronizing=True)
    cmd = ['/usr/bin/python3', '-m', 'scripts.control_ownership.client', 'run', '--owner', 'h2', '--',
           '/usr/bin/python3', '-m', 'scripts.control_ownership.h2',
           '--config', os.environ['H2_CAPTURE_CONFIG'], '--report', str(run/'h2_execution.json'),
           '--allow-real-execution', '--ros-args', '--enclave', '/openarm/h2',
           '-r', '/right_gripper_controller/follow_joint_trajectory:=/authorized/right_gripper_controller/follow_joint_trajectory']
    subprocess.run(cmd, check=True)
    with Client() as client: wait_idle(client)
    verify_pose('home', run/'home_actual_pose.json')
    subprocess.run(['/usr/bin/python3', '-m', 'scripts.control_ownership.client', 'hold-test',
                    '--report', str(run/'home_hold_test.json')], check=True)
    print('HOME_AND_HANDOFF_VERIFIED: start policy and Dora; ACT still requires explicit start.')


def zero_handoff(run):
    """Verify and hand off the recorded visual32 mechanical-zero start pose."""
    if (run/'closed_zero_hold_test.json').exists() or (run/'closed_zero_synchronized_pose.json').exists():
        raise ValueError('visual32 zero handoff was already attempted for these ROS/gateway processes')
    # A failed receive-only acceptance never dispatched a motion. Preserve
    # its evidence and allow an explicit recheck after opening the gripper.
    previous = run/'zero_actual_pose.json'
    if previous.exists():
        failed = json.loads(previous.read_text())
        if failed.get('passed') is not False or failed.get('receive_only') is not True:
            raise ValueError('previous zero acceptance is not a failed receive-only attempt')
        archive = Path(tempfile.mkdtemp(prefix='zero_rejected_', dir=run))
        for name in ('zero_actual_pose.json', 'zero_actual_pose.can.jsonl'):
            if (run/name).exists():
                (run/name).rename(archive/name)
    with Client() as client:
        state = client.request('status')
        required = ('right_joint_trajectory_controller', 'left_joint_trajectory_controller',
                    'right_gripper_controller')
        if state['status']['state'] != 'idle' or any(
                state['controllers'].get(c) != 'active' for c in required):
            raise ValueError('wait for H1 controller activation and gateway idle before zero handoff')
        if (not state.get('supports_dora_arm_velocity') or
                state.get('controller_command_interfaces', {}).get(required[0]) != ['position', 'velocity']):
            raise ValueError('closed-transit handoff requires updated gateway and position+velocity right JTC')
    verify_pose('visual32_zero', run/'zero_actual_pose.json')
    with Client() as client:
        state = client.request('status')
        if state['status']['state'] != 'idle':
            raise ValueError('gateway must be idle before zero handoff synchronization')
        if not state['synchronized']:
            client.request('synchronize-hold')
        wait_idle(client, synchronizing=True)
    verify_pose('visual32_zero', run/'closed_zero_synchronized_pose.json')
    subprocess.run(['/usr/bin/python3', '-m', 'scripts.control_ownership.client', 'hold-test',
                    '--report', str(run/'closed_zero_hold_test.json')], check=True)
    print('ZERO_AND_HANDOFF_VERIFIED: start policy and Dora; ACT still requires explicit start.')


def validate_start(ready, now_ns):
    age = (now_ns-int(ready['monotonic_ns']))/1e9
    if (ready.get('ready_for_start') is not True or ready.get('started') is not False
            or ready.get('stage') != 'holding' or not 0 <= age <= .5):
        raise ValueError('START refused: live bridge is not freshly ready; '
                         f'report_age_s={age:.3f}, ready_for_start={ready.get("ready_for_start")}, '
                         f'started={ready.get("started")}, stage={ready.get("stage")}, '
                         f'action_age_s={ready.get("action_age_s")}, '
                         f'action_receipt_age_s={ready.get("action_receipt_age_s")}; '
                         'inspect ready.json/result.json')


def require_home_handoff(run, profile='commissioning'):
    zero = profile == 'visual32'
    path = run/('closed_zero_hold_test.json' if zero else 'home_hold_test.json')
    label = 'ZERO' if zero else 'HOME'
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f'{label} handoff report is not ready. Wait for terminal 3 to print '
                         f'{label}_AND_HANDOFF_VERIFIED, then retry this software stage. '
                         'Do not restart H1 or repeat an in-progress handoff. Report: '+str(path)) from None
    except json.JSONDecodeError:
        raise SystemExit(f'{label} handoff report is incomplete/invalid. Wait for handoff to finish; '
                         'if it failed, inspect terminal output. Do not bypass this gate. Report: '+str(path)) from None
    if not isinstance(report, dict) or report.get('passed') is not True:
        raise SystemExit(f'{label} handoff test has not passed; inspect terminal output and '+str(path))


def require_fresh_trial(trial):
    """Reject a reused run before spawning any policy/Dora child processes."""
    previous = [name for name in ('start.request', 'started.json', 'stop.request',
                                 'result.json', 'bridge.jsonl', 'bridge_raw_can.jsonl')
                if (trial/name).exists()]
    if previous:
        raise SystemExit('Experiment directory already used: '+', '.join(previous)+'. '
                         'Leave ROS/gateway running; stop old policy/Dora. '
                         'For a stopped motion attempt run: actctl next-trial with the prepared profile. '
                         'For a pre-START failure use: actctl retry-observation. '
                         'Do not delete evidence or rerun H1/H2.')
    try:
        config = json.loads((trial/'trial_config.json').read_text())
        return profile_from_config(config)
    except (OSError, ValueError) as exc:
        raise SystemExit('Prepared experiment profile is missing/invalid: '+str(exc)) from None


def observation_processes(trial, proc=Path('/proc')):
    """Read process metadata only; never signal another process."""
    names = {'logged_policy_server.py', 'queued_policy_client.py', 'ros_bridge.py', 'logged_executor.py',
             'configured_camera.py', 'dora', 'dora-openarm-observer',
             'dora-openarm-local-policy-server'}
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit(): continue
        try:
            args = entry.joinpath('cmdline').read_bytes().split(b'\0')
            if not {Path(a.decode(errors='replace')).name for a in args if a} & names:
                continue
            env = entry.joinpath('environ').read_bytes().split(b'\0')
            if ('ACT_TRIAL_DIR='+str(trial)).encode() in env:
                found.append(int(entry.name))
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError('cannot verify observation process shutdown: '+str(entry)) from exc
    return found


def retry_observation(run, trial, control=None):
    """Archive a failed pre-START attempt; never clear an actual motion attempt."""
    result = json.loads((trial/'result.json').read_text())
    if result.get('started') is not False or any((trial/name).exists() for name in ('start.request', 'started.json')):
        raise ValueError('retry-observation is only for failures BEFORE any START request')
    with (trial/'bridge.jsonl').open() as stream:
        for line in stream:
            if json.loads(line).get('kind') in ('ACT_START', 'ROS_target_published'):
                raise ValueError('motion evidence exists; refusing pre-START recovery')
    config = json.loads((trial/'trial_config.json').read_text())
    profile_name = config.get('execution_profile', {}).get('name', 'commissioning')
    require_home_handoff(run if control is None else control, profile_name)
    processes = observation_processes(trial)
    if processes:
        raise SystemExit('Stop terminal 5 (Dora) and terminal 4 (policy) first; '
                         'leave ROS and gateway running. Observation PIDs: '+str(processes))
    archive = run/('act_failed_before_start_'+str(time.time_ns()))
    trial.rename(archive)
    trial.mkdir()
    for name in ('offline_preflight.json', 'trial_config.json'):
        shutil.copy2(archive/name, trial/name)
    write_json(trial/'recovery.json', {'previous_attempt': str(archive), 'previous_result': result,
                                      'hardware_operations': False, 'automatic_START': False})
    print('PRE_START_ATTEMPT_ARCHIVED: '+str(archive))
    print('Restart policy and Dora with the same environment. Do not repeat H1/H2. Explicit START is still required.')


def next_trial(run, trial, profile='commissioning', control=None):
    """Explicit operator preparation after a motion attempt; never reset/move.

    Retain all old evidence and require already-synchronized idle authority.
    Fresh bridge initialization enforces HOME, gains, CAN and profile checks.
    """
    # Preserve the selected backend and record current implementation metadata
    # below after idle/preflight checks. Historical hashes are not admission gates.
    previous_config = json.loads((trial/'trial_config.json').read_text())
    require_home_handoff(run if control is None else control, profile)
    motion_generator = selected_motion_name(previous_config)
    check_live_host(motion_generator)
    result = json.loads((trial/'result.json').read_text())
    if result.get('started') is not True:
        raise ValueError('next-trial requires a stopped motion attempt; use retry-observation for pre-START failures')
    processes = observation_processes(trial)
    if processes:
        raise SystemExit('Stop terminal 5 (Dora) and terminal 4 (policy) first; '
                         'leave ROS and gateway running. Observation PIDs: '+str(processes))
    with Client() as client:
        state = wait_idle(client, timeout=2.)  # Read-only, no reset/acquire/hold.
        requested_profile = execution_profile(profile)
        if (requested_profile['command_gripper']
                and 'right_arm_gripper' not in state.get('supported_dora_scopes', [])):
            raise ValueError(profile+' requires updated gateway; restart gateway and manually synchronize hold first')
        checkpoint = Path(os.environ.get('OPENARM_PRETRAINED_PATH', str(DEFAULT_CHECKPOINT))).resolve()
        report, config = (offline_preflight(DEFAULT_BUNDLE, checkpoint, None) if profile == 'commissioning'
                          else offline_preflight(DEFAULT_BUNDLE, checkpoint, None, profile=profile))
        if not report['offline_checks_passed']:
            raise ValueError('current offline preflight failed')
        attach_motion(config, report, motion_generator)
        state = wait_idle(client, timeout=2.)
    archive = run/('act_completed_attempt_'+str(time.time_ns()))
    trial.rename(archive)
    trial.mkdir()
    write_json(trial/'offline_preflight.json', report)
    write_json(trial/'trial_config.json', config)
    write_json(trial/'recovery.json', {'previous_attempt':str(archive), 'previous_result':result,
                                     'gateway_status':state, 'hardware_operations':False,
                                     'automatic_START':False, 'fresh_live_HOME_checks_required':True,
                                     'motion_contract_refreshed': previous_config.get('motion_generator', {'name':'legacy'})
                                         != config['motion_generator']})
    print('PREVIOUS_MOTION_ATTEMPT_ARCHIVED: '+str(archive))
    print('Restart policy and Dora. Fresh live checks and explicit START are required; no automatic HOME.')


def prepare_policy_attempt(run, trial, control):
    """Rotate a stopped attempt on explicit policy startup; never move/reset."""
    # Retain the selected backend, then archive completed attempts for the next run.
    check_live_host(selected_motion_name(json.loads((trial/'trial_config.json').read_text())))
    prepared_profile = profile_from_config(json.loads((trial/'trial_config.json').read_text()))
    require_home_handoff(control, prepared_profile['name'])
    if observation_processes(trial):
        raise SystemExit('Stop old policy/Dora first; keep H1/gateway running.')
    if Path(os.environ['SOCKET']).exists():
        raise ValueError('policy socket already exists; refuse duplicate server')
    if not (trial/'result.json').exists():
        return require_fresh_trial(trial)  # Refuse incomplete/crashed attempts.
    profile = profile_from_config(json.loads((trial/'trial_config.json').read_text()))
    result = json.loads((trial/'result.json').read_text())
    if result.get('started') is True:
        next_trial(run, trial, profile=profile['name'], control=control)
    elif result.get('started') is False:
        with Client() as client:
            wait_idle(client, timeout=2.)
        retry_observation(run, trial, control=control)
    else:
        raise ValueError('Invalid previous result; refusing automatic archive')
    print('POLICY_ATTEMPT_PREPARED: previous evidence retained; no HOME or START requested.', flush=True)
    return require_fresh_trial(trial)


def stop_trial_observers(trial, timeout=15.):
    """SIGINT only verified current-trial observers, never a process group.

    pidfds pin identity across exit/PID reuse. No SIGKILL or socket removal.
    """
    for pid in observation_processes(trial):
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            if pid not in observation_processes(trial):
                continue
            status = Path(f'/proc/{pid}/status').read_text()
            if any(line.startswith('State:') and ('T' in line or 't (tracing' in line)
                   for line in status.splitlines()):
                raise ValueError(f'Observer {pid} is suspended; use fg then Ctrl+C in its terminal')
            print(f'HOME cleanup: SIGINT current-trial observer PID {pid}', flush=True)
            signal.pidfd_send_signal(fd, signal.SIGINT)
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)
    deadline = time.monotonic()+timeout
    while observation_processes(trial):
        if time.monotonic() >= deadline:
            raise ValueError('Observers did not exit; refusing HOME. Stop policy/Dora manually; keep H1/gateway.')
        time.sleep(.1)


def wait_home_recovery_ready(client, timeout=5.):
    """Read-only wait for drain evidence; never relax gateway admission."""
    deadline = time.monotonic()+timeout
    while True:
        state = client.request('status')
        status = state['status']
        if status['state'] == 'idle':
            return state
        if (status['state'] != 'fault' or
                status.get('fault') != 'collision in right command proposal sweep'):
            raise ValueError('HOME shortcut refuses this state/fault: '+json.dumps(state))
        if (state.get('drain_ready') and not state.get('pending')
                and not any(state.get('raw_active', {}).values())
                and not state.get('hold_pending') and state.get('synchronized')):
            return state
        if time.monotonic() >= deadline:
            raise ValueError('HOME drain checks did not pass within timeout; no reset/motion: '
                             +json.dumps(state))
        time.sleep(.1)


def home_cycle(run, trial, control, confirm=input):
    """Attended stop/cleanup/plan/execute; never auto-restart ACT."""
    if confirm('确认夹爪空载、无实体碰撞，允许停止本轮推理并恢复 HOME；输入 HOME：') != 'HOME':
        return
    # A newly prepared model session may be used only to recover the robot
    # from the preceding trial. Do not poison that unused session with STOP.
    if observation_processes(trial) or any(
            (trial/name).exists() for name in
            ('start.request', 'started.json', 'ready.json', 'bridge.jsonl',
             'bridge_raw_can.jsonl', 'result.json')):
        (trial/'stop.request').touch()
    stop_trial_observers(trial)
    with Client() as client:
        state = wait_home_recovery_ready(client)
        if state['status']['state'] == 'fault':
            # Only this known proposal rejection is eligible for this shortcut.
            # Other faults require diagnosis, not automatic acknowledgement.
            client.request('reset-fault')
        wait_idle(client, timeout=5.)
    recover_home(run, trial, control=control)
    if confirm('检查上方目录的 plan.json 及实际回程，确认安全后输入 HOME 执行：') != 'HOME':
        print('Plan retained; no HOME motion requested.')
        return
    recover_home(run, trial, execute=True, control=control)


def recover_home(run, trial, execute=False, control=None):
    """Operator-only recovery; plan first, then execute that exact checked plan."""
    if observation_processes(trial):
        raise ValueError('Stop old Dora/policy terminals first; keep ROS/gateway holding')
    run = run if control is None else control
    with Client() as client: wait_idle(client, timeout=2.)
    pointer=run/'return_home_current.path'
    if execute:
        directory=Path(pointer.read_text().strip()).resolve()
        if directory.parent!=run or not directory.name.startswith('return_home_'):
            raise ValueError('invalid return HOME directory')
        if (directory/'execution.json').exists():
            raise ValueError('this return plan was already attempted; no automatic retry')
        plan=json.loads((directory/'plan.json').read_text())
        if not plan.get('success') or plan.get('physical_motion_requested'):
            raise ValueError('a successful plan-only report is required')
    else:
        directory=Path(tempfile.mkdtemp(prefix='return_home_',dir=run))
    command=[str(PYTHON),'-m','scripts.act_trial.return_home','--directory',str(directory)]
    if execute:
        command+=['--execute']
    command+=['--ros-args','--enclave','/openarm/h2','-r',
              '/right_gripper_controller/follow_joint_trajectory:=/authorized/right_gripper_controller/follow_joint_trajectory']
    if execute:
        command=['/usr/bin/python3','-m','scripts.control_ownership.client','run','--owner','h2','--',*command]
    print(('RETURN_HOME: explicit ROS motion requested' if execute else 'PLAN_HOME: no motion requested')+
          '; report directory: '+str(directory),flush=True)
    subprocess.run(command,check=True)
    if not execute:
        pointer.write_text(str(directory)+'\n')
        print('Plan saved. Inspect plan.json, then manually run: actctl return-home');return
    with Client() as client: wait_idle(client)
    verify_pose('home',directory/'home_actual_pose.json')
    subprocess.run(['/usr/bin/python3','-m','scripts.control_ownership.client','hold-test',
                    '--report',str(directory/'home_hold_test.json')],check=True)
    if (run/'home_hold_test.json').exists():
        shutil.copy2(run/'home_hold_test.json',directory/'previous_home_hold_test.json')
    shutil.copy2(directory/'home_hold_test.json',run/'home_hold_test.json')
    print('HOME_AND_HANDOFF_VERIFIED: actctl next-trial --profile task; then policy/Dora/ready/start.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'h1', 'h2', 'zero-handoff', 'zero-open-gripper', 'zero-close-gripper', 'gateway', 'policy', 'prepare-policy', 'dora', 'ready', 'start', 'stop', 'status', 'retry-observation', 'next-trial', 'plan-home', 'return-home', 'home'])
    parser.add_argument('--profile', choices=['commissioning','task','visual32'], default=None,
                        help='prepare/next-trial: visual32 uses the 640x480 mechanical-zero task')
    parser.add_argument('--checkpoint', type=Path, help='prepare only: ACT pretrained_model directory')
    parser.add_argument('--motion-generator', choices=MOTION_MODES, default=None,
                        help='prepare only: closed-transit uses position_stream_v1; legacy profiles default legacy')
    args = parser.parse_args()
    if args.checkpoint is not None and args.stage != 'prepare':
        parser.error('--checkpoint is only allowed with prepare')
    if args.motion_generator is not None and args.stage != 'prepare':
        parser.error('--motion-generator is only allowed with prepare')
    if args.profile is not None and args.stage not in ('prepare','next-trial'):
        parser.error('--profile is only allowed with prepare or next-trial')
    if args.stage == 'prepare':
        prepare(profile=args.profile or 'visual32', checkpoint=args.checkpoint,
                motion_generator=args.motion_generator)
        return
    check_environment()
    if args.stage == 'h1':
        command = ['/usr/bin/python3', '-m', 'scripts.act_trial.h1',
                   '--keystore', os.environ['OPENARM_KEYSTORE'], '--physical']
    elif args.stage == 'gateway':
        # Keep each gateway lifetime's audit log immutable. The gateway still
        # opens with 'x' and enforces its domain lock/socket ownership checks.
        run = runtime_root()
        log = run/'gateway.jsonl'
        if log.exists() or log.is_symlink():
            log = run/f'gateway_restart_{time.time_ns()}_{uuid.uuid4().hex}.jsonl'
        print(f'GATEWAY_LOG: {log}', flush=True)
        command = ['/usr/bin/python3', '-m', 'scripts.control_ownership.gateway',
                   '--socket', os.environ['OPENARM_AUTHORITY_SOCKET'], '--log', str(log),
                   '--ros-args', '--enclave', '/openarm/gateway']
    elif args.stage == 'h2':
        control = control_session()
        with control_operation(control):
            h2(control)
        return
    else:
        command = None
    if command is not None:
        os.execv(command[0], command)
        return
    if args.stage == 'status' and 'ACT_TRIAL_DIR' not in os.environ:
        show_gateway_status()
        print('ACT trial not prepared; H1/H2 do not require an ACT log directory.')
        return
    run, trial = session()
    if args.stage in ('zero-handoff', 'zero-open-gripper', 'zero-close-gripper'):
        profile = profile_from_config(json.loads((trial/'trial_config.json').read_text()))
        if profile['name'] != 'visual32':
            raise ValueError('zero-handoff is only valid for a prepared visual32 trial')
        control = control_session()
        with control_operation(control):
            if args.stage == 'zero-open-gripper':
                raise ValueError('closed-transit ACT starts closed; use zero-close-gripper with an empty gripper')
            if args.stage == 'zero-close-gripper':
                require_fresh_trial(trial)
                if (control/'closed_zero_hold_test.json').exists():
                    raise ValueError('zero handoff already completed; refusing extra setup motion')
                from scripts.act_trial.zero_gripper import open_at_zero
                open_at_zero(Path(tempfile.mkdtemp(prefix='zero_gripper_', dir=control)), trial, target=-.02)
            else:
                zero_handoff(control)
        return
    if args.stage == 'prepare-policy':
        control = control_session()
        with control_operation(control):
            prepare_policy_attempt(run, trial, control)
        return
    if args.stage in ('plan-home','return-home','home'):
        if profile_from_config(json.loads((trial/'trial_config.json').read_text()))['name'] == 'visual32':
            raise ValueError('closed-transit starts/ends at mechanical zero; legacy lifted HOME recovery is not permitted')
        control = control_session()
        with control_operation(control):
            if args.stage == 'home':
                home_cycle(run, trial, control)
            else:
                recover_home(run,trial,execute=args.stage=='return-home',control=control)
        return
    if args.stage == 'retry-observation':
        retry_observation(run, trial, control=control_session()); return
    if args.stage == 'next-trial':
        current = profile_from_config(json.loads((trial/'trial_config.json').read_text()))['name']
        next_trial(run, trial, profile=args.profile or current, control=control_session()); return
    if args.stage in ('policy', 'dora'):
        motion_generator = motion_from_config(json.loads((trial/'trial_config.json').read_text()))
        check_live_host(motion_generator)
        profile = require_fresh_trial(trial)
        require_home_handoff(control_session(), profile['name'])
        print(f'EXPERIMENT: profile={profile["name"]}, maximum_duration_s={profile["duration_s"]}, '
              f'motion_generator={motion_generator}, '
              f'collision_checks_enabled={collision_checks_enabled(profile["name"])}, '
              f'action_schedule={action_queue_contract(profile["name"])["name"]}, '
              f'prediction=50x30Hz, execution={"rolling continuous q/qd" if profile["name"] == "visual32" else "10-step prefix"}', flush=True)
        if args.stage == 'policy':
            if Path(os.environ['SOCKET']).exists():
                raise ValueError('policy socket already exists; refuse duplicate server')
            command = [str(ROOT/'scripts/act_trial/run_node.sh'), 'policy-server']
        else:
            if not Path(os.environ['SOCKET']).is_socket():
                raise ValueError('wait for policy server Listening message')
            os.environ['ROS_SECURITY_ENCLAVE_OVERRIDE'] = '/openarm/observer'
            graph = VISUAL32_GRAPH if profile['name'] == 'visual32' else GRAPH
            command = [str(PYTHON.parent/'dora'), 'run', str(graph)]
            os.chdir(graph.parent)
    elif args.stage == 'ready':
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if (trial/'result.json').exists():
                raise ValueError('trial stopped before START: '+(trial/'result.json').read_text())
            if (trial/'ready.json').exists():
                try:
                    validate_start(json.loads((trial/'ready.json').read_text()), time.monotonic_ns())
                except ValueError:
                    pass
                else:
                    live = json.loads((trial/'ready.json').read_text())
                    print(f'READY: profile={live.get("profile")}, maximum_duration_s={live.get("duration_s")}; '
                          'holding, fresh observations/actions and live checks passed; actctl start requests motion.')
                    return
            time.sleep(.1)
        raise ValueError('Dora not ready within 45s; inspect ready.json, result.json and dora.log; no START sent')
    elif args.stage == 'start':
        profile_from_config(json.loads((trial/'trial_config.json').read_text()))
        motion_generator = motion_from_config(json.loads((trial/'trial_config.json').read_text()))
        check_live_host(motion_generator)
        if (trial/'start.request').exists() or (trial/'stop.request').exists() or (trial/'result.json').exists():
            raise ValueError('trial already requested/stopped; no automatic physical retry')
        ready = json.loads((trial/'ready.json').read_text())
        validate_start(ready, time.monotonic_ns())
        profile = profile_from_config(json.loads((trial/'trial_config.json').read_text()))
        if ready.get('profile') != profile['name'] or ready.get('duration_s') != profile['duration_s']:
            raise ValueError('live bridge profile differs from prepared experiment')
        if ready.get('motion_generator', 'legacy') != motion_generator:
            raise ValueError('live bridge motion generator differs from prepared experiment')
        with Client() as client: wait_idle(client, timeout=1.)
        validate_start(json.loads((trial/'ready.json').read_text()), time.monotonic_ns())
        write_json(trial/'start.request', {'created_monotonic_ns': time.monotonic_ns(),
                                         'operator_command': 'actctl start', 'profile':profile['name'],
                                         'duration_s':profile['duration_s']})
        duration_text = ('no automatic duration stop; manual stop/fault ends the run'
                         if profile['duration_s'] is None else f'up to {profile["duration_s"]:g} seconds')
        print(f'START requested: profile={profile["name"]}, {duration_text}, '
              +('right J1-J8' if profile['command_gripper'] else 'right J1-J7')+'; ROS remains enabled.')
        print('Check actctl status for START acknowledgement and result; request alone is not execution proof.', flush=True)
        return
    elif args.stage == 'stop':
        (trial/'stop.request').touch(exist_ok=True)
        print('ACT stop requested; leave ROS and gateway running. This does not stop a separate H2 process.')
        return
    else:
        show_gateway_status()
        for name in ('ready.json', 'result.json'):
            path = trial/name
            print(name+': '+(path.read_text() if path.exists() else 'not created'))
        return
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
