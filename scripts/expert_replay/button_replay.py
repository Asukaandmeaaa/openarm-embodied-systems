"""Manual B-button expert replay; never ACT inference or automatic startup.

Run only after the operator's secured H1/H2. PLAN ONLY precedes each B wait;
the original player revalidates after B. One press performs one complete replay,
then the next item waits for another press. Stops never advance the playlist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from scripts.control_ownership.client import Client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'openarm_kinesthetic_data/collections/trajectories/20260929_115522_015.json'


def button_pressed(events, index):
    return any(e.kind == 'button' and e.index == index and e.value == 1 and not e.initial
               for e in events)


def idle_status(state):
    return (state['status']['state'] == 'idle' and state['synchronized'] and state['drain_ready']
            and not state['pending'] and not state['hold_pending']
            and not any(state['raw_active'].values()))


def player_command(trajectory, home_report, audit_file, *, plan_only=False):
    return [sys.executable, '-m', 'scripts.expert_replay.secure_player',
            '--audit-file', str(audit_file), '--trajectory', str(trajectory),
            '--home-report', str(home_report), '--playback-rate', '1.0',
            '--plan-only' if plan_only else '--allow-real-execution',
            '--ros-args', '--enclave', '/openarm/h2', '-r',
            '/right_gripper_controller/follow_joint_trajectory:=/authorized/right_gripper_controller/follow_joint_trajectory']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--trajectory', type=Path)
    selection.add_argument('--trajectory-directory', type=Path, default=DEFAULT.parent)
    parser.add_argument('--allow-real-execution', action='store_true', required=True)
    args = parser.parse_args()
    from scripts.act_trial.manual_experiment import session, require_home_handoff, observation_processes
    from openarm_kinesthetic.teaching.gamepad import LinuxJoystick, GamepadMapper, load_yaml, BTN_TL
    from openarm_kinesthetic_recorder.recorded_motion import load_recorded_motion, validate_collection_trajectory
    run, trial = session()
    require_home_handoff(run)
    if observation_processes(trial):
        raise ValueError('stop old policy/Dora before expert replay; keep ROS/gateway holding')
    trajectories = ([args.trajectory.resolve(strict=True)] if args.trajectory else
                    sorted(args.trajectory_directory.resolve(strict=True).glob('*.json')))
    if not trajectories: raise ValueError('no trajectories found')
    playlist = []
    for trajectory in trajectories:
        payload = load_recorded_motion(trajectory)
        validate_collection_trajectory(payload)
        playlist.append({'path':str(trajectory), 'sha256':hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                         'duration_s':payload['duration_s'], 'samples':len(payload['samples'])})
    home_report = run/'h2_execution.json'
    pad = LinuxJoystick.open_configured()
    if pad is None: raise ValueError('configured G3V2 gamepad unavailable')
    mapper = GamepadMapper(load_yaml('gamepad_control.yaml'))
    mapper.configure_axis_codes(pad.axis_codes)
    lb = pad.button_index_for_code(BTN_TL)
    if lb is not None: mapper.button['hand_drag_lb'] = lb
    b_index = mapper.button['episode_capture_b']
    if b_index >= len(pad.buttons): raise ValueError('gamepad B index unavailable')
    out = Path(tempfile.mkdtemp(prefix='expert_replay_', dir=run))
    snapshots = out/'trajectories'; snapshots.mkdir()
    for entry in playlist:
        raw = Path(entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('trajectory changed during preparation')
        snapshot = snapshots/Path(entry['path']).name
        snapshot.write_bytes(raw)
        entry['snapshot'] = str(snapshot)
    stream = (out/'events.jsonl').open('x', buffering=1)
    (out/'playlist.json').write_text(json.dumps(playlist,indent=2)+'\n')
    interrupted = False
    def interrupt(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    def log(kind, **fields):
        stream.write(json.dumps({'kind':kind, 'source':'expert_recording_not_ACT',
                                 'monotonic_ns':time.monotonic_ns(), **fields})+'\n')
    def poll():
        events = pad.poll()
        command = mapper.update(pad.axes, pad.buttons, connected=pad.connected)
        if events: log('gamepad', events=[e.to_dict() for e in events])
        if interrupted: raise RuntimeError('operator Ctrl-C/termination')
        if not pad.connected: raise RuntimeError('gamepad disconnected')
        if command.stop_latched: raise RuntimeError('gamepad RT/RB stop')
        return events, command
    child = None; client = None; result = {'success': False, 'source':'expert_recording_not_ACT',
                                         'playlist':playlist, 'completed':[],
                                         'playback_rate':1., 'motion_requested':False}
    try:
        print(f'EXPERT_REPLAY_LOG: {out}', flush=True)
        with Client() as observer:
            if not idle_status(observer.request('status')):
                raise ValueError('gateway must be idle, synchronized and drained; no automatic reset')
        for index, entry in enumerate(playlist):
            trajectory = Path(entry['snapshot']); result['current_index'] = index
            attempt = out/f'{index+1:03d}_{trajectory.stem}'; attempt.mkdir()
            if hashlib.sha256(trajectory.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('trajectory changed since playlist creation')
            print(f'PLAN_ONLY [{index+1}/{len(playlist)}]: {trajectory.name}; no motion.', flush=True)
            child = subprocess.Popen(player_command(trajectory, home_report, attempt/'plan.jsonl', plan_only=True),
                                     start_new_session=True)
            while child.poll() is None:
                poll(); time.sleep(.02)
            if child.returncode: raise RuntimeError(f'PLAN_ONLY failed: {child.returncode}')
            child = None
            # Discard B presses during earlier motion or planning; no queued start.
            poll()
            released = not bool(pad.buttons[b_index])
            print(f'EXPERT_REPLAY_READY [{index+1}/{len(playlist)}]: {trajectory.name}; reset objects, release LT/LB, press B once. RT/RB stops. NOT ACT.', flush=True)
            while True:
                events, command = poll()
                press = button_pressed(events, b_index)
                if released and press and not command.deadman_pressed and not command.drag_modifier_pressed:
                    break
                released = not bool(pad.buttons[b_index])
                time.sleep(.02)
            if hashlib.sha256(trajectory.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('trajectory changed while waiting')
            client = Client()
            if not idle_status(client.request('status')): raise ValueError('gateway no longer idle')
            client.acquire('h2')  # Existing exclusive validated-trajectory scope.
            env = os.environ.copy(); env['OPENARM_AUTHORITY_TOKEN'] = client.token
            poll()
            print('EXPERT_REPLAY_START: recorded right arm/gripper; rechecking path before motion.', flush=True)
            result['motion_requested'] = True
            child = subprocess.Popen(player_command(trajectory, home_report, attempt/'execution.jsonl'),
                                     env=env, start_new_session=True)
            while child.poll() is None:
                poll(); client.renew(); time.sleep(.05)
            if child.returncode: raise RuntimeError(f'player failed: {child.returncode}; inspect terminal output')
            client.release()
            deadline = time.monotonic()+10
            while not idle_status(client.request('status')):
                if time.monotonic() > deadline: raise RuntimeError('holding acknowledgement timeout')
                time.sleep(.05)
            client.close(); client = None; child = None
            result['completed'].append(trajectory.name)
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'EXPERT_REPLAY_COMPLETE [{index+1}/{len(playlist)}]: ROS holding; next item requires a new B press.', flush=True)
        result['success'] = True
        print('EXPERT_REPLAY_BATCH_COMPLETE: ROS remains holding. No automatic repeat.', flush=True)
        return 0
    except Exception as exc:
        result['reason'] = str(exc)
        if client is not None and client.token is not None:
            try: client.request('freeze', reason='expert replay: '+str(exc))
            except Exception as hold_exc: result['hold_error'] = str(hold_exc)
        print('EXPERT_REPLAY_STOPPED: '+str(exc)+'; keep ROS/gateway running.', flush=True)
        return 2
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=3)
        if client is not None: client.close()
        pad.close()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
