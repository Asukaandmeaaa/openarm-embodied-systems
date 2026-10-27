#!/usr/bin/env python3
"""Playback-only counterpart of episode_capture_button: B = next recording.

Uses the original recorded_motion_player, its direct FJT execution, validation,
markers, HOME return and SIGINT cancellation. No ACT or MoveIt execution adapter.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/home/lushilin/openarm_vla_ws/openarm_kinesthetic_data/collections/trajectories')
XINPUT_DEVICE = '/dev/input/by-id/usb-Gamepad_Controller-joystick'


def configure_replay_gamepad(pad, config, *, xinput=False):
    """Decode input only; never change shared teaching settings or motor limits."""
    from openarm_kinesthetic.teaching.gamepad import GamepadMapper, BTN_TL
    cfg = deepcopy(config)
    if xinput:
        if pad.name != 'Generic X-Box pad':
            raise ValueError(f'unsupported gamepad identity: {pad.name}')
        cfg['mapping']['axis_codes'].update(right_x='ABS_RX', right_y='ABS_RY',
                                             lt='ABS_Z', rt='ABS_RZ')
    mapper = GamepadMapper(cfg)
    resolved = mapper.configure_axis_codes(pad.axis_codes)
    if xinput and not resolved:
        raise ValueError('X-Box axis map incomplete; RT/LT cannot be identified')
    # Use Linux button semantics for the alternate mode instead of copying
    # the original mode's indices (in particular LB differs).
    for semantic, code in [('episode_capture_b', 0x131), ('hand_drag_lb', BTN_TL),
                           ('stop_rb_r1', 0x137)]:
        index = pad.button_index_for_code(code)
        if index is not None:
            mapper.button[semantic] = index
        elif xinput:
            raise ValueError(f'X-Box button map missing {semantic}')
    return mapper


def open_replay_gamepad():
    from openarm_kinesthetic.teaching.gamepad import LinuxJoystick, load_yaml
    cfg = load_yaml('gamepad_control.yaml')
    pad = LinuxJoystick.open_configured(config=cfg)
    xinput = pad is None
    if xinput:
        try:
            pad = LinuxJoystick(XINPUT_DEVICE)
        except OSError as exc:
            raise RuntimeError(f'gamepad unavailable: original G3V2 not found; {XINPUT_DEVICE}: {exc}') from exc
    try:
        mapper = configure_replay_gamepad(pad, cfg, xinput=xinput)
        pad.poll()  # initial/queued events must never start playback
        if not pad.connected:
            raise RuntimeError('gamepad disconnected while opening')
        return pad, mapper
    except BaseException:
        pad.close()
        raise


def player_command(path, *, plan_only=False):
    # Exactly the existing site manual A7/B5 playback parameters.
    return ['ros2', 'run', 'openarm_kinesthetic_recorder', 'recorded_motion_player',
            '--trajectory', str(path), '--plan-only' if plan_only else '--allow-real-execution',
            '--playback-rate', '1.0', '--keyframe-settle-tolerance-rad', '0.04']


def playlist_paths(manifest, only=None):
    """Explicit immutable selection; metadata binds each file to a capture."""
    entries = json.loads(manifest.read_text())['entries']
    if not entries or (only is not None and not 1 <= only <= len(entries)):
        raise ValueError('Invalid playlist selection')
    chosen = entries if only is None else [entries[only-1]]
    paths = []
    for entry in chosen:
        path = Path(entry['trajectory']).resolve(strict=True)
        metadata = Path(entry['metadata']).resolve(strict=True)
        if hashlib.sha256(metadata.read_bytes()).hexdigest() != entry['metadata_sha256']:
            raise ValueError('Capture metadata changed: '+str(metadata))
        meta = json.loads(metadata.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256'] or digest != meta['trajectory_sha256']:
            raise ValueError('Trajectory differs from captured recording: '+str(path))
        if meta.get('review_status') != 'approved' or meta.get('operator_marked_success') is not True:
            raise ValueError('Capture not approved')
        print(f"PLAYLIST {entry['label']}: {path}", flush=True)
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate playlist files')
    return paths


def pressed(events, index):
    return any(e.kind == 'button' and e.index == index and e.value == 1
               and not e.initial for e in events)


def competing_processes():
    """Local process check, not a replacement for DDS authorization."""
    names = {'scripts.control_ownership.gateway', 'scripts.control_ownership.secure_launch',
             'right_kinesthetic_bridge', 'episode_capture_button', 'recorded_motion_player',
             'home_guarded_executor', 'real_gamepad', 'right_motion_recorder', 'scripts.act_trial.ros_bridge',
             'scripts.expert_replay.button_replay', 'scripts.expert_replay.secure_player'}
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, PermissionError):
            continue
        if any(Path(arg).name in names or Path(arg).name.startswith('dora-openarm') for arg in args):
            found.append((int(proc.name), ' '.join(args)))
    return found


def stop_player(child):
    if child is None or child.poll() is not None:
        return
    # Original player handles SIGINT by cancelling its active ROS actions.
    # Never SIGKILL it: a dead client does not guarantee a stopped controller.
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print('STOP UNCONFIRMED: player did not exit. If motion is unsafe use the physical emergency stop. '
              'Keep H1 running; do not start another player.', flush=True)
        child.wait()


def play_sequence(paths, pad, mapper, log):
    child = None
    b = mapper.button['episode_capture_b']

    def poll():
        events = pad.poll()
        command = mapper.update(pad.axes, pad.buttons, connected=pad.connected)
        if not pad.connected or command.stop_latched:
            raise RuntimeError('gamepad disconnected or RT/RB pressed')
        return events, command

    try:
        for index, path in enumerate(paths, 1):
            poll()  # discard queued B events from the previous replay
            released = not bool(pad.buttons[b])
            print(f'REPLAY_READY [{index}/{len(paths)}]: {path.name}; release LT/LB, press B. NOT ACT.', flush=True)
            while True:
                events, command = poll()
                if released and pressed(events, b) and not command.deadman_pressed and not command.drag_modifier_pressed:
                    break
                released = not bool(pad.buttons[b])
                time.sleep(.02)
            conflicts = competing_processes()
            if conflicts:
                raise RuntimeError(f'other command/input processes still running: {conflicts}')
            command = player_command(path)
            log('start', path=str(path), command=command)
            child = subprocess.Popen(command, start_new_session=True)
            while child.poll() is None:
                poll()
                time.sleep(.02)
            code = child.returncode
            log('exit', path=str(path), returncode=code)
            child = None
            if code != 0:
                raise RuntimeError(f'original player failed ({code}); see its error above; playlist stopped')
            print(f'REPLAY_COMPLETE [{index}/{len(paths)}]; reset objects before next B.', flush=True)
        print('REPLAY_BATCH_COMPLETE; H1 remains running.', flush=True)
    finally:
        stop_player(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-directory', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--playlist', type=Path)
    parser.add_argument('--only', type=int, help='1-based index within explicit playlist')
    parser.add_argument('--plan-only', action='store_true', help='ROS validation only; no joystick or execution')
    parser.add_argument('--list', action='store_true', help='list files only; no ROS or joystick')
    parser.add_argument('--allow-real-execution', action='store_true')
    args = parser.parse_args()
    if args.only is not None and args.playlist is None:
        parser.error('--only requires --playlist')
    paths = (playlist_paths(args.playlist, args.only) if args.playlist else
             sorted(args.trajectory_directory.resolve(strict=True).glob('*.json')))
    if not paths:
        parser.error('no trajectory JSON files')
    if args.list:
        for i, path in enumerate(paths, 1):
            print(f'{i:02d} {path}')
        return 0
    if args.plan_only and args.allow_real_execution:
        parser.error('Choose plan-only OR real execution')
    if not args.allow_real_execution and not args.plan_only:
        parser.error('use --allow-real-execution after manual H1/H2, or --list')
    if os.environ.get('ROS_DOMAIN_ID') != '97' or os.environ.get('ROS_SECURITY_ENABLE', '').lower() == 'true':
        parser.error('use the original ROS domain 97 environment; do not source actctl environment.sh')
    conflicts = competing_processes()
    if conflicts:
        parser.error(f'other command/input processes still running: {conflicts}')
    runtime = Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}'))
    lock = (runtime/'openarm-original-button-replay.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.plan_only:
        try:
            for path in paths:
                subprocess.run(player_command(path, plan_only=True), check=True)
        finally:
            lock.close()
        return 0
    sys.path.insert(0, str(ROOT/'src'))
    pad, mapper = open_replay_gamepad()
    print(f'REPLAY_GAMEPAD: {pad.name}; {pad.path}; '
          f'B={mapper.button["episode_capture_b"]}, LB={mapper.button["hand_drag_lb"]}, '
          f'RB={mapper.button["stop_rb_r1"]}, LT_axis={mapper.axis["lt"]}, RT_axis={mapper.axis["rt"]}', flush=True)
    logs = ROOT/'results'; logs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='button_replay_', dir=logs))
    print(f'REPLAY_LOG: {run}', flush=True)
    (run/'playlist.json').write_text(json.dumps([str(p) for p in paths], indent=2)+'\n')
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (run/'events.jsonl').open('x', buffering=1) as stream:
            def log(kind, **fields):
                stream.write(json.dumps(dict(kind=kind, wall_ns=time.time_ns(),
                                             source='expert_recording_not_ACT', **fields))+'\n')
            try:
                play_sequence(paths, pad, mapper, log)
            except (Exception, KeyboardInterrupt) as exc:
                log('stopped', reason=str(exc) or 'operator interrupt')
                print(f'REPLAY_STOPPED: {exc or "operator interrupt"}; leave H1 running.', flush=True)
                return 2
    finally:
        pad.close()
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
