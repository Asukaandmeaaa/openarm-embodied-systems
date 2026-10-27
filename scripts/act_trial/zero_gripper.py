"""Explicit operator gripper opening at mechanical zero via the existing gateway."""
import json
import time

import numpy as np

from scripts.control_ownership.client import Client
from scripts.home_replay.core import write_json


def opening_reference(initial8, elapsed_s, target=-.40):
    """Ten-second quintic: <=0.094 rad/s and <=0.029 rad/s2 over 0.5rad."""
    q = np.asarray(initial8, dtype=float).copy()
    if q.shape != (8,) or not np.all(np.isfinite(q)) or not -.5 <= q[7] <= .05:
        raise ValueError('invalid gripper opening reference')
    if not np.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError('invalid gripper opening time')
    s = min(elapsed_s / 10., 1.)
    blend = 10*s**3 - 15*s**4 + 6*s**5
    if target not in (-.40, -.02):
        raise ValueError('unreviewed empty-gripper setup target')
    q[7] += (target - q[7])*blend
    return q


def check_opening_feedback(snapshot, initial8, bundle):
    from scripts.act_trial.manual_experiment import check_pose
    check_pose(snapshot, 'h1', bundle)
    q = np.asarray(snapshot['positions_rad'])
    cmd = np.asarray(snapshot['command_positions_rad'])
    if np.max(np.abs(q[:7] - initial8[:7])) > .03:
        raise ValueError('arm holding tracking lost during gripper opening')
    if np.max(np.abs(cmd[:7] - initial8[:7])) > .003:
        raise ValueError('arm targets changed during gripper opening')
    if not -.50 <= q[7] <= .08 or abs(snapshot['velocity_rad_s'][7]) > .20:
        raise ValueError('gripper opening position/speed bound exceeded')
    if abs(q[7] - cmd[7]) > .12:
        raise ValueError('gripper opening tracking allowance exceeded')


def open_at_zero(directory, trial, target=-.40):
    """Moves only on this explicit invocation; no ACT or automatic retry."""
    from scripts.act_trial.manual_experiment import check_pose, wait_idle
    from scripts.act_trial.preflight import DEFAULT_BUNDLE
    from scripts.act_trial.telemetry import Telemetry

    report_path = directory/'opening.json'
    with report_path.open('x') as stream:
        stream.write('{}\n')
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    report = {'passed': False, 'motion_requested': False, 'ACT_started': False,
              'target_gripper_rad': target, 'arm_targets_held': True,
              'left_commands_emitted': False}
    telemetry = None
    try:
        with Client() as client:
            state = client.request('status')
            pv = state.get('controller_command_interfaces', {}).get(
                'right_joint_trajectory_controller') == ['position', 'velocity']
            if target == -.02 and not pv:
                raise ValueError('closed-transit setup requires position+velocity right JTC')
            if state['status']['state'] != 'idle':
                raise ValueError('gripper opening requires idle gateway')
            if 'right_arm_gripper' not in state.get('supported_dora_scopes', []):
                raise ValueError('gateway lacks right_arm_gripper support')
            telemetry = Telemetry(directory/'opening.can.jsonl')
            deadline = time.monotonic() + 5.
            while True:
                try:
                    snapshot = telemetry.snapshot()
                    check_pose(snapshot, 'h1', bundle)
                    break
                except (ValueError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            if not state['synchronized']:
                client.request('synchronize-hold')
            wait_idle(client, synchronizing=True)
            snapshot = telemetry.snapshot()
            check_pose(snapshot, 'h1', bundle)
            initial = np.asarray(snapshot['command_positions_rad'][:8], dtype=float)
            check_opening_feedback(snapshot, initial, bundle)
            acquired = client.acquire('dora', scope='right_arm_gripper')
            try:
                reference = np.asarray(acquired['right_reference'])
                if np.max(np.abs(reference - initial[:7])) > .003:
                    raise ValueError('gateway/CAN arm reference disagreement')
                initial[:7] = reference
                report['initial_reference_rad'] = initial.tolist()
                report['motion_requested'] = True
                write_json(report_path, report)
                print(f'ZERO_GRIPPER_SETUP: right gripper -> {target} rad over 10s; '
                      'right arm targets held; keep gripper empty.', flush=True)
                began = previous = time.monotonic()
                elapsed = 0.; settled_since = None
                with (directory/'opening.commands.jsonl').open('x', buffering=1) as log:
                    while True:
                        now = time.monotonic()
                        if now - previous > .10 or now - began > 20.:
                            raise ValueError('gripper opening watchdog/timeout')
                        if (trial/'stop.request').exists():
                            raise ValueError('operator stop requested during gripper opening')
                        elapsed += min(max(now - previous, 0.), .02)
                        previous = now
                        snapshot = telemetry.snapshot()
                        check_opening_feedback(snapshot, initial, bundle)
                        command = opening_reference(initial, elapsed, target=target)
                        client.renew()
                        if pv:
                            client.command(command.tolist(), velocities=[0.]*7)
                        else:
                            client.command(command.tolist())
                        log.write(json.dumps({'monotonic_s': now, 'target_rad': command.tolist(),
                                              'snapshot': snapshot})+'\n')
                        if elapsed >= 10.:
                            try:
                                if target == -.02:
                                    check_pose(snapshot, 'visual32_zero', bundle)
                                if abs(snapshot['positions_rad'][7] - command[7]) > (.035 if target == -.02 else .03):
                                    raise ValueError('gripper has not settled within ACT admission tolerance')
                            except ValueError:
                                settled_since = None
                            else:
                                if settled_since is None:
                                    settled_since = now
                                if now - settled_since >= .5:
                                    break
                        time.sleep(.02)
                client.release()
                wait_idle(client)
                report['passed'] = True
                report['final_snapshot'] = telemetry.snapshot()
            except BaseException as exc:
                if client.token is not None:
                    client.request('freeze', reason='zero gripper opening failed: '+str(exc))
                raise
        print('ZERO_GRIPPER_SETTLED: run ./scripts/actctl zero-handoff next.', flush=True)
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        write_json(report_path, report)
        if telemetry is not None:
            telemetry.close()
