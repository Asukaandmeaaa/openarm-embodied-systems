"""Export a marked capture interval without replacing it with endpoint waypoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from core import NAMES, check_driver_contract, check_start, digest, validate_bundle, write_json


def load(path):
    return json.loads(Path(path).read_text())


def export_capture(directory, phase='H2', source='command', max_age=.1):
    root = Path(directory).resolve()
    manifest = load(root/'manifest.json')
    if not manifest.get('complete') or (root/'recording.ready').exists():
        raise ValueError('capture must be stopped and complete before export')
    if manifest['counts']['socket_drops'] or manifest['counts']['can_errors']:
        raise ValueError('capture contains socket drops or CAN error frames')
    for rel, expected in manifest['files'].items():
        if digest(root/rel) != expected:
            raise ValueError(f'capture hash mismatch: {rel}')
    start = load(root/f'mark_{phase}_start.json')['wall_time_ns']
    end = load(root/f'mark_{phase}_end.json')['wall_time_ns']
    if not 0 < (end-start)/1e9 <= 300:
        raise ValueError('marked interval must be positive and at most 300 s')
    rows = []
    with (root/'can.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if start-int(max_age*1e9) <= row['kernel_wall_ns'] <= end+int(max_age*1e9) and row.get('decoded'):
                if abs(row['receive_wall_ns']-row['kernel_wall_ns']) > .1e9:
                    raise ValueError('CAN receiver latency/clock discrepancy exceeds 100 ms')
                rows.append(row)
    rows.sort(key=lambda r: r['kernel_wall_ns'])
    offsets = [r['receive_wall_ns']-r['receive_monotonic_ns'] for r in rows]
    if offsets and max(offsets)-min(offsets) > 10000000:
        raise ValueError('wall/monotonic clock offset changed by more than 10 ms')
    histories = {kind: [[] for _ in range(16)] for kind in ('command', 'feedback')}
    for row in rows:
        d = row['decoded']
        if d['kind'] in histories and (d['kind'] != 'feedback' or d['status'] == 1):
            axis = d['axis'] + (8 if row['side']=='left' else 0)
            histories[d['kind']][axis].append((row['kernel_wall_ns'], d['q']))
    caches = {'command': {}, 'feedback': {}}
    points, measured, times, axis_times, parameters = [], [], [], [], []
    selected = 'feedback' if source == 'measured' else 'command'
    boundary_checked = False
    def snapshot(kind, stamp):
        cache = caches[kind]
        if len(cache) != 16:
            raise ValueError(f'{phase}: missing initial {kind} axes; do not export a truncated startup')
        ages = [(stamp-cache[i][0])/1e9 for i in range(16)]
        if max(ages) > max_age:
            raise ValueError(f'{phase}: stale {kind} axis {NAMES[int(np.argmax(ages))]}, age={max(ages):.3f}s')
        return [cache[i][1]['q'] for i in range(16)]
    def append(stamp):
        feedback = snapshot('feedback', stamp)
        command = snapshot('command', stamp)
        points.append(feedback if selected == 'feedback' else command)
        measured.append(feedback)
        times.append((stamp-start)/1e9)
        axis_times.append([caches[selected][i][0] for i in range(16)])
        parameters.append([{k:v for k,v in caches['command'][i][1].items()
                            if k in ('dq', 'torque', 'kp', 'kd')} for i in range(16)])
    for row in rows:
        stamp = row['kernel_wall_ns']
        if stamp > end:
            break
        if stamp >= start and not boundary_checked:
            # Demand coverage AT the marked beginning, not after a missing prefix.
            append(start)
            boundary_checked = True
        d = row['decoded']
        kind = d['kind']
        if kind == 'lifecycle':
            if start <= stamp <= end:
                raise ValueError(f'{phase}: lifecycle {d["operation"]} inside marked interval; raw capture retained, not replayable as MIT positions')
            continue
        if kind == 'feedback' and d['status'] != 1:
            if start <= stamp <= end:
                raise ValueError('feedback not ENABLED throughout replay interval')
            continue
        axis = d['axis'] + (8 if row['side']=='left' else 0)
        caches[kind][axis] = (stamp, d)
        if stamp > start and kind == selected and (not times or stamp > start+round(times[-1]*1e9)):
            append(stamp)
    if not boundary_checked or len(points) < 2:
        raise ValueError('no complete captured motion in marked interval')
    if end > start+round(times[-1]*1e9):
        append(end)
    t = np.asarray(times)
    # CAN axes arrive sequentially. Interpolate each axis between its OWN
    # timestamped samples, then evaluate the union of source knot times. A
    # last-value 16D snapshot would create false, extremely fast microsteps.
    def aligned(kind):
        result = []
        for axis, history in enumerate(histories[kind]):
            stamps = np.array([s for s, _ in history], dtype=np.int64)
            if (len(stamps) < 2 or stamps[0] > start or stamps[-1] < end
                    or np.any(np.diff(stamps) <= 0)
                    or np.max(np.diff(stamps)) > max_age*1e9):
                raise ValueError(f'{kind} {NAMES[axis]} has missing boundary coverage or a sample gap')
            result.append(np.interp(t, (stamps-start)/1e9, [v for _, v in history]))
        return np.array(result).T
    q = aligned(selected)
    measured = aligned('feedback')
    points = q.tolist()
    tail = measured[t >= t[-1]-.5]
    if len(tail) < 2 or np.max(np.ptp(tail, axis=0)) > .02:
        raise ValueError('final measured pose is not stable over last 0.5 s')
    home_cfg = yaml.safe_load((root/'snapshots/home_config.yaml').read_text())
    driver_cfg = yaml.safe_load((root/'snapshots/driver_config.yaml').read_text())
    check_driver_contract(driver_cfg)
    ros_errors = []
    topics = {}
    with (root/'ros.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            if not start <= r['receive_wall_ns'] <= end:
                continue
            topic = r['topic']; topics[topic] = topics.get(topic, 0)+1
            if topic != '/joint_states' or r['receive_wall_ns'] < end-500000000:
                continue
            msg = r['message']
            if len(msg['name']) != len(msg['position']) or not set(NAMES).issubset(msg['name']):
                raise ValueError('ROS joint_states does not cover all 16 named axes')
            ros_q = np.array([msg['position'][msg['name'].index(name)] for name in NAMES])
            stamp = msg['header']['stamp']['sec']*10**9+msg['header']['stamp']['nanosec']
            if not start <= stamp <= end or not np.all(np.isfinite(ros_q)):
                raise ValueError('invalid ROS source timestamp or measured position')
            can_q = np.array([np.interp((stamp-start)/1e9, t, measured[:,i]) for i in range(16)])
            ros_errors.append(float(np.max(np.abs(ros_q-can_q))))
    if not ros_errors or max(ros_errors) > .03:
        raise ValueError('post-H2 ROS/CAN measured joint consistency missing or exceeds 0.03 rad')
    if phase == 'H2' and not topics.get('/right_joint_trajectory_controller/controller_state'):
        raise ValueError('no right arm controller reference was recorded during H2')
    h2 = load(root/'h2_execution.json') if phase == 'H2' else None
    if phase == 'H2':
        if not (h2.get('success') and h2.get('state') == 'HOME_READY'
                and h2.get('home_gripper', {}).get('accepted')):
            raise ValueError('H2 report did not reach verified HOME_READY')
        stages = h2.get('stages', [])
        if len(stages) != 2 or any(s.get('state') != 'ACCEPTED' or not s.get('robot_trajectory') for s in stages):
            raise ValueError('H2 report needs both accepted full timed trajectories; rebuild capture-enabled executor')
        for stage in stages:
            trajectory = stage['robot_trajectory']
            serialized = json.dumps(trajectory, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
            if hashlib.sha256(serialized).hexdigest() != stage.get('trajectory_sha256'):
                raise ValueError('full H2 plan does not match its execution trajectory hash')
            joint = trajectory['joint_trajectory']
            if joint['joint_names'] != NAMES[:7] or len(joint['points']) < 2:
                raise ValueError('invalid H2 planned joint ordering/points')
            previous = -1
            for point in joint['points']:
                elapsed = point['time_from_start']['sec']*10**9+point['time_from_start']['nanosec']
                if elapsed <= previous or len(point['positions']) != 7 or not np.all(np.isfinite(point['positions'])):
                    raise ValueError('invalid H2 plan timing/positions')
                previous = elapsed
        if h2['config_sha256'] != digest(root/'snapshots/home_config.yaml'):
            raise ValueError('recorded H2 configuration differs from capture snapshot')
        if not start <= int(h2['started_wall_time_s']*1e9) < int(h2['completed_wall_time_s']*1e9) <= end:
            raise ValueError('H2 report is outside this capture interval (stale/wrong report)')
    bundle = {'schema': 'openarm_recorded_home_v1', 'phase': phase, 'source': source,
              'joint_names': NAMES, 'units': {'positions':'rad', 'time':'s', 'torque':'Nm'},
              'interpolation': 'piecewise_linear_recorded_samples',
              'time_s': times, 'positions_rad': points,
              'measured_positions_rad': measured.tolist(),
              'axis_last_source_wall_ns': axis_times, 'recorded_mit_parameters': parameters,
              'axis_alignment': 'linear interpolation per axis using its kernel timestamp; all selected-source knots retained',
              'measured_start_rad': measured[0].tolist(),
              'measured_end_rad': np.median(tail, axis=0).tolist(),
              'recorded_start_wall_ns': start, 'recorded_end_wall_ns': end,
              'max_source_age_s': max_age, 'home_config': home_cfg, 'driver_config': driver_cfg,
              'h2_execution': h2,
              'ros_topics_in_interval': topics,
              'post_H2_ros_can_comparison': {'sample_count':len(ros_errors),
                                            'maximum_difference_rad':max(ros_errors)},
              'capture_directory': str(root), 'capture_manifest_sha256': digest(root/'manifest.json'),
              'source_hashes': {str(p.relative_to(root)): digest(p) for p in
                                [root/'manifest.json', root/'can.jsonl', root/'ros.jsonl',
                                 root/f'mark_{phase}_start.json', root/f'mark_{phase}_end.json']},
              'physical_replay_authorized': False,
              'notes': ['CAN command timestamps are local loopback observations, not synchronized actuator sampling.',
                        'Measured positions and decoded command positions are preserved separately.',
                        'Archived feedforward torque must not be replayed blindly as gravity compensation.',
                        'This export is for simulation; no physical control equivalence is implied.']}
    validate_bundle(bundle)
    # Archiving is not replay admission. Preserve a command/feedback offset as
    # evidence; the simulator/physical entry gate must still reject a bad start.
    try:
        check_start(bundle, bundle['measured_start_rad'])
        bundle['start_alignment'] = {'passed': True, 'tolerance_rad': .03}
    except ValueError as exc:
        bundle['start_alignment'] = {'passed': False, 'tolerance_rad': .03,
                                     'error': str(exc), 'automatic_alignment_allowed': False}
    prior = np.array(home_cfg['right_home_joint_positions_rad'] + [home_cfg['home_gripper']['open_position_rad']])
    bundle['post_H2_comparison'] = {'configured_right_home_rad': prior.tolist(),
                                  'measured_right_home_rad': bundle['measured_end_rad'][:8],
                                  'measured_minus_configured_rad': (np.array(bundle['measured_end_rad'][:8])-prior).tolist(),
                                  'max_error_rad': float(np.max(np.abs(np.array(bundle['measured_end_rad'][:8])-prior)))}
    return bundle


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture')
    p.add_argument('--phase', choices=['H1','H2'], default='H2')
    p.add_argument('--source', choices=['command','measured'], default='command')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        p.error('output exists; choose a new output path')
    result = export_capture(a.capture, a.phase, a.source)
    write_json(a.output, result)
    print(json.dumps({'file': a.output, 'samples': len(result['time_s']),
                      'duration_s': result['time_s'][-1], 'post_H2_comparison': result['post_H2_comparison']}))
