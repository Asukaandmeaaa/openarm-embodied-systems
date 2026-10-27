"""Local camera/recorder RPC and timing audit; no hardware imports."""
import json
import math
import os
from pathlib import Path
import urllib.request
from bisect import bisect_right

ORDER = [name for side in ('right', 'left') for name in
         [*[f'openarm_{side}_joint{i}' for i in range(1, 8)], f'openarm_{side}_finger_joint1']]
CAMERAS = ('ceiling', 'wrist_right')
TARGET_STREAMS = ('right_arm', 'right_gripper', 'left_arm')


def assemble_actions(path, parked_finger):
    """Join independently recorded controllers by source time after draining.

    Callback arrival order must never decide whether another controller's sample
    is missing. No interpolation, future target, or feedback substitution.
    """
    path = Path(path)
    rows = {key: [json.loads(line) for line in (path/f'target_{key}.jsonl').read_text().splitlines()]
            for key in TARGET_STREAMS}
    times = {key: [r['timestamp_s'] for r in data] for key, data in rows.items()}
    for key, ts in times.items():
        if not ts or not all(math.isfinite(t) for t in ts) or any(b <= a for a, b in zip(ts, ts[1:])):
            raise RuntimeError('Missing/non-monotonic controller source stream: '+key)
    count = 0
    with (path/'demonstration_action.jsonl').open('w') as output:
        for right in rows['right_arm']:
            stamp = right['timestamp_s']
            ids = {key: bisect_right(ts, stamp)-1 for key, ts in times.items()}
            if min(ids.values()) < 0:
                continue
            parts = {key: rows[key][index] for key, index in ids.items()}
            ages = {key: stamp-row['timestamp_s'] for key, row in parts.items()}
            if max(ages.values()) > .10:
                raise RuntimeError('Controller source gap during action join: '+str(ages))
            positions = parts['right_arm']['positions']+parts['right_gripper']['positions']+parts['left_arm']['positions']+[parked_finger]
            output.write(json.dumps(dict(timestamp_s=stamp, positions=positions, order=ORDER,
                source='JTC output.positions joined causally by ROS source time; passive left finger frozen at start',
                component_timestamp_s={key: row['timestamp_s'] for key, row in parts.items()}), allow_nan=False)+'\n')
            count += 1
    return dict(count=count, max_component_age_s=.10, alignment='latest controller source <= right-arm source')


def rpc(operation, **payload):
    url = os.environ['VISION_CAPTURE_SERVICE']
    request = urllib.request.Request(url + '/' + operation,
        data=json.dumps(payload).encode(), headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.environ['VISION_CAPTURE_TOKEN']})
    with urllib.request.urlopen(request, timeout=2 if operation in ('status', 'event') else 30) as response:
        result = json.load(response)
    if not result.get('ok'):
        raise RuntimeError(result.get('error', 'Capture service failed'))
    return result


def event(phase, **detail):
    if os.environ.get('VISION_CAPTURE_SERVICE'):
        rpc('event', phase=phase, detail=detail)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def finish_interrupted_episode(path,reason):
    """Parent fallback when the authority supervisor terminates its executor."""
    metadata_path=Path(path)/'metadata.json'
    if not metadata_path.exists():return
    metadata=json.loads(metadata_path.read_text())
    if 'end_timestamp_s' not in metadata:
        rpc('finish',success=False)
        metadata=json.loads(metadata_path.read_text())
    metadata.update(automatic_task_contract_passed=False,operator_marked_success=False,
                    review_status='pending',terminal_label='failed_or_interrupted',execution_error=reason)
    write_json(metadata_path,metadata)


def timing_summary(timestamps, minimum_hz, maximum_gap):
    if len(timestamps) < 2:
        return dict(passed=False, count=len(timestamps), reason='fewer than two samples')
    gaps = [b-a for a, b in zip(timestamps, timestamps[1:])]
    valid = all(math.isfinite(t) for t in timestamps) and min(gaps) > 0
    hz = (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if valid else 0.
    return dict(count=len(timestamps), measured_hz=hz, max_gap_s=max(gaps),
                passed=valid and hz >= minimum_hz and max(gaps) <= maximum_gap)


def audit_episode(path):
    """Reject gaps, low throughput and missing start/end coverage, not just mean FPS."""
    path = Path(path)
    rows = {key: [json.loads(line) for line in (path/f'{key}.jsonl').read_text().splitlines()]
            for key in ('joint_observation', 'demonstration_action', 'camera')}
    streams = {key: rows[key] for key in ('joint_observation', 'demonstration_action')}
    streams.update({key: [r for r in rows['camera'] if r['key'] == key] for key in CAMERAS})
    limits = {'joint_observation': (60., .05), 'demonstration_action': (35., .10),
              'ceiling': (25., .15), 'wrist_right': (25., .15)}
    report = {key: timing_summary([r['timestamp_s'] for r in stream], *limits[key])
              for key, stream in streams.items()}
    metadata = json.loads((path/'metadata.json').read_text())
    for key, stream in streams.items():
        coverage = bool(stream) and (stream[0]['timestamp_s']-metadata['start_timestamp_s'] <= .15
                    and metadata['end_timestamp_s']-stream[-1]['timestamp_s'] <= .15)
        report[key]['covers_episode'] = coverage
        report[key]['passed'] &= coverage
    for key in CAMERAS:
        stream = streams[key]
        report[key]['unique_frame_ids'] = len({r['frame_number'] for r in stream}) == len(stream)
        report[key]['passed'] &= report[key]['unique_frame_ids']
        report[key]['files_present'] = all((path/r['path']).is_file() for r in stream)
        report[key]['passed'] &= report[key]['files_present']
    return dict(passed=all(r['passed'] for r in report.values()) and not metadata.get('recording_error'),
                streams=report, training_fps=30)


def check_fixed_scene(scene, reference, cube_tolerance=.04, bin_tolerance=.02, yaw_tolerance=.15):
    """Tolerate reset/detection noise but reject a changed collection fixture."""
    def distance(a, b):
        if len(a) != len(b) or not all(math.isfinite(x) for x in [*a, *b]):
            raise ValueError('Invalid detected fixture coordinates')
        return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    current, original = scene['measured'], reference['measured']
    cube_delta = distance(current['cube']['center'], original['cube']['center'])
    bin_delta = distance(current['bin']['center_xy'], original['bin']['center_xy'])
    yaw_delta = abs((current['bin']['yaw']-original['bin']['yaw']+math.pi/2) % math.pi-math.pi/2)
    if cube_delta > cube_tolerance or bin_delta > bin_tolerance or yaw_delta > yaw_tolerance:
        raise RuntimeError(f'Fixed scene changed: cube {cube_delta:.3f}m, bin {bin_delta:.3f}m / {yaw_delta:.3f}rad')
    return dict(cube_delta_m=cube_delta, bin_delta_m=bin_delta, bin_yaw_delta_rad=yaw_delta)
