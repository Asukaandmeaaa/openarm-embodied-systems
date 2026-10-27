"""No-motion integrated recording stress test with repeated live planning."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .collection_io import rpc, write_json
from .cycle import BASE, WORKSPACE, wait_for_gateway_handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--plans', type=int, default=5)
    args = parser.parse_args()
    if args.seconds < 5 or args.plans < 1:
        parser.error('seconds >= 5 and plans >= 1 required')
    lock = (BASE/'.collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cycle_lock = (BASE/'.cycle.lock').open('a')
    fcntl.flock(cycle_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = (args.output or BASE/'runs'/datetime.now().strftime('recording_check_%Y%m%d_%H%M%S_%f')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(no_motion=True, success=False, health=[], plans=[])
    service = child = None
    recording = False
    try:
        wait_for_gateway_handoff()
        with (out/'service.log').open('w') as log:
            service = subprocess.Popen([str(BASE/'run.sh'), 'collection-service', '--ready', str(out/'service.json')],
                                       cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        error = 'not ready'
        while time.monotonic() < deadline:
            if service.poll() is not None:
                raise RuntimeError('Recorder exited: '+str(out/'service.log'))
            if (out/'service.json').exists():
                settings = json.loads((out/'service.json').read_text())
                os.environ['VISION_CAPTURE_SERVICE'] = settings['url']
                os.environ['VISION_CAPTURE_TOKEN'] = settings['token']
                try:
                    rpc('status')
                    break
                except Exception as exc:
                    error = str(exc)
            time.sleep(.2)
        else:
            raise RuntimeError(error)
        rpc('begin', path=str(out/'recording_probe'))
        recording = True
        started = time.monotonic()
        for i in range(args.plans):
            capture = out/f'capture_{i:02d}'
            commands = [('capture', '--output', capture, '--samples', '10'),
                        ('detect', '--capture', capture, '--calibration', BASE/'config/global_camera_20261015.calib'),
                        ('grasp-plan', '--scene', capture/'scene.json', '--registration', BASE/'config/session_registration_20261021.json',
                         '--output', out/f'plan_{i:02d}.json')]
            for cmd in commands:
                path = out/f'{i:02d}_{cmd[0]}.log'
                with path.open('w') as log:
                    child = subprocess.Popen([str(BASE/'run.sh'), *map(str, cmd)], cwd=WORKSPACE,
                                             stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    try:
                        report['health'].append(rpc('status'))
                    except Exception as exc:
                        # No motion is being sent: retain diagnostic capture through the fault.
                        report['health'].append(dict(ok=False, error=str(exc)))
                    time.sleep(.1)
                if child.returncode:
                    raise RuntimeError('Planning check failed: '+str(path)+'\n'+path.read_text()[-2000:])
            report['plans'].append(str(out/f'plan_{i:02d}.json'))
            print(f'NO-MOTION planning+recording {i+1}/{args.plans} completed', flush=True)
        while time.monotonic()-started < args.seconds:
            report['health'].append(rpc('status'))
            time.sleep(.1)
        result = rpc('finish', success=False)
        recording = False
        report['quality'] = result['quality']
        report['success'] = result['quality']['passed'] and all(item['ok'] for item in report['health'])
        # Diagnostic data deliberately never counts as a successful task episode.
        path = out/'recording_probe/metadata.json'
        metadata = json.loads(path.read_text())
        metadata.update(diagnostic_only=True, terminal_label='recording_probe_no_motion')
        write_json(path, metadata)
        # The 30-episode loop keeps this same service alive across saves.
        # Exercise finish -> fresh callbacks -> begin again without restarting it.
        rpc('status')
        rpc('begin', path=str(out/'recording_probe_reuse'))
        recording = True
        for _ in range(30):
            report['health'].append(rpc('status'))
            time.sleep(.1)
        report['reuse_quality'] = rpc('finish', success=False)['quality']
        recording = False
        report['success'] &= report['reuse_quality']['passed']
        metadata_path = out/'recording_probe_reuse/metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata.update(diagnostic_only=True, terminal_label='recording_probe_no_motion')
        write_json(metadata_path, metadata)
        print('SAME-SERVICE second recording: '+str(report['reuse_quality']['passed']), flush=True)
        print(json.dumps(dict(output=str(out), success=report['success'], no_motion=True, quality=report['quality']), indent=2))
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        if child and child.poll() is None:
            # Let plan-only code restore temporary scene attachments in finally.
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.terminate(); child.wait(timeout=5)
        if recording:
            try: report['quality'] = rpc('finish', success=False)['quality']
            except Exception as exc: report['finish_error'] = str(exc)
        if service and service.poll() is None:
            service.terminate()
            try: service.wait(timeout=15)
            except subprocess.TimeoutExpired: service.kill(); service.wait()
        write_json(out/'check_result.json', report)
        cycle_lock.close(); lock.close()
    if not report['success']:
        raise RuntimeError('Integrated recording check failed: '+str(out/'check_result.json'))


if __name__ == '__main__':
    main()
