#!/usr/bin/env python3
"""Collect independent MuJoCo environments in spawned processes; one dataset writer."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import time

# Set these before importing NumPy/MuJoCo, including in spawned workers.
os.environ.setdefault('MUJOCO_GL', 'egl')
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from collect_act_fixed_scene import (
    ExpertRecorder, Pilot, SCENE, manifest, render_record, verify_replay, write_coverage,
)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def initialize_worker():
    # Ctrl-C reaches the whole foreground process group. Let the coordinator
    # drain workers instead of interrupting a render or a staging transaction.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_attempt(args, attempt, fidelity):
    """Only this process touches its model, renderer and staging dataset."""
    started = time.monotonic()
    out = args.output / 'attempts' / f'{attempt:06d}'
    out.mkdir(parents=True, exist_ok=True)
    staging = args.output / '.parallel_work' / f'{attempt:06d}'
    with (out / 'worker.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        pilot = Pilot(args.seed + attempt, out, args.cube_xy_jitter_m,
                      args.cube_yaw_jitter_deg, args.table_xy_jitter_m, args.table_yaw_jitter_deg)
        result = pilot.run()
        dynamics_done = time.monotonic()
        if result['success']:
            replay = verify_replay(pilot, result)
            atomic_json(out / 'replay.json', replay)
            result['replay_success'] = replay['success']
            result['success'] = replay['success']
            if not replay['success']:
                result['failure'] = 'replay: ' + str(replay['failure'] or 'final outcome check failed')
        replay_done = time.monotonic()
        entry = None
        if result['success'] and not args.validate_only:
            details = {
                'seed': args.seed + attempt,
                'source_trajectory_trial_id': f'sim_fixed_{args.seed + attempt}',
                'randomization': result['randomization'],
                'grasp_world_yaw_rad': result.get('grasp_world_yaw_rad'),
                'scene_fidelity': fidelity, 'source_domain': 'simulation',
                'success_reason': 'contact_grasp_transport_planned_release_home_and_30hz_replay',
                'duration_s': result['duration_s'], 'initial_qpos': result['initial_qpos'],
                'attempt_report': str(out.relative_to(args.output)),
            }
            recorder = ExpertRecorder(staging, {}, frame_shape=(360, 640, 3))
            render_record(pilot, recorder, details, attempt)
            entry = recorder.entries[0]
        record = {
            'attempt': attempt, 'seed': args.seed + attempt, 'success': result['success'],
            'randomization': result['randomization'], 'failure': result.get('failure'),
            'replay_success': result.get('replay_success'), 'episode_id': None,
            'worker_pid': os.getpid(),
            'timing_s': {'dynamics': dynamics_done - started,
                         'replay': replay_done - dynamics_done,
                         'render': time.monotonic() - replay_done,
                         'total': time.monotonic() - started},
        }
        # Durable handoff also lets a restarted coordinator finish completed work.
        payload = {'record': record, 'entry': entry}
        atomic_json(out / 'parallel_ready.json', payload)
        return payload


def commit_result(output, recorder, payload):
    """The coordinator alone assigns IDs and replaces shared metadata."""
    record = dict(payload['record'])
    entry = payload['entry']
    if entry is not None:
        # A crash after metadata commit but before the attempt journal must not duplicate it.
        existing = next((e for e in recorder.entries
                         if e.get('source_trajectory_trial_id') == entry['source_trajectory_trial_id']), None)
        if existing is not None:
            record['episode_id'] = int(existing['id'])
        else:
            episode_id = recorder.next_episode_id
            staging = output / '.parallel_work' / f'{record["attempt"]:06d}'
            source = staging / 'episodes' / str(entry['id'])
            destination = output / 'episodes' / str(episode_id)
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            os.replace(source, destination)
            recorder.entries.append({**entry, 'id': str(episode_id)})
            recorder._write_metadata_atomic()
            record['episode_id'] = episode_id
    with (output / 'attempts.jsonl').open('a') as journal:
        journal.write(json.dumps(record) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
    shutil.rmtree(output / '.parallel_work' / f'{record["attempt"]:06d}', ignore_errors=True)
    return record


def collect(args, recorder, fidelity, worker=run_attempt, executor_factory=ProcessPoolExecutor):
    attempt_dirs = args.output / 'attempts'
    attempt_dirs.mkdir(exist_ok=True)
    journal = args.output / 'attempts.jsonl'
    logged = {json.loads(line)['attempt'] for line in journal.read_text().splitlines() if line} if journal.exists() else set()
    successes = len(recorder.entries) if recorder is not None else 0
    for ready in sorted(attempt_dirs.glob('*/parallel_ready.json')):
        payload = json.loads(ready.read_text())
        if payload['record']['attempt'] in logged:
            continue
        entry = payload['entry']
        if entry is not None and recorder is None:
            raise ValueError('Cannot validate-only resume a rendered dataset')
        # Recorder.recover() quarantines a directory if a crash preceded its metadata commit.
        # Preserve that audit evidence; do not admit an incomplete handoff.
        source = args.output / '.parallel_work' / ready.parent.name / 'episodes' / str(entry['id']) if entry else None
        committed = entry and any(e.get('source_trajectory_trial_id') == entry['source_trajectory_trial_id'] for e in recorder.entries)
        if entry and not committed and not source.exists():
            print(f'skipping interrupted handoff: attempt={payload["record"]["attempt"]}', flush=True)
            continue
        if entry and not committed and successes >= args.episodes:
            continue
        commit_result(args.output, recorder, payload)
        successes = len(recorder.entries) if recorder is not None else 0
        write_coverage(args.output, args)
    if successes >= args.episodes:
        print(f'already complete: {successes}', flush=True)
        return
    used = [int(path.name) for path in attempt_dirs.iterdir() if path.is_dir() and path.name.isdigit()]
    attempt = max(used, default=-1) + 1
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        print('Stop requested; finishing in-flight attempts and saving their results.', flush=True)

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    initial_successes = successes
    pending = []
    try:
        with executor_factory(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                              initializer=initialize_worker) as pool:
            while True:
                # No speculative successes beyond the requested total. Consume in seed order
                # so faster trajectories do not get preferential admission at the target.
                capacity = min(args.workers, args.episodes - successes)
                if args.validate_only:
                    capacity = min(args.workers, args.max_attempts - attempt + len(pending))
                while not stopped and attempt < args.max_attempts and len(pending) < capacity:
                    (attempt_dirs / f'{attempt:06d}').mkdir(exist_ok=True)
                    pending.append((attempt, pool.submit(worker, args, attempt, fidelity)))
                    attempt += 1
                if not pending:
                    break
                number, future = pending.pop(0)
                payload = future.result()
                record = commit_result(args.output, recorder, payload)
                successes = len(recorder.entries) if recorder is not None else 0
                write_coverage(args.output, args)
                elapsed = time.monotonic() - started
                rate = (successes - initial_successes) * 3600 / max(elapsed, .001)
                print(f'committed={successes}/{args.episodes}; attempt={number}; '
                      f'success={record["success"]}; workers={args.workers}; '
                      f'new_episodes_per_hour={rate:.1f}; wall_s={elapsed:.1f}', flush=True)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if stopped:
        print(f'stopped cleanly: {successes}/{args.episodes}', flush=True)
        return
    if args.validate_only:
        return
    if successes < args.episodes:
        raise SystemExit(f'Incomplete: {successes}/{args.episodes} successes; attempt budget reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=300, help='total successes including existing episodes')
    parser.add_argument('--max-attempts', type=int, default=1800, help='absolute attempt budget, including existing attempts')
    parser.add_argument('--seed', type=int, default=20262000)
    parser.add_argument('--cube-xy-jitter-m', type=float, default=.01)
    parser.add_argument('--cube-yaw-jitter-deg', type=float, default=30.)
    parser.add_argument('--table-xy-jitter-m', type=float, default=.02)
    parser.add_argument('--table-yaw-jitter-deg', type=float, default=5.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='parallel dynamics/replay only, no images; use a separate output and small max-attempts')
    args = parser.parse_args()
    import math
    bounds = (args.cube_xy_jitter_m, args.cube_yaw_jitter_deg, args.table_xy_jitter_m, args.table_yaw_jitter_deg)
    if min(args.workers, args.episodes, args.max_attempts) < 1 or any(not math.isfinite(b) or b < 0 for b in bounds):
        parser.error('counts must be positive and randomization finite and nonnegative')
    args.output = args.output.resolve()
    cfg = json.loads((SCENE / 'measurements.json').read_text())
    if not args.validate_only and not cfg.get('approved_for_collection'):
        parser.error('scene is not approved for collection')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.collector.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another collector owns this output; stop it before resuming')
        binding = manifest(args)
        binding_path = args.output / 'source_manifest.json'
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ValueError('Existing run source/config changed; use a separate output')
        atomic_json(binding_path, binding)
        metadata = {
            'operation_type': 'expert', 'source_domain': 'simulation', 'scene_id': 'act_fixed_20261015',
            'scene_fidelity': cfg.get('collection_review', {}),
            'tasks': [{'prompt': 'Put the blue cube inside the black box with the right arm.'}],
            'frequencies': {'action': {'arms': {'right': 30, 'left': 30}},
                            'obs': {'arms': {'right': 30, 'left': 30}},
                            'cameras': {'ceiling': 30, 'wrist_right': 30}},
            'camera_shape': [360, 640, 3],
            'state_action_order': 'right J1..J7,gripper; left J1..J7,gripper',
            'expert': binding, 'simulation_duration_limit_s': None,
        }
        recorder = None if args.validate_only else ExpertRecorder(args.output, metadata, frame_shape=(360, 640, 3))
        with (args.output / 'parallel_runs.jsonl').open('a') as history:
            history.write(json.dumps({'started_unix_s': time.time(), 'pid': os.getpid(),
                                      'workers': args.workers, 'start_method': 'spawn',
                                      'coordinator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                      'validate_only': args.validate_only}) + '\n')
        print(f'parallel collector pid={os.getpid()}; workers={args.workers}; output={args.output}', flush=True)
        collect(args, recorder, cfg.get('collection_review', {}))


if __name__ == '__main__':
    main()
