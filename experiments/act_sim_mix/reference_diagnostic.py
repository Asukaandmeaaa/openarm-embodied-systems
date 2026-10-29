"""Compare recorded expert targets and policy from the exact same saved pose.

This is a diagnostic on validation episodes, not the deployment-zero benchmark.
No planner recomputation, object attachments, runtime drivers or real cameras.
The policy path reuses run_scene; its recovery initializer restores the saved
normal frame 0 or expert takeover frame, with source_kind retained in reports.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .closed_loop import Runner, advance_until, execute_block, run_scene, set_recovery_start, task_result
from .contracts import FPS, digest
from .paths import recorded_path
from .policy_limits import DeploymentEnvelope, DeploymentPolicyWorld
from .prepare_eval_scenes import verify_scene_file, runtime_binding
from .sim_expert.world import load_config
from .sim_reference import BiquadLowpass


def recorded_initialization(scene: dict) -> dict:
    # `kind=recovery` selects the existing saved-state initialization only.
    # Success criteria use recovery_type (None for normal scenes), so normal
    # hold/lift criteria remain intact. Never aggregate these as recovery trials.
    return {**scene, 'kind': 'recovery', 'source_kind': scene['kind'],
            'diagnostic_initialization': 'recorded_demonstration_pose',
            'takeover_frame': int(scene.get('takeover_frame', 0))}


def load_reference(scene, max_frames=None):
    raw = recorded_path(scene['collection_root']) / scene['source_episode'] / 'numeric.npz'
    if digest(raw) != scene['numeric_sha256']:
        raise ValueError('Changed reference numeric source')
    start = int(scene['takeover_frame'])
    with np.load(raw) as z:
        if not z['valid'][start:].all():
            raise ValueError('Reference must start at an expert-valid suffix')
        actions = z['action'][start:].copy()
        initial_state = z['state'][start].copy()
    available = len(actions)
    if max_frames is not None:
        actions = actions[:max_frames]
    if len(actions) < 2:
        raise ValueError('Reference needs at least two frames')
    return actions, initial_state, available


def oracle_rollout(scene, cfg, *, mode, steps, observation_wait_s, reference_wait_s, max_frames=None):
    actions, initial_state, available = load_reference(scene, max_frames)
    envelope = DeploymentEnvelope()
    world = DeploymentPolicyWorld(scene['seed'], cfg, jitter=scene.get('jitter'), envelope=envelope)
    try:
        set_recovery_start(world, Path('.'), scene)
        initial_error = float(np.max(np.abs(world.state16() - initial_state)))
        if initial_error > 1e-6:
            raise ValueError('Reference restoration differs from saved feedback')
        left_hold = world.state16()[8:].copy()
        lowpass = BiquadLowpass(250, 15)
        consumed, clipped, failure = 0, 0, None
        blocks = []
        try:
            if mode == 'zoh':
                # Original 30Hz ZOH targets, fixed absolute deadlines; no artificial
                # per-frame camera/inference pauses in the original expert path.
                for i, a in enumerate(actions):
                    command = a.copy()
                    command[8:] = left_hold
                    command, axes = envelope.apply(command)
                    clipped += bool(len(axes))
                    advance_until(world, (i + 1) / FPS, command, 'REFERENCE_ZOH')
                    consumed = i + 1
            else:
                while consumed < len(actions):
                    # Fixed, explicitly reported wait, not claimed measured policy
                    # latency. This isolates reference execution from GPU warmup.
                    world.hold_for(observation_wait_s + reference_wait_s, 'REFERENCE_WAIT')
                    indices = np.minimum(consumed + np.arange(100), len(actions) - 1)
                    prediction = actions[indices]
                    commit = max(2, min(steps, len(actions) - consumed))
                    row = execute_block(world, prediction, commit, lowpass,
                        reset=not blocks, left_hold=left_hold, envelope=envelope)
                    blocks.append(row)
                    clipped += row['clipped_actions']
                    consumed += commit
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
            failed_block = getattr(exc, 'block_diagnostics', None)
        else:
            failed_block = None
        result = task_result(world, recovery_type=scene.get('recovery_type'))
        if failure:
            result.update(success=False, failure_type=failure)
        truncated = len(actions) < available
        return {'mode': mode, 'success': bool(result['success']), 'failure_type': result['failure_type'],
                'result': result, 'reference_frames_available': available,
                'reference_frames_consumed': min(consumed, len(actions)), 'reference_truncated': truncated,
                'diagnostic_complete': not truncated,
                'reference_sequence_completed': failure is None and consumed >= len(actions),
                'initial_state_max_abs_error_rad': initial_error,
                'sim_time_s': float(world.data.time), 'clipped_reference_attempts': int(clipped),
                'blocks': blocks, 'failed_block': failed_block,
                'reference_wait_s': reference_wait_s if mode == 'chunk' else 0.,
                'observation_wait_s': observation_wait_s if mode == 'chunk' else 0.,
                'note': 'Oracle uses stored future expert actions. No success claim for a truncated smoke.'}
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--scene-id', action='append', required=True)
    parser.add_argument('--modes', nargs='+', choices=('zoh', 'chunk', 'policy'), default=['zoh', 'chunk', 'policy'])
    parser.add_argument('--exec-steps', type=int, default=50)
    parser.add_argument('--max-seconds', type=float, default=150.)
    parser.add_argument('--max-reference-frames', type=int, help='Short wiring smoke only; not a full reference test')
    parser.add_argument('--reference-wait-s', type=float, default=.04, help='Explicit fixed oracle inference wait approximation')
    parser.add_argument('--observation-wait-s', type=float, default=1 / FPS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve it and choose a new path')
    if not 2 <= args.exec_steps <= 100 or args.max_seconds <= 0 or args.reference_wait_s < 0 or args.observation_wait_s < 0:
        parser.error('invalid execution budget')
    if args.max_reference_frames is not None and args.max_reference_frames < 2:
        parser.error('short reference requires at least 2 frames')
    if len(set(args.scene_id)) != len(args.scene_id) or len(set(args.modes)) != len(args.modes):
        parser.error('duplicate scene/mode')
    frozen = json.loads(args.scenes.read_text())
    verify_scene_file(frozen)
    if frozen.get('split') != 'validation':
        parser.error('use validation scenes for diagnosis; keep final test held out')
    selected = {s['id']: s for s in frozen['scenes']}
    if any(s not in selected for s in args.scene_id):
        parser.error('unknown scene ID')
    args.output.mkdir(parents=True)
    report = {'schema': 'act_reference_diagnostic_v1', 'model': str(args.model.resolve()),
              'model_files_sha256': {name: digest(args.model / name) for name in ('config.json', 'model.safetensors')},
              'scene_file_sha256': digest(args.scenes), 'source_sha256': digest(Path(__file__)),
              'runtime_sha256': runtime_binding(), 'exec_steps': args.exec_steps,
              'initialization': 'same recorded demonstration pose for all modes; not deployment_zero',
              'policy_wait': 'measured render/inference wall time; fixed reference_wait is an explicit oracle approximation',
              'cases': [], 'hardware_accessed': False,
              'note': 'Diagnostic on selected validation episodes, not a new overall benchmark success rate.'}
    cfg = load_config()
    runner = Runner(args.model.resolve()) if 'policy' in args.modes else None
    for identifier in args.scene_id:
        source = selected[identifier]
        scene = recorded_initialization(source)
        for mode in args.modes:
            print('CASE START', identifier, mode, flush=True)
            if mode == 'policy':
                budget = min(args.max_seconds, args.max_reference_frames / FPS) if args.max_reference_frames else args.max_seconds
                policy_args = argparse.Namespace(command_envelope='deployment', exec_steps=args.exec_steps,
                    max_seconds=budget, max_chunks=math.ceil(budget * FPS / args.exec_steps) + 20,
                    observation_wait_s=args.observation_wait_s)
                result = run_scene(runner, cfg, scene, Path('.'), policy_args)
                result['diagnostic_complete'] = args.max_reference_frames is None
                result['reference_truncated'] = args.max_reference_frames is not None
            else:
                result = oracle_rollout(scene, cfg, mode=mode, steps=args.exec_steps,
                    observation_wait_s=args.observation_wait_s, reference_wait_s=args.reference_wait_s,
                    max_frames=args.max_reference_frames)
            result.update(mode=mode, source_scene_id=identifier, source_kind=source['kind'],
                          initialization_frame=scene['takeover_frame'])
            casefile = args.output / f'{identifier}_{mode}.json'
            casefile.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            brief = {'scene': identifier, 'mode': mode, 'source_kind': source['kind'],
                     'success': result['success'], 'failure': result['failure_type'],
                     'diagnostic_complete': result['diagnostic_complete'], 'file': str(casefile.resolve())}
            report['cases'].append(brief)
            print('CASE RESULT', json.dumps(brief), flush=True)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
