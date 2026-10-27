"""Offline 30-Hz expert-label -> actual executor/Guard q/qd comparison.

No ROS initialization, camera capture, CAN, gateway, or motor commands. This is
an ideal-clock conversion audit, NOT a physics simulation or hardware approval.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from scripts.act_trial.action_queue import ACTION_STEPS, INTERVAL_NS, sample_times
from scripts.act_trial.closed_contract import EPISODE, ROOT, CHECKPOINTS, execution_contract
from scripts.act_trial.guard import Guard
from scripts.act_trial.motion_generator import attach
from scripts.act_trial.preflight import DEFAULT_BUNDLE, run as preflight
from scripts.convert_real_episodes_act import prepare
from vision_grasp_baseline.training_contract import audit_action_contract, read_rows


def executor_module():
    path = ROOT/'official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py'
    spec = importlib.util.spec_from_file_location('_closed_audit_executor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(error):
    return dict(mae=np.mean(abs(error), axis=0).tolist(),
                p95=np.quantile(abs(error), .95, axis=0).tolist(),
                max=np.max(abs(error), axis=0).tolist())


def run(output, episode=EPISODE):
    # Kept only to preserve the provenance of historical reports. This tool
    # used a retired ten-step execution path; do not mislabel it as validation
    # of the current rolling trajectory or silently manufacture new evidence.
    if execution_contract().get('trajectory_contract') == 'rolling_time_aligned_v1':
        raise ValueError('retired ten-step replay audit does not validate rolling execution')
    output, episode = Path(output), Path(episode)
    if output.exists():
        raise FileExistsError(output)
    evidence = audit_action_contract(episode)
    streams, q, ids, alignment = prepare(episode, 30)
    labels = q['demonstration_action'][ids['demonstration_action']]
    observations = q['joint_observation'][ids['joint_observation']]
    raw = [r for r in read_rows(episode/'controller_raw.jsonl') if r['controller'] == 'right_arm']
    native_t = np.array([r['timestamp_s'] for r in raw])
    native_q = np.array([r['message']['output']['positions'] for r in raw])
    native_v = np.array([r['message']['output']['velocities'] for r in raw])
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    initial_command = streams['demonstration_action'][ids['demonstration_action'][0]]['positions'][:8]
    guard = Guard(observations[0], bundle, now=0., initial_command8=initial_command,
                  profile='visual32', motion_generator='position_stream_v1')
    guard.tick(None, observations[0], .01, 0.)
    guard.start(.01)
    original = executor_module()
    lowpass = original.BiquadLowpass(100, 15)
    lowpass.reset_state(labels[0])
    times, evaluation, duration = sample_times(100)
    upsampler = original.HermiteUpsampler(1e9/INTERVAL_NS, (ACTION_STEPS-1)*INTERVAL_NS/1e9)
    generated, requested, stamps, velocities, faults = [], [], [], [], []
    interventions = {}
    for begin in range(0, len(labels), ACTION_STEPS):
        chunk = labels[begin:begin+ACTION_STEPS]
        if len(chunk) < ACTION_STEPS:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], ACTION_STEPS-len(chunk), axis=0)])
        values = upsampler.upsample(chunk, evaluation)
        for offset, value in zip(times, values):
            elapsed = begin*INTERVAL_NS/1e9 + float(offset)
            if elapsed > (len(labels)-1)/30:
                break
            shaped = lowpass.step(value)
            state = observations[min(int(elapsed*30), len(observations)-1)]
            decision = guard.tick(shaped[:8], state, .02+elapsed, 0., .05+float(offset))
            if decision['stage'] == 'fault':
                faults.append(dict(time_s=elapsed, errors=decision['errors']))
                break
            generated.append(decision['dispatched_proposal'])
            velocities.append(decision['proposal_velocity_rad_s'])
            requested.append(shaped[:8])
            stamps.append(elapsed)
            for name in decision['interventions']:
                interventions[name] = interventions.get(name, 0)+1
        if faults:
            break
    if not stamps:
        raise ValueError('no offline conversion samples: '+str(faults))
    stamps, generated, velocities, requested = map(np.asarray, (stamps, generated, velocities, requested))
    source_times = alignment['start_timestamp_s']+stamps
    reference_q = np.column_stack([np.interp(source_times, native_t, native_q[:, j]) for j in range(7)])
    reference_v = np.column_stack([np.interp(source_times, native_t, native_v[:, j]) for j in range(7)])
    q_error, v_error = generated[:, :7]-reference_q, velocities[:, :7]-reference_v
    # Diagnostic review limits only; never a proof of physical safety.
    conversion_within_review_bounds = (not faults and np.max(abs(q_error)) <= .03
                                       and np.max(abs(v_error)) <= .10)
    output.mkdir(parents=True)
    np.savez_compressed(output/'comparison.npz', elapsed_s=stamps, requested=requested,
                        generated_q=generated, generated_qd=velocities,
                        recorded_q=reference_q, recorded_qd=reference_v)
    report = dict(hardware_io_performed=False, physical_replay_authorized=False,
                  episode=str(episode), provenance=evidence, alignment=alignment,
                  deployment=execution_contract(), samples=len(stamps), faults=faults,
                  assumptions=['ideal executor schedule; no inference/IPC latency',
                               'recorded feedback supplied; not simulated response to new commands',
                               'same Hermite/15Hz lowpass and position-stream Guard as deployment',
                               'timestamps interpolated only for offline q/qd comparison'],
                  arm_position_error_rad=metrics(q_error), arm_velocity_error_rad_s=metrics(v_error),
                  executor_position_error_rad=metrics(requested[:, :7]-reference_q),
                  recorded_peak_arm_speed_rad_s=np.max(abs(reference_v), axis=0).tolist(),
                  configured_arm_speed_cap_rad_s=guard.speed[:7].tolist(),
                  configured_acceleration_cap_rad_s2=guard.ACCELERATION,
                  shaping_error_rad=metrics(generated-requested), interventions=interventions,
                  review_bounds=dict(max_arm_position_error_rad=.03, max_arm_velocity_error_rad_s=.10),
                  conversion_within_review_bounds=bool(conversion_within_review_bounds),
                  physical_validation_passed=False)
    from scripts.act_trial.closed_contract import conversion_source_hashes
    report['conversion_source_sha256'] = conversion_source_hashes()
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for step, checkpoint in CHECKPOINTS.items():
        audit, config = preflight(checkpoint_path=checkpoint, profile='visual32')
        attach(config, audit, 'position_stream_v1')
        directory = output/f'{step//1000}K'
        directory.mkdir()
        (directory/'offline_preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
        (directory/'trial_config.json').write_text(json.dumps(config, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({k: report[k] for k in ('samples', 'faults', 'conversion_within_review_bounds',
                                           'arm_position_error_rad', 'arm_velocity_error_rad_s')}, indent=2))


if __name__ == '__main__':
    main()
