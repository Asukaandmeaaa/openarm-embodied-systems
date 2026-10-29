"""Read-only policy/data audit plus short, instrumented MuJoCo rollouts.

Preserves checkpoints, datasets and existing evaluation results. Synthetic
input interventions are sensitivity probes, never physical trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .closed_loop import Runner, run_scene
from .data import CAMS, EpisodeDataset, digest
from .paths import recorded_path
from .prepare_eval_scenes import verify_scene_file
from .reference_diagnostic import recorded_initialization
from .sim_expert.world import load_config
from .policy_limits import DeploymentEnvelope, DeploymentPolicyWorld


def consistent_pose_probes(runner, scene, raw, frames):
    """Static FK/render interventions: change both state and its rendered RGB."""
    import mujoco
    world = DeploymentPolicyWorld(scene['seed'], load_config(), jitter=scene.get('jitter'),
                                  envelope=DeploymentEnvelope())
    rows = []
    try:
        for frame in frames:
            world.data.qpos[:] = raw['qpos'][frame]
            mujoco.mj_forward(world.model, world.data)
            base_state = world.state16()
            baseline, _ = runner.predict(world.render_frames(), base_state)
            for degrees in (-1., 1.):
                q = base_state[:8].copy()
                q[5] += np.deg2rad(degrees)
                world.data.qpos[:] = raw['qpos'][frame]
                world.mapper.set_qpos(world.data.qpos, q, 'right')
                mujoco.mj_forward(world.model, world.data)
                pred, _ = runner.predict(world.render_frames(), world.state16())
                delta = np.rad2deg(pred[49,:7] - baseline[49,:7])
                rows.append(dict(raw_frame=int(frame),phase=str(raw['phase'][frame]),
                    J6_input_change_deg=degrees,h49_joint_change_deg=delta.tolist(),
                    J6_h49_response_ratio=float(delta[5]/degrees)))
    finally:
        world.close()
    return rows


def errors(prediction, target):
    delta = np.asarray(prediction) - np.asarray(target)
    return dict(arm_mae_deg=float(np.rad2deg(np.abs(delta[..., :7])).mean()),
                arm_max_deg=float(np.rad2deg(np.abs(delta[..., :7])).max()),
                joint_signed_mean_deg=np.rad2deg(delta[..., :7].mean(axis=0)).tolist(),
                gripper_mae_rad=float(np.abs(delta[..., 7]).mean()))


def images_of(item):
    return {role.rsplit('.', 1)[-1]: np.rint(item[role].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            for role in CAMS}


def infer_batch(runner, samples):
    batch = {k: torch.stack([s[k] for s in samples])
             for k in ('observation.state', 'action', 'action_is_pad', *CAMS)}
    with torch.inference_mode():
        proc = runner.pre(batch)
        # Exercise the same evaluation path used by train.evaluate and compare
        # to deployment predict_action_chunk. Evaluation must not use labels.
        captured = []
        hook = runner.policy.model.register_forward_hook(lambda _m, _i, out: captured.append(out[0]))
        try:
            runner.policy.forward(proc)
        finally:
            hook.remove()
        forward = captured[0]
        chunk = runner.policy.predict_action_chunk(proc)
        parity = float((forward - chunk).abs().max())
        physical = runner.post(chunk.reshape(-1, 16)).reshape(chunk.shape).cpu().numpy()
        roundtrip = runner.post(proc['action'].reshape(-1, 16)).reshape(proc['action'].shape)
        roundtrip_error = float((roundtrip.cpu() - batch['action']).abs().max())
    return physical, parity, roundtrip_error


class CapturingRunner:
    def __init__(self, runner):
        self.runner, self.chunk_size, self.records = runner, runner.chunk_size, []

    def reset(self):
        self.runner.reset()
        self.records = []

    def predict(self, images, state):
        prediction, elapsed = self.runner.predict(images, state)
        self.records.append((state.copy(), {k: v.copy() for k, v in images.items()}, prediction.copy()))
        return prediction, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--scenes', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--scene-id', action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--closed-seconds', type=float, default=12.)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('preserve existing results; choose a new output')
    if not 0 < args.closed_seconds <= 20:
        ap.error('this diagnostic is limited to 20 simulation seconds')
    frozen = json.loads(args.scenes.read_text())
    verify_scene_file(frozen)
    if frozen['split'] != 'validation':
        ap.error('keep final test held out')
    scenes = {s['id']: s for s in frozen['scenes']}
    if any(s not in scenes for s in args.scene_id):
        ap.error('unknown scene')
    args.output.mkdir(parents=True)
    runner = Runner(args.model)
    torch.set_num_threads(2)
    ds = EpisodeDataset(args.dataset / 'validation', domain='sim')
    train = EpisodeDataset(args.dataset / 'train', domain='sim', images=False)
    from scipy.spatial import cKDTree
    train_q = np.concatenate([e['q'][:,:7] for e in train.episodes])
    train_tree = cKDTree(train_q)
    report = dict(model=str(args.model.resolve()), model_sha256=digest(args.model / 'model.safetensors'),
                  source_sha256=digest(Path(__file__)), hardware_accessed=False,
                  note='Targeted development diagnosis, not an overall success benchmark. Fixed-image state interventions are synthetic sensitivity probes.',
                  cases=[])
    for identifier in args.scene_id:
        scene = scenes[identifier]
        source = recorded_path(scene['collection_root']) / scene['source_episode']
        raw = np.load(source / 'numeric.npz')
        assert digest(source / 'numeric.npz') == scene['numeric_sha256']
        candidates = [i for i, e in enumerate(ds.manifest['episodes'])
                      if recorded_path(e['collection_root']) / e['path'] == source]
        assert len(candidates) == 1
        e = ds.episodes[candidates[0]]
        start = int(ds.manifest['episodes'][candidates[0]]['converted_start_frame'])
        assert np.array_equal(e['q'], raw['state'][start:])
        assert np.array_equal(e['a'], raw['action'][start:])
        phases = raw['phase'][start:]
        boundaries = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1, len(phases)]
        frames = sorted(set(range(0, e['length'], 20)) | {int(x) for x in boundaries[:-1]}
                        | {int((a+b-1)//2) for a,b in zip(boundaries[:-1], boundaries[1:])})
        items, teacher, checks = [], [], []
        for frame in frames:
            item = ds[e['offset'] + frame]
            future = frame + np.arange(100)
            assert np.array_equal(item['action'].numpy(), e['a'][np.minimum(future, e['length']-1)])
            assert np.array_equal(item['action_is_pad'].numpy(), future >= e['length'])
            for key, image in images_of(item).items():
                expected = np.asarray(Image.open(source / 'cameras' / key / f'{frame+start:06d}.jpg').convert('RGB'))
                assert np.array_equal(image, expected)
            items.append(item)
        for offset in range(0, len(items), 8):
            prediction, parity, roundtrip = infer_batch(runner, items[offset:offset+8])
            checks.append(dict(forward_predict_max_abs=parity, normalization_roundtrip_max_rad=roundtrip))
            for sample, pred in zip(items[offset:offset+8], prediction):
                f = int(sample['frame_index'])
                row = dict(frame=f, raw_frame=f+start, phase=str(phases[f]), horizons={})
                for h in (0, 10, 49, 99):
                    if f+h < e['length']:
                        row['horizons'][str(h)] = dict(**errors(pred[h:h+1], e['a'][f+h:f+h+1]),
                            prediction_right=pred[h,:8].tolist(), reference_right=e['a'][f+h,:8].tolist())
                teacher.append(row)
        probes = []
        # Each synthetic probe holds the recorded RGB images fixed. It does not
        # imply the intervened state/image pair is physically realizable.
        for a,b in zip(boundaries[:-1], boundaries[1:]):
            frame = int((a+b-1)//2)
            item = ds[e['offset']+frame]
            images, state = images_of(item), item['observation.state'].numpy()
            baseline, _ = runner.predict(images, state)
            swapped = images_of(ds[e['offset']])
            variants = [('both_black', {k: np.zeros_like(v) for k,v in images.items()}, state),
                        ('recorded_start_images', swapped, state)]
            for joint in (4,5):
                for degrees in (-2.,2.):
                    q = state.copy(); q[joint] += np.deg2rad(degrees)
                    variants.append((f'J{joint+1}_{degrees:+g}deg_fixed_images', images, q))
            for name, rgb, q in variants:
                pred, _ = runner.predict(rgb,q)
                probes.append(dict(frame=frame,phase=str(phases[frame]),variant=name,
                                   h0_change=errors(pred[:1], baseline[:1]),
                                   h49_change=errors(pred[49:50], baseline[49:50])))
        case = dict(scene=identifier, source=str(source), start_frame=start,
                    raw_converted_numeric_exact=True, sampled_images_exact=True,
                    sampled_future_labels_and_padding_exact=True, checks=checks,
                    teacher_forced=teacher, input_sensitivity=probes)
        pose_frames = [int(a+(b-a)//2)+start for a,b in zip(boundaries[:-1],boundaries[1:])
                       if str(phases[a]) in ('PREGRASP','RECOVERY_PREGRASP','APPROACH','CLOSE')]
        case['consistent_pose_sensitivity'] = consistent_pose_probes(runner,scene,raw,pose_frames)
        capture = CapturingRunner(runner)
        rollout_args = argparse.Namespace(command_envelope='deployment', exec_steps=50,
            max_seconds=args.closed_seconds,max_chunks=20,observation_wait_s=1/30)
        rollout = run_scene(capture,load_config(),recorded_initialization(scene),Path('.'),rollout_args)
        (args.output / f'{identifier}_short_rollout.json').write_text(json.dumps(rollout,indent=2)+'\n')
        closed = []
        states, predictions = [], []
        for i,(q,images,pred) in enumerate(capture.records):
            distances = np.abs(e['q'][:,:7]-q[:7]).mean(axis=1)
            nearest = int(distances.argmin())
            delta = np.rad2deg(q[:7]-e['q'][nearest,:7])
            train_distance, train_index = train_tree.query(q[:7],p=1)
            train_episode, train_frame = train.locate(int(train_index))
            closed.append(dict(block=i,state_right=q[:8].tolist(),nearest_reference_frame=nearest,
                nearest_phase=str(phases[nearest]),nearest_arm_mae_deg=float(np.rad2deg(distances[nearest])),
                nearest_sim_train_arm_mae_deg=float(np.rad2deg(train_distance/7)),
                nearest_sim_train_source=train_episode['source'], nearest_sim_train_frame=train_frame,
                nearest_joint_delta_deg=delta.tolist(),h0_right=pred[0,:8].tolist(),h49_right=pred[49,:8].tolist()))
            states.append(q);predictions.append(pred)
            if i in (0,len(capture.records)-1):
                for role,image in images.items():
                    Image.fromarray(image).save(args.output / f'{identifier}_block{i}_{role}.png')
        np.savez_compressed(args.output / f'{identifier}_short_predictions.npz',
                            states=np.asarray(states),predictions=np.asarray(predictions))
        case['closed_loop'] = closed
        report['cases'].append(case)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('DIAGNOSED',identifier,'teacher_frames',len(teacher),'closed_blocks',len(closed),flush=True)
    print('REPORT',args.output/'report.json',flush=True)


if __name__ == '__main__':
    main()
