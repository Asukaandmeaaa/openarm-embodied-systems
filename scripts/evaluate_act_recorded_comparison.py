#!/usr/bin/env python3
"""Offline ACT comparison using stored demonstration and trial observations only.

No ROS, Dora node, camera, CAN, socket, or robot connection is created. This is
teacher-forced TRAINING-data evaluation, not a held-out or closed-loop test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {'ceiling': 'camera_ceiling', 'wrist_right': 'camera_wrist_right'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_server():
    path = ROOT / 'official/dora-openarm-evaluation/src/local_policy_server.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('_offline_act_server', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_observation(table, index, state):
    row = {'id': index, 'position': state.tolist()}
    images = {}
    for role, field in CAMERAS.items():
        value = table[f'observation.images.{role}'][index].as_py()
        if not value['bytes']:
            raise ValueError('Expected embedded dataset images')
        image = np.asarray(Image.open(io.BytesIO(value['bytes'])).convert('RGB'))
        if image.shape != (360, 640, 3):
            raise ValueError(f'Unexpected stored image shape: {image.shape}')
        images[role] = image
        row[field] = image.reshape(-1)
    fields = [pa.array([row['id']], type=pa.int64()),
              pa.array([row['position']], type=pa.list_(pa.float32()))]
    names = ['id', 'position']
    for field in CAMERAS.values():
        fields.append(pa.array([row[field]], type=pa.list_(pa.uint8())))
        names.append(field)
    return pa.StructArray.from_arrays(fields, names=names)[0], images


def metrics(pred, truth, state):
    result = {}
    for n in (1, 10, 50):
        error = np.abs(pred[:n] - truth[:n])
        hold = np.abs(state - truth[:n])
        result[str(n)] = {
            'arm_mae_rad': float(error[:, :7].mean()),
            'gripper_mae_rad': float(error[:, 7].mean()),
            'arm_hold_baseline_mae_rad': float(hold[:, :7].mean()),
            'gripper_hold_baseline_mae_rad': float(hold[:, 7].mean()),
            'arm_prediction_displacement_rad': float(np.abs(pred[:n, :7] - state[:7]).mean()),
            'arm_truth_displacement_rad': float(hold[:, :7].mean()),
            'per_joint_mae_rad': error.mean(axis=0).tolist(),
        }
    return result


def aggregate(rows):
    result = {'samples': len(rows)}
    for n in ('1', '10', '50'):
        result[n] = {k: np.mean([r['metrics'][n][k] for r in rows], axis=0).tolist()
                     for k in rows[0]['metrics'][n]} if rows else {}
        if rows:
            result[n]['fraction_beating_arm_hold_baseline'] = float(np.mean([
                r['metrics'][n]['arm_mae_rad'] < r['metrics'][n]['arm_hold_baseline_mae_rad'] for r in rows]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1000)
    cfg = json.loads((args.checkpoint / 'train_config.json').read_text())
    dataset = Path(cfg['dataset']['root'])
    info = json.loads((dataset / 'meta/info.json').read_text())
    assert info['fps'] == 30 and info['features']['action']['names'] == info['features']['observation.state']['names']
    # CPU deliberately avoids competing with the resident physical policy GPU.
    print('OFFLINE_ONLY: CPU inference, stored inputs, no control connection', flush=True)
    policy = ACTPolicy.from_pretrained(str(args.checkpoint), device='cpu').to('cpu').eval()
    policy.config.device = 'cpu'
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    server = load_server()
    server.DEBUG_OBSERVATIONS = False
    for key, feature in policy.config.input_features.items():
        if key.startswith('observation.images.'):
            server.IMAGE_SIZES[key] = tuple(feature.shape[-2:])

    def predict(observation):
        policy.reset(); pre.reset(); post.reset()
        with torch.inference_mode():
            out = np.asarray(server.infer(policy, pre, post, observation)['positions'])
        assert out.shape == (50, 16) and np.isfinite(out).all()
        return out

    # Read only compact state/action columns first; image columns are read per file.
    columns = ['observation.state', 'action', 'episode_index', 'frame_index', 'index']
    episode_data, file_tables = {}, {}
    for file in sorted((dataset / 'data').rglob('*.parquet')):
        tab = pq.read_table(file, columns=columns)
        file_tables[file] = tab
        ep_ids = np.asarray(tab['episode_index'])
        for episode in np.unique(ep_ids):
            indices = np.flatnonzero(ep_ids == episode)
            records = episode_data.setdefault(int(episode), [])
            for i in indices:
                records.append((int(tab['frame_index'][int(i)].as_py()), file, int(i),
                                tab['observation.state'][int(i)].as_py(), tab['action'][int(i)].as_py()))
    jobs, selections = {}, []
    for episode, records in sorted(episode_data.items()):
        records.sort()
        assert [r[0] for r in records] == list(range(len(records)))
        q = np.array([r[3] for r in records])
        a = np.array([r[4] for r in records])
        assert np.isfinite(q).all() and np.isfinite(a).all()
        last = len(q) - 50
        if last < 1:
            continue
        # This 0.03 rad threshold selects diagnostic movement, not a robot limit.
        baseline = np.median(q[:min(15, len(q))], axis=0)
        moved = np.max(np.abs(q[:, :7] - baseline[:7]), axis=1) > .03
        onset_candidates = np.flatnonzero(np.convolve(moved.astype(int), np.ones(5, dtype=int), mode='valid') == 5)
        onset = int(onset_candidates[0]) if len(onset_candidates) else 0
        excursion = np.max(np.abs(a[49:, :7] - q[:-49, :7]), axis=1)
        peak = int(np.argmax(excursion))
        wanted = [('initial', 0), ('before_onset', max(0, onset-15)), ('onset', onset),
                  ('after_onset_0.5s', onset+15), ('after_onset_1s', onset+30),
                  ('largest_49step_displacement', peak), ('middle', len(q)//2)]
        unique = {}
        for phase, frame in wanted:
            frame = min(last, frame)
            unique.setdefault(frame, []).append(phase)
        selections.append({'episode': episode, 'frames': len(q), 'onset_frame': onset,
                           'selection_threshold_rad': .03, 'selection_persistence_frames': 5})
        for frame, phases in unique.items():
            _, file, index, state, _ = records[frame]
            jobs.setdefault(file, []).append({'episode': episode, 'frame': frame, 'phases': phases,
                                             'file_row': index, 'state': state, 'truth': a[frame:frame+50]})
    rows, examples = [], []
    with (args.output/'predictions.jsonl').open('x') as log:
        for file, samples in sorted(jobs.items()):
            images = pq.read_table(file, columns=[f'observation.images.{r}' for r in CAMERAS])
            for sample in samples:
                state = np.asarray(sample['state'])
                observation, imgs = dataset_observation(images, sample['file_row'], state)
                pred = predict(observation)
                row = {k: v for k, v in sample.items() if k != 'truth'}
                row.update(kind='demonstration', source=str(file), prediction=pred.tolist(),
                           truth=sample['truth'].tolist(), metrics=metrics(pred, sample['truth'], state))
                log.write(json.dumps(row)+'\n'); log.flush()
                rows.append(row)
                if sample['episode'] in (0, 13, 35) and 'onset' in sample['phases']:
                    examples.append((f"demo ep{sample['episode']} onset", imgs))
            print(f'DEMO_PROGRESS samples={len(rows)} file={file.name}', flush=True)
            del images

        result = json.loads((args.trial/'result.json').read_text())
        end = result['monotonic_ns']
        start = end - int(result['execution_profile']['duration_s'] * 1e9)
        live = []
        for file in sorted(args.trial.glob('policy_*.jsonl')):
            for line in file.open():
                row = json.loads(line)
                if row['event'] == 'raw_policy_response' and start <= row['monotonic_ns'] <= end:
                    live.append(row)
        live.sort(key=lambda r: r['monotonic_ns'])
        assert live, 'No recorded policy responses in active interval'
        live_rows = []
        for i in np.unique(np.linspace(0, len(live)-1, 12, dtype=int)):
            recorded = live[int(i)]
            path = Path(recorded['observation_arrow_path'])
            assert sha(path) == recorded['observation_arrow_sha256']
            with pa.memory_map(str(path), 'r') as stream:
                observation = pa.ipc.open_file(stream).get_batch(0).to_struct_array()[0]
            state = np.asarray(observation['position'].as_py())
            pred = predict(observation)
            original = np.asarray(recorded['full_postprocessed_actions'])
            row = {'kind': 'recorded_live', 'source': str(path), 'state': state.tolist(),
                   'prediction': pred.tolist(), 'recorded_prediction': original.tolist(),
                   'rerun_vs_recorded_max_abs_rad': float(np.max(np.abs(pred-original))),
                   'near10_arm_displacement_from_state_rad': float(np.abs(pred[:10, :7]-state[:7]).mean()),
                   'near10_arm_internal_range_rad': float(np.max(np.ptp(pred[:10, :7], axis=0))),
                   'step49_arm_max_displacement_from_state_rad': float(np.max(np.abs(pred[49, :7]-state[:7])))}
            log.write(json.dumps(row)+'\n'); log.flush()
            live_rows.append(row)
            if len(live_rows) == 1:
                imgs = {r: np.array(observation[k].as_py(), dtype=np.uint8).reshape(360, 640, 3)
                        for r, k in CAMERAS.items()}
                examples.append(('recorded live START', imgs))
            print(f'LIVE_PROGRESS samples={len(live_rows)} max_rerun_difference={row["rerun_vs_recorded_max_abs_rad"]:.8f}', flush=True)

    groups = {phase: aggregate([r for r in rows if phase in r['phases']])
              for phase in sorted(set(p for r in rows for p in r['phases']))}
    summary = {'scope': 'TRAINING episodes, teacher-forced prediction; no held-out or closed-loop success claim',
               'checkpoint': str(args.checkpoint.resolve()), 'dataset': str(dataset),
               'source_sha256': {str(p): sha(p) for p in [Path(__file__), args.checkpoint/'model.safetensors',
                                  args.checkpoint/'config.json', args.checkpoint/'train_config.json']},
               'joint_order': info['features']['action']['names'], 'device': 'cpu',
               'frame_alignment': 'prediction[k] versus recorded action[frame+k], no cross-episode padding',
               'selection': selections, 'all_demonstration_samples': aggregate(rows), 'by_phase': groups,
               'live_samples': len(live_rows),
               'live_rerun_max_abs_difference_rad': max(r['rerun_vs_recorded_max_abs_rad'] for r in live_rows),
               'live_near10_arm_displacement_from_state_rad_mean': float(np.mean([r['near10_arm_displacement_from_state_rad'] for r in live_rows])),
               'physical_operations': False, 'runtime_configuration_changes': False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(examples), 2, figsize=(12, 3.5*len(examples)), squeeze=False)
        for i, (name, imgs) in enumerate(examples):
            for j, role in enumerate(CAMERAS):
                axes[i,j].imshow(imgs[role]); axes[i,j].set_title(f'{name}: {role}'); axes[i,j].axis('off')
        fig.tight_layout(); fig.savefig(args.output/'input_comparison.png', dpi=120); plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; numerical report complete', flush=True)
    print('COMPLETE '+str(args.output/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
