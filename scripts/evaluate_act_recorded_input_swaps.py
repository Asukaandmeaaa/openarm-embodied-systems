#!/usr/bin/env python3
"""Synthetic input sensitivity audit. Stored data only; never controls hardware."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from evaluate_act_recorded_comparison import dataset_observation, load_server


def replace(observation, changes):
    arrays = []
    for field in observation.type:
        value = changes.get(field.name, observation[field.name].as_py())
        arrays.append(pa.array([value], type=field.type))
    return pa.StructArray.from_arrays(arrays, fields=list(observation.type))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.comparison/'summary.json').read_text())
    rows = [json.loads(l) for l in (args.comparison/'predictions.jsonl').open()]
    ranking = json.loads((args.comparison/'state_and_onset_comparison.json').read_text())['nearest_demonstration_initial_states']
    closest = [r['episode'] for r in ranking[:3]]
    selected = [r for r in rows if r['kind']=='demonstration' and
                ((r['episode'] in closest and 'before_onset' in r['phases']) or
                 (r['episode']==closest[0] and any(p in r['phases'] for p in ('initial','onset'))))]
    live = next(r for r in rows if r['kind']=='recorded_live')
    with pa.memory_map(live['source'], 'r') as stream:
        live_obs = pa.ipc.open_file(stream).get_batch(0).to_struct_array()[0]
    live_q = np.asarray(live['state'])
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(1000)
    checkpoint = summary['checkpoint']
    policy = ACTPolicy.from_pretrained(checkpoint, device='cpu').to('cpu').eval()
    policy.config.device = 'cpu'
    pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
                                        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    server = load_server(); server.DEBUG_OBSERVATIONS = False
    server.IMAGE_SIZES.update({k: tuple(v.shape[-2:]) for k,v in policy.config.input_features.items() if k.startswith('observation.images.')})
    results = []
    for row in selected:
        table = pq.read_table(row['source'], columns=['observation.images.ceiling','observation.images.wrist_right'])
        q = np.asarray(row['state'])
        demo, _ = dataset_observation(table, row['file_row'], q)
        left_swap = q.copy(); left_swap[8:] = live_q[8:]
        ceiling = live_obs['camera_ceiling'].as_py()
        wrist = live_obs['camera_wrist_right'].as_py()
        variants = {'original_demo': {}, 'live_state_only': {'position': live_q.tolist()},
                    'live_left_state_only': {'position': left_swap.tolist()},
                    'live_images_only': {'camera_ceiling': ceiling,'camera_wrist_right': wrist},
                    'live_ceiling_only': {'camera_ceiling': ceiling},
                    'live_wrist_only': {'camera_wrist_right': wrist}}
        original = np.asarray(row['prediction'])
        for variant, changes in variants.items():
            obs = replace(demo, changes)
            state = np.asarray(obs['position'].as_py())
            policy.reset(); pre.reset(); post.reset()
            with torch.inference_mode():
                pred = np.asarray(server.infer(policy, pre, post, obs)['positions'])
            assert pred.shape==(50,16) and np.isfinite(pred).all()
            results.append({'episode': row['episode'], 'frame': row['frame'], 'phases': row['phases'],
                            'variant': variant, 'prediction': pred.tolist(),
                            'near10_arm_mean_displacement_from_input_state_rad': float(np.abs(pred[:10,:7]-state[:7]).mean()),
                            'full50_arm_mean_displacement_from_input_state_rad': float(np.abs(pred[:,:7]-state[:7]).mean()),
                            'near10_arm_mae_vs_original_prediction_rad': float(np.abs(pred[:10,:7]-original[:10,:7]).mean()),
                            'full50_arm_mae_vs_original_prediction_rad': float(np.abs(pred[:,:7]-original[:,:7]).mean())})
        print(f"SWAP_PROGRESS episode={row['episode']} frame={row['frame']} variants=6", flush=True)
    report={'scope':'Synthetic input swaps measure sensitivity, NOT valid robot observations, causal proof or closed-loop performance.',
            'selection':'Initial-state nearest three training episodes by right J1-J7 RMS distance; before_onset in all three, initial and onset in nearest episode.',
            'selected_episode_order':closest,'live_source':live['source'],
            'comparison_summary':str(args.comparison/'summary.json'), 'physical_operations':False,'results':results}
    with (args.comparison/'input_swap_sensitivity.json').open('x') as out:
        json.dump(report,out,indent=2);out.write('\n')
    print('INPUT_SWAP_COMPLETE',flush=True)


if __name__=='__main__':
    main()
