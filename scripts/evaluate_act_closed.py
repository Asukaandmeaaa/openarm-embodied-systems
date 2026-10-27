#!/usr/bin/env python3
"""Complete held-out offline ACT inference in radians, grouped by task phase.

Stored RGB/state only. No ROS, camera, CAN, gateway or physical success claim.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vision_grasp_baseline.collection_io import write_json


def policy_observation(batch,camera_keys):
    import torch
    observed={k:v.clone() for k,v in batch.items() if k.startswith('observation.')}
    # Installed image datasets already yield float RGB in [0,1]; return_uint8
    # affects video decoding only. Never divide those float images twice.
    for key in camera_keys:
        if observed[key].dtype==torch.uint8:observed[key]=observed[key].float()/255.
    return observed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from vision_grasp_baseline.selected_column_reader import enable_selected_columns
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    saved=json.loads((args.checkpoint/'train_config.json').read_text())
    data=Path(saved['dataset']['root']).parent
    ready=json.loads((data/'READY.json').read_text())
    source=json.loads((data/'validation/meta/visual_expert_manifest.json').read_text())
    cfg=ACTConfig.from_pretrained(str(args.checkpoint));cfg.device=args.device
    policy=ACTPolicy.from_pretrained(str(args.checkpoint),config=cfg).to(args.device).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor':{'device':args.device}})
    ds=LeRobotDataset('local/validation',root=data/'validation',
        delta_timestamps={'action':[i/30 for i in range(cfg.chunk_size)]},return_uint8=True)
    enable_selected_columns(ds)
    phases={}
    for i,row in enumerate(source['episodes']):
        phases[i]=[phase for phase,count in row['phase_coverage']['frame_counts'].items() for _ in range(count)]
        if len(phases[i])!=row['frame_count']:raise ValueError('Phase coverage/frame mismatch')
    totals={};frames=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=0,shuffle=False):
            observed=policy_observation(batch,ds.meta.camera_keys)
            prediction=policy.predict_action_chunk(pre(observed))
            prediction=post(prediction.reshape(-1,16)).reshape(-1,cfg.chunk_size,16).cpu().numpy()
            truth=batch['action'].numpy();state=batch['observation.state'].numpy()
            mask=~batch['action_is_pad'].numpy()
            for i in range(len(truth)):
                phase=phases[int(batch['episode_index'][i])][int(batch['frame_index'][i])]
                for label in ('all',phase):
                    item=totals.setdefault(label,dict(frames=0,horizons={}))
                    item['frames']+=1
                    for horizon in (1,10,cfg.chunk_size):
                        valid=mask[i,:horizon]
                        error=abs(prediction[i,:horizon]-truth[i,:horizon])[valid]
                        hold=abs(state[i]-truth[i,:horizon])[valid]
                        h=item['horizons'].setdefault(str(horizon),dict(valid_steps=0,
                            arm_abs_sum=0.,gripper_abs_sum=0.,arm_hold_abs_sum=0.))
                        h['valid_steps']+=int(valid.sum());h['arm_abs_sum']+=float(error[:,:7].sum())
                        h['gripper_abs_sum']+=float(error[:,7].sum());h['arm_hold_abs_sum']+=float(hold[:,:7].sum())
                frames+=1
            if frames%512<len(truth):print(f'OFFLINE_FULL_VALIDATION {frames}/{len(ds)}',flush=True)
    if frames!=ready['frames']['validation']:raise ValueError('Incomplete validation coverage')
    for item in totals.values():
        for h in item['horizons'].values():
            n=max(h['valid_steps'],1)
            h['right_arm_mae_rad']=h.pop('arm_abs_sum')/(n*7)
            h['right_gripper_mae_rad']=h.pop('gripper_abs_sum')/n
            h['hold_current_arm_baseline_mae_rad']=h.pop('arm_hold_abs_sum')/(n*7)
    write_json(args.output,dict(hardware_io=False,checkpoint=str(args.checkpoint.resolve()),
        validation_frames=frames,scope='Full held-out teacher-forced offline inference; NOT physical success rate',
        phase_assignment='phase of current observation; future action chunk can cross phase boundaries',metrics=totals))
    print(json.dumps(dict(report=str(args.output),frames=frames,all=totals['all']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
