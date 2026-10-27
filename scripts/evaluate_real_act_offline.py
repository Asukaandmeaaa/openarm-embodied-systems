#!/usr/bin/env python3
"""Teacher-forced action error on training episodes, NOT closed-loop success."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--samples-per-episode',type=int,default=20)
    parser.add_argument('--report',type=Path,default=ROOT/'results/act_real_fixed13/offline_training_error.json')
    args=parser.parse_args()
    if args.samples_per_episode < 1:
        parser.error('samples-per-episode must be positive')
    manifest=json.loads((ROOT/'results/act_real_fixed13/conversion_report.json').read_text())
    policy=ACTPolicy.from_pretrained(str(args.checkpoint)).eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(args.checkpoint))
    ds=LeRobotDataset('local/openarm_real_fixed13_20260930',root=manifest['output'])
    errors=[]
    per_episode=[]
    offset=0
    with torch.inference_mode():
        for ep in manifest['episodes']:
            ep_errors=[]
            for i in np.unique(np.linspace(0,ep['frame_count']-1,args.samples_per_episode,dtype=int)):
                row=ds[offset+int(i)]
                obs={k:v for k,v in row.items() if k.startswith('observation.')}
                policy.reset()
                pred=post(policy.select_action(pre(obs))).cpu().reshape(-1)
                if pred.shape != (16,) or not torch.isfinite(pred).all():
                    raise ValueError('Invalid policy output')
                ep_errors.append((pred-row['action'].cpu()).abs().numpy())
            error=np.stack(ep_errors)
            errors.append(error)
            per_episode.append({'episode':ep['source_episode'],'right_arm_mae_rad':float(error[:,:7].mean()),'right_gripper_mae_rad':float(error[:,7].mean())})
            offset+=ep['frame_count']
    error=np.concatenate(errors)
    result={'checkpoint':str(args.checkpoint.resolve()),'scope':'Teacher-forced first-action error on TRAINING episodes. Not held-out accuracy or closed-loop success.',
            'samples':len(error),'joint_order':manifest['joint_order'],'per_joint_mae_rad':error.mean(axis=0).tolist(),
            'right_arm_mae_rad':float(error[:,:7].mean()),'right_gripper_mae_rad':float(error[:,7].mean()),'episodes':per_episode}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
