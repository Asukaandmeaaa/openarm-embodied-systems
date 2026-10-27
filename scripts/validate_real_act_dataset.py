#!/usr/bin/env python3
"""Check pilot LeRobot data against raw source, including ACT terminal padding."""
from pathlib import Path
import json
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from convert_real_episodes_act import ROOT, prepare, CAMERAS


def main():
    report = json.loads((ROOT/'results/act_real_fixed13/conversion_report.json').read_text())
    dataset = LeRobotDataset('local/openarm_real_fixed13_20260930',root=report['output'],
                            delta_timestamps={'action':[i/30 for i in range(50)]})
    assert dataset.num_episodes == 13
    assert len(dataset) == report['total_frames']
    offset, checked = 0, 0
    for ep in report['episodes']:
        path = Path(report['source'])/ep['source_episode']
        streams,q,ids,_ = prepare(path,report['fps'])
        count = ep['frame_count']
        for i in sorted(set([0,1,count//2,count-2,count-1,*range(0,count,200)])):
            row = dataset[offset+i]
            np.testing.assert_array_equal(row['observation.state'].numpy(),q['joint_observation'][ids['joint_observation'][i]])
            expected = np.stack([q['demonstration_action'][ids['demonstration_action'][min(i+j,count-1)]] for j in range(50)])
            np.testing.assert_array_equal(row['action'].numpy(),expected)
            np.testing.assert_array_equal(row['action_is_pad'].numpy(),np.arange(50)+i>=count)
            assert int(row['episode_index']) == ep['episode_index']
            for camera in CAMERAS:
                image = row[f'observation.images.{camera}']
                assert image.shape == (3,360,640)
                assert torch.isfinite(image).all() and float(image.min())>=0 and float(image.max())<=1
            checked += 1
        offset += count
    result = {'passed':True,'episodes':13,'frames':len(dataset),'source_comparison_samples':checked,
              'checks':['joint name ordering','raw measured state equality','raw action chunk equality','episode boundary padding','both decoded RGB images'],
              'scope':'Data integrity only; no held-out or real-world policy success evaluation.'}
    (ROOT/'results/act_real_fixed13/dataset_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
