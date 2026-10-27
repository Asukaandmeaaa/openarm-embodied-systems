#!/usr/bin/env python3
"""Debug ACT policy output to diagnose rollout failure."""

import sys
from pathlib import Path
import torch
import numpy as np
import yaml

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT / 'official/openarm_mujoco/src')]

from lerobot.policies import get_policy_class, make_pre_post_processors

# Load checkpoint
checkpoint = Path('outputs/act_openarm_cube_pick_smoke/checkpoints/000200')
policy = get_policy_class('act').from_pretrained(checkpoint).to('cuda').eval()
pre, post = make_pre_post_processors(
    policy.config,
    pretrained_path=checkpoint,
    preprocessor_overrides={'device_processor': {'device': 'cuda'}},
    postprocessor_overrides={'device_processor': {'device': 'cpu'}}
)

# Load stats to get observation mean
import json
stats = json.load(open('data/lerobot/openarm_cube_pick_expert_pilot_v3/meta/stats.json'))
obs_mean = np.array(stats['observation.state']['mean'])

print('[OBSERVATION.STATE MEAN]')
print('Right arm (j0-j6+gripper):', obs_mean[:8])
print('Left arm (j0-j6+gripper):', obs_mean[8:])
print()

# Create a dummy observation at the mean state
dummy_state = torch.from_numpy(obs_mean).unsqueeze(0).float()
dummy_images_ceiling = torch.zeros(1, 3, 600, 960)
dummy_images_wrist = torch.zeros(1, 3, 600, 960)

batch = {
    'observation.state': dummy_state,
    'observation.images.ceiling': dummy_images_ceiling,
    'observation.images.wrist_right': dummy_images_wrist
}

# Get policy output
with torch.inference_mode():
    actions = post(policy.predict_action_chunk(pre(batch))).numpy()[0]

print(f'[POLICY OUTPUT] Shape: {actions.shape}')
print()
print('[ACTION[0]] First action in chunk:')
print('Right arm (j0-j6+gripper):', actions[0, :8])
print('Left arm (j0-j6+gripper):', actions[0, 8:])
print()
print('[ACTION[-1]] Last action in chunk:')
print('Right arm (j0-j6+gripper):', actions[-1, :8])
print('Left arm (j0-j6+gripper):', actions[-1, 8:])
print()

# Calculate deltas
delta_0 = actions[0] - obs_mean
delta_last = actions[-1] - obs_mean

print('[DELTA] action[0] - observation.state:')
print('Right arm (j0-j6+gripper):', delta_0[:8])
print('Left arm (j0-j6+gripper):', delta_0[8:])
print()
print('[DELTA] action[-1] - observation.state:')
print('Right arm (j0-j6+gripper):', delta_last[:8])
print('Left arm (j0-j6+gripper):', delta_last[8:])
print()

# Check if deltas exceed max_step limits
cfg = yaml.safe_load((ROOT/'configs/collection/openarm_cube_pick_expert_params.yaml').read_text())
max_step = np.array(cfg['guard']['max_step'])
max_velocity = np.array(cfg['guard']['max_velocity'])

print('[GUARD LIMITS]')
print('max_step:', max_step)
print('max_velocity:', max_velocity)
print()

# Assume 30Hz control (33ms per action)
control_dt = 0.033
required_velocity_0 = np.abs(delta_0) / control_dt
required_velocity_last = np.abs(delta_last) / control_dt

print('[REQUIRED VELOCITY] for action[0] at 30Hz:')
print('Right arm:', required_velocity_0[:8])
print('Exceeds limit?', required_velocity_0[:8] > max_velocity[:8])
print()
print('[REQUIRED VELOCITY] for action[-1] at 30Hz:')
print('Right arm:', required_velocity_last[:8])
print('Exceeds limit?', required_velocity_last[:8] > max_velocity[:8])
print()

print('[MAX ABSOLUTE DELTA]')
print('action[0]:', np.max(np.abs(delta_0)))
print('action[-1]:', np.max(np.abs(delta_last)))
print('max_step limit:', np.min(max_step))
