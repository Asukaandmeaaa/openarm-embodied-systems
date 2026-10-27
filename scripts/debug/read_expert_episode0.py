#!/usr/bin/env python3
"""Read expert dataset episode 0, frame 0: observation.state and action."""

import pandas as pd
import numpy as np
from pathlib import Path

dataset_path = Path("/home/lushilin/openarm_vla_ws/data/lerobot/openarm_cube_pick_expert_pilot_v3")
parquet_file = dataset_path / "data/chunk-000/file-000.parquet"

df = pd.read_parquet(parquet_file)

# Filter episode 0
ep0 = df[df['episode_index'] == 0].sort_values('frame_index')

if len(ep0) == 0:
    print("ERROR: No frames found for episode_index=0")
    exit(1)

first_frame = ep0.iloc[0]

print("="*70)
print("EXPERT DATASET - EPISODE 0, FRAME 0")
print("="*70)

# Extract observation.state (16 DOF)
state_cols = [c for c in df.columns if c.startswith('observation.state')]
state_cols_sorted = sorted(state_cols, key=lambda x: int(x.split('_')[-1]))

obs_state = first_frame[state_cols_sorted].values
print(f"\nobservation.state (frame 0):")
print(f"  Shape: {obs_state.shape}")
print(f"  Values: {obs_state}")
print(f"  Right arm (j0-j6+gripper): {obs_state[:8]}")
print(f"  Left arm (j0-j6+gripper): {obs_state[8:]}")

# Extract action (16 DOF)
action_cols = [c for c in df.columns if c.startswith('action_')]
action_cols_sorted = sorted(action_cols, key=lambda x: int(x.split('_')[-1]))

action = first_frame[action_cols_sorted].values
print(f"\naction (frame 0):")
print(f"  Shape: {action.shape}")
print(f"  Values: {action}")
print(f"  Right arm (j0-j6+gripper): {action[:8]}")
print(f"  Left arm (j0-j6+gripper): {action[8:]}")

# Delta
delta = action - obs_state
print(f"\naction - observation.state:")
print(f"  Delta: {delta}")
print(f"  Right arm delta: {delta[:8]}")
print(f"  ||delta||: {np.linalg.norm(delta):.6f}")

print(f"\n" + "="*70)
print("COMPARISON WITH STABILIZE")
print("="*70)
print("\nExpected comparison (after running test_stabilize_debug.py):")
print("  1. observation.state[frame0] should match stabilize final_state")
print("  2. action[frame0] should be close to observation.state[frame0]")
print("  3. If stabilize didn't complete, final_state will differ significantly")
