#!/usr/bin/env python3
"""Diagnose action space mismatch between training and inference."""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load training dataset statistics
train_stats_path = ROOT / "data/lerobot/openarm_cube_pick_expert_pilot_v3/meta/stats.json"
with open(train_stats_path) as f:
    train_stats = json.load(f)

# Load guard configuration
import yaml
guard_config_path = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
with open(guard_config_path) as f:
    guard_cfg = yaml.safe_load(f)

guard = guard_cfg["guard"]

print("=" * 80)
print("ACTION SPACE DIAGNOSIS: Training Data vs. Safety Guards")
print("=" * 80)
print()

# Extract action statistics from training data
action_mean = np.array(train_stats["action"]["mean"])
action_std = np.array(train_stats["action"]["std"])
action_min = np.array(train_stats["action"]["min"])
action_max = np.array(train_stats["action"]["max"])
action_q01 = np.array(train_stats["action"]["q01"])
action_q99 = np.array(train_stats["action"]["q99"])

max_velocity = np.array(guard["max_velocity"])
max_acceleration = np.array(guard["max_acceleration"])
max_step = np.array(guard["max_step"])

joint_names = [
    "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
    "left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper"
]

print("Key Issue: ACT outputs POSITIONS, but guards expect VELOCITIES")
print("-" * 80)
print()

print("Training Data Action Statistics (POSITIONS in radians):")
print("-" * 80)
for i, name in enumerate(joint_names):
    print(f"{name:20s} | mean={action_mean[i]:7.3f} | std={action_std[i]:6.3f} | range=[{action_min[i]:7.3f}, {action_max[i]:7.3f}]")
print()

print("Safety Guard Limits (configured for VELOCITIES in rad/s):")
print("-" * 80)
for i, name in enumerate(joint_names):
    print(f"{name:20s} | max_vel={max_velocity[i]:6.3f} | max_accel={max_acceleration[i]:6.3f} | max_step={max_step[i]:7.4f}")
print()

print("THE MISMATCH:")
print("-" * 80)
print("1. ACT policy outputs TARGET POSITIONS (e.g., right_j1 ~ 2.06 rad)")
print("2. ActionGuard expects POSITION CHANGES (velocities in rad/s or rad/step)")
print("3. When policy outputs absolute positions, guards clip them as if they were velocities")
print()
print("Example from rollout:")
print("  - Policy might output: right_j1 = 2.06 rad (target position)")
print("  - Guard interprets it as: Δposition = 2.06 rad/step")
print("  - Guard clips to max_step[1] = 0.003 rad")
print("  - Result: massive clipping (45,162 interventions)")
print()

print("SOLUTION OPTIONS:")
print("-" * 80)
print("Option 1: Convert policy positions to velocity commands")
print("  Compute: velocity = (target_position - current_position) / dt")
print("  Then apply guards to velocity")
print()
print("Option 2: Use different guards for position-based policies")
print("  Check if current position is near target, allow larger steps if far")
print()
print("Option 3: Check if policy should output deltas instead of absolutes")
print("  Review ACT training: was it trained on position deltas or absolute positions?")
print()

# Compute hypothetical velocity if treating positions as deltas
print("Verification: If policy outputs were DELTAS (Δposition per 33ms):")
print("-" * 80)
dt = 0.033  # 33ms per action execution
hypothetical_velocities = action_std / dt

print("Expected velocity std (if actions were deltas):")
for i, name in enumerate(joint_names):
    ratio = hypothetical_velocities[i] / max_velocity[i]
    status = "✓ OK" if ratio < 0.5 else "✗ EXCEEDS"
    print(f"{name:20s} | std_vel={hypothetical_velocities[i]:6.3f} rad/s | limit={max_velocity[i]:6.3f} | ratio={ratio:5.2f} {status}")
print()

print("DIAGNOSIS COMPLETE")
print("=" * 80)
