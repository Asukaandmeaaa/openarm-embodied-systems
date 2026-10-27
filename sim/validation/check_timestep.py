#!/usr/bin/env python3
"""Check MuJoCo timestep and control frequency."""

import os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "official/openarm_mujoco/src")]

_mujoco_site = str(ROOT / ".venv/lib/python3.12/site-packages")
sys.path.insert(0, _mujoco_site)
import mujoco
sys.path.remove(_mujoco_site)

# Load the model
model = mujoco.MjModel.from_xml_path(str(ROOT / "official/openarm_mujoco/v2/cell/demo.xml"))
data = mujoco.MjData(model)

print("=" * 80)
print("MuJoCo Timing Configuration")
print("=" * 80)
print()
print(f"model.opt.timestep = {model.opt.timestep} seconds")
print(f"Physics frequency = {1.0 / model.opt.timestep:.1f} Hz")
print()

# From config
print("Training Configuration (openarm_cube_pick_expert_params.yaml):")
print(f"  physics_hz: 1000 Hz")
print(f"  record_hz: 30 Hz")
print()

# Calculate action execution
print("Current Rollout Loop (act_closed_loop.py):")
print("  for target in actions:  # 100 actions")
print("      guard.submit(target, data.time)")
print("      for _ in range(33):  # 33 physics steps")
print("          mujoco.mj_step(model, data)")
print()

physics_hz = 1.0 / model.opt.timestep
steps_per_action = 33
time_per_action = steps_per_action * model.opt.timestep
action_hz = 1.0 / time_per_action
total_chunk_time = 100 * time_per_action

print("Calculated Timing:")
print(f"  Physics timestep: {model.opt.timestep * 1000:.1f} ms")
print(f"  Steps per action: {steps_per_action}")
print(f"  Time per action: {time_per_action * 1000:.1f} ms")
print(f"  Action frequency: {action_hz:.1f} Hz")
print(f"  Total chunk duration: {total_chunk_time:.2f} seconds")
print()

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
if abs(action_hz - 30.0) < 0.1:
    print("✓ Action frequency matches record_hz (30 Hz)")
    print("✓ 33 steps × 1ms = 33ms per action is CORRECT")
    print()
    print("ACT chunk semantics:")
    print("  - 100 actions at 30 Hz = 3.33 seconds of trajectory")
    print("  - Each action should be submitted and held for 33ms")
else:
    print("✗ Timing mismatch detected!")
    print(f"  Expected: 30 Hz action rate")
    print(f"  Actual: {action_hz:.1f} Hz")

print()
print("Recommendation for fix:")
print("  Use Option B with ONE submit per action:")
print()
print("  for target in actions:")
print("      guard.submit(target, data.time)  # Submit once")
print("      for _ in range(33):")
print("          result = guard.step(state(), data.time)")
print("          mapper.set_ctrl(...)")
print("          mujoco.mj_step(model, data)")
print()
print("  This maintains:")
print("    - 30 Hz target updates (satisfies guard watchdog)")
print("    - 1000 Hz physics simulation")
print("    - Complete 100-action chunk execution")
