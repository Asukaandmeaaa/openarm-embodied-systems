#!/usr/bin/env python3
"""Debug version: stabilize with periodic target resubmission."""

from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[0]
sys.path[:0] = [str(ROOT), str(ROOT / "official/openarm_mujoco/src")]

_mujoco_site = str(ROOT / ".venv/lib/python3.12/site-packages")
sys.path.insert(0, _mujoco_site)
import mujoco
sys.path.remove(_mujoco_site)
import numpy as np
import yaml
from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.guard import ActionGuard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--stabilize-seconds", type=float, default=1.5)
    parser.add_argument("--resubmit-interval-ms", type=int, default=30)
    a = parser.parse_args()

    cfg = yaml.safe_load((ROOT/"configs/collection/openarm_cube_pick_expert_params.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT/"official/openarm_mujoco/v2/cell/demo.xml"))
    data = mujoco.MjData(model)
    mapper = JointResolver(model)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
    joint = int(model.body_jntadr[cube])
    qadr = int(model.jnt_qposadr[joint])

    rng = np.random.default_rng(a.seed)
    data.qpos[qadr:qadr+2] = [.33+rng.uniform(-.02,.02), -.28+rng.uniform(-.02,.02)]
    yaw = rng.uniform(-math.radians(15), math.radians(15))
    data.qpos[qadr+3:qadr+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    data.qvel.fill(0)

    def state():
        r, rg = mapper.get_driver(data.qpos, "right")
        l, lg = mapper.get_driver(data.qpos, "left")
        return combine_action(np.r_[r, rg], np.r_[l, lg])

    lower = np.r_[model.actuator_ctrlrange[9:17,0], model.actuator_ctrlrange[1:9,0]]
    upper = np.r_[model.actuator_ctrlrange[9:17,1], model.actuator_ctrlrange[1:9,1]]
    g = cfg["guard"]

    guard = ActionGuard(lower, upper, g["max_velocity"], g["max_acceleration"],
                       g["max_step"], g["watchdog_timeout_s"], g["limit_policy"])

    initial_state = state()
    initial_target = initial_state.copy()
    initial_target[7] = cfg["pregrasp_right_gripper_rad"]

    print("="*70)
    print("STABILIZE DEBUG TEST")
    print("="*70)
    print(f"\nGuard parameters:")
    print(f"  watchdog_timeout_s: {g['watchdog_timeout_s']}")
    print(f"  limit_policy: {g['limit_policy']}")
    print(f"\nStabilize configuration:")
    print(f"  Duration: {a.stabilize_seconds}s ({int(a.stabilize_seconds * 1000)}ms)")
    print(f"  Resubmit interval: {a.resubmit_interval_ms}ms")
    print(f"  Expected resubmits: {int(a.stabilize_seconds * 1000 / a.resubmit_interval_ms)}")

    print(f"\n[BEFORE STABILIZE]")
    print(f"  Target: {initial_target}")
    print(f"  State:  {initial_state}")
    print(f"  Delta:  {initial_target - initial_state}")
    print(f"  Cube position: {data.qpos[qadr:qadr+3]}")

    # Stabilize with periodic resubmission
    stabilize_steps = int(a.stabilize_seconds * 1000)  # 1ms per step
    resubmit_interval = a.resubmit_interval_ms
    next_resubmit = 0
    resubmit_count = 0

    guard.submit(initial_target, data.time)
    resubmit_count += 1

    for i in range(stabilize_steps):
        # Resubmit target periodically
        if i >= next_resubmit and i > 0:
            guard.submit(initial_target, data.time)
            resubmit_count += 1
            next_resubmit = i + resubmit_interval

        result = guard.step(state(), data.time)
        mapper.set_ctrl(data.ctrl, result.action[:8], "right")
        mapper.set_ctrl(data.ctrl, result.action[8:], "left")
        mujoco.mj_step(model, data)

    final_state = state()
    stabilize_stats = dict(guard.stats)

    print(f"\n[AFTER STABILIZE]")
    print(f"  Target: {initial_target}")
    print(f"  State:  {final_state}")
    print(f"  Delta:  {final_state - initial_target}")
    print(f"  Cube position: {data.qpos[qadr:qadr+3]}")

    print(f"\n[STABILIZE STATS]")
    print(f"  Actual resubmits: {resubmit_count}")
    print(f"  Guard stats: {json.dumps(stabilize_stats, indent=4)}")

    # Analysis
    delta_norm = np.linalg.norm(final_state - initial_target)
    hold_watchdog = stabilize_stats.get("hold_watchdog", 0)

    print(f"\n[ANALYSIS]")
    print(f"  ||final_state - target||: {delta_norm:.6f}")
    print(f"  hold_watchdog: {hold_watchdog} ms")
    print(f"  Expected hold_watchdog: ~0 ms (if resubmit works)")

    if hold_watchdog < 10:
        print(f"  ✓ SUCCESS: hold_watchdog near zero, stabilization worked!")
    else:
        print(f"  ✗ FAILURE: hold_watchdog = {hold_watchdog}, watchdog still triggered")

    report = {
        "seed": a.seed,
        "stabilize_seconds": a.stabilize_seconds,
        "resubmit_interval_ms": a.resubmit_interval_ms,
        "resubmit_count": resubmit_count,
        "initial_target": initial_target.tolist(),
        "initial_state": initial_state.tolist(),
        "final_state": final_state.tolist(),
        "delta": (final_state - initial_target).tolist(),
        "delta_norm": float(delta_norm),
        "guard_stats": stabilize_stats,
        "hold_watchdog_ms": hold_watchdog
    }

    print(f"\n{json.dumps(report, indent=2)}")

if __name__ == "__main__":
    main()
