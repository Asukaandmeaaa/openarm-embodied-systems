#!/usr/bin/env python3
"""Quick test: 5x velocity limits to verify ACT policy direction."""

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
import torch
import yaml
from lerobot.policies import get_policy_class, make_pre_post_processors
from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.guard import ActionGuard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-seconds", type=float, default=5.0)
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

    # Multiply max_velocity by 5 (element-wise)
    max_velocity = [v * 5.0 for v in g["max_velocity"]]
    max_acceleration = g["max_acceleration"]
    max_step = g["max_step"]

    print(f"[DEBUG] lower length: {len(lower)}")
    print(f"[DEBUG] upper length: {len(upper)}")
    print(f"[DEBUG] max_velocity length: {len(max_velocity)}")
    print(f"[DEBUG] max_acceleration length: {len(max_acceleration)}")
    print(f"[DEBUG] max_step length: {len(max_step)}")

    guard = ActionGuard(lower, upper, max_velocity, max_acceleration,
                       max_step, g["watchdog_timeout_s"], g["limit_policy"])

    initial = state()
    initial[7] = cfg["pregrasp_right_gripper_rad"]
    print(f"[INIT] Initial state: right arm = {initial[:8]}")
    print(f"[INIT] Cube position: {data.qpos[qadr:qadr+3]}")

    guard.submit(initial, 0)

    # Stabilize 1.5s
    for i in range(1500):
        result = guard.step(state(), data.time)
        mapper.set_ctrl(data.ctrl, result.action[:8], "right")
        mapper.set_ctrl(data.ctrl, result.action[8:], "left")
        mujoco.mj_step(model, data)

    print(f"[STAB] After stabilize: right arm = {state()[:8]}")

    renderers = {n: mujoco.Renderer(model, height=600, width=960)
                 for n in ("ceiling", "wrist_right")}

    policy = get_policy_class("act").from_pretrained(a.checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=a.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    policy.reset()

    chunks = 0
    initial_cube_pos = data.qpos[qadr:qadr+3].copy()

    try:
        while data.time < 1.5 + a.max_seconds:
            batch = {"observation.state": torch.from_numpy(state()).unsqueeze(0)}
            for key, camera in (("ceiling", "camera_ceiling"),
                               ("wrist_right", "camera_wrist_right")):
                renderers[key].update_scene(data, camera=camera)
                rgb = renderers[key].render().copy()
                batch[f"observation.images.{key}"] = torch.from_numpy(rgb).permute(2,0,1).float().div(255).unsqueeze(0)

            with torch.inference_mode():
                actions = post(policy.predict_action_chunk(pre(batch))).numpy()[0]

            chunks += 1

            if chunks == 1:
                print(f"\n[CHUNK 1] Policy first action: {actions[0, :8]}")
                print(f"[CHUNK 1] Current state: {state()[:8]}")
                print(f"[CHUNK 1] Delta: {actions[0, :8] - state()[:8]}")

            for target in actions:
                guard.submit(target, data.time)
                for _ in range(33):
                    result = guard.step(state(), data.time)
                    mapper.set_ctrl(data.ctrl, result.action[:8], "right")
                    mapper.set_ctrl(data.ctrl, result.action[8:], "left")
                    mujoco.mj_step(model, data)

                if data.time >= 1.5 + a.max_seconds:
                    break

            if data.time >= 1.5 + a.max_seconds:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()

    final_state = state()
    final_cube_pos = data.qpos[qadr:qadr+3]

    print(f"\n[RESULT] Chunks executed: {chunks}")
    print(f"[RESULT] Final state: right arm = {final_state[:8]}")
    print(f"[RESULT] Final cube position: {final_cube_pos}")
    print(f"[RESULT] Cube moved: {np.linalg.norm(final_cube_pos - initial_cube_pos):.4f} m")
    print(f"[RESULT] Guard stats: {dict(guard.stats)}")

    report = {
        "seed": a.seed,
        "chunks": chunks,
        "sim_time_s": float(data.time),
        "initial_cube_pos": initial_cube_pos.tolist(),
        "final_cube_pos": final_cube_pos.tolist(),
        "cube_displacement_m": float(np.linalg.norm(final_cube_pos - initial_cube_pos)),
        "guard_stats": dict(guard.stats)
    }

    print(f"\n{json.dumps(report, indent=2)}")

if __name__ == "__main__":
    main()
