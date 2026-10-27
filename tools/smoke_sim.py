"""CPU-only smoke: load the shipped MJCF and step physics without rendering."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "official/openarm_mujoco/src")]
import mujoco
import numpy as np
from openarm_cube_pick_expert.env import OpenArmPickCubeEnv


def main():
    env = OpenArmPickCubeEnv()
    obs, info = env.reset(seed=20260909)
    initial = obs["observation.state"]
    if initial.shape != (16,) or not np.isfinite(initial).all():
        raise RuntimeError("invalid initial observation")
    for _ in range(100):
        mujoco.mj_step(env.model, env.data)
    if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.state()).all():
        raise RuntimeError("nonfinite physics state")
    print(json.dumps({"hardware_io": False, "steps": 100,
                     "simulation_time_s": float(env.data.time),
                     "observation_shape": list(env.state().shape),
                     "scene_sha256": info["scene_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
