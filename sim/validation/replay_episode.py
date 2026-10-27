#!/usr/bin/env python3
"""Replay one recorded action stream through MuJoCo position actuators."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from openarm_mujoco.v2 import JointResolver


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"


def load_actions(episode: Path, side: str) -> np.ndarray:
    path = episode / "action" / "arms" / side / "state.parquet"
    return np.asarray(pq.read_table(path, columns=["qpos"])["qpos"].to_pylist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--steps-per-action", type=int, default=34)
    args = parser.parse_args()

    right = load_actions(args.episode, "right")
    left = load_actions(args.episode, "left")
    count = min(len(right), len(left))
    if count < 2 or right.shape[1:] != (8,) or left.shape[1:] != (8,):
        raise SystemExit(f"invalid action shapes: right={right.shape}, left={left.shape}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mapper = JointResolver(model)
    initial_qpos = data.qpos.copy()
    for index in range(count):
        mapper.set_ctrl(data.ctrl, right[index], "right")
        mapper.set_ctrl(data.ctrl, left[index], "left")
        for _ in range(args.steps_per_action):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise SystemExit(f"NaN/Inf during replay at action {index}")
    delta = float(np.max(np.abs(data.qpos - initial_qpos)))
    if delta < 1e-3:
        raise SystemExit("replay produced no physical qpos movement")
    print(f"PASS actions={count} max_qpos_delta={delta:.6f} m/rad")


if __name__ == "__main__":
    main()
