#!/usr/bin/env python3
"""Minimal OpenArm MuJoCo position-actuator smoke test (no dora required)."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_XML = Path(__file__).resolve().parent / "v2" / "cell" / "demo.xml"


def _object_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    key_id = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Initialize all position targets to their corresponding current qpos so
    # only the selected actuator receives a changed target.
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]

    joint_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint1"
    )
    actuator_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint1_ctrl"
    )
    qpos_id = int(model.jnt_qposadr[joint_id])

    before_qpos = data.qpos.copy()
    ctrl_before = data.ctrl.copy()
    command = float(before_qpos[qpos_id] + 0.2)
    data.ctrl[actuator_id] = command
    ctrl_after = data.ctrl.copy()

    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    after_qpos = data.qpos.copy()
    delta = float(after_qpos[qpos_id] - before_qpos[qpos_id])
    print(f"before qpos: {before_qpos.tolist()}")
    print(f"command: right_joint1={command}")
    print(f"ctrl before: {ctrl_before.tolist()}")
    print(f"ctrl: {ctrl_after.tolist()}")
    print(f"after qpos: {after_qpos.tolist()}")
    print(f"right_joint1 delta: {delta}")

    if np.isclose(delta, 0.0):
        raise SystemExit("FAIL: openarm_right_joint1 qpos did not change")
    print("PASS: openarm_right_joint1 qpos changed")


if __name__ == "__main__":
    main()
