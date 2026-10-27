"""Load the three copied MuJoCo models and enforce the name/pose contracts."""

from __future__ import annotations

import argparse
import math

from ..config import load_yaml, validate_task_contract
from ..paths import vendor_path


MODELS = (
    "v2/openarm_bimanual.xml",
    "v2/pedestal/pedestal.xml",
    "v2/cell/demo.xml",
)


def _id(mujoco, model, object_type, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise RuntimeError(f"required MuJoCo name is missing: {name}")
    return value


def inspect_model(relative_path: str) -> dict:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required; run with the documented local VLA Python environment") from exc

    path = vendor_path(*relative_path.split("/"))
    model = mujoco.MjModel.from_xml_path(str(path))
    left = [f"openarm_left_joint{i}" for i in range(1, 8)]
    right = [f"openarm_right_joint{i}" for i in range(1, 8)]
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    for name in left + right + ["openarm_right_finger_joint1"]:
        _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
    _id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger1_ctrl")
    _id(mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_wrist_right")
    _id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
    left_indices = [joint_names.index(name) for name in left]
    right_indices = [joint_names.index(name) for name in right]
    if left_indices != list(range(left_indices[0], left_indices[0] + 7)):
        raise RuntimeError("left arm joints are not contiguous in MuJoCo order")
    if right_indices != list(range(right_indices[0], right_indices[0] + 7)):
        raise RuntimeError("right arm joints are not contiguous in MuJoCo order")
    if max(left_indices) >= min(right_indices):
        raise RuntimeError("copied MuJoCo model no longer has left-before-right model order")

    result = {
        "model": relative_path,
        "nq": int(model.nq),
        "nu": int(model.nu),
        "right_joint_count": len(right_indices),
        "right_gripper_joint": "openarm_right_finger_joint1",
        "wrist_camera": "camera_wrist_right",
        "grasp_site": "right_grasp_frame",
        "model_joint_order": "left_then_right",
        "dataset_order": "right8_then_left8",
        "home_j4_rad": None,
    }
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        values = []
        for name in ("openarm_left_joint4", "openarm_right_joint4"):
            joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
            values.append(float(model.key_qpos[home, model.jnt_qposadr[joint]]))
        if any(not math.isclose(value, math.pi / 2, abs_tol=1e-5) for value in values):
            raise RuntimeError(f"{relative_path}: pedestal/cell home J4 is not pi/2: {values}")
        result["home_j4_rad"] = values
    if relative_path == "v2/pedestal/pedestal.xml" and result["home_j4_rad"] is None:
        raise RuntimeError("pedestal scene has no home keyframe")
    return result


def run_smoke() -> list[dict]:
    contract = load_yaml("task_contract.yaml")
    validate_task_contract(contract)
    expected = [
        "right_joint1", "right_joint2", "right_joint3", "right_joint4",
        "right_joint5", "right_joint6", "right_joint7", "right_gripper",
        "left_joint1", "left_joint2", "left_joint3", "left_joint4",
        "left_joint5", "left_joint6", "left_joint7", "left_gripper",
    ]
    if contract["ordering"]["state_action"] != expected:
        raise RuntimeError("dataset state/action order is not right8 then left8")
    return [inspect_model(path) for path in MODELS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for result in run_smoke():
        home = result["home_j4_rad"]
        suffix = ""
        if home is not None:
            suffix = f"; home J4={home} rad (~pi/2), NOT all-zero mechanical motor zero"
        print(
            f"PASS: {result['model']} nq={result['nq']} nu={result['nu']}; "
            "right J1-J7, right gripper, wrist camera, and right_grasp_frame resolved; "
            f"model order left->right; dataset order right8->left8{suffix}"
        )
    print("PASS: pedestal home J4 is ~pi/2; task-ready/ACT Home is NOT all-zero mechanical motor zero")


if __name__ == "__main__":
    main()
