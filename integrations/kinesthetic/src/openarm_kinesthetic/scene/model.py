"""Load and introspect the project-owned real_table_v1 MuJoCo overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..paths import workspace_path


RIGHT_JOINTS = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
LEFT_JOINTS = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
RIGHT_ACTUATORS = tuple(f"right_joint{i}_ctrl" for i in range(1, 8))
LEFT_ACTUATORS = tuple(f"left_joint{i}_ctrl" for i in range(1, 8))


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("MuJoCo is required in the documented local VLA environment") from exc
    return mujoco


def name_id(model, object_type, name: str) -> int:
    mujoco = _mujoco()
    index = int(mujoco.mj_name2id(model, object_type, name))
    if index < 0:
        raise KeyError(f"MuJoCo object is missing: {name}")
    return index


def load_model(path: str | Path = "sim/scenes/real_table_v1.xml"):
    mujoco = _mujoco()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_path(*candidate.parts)
    return mujoco.MjModel.from_xml_path(str(candidate))


@dataclass(frozen=True)
class ModelMap:
    right_qpos: np.ndarray
    right_dof: np.ndarray
    left_qpos: np.ndarray
    left_dof: np.ndarray
    right_actuators: np.ndarray
    left_actuators: np.ndarray
    right_gripper_qpos: int
    left_gripper_qpos: int
    right_gripper_actuator: int
    left_gripper_actuator: int
    cube_qpos: int
    cube_dof: int
    grasp_site: int
    cube_body: int

    @classmethod
    def from_model(cls, model) -> "ModelMap":
        mujoco = _mujoco()

        def joint_addresses(names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
            ids = [name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            return (
                np.asarray([model.jnt_qposadr[i] for i in ids], dtype=int),
                np.asarray([model.jnt_dofadr[i] for i in ids], dtype=int),
            )

        right_qpos, right_dof = joint_addresses(RIGHT_JOINTS)
        left_qpos, left_dof = joint_addresses(LEFT_JOINTS)
        right_finger = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint1")
        left_finger = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_left_finger_joint1")
        cube = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "blue_cube_free")
        return cls(
            right_qpos=right_qpos,
            right_dof=right_dof,
            left_qpos=left_qpos,
            left_dof=left_dof,
            right_actuators=np.asarray([name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in RIGHT_ACTUATORS]),
            left_actuators=np.asarray([name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEFT_ACTUATORS]),
            right_gripper_qpos=int(model.jnt_qposadr[right_finger]),
            left_gripper_qpos=int(model.jnt_qposadr[left_finger]),
            right_gripper_actuator=name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger1_ctrl"),
            left_gripper_actuator=name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger1_ctrl"),
            cube_qpos=int(model.jnt_qposadr[cube]),
            cube_dof=int(model.jnt_dofadr[cube]),
            grasp_site=name_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame"),
            cube_body=name_id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube"),
        )

    def dataset_vector(self, qpos: np.ndarray, *, right_gripper_command: float | None = None) -> np.ndarray:
        """Return the authoritative right8-then-left8 state/action order."""
        rg = float(qpos[self.right_gripper_qpos]) if right_gripper_command is None else float(right_gripper_command)
        return np.asarray(
            [*qpos[self.right_qpos], rg, *qpos[self.left_qpos], qpos[self.left_gripper_qpos]],
            dtype=np.float32,
        )


def reset_task_ready(model, data) -> ModelMap:
    mujoco = _mujoco()
    mapping = ModelMap.from_model(model)
    key = name_id(model, mujoco.mjtObj.mjOBJ_KEY, "task_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return mapping


def ros_optical_pose_to_mujoco_camera(
    translation_xyz: Iterable[float], quaternion_xyzw: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ROS optical (+Z forward, +Y down) to MuJoCo (-Z forward, +Y up)."""
    mujoco = _mujoco()
    xyzw = np.asarray(tuple(quaternion_xyzw), dtype=float)
    if xyzw.shape != (4,) or not np.all(np.isfinite(xyzw)):
        raise ValueError("quaternion_xyzw must be four finite values")
    norm = float(np.linalg.norm(xyzw))
    if norm <= 1e-12:
        raise ValueError("zero quaternion is invalid")
    ros_wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float) / norm
    ros_matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(ros_matrix, ros_wxyz)
    camera_matrix = ros_matrix.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    camera_wxyz = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(camera_wxyz, camera_matrix.ravel())
    translation = np.asarray(tuple(translation_xyz), dtype=float)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_xyz must be three finite values")
    return translation, camera_wxyz
