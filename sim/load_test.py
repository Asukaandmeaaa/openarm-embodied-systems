"""MuJoCo 加载 OpenArm URDF 的冒烟测试。

验证：模型加载、自由度、关节命名、网格渲染。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

URDF = Path(__file__).resolve().parent.parent / "urdf/openarm_v20.urdf"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(URDF))
    data = mujoco.MjData(model)

    print(f"[load_test] nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"[load_test] nmesh={model.nmesh} ngeom={model.ngeom}")

    # 可动关节清单（revolute/prismatic，排除 fixed）
    joint_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            joint_names.append(model.jnt_name(j))
    print(f"[load_test] 可动关节 ({len(joint_names)}):")
    for name in joint_names:
        qpos_addr = model.jnt_qposadr[model.jnt_name2id(name.encode())]
        print(f"    {name:<32} qpos_adr={qpos_addr}")

    # 取默认位姿跑几步，确认动力学稳定
    mujoco.mj_step(model, data)
    mujoco.mj_step(model, data)
    print(f"[load_test] 两步后 qpos[0:8] = {np.round(data.qpos[:8], 3)}")

    # 渲染一张图，确认 mesh 正常加载（有内容、非全黑）
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 640, 480)
    renderer.update_scene(data, camera=-1)
    img = renderer.render()
    print(f"[load_test] 渲染 {img.shape} 像素均值={img.mean():.2f} 方差={img.std():.2f}")
    if img.std() < 1.0:
        print("[load_test] WARNING: 渲染接近纯色，mesh 可能未加载")
        sys.exit(1)
    print("[load_test] OK: 模型加载 + 渲染正常")


if __name__ == "__main__":
    main()
