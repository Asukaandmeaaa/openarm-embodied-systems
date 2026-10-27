"""ManiSkill-style MuJoCo environment boundary for OpenArm PickCube."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from openarm_mujoco.v2 import JointResolver

from .config import SCENE, frozen_hashes
from .geometry import combine_action


class OpenArmPickCubeEnv:
    """Own runtime physics; planner scratch state never replaces ``data``."""

    def __init__(self, scene: Path = SCENE):
        self.scene = Path(scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.mapper = JointResolver(self.model)
        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
        joint = int(self.model.body_jntadr[self.cube_body])
        self.cube_qadr = int(self.model.jnt_qposadr[joint])
        self.cube_dadr = int(self.model.jnt_dofadr[joint])
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
        self.reset_count = 0

    def state(self) -> np.ndarray:
        right, rg = self.mapper.get_driver(self.data.qpos, "right")
        left, lg = self.mapper.get_driver(self.data.qpos, "left")
        return combine_action(np.r_[right, rg], np.r_[left, lg])

    def finger_contacts(self) -> tuple[str, ...]:
        names = set()
        for index in range(self.data.ncon):
            c = self.data.contact[index]
            bodies = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            if self.cube_body not in bodies or len(bodies) < 2:
                continue
            body = (bodies - {self.cube_body}).pop()
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if name in {"openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger"}:
                names.add(name)
        return tuple(sorted(names))

    def privileged_info(self) -> dict:
        contacts = self.finger_contacts()
        return {
            "cube_pose_wxyz": self.data.qpos[self.cube_qadr:self.cube_qadr + 7].copy(),
            "cube_velocity": self.data.qvel[self.cube_dadr:self.cube_dadr + 6].copy(),
            "tcp_position": self.data.site_xpos[self.tcp_site].copy(),
            "finger_contacts": contacts,
            "bilateral_contact": len(contacts) == 2,
        }

    def observation(self) -> dict:
        return {"observation.state": self.state()}

    def reset(self, seed: int, options: dict | None = None):
        options = options or {}
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        rng = np.random.default_rng(int(seed))
        center = np.asarray(options.get("cube_xy", [0.33, -0.28]), dtype=float)
        spread = float(options.get("xy_randomization_m", 0.02))
        yaw_limit = math.radians(float(options.get("yaw_randomization_deg", 15.0)))
        xy = center + rng.uniform(-spread, spread, 2)
        yaw = float(rng.uniform(-yaw_limit, yaw_limit))
        # Runtime qpos mutation is intentionally confined to reset.
        self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = [xy[0], xy[1], options.get("cube_z", 1.05)]
        self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel.fill(0)
        mujoco.mj_forward(self.model, self.data)
        self.reset_count += 1
        info = {
            "seed": int(seed), "reset_index": self.reset_count - 1, "cube_yaw_rad": yaw,
            "reset_cube_pose_wxyz": self.data.qpos[self.cube_qadr:self.cube_qadr + 7].copy(),
            "initial_qpos16": self.state().copy(), **frozen_hashes(),
        }
        return self.observation(), info

    def step(self, action16):
        value = np.asarray(action16, dtype=np.float32)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("action must be finite float32[16]")
        self.mapper.set_ctrl(self.data.ctrl, value[:8], "right")
        self.mapper.set_ctrl(self.data.ctrl, value[8:], "left")
        mujoco.mj_step(self.model, self.data)
        return self.observation(), self.privileged_info()

    def evaluate(self) -> dict:
        return self.privileged_info()
