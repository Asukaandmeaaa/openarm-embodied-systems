"""MuJoCo world, contact-dynamics expert execution and independent replay."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation

from .mapping import SimMapper
from .planner import MujocoGraspPlanner

ROOT = Path(__file__).resolve().parents[3]
SCENE = Path(__file__).resolve().parents[1] / "assets/scene.xml"
FPS = 30


def load_config(path: Path | None = None) -> dict:
    import yaml

    path = path or Path(__file__).with_name("config.yaml")
    return yaml.safe_load(path.read_text())


class SimWorld:
    """One randomized scene.  Physics is always advanced by mj_step."""

    def __init__(self, seed: int, cfg: dict, *, jitter: dict | None = None):
        self.cfg = cfg
        self.seed = int(seed)
        jitter = jitter or {}
        self.jitter = {
            "cube_xy_jitter_m": float(jitter.get("cube_xy_jitter_m", cfg["cube_xy_jitter_m"])),
            "cube_yaw_jitter_deg": float(jitter.get("cube_yaw_jitter_deg", cfg["cube_yaw_jitter_deg"])),
            "table_xy_jitter_m": float(jitter.get("table_xy_jitter_m", cfg["table_xy_jitter_m"])),
            "table_yaw_jitter_deg": float(jitter.get("table_yaw_jitter_deg", cfg["table_yaw_jitter_deg"])),
        }
        spec = mujoco.MjSpec.from_file(str(SCENE))
        rng = np.random.default_rng(self.seed)
        cube_delta = rng.uniform(-self.jitter["cube_xy_jitter_m"], self.jitter["cube_xy_jitter_m"], 2)
        cube_angle = float(rng.uniform(-self.jitter["cube_yaw_jitter_deg"], self.jitter["cube_yaw_jitter_deg"]))
        table_delta = rng.uniform(-self.jitter["table_xy_jitter_m"], self.jitter["table_xy_jitter_m"], 2)
        table_angle = float(rng.uniform(-self.jitter["table_yaw_jitter_deg"], self.jitter["table_yaw_jitter_deg"]))
        rotation = Rotation.from_euler("z", table_angle, degrees=True)
        matrix = rotation.as_matrix()
        pivot = np.array(spec.body("table").pos).copy()
        for name in ("table", "black_frame"):
            body = spec.body(name)
            body.pos = pivot + matrix @ (np.array(body.pos) - pivot) + np.r_[table_delta, 0.0]
            old = Rotation.from_quat(np.array(body.quat)[[1, 2, 3, 0]])
            body.quat = (rotation * old).as_quat()[[3, 0, 1, 2]]
        self.randomization = {
            "table_translation_xy_m": table_delta.tolist(),
            "table_yaw_deg": table_angle,
            "cube_translation_in_table_xy_m": cube_delta.tolist(),
            "cube_yaw_relative_table_deg": cube_angle,
            "cube_world_yaw_deg": cube_angle + table_angle,
            "table_rotation_pivot_world_m": pivot.tolist(),
            "camera_pose_randomized": False,
            "robot_base_randomized": False,
            "rng_seed": self.seed,
        }
        # Ideal collection-style gravity feedforward.  This is an explicit
        # simulation assumption, not an identified hardware model.
        for body in spec.bodies:
            if body.name.startswith("openarm_"):
                body.gravcomp = 1
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.mapper = SimMapper(self.model)
        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.model.key("physical_home_reference").id,
        )
        right = np.r_[
            np.array(self.data.qpos[self.mapper._qpos["right"]], dtype=float),
            self.cfg["return_closed_right_gripper_rad"],
        ]
        left = np.r_[np.zeros(7), self.cfg["left_gripper_hold_rad"]]
        self.home_driver = np.r_[right, left]
        self.mapper.set_qpos(self.data.qpos, right, "right")
        self.mapper.set_qpos(self.data.qpos, left, "left")
        self.mapper.set_ctrl(self.data.ctrl, right, "right")
        self.mapper.set_ctrl(self.data.ctrl, left, "left")
        mujoco.mj_forward(self.model, self.data)
        self.site = self.model.site("right_grasp_frame").id
        self.cube = self.model.body("orange_cube").id
        self.box = self.model.body("black_frame").id
        cube_joint = self.model.body_jntadr[self.cube]
        self.cube_qadr = int(self.model.jnt_qposadr[cube_joint])
        nominal = self.data.qpos[self.cube_qadr : self.cube_qadr + 3].copy() + np.r_[cube_delta, 0.0]
        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = (
            pivot + matrix @ (nominal - pivot) + np.r_[table_delta, 0.0]
        )
        self.cube_yaw = math.radians(cube_angle + table_angle)
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = [
            math.cos(self.cube_yaw / 2),
            0,
            0,
            math.sin(self.cube_yaw / 2),
        ]
        mujoco.mj_forward(self.model, self.data)
        self.planner = MujocoGraspPlanner(
            self.model, self.mapper, self.cfg, self.seed, runtime_data=self.data
        )
        limits = np.array(self.cfg["position_limits_rad"], dtype=float)
        self.planner.lower = np.maximum(self.planner.lower, limits[:7, 0])
        self.planner.upper = np.minimum(self.planner.upper, limits[:7, 1])
        self.limits = limits
        self.dt = float(self.model.opt.timestep)
        self.cmd_limits = np.array(self.cfg["gripper_command_range_rad"], dtype=float)
        self.nominal_cube_xyz = self.data.xpos[self.cube].copy()
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self.samples: list[dict] = []
        self.next_sample = 0.0
        self.last_q = self.home_driver[:8].copy()
        self.last_command16 = self.home_driver.copy()
        self.last_v = np.zeros(8)
        self.max_v = 0.0
        self.max_a = 0.0
        self.max_command_step = 0.0
        self.forbidden_contacts: list[list[str]] = []
        self.peak_lift = 0.0
        self.continuous_hold = 0.0
        self.contacts_duration = 0.0
        self.grasp_relative = None
        self.events: list[dict] = []

    # ----- state ---------------------------------------------------------
    def state16(self) -> np.ndarray:
        right, _ = self.mapper.get_driver(self.data.qpos, "right")
        left, _ = self.mapper.get_driver(self.data.qpos, "left")
        return np.r_[right, left].astype(np.float32)

    def render_frames(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        out = {}
        for role in ("ceiling", "wrist_right"):
            self._renderer.update_scene(self.data, camera=f"camera_{role}")
            out[role] = self._renderer.render().copy()
        return out

    def close(self) -> None:
        if hasattr(self, "_renderer"):
            self._renderer.close()
            del self._renderer

    def cube_xyz(self) -> np.ndarray:
        return self.data.xpos[self.cube].copy()

    def tcp_xyz(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def event(self, name: str, **detail) -> None:
        self.events.append(
            dict(t=float(self.data.time), name=name, detail=detail)
        )

    # ----- contact / success --------------------------------------------
    def contacts(self) -> tuple[bool, list[list[str]]]:
        fingers: set[str] = set()
        forbidden: list[list[str]] = []
        for contact in self.data.contact:
            names = [
                self.model.body(self.model.geom_bodyid[x]).name or ""
                for x in (contact.geom1, contact.geom2)
            ]
            if "orange_cube" in names:
                fingers.update(
                    n
                    for n in names
                    if n
                    in (
                        "openarm_right_ee_inner_finger",
                        "openarm_right_ee_outer_finger",
                    )
                )
                if any(n.startswith("openarm_") and n not in fingers for n in names):
                    forbidden.append(names)
                continue
            if contact.dist >= -0.001:
                continue
            if any(n.startswith("openarm_") for n in names):
                if all("finger" in n for n in names):
                    continue
                forbidden.append(names)
        self.finger_count = len(fingers)
        return len(fingers) == 2, forbidden

    def box_outcome(self) -> tuple[bool, bool]:
        rotation = self.data.xmat[self.box].reshape(3, 3)
        local = rotation.T @ (self.data.xpos[self.cube] - self.data.xpos[self.box])
        half = np.abs(rotation.T @ self.data.xmat[self.cube].reshape(3, 3)) @ np.full(3, 0.02)
        inside = bool(np.all(np.abs(local[:2]) + half[:2] <= 0.06))
        supported = bool(abs(local[2] - half[2] - 0.01) < 0.002)
        return inside, supported

    def final_success(self) -> dict:
        inside, supported = self.box_outcome()
        home_error = float(
            np.max(np.abs(self.state16()[:7] - self.home_driver[:7]))
        )
        bilateral, forbidden = self.contacts()
        cube_velocity = float(np.linalg.norm(self.data.cvel[self.cube][3:]))
        return {
            "cube_inside": inside,
            "cube_supported": supported,
            "grader_released": self.finger_count == 0,
            "home_error_rad": home_error,
            "home_ok": home_error <= self.cfg["max_home_error_rad"],
            "peak_lift_m": self.peak_lift,
            "lift_ok": self.peak_lift >= self.cfg["hold_height_m"],
            "bilateral_lift_s": self.contacts_duration,
            "hold_ok": self.contacts_duration >= self.cfg["hold_s"],
            "final_cube_speed_m_s": cube_velocity,
            "settled": cube_velocity < 0.01,
            "forbidden_contacts": forbidden,
            "success": bool(
                inside
                and supported
                and self.finger_count == 0
                and home_error <= self.cfg["max_home_error_rad"]
                and self.peak_lift >= self.cfg["hold_height_m"]
                and self.contacts_duration >= self.cfg["hold_s"]
                and not forbidden
                and cube_velocity < 0.01
            ),
        }

    # ----- stepping ------------------------------------------------------
    def validate_command(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("invalid 16-dimensional command")
        if np.any(action[:7] < self.limits[:7, 0]) or np.any(action[:7] > self.limits[:7, 1]):
            raise RuntimeError("right joint command outside unchanged real ACT range")
        if not self.cmd_limits[0] - 1e-9 <= action[7] <= self.cmd_limits[1] + 1e-9:
            raise RuntimeError("right gripper command outside real ACT range")
        return action

    def step(self, action: np.ndarray, phase: str, *, valid: bool = True) -> None:
        action = self.validate_command(action)
        right = action[:8]
        left = action[8:]
        dt = self.dt
        velocity = (right - self.last_q) / dt
        acceleration = (velocity - self.last_v) / dt
        self.max_v = max(self.max_v, float(np.max(np.abs(velocity))))
        self.max_a = max(self.max_a, float(np.max(np.abs(acceleration))))
        self.last_q = right.copy()
        self.last_command16 = action.copy()
        self.last_v = velocity.copy()
        self.mapper.set_ctrl(self.data.ctrl, right, "right")
        self.mapper.set_ctrl(self.data.ctrl, left, "left")
        mujoco.mj_step(self.model, self.data)
        bilateral, forbidden = self.contacts()
        if forbidden:
            self.forbidden_contacts.append(forbidden[:3])
            raise RuntimeError("forbidden_contact:" + str(forbidden[:3]))
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("nonfinite physics")
        cube = self.cube_xyz()
        self.peak_lift = max(self.peak_lift, float(cube[2] - self.nominal_cube_xyz[2]))
        if phase in {"LIFT", "HOLD", "TRANSPORT", "PLACE_ABOVE_BOX"} and self.grasp_relative is not None:
            relative = (
                self.data.site_xmat[self.site].reshape(3, 3).T
                @ (cube - self.data.site_xpos[self.site])
            )
            if np.linalg.norm(relative - self.grasp_relative) > self.cfg["max_relative_drift_m"]:
                raise RuntimeError(f"{phase}:grasp_slip_exceeded_existing_25mm_drift_bound")
        if bilateral and cube[2] - self.nominal_cube_xyz[2] >= self.cfg["hold_height_m"]:
            self.continuous_hold += dt
            self.contacts_duration = max(self.contacts_duration, self.continuous_hold)
        else:
            self.continuous_hold = 0.0
        if self.data.time + 1e-12 >= self.next_sample:
            self.record(phase=phase, valid=valid)
            self.next_sample += 1.0 / FPS

    def record(self, *, phase: str, valid: bool) -> None:
        # The 30 Hz grid is exact in the dataset; physics is sampled on the
        # next 1 ms step (<=1 ms after the nominal instant).
        self.samples.append(
            dict(
                time=len(self.samples) / FPS,
                physics_time=float(self.data.time),
                phase=phase,
                state=self.state16().copy(),
                action=self.last_command16.astype(np.float32).copy(),
                valid=bool(valid),
                cube=self.cube_xyz().copy(),
                cube_speed=float(np.linalg.norm(self.data.cvel[self.cube][3:])),
                cube_half_box=(np.abs(self.data.xmat[self.box].reshape(3, 3).T
                                    @ self.data.xmat[self.cube].reshape(3, 3)) @ np.full(3, 0.02)),
                tcp=self.tcp_xyz().copy(),
                bilateral=bool(self.contacts()[0]),
                finger_count=self.finger_count,
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
            )
        )

    def hold_for(self, seconds: float, phase: str = "HOLD") -> None:
        """Hold the last target while physics and gravity continue."""
        if seconds <= 0:
            return
        command = self.last_command16.copy()
        for _ in range(int(np.ceil(seconds / self.dt))):
            self.step(command, phase)

    # ----- expert motion -------------------------------------------------
    def follow(self, phase: str, points, min_duration: float = 0.0, *, valid: bool = True) -> None:
        points = np.asarray(points, dtype=float)
        points = np.vstack([self.last_q, points])
        arm_points = points[:, :7]
        grip_start = float(points[0, 7])
        grip_end = float(points[-1, 7])
        spline = CubicSpline(np.linspace(0, 1, len(arm_points)), arm_points, bc_type="natural")
        probe = spline(np.linspace(0, 1, 2001))
        if np.any(probe[:, :] < self.limits[:7, 0]) or np.any(probe[:, :] > self.limits[:7, 1]):
            spline = PchipInterpolator(np.linspace(0, 1, len(arm_points)), arm_points)

        def curve(u, derivative=0):
            u = np.asarray(u)
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            ds = 30 * u**2 - 60 * u**3 + 30 * u**4
            dds = 60 * u - 180 * u**2 + 120 * u**3
            grip = grip_start + (grip_end - grip_start) * s
            if derivative == 1:
                arm = spline(s, 1) * ds[..., None]
                return np.concatenate([arm, ((grip_end - grip_start) * ds)[..., None]], axis=-1)
            if derivative == 2:
                arm = spline(s, 2) * ds[..., None] ** 2 + spline(s, 1) * dds[..., None]
                return np.concatenate([arm, ((grip_end - grip_start) * dds)[..., None]], axis=-1)
            return np.concatenate([spline(s), grip[..., None]], axis=-1)

        u = np.linspace(0, 1, 2001)
        velocity = float(self.cfg["max_velocity_rad_s"])
        acceleration = float(self.cfg["max_acceleration_rad_s2"])
        duration = max(
            min_duration,
            1.02 * np.max(np.abs(curve(u, 1))) / velocity,
            math.sqrt(1.02 * np.max(np.abs(curve(u, 2))) / acceleration),
            0.1,
        )
        count = math.ceil(duration / self.dt)
        for i in range(1, count + 1):
            command = np.r_[curve(i / count), self.home_driver[8:]]
            self.step(command, phase, valid=valid)
        self.event("phase", phase=phase, time_s=float(self.data.time))

    def cartesian(self, phase, target_pos, target_rotation, gripper, *, valid=True, offset=None):
        if offset is not None:
            target_pos = np.asarray(target_pos, dtype=float) + np.asarray(offset, dtype=float)
            target_rotation = self.planner.target_rotation(self.cube_yaw) if target_rotation is None else target_rotation
        start = self.data.site_xpos[self.site].copy()
        start_r = self.data.site_xmat[self.site].reshape(3, 3).copy()
        q = self.state16()[:7].copy()
        path = []
        n = max(20, math.ceil(np.linalg.norm(np.asarray(target_pos) - start) / 0.005))
        rotations = Rotation.from_matrix(np.stack([start_r, target_rotation]))
        rotations = _slerp(rotations)
        for i in range(1, n + 1):
            u = i / n
            qn = self.planner.solve_pose(
                start + (np.asarray(target_pos) - start) * u,
                rotations(u).as_matrix(),
                q,
                prefer_continuity=True,
            )
            if qn is None:
                raise RuntimeError(f"{phase}:IK at {i}/{n}")
            if np.max(np.abs(qn - q)) > self.cfg["max_ik_branch_jump_rad"]:
                raise RuntimeError(f"{phase}:IK branch discontinuity")
            if self.planner._has_forbidden_collision(qn, allow_cube=True, gripper=gripper):
                raise RuntimeError(f"{phase}:planned collision")
            path.append(np.r_[qn, gripper])
            q = qn
        shortcut = np.linspace(
            self.last_q,
            np.asarray(path[-1]),
            max(100, math.ceil(np.max(np.abs(path[-1] - self.last_q)) / 0.005)),
        )
        if all(
            not self.planner._has_forbidden_collision(q[:7], allow_cube=True, gripper=q[7])
            for q in shortcut
        ):
            self.follow(phase, [path[-1]], valid=valid)
        else:
            self.follow(phase, path, valid=valid)

    def cartesian_to_cube(self, phase, position, gripper, *, valid=True):
        """Cartesian move with an explicit position target (no random offset)."""
        rotation = self.planner.target_rotation(self.cube_yaw)
        self.cartesian(phase, position, rotation, gripper, valid=valid)


def _slerp(rotations):
    from scipy.spatial.transform import Slerp

    return Slerp([0, 1], rotations)
