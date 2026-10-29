"""Damped-least-squares grasp planner, ported from the old expert.

Source: openarm_cube_pick_expert/expert.py (historical port; see config/port_manifest.json)
The planning math and candidate rejection rules are retained.  No old-workspace
module is imported at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .geometry import yaw_candidates


class IKTimeout(RuntimeError):
    pass


@dataclass
class GraspCandidate:
    index: int
    yaw: float
    waypoints: list[np.ndarray] = field(default_factory=list)
    transit_waypoints: list[np.ndarray] = field(default_factory=list)
    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_waypoints: list[np.ndarray] = field(default_factory=list)
    reachable: bool = False
    collision_free: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    score: tuple[float, ...] = (math.inf,)


def rank_candidates(candidates: list[GraspCandidate]) -> list[GraspCandidate]:
    return sorted(
        (c for c in candidates if c.reachable and c.collision_free),
        key=lambda c: c.score,
    )


class MujocoGraspPlanner:
    def __init__(self, model, mapper, config, seed=0, runtime_data=None):
        self.model, self.mapper, self.cfg = model, mapper, config
        self.data = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
        if self.site < 0:
            raise ValueError("MJCF is missing right_grasp_frame")
        self.qidx = mapper._qpos["right"]
        self.didx = mapper._dof["right"]
        joint_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}"
            )
            for i in range(1, 8)
        ]
        self.lower = np.array([model.jnt_range[j, 0] for j in joint_ids])
        self.upper = np.array([model.jnt_range[j, 1] for j in joint_ids])
        if runtime_data is not None:
            self.data.qpos[:] = runtime_data.qpos
            self.data.qvel[:] = runtime_data.qvel
        mapper.set_qpos(
            self.data.qpos,
            np.r_[np.zeros(7), config["pregrasp_right_gripper_rad"]],
            "right",
        )
        mujoco.mj_forward(self.model, self.data)
        self.baseline_left_contacts = self._penetrating_contact_pairs(left_only=True)
        self.rng = np.random.default_rng(seed)
        self.ik_timeout_count = 0

    @staticmethod
    def target_rotation(yaw: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        cp, sp = math.cos(math.pi / 2), math.sin(math.pi / 2)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        return rz @ ry

    def _solve(self, position, rotation, seed, *, deadline=None):
        q = np.asarray(seed, dtype=float).copy()
        pos_tol = self.cfg["ik_position_tolerance_m"]
        rot_tol = self.cfg["ik_orientation_tolerance_rad"]

        def residual(value):
            if deadline is not None and time.monotonic() > deadline:
                raise IKTimeout("IK attempt deadline exceeded")
            self.data.qpos[self.qidx] = value
            mujoco.mj_forward(self.model, self.data)
            current_r = self.data.site_xmat[self.site].reshape(3, 3).copy()
            ep = np.asarray(position) - self.data.site_xpos[self.site]
            er = Rotation.from_matrix(rotation @ current_r.T).as_rotvec()
            return np.concatenate((ep, 0.35 * er))

        result = least_squares(
            residual,
            np.clip(q, self.lower + 0.002, self.upper - 0.002),
            bounds=(self.lower + 0.002, self.upper - 0.002),
            max_nfev=self.cfg["ik_iterations"],
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        error = residual(result.x)
        if (
            np.linalg.norm(error[:3]) <= pos_tol
            and np.linalg.norm(error[3:]) / 0.35 <= rot_tol
        ):
            return result.x.copy()
        return None

    def solve_pose(self, position, rotation, current, *, prefer_continuity=False, deadline=None):
        def attempt(seed):
            if deadline is not None and time.monotonic() >= deadline:
                raise IKTimeout("global planning deadline exceeded")
            attempt_deadline = time.monotonic() + float(self.cfg.get("ik_attempt_timeout_s", 2.0))
            if deadline is not None:
                attempt_deadline = min(attempt_deadline, deadline)
            try:
                return self._solve(position, rotation, seed, deadline=attempt_deadline)
            except IKTimeout:
                self.ik_timeout_count += 1
                return None

        if prefer_continuity:
            continued = attempt(current)
            if continued is not None:
                wrist_margin = float(
                    np.min(
                        np.minimum(
                            continued[5:7] - self.lower[5:7],
                            self.upper[5:7] - continued[5:7],
                        )
                    )
                )
                if wrist_margin >= self.cfg["joint_margin_rad"]:
                    return continued
        seeds = [np.asarray(current)]
        seeds.append(0.5 * (self.lower + self.upper))
        if "ik_reference_seed" in self.cfg:
            seeds.append(np.asarray(self.cfg["ik_reference_seed"], dtype=float))
        for _ in range(max(0, self.cfg["ik_attempts"] - len(seeds))):
            seeds.append(
                np.clip(current + self.rng.normal(0, 0.45, 7), self.lower, self.upper)
            )
        solutions = []
        for seed in seeds:
            result = attempt(seed)
            if result is not None:
                wrist_margin = float(
                    np.min(
                        np.minimum(
                            result[5:7] - self.lower[5:7],
                            self.upper[5:7] - result[5:7],
                        )
                    )
                )
                solutions.append(
                    (wrist_margin, float(np.linalg.norm(result - current)), result)
                )
        if not solutions:
            return None
        safe = [item for item in solutions if item[0] >= self.cfg["joint_margin_rad"]]
        pool = safe or solutions
        if prefer_continuity:
            return min(pool, key=lambda item: (item[1], -item[0]))[2]
        return max(pool, key=lambda item: (item[0], -item[1]))[2]

    def _penetrating_contact_pairs(self, *, left_only=False) -> set[tuple[str, str]]:
        pairs = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist >= -0.001:
                continue
            names = tuple(
                sorted(
                    (
                        mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                        )
                        or "",
                        mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                        )
                        or "",
                    )
                )
            )
            if not left_only or any("left" in name for name in names):
                pairs.add(names)
        return pairs

    def _has_forbidden_collision(self, q, allow_cube=False, gripper=None):
        if gripper is None:
            self.data.qpos[self.qidx] = q
        else:
            self.mapper.set_qpos(self.data.qpos, np.r_[q, gripper], "right")
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            names = n1 + " " + n2
            pair = tuple(sorted((n1, n2)))
            if pair in self.baseline_left_contacts:
                continue
            if "finger_inner" in names and "finger_outer" in names:
                continue
            b1 = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[c.geom1]
                )
                or ""
            )
            b2 = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[c.geom2]
                )
                or ""
            )
            bodies = {b1, b2}
            if allow_cube and "orange_cube" in bodies and bodies & {
                "openarm_right_ee_inner_finger",
                "openarm_right_ee_outer_finger",
            }:
                continue
            if c.dist < -0.001:
                return True
        return False

    def plan(self, cube_position, cube_yaw, current):
        start = time.monotonic()
        deadline = start + self.cfg["planning_timeout_s"]
        candidates = []
        grasp = np.asarray(cube_position, dtype=float).copy()
        grasp[2] += self.cfg["grasp_center_z_offset_m"]
        for index, yaw in enumerate(yaw_candidates(cube_yaw)):
            c = GraspCandidate(index=index, yaw=yaw)
            candidates.append(c)
            if time.monotonic() >= deadline:
                c.failure_reasons.append("planning_timeout")
                continue
            rot = self.target_rotation(yaw)
            pre = grasp + np.array([0, 0, self.cfg["pregrasp_offset_m"]])
            qpre = self.solve_pose(pre, rot, current, deadline=deadline)
            if qpre is None:
                c.failure_reasons.append("pregrasp_ik")
                continue
            transit_steps = max(
                60,
                math.ceil(
                    float(np.max(np.abs(qpre - current)))
                    / self.cfg.get("transit_joint_step_rad", 0.04)
                ),
            )
            transit = [
                current + (qpre - current) * k / transit_steps
                for k in range(1, transit_steps + 1)
            ]
            if any(self._has_forbidden_collision(q) for q in transit):
                c.failure_reasons.append("home_to_pregrasp_collision")
                continue
            path = [qpre]
            q = qpre
            distance = float(pre[2] - grasp[2])
            steps = max(1, math.ceil(distance / self.cfg["approach_step_m"]))
            orientation_path = Slerp([0, 1], Rotation.from_matrix(np.stack([rot, rot])))
            opened = self.cfg["pregrasp_right_gripper_rad"]
            closed = self.cfg["approach_right_gripper_rad"]
            failed = False
            for k in range(1, steps + 1):
                pos = pre + (grasp - pre) * k / steps
                qnext = self.solve_pose(
                    pos,
                    orientation_path(k / steps).as_matrix(),
                    q,
                    prefer_continuity=True,
                    deadline=deadline,
                )
                if qnext is None:
                    c.failure_reasons.append(f"cartesian_ik_step_{k}")
                    failed = True
                    break
                gripper = opened + (closed - opened) * k / steps
                if self._has_forbidden_collision(qnext, allow_cube=True, gripper=gripper):
                    c.failure_reasons.append(f"collision_step_{k}")
                    failed = True
                    break
                if np.max(np.abs(qnext - q)) > self.cfg.get("max_ik_branch_jump_rad", 0.35):
                    c.failure_reasons.append(f"ik_branch_jump_step_{k}")
                    failed = True
                    break
                path.append(qnext)
                q = qnext
            if failed:
                continue
            lift = []
            lift_steps = max(1, math.ceil(self.cfg["lift_m"] / self.cfg["approach_step_m"]))
            for k in range(1, lift_steps + 1):
                pos = grasp + np.array([0.0, 0.0, self.cfg["lift_m"] * k / lift_steps])
                qnext = self.solve_pose(
                    pos, rot, q, prefer_continuity=True, deadline=deadline
                )
                if qnext is None:
                    c.failure_reasons.append(f"lift_ik_step_{k}")
                    failed = True
                    break
                if np.max(np.abs(qnext - q)) > self.cfg.get("max_ik_branch_jump_rad", 0.35):
                    c.failure_reasons.append(f"lift_branch_jump_step_{k}")
                    failed = True
                    break
                if self._has_forbidden_collision(qnext, allow_cube=True, gripper=closed):
                    c.failure_reasons.append(f"lift_collision_step_{k}")
                    failed = True
                    break
                lift.append(qnext)
                q = qnext
            if failed or len(lift) != lift_steps:
                if not failed:
                    c.failure_reasons.append("incomplete_lift_path")
                continue
            margin = np.minimum(q - self.lower, self.upper - q)
            if np.any(margin[5:7] < self.cfg["joint_margin_rad"]):
                c.failure_reasons.append("joint_limit_margin")
                continue
            c.transit_waypoints = [x.copy() for x in transit]
            c.approach_waypoints = [x.copy() for x in path[1:]]
            c.lift_waypoints = [x.copy() for x in lift]
            c.waypoints, c.reachable, c.collision_free = path, True, True
            wrist_margin = min(margin[5], margin[6])
            c.score = (
                -float(wrist_margin),
                float(np.linalg.norm(qpre - current)),
                float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:]))),
            )
        valid = rank_candidates(candidates)
        return (valid[0] if valid else None), candidates
