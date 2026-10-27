"""Contact-dynamic automatic expert for the provisional real_table_v1 scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import load_yaml
from ..planning.box_path import BoxPathPlan, plan_box_path
from ..planning.kinematics import contact_failures, solve_site_pose
from ..scene.model import load_model, reset_task_ready
from ..task import EvaluationSample, TaskEvaluator, TaskResult


@dataclass
class ExpertRun:
    seed: int
    times: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    cube_poses: np.ndarray
    right_qvel: np.ndarray
    result: TaskResult
    commands_1000hz: np.ndarray
    box_path_plan: dict


class AutomaticExpert:
    """Uses normal position actuators and physical finger contacts; no weld/qpos shortcut."""

    def __init__(self, seed: int = 20260911):
        import mujoco

        self.mujoco = mujoco; self.seed = int(seed); self.model = load_model()
        self.data = mujoco.MjData(self.model); self.mapping = reset_task_ready(self.model, self.data)
        self.ready = self.data.qpos[self.mapping.right_qpos].copy()
        self.open_target = -0.55; self.close_target = -0.10
        self._plan()

    def _plan(self) -> None:
        scratch = self.mujoco.MjData(self.model); reset_task_ready(self.model, scratch)
        cfg = load_yaml("layout_scan.yaml")
        scene = load_yaml("scene_real_table_v1.yaml")
        table_xy = np.asarray(scene["table"]["center_xy_world_m"]["value"], dtype=float)
        cube_xy = table_xy + np.asarray(scene["cube"]["source_center_xy_table_m"]["value"], dtype=float)
        box_xy = table_xy + np.asarray(scene["black_box"]["center_xy_table_m"]["value"], dtype=float)
        rotation = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
        self.pregrasp = solve_site_pose(
            self.model, scratch, self.mapping, np.r_[cube_xy, 0.546], rotation, self.ready,
            tolerance_m=float(cfg["ik_position_tolerance_m"]), iterations=int(cfg["ik_iterations"]),
            margin_rad=float(cfg["joint_margin_rad"]),
        ).q
        self.grasp = solve_site_pose(
            self.model, scratch, self.mapping, np.r_[cube_xy, 0.456], rotation, self.pregrasp,
            tolerance_m=float(cfg["ik_position_tolerance_m"]), iterations=int(cfg["ik_iterations"]),
            margin_rad=float(cfg["joint_margin_rad"]),
        ).q
        self.box_path: BoxPathPlan = plan_box_path(self.model, scratch, self.mapping, self.ready, box_xy, cfg)
        self.above_box = self.box_path.above_box_q
        self.place = self.box_path.place_q
        self.retreat = self.box_path.retreat_q

    def _finger_contacts(self) -> tuple[bool, bool]:
        cube = self.mapping.cube_body; found = [False, False]
        finger_names = ("openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger")
        finger_ids = [self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name) for name in finger_names]
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            for index, finger in enumerate(finger_ids):
                if cube in bodies and finger in bodies: found[index] = True
        return bool(found[0]), bool(found[1])

    def run(self) -> ExpertRun:
        self.mapping = reset_task_ready(self.model, self.data)
        evaluator = TaskEvaluator.from_model(
            self.model, self.data, self.mapping, self.ready
        )
        times=[]; states=[]; actions=[]; phases=[]; cubes=[]; qvel=[]; commands=[]
        time_s = 0.0
        reference_relative = None

        def segment(name: str, target: np.ndarray, gripper: float, duration_s: float) -> None:
            nonlocal time_s, reference_relative
            start = self.data.ctrl[self.mapping.right_actuators].copy(); start_g = float(self.data.ctrl[self.mapping.right_gripper_actuator])
            count = int(round(duration_s / self.model.opt.timestep))
            for step in range(count):
                u = (step + 1) / count; blend = 10*u**3 - 15*u**4 + 6*u**5
                command = start + blend * (target - start); effective_g = start_g + blend * (gripper - start_g)
                self.data.ctrl[self.mapping.right_actuators] = command
                self.data.ctrl[self.mapping.right_gripper_actuator] = effective_g
                self.mujoco.mj_step(self.model, self.data); time_s += float(self.model.opt.timestep)
                cube_xyz = self.data.qpos[self.mapping.cube_qpos:self.mapping.cube_qpos+3].copy()
                cube_vel = self.data.qvel[self.mapping.cube_dof:self.mapping.cube_dof+3].copy()
                contacts=self._finger_contacts();relative=cube_xyz-self.data.site_xpos[self.mapping.grasp_site]
                if all(contacts) and reference_relative is None:reference_relative=relative.copy()
                drift=float(np.linalg.norm(relative-reference_relative)) if reference_relative is not None else 0.0
                evaluator.update(EvaluationSample(
                    time_s, tuple(cube_xyz), tuple(cube_vel), self._finger_contacts(),
                    tuple(self.data.qpos[self.mapping.right_qpos]), relative_grasp_drift_m=drift,
                    forbidden_collision=bool(contact_failures(self.model, self.data)),
                ))
                if step % 10 == 0:
                    times.append(time_s); states.append(self.mapping.dataset_vector(self.data.qpos))
                    actions.append(self.mapping.dataset_vector(self.data.qpos, right_gripper_command=effective_g))
                    actions[-1][:7] = command.astype(np.float32)
                    actions[-1][8:16] = 0.0
                    phases.append(name); cubes.append(self.data.qpos[self.mapping.cube_qpos:self.mapping.cube_qpos+7].copy())
                    qvel.append(self.data.qvel[self.mapping.right_dof].copy())
                commands.append(np.r_[command, effective_g])

        sequence = [
            ("STABILIZE", self.ready, self.open_target, 1.0),
            ("PREGRASP", self.pregrasp, self.open_target, 2.0),
            ("APPROACH", self.grasp, self.open_target, 2.0),
            ("CLOSE", self.grasp, self.close_target, 1.5),
            ("LIFT", self.ready, self.close_target, 2.0),
            ("LIFT_HOLD", self.ready, self.close_target, 2.2),
            ("TRANSPORT", self.above_box, self.close_target, 8.0),
            ("PLACE", self.place, self.close_target, 3.0),
            ("PLACE_STABLE", self.place, self.close_target, 1.0),
            ("RELEASE", self.place, self.open_target, 2.0),
            ("RETREAT", self.retreat, self.open_target, 3.0),
            ("RETURN_READY", self.ready, self.open_target, 8.0),
            ("READY_AND_TAIL", self.ready, self.open_target, 2.5),
        ]
        for args in sequence: segment(*args)
        return ExpertRun(
            seed=self.seed, times=np.asarray(times), states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32), phases=np.asarray(phases), cube_poses=np.asarray(cubes),
            right_qvel=np.asarray(qvel), result=evaluator.result(), commands_1000hz=np.asarray(commands, dtype=np.float32),
            box_path_plan=self.box_path.as_dict(),
        )
