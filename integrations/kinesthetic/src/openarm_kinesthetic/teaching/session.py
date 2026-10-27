"""Interactive MuJoCo virtual-handle collection session (never touches ROS/CAN)."""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np

from ..config import load_yaml
from ..control import GripperController, RightArmPassiveMode
from ..initialization.collision import initialization_contacts
from ..initialization.execute import _segment_durations, gravity_feedforward, reference_at
from ..initialization.pipeline import load_initialization_config, run_initialization, simulation_start_q
from ..initialization.teaching_ready import rotation_error_rad
from ..paths import workspace_path
from ..recording import EpisodeRecorder, RecorderStateError
from ..planning.kinematics import contact_failures, normalized_joint_margin
from ..scene.model import load_model, reset_task_ready
from ..task import EvaluationSample, TaskEvaluator
from .gamepad import GamepadMapper, LinuxJoystick
from .gamepad_overlay import GamepadOverlay
from .virtual_handle import VirtualHandle


_SPECIAL_KEYS = {262: "Right", 263: "Left", 264: "Down", 265: "Up"}
_JOG_DIRECTIONS = {
    "Up": np.array([1.0, 0.0, 0.0]),
    "Down": np.array([-1.0, 0.0, 0.0]),
    "Left": np.array([0.0, 1.0, 0.0]),
    "Right": np.array([0.0, -1.0, 0.0]),
    "U": np.array([0.0, 0.0, 1.0]),
    "D": np.array([0.0, 0.0, -1.0]),
}

# The interactive teaching station uses the same table convention as the
# initialization pipeline: X is the 0.40 m robot-to-table depth and Y is the
# 0.60 m lateral width.  The older real_table_v1 scene has those extents swapped
# and is retained only for its historical regression gates.
DEFAULT_INTERACTIVE_SCENE = "sim/scenes/prior_table_v1.xml"


def _configured_simulation_start_q(initialization_config) -> np.ndarray:
    """Return the configured simulation start pose."""
    return simulation_start_q(initialization_config)


def _set_right_arm_state(model, data, mapping, q: np.ndarray) -> None:
    """Install a simulation state, then let planning read it back as current q."""
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    data.ctrl[mapping.right_actuators] = np.asarray(q, dtype=float)
    mujoco.mj_forward(model, data)


def _prepare_interactive_initialization(model, data, mapping):
    """Plan from the configured simulation start and restore that start for playback."""
    import mujoco

    config = load_initialization_config()
    start_q = _configured_simulation_start_q(config)
    planning_data = mujoco.MjData(model)
    reset_task_ready(model, planning_data)
    _set_interactive_right_gripper_opening(model, planning_data, -0.35)
    _set_right_arm_state(model, planning_data, mapping, start_q)
    # `run_initialization` receives the state read from MuJoCo.  The reference
    # task-ready joints remain only an FK orientation and IK seed.
    current_q = planning_data.qpos[mapping.right_qpos].copy()
    report = run_initialization(
        execute=False,
        start_q=current_q,
        config=config,
        model=model,
        data=planning_data,
        mapping=mapping,
    )
    if not report["success"]:
        failure = report.get("state", {}).get("error") or "SIM_INITIALIZATION_PLAN_FAILED"
        raise RuntimeError(f"interactive initialization rejected: {failure}")
    reset_task_ready(model, data)
    _set_right_arm_state(model, data, mapping, current_q)
    waypoints = [np.asarray(q, dtype=float) for q in report["planner"]["waypoints"]]
    durations = _segment_durations(waypoints, config["planning"]["time_scaling"])
    return config, report, waypoints, durations


class _RealtimeStepScheduler:
    """Convert wall-clock progress into bounded fixed-size physics substeps."""

    def __init__(self, timestep_s: float, now_s: float, *, maximum_catch_up_s: float = 0.05):
        if timestep_s <= 0 or maximum_catch_up_s < timestep_s:
            raise ValueError("invalid realtime scheduler timing")
        self.timestep_s = float(timestep_s)
        self.maximum_catch_up_s = float(maximum_catch_up_s)
        self.last_wall_s = float(now_s)
        self.accumulator_s = self.timestep_s

    def steps_due(self, now_s: float) -> int:
        elapsed_s = max(0.0, float(now_s) - self.last_wall_s)
        self.last_wall_s = float(now_s)
        self.accumulator_s = min(
            self.accumulator_s + elapsed_s,
            self.maximum_catch_up_s,
        )
        steps = int((self.accumulator_s + 1e-12) / self.timestep_s)
        self.accumulator_s -= steps * self.timestep_s
        return steps


def _key_name(code: int) -> str:
    if code == 32:
        return "Space"
    if code == 256:
        return "Esc"
    if code in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[code]
    return chr(code) if 0 <= code < 256 else str(code)


def _keyboard_jog_delta(key: str, step_m: float) -> np.ndarray | None:
    direction = _JOG_DIRECTIONS.get(key)
    return None if direction is None else direction * float(step_m)


def _initialize_perturbation(model, data, camera, option, perturb) -> None:
    """Initialize a selected body against a scene with valid camera frusta."""
    import mujoco

    scene = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom + 100))
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        perturb,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    mujoco.mjv_initPerturb(model, data, scene, perturb)


def _clear_viewer_perturbation_force(data) -> None:
    """Discard the native viewer wrench; VirtualHandle supplies the only wrench."""
    data.xfrc_applied[:] = 0.0


def _sync_mocap_target_to_site(data, mocap_id: int, site_id: int) -> None:
    import mujoco

    data.mocap_pos[mocap_id] = data.site_xpos[site_id]
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site_xmat[site_id])


def _set_interactive_right_gripper_opening(model, data, target: float) -> None:
    """Set both coupled right fingers for the pre-recording interactive state."""
    import mujoco

    for name in ("openarm_right_finger_joint1", "openarm_right_finger_joint2"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"MuJoCo object is missing: {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(target)
    mujoco.mj_forward(model, data)


def _mocap_rotation(data, mocap_id: int) -> np.ndarray:
    import mujoco

    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.mocap_quat[mocap_id])
    return rotation.reshape(3, 3)


def _integrate_local_rotation(rotation: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    omega_dt = np.asarray(angular_velocity, dtype=float) * float(dt)
    angle = float(np.linalg.norm(omega_dt))
    if angle <= 1e-12:
        return np.asarray(rotation, dtype=float).copy()
    axis = omega_dt / angle
    skew = np.asarray([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    delta = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return np.asarray(rotation, dtype=float) @ delta


def run_interactive(
    output: Path,
    *,
    width: int = 320,
    height: int = 240,
    jog_step_m: float = 0.01,
    gamepad_device: str = "auto",
    scene: str = DEFAULT_INTERACTIVE_SCENE,
) -> None:  # pragma: no cover - operator GUI
    import mujoco
    import mujoco.viewer

    if not 0 < jog_step_m <= 0.05:
        raise ValueError("jog_step_m must be in (0, 0.05]")
    gamepad_config=load_yaml("gamepad_control.yaml");gamepad=LinuxJoystick.open_configured(gamepad_device,gamepad_config)
    gamepad_mapper=GamepadMapper(gamepad_config);gamepad_name=gamepad.name if gamepad is not None else "not_connected"
    gamepad_overlay=GamepadOverlay(gamepad_config)
    if gamepad is None:
        print(f"[gamepad] not connected (request={gamepad_device!r}); mouse/keyboard control remains available",flush=True)
    else:
        print(
            f"[gamepad] connected path={gamepad.path!r} name={gamepad.name!r}; "
            "hold LT while moving a stick (LT is not a toggle)",
            flush=True,
        )
    model=load_model(scene);data=mujoco.MjData(model);mapping=reset_task_ready(model,data)
    initialization_config,initialization_report,initialization_waypoints,initialization_durations=_prepare_interactive_initialization(model,data,mapping)
    initialization_goal_q=np.asarray(initialization_report["ik"]["q_teaching_ready"],dtype=float)
    initialization_target_position=np.asarray(initialization_report["teaching_ready"]["tcp_position_world_m"],dtype=float)
    initialization_target_rotation=np.asarray(initialization_report["teaching_ready"]["tcp_orientation_matrix"],dtype=float)
    initialization_preview_s=float(initialization_config["execution"].get("start_preview_s",1.0))
    initialization_trajectory_s=float(sum(initialization_durations))
    initialization_settle_s=float(initialization_config["execution"].get("settle_s",0.5))
    initialization_total_s=initialization_preview_s+initialization_trajectory_s+initialization_settle_s
    initialization_elapsed_s=0.0;initialization_failure=None;manual_ready=False
    initialization_inverse_data=mujoco.MjData(model)
    print(
        f"[initialization] planned from current sim q={np.round(initialization_report['start_q'],4).tolist()} "
        f"to teaching-ready; viewer playback={initialization_total_s:.2f}s",
        flush=True,
    )
    from PIL import Image
    renderer=mujoco.Renderer(model,height=height,width=width)
    recording_scene_option=mujoco.MjvOption();recording_scene_option.geomgroup[5]=0
    passive=RightArmPassiveMode(model,mapping);gripper=GripperController("mujoco")
    _set_interactive_right_gripper_opening(model,data,gripper.OPEN_TARGET)
    recorder=EpisodeRecorder(output);evaluator=None;recording=False;paused=False;last_callback={}
    pending=[];pending_jogs=[];pending_gamepad_events=[]
    interrupt_requested=False

    def request_interrupt(_signum, _frame) -> None:
        nonlocal interrupt_requested
        interrupt_requested=True

    previous_sigint_handler=signal.signal(signal.SIGINT,request_interrupt)

    def handle_operator_key(key: str, now: float) -> None:
        nonlocal recording,paused,evaluator,reference_relative
        if not manual_ready and key != "Esc":
            return
        normalized=gripper.normalize(key)
        if now-last_callback.get(normalized,-1e9)>0.25:gripper.key_up(normalized)
        event=gripper.key_down(normalized,now);last_callback[normalized]=now;pending.append(event)
        if not event.accepted:return
        if normalized=="R" and not recording:
            recorder.begin({"source":"human_virtual_handle","virtual_handle_inputs":["mocap_mouse_target","keyboard_cartesian_jog","machenike_g3v2_gamepad"],"mouse_target_execution":"compliant_6d_wrench_with_nullspace_posture","operator_target_visible_in_recorded_images":False,"keyboard_jog_frame":"world","keyboard_jog_step_m":jog_step_m,"gamepad":{"device_name":gamepad_name,"mapping_config":"configs/gamepad_control.yaml","activation":"hold_lt_deadman","input_loss_behavior":"release_virtual_handle"},"task":"Put the blue cube inside the fixed black box with the right arm, then return the right arm to the task-ready pose.","initialization_in_episode":False})
            evaluator=TaskEvaluator.from_model(model,data,mapping,data.qpos[mapping.right_qpos].copy());reference_relative=None;recording=True
        elif normalized=="S" and recording:
            try:recorder.finish_success(evaluator.result())
            except RecorderStateError:pass
            else:recording=False
        elif normalized=="F" and recording:recorder.finish_failure("operator_marked_failure");recording=False
        elif normalized=="X" and recording:recorder.cancel();recording=False
        elif normalized=="Space":
            paused=gripper.paused
            if recording and evaluator is not None and "pause_during_active_episode" not in evaluator.failures:evaluator.failures.append("pause_during_active_episode")
        elif normalized=="Esc" and recording and evaluator is not None:
            evaluator.failures.append("software_stop_during_active_episode")

    def callback(code: int) -> None:
        key=_key_name(code);now=time.monotonic()
        jog_delta=_keyboard_jog_delta(key,jog_step_m)
        if jog_delta is not None:
            pending_jogs.append((now,key,jog_delta))
            return
        handle_operator_key(key,now)

    with mujoco.viewer.launch_passive(model,data,key_callback=callback) as viewer:
        target_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_handle_target")
        target_mocap=int(model.body_mocapid[target_body])
        _sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site);mujoco.mj_forward(model,data)
        viewer.perturb.select=target_body
        _initialize_perturbation(model,data,viewer.cam,viewer.opt,viewer.perturb)
        handle=VirtualHandle(
            mapping.grasp_site,
            right_qpos=mapping.right_qpos,
            right_dofs=mapping.right_dof,
            preferred_q=initialization_goal_q,
        )
        keyboard_jog_active=False;keyboard_jog_deadline=0.0;mouse_was_active=False;gamepad_was_active=False;handle_input_mode="idle"
        next_joint=next_camera=0.0;reference_relative=None
        scheduler=_RealtimeStepScheduler(float(model.opt.timestep),time.monotonic())
        gamepad_command=gamepad_mapper.update(
            np.zeros(8,dtype=np.int16),np.zeros(15,dtype=np.uint8),connected=False
        )
        while viewer.is_running() and not gripper.stop_latched and not interrupt_requested:
            started=time.monotonic()
            physics_steps=scheduler.steps_due(started)
            if gamepad is not None:
                gamepad_events=gamepad.poll()
                if recording:pending_gamepad_events.extend(event for event in gamepad_events if not event.initial)
                gamepad_command=gamepad_mapper.update(gamepad.axes,gamepad.buttons,connected=gamepad.connected)
                if gamepad_command.open_pressed:handle_operator_key("O",time.monotonic())
                if gamepad_command.open_released:gripper.key_up("O")
                if gamepad_command.close_pressed:handle_operator_key("C",time.monotonic())
                if gamepad_command.close_released:gripper.key_up("C")
            else:
                gamepad_command=gamepad_mapper.update(np.zeros(8,dtype=np.int16),np.zeros(15,dtype=np.uint8),connected=False)

            for _ in range(physics_steps):
                data.qfrc_applied[:]=0
                _clear_viewer_perturbation_force(data)
                if not manual_ready:
                    if initialization_failure is None:
                        trajectory_time=max(0.0,initialization_elapsed_s-initialization_preview_s)
                        reference=reference_at(
                            initialization_waypoints,
                            initialization_durations,
                            min(trajectory_time,initialization_trajectory_s),
                        )
                        data.ctrl[mapping.right_actuators]=reference
                        data.qfrc_applied[mapping.right_dof]=gravity_feedforward(
                            model,data,mapping,initialization_inverse_data
                        )
                        data.ctrl[mapping.right_gripper_actuator]=gripper.effective_target
                        mujoco.mj_step(model,data)
                        initialization_elapsed_s+=float(model.opt.timestep)
                        forbidden=initialization_contacts(model,data)
                        if forbidden:
                            initialization_failure="|".join(sorted(forbidden))
                        elif initialization_elapsed_s>=initialization_total_s:
                            position_error=float(np.linalg.norm(data.site_xpos[mapping.grasp_site]-initialization_target_position))
                            orientation_error=rotation_error_rad(data.site_xmat[mapping.grasp_site].reshape(3,3),initialization_target_rotation)
                            joint_error=float(np.max(np.abs(data.qpos[mapping.right_qpos]-initialization_goal_q)))
                            execution_config=initialization_config["execution"]
                            if (
                                position_error>float(execution_config["final_position_tolerance_m"])
                                or orientation_error>float(execution_config["final_orientation_tolerance_rad"])
                                or joint_error>float(execution_config["final_joint_tolerance_rad"])
                            ):
                                initialization_failure=(
                                    f"final_tolerance pos={position_error:.4f} "
                                    f"rot={orientation_error:.4f} joint={joint_error:.4f}"
                                )
                            else:
                                passive.enable();manual_ready=True
                                handle.release(data);_sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                                print(
                                    f"[initialization] INITIALIZATION_REACHED -> MANUAL_TEACHING "
                                    f"pos_error={position_error:.6f}m rot_error={orientation_error:.6f}rad "
                                    f"joint_error={joint_error:.6f}rad",
                                    flush=True,
                                )
                    else:
                        data.ctrl[mapping.right_actuators]=data.qpos[mapping.right_qpos]
                        mujoco.mj_step(model,data)
                    _sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                    pending.clear();pending_jogs.clear();pending_gamepad_events.clear()
                    continue
                passive.apply_gravity_compensation(data)
                if pending_jogs and not gamepad_command.active:
                    combined_delta=np.sum([entry[2] for entry in pending_jogs],axis=0)
                    pending_jogs.clear()
                    handle.engage_at_current(data)
                    handle.set_target(data.site_xpos[mapping.grasp_site]+combined_delta,data.site_xmat[mapping.grasp_site].reshape(3,3))
                    data.mocap_pos[target_mocap]=handle.target_position
                    mujoco.mju_mat2Quat(data.mocap_quat[target_mocap],handle.target_rotation.ravel())
                    keyboard_jog_active=bool(np.linalg.norm(combined_delta)>0)
                    keyboard_jog_deadline=float(data.time)+0.5
                if gamepad_command.active:
                    if not gamepad_was_active:
                        handle.engage_at_current(data)
                    target=handle.target_position+gamepad_command.linear_velocity_world_m_s*float(model.opt.timestep)
                    offset=target-data.site_xpos[mapping.grasp_site]
                    distance=float(np.linalg.norm(offset))
                    if distance>gamepad_mapper.maximum_target_error:
                        target=data.site_xpos[mapping.grasp_site]+offset*(gamepad_mapper.maximum_target_error/distance)
                    rotation=_integrate_local_rotation(handle.target_rotation,gamepad_command.angular_velocity_local_rad_s,float(model.opt.timestep))
                    handle.set_target(target,rotation)
                    data.mocap_pos[target_mocap]=target
                    mujoco.mju_mat2Quat(data.mocap_quat[target_mocap],rotation.ravel())
                mouse_active=int(viewer.perturb.active)!=0 and not gamepad_command.active
                if paused:
                    handle.release();keyboard_jog_active=False;handle_input_mode="paused";_sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                elif gamepad_command.active:
                    keyboard_jog_active=False;handle_input_mode="machenike_g3v2_gamepad";handle.apply(model,data)
                elif gamepad_was_active:
                    handle.release();keyboard_jog_active=False;handle_input_mode="gamepad_stopped";_sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                elif mouse_active:
                    keyboard_jog_active=False;handle_input_mode="mocap_mouse_target"
                    handle.set_target(data.mocap_pos[target_mocap],_mocap_rotation(data,target_mocap));handle.apply(model,data)
                elif mouse_was_active:
                    handle.release();keyboard_jog_active=False;handle_input_mode="idle";_sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                elif keyboard_jog_active:
                    handle_input_mode="keyboard_cartesian_jog";handle.apply(model,data)
                    target_error=float(np.linalg.norm(handle.target_position-data.site_xpos[mapping.grasp_site]))
                    if target_error<=0.0015 or float(data.time)>=keyboard_jog_deadline:
                        handle.release();keyboard_jog_active=False;_sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                else:
                    handle.release();handle_input_mode="idle";_sync_mocap_target_to_site(data,target_mocap,mapping.grasp_site)
                mouse_was_active=mouse_active;gamepad_was_active=gamepad_command.active
                data.ctrl[mapping.right_gripper_actuator]=gripper.effective_target
                mujoco.mj_step(model,data);now=float(data.time)
                contacts=[False,False]
                for c in data.contact[:data.ncon]:
                    bodies={int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])}
                    for i,name in enumerate(("openarm_right_ee_inner_finger","openarm_right_ee_outer_finger")):
                        finger=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                        if mapping.cube_body in bodies and finger in bodies:contacts[i]=True
                gripper.update(float(data.qpos[mapping.right_gripper_qpos]),bilateral_contact=all(contacts))
                if recording and evaluator is not None:
                    left_error=float(np.max(np.abs(data.qpos[mapping.left_qpos])))
                    if left_error>0.03 and "left_arm_departed_park" not in evaluator.failures:evaluator.failures.append("left_arm_departed_park")
                    lower=model.jnt_range[model.dof_jntid[mapping.right_dof],0];upper=model.jnt_range[model.dof_jntid[mapping.right_dof],1]
                    if np.any(data.qpos[mapping.right_qpos]<lower) or np.any(data.qpos[mapping.right_qpos]>upper):evaluator.failures.append("right_joint_hard_limit")
                    relative=data.qpos[mapping.cube_qpos:mapping.cube_qpos+3]-data.site_xpos[mapping.grasp_site]
                    if all(contacts) and reference_relative is None:reference_relative=relative.copy()
                    drift=float(np.linalg.norm(relative-reference_relative)) if reference_relative is not None else 0.0
                    evaluator.update(EvaluationSample(now,tuple(data.qpos[mapping.cube_qpos:mapping.cube_qpos+3]),tuple(data.qvel[mapping.cube_dof:mapping.cube_dof+3]),tuple(contacts),tuple(data.qpos[mapping.right_qpos]),relative_grasp_drift_m=drift,forbidden_collision=bool(contact_failures(model,data))))
                    recorder.append("physics",now,{"qpos_right":data.qpos[mapping.right_qpos].tolist(),"qvel_right":data.qvel[mapping.right_dof].tolist()})
                    if now>=next_joint:
                        state=mapping.dataset_vector(data.qpos);action=mapping.dataset_vector(data.qpos,right_gripper_command=gripper.effective_target);action[8:16]=0
                        recorder.append("joint_observation",now,{"state":state.tolist()});recorder.append("demonstration_action",now,{"action":action.tolist()})
                        contact_records=[]
                        for index,c in enumerate(data.contact[:data.ncon]):
                            force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
                            contact_records.append({"geom1":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),"geom2":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),"distance_m":float(c.dist),"force6":force.tolist()})
                        recorder.append("sidecar",now,{"right_qvel":data.qvel[mapping.right_dof].tolist(),"right_actuator_torque":data.qfrc_actuator[mapping.right_dof].tolist(),"contacts":contact_records,"cube_pose":data.qpos[mapping.cube_qpos:mapping.cube_qpos+7].tolist(),"cube_velocity":data.qvel[mapping.cube_dof:mapping.cube_dof+6].tolist(),"tcp_pose":{"xyz":data.site_xpos[mapping.grasp_site].tolist(),"xmat":data.site_xmat[mapping.grasp_site].tolist()},"virtual_handle":{"input_mode":handle_input_mode,"active":handle.active,"target_xyz":handle.target_position.tolist()},"gamepad":{"connected":bool(gamepad is not None and gamepad.connected),"deadman_pressed":gamepad_command.deadman_pressed,"active":gamepad_command.active,"stop_latched":gamepad_command.stop_latched,"axes_raw":gamepad.axes.astype(int).tolist() if gamepad is not None else [],"buttons_raw":gamepad.buttons.astype(int).tolist() if gamepad is not None else []},"gripper_state":gripper.state.value,"task_phase":evaluator.phase,"tcp_table_clearance_m":float(data.site_xpos[mapping.grasp_site,2]-0.40),"right_joint_margin_normalized":normalized_joint_margin(model,mapping,data.qpos[mapping.right_qpos]),"forbidden_collisions":sorted(contact_failures(model,data)),"source_clocks":{"simulation_s":now,"host_monotonic_s":time.monotonic()}})
                        next_joint=now+0.01
                    if now>=next_camera:
                        for key,camera_name in (("ceiling","ceiling"),("wrist_right","camera_wrist_right")):
                            renderer.update_scene(data,camera=camera_name,scene_option=recording_scene_option);rgb=renderer.render().copy()
                            image_dir=recorder.active/"images"/key;image_dir.mkdir(parents=True,exist_ok=True)
                            relative=Path("images")/key/f"{now:012.6f}.jpg";Image.fromarray(rgb,"RGB").save(recorder.active/relative,format="JPEG",quality=90)
                            recorder.append("camera",now,{"key":key,"encoding":"JPEG","decoded_color":"RGB","resolution_wh":[width,height],"relative_path":str(relative)})
                        next_camera=now+1/30
                    while pending:
                        event=pending.pop(0);recorder.append("keyboard_events",event.raw_timestamp_s,event.to_dict())
                    while pending_gamepad_events:
                        event=pending_gamepad_events.pop(0);recorder.append("gamepad_events",event.host_monotonic_s,event.to_dict())
                else:
                    pending.clear();pending_gamepad_events.clear()
            initialization_status=(
                f"INIT FAILED: {initialization_failure}" if initialization_failure else
                (f"INITIALIZING {min(100.0,100.0*initialization_elapsed_s/initialization_total_s):.0f}% "
                 f"(manual controls locked)" if not manual_ready else "MANUAL_TEACHING ready; press R to record")
            )
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "OpenArm kinesthetic\nG3V2: hold LT=move | left stick=X/Y | D-pad up/down=Z | right stick=pose\nA=open Y=close RB/RT=stop | Mouse and keyboard remain available",
                f"gamepad={gamepad_name} connected={bool(gamepad is not None and gamepad.connected)}\n"
                f"LT_raw={int(gamepad.axes[gamepad_mapper.axis['lt']]) if gamepad is not None and gamepad_mapper.axis['lt'] < len(gamepad.axes) else 'n/a'} "
                f"deadman={gamepad_command.deadman_pressed} active={gamepad_command.active} stop={gamepad_command.stop_latched}\n"
                f"physics_steps/frame={physics_steps} v_world={np.round(gamepad_command.linear_velocity_world_m_s,3).tolist()} "
                f"w_local={np.round(gamepad_command.angular_velocity_local_rad_s,3).tolist()}\n"
                f"mode={initialization_status}\n"
                f"gripper={gripper.state.value} recording={recording} paused={paused}\ncontract={evaluator.phase if evaluator else 'IDLE'}",
            ))
            viewport=viewer.viewport
            if viewport is not None and viewport.width>=GamepadOverlay.WIDTH and viewport.height>=GamepadOverlay.HEIGHT+20:
                overlay_axes=gamepad.axes if gamepad is not None else np.zeros(8,dtype=np.int16)
                overlay_buttons=gamepad.buttons if gamepad is not None else np.zeros(15,dtype=np.uint8)
                overlay_image=gamepad_overlay.render(
                    overlay_axes,
                    overlay_buttons,
                    connected=bool(gamepad is not None and gamepad.connected),
                    active=gamepad_command.active,
                    stop_latched=gamepad_command.stop_latched,
                    now_s=time.monotonic(),
                )
                overlay_left=min(245,max(0,int(viewport.width)-GamepadOverlay.WIDTH))
                overlay_viewport=mujoco.MjrRect(
                    overlay_left,20,GamepadOverlay.WIDTH,GamepadOverlay.HEIGHT
                )
                viewer.set_images((overlay_viewport,overlay_image))
            viewer.sync();remaining=model.opt.timestep-(time.monotonic()-started)
            if remaining>0:time.sleep(remaining)
    if gamepad is not None:gamepad.close()
    renderer.close()
    signal.signal(signal.SIGINT,previous_sigint_handler)
    if interrupt_requested:print("[session] Ctrl+C received; viewer closed cleanly",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=workspace_path("data","pilot"));parser.add_argument("--width",type=int,default=320);parser.add_argument("--height",type=int,default=240);parser.add_argument("--jog-step-mm",type=float,default=10.0);parser.add_argument("--gamepad-device",default="auto",help="auto, off, or /dev/input/js* path");parser.add_argument("--scene",default=DEFAULT_INTERACTIVE_SCENE,help="workspace-relative MuJoCo scene (default: prior_table_v1 with 0.40 m X depth x 0.60 m Y width)");args=parser.parse_args();run_interactive(args.output,width=args.width,height=args.height,jog_step_m=args.jog_step_mm/1000.0,gamepad_device=args.gamepad_device,scene=args.scene)


if __name__=="__main__":main()
