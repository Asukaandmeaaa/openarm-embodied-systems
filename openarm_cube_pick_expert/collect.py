"""Deterministic actuator-only MuJoCo expert batch collector."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
from tqdm.auto import tqdm

from openarm_mujoco.v2 import JointResolver
from .expert import ExpertPhase, MujocoGraspPlanner
from .fsm import ExpertFSM
from .geometry import combine_action, right_gripper_target
from .guard import ActionGuard
from .recording import ExpertRecorder
from .task import PickCubeEvaluator
from .config import ROOT, SCENE, PARAMS, METADATA, sha256

class Collector:
    def __init__(self, output: Path, base_seed: int, render=True):
        self.cfg = yaml.safe_load(PARAMS.read_text())
        self.model = mujoco.MjModel.from_xml_path(str(SCENE)); self.data = mujoco.MjData(self.model)
        self.mapper = JointResolver(self.model); self.base_seed = int(base_seed); self.render = render
        self.key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "")
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
        self.cube_qadr = int(self.model.jnt_qposadr[self.model.body_jntadr[self.cube_body]])
        self.cube_dadr = int(self.model.jnt_dofadr[self.model.body_jntadr[self.cube_body]])
        # Keep the upstream scene/body name for compatibility, but make the
        # collected observations match the physical blue training object.
        self.cube_geom_ids = np.flatnonzero(self.model.geom_bodyid == self.cube_body)
        self.model.geom_rgba[self.cube_geom_ids] = np.array([0.04, 0.18, 0.90, 1.0])
        lower = np.r_[self.model.actuator_ctrlrange[9:17, 0], self.model.actuator_ctrlrange[1:9, 0]]
        upper = np.r_[self.model.actuator_ctrlrange[9:17, 1], self.model.actuator_ctrlrange[1:9, 1]]
        g = self.cfg["guard"]
        self.guard = ActionGuard(
            lower,
            upper,
            g["max_velocity"],
            g["max_acceleration"],
            g["max_step"],
            g["watchdog_timeout_s"],
            g["limit_policy"],
            g.get("target_response_time_s", 0.4),
        )
        self.fsm = ExpertFSM()
        meta = yaml.safe_load(METADATA.read_text()); meta["operation_type"] = "expert"
        meta["frequencies"] = {"action":{"arms":{"right":30,"left":30}},"obs":{"arms":{"right":30,"left":30}},"cameras":{"ceiling":30,"wrist_right":30}}
        meta["expert"] = {"scene":str(SCENE.relative_to(ROOT)),"scene_sha256":sha256(SCENE),"params_sha256":sha256(PARAMS),"no_weld":True,"actuator_control":True}
        self.recorder = ExpertRecorder(output, meta)
        self.attempts_path = Path(output) / "attempts.jsonl"
        self.renderers = {n:mujoco.Renderer(self.model, height=600, width=960) for n in ("ceiling","wrist_right")} if render else {}

    def state(self):
        r,rg=self.mapper.get_driver(self.data.qpos,"right"); l,lg=self.mapper.get_driver(self.data.qpos,"left")
        return combine_action(np.r_[r,rg],np.r_[l,lg])

    def command(self, target, now):
        self.guard.submit(np.asarray(target,dtype=np.float32),now); result=self.guard.step(self.state(),now)
        self.mapper.set_ctrl(self.data.ctrl,result.action[:8],"right"); self.mapper.set_ctrl(self.data.ctrl,result.action[8:],"left")
        return result.action

    def frames(self):
        out={}
        for key,camera in (("ceiling","camera_ceiling"),("wrist_right","camera_wrist_right")):
            renderer=self.renderers[key]; renderer.update_scene(self.data,camera=camera); out[key]=renderer.render().copy()
        return out

    def cube_contact(self):
        return bool(self.cube_contact_bodies())

    def bilateral_contact(self):
        return set(self.cube_contact_bodies()) == {
            "openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger"
        }

    def cube_contact_bodies(self):
        contacts=[]
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            bodies={int(self.model.geom_bodyid[c.geom1]),int(self.model.geom_bodyid[c.geom2])}
            if self.cube_body not in bodies: continue
            other=(bodies-{self.cube_body}).pop() if len(bodies)>1 else -1
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,other) or ""
            if "openarm_right_ee_" in name: contacts.append(name)
        return sorted(set(contacts))

    def reset(self, episode_seed):
        self.recorder.cancel()
        mujoco.mj_resetDataKeyframe(self.model,self.data,self.key); rng=np.random.default_rng(episode_seed)
        self.fsm.reset(float(self.data.time))
        center=np.asarray(self.cfg["cube_xy_center_m"],dtype=float); half=np.asarray(self.cfg["cube_xy_half_range_m"],dtype=float)
        cube_xy=center+rng.uniform(-half,half)
        self.data.qpos[self.cube_qadr:self.cube_qadr+3]=[cube_xy[0],cube_xy[1],1.05]
        yaw=rng.uniform(-math.radians(15),math.radians(15)); self.data.qpos[self.cube_qadr+3:self.cube_qadr+7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        if "fixed_place_xy_m" in self.cfg:
            self.place_xy=np.asarray(self.cfg["fixed_place_xy_m"],dtype=float)
        else:
            place_half=np.asarray(self.cfg["place_xy_half_range_m"],dtype=float)
            self.place_xy=None
            for _ in range(100):
                candidate=center+rng.uniform(-place_half,place_half)
                if np.linalg.norm(candidate-cube_xy)>=self.cfg["place_min_distance_m"]:
                    self.place_xy=candidate;break
            if self.place_xy is None:
                direction=np.array([1.0,-1.0]);direction/=np.linalg.norm(direction)
                self.place_xy=np.clip(cube_xy+self.cfg["place_min_distance_m"]*direction,center-place_half,center+place_half)
        self.data.qvel.fill(0); self.guard.reset(); left=self.state()[8:]
        if "left_gripper_open_rad" in self.cfg:left[7]=self.cfg["left_gripper_open_rad"]
        target=combine_action(np.r_[self.state()[:7],self.cfg["pregrasp_right_gripper_rad"]],left)
        for _ in range(int(self.cfg["stabilize_s"]*1000)):
            self.command(target,self.data.time); mujoco.mj_step(self.model,self.data)
        self.fsm.transition(ExpertPhase.PLAN, float(self.data.time), "physics_stabilized")
        return yaw,target

    def record_attempt(self, details):
        """Durably append one success/failure/interruption diagnostic record."""
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(details, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.attempts_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)

    def run_episode(self,index,record=True,candidate_index=None):
        seed=self.base_seed+index; yaw,target=self.reset(seed); reset_state=self.state().copy(); cube0=self.data.qpos[self.cube_qadr:self.cube_qadr+7].copy()
        reset_z=float(cube0[2]); phase_times={item["phase"]:item["time_s"] for item in self.fsm.transitions}
        planner=MujocoGraspPlanner(self.model,self.mapper,self.cfg,seed,runtime_data=self.data); selected,candidates=planner.plan(cube0[:3],yaw,reset_state[:7])
        preferred_index=(candidate_index if candidate_index is not None
                         else self.cfg.get("preferred_candidate_index"))
        if preferred_index is not None:
            preferred=[item for item in candidates if item.index==int(preferred_index) and item.reachable and item.collision_free]
            # An explicitly requested candidate is used by simulation
            # preflight.  Falling back here would attribute another branch's
            # result to the candidate being tested.
            selected=preferred[0] if preferred else None
        details={"seed":seed,"reset_index":index,"objects":{"orange_cube":{"pose":cube0.tolist(),"color":"blue"}},"place_target_xy_m":self.place_xy.tolist(),"initial_qpos":reset_state.tolist(),"candidate_failures":[{"index":c.index,"yaw":c.yaw,"reasons":c.failure_reasons} for c in candidates],"phase_timestamps":phase_times,"scene_sha256":sha256(SCENE),"params_sha256":sha256(PARAMS)}
        if selected is None:
            self.fsm.transition(ExpertPhase.FAIL, float(self.data.time), "all_candidates_failed")
            details["phase_timestamps"][ExpertPhase.FAIL.value] = float(self.data.time)
            details["phase_transitions"] = list(self.fsm.transitions)
            details.update(success_reason=None,fail_reason="unreachable",selected_candidate=None,max_lift_height_m=0.0,hold_duration_s=0.0,action_guard=dict(self.guard.stats)); return False,details
        details["selected_candidate"]={"index":selected.index,"yaw":selected.yaw,"score":list(selected.score)}
        # Stabilization has its own target and velocity history. The expert
        # trajectory must start from the measured reset state, otherwise the
        # stale Guard integrator moves the nominally stationary left arm.
        self.guard.reset()
        if record:self.recorder.start(index)
        start=float(self.data.time); next_record=start; tracker=PickCubeEvaluator(reset_z,self.cfg["hold_height_m"],self.cfg["hold_s"],self.cfg["tcp_distance_m"],self.cfg.get("max_relative_drift_m",.025))
        right_open=self.cfg["pregrasp_right_gripper_rad"]; right_close=self.cfg.get("forced_right_close_rad",right_gripper_target(self.cfg["object_width_m"],preload=self.cfg["aperture_preload_m"],contact_radius=self.cfg["finger_contact_radius_m"],reference_width=self.cfg["reference_width_m"],reference_angle=self.cfg["reference_right_gripper_rad"]))
        left=reset_state[8:].copy(); left[7]=self.cfg.get("left_gripper_open_rad",0.0)
        rot=planner.target_rotation(selected.yaw)
        site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,"right_grasp_frame")
        stream_step=float(self.cfg["stream_waypoint_s"])
        settle_count=max(1,math.ceil(self.cfg["stream_settle_s"]/stream_step))
        def with_settle(waypoints,count=settle_count):
            if waypoints:waypoints.extend([waypoints[-1].copy() for _ in range(count)])
            return waypoints
        # Grasp orientation depends on the cube yaw, but carrying to the box
        # should use one repeatable wrist orientation. This avoids selecting a
        # grasp that is valid at the cube but becomes singular near the box.
        def cartesian_path(start_pos,end_pos,start_q,gripper,rotation=rot,
                           start_rotation=rot):
            distance=float(np.linalg.norm(np.asarray(end_pos)-np.asarray(start_pos)))
            steps=max(1,math.ceil(distance/self.cfg["transport_step_m"]));out=[];q=np.asarray(start_q,dtype=float)
            deadline=time.monotonic()+float(self.cfg.get("planning_timeout_s",12.0))
            orientation_path=Slerp([0.0,1.0], Rotation.from_matrix([start_rotation,rotation]))
            for k in range(1,steps+1):
                if time.monotonic() >= deadline:
                    return None
                pos=np.asarray(start_pos)+(np.asarray(end_pos)-np.asarray(start_pos))*k/steps
                alpha=min(1.0, k / max(1, steps))
                interp=orientation_path(alpha).as_matrix()
                qnext=planner.solve_pose(pos,interp,q,prefer_continuity=True,deadline=deadline)
                if qnext is None or np.max(np.abs(qnext-q))>self.cfg.get("max_ik_branch_jump_rad",.35):return None
                out.append(np.r_[qnext,gripper]);q=qnext
            return with_settle(out)
        q_pre=selected.waypoints[0]
        transit_path=[np.r_[qn,right_open] for qn in selected.transit_waypoints]
        transit=with_settle([item.copy() for item in transit_path])
        approach=[]; approach_end=self.cfg["approach_right_gripper_rad"]
        for k,qn in enumerate(selected.approach_waypoints,1):
            approach.append(np.r_[qn,right_open+(approach_end-right_open)*k/len(selected.approach_waypoints)])
        approach_step=float(self.cfg["approach_waypoint_s"])
        approach_settle=max(1,math.ceil(self.cfg["approach_settle_s"]/approach_step))
        approach_path=[item.copy() for item in approach]
        with_settle(approach,approach_settle)
        grasp_q=selected.waypoints[-1].copy()
        phases=[(ExpertPhase.MOVE_PREGRASP,transit),(ExpertPhase.APPROACH,approach),(ExpertPhase.CLOSE,[np.r_[grasp_q,right_close]])]
        phases += [(ExpertPhase.VERIFY_GRASP,[np.r_[grasp_q,right_close]]),(ExpertPhase.LIFT,[]),(ExpertPhase.HOLD,[]),
                   (ExpertPhase.TRANSPORT,[]),(ExpertPhase.PLACE,[]),(ExpertPhase.RELEASE,[]),
                   (ExpertPhase.RETREAT,[]),(ExpertPhase.RETURN_HOME,[])]
        success=False; fail_reason="timeout"; frame_index=0; phase_snapshots={};live_hold=None;verified_rel=None;verify_contact_since=None;pick_hold_complete=False;released=False;lift_path=None;transport_path=None;place_path=None;place_stable_since=None;home_stable_since=None
        stream_phases={ExpertPhase.MOVE_PREGRASP,ExpertPhase.APPROACH,ExpertPhase.LIFT,ExpertPhase.TRANSPORT,ExpertPhase.PLACE,ExpertPhase.RETREAT,ExpertPhase.RETURN_HOME}
        try:
            for phase,waypoints in phases:
                self.fsm.transition(phase, float(self.data.time), f"enter_{phase.value.lower()}")
                phase_times[phase.value]=float(self.data.time)
                if phase == ExpertPhase.CLOSE:
                    for _ in range(0):
                        live_cube=self.data.xpos[self.cube_body].copy(); desired=live_cube.copy();desired[2]+=self.cfg["grasp_center_z_offset_m"]
                        aligned=planner._solve(desired,rot,self.state()[:7])
                        if aligned is None: break
                        align_action=combine_action(np.r_[aligned,self.cfg["approach_right_gripper_rad"]],left)
                        for _ in range(400):
                            action=self.command(align_action,self.data.time);mujoco.mj_step(self.model,self.data)
                            if record and self.data.time+1e-9>=next_record:
                                self.recorder.sample(int(self.data.time*1e9),action,self.state(),self.frames());next_record+=1/self.cfg["record_hz"]
                if phase == ExpertPhase.LIFT:
                    live_base=self.data.site_xpos[site].copy();live_q=self.state()[:7].astype(float);waypoints=[]
                    lift_steps=max(1,math.ceil(self.cfg["lift_m"]/self.cfg["approach_step_m"]))
                    lift_deadline=time.monotonic()+float(self.cfg.get("planning_timeout_s",12.0))
                    for k in range(1,lift_steps+1):
                        if time.monotonic() >= lift_deadline:
                            break
                        dz=self.cfg["lift_m"]*k/lift_steps
                        qn=planner._solve(live_base+[0,0,dz],rot,live_q,deadline=lift_deadline)
                        if qn is None or np.max(np.abs(qn-live_q))>self.cfg.get("max_ik_branch_jump_rad",.35):break
                        waypoints.append(np.r_[qn,right_close]);live_q=qn
                    if len(waypoints)!=lift_steps:
                        fail_reason="incomplete_live_lift_path"; break
                    lift_path=[item.copy() for item in waypoints]
                    waypoints=with_settle(waypoints);live_hold=waypoints[-1].copy()
                elif phase == ExpertPhase.HOLD and live_hold is not None:
                    waypoints=[live_hold]
                elif phase == ExpertPhase.TRANSPORT:
                    start_pos=self.data.site_xpos[site].copy();cube_now=self.data.xpos[self.cube_body].copy();end_pos=start_pos.copy();end_pos[:2]+=self.place_xy-cube_now[:2]
                    # Clear the table and the box rim before the long
                    # horizontal transfer. The old nearby-placement task did
                    # not need this intermediate clearance waypoint.
                    high_pos=start_pos.copy()
                    high_pos[2]=max(float(start_pos[2]), float(self.cfg.get("transport_clearance_z_m",1.22)))
                    mid_pos=high_pos.copy()
                    mid_pos[:2]=high_pos[:2]+(
                        self.place_xy-high_pos[:2]
                    )*float(self.cfg.get("transport_midpoint_fraction",0.5))
                    first=cartesian_path(start_pos,high_pos,self.state()[:7],right_close)
                    if first is None:
                        first_q=planner.solve_pose(high_pos,rot,self.state()[:7],
                                                   prefer_continuity=True)
                        if first_q is None:
                            fail_reason="transport_ik_failed";break
                        n=max(30,math.ceil(float(np.max(np.abs(first_q-self.state()[:7])))/0.015))
                        first=[np.r_[self.state()[:7]+(first_q-self.state()[:7])*k/n,right_close] for k in range(1,n+1)]
                    second=cartesian_path(high_pos,mid_pos,first[-1][:7],right_close)
                    if second is None:
                        end_q=planner.solve_pose(mid_pos,rot,first[-1][:7],
                                                 prefer_continuity=True)
                        if end_q is None:
                            fail_reason="transport_ik_failed";break
                        n=max(30,math.ceil(float(np.max(np.abs(end_q-first[-1][:7])))/0.015))
                        second=[np.r_[first[-1][:7]+(end_q-first[-1][:7])*k/n,right_close] for k in range(1,n+1)]
                    third=cartesian_path(mid_pos,end_pos,second[-1][:7],right_close)
                    if third is None:
                        end_q=planner.solve_pose(end_pos,rot,second[-1][:7],
                                                 prefer_continuity=True)
                        if end_q is None:
                            fail_reason="transport_ik_failed";break
                        n=max(30,math.ceil(float(np.max(np.abs(end_q-second[-1][:7])))/0.015))
                        third=[np.r_[second[-1][:7]+(end_q-second[-1][:7])*k/n,right_close] for k in range(1,n+1)]
                    waypoints=first[:-settle_count]+second[:-settle_count]+third
                    transport_path=[item.copy() for item in waypoints[:-settle_count]]
                elif phase == ExpertPhase.PLACE:
                    start_pos=self.data.site_xpos[site].copy();end_pos=start_pos.copy()
                    if not self.cfg.get("release_above_box",False):
                        end_pos[2]+=reset_z-float(self.data.xpos[self.cube_body,2])
                    waypoints=cartesian_path(start_pos,end_pos,self.state()[:7],right_close)
                    if waypoints is None:fail_reason="place_ik_failed";break
                    place_path=[item.copy() for item in waypoints[:-settle_count]]
                elif phase == ExpertPhase.RELEASE:
                    waypoints=[np.r_[self.state()[:7],right_open]]
                elif phase == ExpertPhase.RETREAT:
                    if not place_path or not transport_path or not lift_path:fail_reason="missing_reverse_path";break
                    # First lift clear of the placed cube, then reverse the
                    # validated transport corridor before returning home. This
                    # prevents the arm from sweeping through the placed object.
                    retreat=[*reversed(place_path),*reversed(transport_path),
                             *reversed(lift_path),*reversed(approach_path),
                             *reversed(transit_path)]
                    waypoints=with_settle([np.r_[item[:7],right_open] for item in retreat])
                elif phase == ExpertPhase.RETURN_HOME:
                    current=self.state()[:7].astype(float);home=reset_state[:7].astype(float);steps=max(30,math.ceil(float(np.max(np.abs(home-current)))/.015))
                    return_settle=max(1,math.ceil(self.cfg["return_settle_s"]/stream_step))
                    waypoints=with_settle([np.r_[current+(home-current)*k/steps,reset_state[7]] for k in range(1,steps+1)],return_settle)
                inner=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"openarm_right_ee_inner_finger");outer=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"openarm_right_ee_outer_finger")
                phase_snapshots[phase.value]={"cube":self.data.xpos[self.cube_body].tolist(),"tcp":self.data.site_xpos[site].tolist(),"right":self.state()[:8].tolist(),"contact":self.cube_contact(),"contact_bodies":self.cube_contact_bodies(),"finger_centers":{"inner":self.data.xpos[inner].tolist(),"outer":self.data.xpos[outer].tolist()}}
                for waypoint in waypoints:
                    durations={ExpertPhase.CLOSE:self.cfg["close_timeout_s"],ExpertPhase.VERIFY_GRASP:1.0,ExpertPhase.HOLD:self.cfg["hold_s"]+.5,ExpertPhase.RELEASE:self.cfg["release_s"]}
                    if phase==ExpertPhase.APPROACH:phase_step=approach_step
                    elif phase==ExpertPhase.RETREAT:phase_step=self.cfg["retreat_waypoint_s"]
                    elif phase in (ExpertPhase.TRANSPORT,ExpertPhase.PLACE):phase_step=self.cfg["transport_waypoint_s"]
                    else:phase_step=stream_step
                    combined=combine_action(waypoint,left); deadline=self.data.time+(phase_step if phase in stream_phases else durations[phase])
                    while self.data.time<deadline:
                        action=self.command(combined,self.data.time);mujoco.mj_step(self.model,self.data)
                        if record and self.data.time+1e-9>=next_record:
                            self.recorder.sample(int(self.data.time*1e9),action,self.state(),self.frames());next_record+=1/self.cfg["record_hz"]
                        cube_z=float(self.data.qpos[self.cube_qadr+2]); relative=self.data.xpos[self.cube_body]-self.data.site_xpos[site];tcp_dist=float(np.linalg.norm(relative));vz=float(self.data.qvel[self.cube_dadr+2])
                        if phase == ExpertPhase.VERIFY_GRASP:
                            if self.bilateral_contact():
                                if verify_contact_since is None: verify_contact_since=float(self.data.time)
                                if self.data.time-verify_contact_since>=self.cfg.get("verify_contact_s",.2):
                                    verified_rel=relative.copy();break
                            else: verify_contact_since=None
                        if phase == ExpertPhase.HOLD and verified_rel is not None and tracker.update(now_s=self.data.time,cube_z=cube_z,bilateral_contact=self.bilateral_contact(),tcp_distance_m=tcp_dist,relative_drift_m=float(np.linalg.norm(relative-verified_rel)),cube_vertical_velocity_m_s=vz):
                            pick_hold_complete=True;break
                        if phase == ExpertPhase.PLACE and "placement_stable_s" in self.cfg:
                            cube_xyz=self.data.xpos[self.cube_body].copy();center=np.asarray(self.cfg.get("success_place_xy_m",self.place_xy),dtype=float)
                            in_box=bool(np.all(np.abs(cube_xyz[:2]-center)<=self.cfg["place_tolerance_m"]) and abs(float(cube_xyz[2])-self.cfg["box_cube_center_z_m"])<=self.cfg["box_cube_center_z_tolerance_m"])
                            stable=in_box and abs(vz)<=0.03
                            if stable:
                                if place_stable_since is None:place_stable_since=float(self.data.time)
                            else:place_stable_since=None
                        if phase == ExpertPhase.RETURN_HOME and "home_stable_s" in self.cfg:
                            live_state=self.state();joint_error=max(float(np.max(np.abs(live_state[:7]-reset_state[:7]))),float(np.max(np.abs(live_state[8:15]-reset_state[8:15]))))
                            if joint_error<=.03:
                                if home_stable_since is None:home_stable_since=float(self.data.time)
                            else:home_stable_since=None
                        error=np.abs(self.state()-combined)
                        if np.max(error[:7])<.008 and np.max(error[8:15])<.008 and error[7]<.03 and error[15]<.03 and phase not in stream_phases and phase not in (ExpertPhase.VERIFY_GRASP,ExpertPhase.HOLD,ExpertPhase.RELEASE): break
                        if self.data.time-start>self.cfg["episode_timeout_s"]: break
                    if pick_hold_complete and phase==ExpertPhase.HOLD:break
                    if self.data.time-start>self.cfg["episode_timeout_s"]:break
                if phase == ExpertPhase.VERIFY_GRASP and verified_rel is None:
                    fail_reason="bilateral_contact_not_stable";break
                if phase in (ExpertPhase.MOVE_PREGRASP, ExpertPhase.APPROACH):
                    pregrasp_cube_displacement = float(
                        np.linalg.norm(self.data.xpos[self.cube_body] - cube0[:3])
                    )
                    if pregrasp_cube_displacement > self.cfg.get(
                        "max_pregrasp_cube_displacement_m", math.inf
                    ):
                        fail_reason="cube_disturbed_before_grasp";break
                if phase == ExpertPhase.HOLD and not pick_hold_complete:
                    fail_reason="lift_hold_failed";break
                if phase == ExpertPhase.PLACE and "placement_stable_s" in self.cfg and self.cfg.get("placement_stable_s",0.0)>0:
                    place_stable_duration=0.0 if place_stable_since is None else float(self.data.time)-place_stable_since
                    if place_stable_duration+1e-9<self.cfg["placement_stable_s"]:fail_reason="placement_not_stable";break
                if phase == ExpertPhase.RELEASE:
                    released=not self.cube_contact()
                    if not released:fail_reason="release_failed";break
                if phase == ExpertPhase.RETURN_HOME:
                    cube_final=self.data.xpos[self.cube_body].copy();success_xy=np.asarray(self.cfg.get("success_place_xy_m",self.place_xy),dtype=float);place_delta=np.abs(cube_final[:2]-success_xy);place_error=float(np.linalg.norm(place_delta));height_error=abs(float(cube_final[2]-self.cfg.get("box_cube_center_z_m",reset_z)));live_state=self.state();home_error=max(float(np.max(np.abs(live_state[:7]-reset_state[:7]))),float(np.max(np.abs(live_state[8:15]-reset_state[8:15]))))
                    if self.cfg.get("place_axis_tolerance",False):place_valid=bool(np.all(place_delta<=self.cfg["place_tolerance_m"]))
                    else:place_valid=place_error<=self.cfg["place_tolerance_m"]
                    height_limit=self.cfg.get("box_cube_center_z_tolerance_m",self.cfg["place_height_tolerance_m"])
                    home_stable_duration=0.0 if home_stable_since is None else float(self.data.time)-home_stable_since
                    # Releasing above a target changes the motion, never the
                    # task success definition.  The cube must actually finish
                    # at the requested placement pose.
                    placement_ok = place_valid and height_error<=height_limit
                    success=bool(pick_hold_complete and released and placement_ok and home_error<=.03 and home_stable_duration+1e-9>=self.cfg.get("home_stable_s",0.0))
                    fail_reason=None if success else "final_place_or_home_validation_failed"
                if self.data.time-start>self.cfg["episode_timeout_s"]:break
            if success and self.cfg.get("post_success_tail_s",0) > 0:
                # Continue converging arm joints to home during the static
                # tail, but preserve the released gripper commands. Freezing
                # the merely threshold-valid live state left too little margin
                # for strict 30 Hz actuator replay.
                tail_action=self.state().copy()
                tail_action[:7]=reset_state[:7]
                tail_action[8:15]=reset_state[8:15]
                settle_deadline=self.data.time+3.0
                while self.data.time<settle_deadline:
                    action=self.command(tail_action,self.data.time);mujoco.mj_step(self.model,self.data)
                tail_reference=self.state().copy()
                tail_deadline=self.data.time+float(self.cfg["post_success_tail_s"])
                tail_max_delta=0.0
                while self.data.time<tail_deadline:
                    action=self.command(tail_action,self.data.time);mujoco.mj_step(self.model,self.data)
                    tail_max_delta=max(tail_max_delta,float(np.max(np.abs(self.state()-tail_reference))))
                    if record and self.data.time+1e-9>=next_record:
                        self.recorder.sample(int(self.data.time*1e9),action,self.state(),self.frames());next_record+=1/self.cfg["record_hz"]
                tail_cube=self.data.xpos[self.cube_body].copy();tail_delta=np.abs(tail_cube[:2]-np.asarray(self.cfg.get("success_place_xy_m",self.place_xy),dtype=float));tail_box_valid=bool(np.all(tail_delta<=self.cfg["place_tolerance_m"]) and abs(float(tail_cube[2])-self.cfg.get("box_cube_center_z_m",float(tail_cube[2])))<=self.cfg.get("box_cube_center_z_tolerance_m",self.cfg["place_height_tolerance_m"]))
                if tail_max_delta>self.cfg.get("static_joint_tolerance_rad",.005) or not tail_box_valid:
                    success=False;fail_reason="static_tail_failed"
            terminal = ExpertPhase.SUCCESS if success else ExpertPhase.FAIL
            self.fsm.transition(terminal, float(self.data.time), "success_criteria_met" if success else fail_reason)
            phase_times[terminal.value]=float(self.data.time)
            details.update(success_reason="pick_hold_place_release_return_home" if success else None,fail_reason=fail_reason,**tracker.evidence(),phase_timestamps=phase_times,phase_transitions=list(self.fsm.transitions),phase_snapshots=phase_snapshots,final_cube_pose=self.data.qpos[self.cube_qadr:self.cube_qadr+7].tolist(),final_tcp_pose=self.data.site_xpos[site].tolist(),place_error_m=locals().get("place_error"),place_height_error_m=locals().get("height_error"),placement_stable_duration_s=locals().get("place_stable_duration"),pregrasp_cube_displacement_m=locals().get("pregrasp_cube_displacement"),return_home_error_rad=locals().get("home_error"),home_stable_duration_s=locals().get("home_stable_duration"),static_tail_duration_s=self.cfg.get("post_success_tail_s",0) if success else 0.0,static_tail_max_joint_delta_rad=locals().get("tail_max_delta"),released=released,action_guard=dict(self.guard.stats),gripper_calibration={"object_width_m":self.cfg["object_width_m"],"aperture_preload_m":self.cfg["aperture_preload_m"],"real_reference_joint_rad":self.cfg["reference_right_gripper_rad"],"mujoco_commanded_joint_rad":right_close})
            details.update(getattr(self, "_episode_context", {}))
            if record:
                if success:self.recorder.finish(True,details)
                else:self.recorder.cancel()
            return success,details
        except BaseException:
            if record:self.recorder.cancel()
            raise

    def close(self):
        for renderer in self.renderers.values(): renderer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--episodes",type=int,default=5,help="total number of successful episodes desired (resumes an existing output)");p.add_argument("--max-attempts",type=int);p.add_argument("--seed",type=int,default=20260902);p.add_argument("--output",type=Path,default=Path("data/raw/openarm_cube_pick/expert_pilot"));p.add_argument("--no-render",action="store_true")
    a=p.parse_args(); collector=Collector(a.output,a.seed,not a.no_render)

    # Reconstruct lightweight run counters without retaining the very large
    # per-frame hold evidence from every attempts.jsonl record in memory.
    prior_attempts=0; prior_failures=Counter(); prior_seeds=[]; used_indices=[]
    if collector.attempts_path.exists():
        with collector.attempts_path.open(encoding="utf-8") as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():continue
                try: item=json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid attempts.jsonl line {line_number}") from exc
                prior_attempts+=1
                if item.get("fail_reason"):prior_failures[item["fail_reason"]]+=1
                if "seed" in item:prior_seeds.append(int(item["seed"]))
                if "reset_index" in item:used_indices.append(int(item["reset_index"]))

    for entry in collector.recorder.entries:
        if "reset_index" in entry:used_indices.append(int(entry["reset_index"]))
    inferred_bases={seed-index for seed,index in zip(prior_seeds,used_indices[:len(prior_seeds)])}
    if inferred_bases and inferred_bases!={a.seed}:
        raise RuntimeError(
            f"cannot resume {a.output}: existing base seed(s)={sorted(inferred_bases)}, requested={a.seed}"
        )

    initial_successes=len(collector.recorder.entries)
    next_attempt=max(used_indices,default=-1)+1
    max_attempts=a.max_attempts or a.episodes*10
    results=[]
    progress=tqdm(
        total=a.episodes,
        initial=min(initial_successes,a.episodes),
        desc="Expert successes",
        unit="episode",
        dynamic_ncols=True,
        smoothing=0.1,
    )
    if initial_successes:
        tqdm.write(
            f"Resuming {a.output}: {initial_successes}/{a.episodes} successes, "
            f"{prior_attempts} attempts; next_index={next_attempt}, "
            f"next_seed={a.seed+next_attempt}"
        )
    try:
        for i in range(next_attempt,max_attempts):
            try:
                ok,info=collector.run_episode(i,record=not a.no_render)
            except BaseException as exc:
                collector.recorder.cancel()
                collector.record_attempt({"seed":collector.base_seed+i,"reset_index":i,"success_reason":None,"fail_reason":"interrupted" if isinstance(exc,KeyboardInterrupt) else "exception","exception_type":type(exc).__name__,"scene_sha256":sha256(SCENE),"params_sha256":sha256(PARAMS)})
                raise
            results.append(info);collector.record_attempt(info)
            successes=initial_successes+sum(r.get("success_reason") is not None for r in results)
            attempts=prior_attempts+len(results)
            if ok:progress.update(1)
            progress.set_postfix(attempts=attempts,rate=f"{successes/max(1,attempts):.1%}",last_fail=info.get("fail_reason") or "none",refresh=True)
            tqdm.write(f"episode={i} seed={info['seed']} success={ok} fail={info.get('fail_reason')} max_lift={info.get('max_lift_height_m',0):.4f} hold={info.get('hold_duration_s',0):.3f}")
            if successes>=a.episodes:break
    finally:
        progress.close();collector.close()
    successes=initial_successes+sum(r.get("success_reason") is not None for r in results)
    failures=prior_failures+Counter(r.get("fail_reason") for r in results if r.get("fail_reason"))
    attempts=prior_attempts+len(results)
    summary={"attempts":attempts,"successes":successes,"requested_successes":a.episodes,"success_rate":successes/max(1,attempts),"failures":dict(failures),"seeds":prior_seeds+[r["seed"] for r in results]}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"expert_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));raise SystemExit(0 if successes>=a.episodes else 2)

if __name__=="__main__":main()
