#!/usr/bin/env python3
"""Offline contact-dynamics pilot; never connects to robot or commits training data."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'official/openarm_mujoco/src')]
import mujoco
import numpy as np
import yaml
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.spatial.transform import Rotation,Slerp
from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.expert import MujocoGraspPlanner


class Pilot:
    def __init__(self,seed,output,xy_jitter=0.,yaw_jitter_deg=0.,table_xy_jitter=0.,table_yaw_jitter_deg=0.):
        bounds=np.array([xy_jitter,yaw_jitter_deg,table_xy_jitter,table_yaw_jitter_deg])
        if not np.isfinite(bounds).all() or np.any(bounds<0):raise ValueError('invalid randomization bounds')
        self.output=output;output.mkdir(parents=True,exist_ok=True)
        spec=mujoco.MjSpec.from_file(str(ROOT/'sim/scenes/act_fixed_20261015/scene.xml'))
        rng=np.random.default_rng(seed)
        cube_delta=rng.uniform(-xy_jitter,xy_jitter,2)
        cube_angle=float(rng.uniform(-yaw_jitter_deg,yaw_jitter_deg))
        table_delta=rng.uniform(-table_xy_jitter,table_xy_jitter,2)
        table_angle=float(rng.uniform(-table_yaw_jitter_deg,table_yaw_jitter_deg))
        table_rotation=Rotation.from_euler('z',table_angle,degrees=True)
        table_matrix=table_rotation.as_matrix()
        pivot=np.array(spec.body('table').pos).copy()
        for name in ('table','black_frame'):
            body=spec.body(name)
            body.pos=pivot+table_matrix@(np.array(body.pos)-pivot)+np.r_[table_delta,0.]
            old=Rotation.from_quat(np.array(body.quat)[[1,2,3,0]])
            body.quat=(table_rotation*old).as_quat()[[3,0,1,2]]
        self.randomization={'table_translation_xy_m':table_delta.tolist(),'table_yaw_deg':table_angle,
                            'cube_translation_in_table_xy_m':cube_delta.tolist(),'cube_yaw_relative_table_deg':cube_angle,
                            'cube_world_yaw_deg':cube_angle+table_angle,
                            'table_rotation_pivot_world_m':pivot.tolist(),
                            'camera_pose_randomized':False,'robot_base_randomized':False}
        (output/'randomization.json').write_text(json.dumps(self.randomization,indent=2)+'\n')
        # Set before compilation: MuJoCo precomputes ngravcomp. Mutating only
        # body_gravcomp after compile leaves that aggregate stale at zero.
        for body in spec.bodies:
            if body.name.startswith('openarm_'):body.gravcomp=1
        self.model=spec.compile()
        self.data=mujoco.MjData(self.model);self.mapper=JointResolver(self.model)
        mujoco.mj_resetDataKeyframe(self.model,self.data,self.model.key('physical_home_reference').id)
        self.mapper.set_qpos(self.data.qpos,np.zeros(8),'left');self.mapper.set_ctrl(self.data.ctrl,np.zeros(8),'left')
        # ROS H1 uses gravity compensation. This is ideal simulated compensation,
        # recorded explicitly, not an identified model of the hardware response.
        assert self.model.ngravcomp>0
        self.cfg=yaml.safe_load((ROOT/'configs/collection/openarm_cube_into_black_box_expert_params.yaml').read_text())
        self.cfg.update(pregrasp_right_gripper_rad=-.35,approach_right_gripper_rad=-.35)
        # Validate grasp locally; transport has an explicit orientation transition.
        self.cfg.pop('fixed_place_xy_m')
        self.contract=json.loads((ROOT/'sim/scenes/act_fixed_20261015/control_contract.json').read_text())
        self.limits=np.array(self.contract['position_limits_rad'])
        mujoco.mj_forward(self.model,self.data)
        self.site=self.model.site('right_grasp_frame').id
        self.cube=self.model.body('orange_cube').id
        self.cube_joint=self.model.body_jntadr[self.cube]
        self.cube_qadr=self.model.jnt_qposadr[self.cube_joint]
        nominal=self.data.qpos[self.cube_qadr:self.cube_qadr+3].copy()+np.r_[cube_delta,0.]
        self.data.qpos[self.cube_qadr:self.cube_qadr+3]=pivot+table_matrix@(nominal-pivot)+np.r_[table_delta,0.]
        self.cube_yaw=math.radians(cube_angle+table_angle)
        self.data.qpos[self.cube_qadr+3:self.cube_qadr+7]=[math.cos(self.cube_yaw/2),0,0,math.sin(self.cube_yaw/2)]
        mujoco.mj_forward(self.model,self.data)
        self.planner=MujocoGraspPlanner(self.model,self.mapper,self.cfg,seed,runtime_data=self.data)
        self.planner.lower=np.maximum(self.planner.lower,self.limits[:7,0])
        self.planner.upper=np.minimum(self.planner.upper,self.limits[:7,1])
        self.initial_qpos=self.data.qpos.copy();self.initial_ctrl=self.data.ctrl.copy();self.seed=seed
        self.home=self.state()[:8].copy();self.start_cube=self.data.xpos[self.cube].copy()
        self.samples=[];self.trace=[];self.next_sample=0.;self.peak_lift=0.;self.contacts_duration=0.
        self.continuous_hold=0.;self.finger_count=0
        self.grasp_relative=None
        self.last_q=self.home.copy();self.last_v=np.zeros(8);self.max_v=0.;self.max_a=0.

    def box_outcome(self):
        """Test full cube bounds in the actual rotated box frame."""
        box=self.model.body('black_frame').id
        rotation=self.data.xmat[box].reshape(3,3)
        local=rotation.T@(self.data.xpos[self.cube]-self.data.xpos[box])
        half=np.abs(rotation.T@self.data.xmat[self.cube].reshape(3,3))@np.full(3,.02)
        return bool(np.all(np.abs(local[:2])+half[:2]<=.06)),bool(abs(local[2]-half[2]-.01)<.002)

    def state(self):
        r,g=self.mapper.get_driver(self.data.qpos,'right');l,h=self.mapper.get_driver(self.data.qpos,'left')
        return np.r_[r,g,l,h]

    def contacts(self):
        fingers=set();forbidden=[]
        for c in self.data.contact:
            names=[self.model.body(self.model.geom_bodyid[x]).name or '' for x in (c.geom1,c.geom2)]
            if 'orange_cube' in names:
                fingers.update(n for n in names if n in ['openarm_right_ee_inner_finger','openarm_right_ee_outer_finger'])
                if any(n.startswith('openarm_') and n not in fingers for n in names):
                    forbidden.append(names)
                continue
            if c.dist>=-.001:continue  # inherited planner penetration criterion
            if any(n.startswith('openarm_') for n in names):
                if all('finger' in n for n in names):continue
                forbidden.append(names)
        self.finger_count=len(fingers)
        return len(fingers)==2,forbidden

    def step(self,action,phase):
        action=np.asarray(action)
        if np.any(action[:7]<self.limits[:7,0]) or np.any(action[:7]>self.limits[:7,1]) or not -.350001<=action[7]<=-.099999:
            raise RuntimeError('command outside unchanged real ACT range')
        dt=float(self.model.opt.timestep);v=(action-self.last_q)/dt;a=(v-self.last_v)/dt
        self.max_v=max(self.max_v,float(np.max(np.abs(v))));self.max_a=max(self.max_a,float(np.max(np.abs(a))))
        self.last_q=action.copy();self.last_v=v.copy()
        self.mapper.set_ctrl(self.data.ctrl,action,'right');mujoco.mj_step(self.model,self.data)
        bilateral,forbidden=self.contacts()
        if forbidden:raise RuntimeError('forbidden_contact:'+str(forbidden[:3]))
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('nonfinite physics')
        cube=self.data.xpos[self.cube].copy();self.peak_lift=max(self.peak_lift,float(cube[2]-self.start_cube[2]))
        if phase in {'LIFT','HOLD','TRANSPORT','PLACE_ABOVE_BOX'} and self.grasp_relative is not None:
            relative=self.data.site_xmat[self.site].reshape(3,3).T@(cube-self.data.site_xpos[self.site])
            if np.linalg.norm(relative-self.grasp_relative)>self.cfg['max_relative_drift_m']:
                raise RuntimeError(f'{phase}:grasp_slip_exceeded_existing_25mm_drift_bound')
        if bilateral and cube[2]-self.start_cube[2]>=self.cfg['hold_height_m']:
            self.continuous_hold+=dt;self.contacts_duration=max(self.contacts_duration,self.continuous_hold)
        else:self.continuous_hold=0.
        if self.data.time>=self.next_sample:
            self.samples.append({'time':float(self.data.time),'phase':phase,'state':self.state().tolist(),
                                 'action':np.r_[action,np.zeros(8)].tolist(),'cube':cube.tolist(),
                                 'bilateral':bilateral,'qpos':self.data.qpos.tolist()})
            self.next_sample+=1/30

    def follow(self,phase,points,min_duration=0.):
        points=np.asarray(points,dtype=float)
        points=np.vstack([self.last_q,points])
        spline=CubicSpline(np.linspace(0,1,len(points)),points,bc_type='natural')
        # Cubic interpolation may leave the bounded IK waypoint range. Use a
        # shape-preserving interpolant in that case; never clip joint commands
        # or relax the existing position limits to admit an overshooting path.
        probe=spline(np.linspace(0,1,2001))
        if np.any(probe[:,:7]<self.limits[:7,0]) or np.any(probe[:,:7]>self.limits[:7,1]):
            spline=PchipInterpolator(np.linspace(0,1,len(points)),points)
        def curve(u,derivative=0):
            u=np.asarray(u);s=10*u**3-15*u**4+6*u**5
            ds=30*u**2-60*u**3+30*u**4;dds=60*u-180*u**2+120*u**3
            if derivative==1:return spline(s,1)*ds[...,None]
            if derivative==2:return spline(s,2)*ds[...,None]**2+spline(s,1)*dds[...,None]
            return spline(s)
        u=np.linspace(0,1,2001)
        velocity=float(self.contract['max_velocity_rad_s']);acceleration=float(self.contract['max_acceleration_rad_s2'])
        duration=max(min_duration,1.02*np.max(np.abs(curve(u,1)))/velocity,
                     math.sqrt(1.02*np.max(np.abs(curve(u,2)))/acceleration),.1)
        count=math.ceil(duration/self.model.opt.timestep)
        for i in range(1,count+1):self.step(curve(i/count),phase)
        self.trace.append({'phase':phase,'time':float(self.data.time),'cube':self.data.xpos[self.cube].tolist(),
                           'tcp':self.data.site_xpos[self.site].tolist(),'bilateral':self.contacts()[0]})
        print(json.dumps(self.trace[-1]),flush=True)

    def cartesian(self,phase,target_pos,target_rotation,gripper):
        start=self.data.site_xpos[self.site].copy();start_r=self.data.site_xmat[self.site].reshape(3,3).copy()
        q=self.state()[:7].copy();path=[]
        n=max(20,math.ceil(np.linalg.norm(np.asarray(target_pos)-start)/.005))
        rotations=Slerp([0,1],Rotation.from_matrix([start_r,target_rotation]))
        for i in range(1,n+1):
            u=i/n;qn=self.planner.solve_pose(start+(np.asarray(target_pos)-start)*u,rotations(u).as_matrix(),q,prefer_continuity=True)
            if qn is None:raise RuntimeError(f'{phase}:IK at {i}/{n}')
            if np.max(np.abs(qn-q))>self.cfg['max_ik_branch_jump_rad']:raise RuntimeError(f'{phase}:IK branch discontinuity')
            if self.planner._has_forbidden_collision(qn,allow_cube=True,gripper=gripper):raise RuntimeError(f'{phase}:planned collision')
            path.append(np.r_[qn,gripper]);q=qn
        # A direct joint corridor removes redundant-IK oscillations. It is used
        # only when every sampled configuration passes the same collision test.
        # Dynamic contact/slip checks remain active during the entire rollout.
        shortcut=np.linspace(self.last_q,np.asarray(path[-1]),max(100,math.ceil(np.max(np.abs(path[-1]-self.last_q))/.005)))
        if all(not self.planner._has_forbidden_collision(q[:7],allow_cube=True,gripper=q[7]) for q in shortcut):
            self.follow(phase,[path[-1]])
        else:self.follow(phase,path)

    def run(self):
        result={'success':False,'dataset_committed':False,'ideal_gravity_compensation':True,
                'ngravcomp':self.model.ngravcomp}
        try:
            self.follow('STABILIZE',[self.home],1.)
            selected,candidates=self.planner.plan(self.data.xpos[self.cube],self.cube_yaw,self.state()[:7])
            result['candidate_failures']=[{'index':c.index,'reasons':c.failure_reasons} for c in candidates]
            if selected is None:raise RuntimeError('no_grasp_candidate')
            result['selected_candidate']=selected.index
            result['grasp_world_yaw_rad']=selected.yaw
            self.follow('PREGRASP',[np.r_[q,-.35] for q in selected.transit_waypoints])
            self.follow('APPROACH',[np.r_[q,-.35] for q in selected.approach_waypoints])
            self.follow('CLOSE',[np.r_[self.last_q[:7],-.10]],1.5)
            self.follow('VERIFY',[self.last_q],1.)
            if not self.contacts()[0]:raise RuntimeError('no_bilateral_grasp')
            self.grasp_relative=self.data.site_xmat[self.site].reshape(3,3).T@(self.data.xpos[self.cube]-self.data.site_xpos[self.site])
            lift=self.data.site_xpos[self.site].copy()+[0,0,.065]
            self.cartesian('LIFT',lift,self.data.site_xmat[self.site].reshape(3,3).copy(),-.10)
            self.follow('HOLD',[self.last_q],2.)
            if self.contacts_duration<self.cfg['hold_s']:raise RuntimeError('lift_hold_failed')
            # An angled carrying posture is selected for this low tabletop.
            rot=Rotation.from_euler('z',45,degrees=True).as_matrix()@Rotation.from_euler('y',45,degrees=True).as_matrix()
            box_position=self.data.xpos[self.model.body('black_frame').id].copy()
            self.cartesian('TRANSPORT',box_position+[0,0,.13],rot,-.10)
            if not self.contacts()[0]:raise RuntimeError('grasp_lost_before_planned_release')
            # Use measured object-to-TCP offset so the object is above the box.
            cube=self.data.xpos[self.cube].copy();target=self.data.site_xpos[self.site].copy()
            target+=box_position+[0,0,.08]-cube
            self.cartesian('PLACE_ABOVE_BOX',target,rot,-.10)
            if not self.contacts()[0]:raise RuntimeError('grasp_lost_before_planned_release')
            self.follow('RELEASE',[np.r_[self.last_q[:7],-.35]],1.5)
            self.follow('SETTLE',[self.last_q],2.)
            self.cartesian('RETREAT',self.data.site_xpos[self.site]+[0,0,.07],rot,-.35)
            self.follow('RETURN_HOME',[self.home],3.)
            self.follow('TAIL',[self.home],3.)
            cube=self.data.xpos[self.cube]
            # Entire cube must be inside the cavity, rest on the box floor,
            # be released, and arm must have returned to the fixed real HOME.
            inside,supported=self.box_outcome()
            home_error=float(np.max(np.abs(self.state()[:7]-self.home[:7])))
            self.contacts()
            result.update(success=inside and supported and self.finger_count==0 and home_error<=.03,
                          cube_inside=inside,cube_supported=supported,home_error_rad=home_error)
        except Exception as exc:
            result['failure']=str(exc)
        result.update(seed=self.seed,randomization=self.randomization,initial_qpos=self.initial_qpos.tolist(),initial_ctrl=self.initial_ctrl.tolist(),
                      duration_s=float(self.data.time),peak_lift_m=self.peak_lift,bilateral_lift_duration_s=self.contacts_duration,
                      max_command_velocity_rad_s=self.max_v,max_command_acceleration_rad_s2=self.max_a,
                      phases=self.trace,final_cube=self.data.xpos[self.cube].tolist())
        result['simulation_duration_limit_s']=None
        (self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        with (self.output/'trace.jsonl').open('w') as f:
            for sample in self.samples:f.write(json.dumps(sample)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='phases'},indent=2))
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--cube-xy-jitter-m',type=float,default=0.)
    p.add_argument('--cube-yaw-jitter-deg',type=float,default=0.)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use a new pilot output to preserve prior evidence')
    result=Pilot(a.seed,a.output,a.cube_xy_jitter_m,a.cube_yaw_jitter_deg).run();sys.exit(0 if result['success'] else 2)


if __name__=='__main__':main()
