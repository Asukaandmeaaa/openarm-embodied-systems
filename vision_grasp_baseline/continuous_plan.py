"""Plan and validate the entire pick/place/zero timeline without motion goals."""
from .gripper_settings import (OPEN_POSITION,CLOSE_POSITION,TRANSIT_POSITION,COLLECTION_POLICY,
                               require_current_plan,require_closed_transit)

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from moveit_msgs.srv import GetStateValidity
from rosidl_runtime_py.convert import message_to_ordereddict
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
import yaml

from .collection_io import write_json
from .continuous_timing import controller_bounds
from .grasp import GraspNode, FINGER
from .hover import NAMES, TCP, pose_matrix, box
from .fast_fk import ChainFK
from .linear_place_plan import plan_linear,release_pose
from .waypoint_place_test import scene_attachment
from .table_envelope import planning_scene,admit_scene,with_top_clearance,with_edge_clearance
from .scene_estimation import stabilize_scene,UnstableSceneError
from .pickup_clearance import PICKUP_CLEARANCE_M,padded_cube,require_pickup_clearance
from scripts.control_ownership.h2 import validated_token

BASE=Path(__file__).resolve().parent
TABLE_EDGE_CLEARANCE_M=.030


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_limits(model):
    from ament_index_python.packages import get_package_share_directory
    config=Path(get_package_share_directory('openarm_bimanual_moveit_config'))/'config/openarm_v2.0/joint_limits.yaml'
    configured=yaml.safe_load(config.read_text())['joint_limits']
    xml=ET.parse(model).getroot()
    rows=[]
    for name in [*NAMES,FINGER]:
        cfg=configured[name];urdf=xml.find(f"joint[@name='{name}']/limit")
        if urdf is None or not all(cfg.get('has_'+k+'_limits') for k in ('velocity','acceleration','jerk')):
            raise ValueError('Explicit velocity/acceleration/jerk limits required: '+name)
        row=dict(velocity=min(float(urdf.get('velocity')),cfg['max_velocity']),
                 acceleration=cfg['max_acceleration'],jerk=cfg['max_jerk'],
                 lower=float(urdf.get('lower')),upper=float(urdf.get('upper')))
        if cfg.get('has_position_limits'):
            row['lower']=max(row['lower'],cfg['min_position']);row['upper']=min(row['upper'],cfg['max_position'])
        if name==NAMES[3]:row['lower']=max(row['lower'],-.02)
        rows.append(row)
    return ({key:[row[key] for row in rows[:7]] for key in rows[0]},rows[-1],
            dict(path=str(config),sha256=file_hash(config),source='installed MoveIt limits intersected with live URDF'))


def end_q(plan):
    return list(plan['trajectory']['joint_trajectory']['points'][-1]['positions'])


def pickup_target_xy(center,bias):
    bias=np.asarray(bias,float)
    if bias.shape!=(3,) or not np.isfinite(bias).all() or bias[2]!=0:
        raise ValueError('Pickup correction must be a finite horizontal world translation')
    return np.asarray(center)[:2]+bias[:2]


def check_gripper_sweep(node,q,start,end,label):
    """Pre-motion sampled sweep with the current unrelaxed collision scene."""
    count=max(2,int(math.ceil(abs(end-start)/.01))+1)
    for finger in np.linspace(start,end,count):
        query=GetStateValidity.Request();query.group_name='right_arm'
        query.robot_state=node.state(list(q),float(finger))
        result=node.call(node.valid_client,query)
        if not result.valid:
            raise RuntimeError(label+': gripper sweep collision')
    return count


def plan_transfer(node,q,start,release,attached):
    """Try smaller extra lift when fixed-orientation upward motion has no IK.

    The lowest candidate remains above both lift and release TCP endpoints;
    release_pose already places the object above the bin rim. Each segment
    retains its collision, joint, line and orientation checks.
    """
    failures=[]
    for extra in (.03,.015,0.):
        height=max(start[2,3],release[2,3])+extra
        rise=start.copy();rise[2,3]=height
        oriented=rise.copy();oriented[:3,:3]=release[:3,:3]
        above=release.copy();above[2,3]=height
        candidate=[];current=list(q);current_pose=start
        try:
            steps=[('vertical_clearance',rise,False),('orientation_at_clearance',oriented,True),
                   ('transfer',above,False),('vertical_release',release,False)]
            for name,pose,rotation in steps:
                if np.allclose(pose,current_pose,atol=1e-7,rtol=0.):continue
                current,segment=plan_linear(node,name,current,CLOSE_POSITION,pose,attached,
                                            allow_rotation=rotation,constant_orientation_tolerance_rad=.10)
                candidate.append(segment)
                current_pose=pose
        except RuntimeError as exc:
            failures.append(dict(extra_clearance_m=extra,error=str(exc)))
            print('搬运高度候选未通过：'+json.dumps(failures[-1]),flush=True)
            continue
        for segment in candidate:
            segment['transfer_extra_clearance_m']=extra
        return current,candidate,rise,oriented,above,failures
    raise RuntimeError('No validated transfer height: '+json.dumps(failures))


def plan_pickup(node,scene,offset,current,pickup_bias):
    """Try square-equivalent grasp directions before rejecting a rotated cube."""
    failures=[]
    for height in (.11,.09,.075):
        for turn in (0.,math.pi/2,-math.pi/2,math.pi):
            measured,obj=node.precise_scene(with_top_clearance(scene,.015),offset)
            node.finger_contacts(False)
            node.update_object(padded_cube(obj))
            target=np.eye(4)
            target[:3,:3]=Rotation.from_euler('xyz',[0.,math.pi/2,measured['cube_yaw']+math.pi/2+turn]).as_matrix()
            target[:3,3]=[*pickup_target_xy(measured['cube_center'],pickup_bias),measured['support_z']+height]
            try:
                _,pre=node.plan_pose('pregrasp',target,current,TRANSIT_POSITION)
                check_gripper_sweep(node,end_q(pre),TRANSIT_POSITION,OPEN_POSITION,'pregrasp_open')
                measured,obj=node.precise_scene(scene,offset);node.finger_contacts(False)
                target[2,3]=measured['support_z']+.055
                _,approach=node.plan_pose('vertical_approach',target,end_q(pre),OPEN_POSITION,vertical=True)
            except RuntimeError as exc:
                failures.append(dict(pregrasp_height_m=height,square_turn_rad=turn,error=str(exc)))
                print('抓取姿态候选未通过：'+json.dumps(failures[-1]),flush=True)
                continue
            pre['grasp_candidate']=dict(pregrasp_height_m=height,square_turn_rad=turn,rejected=failures)
            return measured,obj,target,pre,approach
    raise RuntimeError('No validated square grasp candidate: '+json.dumps(failures))


def plan_empty_zero(node,q):
    # HomeRuntimeNode.plan zips the seven arm goals with joint_names: the
    # eighth entry supplies ONLY the fixed finger in start_state, not a finger
    # goal for the seven-axis arm planning group. Do not inherit the live finger.
    payload=node.payload(q)
    payload['joint_names']=[*NAMES,FINGER]
    payload['start_positions_rad']=[*q,TRANSIT_POSITION]
    return node.plan(payload,[0.]*7,20.,.1,.1)


def plan_geometry(node,scene,offset,pickup_bias=(0.,0.,0.)):
    current=list(node.refresh_rigid_state(10.))
    if max(abs(x) for x in current)>.06:raise ValueError('Complete planning requires mechanical zero')
    values=dict(zip(node.full_joints.name,node.full_joints.position))
    require_closed_transit(float(values[FINGER]))
    # Reserve clearance for offline blending on the free-space approach. The
    # contact approach/lift uses the original measured tabletop geometry.
    scene=with_edge_clearance(scene,TABLE_EDGE_CLEARANCE_M)
    buffered=with_top_clearance(scene,.015)
    phases=[[],[],[],[],[]]
    measured,obj,target,pre,approach=plan_pickup(node,scene,offset,current,pickup_bias)
    phases[0].append(pre);phases[1].append(approach)
    attached=node.attached_at(obj,end_q(approach))
    candidate=json.loads((BASE/'runs/linear_release_20261021_01/original_bin_waypoints_02.json').read_text())
    fixed_rotation=np.asarray(next(s for s in candidate['stages'] if s['name']=='vertical_release')['target_tcp_world'])[:3,:3]
    with node.planning_attachment(obj,attached):
        node.finger_contacts(True,support_contact=True)
        target[2,3]=measured['support_z']+.12
        _,lift=node.plan_pose('vertical_lift',target,end_q(approach),CLOSE_POSITION,attached,vertical=True);phases[2].append(lift)
        node.finger_contacts(True)
        q=end_q(lift)
        start=pose_matrix(node.fk_pose({**node.payload(q),'end_effector_link':TCP},q,10.).pose)
        tcp_object=np.linalg.inv(node.tcp_transform(q))@pose_matrix(attached.object.primitive_poses[0])
        release=release_pose(fixed_rotation,tcp_object,scene['measured']['bin'],offset)
        q,transfer,rise,oriented,above,failures=plan_transfer(node,q,start,release,attached)
        transfer[0]['rejected_transfer_heights']=failures
        phases[2].extend(transfer)
        b=scene['measured']['bin']
        placed=box(obj.id,[*(np.asarray(b['center_xy'])+offset[:2]),b['support_z']+offset[2]+.030],[.04]*3,b['yaw'])
        scene_attachment(node,attached,remove=True,world=placed)
        node.finger_contacts(False)
        q,retreat=plan_linear(node,'vertical_retreat',q,OPEN_POSITION,above,None,
                              constant_orientation_tolerance_rad=.10)
        phases[3].append(retreat)
        check_gripper_sweep(node,q,OPEN_POSITION,TRANSIT_POSITION,'transit_close')
        for name,pose,rotation in [('return',oriented,False),('restore_empty_orientation',rise,True)]:
            q,plan=plan_linear(node,name,q,TRANSIT_POSITION,pose,None,allow_rotation=rotation,constant_orientation_tolerance_rad=.10)
            phases[4].append(plan)
        # The legacy API applies its own .10 admission cap. Its timing is
        # discarded; the complete return phase is subsequently retimed once.
        node.precise_scene(buffered,offset)
        node.update_object(placed)
        response=plan_empty_zero(node,q)
        if response.error_code.val!=1:raise RuntimeError('Cannot preplan final mechanical zero')
        phases[4].append(dict(name='mechanical_zero',trajectory=message_to_ordereddict(response.trajectory)))
    node.finger_contacts(False)
    return phases,obj,attached,placed


def validate_timeline(node,plan,scene,offset,model):
    """Check the actual quintic controller path, including the gripper sweeps.

    Every knot and adaptively subdivided interval is tested. This is discrete
    collision validation, not a claim of continuous collision detection.
    """
    require_current_plan(plan)
    cube_margin=require_pickup_clearance(plan)
    limits,grip,source=read_limits(model)
    # A fresh frame may change movable objects, but cannot silently replace
    # the fixed table volume against which the complete path was planned.
    if plan.get('scene',{}).get('table_collision_envelope'):
        scene=admit_scene(plan['scene'],scene)
    # Check the final blended path with the same lateral reserve as planning.
    # Execution monitors the original measured envelope, leaving this reserve
    # available for real tracking error rather than consuming it in smoothing.
    scene=with_edge_clearance(scene,plan.get('table_edge_clearance_m',0.))
    if plan['limits']!=limits or plan['gripper_limits']!=grip:
        raise ValueError('Robot limits changed after trajectory creation')
    rows=plan['samples'];t=np.asarray([r['time_s'] for r in rows])
    q,v,a=[np.asarray([r[key] for r in rows]) for key in ('positions','velocities','accelerations')]
    if len(t)<3 or t[0]!=0 or np.any(np.diff(t)<=0) or not np.isfinite(np.c_[t,q,v,a]).all():
        raise ValueError('Invalid complete timeline')
    if max(abs(q[0,:7]))>.06 or max(abs(q[-1,:7]))>.06:
        raise ValueError('Timeline must start and end at mechanical zero')
    if np.max(abs(v[[0,-1]]))>1e-8 or np.max(abs(a[[0,-1]]))>1e-8:
        raise ValueError('Timeline endpoint must be at rest')
    bounds,lo,hi=controller_bounds(t,q,v,a)
    for i,key in enumerate(('velocity','acceleration','jerk')):
        if np.any(bounds[i]>np.r_[limits[key],grip[key]]*1.000001):
            raise ValueError('Controller interpolant exceeds '+key)
    if np.any(lo<np.r_[limits['lower'],grip['lower']]-1e-7) or np.any(hi>np.r_[limits['upper'],grip['upper']]+1e-7):
        raise ValueError('Controller interpolant exceeds position bounds')
    events={e['name']:e['time_s'] for e in plan['events']}
    if not (events['reach']==0<events['close']<events['carry']<events['open']<events['return']<t[-1]):
        raise ValueError('Invalid grasp/release event order')
    _,obj=node.precise_scene(scene,offset)
    grasp=q[np.argmin(abs(t-events['close'])),:7]
    attached=node.attached_at(obj,grasp.tolist())
    b=scene['measured']['bin']
    placed=box(obj.id,[*(np.asarray(b['center_xy'])+offset[:2]),b['support_z']+offset[2]+.030],[.04]*3,b['yaw'])
    node.finger_contacts(False)
    count=0;phase='reach';support=False;touch=False
    chain=ChainFK(model)
    pickup=chain.pose(dict(zip(NAMES,grasp)))
    last_print=time.monotonic()
    try:
        # Reserve tracking room throughout free-space reach AND stationary
        # opening, not only at OMPL knots. Restore nominal contact geometry
        # before the open-gripper descent. Never allow premature finger contact.
        node.update_object(padded_cube(obj,cube_margin));padded=True
        for i in range(len(t)-1):
            poly=BPoly.from_derivatives(t[i:i+2],[[q[i],v[i],a[i]],[q[i+1],v[i+1],a[i+1]]])
            # <=20ms and <=.01rad/control-point span per collision sample.
            subdivisions=max(2,int(np.ceil((t[i+1]-t[i])/.02)),int(np.ceil(np.max(np.ptp(poly.c,axis=0))/.01)))
            for stamp in np.linspace(t[i],t[i+1],subdivisions+1)[:-1]:
                state=poly(stamp)
                if padded and stamp>=events['approach']:
                    node.update_object(obj);padded=False
                next_phase='return' if stamp>=events['return'] else 'carry' if stamp>=events['carry'] else 'reach'
                if next_phase!=phase:
                    if next_phase=='carry':scene_attachment(node,attached)
                    else:scene_attachment(node,attached,remove=True,world=placed)
                    phase=next_phase
                # Cube/table contact only while departing the pickup support.
                tcp=chain.pose(dict(zip(NAMES,state[:7])))
                supported=bool(phase=='carry' and tcp[2,3]<pickup[2,3]+.025
                               and np.linalg.norm(tcp[:2,3]-pickup[:2,3])<.02)
                allow_touch=phase=='carry' or (phase=='reach' and stamp>=events['close'])
                if supported!=support or allow_touch!=touch:
                    node.finger_contacts(allow_touch,support_contact=supported)
                    support=supported;touch=allow_touch
                query=GetStateValidity.Request();query.group_name='right_arm'
                query.robot_state=node.state(state[:7].tolist(),float(state[7]),attached if phase=='carry' else None)
                checked=node.call(node.valid_client,query)
                if not checked.valid:
                    raise RuntimeError(f'Interpolated path collision at {stamp:.3f}s: '+str([(c.contact_body_1,c.contact_body_2) for c in checked.contacts]))
                count+=1
            if time.monotonic()-last_print>5:
                print(f'完整轨迹碰撞检查 {t[i]:.1f}/{t[-1]:.1f}s，已检查 {count} 个状态',flush=True);last_print=time.monotonic()
        query=GetStateValidity.Request();query.group_name='right_arm';query.robot_state=node.state(q[-1,:7].tolist(),float(q[-1,7]))
        if not node.call(node.valid_client,query).valid:raise RuntimeError('Final zero is in collision')
    finally:
        scene_attachment(node,attached,remove=True,world=obj)
        node.finger_contacts(False)
    return dict(passed=True,checked_states=count+1,controller_interpolation='quintic',
                max_time_step_s=.02,max_control_polygon_step_rad=.01,limits_source=source,
                pickup_clearance_m=cube_margin,pickup_clearance_until='approach')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--robot-model',type=Path,required=True)
    p.add_argument('--registration',type=Path,default=BASE/'config/session_registration_20261021.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(remove_ros_args()[1:]);args.output.mkdir(parents=True,exist_ok=False)
    validated_token(False);rclpy.init();node=GraspNode()
    report=dict(success=False,physical_motion=False,started_wall_ns=time.time_ns())
    try:
        scene=planning_scene(stabilize_scene(json.loads(args.scene.read_text())));offset=np.asarray(json.loads(args.registration.read_text())['translation_world_m'])
        correction=json.loads((BASE/'config/grasp_target_offset.json').read_text())
        if correction['frame']!='world':raise ValueError('Pickup correction must use world frame')
        pickup_bias=correction['translation_m']
        phases,_,_,_=plan_geometry(node,scene,offset,pickup_bias)
        limits,grip,source=read_limits(args.robot_model)
        rejected=[]
        # Temporal blending can cut a corner of a valid geometric path. Reduce
        # blending before motion, re-auditing both dynamics and collisions each
        # time. Never bypass a collision or change a running trajectory.
        for sigma in (.12,.06,.03):
            write_json(args.output/'geometry.json',dict(phase_plans=phases,limits=limits,gripper_limits=grip,
                gripper_policy=COLLECTION_POLICY,smoothing_sigma_s=sigma))
            subprocess.run([str(BASE.parent/'.venv-trajectory-processing/bin/python'),'-m','vision_grasp_baseline.continuous_timing',
                            str(args.output/'geometry.json'),str(args.output/'timing.json')],cwd=BASE.parent,check=True)
            plan=json.loads((args.output/'timing.json').read_text())
            plan.update(schema=3,scene=scene,local_geometry_translation_m=offset.tolist(),limits=limits,gripper_limits=grip,
                        model_sha256=file_hash(args.robot_model),limits_source=source,created_wall_ns=time.time_ns(),
                        grasp_target_offset_world_m=pickup_bias)
            plan['table_edge_clearance_m']=TABLE_EDGE_CLEARANCE_M
            plan['pickup_clearance_m']=PICKUP_CLEARANCE_M
            try:
                plan['validation']=validate_timeline(node,plan,scene,offset,args.robot_model)
                break
            except RuntimeError as exc:
                if not str(exc).startswith('Interpolated path collision'):raise
                rejected.append(dict(smoothing_sigma_s=sigma,error=str(exc)))
                write_json(args.output/'rejected_smoothing.json',rejected)
                if sigma==.03:raise
                print('平滑轨迹碰撞检查未通过，减小过渡平滑范围后重新检查：'+str(exc),flush=True)
        plan['rejected_smoothing_attempts']=rejected
        write_json(args.output/'trajectory.json',plan)
        report.update(success=True,duration_s=plan['duration_s'],events=plan['events'],validation=plan['validation'],
                      trajectory=str((args.output/'trajectory.json').resolve()))
        print(f"整轮轨迹已保存：预计执行 {plan['duration_s']:.2f}s；执行中不再调用规划器。",flush=True)
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        report['finished_wall_ns']=time.time_ns();write_json(args.output/'result.json',report)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    try:main()
    except UnstableSceneError as exc:
        print('Input scene is unstable; recapture images before planning: '+str(exc),flush=True)
        raise SystemExit(3)
