"""Mock-only complete plan with explicit intermediate points at the original bin.

Search a small orientation adjustment, rotate only over the pickup position,
translate through four intermediate goals, then lower and retrace. No actions.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from moveit_msgs.msg import AttachedCollisionObject
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from rosidl_runtime_py.set_message import set_message_fields
from scipy.spatial.transform import Rotation

from .grasp import GraspNode
from .hover import EE,TCP,NAMES,box,matrix_pose,pose_matrix
from .linear_place_plan import plan_linear,release_pose
from scripts.control_ownership.h2 import validated_token


def intermediate_poses(start_tcp,release,segments=4):
    if not isinstance(segments,int) or not 2<=segments<=12:
        raise ValueError('Use 2 to 12 translation segments')
    height=max(start_tcp[2,3],release[2,3])+.03
    rise=start_tcp.copy();rise[2,3]=height
    oriented=rise.copy();oriented[:3,:3]=release[:3,:3]
    steps=[('vertical_clearance',rise,False),('orientation_at_clearance',oriented,True)]
    for i in range(1,segments+1):
        target=oriented.copy();target[:2,3]=oriented[:2,3]+(release[:2,3]-oriented[:2,3])*i/segments
        steps.append((f'transfer_{i}_of_{segments}',target,False))
    above=steps[-1][1].copy()
    steps.extend([('vertical_release',release,False),('vertical_retreat',above,False)])
    for i in range(1,segments+1):
        target=above.copy();target[:2,3]=above[:2,3]+(oriented[:2,3]-above[:2,3])*i/segments
        steps.append((f'return_{i}_of_{segments}',target,False))
    steps.append(('restore_empty_orientation',rise,True))
    return steps


def inverse_solution(node,target,ee_tcp,start,gripper,attached):
    seeds=[start,[0.,.5,0.,1.2,0.,-.4,0.],[.8,.5,-.8,1.2,-.6,.5,1.4]]
    for seed in seeds:
        req=GetPositionIK.Request();ik=req.ik_request
        ik.group_name='right_arm';ik.ik_link_name=EE;ik.avoid_collisions=False
        ik.robot_state=node.state(seed,gripper,attached);ik.timeout.nanosec=50000000
        ik.pose_stamped.header.frame_id='world';ik.pose_stamped.pose=matrix_pose(target@np.linalg.inv(ee_tcp))
        response=node.call(node.ik_client,req)
        if response.error_code.val!=1:continue
        positions=dict(zip(response.solution.joint_state.name,response.solution.joint_state.position))
        q=[positions[name] for name in NAMES]
        v=GetStateValidity.Request();v.group_name='right_arm';v.robot_state=node.state(q,gripper,attached)
        if node.call(node.valid_client,v).valid:return q
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grasp-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--segments',type=int,default=4)
    a=p.parse_args(remove_ros_args()[1:])
    if not 150<=int(os.environ.get('ROS_DOMAIN_ID','-1'))<=232:
        raise RuntimeError('This planner refuses physical domains')
    if a.output.exists():raise ValueError('Use a new output file')
    previous=json.loads(a.grasp_result.read_text());scene=json.loads(Path(previous['scene']).read_text())
    correction=np.asarray(previous['local_geometry_translation_m'])
    validated_token(False);rclpy.init();node=GraspNode()
    report=dict(no_motion=True,mock_only=True,success=False,bin_moved=False,candidates=[],stages=[],
                limitation='Historical scene/model only; actual table collision has not been reproduced or cleared')
    try:
        _,obj=node.precise_scene(scene,correction)
        original=previous['stages'][-1]['actual_joint_positions']
        tcp=pose_matrix(node.fk_pose({**node.payload(original),'end_effector_link':TCP},original,10.).pose)
        ee_tcp=node.tcp_transform(original)
        attached=AttachedCollisionObject();record=deepcopy(previous['attached_object']);record['object'].pop('operation',None)
        set_message_fields(attached,record)
        tcp_object=np.linalg.inv(ee_tcp)@pose_matrix(attached.object.primitive_poses[0])
        node.update_object(obj,remove=True);node.finger_contacts(True)
        cube_world_present=False
        success=False
        for degrees in [0,5,10,15,20,25,30,35,40,45,50]:
            for heading in ([0] if degrees==0 else range(0,360,45)):
                # Restart every attempt with the same virtual held pose and no world copy.
                if cube_world_present:
                    node.update_object(obj,remove=True);cube_world_present=False
                axis=np.array([math.cos(math.radians(heading)),math.sin(math.radians(heading)),0.])
                rotation=Rotation.from_rotvec(axis*math.radians(degrees)).as_matrix()@tcp[:3,:3]
                release=release_pose(rotation,tcp_object,scene['measured']['bin'],correction)
                waypoints=intermediate_poses(tcp,release,a.segments)
                high=next(t for name,t,_ in waypoints if name=='vertical_retreat')
                candidate=dict(tilt_adjustment_deg=degrees,tilt_axis_heading_deg=heading,
                               release_ik=False,transfer_ik=False,success=False)
                report['candidates'].append(candidate)
                release_q=inverse_solution(node,release,ee_tcp,original,-.255,attached)
                if release_q is None:continue
                candidate['release_ik']=True
                if inverse_solution(node,high,ee_tcp,release_q,-.255,attached) is None:continue
                candidate['transfer_ik']=True
                print(json.dumps(dict(candidate=candidate)),flush=True)
                planned_stages=[];q=list(original)
                try:
                    for name,target,rotating in waypoints:
                        held=attached if name in ('vertical_clearance','orientation_at_clearance','vertical_release') or name.startswith('transfer_') else None
                        if name=='vertical_retreat':
                            b=scene['measured']['bin'];center=[*(np.asarray(b['center_xy'])+correction[:2]),b['support_z']+correction[2]+.030]
                            node.update_object(box(obj.id,center,[.04]*3,b['yaw']))
                            cube_world_present=True
                        q,stage=plan_linear(node,name,q,-.255 if held else -.35,target,held,allow_rotation=rotating)
                        planned_stages.append(stage)
                        print(json.dumps(dict(stage=name,points=stage['audit']['point_count'],checked_samples=stage['checked_samples'])),flush=True)
                    candidate['success']=True;report['stages']=planned_stages
                    report['waypoints']=[dict(name=name,target_tcp_world=t.tolist(),orientation_transition=rot) for name,t,rot in waypoints]
                    report['selected_candidate']={k:v for k,v in candidate.items()}
                    success=True;break
                except RuntimeError as exc:
                    candidate['planning_error']=str(exc);candidate['completed_segments']=len(planned_stages)
                    print(json.dumps(dict(candidate_failed=candidate)),flush=True)
            if success:break
        if not success:raise RuntimeError('No complete intermediate-waypoint plan found in this bounded search')
        report['success']=True
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        report['finished_wall_ns']=time.time_ns();a.output.write_text(json.dumps(report,indent=2)+'\n')
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
