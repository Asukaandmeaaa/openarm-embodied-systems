"""Mock-only audit and Pilz LIN plan for lift, horizontal transfer, lower and return.

No action clients are called. Refuses the physical DDS domain. Uses a recorded
held grasp and measured scene; this is a candidate, not a physical approval.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from moveit_msgs.msg import AttachedCollisionObject, Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetMotionPlan, GetStateValidity
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from shape_msgs.msg import SolidPrimitive
from scipy.spatial.transform import Rotation

from .grasp import GraspNode
from .hover import EE, TCP, NAMES, box, matrix_pose, pose_matrix
from .execution_quality import task_joint4_bounds
from scripts.control_ownership.h2 import validated_token
from openarm_kinesthetic_recorder.execution_guard import audit_joint_trajectory


def release_pose(rotation,tcp_object,bin_,correction):
    """Place the lowest cube corner 20mm over the rim in the selected orientation."""
    object_r=rotation @ tcp_object[:3,:3]
    half=.020*np.sum(abs(object_r),axis=1)
    rb=Rotation.from_euler('z',bin_['yaw']).as_matrix()
    footprint=.020*np.sum(abs(rb.T @ object_r),axis=1)
    inner=np.asarray(bin_['outer_size_xy'])-.020
    if np.any(footprint[:2]+.010>inner/2):
        raise ValueError('Held cube does not fit measured nominal bin aperture')
    rim=bin_['support_z']+correction[2]+bin_['rim_height']
    object_center=np.r_[np.asarray(bin_['center_xy'])+correction[:2],rim+half[2]+.020]
    release=np.eye(4);release[:3,:3]=rotation;release[:3,3]=object_center-rotation@tcp_object[:3,3]
    return release


def release_waypoints(start_tcp, tcp_object, bin_, correction, clearance=.03):
    """Keep the measured grasp orientation throughout transfer and release."""
    release=release_pose(start_tcp[:3,:3],tcp_object,bin_,correction)
    z=max(start_tcp[2,3],release[2,3])+clearance
    lift=start_tcp.copy();lift[2,3]=z
    transfer=release.copy();transfer[2,3]=z
    return [('vertical_clearance',lift),('horizontal_transfer',transfer),
            ('vertical_release',release),('vertical_retreat',transfer),('horizontal_return',lift)]


def request_linear_trajectory(node,request,label):
    """Retry Cartesian generation at lower speed before whole-cycle retiming.

    Pilz can reject the Cartesian-to-joint conversion on acceleration even
    though the geometric path is usable. Keep the pose, seed, constraints and
    joint limits unchanged; every successful path still receives our audits.
    """
    motion=request.motion_plan_request
    velocity=motion.max_velocity_scaling_factor
    acceleration=motion.max_acceleration_scaling_factor
    attempts=[]
    for factor in (1.,.5,.25,.125):
        motion.max_velocity_scaling_factor=velocity*factor
        motion.max_acceleration_scaling_factor=acceleration*factor
        result=node.call(node.plan_client,request,30.).motion_plan_response
        attempts.append(dict(velocity_scaling=motion.max_velocity_scaling_factor,
                             acceleration_scaling=motion.max_acceleration_scaling_factor,
                             error_code=result.error_code.val,message=result.error_code.message))
        if result.error_code.val==1:return result,attempts
        # Invalid requests/start/goal conditions do not benefit from retiming.
        if result.error_code.val not in (-1,99999):break
        print(json.dumps(dict(stage=label,lin_generation_attempt=attempts[-1],
                              retry_lower_generation_speed=factor!=.125)),flush=True)
    raise RuntimeError(label+': LIN generation failed after '+str(len(attempts))+
                       ' attempt(s): '+json.dumps(attempts)+
                       '; inspect MoveIt log for joint conversion/IK details')


def plan_linear(node,label,start,gripper,target,attached,*,allow_rotation=False,
                constant_orientation_tolerance_rad=.01):
    ee_tcp=node.tcp_transform(start);ee_target=target@np.linalg.inv(ee_tcp)
    # Rotation about the grasp point must be planned at that point. A LIN at
    # the wrist frame makes the offset TCP sweep an arc during reorientation.
    controlled_link=TCP if allow_rotation else EE
    controlled_target=target if allow_rotation else ee_target
    request=GetMotionPlan.Request();m=request.motion_plan_request
    m.group_name='right_arm';m.pipeline_id='pilz_industrial_motion_planner';m.planner_id='LIN'
    m.allowed_planning_time=8.;m.num_planning_attempts=1
    m.max_velocity_scaling_factor=.03;m.max_acceleration_scaling_factor=.01
    m.start_state=node.state(start,gripper,attached);m.start_state.joint_state.velocity=[0.]*8
    constraints=Constraints();constraints.name=label
    pc=PositionConstraint();pc.header.frame_id='world';pc.link_name=controlled_link;pc.weight=1.
    shape=SolidPrimitive();shape.type=SolidPrimitive.SPHERE;shape.dimensions=[.0001]
    pc.constraint_region.primitives=[shape];pc.constraint_region.primitive_poses=[matrix_pose(controlled_target)]
    oc=OrientationConstraint();oc.header.frame_id='world';oc.link_name=controlled_link;oc.weight=1.
    oc.orientation=matrix_pose(controlled_target).orientation
    oc.absolute_x_axis_tolerance=oc.absolute_y_axis_tolerance=oc.absolute_z_axis_tolerance=.001
    constraints.position_constraints=[pc];constraints.orientation_constraints=[oc]
    m.goal_constraints=[constraints]
    result,attempts=request_linear_trajectory(node,request,label)
    jt=result.trajectory.joint_trajectory
    task_joint4_bounds(jt.joint_names,jt.points)
    for point in [jt.points[0],jt.points[-1]]:point.velocities=[0.]*7;point.accelerations=[0.]*7
    goal=list(jt.points[-1].positions)
    audit=audit_joint_trajectory(jt,NAMES,start,goal,.015).as_dict()
    current=pose_matrix(node.fk_pose({**node.payload(start),'end_effector_link':TCP},start,10.).pose)
    delta=target[:3,3]-current[:3,3];length=float(np.linalg.norm(delta));direction=delta/max(length,1e-12)
    samples=[]
    for a,b in zip(jt.points,jt.points[1:]):
        samples.extend([list(a.positions),((np.asarray(a.positions)+b.positions)/2).tolist()])
    samples.append(goal)
    max_line=max_rotation=max_rotation_path_error=0.
    target_rotation=Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).as_rotvec()
    rotation_length=float(np.linalg.norm(target_rotation))
    for q in samples:
        validity=GetStateValidity.Request();validity.group_name='right_arm'
        validity.robot_state=node.state(q,gripper,attached)
        v=node.call(node.valid_client,validity)
        if not v.valid:raise RuntimeError(label+': path collision '+str([(c.contact_body_1,c.contact_body_2) for c in v.contacts]))
        fk=pose_matrix(node.fk_pose({**node.payload(q),'end_effector_link':TCP},q,10.).pose)
        displacement=fk[:3,3]-current[:3,3]
        along=float(np.clip(displacement@direction,0.,length))
        max_line=max(max_line,float(np.linalg.norm(displacement-along*direction)))
        relative=Rotation.from_matrix(fk[:3,:3]@current[:3,:3].T).as_rotvec()
        max_rotation=max(max_rotation,float(np.linalg.norm(relative)))
        fraction=float(np.clip(relative@target_rotation/max(rotation_length**2,1e-16),0.,1.))
        expected_rotation=Rotation.from_rotvec(target_rotation*fraction).as_matrix()@current[:3,:3]
        max_rotation_path_error=max(max_rotation_path_error,float(Rotation.from_matrix(fk[:3,:3]@expected_rotation.T).magnitude()))
    if max_line>.003 or max_rotation_path_error>.01 or (not allow_rotation and max_rotation>constant_orientation_tolerance_rad):
        raise RuntimeError(label+f': generated path did not retain line/orientation: line={max_line:.6f}m, '
                           f'rotation_path={max_rotation_path_error:.6f}rad, rotation_change={max_rotation:.6f}rad')
    return goal,dict(name=label,planner='Pilz LIN',generation_attempts=attempts,audit=audit,checked_samples=len(samples),
                     max_line_error_m=max_line,max_orientation_change_rad=max_rotation,
                     max_rotation_path_error_rad=max_rotation_path_error,orientation_transition=allow_rotation,
                     target_tcp_world=target.tolist(),trajectory=message_to_ordereddict(result.trajectory))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grasp-result',type=Path,required=True)
    p.add_argument('--previous-place',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bin-offset-xy',nargs=2,type=float,default=[0.,0.],
                   help='Mock-only prospective bin translation; never changes physical configuration')
    a=p.parse_args(remove_ros_args()[1:])
    if not 150<=int(os.environ.get('ROS_DOMAIN_ID','-1'))<=232:
        raise RuntimeError('This diagnostic refuses physical ROS domains')
    if a.output.exists():raise ValueError('Use a new output file')
    previous=json.loads(a.grasp_result.read_text());scene=json.loads(Path(previous['scene']).read_text())
    scene['measured']['bin']['center_xy']=(np.asarray(scene['measured']['bin']['center_xy'])+a.bin_offset_xy).tolist()
    correction=np.asarray(previous['local_geometry_translation_m'])
    validated_token(False);rclpy.init();node=GraspNode()
    report=dict(no_motion=True,mock_only=True,success=False,stages=[],previous_actual_endpoints=[],
                prospective_bin_offset_world_m=a.bin_offset_xy)
    try:
        geometry,obj=node.precise_scene(scene,correction)
        old=json.loads(a.previous_place.read_text())
        for stage in old['stages']:
            query=GetStateValidity.Request();query.group_name='right_arm'
            query.robot_state=node.state(stage['actual_joint_positions'],-.35)
            v=node.call(node.valid_client,query)
            report['previous_actual_endpoints'].append(dict(name=stage['name'],valid=v.valid,
                contacts=[(c.contact_body_1,c.contact_body_2) for c in v.contacts]))
        start=previous['stages'][-1]['actual_joint_positions']
        start_tcp=pose_matrix(node.fk_pose({**node.payload(start),'end_effector_link':TCP},start,10.).pose)
        attached=AttachedCollisionObject();record=deepcopy(previous['attached_object'])
        record['object'].pop('operation',None) # serialized uint8 can be a character
        set_message_fields(attached,record)
        ee_tcp=node.tcp_transform(start)
        tcp_object=np.linalg.inv(ee_tcp)@pose_matrix(attached.object.primitive_poses[0])
        report['start_tcp_world']=start_tcp.tolist()
        node.update_object(obj,remove=True);node.finger_contacts(True)
        waypoints=release_waypoints(start_tcp,tcp_object,scene['measured']['bin'],correction)
        report['waypoints']=[dict(name=name,tcp_world=t.tolist()) for name,t in waypoints]
        for name,target in waypoints:
            if name=='vertical_retreat':
                b=scene['measured']['bin']
                # Empty-arm checks must still include the released cube. The
                # nominal 10mm floor is the same assumption as the old test.
                center=[*(np.asarray(b['center_xy'])+correction[:2]),
                        b['support_z']+correction[2]+.010+.020]
                node.update_object(box('vision_baseline_cube',center,[.04]*3,b['yaw']))
                report['released_cube_proxy_center_world']=center
            held=attached if name in ('vertical_clearance','horizontal_transfer','vertical_release') else None
            start,planned=plan_linear(node,name,start,-.255 if held else -.35,target,held)
            report['stages'].append(planned)
            print(json.dumps({k:v for k,v in planned.items() if k not in ('trajectory','target_tcp_world')}),flush=True)
        report['success']=True
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        report['finished_wall_ns']=time.time_ns();a.output.write_text(json.dumps(report,indent=2)+'\n')
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
