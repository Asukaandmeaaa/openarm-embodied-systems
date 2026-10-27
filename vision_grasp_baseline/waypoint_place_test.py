"""Attended execution of fixed-orientation intermediate-point placement.

Plans every segment before moving, then replans each segment from fresh feedback.
Uses existing connection-bound H2 actions; no direct motor commands.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from moveit_msgs.msg import CollisionObject,PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene
from scipy.spatial.transform import Rotation

from .grasp_test import TestNode,capture
from .collection_io import event
from .grasp import FINGER
from .hover import NAMES,TCP,pose_matrix,box
from .fast_fk import ChainFK
from .execution_quality import endpoint_quality
from .linear_place_plan import release_pose,plan_linear
from .waypoint_place_plan import intermediate_poses
from .place_test import verify_placed
from scripts.act_trial.isolated_telemetry import Telemetry
from scripts.control_ownership.h2 import validated_token,inject_action_capabilities


class WaypointNode(TestNode):
    def __init__(self,model,*,tracking_abort=True):
        super().__init__();self.chain=ChainFK(model);self.tracking_error_since=None
        self.max_tracking_error_m=0.
        self.tracking_abort=tracking_abort

    def execution_table_guard_failure(self,now=None):
        failure=super().execution_table_guard_failure(now)
        if failure or self.telemetry is None:return failure
        try:
            s=self.telemetry.snapshot()
            actual=self.chain.pose(dict(zip(NAMES,s['positions_rad'][:7])))
            command=self.chain.pose(dict(zip(NAMES,s['command_positions_rad'][:7])))
        except (RuntimeError,ValueError,KeyError) as exc:
            return 'Live TCP tracking monitor failed: '+str(exc)
        error=float(np.linalg.norm(actual[:3,3]-command[:3,3]));self.max_tracking_error_m=max(self.max_tracking_error_m,error)
        if self.tracking_abort and error>.025:
            if self.tracking_error_since is None:self.tracking_error_since=time.monotonic()
            if time.monotonic()-self.tracking_error_since>.20:
                return f'Actual TCP differs from live commanded TCP by {error:.3f}m'
        else:self.tracking_error_since=None
        return ''


def scene_attachment(node,attached,*,remove=False,world=None):
    # MoveIt attaches before processing world updates, and attaching already
    # consumes the same-ID world object. Do not REMOVE it again in this diff.
    if not remove and world is not None and world.id==attached.object.id:
        raise ValueError('Attach consumes the same-ID world object; omit the world update')
    request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
    body=deepcopy(attached);body.object.operation=CollisionObject.REMOVE if remove else CollisionObject.ADD
    request.scene.robot_state.attached_collision_objects=[body]
    if world is not None:request.scene.world.collision_objects=[world]
    if not node.call(node.apply_client,request).success:raise RuntimeError('Attachment transition failed')


def is_held_stage(name):
    return name in ('vertical_clearance','orientation_at_clearance','vertical_release') or name.startswith('transfer_')


def prepare_trajectory(planned,output):
    """Save a JSON plan and build a goal without mutating its nested lists.

    Jazzy's set_message_fields replaces entries of input message sequences in
    place, so it must receive a private copy of the trajectory dictionary.
    """
    from moveit_msgs.msg import RobotTrajectory
    from rosidl_runtime_py.set_message import set_message_fields
    serialized=json.dumps(planned,indent=2,allow_nan=False)+'\n'
    trajectory=RobotTrajectory()
    set_message_fields(trajectory,deepcopy(planned['trajectory']))
    output.write_text(serialized)
    return trajectory


def preflight_waypoints(node,attached,cube,current,steps,output):
    plans=[];q=list(current);virtual_release=False
    report=dict(success=False,stages=plans)
    try:
        for name,target,rotating in steps:
            if name=='vertical_retreat':
                virtual_release=True
                scene_attachment(node,attached,remove=True,world=cube)
            body=attached if is_held_stage(name) else None
            q,planned=plan_linear(node,name,q,-.255 if body else -.35,target,body,allow_rotation=rotating,
                                  constant_orientation_tolerance_rad=.10)
            plans.append(planned)
        report['success']=True
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:
        try:
            if virtual_release:
                # Supply the original attachment geometry. MoveIt removes the
                # virtual world copy as part of ADD, without another REMOVE.
                scene_attachment(node,attached)
        except BaseException as exc:
            report.update(success=False,restoration_error=str(exc))
            raise
        finally:
            output.write_text(json.dumps(report,indent=2)+'\n')
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grasp-result',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--robot-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true',help='Check all segments and restore the held scene; no action goals')
    a=p.parse_args(remove_ros_args()[1:]);previous=json.loads(a.grasp_result.read_text())
    proof=json.loads(a.candidate.read_text())
    if not previous['success'] or previous['state']!='HOLDING_AFTER_LIFT' or not proof['success']:
        raise ValueError('Verified grasp and successful waypoint candidate required')
    a.output.mkdir(parents=True,exist_ok=False)
    scene=json.loads(Path(previous['scene']).read_text());offset=np.array(previous['local_geometry_translation_m'])
    token=validated_token(require_token=not a.plan_only);rclpy.init();node=WaypointNode(a.robot_model);telemetry=Telemetry(a.output/'raw_can.jsonl')
    report=dict(workflow='ATTENDED_INTERMEDIATE_WAYPOINT_RELEASE',success=False,stages=[],
                home_commanded=False,bin_moved=False,started_wall_ns=time.time_ns(),state='HOLDING_AFTER_LIFT',
                plan_only=a.plan_only,physical_motion=False)
    try:
        node.bind_telemetry(telemetry)
        query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        held=node.call(node.scene_client,query).scene.robot_state.attached_collision_objects
        attached=next(x for x in held if x.object.id=='vision_baseline_cube')
        current=node.refresh_rigid_state(10.);start=pose_matrix(node.fk_pose({**node.payload(current),'end_effector_link':TCP},current,10.).pose)
        direct=node.chain.pose(dict(zip(NAMES,current)))
        if np.max(abs(direct-start))>1e-7:raise RuntimeError('Tracking FK disagrees with MoveIt FK')
        orientation=attached.object.pose.orientation
        root=np.eye(4) if sum(v*v for v in (orientation.x,orientation.y,orientation.z,orientation.w))<.5 else pose_matrix(attached.object.pose)
        tcp_object=np.linalg.inv(node.tcp_transform(current))@root@pose_matrix(attached.object.primitive_poses[0])
        release_record=next(s for s in proof['stages'] if s['name']=='vertical_release')
        fixed_rotation=np.array(release_record['target_tcp_world'])[:3,:3]
        release=release_pose(fixed_rotation,tcp_object,scene['measured']['bin'],offset)
        steps=intermediate_poses(start,release)
        b=scene['measured']['bin'];cube_center=[*(np.array(b['center_xy'])+offset[:2]),b['support_z']+offset[2]+.030]
        cube=box(attached.object.id,cube_center,[.04]*3,b['yaw'])
        # Virtual release for full preflight; restore the actual held body even on failure.
        preflight=preflight_waypoints(node,attached,cube,current,steps,a.output/'preflight.json')
        report['preflight_segments']=len(preflight)
        # Check the same JSON-to-goal conversion used during execution before
        # moving, including when called through the no-motion planning entry.
        for planned in preflight:
            prepare_trajectory(planned,a.output/f"preflight_{planned['name']}_plan.json")
        report['preflight_goals_prepared']=len(preflight)
        if a.plan_only:
            report['success']=True
            print(json.dumps(dict(success=True,physical_motion=False,preflight_segments=len(preflight))))
            return
        with inject_action_capabilities(token):
            for name,target,rotating in steps:
                failure=node.execution_table_guard_failure()
                if failure:raise RuntimeError(failure)
                current=node.refresh_rigid_state(10.);values=dict(zip(node.full_joints.name,node.full_joints.position))
                body=attached if is_held_stage(name) else None
                event('planning_begin',segment=name)
                _,planned=plan_linear(node,name,current,float(values[FINGER]),target,body,allow_rotation=rotating,
                                      constant_orientation_tolerance_rad=.10)
                trajectory=prepare_trajectory(planned,a.output/f'{name}_plan.json')
                print(json.dumps(dict(stage=name,executing=True,duration_s=planned['audit']['duration_s'])),flush=True)
                report['physical_motion']=True
                event('motion_begin',segment=name)
                result=node.execute_same_trajectory(trajectory,planned['audit']['duration_s'],20.)
                event('motion_end',segment=name)
                actual_q=node.refresh_rigid_state(10.)
                actual=pose_matrix(node.fk_pose({**node.payload(actual_q),'end_effector_link':TCP},actual_q,10.).pose)
                arrival=endpoint_quality(target,actual)
                stage=dict(name=name,execution=result,actual_q=actual_q,actual_tcp_world=actual.tolist(),arrival=arrival)
                report['stages'].append(stage)
                (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
                if result['action_status']!=4 or result['moveit_error_code']!=1 or not arrival['accepted']:
                    raise RuntimeError(name+': execution/actual arrival failed')
                if name=='vertical_release':
                    report['gripper_release']=node.gripper(-.35)
                    scene_attachment(node,attached,remove=True,world=cube)
                    node.finger_contacts(False);report['state']='RELEASED'
        capture(a.output/'after_release')
        report['placement_verification']=verify_placed(scene,a.output/'after_release',offset)
        report['success']=report['placement_verification']['verified']
        report['state']='PLACED_AND_RETURNED' if report['success'] else 'PLACEMENT_NOT_VERIFIED'
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        report['max_live_tcp_tracking_error_m']=node.max_tracking_error_m
        report['finished_wall_ns']=time.time_ns();(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        telemetry.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
