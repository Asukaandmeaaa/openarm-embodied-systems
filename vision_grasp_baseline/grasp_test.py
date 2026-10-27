"""Attended visual grasp-and-lift test; no HOME and no bulk data collection."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import ApplyPlanningScene
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args
from rosidl_runtime_py.convert import message_to_ordereddict
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectoryPoint

from .grasp import GraspNode,FINGER
from .hover import TCP,pose_matrix
from .execution_quality import endpoint_quality
from .collection_io import event
from scripts.act_trial.isolated_telemetry import Telemetry
from scripts.control_ownership.h2 import inject_action_capabilities,validated_token

ROOT=Path(__file__).resolve().parent


class TestNode(GraspNode):
    def __init__(self):
        super().__init__();self.telemetry=None
        self.gripper_client=ActionClient(self,FollowJointTrajectory,'/authorized/right_gripper_controller/follow_joint_trajectory')

    def execution_table_guard_failure(self,now=None):
        if (ROOT/'STOP').exists():return 'operator requested visual baseline stop'
        if self.telemetry is not None:
            try:self.telemetry.snapshot()
            except RuntimeError as exc:return 'Motor feedback check failed: '+str(exc)
        return ''

    def bind_telemetry(self,observer):
        deadline=time.monotonic()+3.
        while True:
            try:observer.snapshot();break
            except RuntimeError:
                if time.monotonic()>=deadline:raise
                rclpy.spin_once(self,timeout_sec=.05)
        self.telemetry=observer

    def gripper(self,target):
        event('gripper_begin',target=target)
        failure=self.execution_table_guard_failure()
        if failure:raise RuntimeError(failure)
        self.refresh_rigid_state(10.)
        if self.full_joints is None:raise RuntimeError('Full joint state unavailable')
        values=dict(zip(self.full_joints.name,self.full_joints.position))
        start=float(values[FINGER])
        if not self.gripper_client.wait_for_server(timeout_sec=10.):raise RuntimeError('Authorized gripper action unavailable')
        goal=FollowJointTrajectory.Goal();goal.trajectory.joint_names=[FINGER]
        first=JointTrajectoryPoint();first.positions=[start];first.velocities=[0.];first.accelerations=[0.]
        last=JointTrajectoryPoint();last.positions=[float(target)];last.velocities=[0.];last.accelerations=[0.];last.time_from_start.sec=3
        goal.trajectory.points=[first,last]
        f=self.gripper_client.send_goal_async(goal);rclpy.spin_until_future_complete(self,f,timeout_sec=10.)
        handle=f.result()
        if handle is None or not handle.accepted:raise RuntimeError('Gripper goal rejected')
        result=handle.get_result_async();deadline=time.monotonic()+12.
        while not result.done() and time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.05)
            failure=self.execution_table_guard_failure()
            if failure:
                cancel=handle.cancel_goal_async();rclpy.spin_until_future_complete(self,cancel,timeout_sec=3.)
                raise RuntimeError(failure)
        if not result.done():
            cancel=handle.cancel_goal_async();rclpy.spin_until_future_complete(self,cancel,timeout_sec=3.)
            raise RuntimeError('Gripper action timed out')
        wrapper=result.result()
        if wrapper.status!=4 or wrapper.result.error_code!=0:
            raise RuntimeError('Gripper action failed: '+wrapper.result.error_string)
        event('gripper_end',target=target)
        return dict(start=start,target=target,status=wrapper.status,error_code=wrapper.result.error_code)

    def attach(self,attached):
        request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
        request.scene.robot_state.attached_collision_objects=[attached]
        if not self.call(self.apply_client,request).success:raise RuntimeError('Could not register grasped object')

    def move(self,name,target,report,output,*,vertical=False,attached=None):
        failure=self.execution_table_guard_failure()
        if failure:raise RuntimeError(failure)
        current=self.refresh_rigid_state(10.)
        values=dict(zip(self.full_joints.name,self.full_joints.position))
        event('planning_begin',segment=name)
        trajectory,planned=self.plan_pose(name,target,current,float(values[FINGER]),attached,vertical)
        path=output/f'{name}_plan.json';path.write_text(json.dumps(planned,indent=2)+'\n')
        self.verify_controller_active('right_joint_trajectory_controller',10.)
        event('motion_begin',segment=name)
        result=self.execute_same_trajectory(trajectory,planned['audit']['duration_s'],20.)
        event('motion_end',segment=name)
        stage=dict(name=name,plan=str(path),execution=result)
        report['stages'].append(stage)
        if result['action_status']!=4 or result['moveit_error_code']!=1:
            raise RuntimeError(name+': execution did not succeed')
        final=self.refresh_rigid_state(10.)
        fk=self.fk_pose({**self.payload(final),'end_effector_link':TCP},final,10.).pose
        stage.update(actual_joint_positions=final,actual_tcp_world=pose_matrix(fk).tolist())
        stage['arrival']=endpoint_quality(target,stage['actual_tcp_world'])
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        if not stage['arrival']['accepted']:
            raise RuntimeError(name+': actual TCP did not reach planned endpoint: '+str(stage['arrival']))
        return final


def capture(directory):
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'capture.py'),'--output',str(directory),'--samples','10'],check=True)


def verify_lift(scene,capture_path):
    import cv2
    metadata=json.loads((capture_path/'capture.json').read_text());i=metadata['color_intrinsics']
    wc=np.array(scene['world_from_color_optical']);normal=np.array(scene['measured']['table']['normal'])
    d=scene['measured']['table']['plane_d'];heights=[]
    for frame in metadata['frames']:
        index=frame['index'];image=cv2.imread(str(capture_path/f'color_{index:03d}.png'))
        z=np.load(capture_path/f'depth_{index:03d}.npy')*metadata['depth_scale_m'];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(90,150,65),(130,255,255))>0
        v,u=np.where(mask&(z>.1)&(z<1.5))
        points=np.column_stack(((u-i['ppx'])*z[v,u]/i['fx'],(v-i['ppy'])*z[v,u]/i['fy'],z[v,u]))
        world=points @ wc[:3,:3].T+wc[:3,3]
        near=np.linalg.norm(world[:,:2]-np.array(scene['measured']['cube']['center'][:2]),axis=1)<.08
        if near.sum()<50:continue
        heights.append(float(np.percentile(world[near] @ normal+d,85)))
    before=scene['measured']['cube']['size'][2]
    positive=[h>before+.025 for h in heights]
    return dict(before_top_height_m=before,after_top_heights_m=heights,
                verified=len(positive)>=6 and sum(positive)>=6,
                interpretation='Blue object upper surface rose at least 25mm in at least 6 RGB-D frames')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--registration',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-proof',type=Path,required=True)
    args=parser.parse_args(remove_ros_args()[1:])
    proof=json.loads(args.plan_proof.read_text())
    if not proof['success']:raise RuntimeError('Complete grasp/lift planning proof required')
    scene=json.loads(args.scene.read_text())
    if (time.time_ns()-scene['capture_wall_ns'])/1e9>120:raise RuntimeError('Capture a fresh scene before this test')
    args.output.mkdir(parents=True,exist_ok=False)
    correction=[0.,0.,0.] if args.registration is None else json.loads(args.registration.read_text())['translation_world_m']
    token=validated_token();rclpy.init();node=TestNode();telemetry=Telemetry(args.output/'raw_can.jsonl')
    report=dict(workflow='ATTENDED_VISUAL_GRASP_LIFT',success=False,stages=[],started_wall_ns=time.time_ns(),
                scene=str(args.scene),plan_proof=str(args.plan_proof),home_commanded=False,bulk_collection=False,
                local_geometry_translation_m=correction,original_calibration_modified=False)
    attached=None
    try:
        node.bind_telemetry(telemetry)
        with inject_action_capabilities(token):
            measured,object_=node.precise_scene(scene,correction);report['geometry']=measured
            target=np.eye(4);target[:3,:3]=Rotation.from_euler('xyz',[0.,math.pi/2,measured['cube_yaw']+math.pi/2]).as_matrix()
            target[:3,3]=[*measured['cube_center'][:2],measured['support_z']+.11]
            node.finger_contacts(True)
            node.move('pregrasp',target,report,args.output)
            report['gripper_open']=node.gripper(-.35)
            target[2,3]=measured['support_z']+.055
            grasp_q=node.move('vertical_approach',target,report,args.output,vertical=True)
            report['gripper_close']=node.gripper(-.085)
            grasp_q=node.refresh_rigid_state(10.)
            attached=node.attached_at(object_,grasp_q)
            node.update_object(object_,remove=True);node.attach(attached)
            report['attached_object']=message_to_ordereddict(attached)
            node.finger_contacts(True,support_contact=True)
            target[2,3]=measured['support_z']+.12
            node.move('vertical_lift',target,report,args.output,vertical=True,attached=attached)
            node.finger_contacts(True,support_contact=False)
            capture(args.output/'after_lift')
            report['lift_verification']=verify_lift(scene,args.output/'after_lift')
            report['success']=report['lift_verification']['verified']
            report['state']='HOLDING_AFTER_LIFT' if report['success'] else 'LIFT_NOT_VERIFIED'
            # Do not drop an object automatically when a camera cannot verify it.
            # Fixed release is a separate phase after inspecting this test.
    except BaseException as exc:
        report['error']=str(exc);raise
    finally:
        report['finished_wall_ns']=time.time_ns()
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        telemetry.close();node.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in report.items() if k not in ['geometry','attached_object']},indent=2))


if __name__=='__main__':main()
