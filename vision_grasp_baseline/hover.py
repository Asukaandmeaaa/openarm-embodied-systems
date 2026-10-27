"""Independent ROS/MoveIt point check, starting at mechanical zero, no HOME.

Reuse the installed runtime's fresh-state and execution monitoring primitives.
The target is above the object; this program never closes/opens the gripper or
descends to grasp. Commands use the existing connection-bound H2 gateway.
"""
import argparse
import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK, GetPlanningScene, GetStateValidity
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import RobotTrajectory
from scipy.spatial.transform import Rotation

from openarm_kinesthetic_recorder.home_runtime import HomeRuntimeNode
from openarm_kinesthetic_recorder.execution_guard import audit_joint_trajectory
from scripts.control_ownership.h2 import inject_action_capabilities, validated_token

NAMES = [f'openarm_right_joint{i}' for i in range(1,8)]
EE = 'openarm_right_ee_base_link'
TCP = 'openarm_right_grasp_frame'


def pose_matrix(p):
    t=np.eye(4)
    t[:3,:3]=Rotation.from_quat([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]).as_matrix()
    t[:3,3]=[p.position.x,p.position.y,p.position.z]
    return t


def matrix_pose(t):
    p=Pose()
    p.position.x,p.position.y,p.position.z=map(float,t[:3,3])
    p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=map(float,Rotation.from_matrix(t[:3,:3]).as_quat())
    return p


def box(identifier,position,size,yaw=0.):
    obj=CollisionObject()
    obj.id=identifier; obj.header.frame_id='world';obj.operation=CollisionObject.ADD
    shape=SolidPrimitive();shape.type=SolidPrimitive.BOX;shape.dimensions=list(map(float,size))
    p=Pose();p.position.x,p.position.y,p.position.z=map(float,position)
    p.orientation.z=math.sin(yaw/2);p.orientation.w=math.cos(yaw/2)
    obj.primitives=[shape];obj.primitive_poses=[p]
    return obj


class HoverNode(HomeRuntimeNode):
    def __init__(self):
        super().__init__('home_guarded_executor') # Existing H2 DDS service namespace.
        self.ik_client=self.create_client(GetPositionIK,'/compute_ik')
        self.apply_client=self.create_client(ApplyPlanningScene,'/apply_planning_scene')
        self.valid_client=self.create_client(GetStateValidity,'/check_state_validity')

    def call(self,client,request,timeout=15.):
        self.wait_for_service(client,client.srv_name,timeout)
        f=client.call_async(request)
        rclpy.spin_until_future_complete(self,f,timeout_sec=timeout)
        if not f.done() or f.result() is None:
            raise RuntimeError('Service timed out: '+client.srv_name)
        return f.result()

    def apply_scene(self,scene):
        m=scene['measured'];table=m['table'];cube=m['cube'];bin_=m['bin']
        # Extend the partial RGB-D footprint to at least the recorded 40x60cm.
        size=np.maximum(table['size_xy'],[.40,.60])+.04
        c=np.asarray(table['center_xy']);n=np.asarray(table['normal']);d=table['plane_d']
        r=Rotation.from_euler('z',table['yaw']).as_matrix()[:2,:2]
        corners=np.asarray([[a*size[0]/2,b*size[1]/2] for a in [-1,1] for b in [-1,1]]) @ r.T+c
        ceiling=float(max(-(corners@n[:2]+d)/n[2])+.012)
        objects=[box('vision_baseline_table',[*c,(ceiling-.05)/2],[*size,ceiling+.05],table['yaw']),
                 box('vision_baseline_bin', [*bin_['center_xy'],bin_['support_z']+bin_['rim_height']/2],
                     [*(np.asarray(bin_['outer_size_xy'])+.008),bin_['rim_height']+.008],bin_['yaw']),
                 box('vision_baseline_cube',cube['center'],np.maximum(cube['size'],[.04,.04,.04]),cube['yaw'])]
        request=ApplyPlanningScene.Request();request.scene.is_diff=True
        request.scene.robot_state.is_diff=True;request.scene.world.collision_objects=objects
        if not self.call(self.apply_client,request).success:
            raise RuntimeError('MoveIt refused current visual collision scene')
        query=GetPlanningScene.Request();query.components.components=16 # WORLD_OBJECT_GEOMETRY
        response=self.call(self.scene_client,query)
        found={o.id:o for o in response.scene.world.collision_objects}
        for o in objects:
            if o.id not in found or message_to_ordereddict(found[o.id])['primitives'] != message_to_ordereddict(o)['primitives']:
                raise RuntimeError('Visual collision scene readback mismatch: '+o.id)
        return dict(table_conservative_top_z=ceiling,collision_objects=[message_to_ordereddict(o) for o in objects])

    def payload(self,q):
        return dict(joint_names=NAMES,start_positions_rad=list(q),world_frame='world',move_group='right_arm')

    def tcp_transform(self,q):
        payload=self.payload(q)
        ee=self.fk_pose({**payload,'end_effector_link':EE},q,10.).pose
        tcp=self.fk_pose({**payload,'end_effector_link':TCP},q,10.).pose
        return np.linalg.inv(pose_matrix(ee)) @ pose_matrix(tcp)

    def check_path(self,trajectory):
        # Check every waypoint and all segment midpoints against the real
        # MoveIt scene, with self-collision and the other arm present.
        points=trajectory.joint_trajectory.points
        sampled=[]
        for a,b in zip(points,points[1:]):
            sampled.extend([list(a.positions),((np.asarray(a.positions)+b.positions)/2).tolist()])
        sampled.append(list(points[-1].positions))
        for q in sampled:
            req=GetStateValidity.Request();req.group_name='right_arm'
            req.robot_state.joint_state.name=NAMES;req.robot_state.joint_state.position=q
            req.robot_state.is_diff=True
            result=self.call(self.valid_client,req)
            if not result.valid:
                raise RuntimeError('Invalid planned state: '+str([(c.contact_body_1,c.contact_body_2) for c in result.contacts]))
        return len(sampled)

    def make_plan(self,scene,*,from_current=False,target_xyz=None):
        start=self.refresh_rigid_state(10.)
        if not from_current and max(abs(q) for q in start)>.06:
            raise RuntimeError('Point check requires mechanical-zero start; no automatic HOME/recovery')
        collision=self.apply_scene(scene)
        ee_tcp=self.tcp_transform(start)
        cube=scene['measured']['cube']
        # 16cm above the local tabletop, and at least 12cm above the
        # conservative collision ceiling. This is an XY alignment check.
        xyz=[*cube['center'][:2],max(cube['support_z']+.16,collision['table_conservative_top_z']+.12)]
        if target_xyz is not None:
            xyz=list(target_xyz)
            if xyz[2]<collision['table_conservative_top_z']+.10:
                raise ValueError('An alignment-check point must clear the conservative table by 10cm')
        grasp_yaw=cube['yaw']+math.pi/2
        target=np.eye(4);target[:3,:3]=Rotation.from_euler('xyz',[0.,math.pi/2,grasp_yaw]).as_matrix();target[:3,3]=xyz
        ee_target=target @ np.linalg.inv(ee_tcp)
        candidates=[]
        for elbow in [0.,.8,1.2,1.6]:
            seed=list(start);seed[1]=.5;seed[3]=elbow;seed[5]=-.4
            req=GetPositionIK.Request();ik=req.ik_request
            ik.group_name='right_arm';ik.ik_link_name=EE;ik.avoid_collisions=True
            ik.pose_stamped.header.frame_id='world';ik.pose_stamped.pose=matrix_pose(ee_target)
            ik.robot_state.joint_state.name=NAMES;ik.robot_state.joint_state.position=seed;ik.robot_state.is_diff=True
            ik.timeout.sec=2
            result=self.call(self.ik_client,req,10.)
            print(json.dumps(dict(ik_seed_elbow=elbow,error_code=result.error_code.val,
                                  target_tcp=xyz,yaw=grasp_yaw)),flush=True)
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            values=dict(zip(result.solution.joint_state.name,result.solution.joint_state.position))
            q=[values[n] for n in NAMES]
            candidates.append(q)
        if not candidates:
            raise RuntimeError('No collision-free IK for the fixed vertical hover pose')
        candidates.sort(key=lambda q:np.linalg.norm(np.asarray(q)-start))
        for goal in candidates:
            goal_tcp=self.fk_pose({**self.payload(goal),'end_effector_link':TCP},goal,10.).pose
            actual=pose_matrix(goal_tcp)
            if np.linalg.norm(actual[:3,3]-target[:3,3])>.005 or Rotation.from_matrix(actual[:3,:3] @ target[:3,:3].T).magnitude()>.03:
                continue
            response=self.plan(self.payload(start),goal,20.,.10,.10)
            if response.error_code.val==MoveItErrorCodes.SUCCESS:
                break
        else:
            raise RuntimeError('No collision-free path from zero to the hover pose')
        jt=response.trajectory.joint_trajectory
        # Controllers require a stopped boundary between independent segments.
        for p in [jt.points[0],jt.points[-1]]:
            p.velocities=[0.]*len(NAMES);p.accelerations=[0.]*len(NAMES)
        audit=audit_joint_trajectory(jt,NAMES,start,goal,.015).as_dict()
        path_samples=self.check_path(response.trajectory)
        self.publish_display(response)
        return dict(schema_version=1,workflow='CURRENT_TO_ALIGNMENT_HOVER' if from_current else 'ZERO_TO_VISUAL_HOVER',created_wall_ns=time.time_ns(),
                    scene=scene,collision=collision,start_positions_rad=start,goal_positions_rad=goal,
                    target_tcp_world=target.tolist(),ee_to_tcp=ee_tcp.tolist(),audit=audit,
                    path_validity_samples=path_samples,trajectory=message_to_ordereddict(response.trajectory),
                    gripper_commanded=False,home_commanded=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['plan','execute'],required=True)
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--from-current',action='store_true',help='Plan another high alignment point without HOME')
    parser.add_argument('--target-xyz',type=float,nargs=3,help='Optional high alignment point in world coordinates')
    args=parser.parse_args(remove_ros_args()[1:])
    token=validated_token(require_token=args.mode=='execute')
    rclpy.init();node=HoverNode()
    try:
        if args.mode=='plan':
            if args.scene is None or args.plan.exists():
                raise ValueError('Supply --scene and a new --plan path')
            plan=node.make_plan(json.loads(args.scene.read_text()),from_current=args.from_current,target_xyz=args.target_xyz)
            args.plan.write_text(json.dumps(plan,indent=2)+'\n')
            print(json.dumps({k:v for k,v in plan.items() if k not in ('trajectory','scene')},indent=2))
        else:
            if args.report is None or args.report.exists():
                raise ValueError('Supply a new --report path')
            plan=json.loads(args.plan.read_text())
            trajectory=RobotTrajectory();set_message_fields(trajectory,plan['trajectory'])
            current=node.refresh_rigid_state(10.)
            if max(abs(a-b) for a,b in zip(current,plan['start_positions_rad']))>.015:
                raise RuntimeError('Live start differs from planned start')
            if (time.time_ns()-plan['scene']['capture_wall_ns'])/1e9>120:
                raise RuntimeError('Scene snapshot older than two minutes; recapture before point check')
            node.apply_scene(plan['scene'])
            node.check_path(trajectory)
            report=dict(workflow=plan['workflow'],plan=str(args.plan),started_wall_ns=time.time_ns(),success=False)
            try:
                with inject_action_capabilities(token):
                    result=node.execute_same_trajectory(trajectory,plan['audit']['duration_s'],15.)
                report['execution']=result
                if result['moveit_error_code']!=1 or result['action_status']!=4:
                    raise RuntimeError('Hover action did not succeed')
                final=node.refresh_rigid_state(10.)
                report['final_positions_rad']=final
                tcp=node.fk_pose({**node.payload(final),'end_effector_link':TCP},final,10.).pose
                report['final_tcp_world']=pose_matrix(tcp).tolist()
                report['final_joint_error_rad']=[a-b for a,b in zip(final,plan['goal_positions_rad'])]
                report['maximum_final_joint_error_rad']=max(abs(v) for v in report['final_joint_error_rad'])
                report['final_tcp_position_error_m']=float(np.linalg.norm(
                    np.asarray(report['final_tcp_world'])[:3,3]-np.asarray(plan['target_tcp_world'])[:3,3]))
                # JTC/MoveIt already evaluated the configured execution
                # tolerances. Do not layer HOME's unrelated .03rad terminal
                # threshold over this attended visual hover point check.
                report['success']=True
                report['acceptance']='Existing controller and MoveIt execution result; tracking errors recorded'
            except BaseException as exc:
                report['error']=str(exc)
                raise
            finally:
                report['finished_wall_ns']=time.time_ns()
                args.report.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(report,indent=2))
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    main()
