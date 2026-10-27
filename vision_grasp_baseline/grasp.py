"""Independent measured-scene grasp/lift planning, using MoveIt services.

Starts at the live held pose. Physical execution is separate from planning.
The original ACT, MTC workspace and calibration file are never modified.
"""
import argparse
from copy import deepcopy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject,
                             CollisionObject, Constraints, JointConstraint, OrientationConstraint,
                             PlanningSceneComponents, PositionConstraint, RobotState)
from moveit_msgs.srv import GetMotionPlan,GetPlanningScene,GetPositionIK,ApplyPlanningScene,GetStateValidity
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from rosidl_runtime_py.convert import message_to_ordereddict
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive

from .hover import HoverNode,NAMES,EE,TCP,box,matrix_pose,pose_matrix
from .execution_quality import task_joint4_bounds
from scripts.control_ownership.h2 import validated_token
from openarm_kinesthetic_recorder.execution_guard import audit_joint_trajectory

FINGER='openarm_right_finger_joint1'
TOUCH=['openarm_right_ee_link1','openarm_right_ee_link2']


class GraspNode(HoverNode):
    def __init__(self):
        # Planning and execution both need the measured finger, not just the
        # seven arm joints retained by HomeRuntimeNode. Reuse its subscription
        # so arm/finger acceptance comes from the very same JointState message.
        self.full_joints=None
        self.full_joint_state_monotonic_s=None
        super().__init__()

    def _joint_state(self,message):
        self.full_joints=None
        self.full_joint_state_monotonic_s=None
        names=list(message.name)
        if len(names)!=len(message.position) or len(set(names))!=len(names):
            return
        values=dict(zip(names,message.position))
        if any(name not in values or not math.isfinite(values[name]) for name in [*NAMES,FINGER]):
            return
        super()._joint_state(message)
        self.full_joints=message
        self.full_joint_state_monotonic_s=self.current_state_monotonic_s

    def refresh_rigid_state(self,timeout_s):
        # Never let an earlier full-state cache satisfy a new fresh-arm read.
        self.full_joints=None
        self.full_joint_state_monotonic_s=None
        current=super().refresh_rigid_state(timeout_s)
        if (self.full_joints is None or self.full_joint_state_monotonic_s is None
                or not 0<=time.monotonic()-self.full_joint_state_monotonic_s<=.5):
            raise RuntimeError('Fresh complete arm/gripper /joint_states unavailable')
        return current

    @contextmanager
    def planning_attachment(self,object_,attached):
        """Keep an attachment in the monitored scene during virtual lift planning.

        A world REMOVE with only request-local attached bodies leaves the global
        state unattached. PlanningScene::pushDiffs then removes this object's ACM
        entry when it processes DESTROY, racing subsequent support-contact rules.
        Apply the same complete attachment transition as physical grasp; restore
        the real stationary world object even when planning fails. No joint goals.
        """
        request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
        request.scene.robot_state.attached_collision_objects=[deepcopy(attached)]
        try:
            if not self.call(self.apply_client,request).success:
                raise RuntimeError('Temporary planning attachment rejected')
            query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            held=self.call(self.scene_client,query).scene.robot_state.attached_collision_objects
            if object_.id not in [body.object.id for body in held]:
                raise RuntimeError('Temporary planning attachment readback missing')
            yield
        finally:
            restore=ApplyPlanningScene.Request();restore.scene.is_diff=True;restore.scene.robot_state.is_diff=True
            body=deepcopy(attached);body.object.operation=CollisionObject.REMOVE
            restore.scene.robot_state.attached_collision_objects=[body]
            restore.scene.world.collision_objects=[deepcopy(object_)]
            if not self.call(self.apply_client,restore).success:
                raise RuntimeError('Could not restore world cube after virtual planning')
            query=GetPlanningScene.Request()
            query.components.components=(PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
                                         PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
            restored=self.call(self.scene_client,query).scene
            if (object_.id in [b.object.id for b in restored.robot_state.attached_collision_objects]
                    or object_.id not in [o.id for o in restored.world.collision_objects]):
                raise RuntimeError('World cube restoration readback failed')

    def update_object(self,object_,*,remove=False):
        request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
        message=deepcopy(object_)
        if remove:message.operation=CollisionObject.REMOVE
        request.scene.world.collision_objects=[message]
        if not self.call(self.apply_client,request).success:raise RuntimeError('Scene object update failed')
        query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        objects=self.call(self.scene_client,query).scene.world.collision_objects
        if ((object_.id in [o.id for o in objects]) == remove):
            raise RuntimeError('Scene object update readback mismatch')

    def precise_scene(self,scene,correction,*,table_edge_padding_m=.03):
        if not math.isfinite(table_edge_padding_m) or table_edge_padding_m<0:
            raise ValueError('Table edge padding must be finite and nonnegative')
        check=GetPlanningScene.Request();check.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        if any(x.object.id=='vision_baseline_cube' for x in self.call(self.scene_client,check).scene.robot_state.attached_collision_objects):
            raise RuntimeError('A cube is still attached; continue its release before starting another grasp')
        s=deepcopy(scene);offset=np.asarray(correction)
        m=s['measured'];table=m['table'];cube=m['cube'];bin_=m['bin']
        n=np.asarray(table['normal']);table_xy=np.asarray(table['center_xy'])+offset[:2]
        table_xyz=np.r_[table_xy,table['top_z']+offset[2]]
        x=np.array([math.cos(table['yaw']),math.sin(table['yaw']),0.])
        x-=n*(x@n);x/=np.linalg.norm(x);y=np.cross(n,x)
        r=np.column_stack((x,y,n));center=table_xyz-n*(.35/2-.003)
        support=box('vision_baseline_table',center,[*np.maximum(table['size_xy'],[.4,.6])+2*table_edge_padding_m,.35])
        support.primitive_poses[0]=matrix_pose(np.block([[r,center[:,None]],[np.zeros((1,3)),np.ones((1,1))]]))
        if 'table_collision_envelope' in s:
            envelope=s['table_collision_envelope']
            support=box('vision_baseline_table',np.asarray(envelope['center'])+offset,envelope['size'])
            transform=np.eye(4);transform[:3,:3]=envelope['rotation'];transform[:3,3]=np.asarray(envelope['center'])+offset
            support.primitive_poses[0]=matrix_pose(transform)
        # This known fixture is a measured nominal 40mm cube. Infer its center
        # from the upper surface rather than treating noisy visible widths as
        # a smaller physical object.
        cube_center=np.array(cube['center'])+offset
        cube_center[2]=cube['center'][2]+cube['size'][2]/2-.020+offset[2]
        object_=box('vision_baseline_cube',cube_center,[.04]*3,cube['yaw'])
        bin_center=[*(np.asarray(bin_['center_xy'])+offset[:2]),bin_['support_z']+offset[2]+bin_['rim_height']/2]
        bin_obj=box('vision_baseline_bin',bin_center,[*(np.asarray(bin_['outer_size_xy'])+.008),bin_['rim_height']+.006],bin_['yaw'])
        request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
        request.scene.world.collision_objects=[support,bin_obj,object_]
        if not self.call(self.apply_client,request).success:raise RuntimeError('Measured scene rejected')
        return dict(correction_world_m=offset.tolist(),cube_center=cube_center.tolist(),cube_yaw=cube['yaw'],
                    support_z=cube['support_z']+offset[2],table_edge_padding_m=table_edge_padding_m,
                    objects=[message_to_ordereddict(o) for o in request.scene.world.collision_objects]),object_

    def finger_contacts(self,allowed,*,support_contact=False):
        # ROS bool[] conversion asserts on numpy.bool_ (rather than raising).
        allowed=bool(allowed);support_contact=bool(support_contact)
        query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm=self.call(self.scene_client,query).scene.allowed_collision_matrix
        names=list(acm.entry_names);old={name:list(row.enabled) for name,row in zip(names,acm.entry_values)}
        for key in ['vision_baseline_cube','vision_baseline_table',*TOUCH]:
            if key not in names:names.append(key)
        rows=[]
        for name in names:
            row=AllowedCollisionEntry();row.enabled=[old.get(name,[])[acm.entry_names.index(other)] if name in old and other in acm.entry_names else False for other in names]
            rows.append(row)
        for finger in TOUCH:
            a,b=names.index(finger),names.index('vision_baseline_cube')
            rows[a].enabled[b]=allowed;rows[b].enabled[a]=allowed
        a,b=names.index('vision_baseline_cube'),names.index('vision_baseline_table')
        rows[a].enabled[b]=support_contact;rows[b].enabled[a]=support_contact
        acm.entry_names=names;acm.entry_values=rows
        request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
        request.scene.allowed_collision_matrix=acm
        if not self.call(self.apply_client,request).success:raise RuntimeError('Finger contact update rejected')

    def state(self,q,gripper,attached=None):
        state=RobotState();state.is_diff=True
        state.joint_state.name=[*NAMES,FINGER];state.joint_state.position=[*q,gripper]
        if attached is not None:state.attached_collision_objects=[attached]
        return state

    def plan_pose(self,label,target,start,gripper,attached=None,vertical=False):
        ee_tcp=self.tcp_transform(start);ee_target=target @ np.linalg.inv(ee_tcp)
        start_query=GetStateValidity.Request();start_query.group_name='right_arm'
        start_query.robot_state=self.state(start,gripper,attached)
        start_valid=self.call(self.valid_client,start_query)
        print(json.dumps(dict(stage=label,start_valid=start_valid.valid,
                              start_contacts=[(c.contact_body_1,c.contact_body_2) for c in start_valid.contacts])),flush=True)
        if not start_valid.valid:raise RuntimeError(label+': start state is in collision')
        goals=[]
        seeds=[start,[0.,.5,0.,1.2,0.,-.4,0.],[.8,.5,-.8,1.2,-.6,.5,1.4]]
        for seed in seeds:
            request=GetPositionIK.Request();ik=request.ik_request;ik.group_name='right_arm';ik.ik_link_name=EE
            ik.robot_state=self.state(seed,gripper,attached);ik.avoid_collisions=False;ik.timeout.nanosec=400000000
            ik.pose_stamped.header.frame_id='world';ik.pose_stamped.pose=matrix_pose(ee_target)
            response=self.call(self.ik_client,request)
            if response.error_code.val!=1:continue
            values=dict(zip(response.solution.joint_state.name,response.solution.joint_state.position))
            q=[values[k] for k in NAMES]
            validity=GetStateValidity.Request();validity.group_name='right_arm';validity.robot_state=self.state(q,gripper,attached)
            checked=self.call(self.valid_client,validity)
            if not checked.valid:
                print(json.dumps(dict(stage=label,goal_collision=[(c.contact_body_1,c.contact_body_2) for c in checked.contacts])),flush=True)
                continue
            goals.append(q)
        if not goals:raise RuntimeError(label+': no collision-free goal IK')
        goals.sort(key=lambda q:np.linalg.norm(np.asarray(q)-start))
        for goal in goals:
            req=GetMotionPlan.Request();motion=req.motion_plan_request
            motion.group_name='right_arm';motion.pipeline_id='ompl';motion.num_planning_attempts=10
            motion.allowed_planning_time=8.;motion.max_velocity_scaling_factor=.1;motion.max_acceleration_scaling_factor=.1
            motion.start_state=self.state(start,gripper,attached)
            constraints=Constraints();constraints.name=label
            for name,value in zip(NAMES,goal):
                joint=JointConstraint();joint.joint_name=name;joint.position=value
                joint.tolerance_above=.003;joint.tolerance_below=.003;joint.weight=1.;constraints.joint_constraints.append(joint)
            motion.goal_constraints=[constraints]
            if vertical:
                current=pose_matrix(self.fk_pose({**self.payload(start),'end_effector_link':TCP},start,10.).pose)
                pc=PositionConstraint();pc.header.frame_id='world';pc.link_name=EE;pc.weight=1.
                pc.target_point_offset.x,pc.target_point_offset.y,pc.target_point_offset.z=map(float,ee_tcp[:3,3])
                shape=SolidPrimitive();shape.type=SolidPrimitive.CYLINDER
                radius=max(.005,float(np.linalg.norm(current[:2,3]-target[:2,3]))+.002)
                shape.dimensions=[.5,radius]
                center=Pose();center.position.x=float(target[0,3]);center.position.y=float(target[1,3]);center.position.z=float((current[2,3]+target[2,3])/2);center.orientation.w=1.
                pc.constraint_region.primitives=[shape];pc.constraint_region.primitive_poses=[center]
                oc=OrientationConstraint();oc.header.frame_id='world';oc.link_name=EE;oc.orientation=matrix_pose(ee_target).orientation
                oc.absolute_x_axis_tolerance=.06;oc.absolute_y_axis_tolerance=.06;oc.absolute_z_axis_tolerance=.06;oc.weight=1.
                motion.path_constraints.position_constraints=[pc];motion.path_constraints.orientation_constraints=[oc]
                start_query.constraints=motion.path_constraints
                constrained=self.call(self.valid_client,start_query)
                print(json.dumps(dict(stage=label,constrained_start_valid=constrained.valid,
                                      start_contacts=[(c.contact_body_1,c.contact_body_2) for c in constrained.contacts],
                                      constraint_results=[message_to_ordereddict(x) for x in constrained.constraint_result])),flush=True)
                if not constrained.valid:raise RuntimeError(label+': constrained start state is invalid (collision or path constraints): '+str([(c.contact_body_1,c.contact_body_2) for c in constrained.contacts]))
            result=self.call(self.plan_client,req,30.).motion_plan_response
            if result.error_code.val!=1:
                print(json.dumps(dict(stage=label,planning_error_code=result.error_code.val,
                                      message=result.error_code.message)),flush=True)
            if result.error_code.val==1:break
        else:raise RuntimeError(label+': path planning failed')
        jt=result.trajectory.joint_trajectory
        task_joint4_bounds(jt.joint_names,jt.points)
        for p in [jt.points[0],jt.points[-1]]:p.velocities=[0.]*7;p.accelerations=[0.]*7
        audit=audit_joint_trajectory(jt,NAMES,start,goal,.015).as_dict()
        print(json.dumps(dict(stage=label,planned=True,duration_s=audit['duration_s'],vertical_constraint=vertical)),flush=True)
        return result.trajectory,dict(name=label,target_tcp_world=target.tolist(),start=start,goal=goal,audit=audit,
                                      trajectory=message_to_ordereddict(result.trajectory),vertical_constraint=vertical)

    def attached_at(self,object_,q):
        ee=pose_matrix(self.fk_pose({**self.payload(q),'end_effector_link':EE},q,10.).pose)
        relative=np.linalg.inv(ee) @ pose_matrix(object_.primitive_poses[0])
        attached=AttachedCollisionObject();attached.link_name=EE;attached.touch_links=TOUCH
        attached.object=deepcopy(object_);attached.object.header.frame_id=EE
        attached.object.primitive_poses=[matrix_pose(relative)]
        return attached


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--registration',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-zero',action='store_true',help='Plan-only virtual zero start, with no physical reset')
    a=p.parse_args(remove_ros_args()[1:])
    if a.output.exists():raise ValueError('Output already exists')
    source=json.loads(a.scene.read_text());correction=[0.,0.,0.]
    if a.registration:correction=json.loads(a.registration.read_text())['translation_world_m']
    validated_token(require_token=False);rclpy.init();n=GraspNode()
    report=dict(workflow='VISUAL_GRASP_LIFT_PLAN_ONLY',no_motion=True,success=False,scene=str(a.scene),stages=[])
    try:
        current=n.refresh_rigid_state(10.)
        if a.start_zero:current=[0.]*7
        measured,object_=n.precise_scene(source,correction)
        report['geometry']=measured;n.finger_contacts(True)
        target=np.eye(4);target[:3,:3]=Rotation.from_euler('xyz',[0.,math.pi/2,measured['cube_yaw']+math.pi/2]).as_matrix()
        target[:3,3]=[*measured['cube_center'][:2],measured['support_z']+.11]
        _,pre=n.plan_pose('pregrasp',target,current,0.)
        report['stages'].append(pre)
        target[2,3]=measured['support_z']+.055
        _,grasp=n.plan_pose('vertical_approach',target,pre['goal'],-.35,vertical=True)
        report['stages'].append(grasp)
        attached=n.attached_at(object_,grasp['goal'])
        with n.planning_attachment(object_,attached):
            # Allow the supported cube/table pair while lifting away. Robot/table
            # remains checked; the monitored attachment keeps this ACM entry alive.
            n.finger_contacts(True,support_contact=True)
            target[2,3]=measured['support_z']+.12
            _,lift=n.plan_pose('vertical_lift',target,grasp['goal'],-.085,attached,vertical=True)
        report['stages'].append(lift);report['success']=True
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        if 'object_' in locals():n.update_object(object_)
        n.finger_contacts(False)
        report['finished_wall_ns']=time.time_ns();a.output.write_text(json.dumps(report,indent=2)+'\n')
        n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
