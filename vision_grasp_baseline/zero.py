"""Plan a collision-checked return to mechanical zero for a fresh expert trial.

This is the seven-joint zero target, not the legacy lifted HOME trajectory.
"""
from .gripper_settings import TRANSIT_POSITION,COLLECTION_POLICY

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.utilities import remove_ros_args

from .grasp_test import TestNode
from scripts.act_trial.isolated_telemetry import Telemetry
from scripts.control_ownership.h2 import validated_token,inject_action_capabilities
from openarm_kinesthetic_recorder.execution_guard import audit_joint_trajectory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--preserve-scene',action='store_true',help='Return after release using the current placed-cube collision scene')
    a=p.parse_args(remove_ros_args()[1:])
    if a.output.exists():raise ValueError('Output already exists')
    scene=json.loads(a.scene.read_text());offset=json.loads(a.registration.read_text())['translation_world_m']
    token=validated_token(require_token=a.execute);rclpy.init();n=TestNode()
    telemetry=None
    report=dict(workflow='RETURN_MECHANICAL_ZERO',legacy_home=False,success=False,executed=False)
    report['gripper_policy']=COLLECTION_POLICY
    try:
        if a.execute:
            telemetry=Telemetry(a.output.with_suffix('.raw_can.jsonl'));n.bind_telemetry(telemetry)
        if a.preserve_scene:
            from moveit_msgs.msg import PlanningSceneComponents
            from moveit_msgs.srv import GetPlanningScene
            request=GetPlanningScene.Request()
            request.components.components=(PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
                                           PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
            live=n.call(n.scene_client,request).scene
            if live.robot_state.attached_collision_objects:
                raise RuntimeError('Cannot return empty-handed while an object remains attached')
            required={'vision_baseline_cube','vision_baseline_table','vision_baseline_bin'}
            if not required.issubset({o.id for o in live.world.collision_objects}):
                raise RuntimeError('Complete released-object collision scene required')
            report['scene_preserved_after_release']=True
        else:
            geometry,_=n.precise_scene(scene,offset);report['table_edge_padding_m']=geometry['table_edge_padding_m']
        current=n.refresh_rigid_state(10.);report['initial_q']=list(current)
        if max(abs(q) for q in current)<=.06:
            report.update(success=True,already_at_zero=True)
        else:
            response=n.plan(n.payload(current),[0.]*7,20.,.10,.10)
            if response.error_code.val!=1:raise RuntimeError('No collision-free return-to-zero path')
            jt=response.trajectory.joint_trajectory
            # A wider startup model may admit an already measured slight elbow
            # extension; recovery must never extend farther than that position.
            j4=list(jt.joint_names).index('openarm_right_joint4')
            if min(p.positions[j4] for p in jt.points)<min(current[3],-.02)-.001:
                raise RuntimeError('Return-to-zero path extends elbow beyond measured startup pose')
            for point in [jt.points[0],jt.points[-1]]:point.velocities=[0.]*7;point.accelerations=[0.]*7
            audit=audit_joint_trajectory(jt,n.expected_names,current,[0.]*7,.015).as_dict()
            report['audit']=audit;report['path_validity_samples']=n.check_path(response.trajectory)
            if a.execute:
                with inject_action_capabilities(token):
                    result=n.execute_same_trajectory(response.trajectory,audit['duration_s'],20.)
                report.update(executed=True,execution=result,success=result['action_status']==4 and result['moveit_error_code']==1)
            else:report.update(success=True,plan_only=True)
        if a.execute and report['success']:
            final=n.refresh_rigid_state(10.);report['final_q']=list(final)
            report['max_actual_zero_error_rad']=max(abs(q) for q in final)
            if report['max_actual_zero_error_rad']>.06:
                report['success']=False
                raise RuntimeError('Actual joints did not arrive at mechanical zero; inspect final_q')
            from .continuous_plan import check_gripper_sweep
            from .grasp import FINGER
            values=dict(zip(n.full_joints.name,n.full_joints.position))
            check_gripper_sweep(n,final,float(values[FINGER]),TRANSIT_POSITION,'zero_transit_close')
            with inject_action_capabilities(token):report['initial_gripper_closed']=n.gripper(TRANSIT_POSITION)
            n.refresh_rigid_state(10.)
            from .empty_gripper import verify_stationary_empty_gripper
            report['empty_gripper_acceptance']=verify_stationary_empty_gripper(n,final)
    except BaseException as exc:report.update(success=False,error=str(exc));raise
    finally:
        report['finished_wall_ns']=time.time_ns();a.output.write_text(json.dumps(report,indent=2)+'\n')
        if telemetry is not None:telemetry.close()
        n.destroy_node();rclpy.shutdown();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
