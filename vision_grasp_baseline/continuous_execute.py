"""Replay one prevalidated complete timeline through two synchronized JTC goals."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.utilities import remove_ros_args
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from .collection_io import rpc,event,write_json
from .continuous_plan import BASE,file_hash,validate_timeline
from .grasp import FINGER
from .grasp_test import capture
from .hover import NAMES,box
from .place_test import verify_placed
from .execution_quality import endpoint_quality
from .waypoint_place_test import WaypointNode,scene_attachment
from .table_envelope import admit_scene
from .scene_estimation import check_replay_fixture
from .live_collision import LiveCollisionMonitor
from .controller_readiness import verify_required_controllers
from .runtime_recording import RuntimeRecording
from .pickup_clearance import require_pickup_clearance
from .gripper_settings import COLLECTION_POLICY,require_current_plan,require_closed_transit
from .empty_gripper import ReturnClosureGuard
from scripts.act_trial.isolated_telemetry import Telemetry
from scripts.control_ownership.h2 import validated_token,inject_action_capabilities


def validate_proof(proof,binding,now_ns):
    age=(now_ns-proof.get('finished_wall_ns',0))/1e9
    if (not proof.get('success') or not proof.get('check_only') or proof.get('physical_motion') is not False
            or not proof.get('validation',{}).get('passed') or proof.get('file_binding')!=binding or not 0<=age<=30):
        raise ValueError('Missing, stale or mismatched read-only validation proof')
    return proof['validation']


def build_goals(plan):
    # Use the existing H2 permission boundary: MoveIt forwards the arm action
    # through the gateway; H2 sends the gripper action through that same gateway.
    # Each gets its own connection-bound nonce and the same absolute start time.
    goals=[ExecuteTrajectory.Goal(),FollowJointTrajectory.Goal()]
    goals[0].controller_names=['right_joint_trajectory_controller']
    trajectories=[goals[0].trajectory.joint_trajectory,goals[1].trajectory]
    trajectories[0].joint_names=NAMES;trajectories[1].joint_names=[FINGER]
    for row in plan['samples']:
        ns=round(row['time_s']*1e9)
        for trajectory,indices in zip(trajectories,(slice(0,7),slice(7,8))):
            point=JointTrajectoryPoint()
            for key in ('positions','velocities','accelerations'):setattr(point,key,row[key][indices])
            point.time_from_start.sec,point.time_from_start.nanosec=divmod(ns,10**9)
            trajectory.points.append(point)
    return goals


def execute(node,plan,goals,recording,on_event=None,before_dispatch=None,collision_monitor=None,on_dispatch=None):
    """One shared ROS start time; cancel both if either action/guard fails."""
    require_current_plan(plan)
    clients=[node.execute_client,node.gripper_client]
    handles=[];send_futures=[];results=[];observed=[];recorder_io=None
    for client in clients:
        if not client.wait_for_server(timeout_sec=10.):raise RuntimeError('Authorized trajectory action unavailable')
    controller_readiness=verify_required_controllers(node,('right_joint_trajectory_controller','right_gripper_controller'))
    if collision_monitor is not None:collision_monitor.prepare()
    q=node.refresh_rigid_state(10.)
    if node.passive_state is not False:
        raise RuntimeError('Fresh rigid-mode confirmation required before dispatch')
    first=plan['samples'][0]['positions']
    values=dict(zip(node.full_joints.name,node.full_joints.position))
    if max(abs(np.asarray(q)-first[:7]))>.015 or abs(values[FINGER]-first[7])>.04:
        raise RuntimeError('Actual start differs from cached trajectory; verify closed-gripper zero and replan')
    require_closed_transit(float(values[FINGER]))
    closure_guard=ReturnClosureGuard(plan)
    if before_dispatch is not None:before_dispatch()
    if recording:recorder_io=RuntimeRecording()
    start=node.get_clock().now()+Duration(seconds=1.)
    goals[0].trajectory.joint_trajectory.header.stamp=start.to_msg()
    goals[1].trajectory.header.stamp=start.to_msg()
    def succeeded(wrapper,index):
        if wrapper is None or wrapper.status!=4:return False
        return wrapper.result.error_code.val==1 if index==0 else wrapper.result.error_code==0
    try:
        if on_dispatch is not None:on_dispatch()
        send_futures=[client.send_goal_async(goal) for client,goal in zip(clients,goals)]
        deadline=time.monotonic()+.65
        while not all(f.done() for f in send_futures) and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.005)
        if not all(f.done() for f in send_futures):raise RuntimeError('Both controllers must accept before shared start time')
        handles=[f.result() for f in send_futures]
        if any(h is None or not h.accepted for h in handles):raise RuntimeError('A synchronized trajectory was rejected')
        results=[h.get_result_async() for h in handles]
        if (start-node.get_clock().now()).nanoseconds<100_000_000:
            raise RuntimeError('Insufficient controller dispatch margin')
        emit=recorder_io.event if recorder_io is not None else event
        emit('continuous_dispatch',duration_s=plan['duration_s'],arm_goals=1,gripper_goals=1,
              start_ros_ns=start.nanoseconds,planning_during_execution=False)
        next_event=0;deadline=time.monotonic()+plan['duration_s']+12.
        while not all(f.done() for f in results):
            rclpy.spin_once(node,timeout_sec=.005)
            if time.monotonic()>deadline:raise RuntimeError('Complete timeline execution timed out')
            elapsed=(node.get_clock().now()-start).nanoseconds/1e9
            if node.passive_state is not False or node.current_state_monotonic_s is None or time.monotonic()-node.current_state_monotonic_s>.5:
                age=None if node.current_state_monotonic_s is None else time.monotonic()-node.current_state_monotonic_s
                raise RuntimeError(f'Rigid mode / fresh joint feedback lost: passive={node.passive_state}, state_age_s={age}')
            failure=node.execution_table_guard_failure()
            if failure:raise RuntimeError(failure)
            measured=dict(zip(node.full_joints.name,node.full_joints.position))[FINGER]
            closure_guard.check(elapsed,float(measured),node.full_joint_state_monotonic_s,time.monotonic())
            if collision_monitor is not None:collision_monitor.check()
            for index,future in enumerate(results):
                if future.done():
                    wrapper=future.result()
                    if not succeeded(wrapper,index):
                        raise RuntimeError('Controller execution failed: '+str(wrapper))
            if recorder_io is not None:recorder_io.check()
            while next_event<len(plan['events']) and elapsed>=plan['events'][next_event]['time_s']:
                mark=plan['events'][next_event]
                if mark['name']=='return_closed':
                    require_closed_transit(float(dict(zip(node.full_joints.name,node.full_joints.position))[FINGER]))
                    if not closure_guard.confirmed:raise RuntimeError('Missing pre-return empty-gripper acceptance')
                if mark['name']=='approach':
                    # Empty-gripper transitions must actually finish before
                    # descending or starting the closed return. Never wait or
                    # silently retime a running synchronized trajectory.
                    index=min(range(len(plan['samples'])),key=lambda i:abs(plan['samples'][i]['time_s']-mark['time_s']))
                    values=dict(zip(node.full_joints.name,node.full_joints.position))
                    if abs(values[FINGER]-plan['samples'][index]['positions'][7])>.04:
                        raise RuntimeError('Empty gripper did not reach '+mark['name']+' reference')
                # The timeline continues autonomously; endpoint checking does
                # not request a new path or wait for a service between phases.
                if mark['name'] in ('pregrasp_open','close','open','transit_close'):
                    index=min(range(len(plan['samples'])),key=lambda i:abs(plan['samples'][i]['time_s']-mark['time_s']))
                    expected=node.chain.pose(dict(zip(NAMES,plan['samples'][index]['positions'][:7])))
                    actual=node.chain.pose(dict(zip(NAMES,node.current_positions)))
                    if not endpoint_quality(expected,actual)['accepted']:
                        raise RuntimeError('TCP did not reach '+mark['name']+' position before scheduled gripper motion')
                observed.append(dict(**mark,observed_time_s=elapsed))
                if on_event is not None:on_event(mark)
                emit('timeline_event',**observed[-1]);next_event+=1
        for index,future in enumerate(results):
            wrapper=future.result()
            if not succeeded(wrapper,index):
                raise RuntimeError('Controller terminal status failed')
        final=node.refresh_rigid_state(10.)
        if max(abs(x) for x in final)>.06:raise RuntimeError('Final mechanical zero not reached')
        values=dict(zip(node.full_joints.name,node.full_joints.position))
        require_closed_transit(float(values[FINGER]))
        if not closure_guard.confirmed:raise RuntimeError('Empty-gripper return acceptance was not observed')
        if recorder_io is not None:recorder_io.close(drain=True)
        return dict(success=True,events=observed,final_q=final,final_gripper=float(values[FINGER]),
                    controller_readiness=controller_readiness,
                    empty_return_closure_verified=True,
                    gripper_policy=COLLECTION_POLICY,arm_goals=1,gripper_goals=1,planning_calls_during_execution=0)
    except BaseException as exc:
        # Cancellation revokes H2 authority; log the primary cause before the
        # lease supervisor sees that secondary renewal failure.
        print('Continuous execution stopped: '+str(exc),flush=True)
        # A goal response may arrive after the dispatch timeout. Collect it
        # briefly so an admitted companion cannot remain unattended.
        deadline=time.monotonic()+1.
        while any(not f.done() for f in send_futures) and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.01)
        for future in send_futures:
            if future.done():
                handle=future.result()
                if handle is not None and handle.accepted:
                    cancel=handle.cancel_goal_async();rclpy.spin_until_future_complete(node,cancel,timeout_sec=3.)
        raise
    finally:
        if recorder_io is not None:recorder_io.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--robot-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record-episode',type=Path)
    p.add_argument('--check-only',action='store_true',help='Validate without acquiring motion authority or sending goals')
    p.add_argument('--validation-proof',type=Path,help='Fresh read-only validation result bound to these exact files')
    a=p.parse_args(remove_ros_args()[1:]);a.output.mkdir(parents=True,exist_ok=False)
    if a.check_only and a.record_episode:p.error('--check-only does not record')
    plan=json.loads(a.plan.read_text());scene=json.loads(a.scene.read_text())
    require_current_plan(plan)
    require_pickup_clearance(plan)
    if file_hash(a.robot_model)!=plan['model_sha256']:raise ValueError('Cached model differs from live robot model')
    if not plan.get('validation',{}).get('passed'):raise ValueError('Complete trajectory validation missing')
    if (time.time_ns()-scene['capture_wall_ns'])/1e9>120:raise ValueError('Fresh scene required before complete replay')
    # Replaying a fixed trajectory needs a much tighter fixture gate than a
    # session whose geometry is replanned each round.
    scene=check_replay_fixture(scene,plan['scene'])
    scene=admit_scene(plan['scene'],scene)
    token=validated_token(require_token=not a.check_only);rclpy.init();node=WaypointNode(a.robot_model,tracking_abort=False);telemetry=None;recording=False;monitor=None
    report=dict(success=False,physical_motion=False,trajectory=str(a.plan.resolve()),trajectory_sha256=file_hash(a.plan),
                started_wall_ns=time.time_ns(),workflow='COMPLETE_TIMELINE_REPLAY')
    try:
        offset=np.asarray(plan['local_geometry_translation_m'])
        # Revalidate against the fresh detection before recording or sending
        # goals. Cached geometry is never silently adapted during replay.
        binding=dict(trajectory_sha256=file_hash(a.plan),scene_sha256=file_hash(a.scene),model_sha256=file_hash(a.robot_model))
        if a.validation_proof and not a.check_only:
            proof=json.loads(a.validation_proof.read_text())
            report['validation']=validate_proof(proof,binding,time.time_ns())
            from .continuous_plan import read_limits
            limits,grip,_=read_limits(a.robot_model)
            if plan['limits']!=limits or plan['gripper_limits']!=grip:raise ValueError('Motion limits changed after preflight')
        else:
            report['validation']=validate_timeline(node,plan,scene,offset,a.robot_model)
        report['file_binding']=binding
        if a.check_only:
            report.update(success=True,check_only=True)
            print('整轮执行前检查通过；未申请运动权限、未发送运动目标。',flush=True)
            return
        _,obj=node.precise_scene(scene,offset)
        close_time=next(e['time_s'] for e in plan['events'] if e['name']=='close')
        grasp=min(plan['samples'],key=lambda s:abs(s['time_s']-close_time))['positions'][:7]
        attached=node.attached_at(obj,grasp)
        pickup=node.chain.pose(dict(zip(NAMES,grasp)))
        b=scene['measured']['bin']
        placed=box(obj.id,[*(np.asarray(b['center_xy'])+offset[:2]),b['support_z']+offset[2]+.030],[.04]*3,b['yaw'])
        def transition(mark):
            report['last_timeline_event']=mark['name']
            if mark['name']=='close':
                monitor.transition('grasp')
            elif mark['name']=='carry':
                scene_attachment(node,attached);monitor.transition('carry',attached,pickup)
            elif mark['name']=='return':
                scene_attachment(node,attached,remove=True,world=placed);monitor.transition('return')
        goals=build_goals(plan)
        telemetry=Telemetry(a.output/'raw_can.jsonl');node.bind_telemetry(telemetry)
        monitor=LiveCollisionMonitor(node,a.robot_model);monitor.transition('reach')
        def begin():
            nonlocal recording
            rpc('begin',path=str(a.record_episode.resolve()));recording=True
        with inject_action_capabilities(token):
            report['execution']=execute(node,plan,goals,bool(a.record_episode),on_event=transition,
                                        before_dispatch=begin if a.record_episode else None,collision_monitor=monitor,
                                        on_dispatch=lambda:report.update(physical_motion=True))
        report['returned_mechanical_zero']=True
        if recording:
            report['recording_quality']=rpc('finish',success=True)['quality'];recording=False
        # Post-task verification is deliberately outside the demonstration.
        capture(a.output/'after_release')
        report['placement_verification']=verify_placed(scene,a.output/'after_release',offset)
        report['success']=report['placement_verification']['verified'] and report.get('recording_quality',{'passed':True})['passed']
        b=scene['measured']['bin']
        node.update_object(box('vision_baseline_cube',[*(np.asarray(b['center_xy'])+offset[:2]),b['support_z']+offset[2]+.030],[.04]*3,b['yaw']))
        if not report['success']:raise RuntimeError('Placement / recording verification failed')
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        if recording:
            try:report['recording_quality']=rpc('finish',success=False)['quality']
            except Exception as exc:report['recording_error']=str(exc)
        if a.record_episode and (a.record_episode/'metadata.json').exists():
            metadata=json.loads((a.record_episode/'metadata.json').read_text())
            metadata.update(automatic_task_contract_passed=report['success'],
                            gripper_policy=COLLECTION_POLICY,
                            terminal_label='returned_zero' if report['success'] else 'failed_or_interrupted',
                            complete_trajectory_sha256=report['trajectory_sha256'],complete_trajectory_path=report['trajectory'],
                            planning_during_recording=False,
                            grasp_target_offset_world_m=plan.get('grasp_target_offset_world_m',[0.,0.,0.]),
                            tracking_error_policy='diagnostic_only_with_measured_collision_checks')
            write_json(a.record_episode/'metadata.json',metadata)
        report.update(finished_wall_ns=time.time_ns(),max_live_tcp_tracking_error_m=node.max_tracking_error_m)
        if hasattr(node,'controller_readiness_diagnostics'):
            report['controller_readiness']=node.controller_readiness_diagnostics
        if monitor is not None:
            report['live_collision_monitor']=monitor.report()
            monitor.close()
        write_json(a.output/'result.json',report)
        if telemetry is not None:telemetry.close()
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
