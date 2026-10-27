"""Manual, two-stage ROS return to HOME. No enable, disable, zero or left command.

Plan-only never sends an action. Execution revalidates the exact saved plan,
current scene and full measured state under the existing H2 gateway lease.
"""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.act_trial.preflight import DEFAULT_BUNDLE
from scripts.act_trial.tracking_contract import stationary_recovery_contract
from scripts.control_ownership.h2 import inject_action_capabilities, validated_token
from scripts.home_replay.core import NAMES, digest, write_json


def check_snapshot(state, bundle, previous=None):
    q = np.asarray(state['positions_rad'], dtype=float)
    cmd = np.asarray(state['command_positions_rad'], dtype=float)
    v = np.asarray(state['velocity_rad_s'], dtype=float)
    if any(a.shape != (16,) or not np.all(np.isfinite(a)) for a in (q,cmd,v)):
        raise ValueError('invalid full CAN feedback/command state')
    if state['motor_status'] != [1]*16 or any(not 0 <= state[k] <= .1 for k in ('max_age_s','command_max_age_s')):
        raise ValueError('CAN feedback/commands stale or motor fault')
    if np.any(np.abs(v) > .05):
        raise ValueError('robot not stationary; return HOME refused')
    tolerance = np.array(stationary_recovery_contract(bundle)['limits_rad'])
    if np.any(np.abs(q[:8]-cmd[:8]) > tolerance):
        raise ValueError('existing right command/feedback error exceeds stationary recovery allowance')
    home_grip = bundle['home_config']['home_gripper']['open_position_rad']
    if max(abs(q[7]-home_grip),abs(cmd[7]-home_grip)) > .03:
        raise ValueError('this recovery requires the right gripper already open; it never opens a grasp')
    if previous is not None and np.max(np.abs(q-np.asarray(previous))) > .01:
        raise ValueError('full robot start changed by more than ROS planning admission 0.01 rad; replan')
    return q


def validate_path(trajectory, start16, reference16, goal7, limits, velocity_limits, scene, acceleration_limits=None):
    """Sample the JTC linear/cubic/quintic polynomial, not only its waypoints.

    Keep all other measured axes fixed as collision context. Discrete geometry
    is not a certified continuous collision or dynamics proof.
    """
    from scipy.interpolate import BPoly
    from openarm_kinesthetic_recorder.execution_guard import audit_joint_trajectory
    points = trajectory.joint_trajectory.points
    audit = audit_joint_trajectory(trajectory.joint_trajectory,NAMES[:7],start16[:7],goal7,.01)
    if points[0].time_from_start.sec or points[0].time_from_start.nanosec:
        raise ValueError('recovery requires a zero-time first point')
    if trajectory.multi_dof_joint_trajectory.points:
        raise ValueError('unexpected multi-DOF recovery trajectory')
    count = 0
    def check(q):
        nonlocal count
        count += 1
        if np.any(q[:8]<limits[:8,0]) or np.any(q[:8]>limits[:8,1]):
            raise ValueError('return HOME exceeds canonical right joint limits')
        if scene.put(q):
            raise ValueError('return HOME collision')
    start16=np.asarray(start16,dtype=float); reference16=np.asarray(reference16,dtype=float)
    # Include the holding reference-to-measured transition at controller entry.
    for u in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(start16[:8]-reference16[:8]))/.002))+1)):
        q=start16.copy(); q[:8]=reference16[:8]+u*(start16[:8]-reference16[:8]); check(q)
    previous=start16.copy()
    vmax=np.minimum(np.asarray(velocity_limits)[:7]*.10,.25)
    amax=np.full(7,.05) if acceleration_limits is None else np.minimum(.05,acceleration_limits)
    if amax.shape!=(7,) or not np.all(np.isfinite(amax)) or np.any(amax<=0):
        raise ValueError('invalid ROS recovery acceleration limits')
    for left,right in zip(points,points[1:]):
        t0=left.time_from_start.sec+left.time_from_start.nanosec/1e9
        t1=right.time_from_start.sec+right.time_from_start.nanosec/1e9
        derivatives=[]
        for p in (left,right):
            row=[p.positions]
            if p.velocities:
                row.append(p.velocities)
                if p.accelerations: row.append(p.accelerations)
            elif p.accelerations:
                raise ValueError('acceleration without velocity is unsupported')
            derivatives.append(row)
        if len(derivatives[0])!=len(derivatives[1]):
            raise ValueError('inconsistent trajectory derivatives')
        poly=BPoly.from_derivatives([t0,t1],derivatives)
        if t1-t0>60 or count>100000:
            raise ValueError('recovery validation budget exceeded')
        for t in np.linspace(t0,t1,max(2,int(np.ceil((t1-t0)/.005))+1)):
            q=start16.copy();q[:7]=poly(t)
            if not np.all(np.isfinite(q)) or np.max(np.abs(q[:7]-previous[:7]))>.002+1e-9:
                raise ValueError('return HOME interpolation requires finer validation or start realignment')
            if np.any(np.abs(poly.derivative()(t))>vmax+1e-6):
                raise ValueError('return HOME exceeds reviewed ROS recovery speed')
            # Existing ROS return uses 0.10 of the 0.5 rad/s² planning cap.
            if np.any(np.abs(poly.derivative(2)(t))>amax+1e-6):
                actual = np.abs(poly.derivative(2)(t))
                raise ValueError('return HOME exceeds ROS recovery acceleration: '
                                 +json.dumps({'time_s':float(t), 'abs_rad_s2':actual.tolist(),
                                              'limits_rad_s2':amax.tolist()}))
            check(q); previous=q
    return {'trajectory':audit.as_dict(),'collision_samples':count,
            'interpolation':'JTC position/velocity/acceleration polynomial',
            'left_commanded':False,'gripper_commanded':False,'continuous_collision_proof':False}


def run(args):
    # All ROS/hardware imports are deferred; importing this module is offline.
    import rclpy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.set_message import set_message_fields
    from moveit_msgs.msg import RobotTrajectory
    from openarm_kinesthetic_recorder.home_guarded_executor import (
        HomeInitializationNode, load_home_config, verify_static_collision_snapshot,
        wait_for_home_settle)
    from openarm_kinesthetic_recorder.home_measurements import require_current_table_collision
    from scripts.act_trial.telemetry import Telemetry
    from scripts.home_replay.simulate import Scene
    from scripts.home_replay.validation_contract import ros_limits
    from scripts.act_trial.ros_bridge import Inputs
    from scripts.act_trial.speed_contract import ROS_PLANNING_CONFIG
    import yaml
    token=validated_token(require_token=args.execute)
    bundle=json.loads(DEFAULT_BUNDLE.read_text())
    config=load_home_config(Path(os.environ['H2_CAPTURE_CONFIG']))
    if config!=bundle['home_config']: raise ValueError('HOME config differs from sealed ROS capture')
    require_current_table_collision(config)
    limits,velocities,_,_=ros_limits(bundle,None,None,required=True)
    planning_limits=yaml.safe_load(ROS_PLANNING_CONFIG.read_text())['joint_limits']
    if any(planning_limits[n].get('has_acceleration_limits') is not True for n in NAMES[:7]):
        raise ValueError('reviewed ROS acceleration limits required')
    accelerations=np.array([float(planning_limits[n]['max_acceleration'])*.10 for n in NAMES[:7]])
    goal=config['right_home_joint_positions_rad']; collision=config['static_collision']
    timeout=float(config['planning']['service_timeout_s'])
    # Inputs runs its own ROS executor; it audits the validated live backend.
    inputs=node=telemetry=None
    record={'success':False,'execution_requested':args.execute,'physical_motion_requested':False,
            'bundle_sha256':digest(DEFAULT_BUNDLE),'start_wall_ns':time.time_ns(),
            'planning_limits_sha256':digest(ROS_PLANNING_CONFIG)}
    report=args.directory/('execution.json' if args.execute else 'plan.json')
    def fresh():
        deadline=time.monotonic()+10
        while True:
            try:
                ros=inputs.snapshot(time.monotonic()); can=telemetry.snapshot()
                q=check_snapshot(can,bundle)
                if np.max(np.abs(ros['actual']-q))>.03: raise ValueError('ROS/CAN state disagreement')
                return q,can
            except (ValueError,RuntimeError):
                if time.monotonic()>=deadline:raise
                rclpy.spin_once(node,timeout_sec=.02)
    try:
        inputs=Inputs(start_parameter_services=False)
        node=HomeInitializationNode()
        telemetry=Telemetry(args.directory/('execution_can.jsonl' if args.execute else 'planning_can.jsonl'))
        q,can=fresh()
        # Wait for transient robot_description delivery before the one-time audit.
        deadline=time.monotonic()+10
        while True:
            try: inputs.get('/robot_description');break
            except ValueError:
                if time.monotonic()>deadline:raise
                time.sleep(.02)
        record['backend_audit']=inputs.audit(bundle)
        arm_params=record['backend_audit']['live_controller_parameters']['right_joint_trajectory_controller']
        if arm_params['interpolation_method']!='splines' or arm_params['interpolate_from_desired_state']:
            raise ValueError('recovery requires the validated measured-start spline controller')
        for key,quantization in (('kp',500/4095),('kd',5/4095)):
            if np.max(np.abs(np.array(can['command_'+key])[:8]-record['backend_audit']['expected_right_gains'][key]))>quantization+1e-8:
                raise ValueError('CAN gains differ from validated ROS')
        snapshot=node.guard_snapshot(collision['object_id'],collision['dimensions_xyz_m'],timeout)
        verify_static_collision_snapshot(snapshot,collision)
        node.refresh_rigid_state(timeout)
        if args.execute:
            saved=json.loads((args.directory/'plan.json').read_text())
            if not saved.get('success') or saved.get('bundle_sha256')!=digest(DEFAULT_BUNDLE):
                raise ValueError('successful matching saved plan required')
            check_snapshot(can,bundle,saved['start16'])
            if np.max(np.abs(np.array(can['command_positions_rad'])-saved['reference16']))>.01:
                raise ValueError('holding reference changed since planning; replan')
            trajectory=RobotTrajectory();set_message_fields(trajectory,saved['robot_trajectory'])
            record['plan_sha256']=digest(args.directory/'plan.json')
            start=np.array(saved['start16'])
        else:
            start=q.copy()
            response=node.plan({'joint_names':NAMES[:7],'start_positions_rad':start[:7].tolist(),
                                'move_group':'right_arm'},goal,timeout,.10,.10)
            if response.error_code.val!=1:raise ValueError('ROS return HOME planning failed')
            trajectory=deepcopy(response.trajectory)
            from scripts.act_trial.retime_home import slow_to_limits
            # Save the rejected/original candidate too, even if retiming fails.
            record['original_robot_trajectory'] = message_to_ordereddict(trajectory)
            trajectory, record['retiming'] = slow_to_limits(trajectory, velocities, accelerations)
        scene=Scene(bundle)
        audit=validate_path(trajectory,start,np.array(can['command_positions_rad']),goal,limits,velocities,scene,accelerations)
        record.update(start16=start.tolist(),reference16=can['command_positions_rad'],
                      robot_trajectory=message_to_ordereddict(trajectory),validation=audit)
        if not args.execute:
            record['success']=True
            print('HOME_PLAN_CHECKED: no action sent; inspect plan.json before return-home.',flush=True)
            return
        # Re-read after planning/model work; no movement or changed scene allowed.
        current,can=fresh();check_snapshot(can,bundle,start)
        if np.max(np.abs(np.array(can['command_positions_rad'])-record['reference16']))>.01:
            raise ValueError('holding reference changed before execution')
        verify_static_collision_snapshot(node.guard_snapshot(collision['object_id'],collision['dimensions_xyz_m'],timeout),collision)
        record['physical_motion_requested']=True
        write_json(report,record)
        print('ROS_RETURN_HOME_EXECUTING: right J1-J7 only; existing ROS torque remains enabled.',flush=True)
        with inject_action_capabilities(token):
            record['execution']=node.execute_same_trajectory(trajectory,audit['trajectory']['duration_s'],30.)
        if record['execution']['action_status']!=4 or record['execution']['moveit_error_code']!=1:
            raise ValueError('ROS return HOME execution failed')
        record['settling']=wait_for_home_settle(node,goal,config)
        if not record['settling']['accepted']:raise ValueError('ROS HOME settling failed')
        q,can=fresh()
        from scripts.act_trial.manual_experiment import check_pose
        check_pose(can,'home',bundle)
        record['success']=True
        print('ROS_RETURN_HOME_ACCEPTED: arm/gripper HOME verified; no ACT started.',flush=True)
    except BaseException as exc:
        record['error']=str(exc);raise
    finally:
        write_json(report,record)
        if telemetry is not None: telemetry.close()
        if node is not None: node.destroy_node()
        if inputs is not None: inputs.close()


def main():
    from rclpy.utilities import remove_ros_args
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(remove_ros_args()[1:])
    args.directory=args.directory.resolve()
    if not args.directory.is_dir():raise ValueError('prepare a recovery directory first')
    target=args.directory/('execution.json' if args.execute else 'plan.json')
    # Reserve the artifact before constructing ROS nodes; never retry in place.
    with target.open('x') as f:f.write('{}\n')
    run(args)


if __name__=='__main__':main()
