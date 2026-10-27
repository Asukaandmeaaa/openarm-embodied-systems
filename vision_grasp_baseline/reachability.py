"""Read-only grasp/release IK audit; no trajectory or gripper goals are sent.

Collision checking is deliberately disabled ONLY for these IK diagnostics:
the scene currently contains the conservative solid-bin/raised-table objects
used for hovering. These are not executable plans or collision approvals.
"""
import argparse
import json
import math
from pathlib import Path
import time

from moveit_msgs.srv import GetPositionIK
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation

from .hover import HoverNode,NAMES,EE,TCP,matrix_pose,pose_matrix
from scripts.control_ownership.h2 import validated_token


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(remove_ros_args()[1:])
    if a.output.exists():raise ValueError('Output already exists')
    s=json.loads(a.scene.read_text());c=s['measured']['cube'];b=s['measured']['bin']
    validated_token(require_token=False);rclpy.init();n=HoverNode()
    rows=[]
    try:
        current=n.refresh_rigid_state(10.);ee_tcp=n.tcp_transform(current)
        positions=[('grasp',[*c['center'][:2],c['support_z']+.045],math.pi/2,c['yaw']+math.pi/2),
                   ('release_vertical',[*b['center_xy'],b['release_object_center_z']+.025],math.pi/2,c['yaw']+math.pi/2)]
        for pitch in [.5,.7,.9,1.1]:
            positions.append((f'release_pitch_{pitch}',[*b['center_xy'],b['release_object_center_z']+.025],pitch,0.))
        for name,xyz,pitch,yaw in positions:
            t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',[0.,pitch,yaw]).as_matrix();t[:3,3]=xyz
            row=dict(name=name,target_tcp_world=t.tolist(),collision_checked=False,solutions=[])
            for seed in [current,[0,.5,0,1.2,0,-.4,0],[.2,.2,0,.8,0,-.5,0],[.8,.5,-.8,1.2,-.6,.5,1.4]]:
                req=GetPositionIK.Request();ik=req.ik_request;ik.group_name='right_arm';ik.ik_link_name=EE
                ik.avoid_collisions=False;ik.timeout.nanosec=300000000
                ik.robot_state.joint_state.name=NAMES;ik.robot_state.joint_state.position=seed;ik.robot_state.is_diff=True
                ik.pose_stamped.header.frame_id='world';ik.pose_stamped.pose=matrix_pose(t @ np.linalg.inv(ee_tcp))
                result=n.call(n.ik_client,req,10.)
                if result.error_code.val!=1:continue
                values=dict(zip(result.solution.joint_state.name,result.solution.joint_state.position));q=[values[k] for k in NAMES]
                fk=n.fk_pose({**n.payload(q),'end_effector_link':TCP},q,10.).pose;actual=pose_matrix(fk)
                row['solutions'].append(dict(positions_rad=q,position_error_m=float(np.linalg.norm(actual[:3,3]-t[:3,3])),
                                             rotation_error_rad=float(Rotation.from_matrix(actual[:3,:3] @ t[:3,:3].T).magnitude())))
            rows.append(row)
            print(json.dumps(dict(name=name,solutions=len(row['solutions']),no_motion=True)),flush=True)
        report=dict(wall_ns=time.time_ns(),no_motion=True,scene=str(a.scene),poses=rows,
                    limitation='IK availability only; not collision-checked and not executable.')
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
