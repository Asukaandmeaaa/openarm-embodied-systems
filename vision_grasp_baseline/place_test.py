"""Continue a verified grasp with one fixed tilted bin-release posture."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from moveit_msgs.msg import AttachedCollisionObject,CollisionObject,PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation

from .grasp_test import TestNode,capture
from .hover import matrix_pose,pose_matrix,box
from scripts.act_trial.isolated_telemetry import Telemetry
from scripts.control_ownership.h2 import inject_action_capabilities,validated_token


def inside_aperture(extent,inner):
    """Containment only: the task does not require centering or wall clearance."""
    extent=np.asarray(extent,float);inner=np.asarray(inner,float)
    if (extent.shape!=(2,) or inner.shape!=(2,) or not np.isfinite([extent,inner]).all()
            or np.any(extent<0) or np.any(inner<=0)):
        return False
    return bool(np.all(extent<=inner/2))


def verify_placed(scene,path,correction):
    import cv2
    metadata=json.loads((path/'capture.json').read_text());i=metadata['color_intrinsics']
    wc=np.array(scene['world_from_color_optical']);normal=np.array(scene['measured']['table']['normal'])
    d=scene['measured']['table']['plane_d'];bin_=scene['measured']['bin']
    yaw=bin_['yaw'];r=Rotation.from_euler('z',yaw).as_matrix()[:2,:2]
    center=np.array(bin_['center_xy']);inner=np.asarray(bin_['outer_size_xy'])-.020
    observations=[]
    for frame in metadata['frames']:
        idx=frame['index'];im=cv2.imread(str(path/f'color_{idx:03d}.png'))
        z=np.load(path/f'depth_{idx:03d}.npy')*metadata['depth_scale_m'];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(90,150,65),(130,255,255))>0
        v,u=np.where(mask&(z>.1)&(z<1.5))
        points=np.column_stack(((u-i['ppx'])*z[v,u]/i['fx'],(v-i['ppy'])*z[v,u]/i['fy'],z[v,u]))
        world=points @ wc[:3,:3].T+wc[:3,3]
        if len(world)<50:continue
        local=(world[:,:2]-center) @ r
        # Count all substantial blue pixels, so an outside cube is not hidden
        # by selecting only a small blue patch inside the target footprint.
        extent=np.percentile(abs(local),98,axis=0)
        height=float(np.percentile(world@normal+d,85))
        inside=inside_aperture(extent,inner)
        observations.append(dict(inside=inside,extent_xy=extent.tolist(),top_height_m=height,
                                 aperture_clearance_xy_m=(inner/2-extent).tolist(),
                                 center_world=(np.median(world,axis=0)+correction).tolist()))
    valid=[x['inside'] and .025<x['top_height_m']<.075 for x in observations]
    return dict(verified=len(valid)>=6 and sum(valid)>=6,frames=observations,
                acceptance_policy='inside_bin_v2',required_extra_wall_clearance_m=0.,
                inner_aperture_basis='Measured outer footprint minus nominal 10mm walls per side',
                inner_dimensions_xy_m=inner.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grasp-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(remove_ros_args()[1:])
    previous=json.loads(a.grasp_result.read_text())
    if not previous['success'] or previous['state']!='HOLDING_AFTER_LIFT':
        raise RuntimeError('A verified held grasp is required')
    scene=json.loads(Path(previous['scene']).read_text());correction=np.asarray(previous['local_geometry_translation_m'])
    a.output.mkdir(parents=True,exist_ok=False)
    token=validated_token();rclpy.init();node=TestNode();telemetry=Telemetry(a.output/'raw_can.jsonl')
    report=dict(workflow='FIXED_TILTED_BIN_RELEASE',success=False,stages=[],started_wall_ns=time.time_ns(),
                grasp_result=str(a.grasp_result),home_commanded=False,bulk_collection=False)
    try:
        node.bind_telemetry(telemetry)
        with inject_action_capabilities(token):
            query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            held=node.call(node.scene_client,query).scene.robot_state.attached_collision_objects
            ids=[x.object.id for x in held]
            if 'vision_baseline_cube' not in ids:raise RuntimeError('Held-object scene state missing; refusing blind release continuation')
            attached=next(x for x in held if x.object.id=='vision_baseline_cube')
            if attached.object.header.frame_id!=attached.link_name:
                raise RuntimeError('Attached-object geometry is not expressed in its attachment frame')
            current=node.refresh_rigid_state(10.);ee_tcp=node.tcp_transform(current)
            q=attached.object.pose.orientation
            root_pose=np.eye(4) if q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w<.5 else pose_matrix(attached.object.pose)
            tcp_object=np.linalg.inv(ee_tcp) @ root_pose @ pose_matrix(attached.object.primitive_poses[0])
            release_r=Rotation.from_euler('xyz',[0.,.7,0.]).as_matrix()
            object_r=release_r @ tcp_object[:3,:3]
            half_extents=.020*np.sum(abs(object_r),axis=1)
            bin_=scene['measured']['bin'];bin_yaw=bin_['yaw']
            rb=Rotation.from_euler('z',bin_yaw).as_matrix()
            in_bin_half=.020*np.sum(abs(rb.T @ object_r),axis=1)
            inner=np.asarray(bin_['outer_size_xy'])-.020
            if np.any(in_bin_half[:2]+.010>inner/2):raise RuntimeError('Rotated cube does not fit the nominal measured bin aperture')
            desired_object=np.r_[np.asarray(bin_['center_xy'])+correction[:2],
                                bin_['support_z']+correction[2]+bin_['rim_height']+half_extents[2]+.020]
            target=np.eye(4);target[:3,:3]=release_r;target[:3,3]=desired_object-release_r@tcp_object[:3,3]
            report['release_tcp_world']=target.tolist();report['release_object_center_world']=desired_object.tolist()
            upper=target.copy();upper[2,3]+=.070
            node.move('preplace',upper,report,a.output,attached=attached)
            node.move('vertical_lower',target,report,a.output,vertical=True,attached=attached)
            report['gripper_release']=node.gripper(-.35)
            detach=deepcopy(attached);detach.object.operation=CollisionObject.REMOVE
            request=ApplyPlanningScene.Request();request.scene.is_diff=True;request.scene.robot_state.is_diff=True
            request.scene.robot_state.attached_collision_objects=[detach]
            if not node.call(node.apply_client,request).success:raise RuntimeError('Detach scene update failed')
            upper[2,3]+=.030
            node.move('vertical_retreat',upper,report,a.output,vertical=True)
            node.finger_contacts(False)
            # Remain over the bin after vertical retreat. The former arbitrary
            # side-view target had a 12.8cm physical arrival error and is removed.
            # Occluded visual verification is an unverified result, not a reason
            # to command another uncommissioned arm posture.
            report['observation_clear_removed']=True
            capture(a.output/'after_release')
            report['placement_verification']=verify_placed(scene,a.output/'after_release',correction)
            report['success']=report['placement_verification']['verified']
            report['state']='PLACED_AND_RETREATED' if report['success'] else 'PLACEMENT_NOT_VERIFIED'
            if report['success']:
                frames=report['placement_verification']['frames']
                xyz=np.median([x['center_world'] for x in frames],axis=0)
                # Visible blue points lie near the upper surface; use the known
                # 40mm fixture height for the resting collision-box center.
                xyz[2]=bin_['support_z']+correction[2]+.010+.020
                node.update_object(box('vision_baseline_cube',xyz,[.04]*3,bin_yaw))
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        report['finished_wall_ns']=time.time_ns();(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        telemetry.close();node.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in report.items() if k!='stages'},indent=2))


if __name__=='__main__':main()
