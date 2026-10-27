"""Move an already-open gripper to the collection's side observation point."""
import argparse
import json
from pathlib import Path
import time

from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation

from .grasp_test import TestNode,capture
from .place_test import verify_placed
from .hover import box
from scripts.control_ownership.h2 import validated_token,inject_action_capabilities


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grasp-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(remove_ros_args()[1:]);a.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads(a.grasp_result.read_text());scene=json.loads(Path(previous['scene']).read_text())
    token=validated_token();rclpy.init();node=TestNode();report=dict(workflow='POST_RELEASE_CLEAR_VIEW',success=False,stages=[],started_wall_ns=time.time_ns())
    try:
        query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        if node.call(node.scene_client,query).scene.robot_state.attached_collision_objects:
            raise RuntimeError('An object is still registered as attached; use the held-object workflow')
        with inject_action_capabilities(token):
            target=np.eye(4);target[:3,:3]=Rotation.from_euler('xyz',[0.,.7,0.]).as_matrix();target[:3,3]=[.28,-.20,.43]
            node.move('observation_clear',target,report,a.output)
            capture(a.output/'after_release')
            report['placement_verification']=verify_placed(scene,a.output/'after_release',np.asarray(previous['local_geometry_translation_m']))
            report['success']=report['placement_verification']['verified']
            if report['success']:
                b=scene['measured']['bin'];offset=np.asarray(previous['local_geometry_translation_m'])
                frames=report['placement_verification']['frames']
                xyz=np.median([x['center_world'] for x in frames],axis=0)
                xyz[2]=b['support_z']+offset[2]+.030
                node.update_object(box('vision_baseline_cube',xyz,[.04]*3,b['yaw']))
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        report['finished_wall_ns']=time.time_ns();(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        node.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in report.items() if k!='stages'},indent=2))


if __name__=='__main__':main()
