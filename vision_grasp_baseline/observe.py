"""Read-only live joint state, full right-arm FK and live robot model export."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from moveit_msgs.srv import GetPositionFK
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rosidl_runtime_py.convert import message_to_ordereddict

from .hover import HoverNode,pose_matrix
from scripts.control_ownership.h2 import validated_token


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(remove_ros_args()[1:]);a.output.mkdir(parents=True,exist_ok=False)
    validated_token(require_token=False)
    rclpy.init();n=HoverNode();got={}
    n.create_subscription(String,'/robot_description',lambda m:got.update(urdf=m.data),
                          QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    n.create_subscription(JointState,'/joint_states',lambda m:got.update(joints=m,received=time.monotonic()),10)
    try:
        deadline=time.monotonic()+10.
        while ('urdf' not in got or 'joints' not in got) and time.monotonic()<deadline:
            rclpy.spin_once(n,timeout_sec=.1)
        if 'urdf' not in got or 'joints' not in got:
            raise RuntimeError('Live robot model/joint state unavailable')
        root=ET.fromstring(got['urdf'])
        names=[l.attrib['name'] for l in root.findall('link') if l.attrib['name'].startswith('openarm_right')]
        request=GetPositionFK.Request();request.header.frame_id='world';request.fk_link_names=names
        request.robot_state.joint_state=got['joints'];request.robot_state.is_diff=True
        result=n.call(n.fk_client,request)
        if result.error_code.val!=1 or len(result.pose_stamped)!=len(names):
            raise RuntimeError('Live FK export failed')
        out=dict(wall_ns=time.time_ns(),hardware_motion=False,
                 joint_state=message_to_ordereddict(got['joints']),
                 world_from_link={k:pose_matrix(v.pose).tolist() for k,v in zip(names,result.pose_stamped)})
        (a.output/'robot.urdf').write_text(got['urdf'])
        (a.output/'state.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(output=str(a.output),links=len(names),motion=False)))
    finally:n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
