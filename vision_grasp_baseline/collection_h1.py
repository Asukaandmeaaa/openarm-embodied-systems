"""Explicit closed-transit collection H1; legacy ACT H1 is unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.control_ownership import secure_launch
from scripts.control_ownership.security import environment
from .gripper_settings import COLLECTION_POLICY,CLOSE_POSITION,TRANSIT_POSITION
from .recovery_launch import linear_planner_node


def description_transform(xml):
    root=ET.fromstring(xml)
    controls=root.findall('ros2_control')
    if len(controls)!=2:raise ValueError('Expected exactly two hardware instances')
    found=[]
    for control in controls:
        hardware=control.find('hardware')
        if hardware is None or hardware.findtext('plugin')!='openarm_hardware/OpenArmHW':
            raise ValueError('Collection H1 requires physical OpenArmHW, not a silent fallback')
        prefix=hardware.findtext("param[@name='arm_prefix']")
        if prefix not in ('right_','left_') or prefix in found:raise ValueError('Unexpected arm prefixes')
        found.append(prefix)
        if prefix=='right_':
            if hardware.findtext("param[@name='ee_type']")!='pinch_gripper':
                raise ValueError('Closed-transit startup requires the right pinch gripper')
            if hardware.find("param[@name='startup_gripper_closed']") is not None:
                raise ValueError('Unexpected preexisting startup override')
            ET.SubElement(hardware,'param',name='startup_gripper_closed').text='true'
    return ET.tostring(root,encoding='unicode')


def require_library_marker(library):
    symbols=subprocess.run(['nm','-D','--defined-only',str(library)],
                           check=True,capture_output=True,text=True).stdout
    if ' openarm_closed_transit_startup_version' not in symbols:
        raise ValueError('Collection H1 library not installed. With robot power OFF and H1/gateway stopped, '
                         'run ./vision_grasp_baseline/run.sh install-h1-fix')


def main(*, node_transform=linear_planner_node):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keystore',required=True)
    parser.add_argument('--physical',action='store_true')
    args=parser.parse_args()
    security=environment(args.keystore)
    print(json.dumps(dict(gripper_policy=COLLECTION_POLICY,right_gripper_target_rad=CLOSE_POSITION,
        pre_recording_empty_gripper_target_rad=TRANSIT_POSITION,
        physical_requested=args.physical,legacy_act_h1_unchanged=True,
        right_arm_command_interfaces=['position','velocity'],
        shutdown='Support both arms before ROS shutdown; motors disable on shutdown'),indent=2),flush=True)
    if not args.physical:return
    security,library=secure_launch.preflight(args.keystore)
    require_library_marker(library)
    report=Path(tempfile.mkdtemp(prefix='collection_h1_closed_',dir=secure_launch.ROOT/'results'))/'motor_startup.json'
    print('COLLECTION_H1_STARTUP_REPORT: '+str(report),flush=True)
    secure_launch.verify_motor_responses(report)
    os.environ.update(security)
    os.environ.pop('ROS_LOCALHOST_ONLY',None);os.environ.pop('ROS_SECURITY_ENCLAVE_OVERRIDE',None)
    from launch import LaunchService
    service=LaunchService()
    service.include_launch_description(secure_launch.create_launch_description(security,
        robot_description_transform=description_transform,node_transform=node_transform))
    print('COLLECTION_H1: startup hold -0.085 rad; zero-test verifies tighter empty closure before recording; left hold unchanged',flush=True)
    raise SystemExit(service.run())


if __name__=='__main__':main()
