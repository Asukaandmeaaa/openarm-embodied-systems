"""Isolated measured-pose hold startup; a separate H2 command plans zero motion."""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.control_ownership import secure_launch
from scripts.control_ownership.security import environment

BASE = Path(__file__).resolve().parent
PREFIX = BASE / '.recovery_overlay/install/openarm_hardware'


def startup_joint4_model(xml,lower):
    if lower is None:return xml
    if not -.08<=lower<=-.02:raise ValueError('Startup J4 lower must be in [-.08,-.02]')
    root=ET.fromstring(xml)
    limit=root.find("joint[@name='openarm_right_joint4']/limit")
    if limit is None:raise ValueError('Right elbow limit missing')
    limit.set('lower',str(lower))
    return ET.tostring(root,encoding='unicode')


def linear_planner_node(kwargs,*,joint4_lower=None,position_velocity=True):
    """Expose Pilz LIN only in this independent baseline, using existing services."""
    if (position_velocity and kwargs.get('package')=='controller_manager'
            and kwargs.get('executable')=='spawner'
            and 'right_joint_trajectory_controller' in kwargs.get('arguments',[])):
        kwargs['arguments']=[*kwargs['arguments'],'--param-file',
                            str(BASE/'config/right_arm_position_velocity.yaml')]
    if kwargs.get('package') == 'moveit_ros_move_group':
        if joint4_lower is not None:
            params=[];changed=False
            for item in kwargs.get('parameters',[]):
                if isinstance(item,dict) and 'robot_description' in item:
                    item=dict(item);item['robot_description']=startup_joint4_model(item['robot_description'],joint4_lower);changed=True
                params.append(item)
            if not changed:raise ValueError('Cannot apply same startup limit to MoveIt model')
            kwargs['parameters']=params
        kwargs['parameters'] = [*kwargs.get('parameters', []), {
            'planning_pipelines': ['ompl', 'pilz_industrial_motion_planner'],
            'pilz_industrial_motion_planner.planning_plugins': ['pilz_industrial_motion_planner/CommandPlanner'],
            'robot_description_planning.cartesian_limits.max_trans_vel': .3,
            'robot_description_planning.cartesian_limits.max_trans_acc': .5,
            'robot_description_planning.cartesian_limits.max_trans_dec': -.5,
            'robot_description_planning.cartesian_limits.max_rot_vel': .5,
        }]
    return kwargs


def description_transform(xml, *, mock_positions=None,joint4_lower=None):
    root = ET.fromstring(startup_joint4_model(xml,joint4_lower))
    controls = root.findall('ros2_control')
    if len(controls) != 2:
        raise ValueError('Expected exactly two arm hardware instances')
    for control in controls:
        hardware = control.find('hardware')
        plugin = hardware.findtext('plugin')
        if mock_positions is None:
            if plugin != 'openarm_hardware/OpenArmHW':
                raise ValueError('Unexpected physical hardware plugin: ' + str(plugin))
            ET.SubElement(hardware, 'param', name='startup_hold_current').text = 'true'
        else:
            if plugin != 'mock_components/GenericSystem':
                raise ValueError('Mock test refuses physical hardware')
            for joint in control.findall('joint'):
                name = joint.get('name')
                if name not in mock_positions:
                    raise ValueError('Missing mock initial position: ' + str(name))
                param = joint.find("state_interface[@name='position']/param[@name='initial_value']")
                if param is None:
                    interface = joint.find("state_interface[@name='position']")
                    param = ET.SubElement(interface, 'param', name='initial_value')
                param.text = str(mock_positions[name])
    return ET.tostring(root, encoding='unicode')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keystore', required=True)
    p.add_argument('--physical', action='store_true')
    p.add_argument('--mock-model', type=Path)
    p.add_argument('--positions-report', type=Path)
    p.add_argument('--startup-report', type=Path)
    p.add_argument('--position-only',action='store_true',
                   help='Rollback right-arm controller to original position-only interface')
    p.add_argument('--joint4-start-lower',type=float,default=-.08,
                   help='Startup/zero admission only, [-.08,-.02]rad (default: -.08); task paths retain original -.02 lower')
    a = p.parse_args()
    if a.physical == (a.mock_model is not None):
        p.error('Select exactly one of --physical and --mock-model')
    if a.physical:
        secure_launch.HARDWARE_PREFIX = PREFIX
        security, _ = secure_launch.preflight(a.keystore)
        secure_launch.verify_motor_responses(a.startup_report)
        transform = lambda xml: description_transform(xml,joint4_lower=a.joint4_start_lower)
    else:
        security = environment(a.keystore)
        if not 150 <= int(security['ROS_DOMAIN_ID']) <= 232:
            raise ValueError('Mock domain must be isolated from physical robot')
        snapshot = json.loads(a.positions_report.read_text())['positions_rad']
        positions = {}
        for interface, side in [('can0', 'right'), ('can1', 'left')]:
            for i, q in enumerate(snapshot[interface]):
                name = f'openarm_{side}_joint{i+1}' if i < 7 else f'openarm_{side}_finger_joint1'
                positions[name] = q
        transform = lambda xml: description_transform(xml, mock_positions=positions,joint4_lower=a.joint4_start_lower)
    os.environ.update(security)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('ROS_SECURITY_ENCLAVE_OVERRIDE', None)
    from launch import LaunchService
    service = LaunchService()
    service.include_launch_description(secure_launch.create_launch_description(
        security, software_test_directory=a.mock_model, robot_description_transform=transform,
        node_transform=lambda kwargs:linear_planner_node(kwargs,joint4_lower=a.joint4_start_lower,
                                                       position_velocity=not a.position_only)))
    print('RECOVERY_STARTUP: measured-position hold only; no automatic zero/HOME motion', flush=True)
    print('RIGHT_ARM_COMMAND_INTERFACES: '+('position' if a.position_only else 'position, velocity'),flush=True)
    raise SystemExit(service.run())


if __name__ == '__main__':
    main()
