"""Review or manually launch the validated ROS backend with enforced ownership.

Default is render-only: no ROS node, socket, URDF generation or hardware I/O.
--physical deliberately runs the normal H1 activation/return_to_zero workflow.
It is for the supported mechanical-zero procedure, never arbitrary-pose recovery.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.control_ownership.security import environment
from scripts.control_ownership.host_setup import require_can_registry

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / '.control_ownership_overlay/install'
HARDWARE_PREFIX = OVERLAY / 'openarm_hardware'
CONTROLLERS = ('left_joint_trajectory_controller', 'right_joint_trajectory_controller',
               'left_gripper_controller', 'right_gripper_controller')
VALIDATED_ARGUMENTS = {
    'arm_type': 'openarm_v2.0', 'use_fake_hardware': 'false',
    'launch_rviz': 'false', 'robot_controller': 'joint_trajectory_controller',
    'right_can_interface': 'can0', 'left_can_interface': 'can1',
    'gravity_compensation': 'true', 'gravity_scale': '1.0',
    'gripper_contact_mode': 'monitor', 'gripper_torque_threshold': '0.0',
    'gripper_torque_release_threshold': '0.0', 'gripper_hard_torque_limit': '0.0',
}


def moveit_remaps():
    result = [(f'/{name}/follow_joint_trajectory',
               f'/authorized/{name}/follow_joint_trajectory') for name in CONTROLLERS]
    # The configured, unused left GripperCommand handle must not retain raw access.
    result += [('/left_gripper_controller/gripper_cmd',
                '/authorized/left_gripper_controller/gripper_cmd')]
    return result


def bootstrap_arguments(arguments):
    """Right-arm commissioning: keep left hardware hold, do not load left J8 JTC."""
    values = list(arguments)
    if 'left_gripper_controller' in values and 'right_gripper_controller' in values:
        values.remove('left_gripper_controller')
    return values


def verify_overlay():
    from ament_index_python.packages import get_package_prefix
    prefix = Path(get_package_prefix('openarm_hardware')).resolve()
    if prefix != HARDWARE_PREFIX.resolve():
        raise ValueError('patched hardware overlay not sourced last: ' + str(prefix))
    library = prefix / 'lib/libopenarm_hardware.so'
    symbols = subprocess.run(['nm', '-D', '--defined-only', str(library)],
                             check=True, text=True, capture_output=True).stdout
    if ' openarm_can_ownership_protocol_version' not in symbols:
        raise ValueError('hardware library lacks CAN ownership protocol marker')
    return str(library)


def conflicting_processes(proc=Path('/proc')):
    conflicts = []
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            args = item.joinpath('cmdline').read_bytes().split(b'\0')
            # Match executable basenames rather than this script's command text.
            bases = {Path(a.decode(errors='replace')).name for a in args if a}
            if bases & {'ros2_control_node', 'dora-openarm', 'dora_openarm'}:
                conflicts.append({'pid': int(item.name), 'argv': [a.decode(errors='replace') for a in args if a]})
        except (OSError, PermissionError):
            continue
    return conflicts


def existing_can_receivers(proc_can=Path('/proc/net/can')):
    # A receiver could be a writer. Refuse conservatively; start the recorder
    # after H1 with this entrypoint. Do not signal or modify existing processes.
    require_can_registry(proc_can)
    registrations = []
    for path in sorted(proc_can.glob('rcvlist_*')):
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ('can0', 'can1'):
                registrations.append({'file': path.name, 'line': line.strip()})
    return registrations


def verify_empty_can_tx_queues():
    """Read qdisc statistics only; never reset a link or transmit a CAN frame.

    A stopped producer can leave old commands queued. Interface UP/ERROR-ACTIVE
    is not proof of a working transmit path. Do not enable against that backlog.
    Cumulative drops/requeues are diagnostic history, not current queue occupancy.
    """
    for interface in ('can0', 'can1'):
        try:
            result = subprocess.run(['tc', '-j', '-s', 'qdisc', 'show', 'dev', interface],
                                    check=True, text=True, capture_output=True, timeout=5.)
            queues = json.loads(result.stdout)
            if not isinstance(queues, list) or not queues:
                raise ValueError('no qdisc statistics')
            for queue in queues:
                if not isinstance(queue, dict) or any(
                        type(queue.get(key)) is not int or queue[key] < 0
                        for key in ('backlog', 'qlen')):
                    raise ValueError('missing/invalid backlog or qlen statistics')
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ValueError(f'cannot verify CAN TX queue for {interface}; refusing startup: {exc}') from exc
        if any(queue['backlog'] or queue['qlen'] for queue in queues):
            raise ValueError(f'pending CAN TX frames on {interface}; refusing startup: '
                             + json.dumps(queues)
                             + '; support both arms and isolate motor power before manual recovery; '
                               'no automatic queue flush or interface reset')


def preflight(keystore):
    security = environment(keystore)
    library = verify_overlay()
    conflicts = conflicting_processes()
    receivers = existing_can_receivers()
    if conflicts or receivers:
        raise ValueError('existing control process/CAN receiver; no automatic shutdown: '
                         + json.dumps({'processes': conflicts, 'can_receivers': receivers}))
    lockdir = Path('/run/lock/openarm-can')
    if not lockdir.is_dir() or not os.access(lockdir, os.W_OK | os.X_OK):
        raise ValueError('CAN lock directory missing/unwritable; run ./scripts/actctl prepare-host')
    verify_empty_can_tx_queues()
    return security, library


def verify_motor_responses(report_path=None):
    """Physical-entry-only bounded parameter reads, before ROS can enable motors.

    This is deliberately absent from --check-only. Queries never enable, disable,
    move, or write motor parameters. A responding device is not proof of homing.
    """
    from scripts.probe_openarm_can import probe
    report = {'motor_enable_requested': False, 'homing_verified': False, 'interfaces': {}}
    try:
        for interface in ('can0', 'can1'):
            result = probe(interface, list(range(1, 9)))
            report['interfaces'][interface] = result
            if not result['ok']:
                raise ValueError(f'{interface}: motor ID/mode query failed before enable; '
                                 + str(result.get('error', 'missing or unexpected replies')))
        # Replies do not license activation if pending frames remain.
        verify_empty_can_tx_queues()
        report['passed'] = True
    except Exception as exc:
        report.update(passed=False, error=str(exc))
        raise
    finally:
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'motor_communication_preflight': report}), flush=True)


def create_launch_description(security, *, software_test_directory=None, software_test_offset=0.,
                              robot_description_transform=None, node_transform=None):
    """Reuse the installed launch implementation; change identities and routes only."""
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import SetLaunchConfiguration
    from launch_ros.actions import Node
    source = Path(get_package_share_directory('openarm_bimanual_moveit_config')) / 'launch/demo.launch.py'
    spec = importlib.util.spec_from_file_location('openarm_validated_secure_demo', source)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    launch_arguments = dict(VALIDATED_ARGUMENTS)
    if software_test_directory is not None:
        # Test-only API: force fake hardware AND isolate the upstream launch's
        # fixed gravity URDF filename from every physical experiment.
        from types import SimpleNamespace
        test_directory = Path(software_test_directory).resolve()
        test_directory.mkdir(parents=True, exist_ok=False)
        demo.tempfile = SimpleNamespace(gettempdir=lambda: str(test_directory))
        launch_arguments['use_fake_hardware'] = 'true'
        import xml.etree.ElementTree as ET
        if not 0. <= float(software_test_offset) <= .02:
            raise ValueError('mock following offset must be in [0,.02]')
        original_description = demo.generate_robot_description
        def checked_mock_description(*args, **kwargs):
            tree = ET.fromstring(original_description(*args, **kwargs))
            controls = tree.findall('ros2_control')
            if len(controls) != 2:
                raise ValueError('test model requires exactly two GenericSystem controls')
            for control in controls:
                hardware = control.find('hardware')
                if hardware.findtext('plugin') != 'mock_components/GenericSystem':
                    raise ValueError('software test refuses non-GenericSystem hardware BEFORE node launch')
                if software_test_offset:
                    ET.SubElement(hardware, 'param', name='position_state_following_offset').text = str(software_test_offset)
                    # Keep the mock right finger away from its0rad upper limit
                    # so the artificial positive offset is not an activation fault.
                    finger = control.find("joint[@name='openarm_right_finger_joint1']/state_interface[@name='position']/param[@name='initial_value']")
                    if finger is not None:
                        finger.text = '-0.35'
            xml = ET.tostring(tree, encoding='unicode')
            (test_directory / 'openarm_bimanual.urdf').write_text(xml)
            return xml
        demo.generate_robot_description = checked_mock_description

    if robot_description_transform is not None:
        untransformed_description = demo.generate_robot_description
        def transformed_description(*args, **kwargs):
            return robot_description_transform(untransformed_description(*args, **kwargs))
        demo.generate_robot_description = transformed_description

    def secured_node(*args, **kwargs):
        if node_transform is not None:
            kwargs = node_transform(dict(kwargs))
        package = kwargs.get('package')
        executable = kwargs.get('executable')
        if package == 'controller_manager' and executable == 'spawner':
            enclave = 'bootstrap'
            kwargs['arguments'] = bootstrap_arguments(kwargs.get('arguments', []))
        elif package == 'moveit_ros_move_group':
            enclave = 'moveit'
            kwargs['remappings'] = list(kwargs.get('remappings', [])) + moveit_remaps()
        elif package in ('controller_manager', 'robot_state_publisher'):
            enclave = 'backend'
        elif package == 'rviz2':
            enclave = 'observer'  # launch_rviz=false; no control permission if changed.
        else:
            raise ValueError('unreviewed launch node: ' + str((package, executable)))
        kwargs['ros_arguments'] = list(kwargs.get('ros_arguments', [])) + ['--enclave', '/openarm/' + enclave]
        kwargs['additional_env'] = {**kwargs.get('additional_env', {}), **security,
                                   'ROS_SECURITY_ENCLAVE_OVERRIDE': '/openarm/' + enclave}
        return Node(*args, **kwargs)

    demo.Node = secured_node
    return LaunchDescription([
        *[SetLaunchConfiguration(key, value) for key, value in launch_arguments.items()],
        *demo.generate_launch_description().entities,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keystore', required=True)
    parser.add_argument('--physical', action='store_true',
                        help='MANUAL H1: enables motors and performs validated return_to_zero')
    parser.add_argument('--render-only', action='store_true', help='explicit default: review only')
    parser.add_argument('--check-only', action='store_true',
                        help='read-only security/overlay/process/CAN/lock preflight; starts nothing')
    parser.add_argument('--startup-report', type=Path,
                        help='new report file for physical-entry motor communication checks')
    args = parser.parse_args()
    if sum((args.physical, args.render_only, args.check_only)) > 1:
        parser.error('--physical, --render-only and --check-only are mutually exclusive')
    security = environment(args.keystore)
    review = {'physical_requested': args.physical, 'launch_arguments': VALIDATED_ARGUMENTS,
              'security_environment': security, 'moveit_remappings': moveit_remaps(),
              'hardware_overlay': str(HARDWARE_PREFIX),
              'trusted_bootstrap': 'controller spawners only; must finish before lease acquisition',
              'left_gripper_controller': 'not loaded; ROS hardware preserves its initial holding reference',
              'shutdown': 'normal ROS shutdown disables motors; support both arms first'}
    print(json.dumps(review, indent=2), flush=True)
    if args.check_only:
        try:
            _, library = preflight(args.keystore)
        except ValueError as exc:
            print(json.dumps({'preflight_passed': False, 'reason': str(exc), 'hardware_operations': False}))
            raise SystemExit(2)
        print(json.dumps({'preflight_passed': True, 'hardware_library': library, 'hardware_operations': False}))
        return
    if not args.physical:
        return
    if args.startup_report is not None:
        if args.startup_report.exists() or not args.startup_report.parent.is_dir():
            parser.error('--startup-report must be a new file in an existing directory')
    security, _ = preflight(args.keystore)
    verify_motor_responses(args.startup_report)
    os.environ.update(security)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('ROS_SECURITY_ENCLAVE_OVERRIDE', None)
    from launch import LaunchService
    description = create_launch_description(security)
    service = LaunchService()
    service.include_launch_description(description)
    print('MANUAL_PHYSICAL_H1_START: ROS activation/return_to_zero will begin.', flush=True)
    sys.exit(service.run())


if __name__ == '__main__':
    main()
