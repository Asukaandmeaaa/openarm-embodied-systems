"""Receive-only SocketCAN + ROS capture; start BEFORE H1. Never transmits CAN."""
import argparse
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import socket
import struct
import threading
import time

from core import NAMES, MOTORS, check_driver_contract, decode, digest, preserve_nonfinite, write_json


def mark(directory, name):
    directory = Path(directory)
    if not (directory/'recording.ready').exists():
        raise RuntimeError('capture is not ready/running')
    if (directory/f'mark_{name}.json').exists():
        raise RuntimeError('phase marker already exists; use a new capture for a new H1/H2 run')
    stamp = time.time_ns()
    write_json(directory/f'mark_{name}.json', {'name': name, 'wall_time_ns': stamp,
                                             'monotonic_ns': time.monotonic_ns()})


def record(args):
    import yaml
    check_driver_contract(yaml.safe_load(Path(args.driver_config).read_text()))
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import JointState
    from control_msgs.msg import JointTrajectoryControllerState
    from moveit_msgs.msg import DisplayTrajectory, PlanningScene
    from std_msgs.msg import String
    from rcl_interfaces.msg import ParameterEvent
    from tf2_msgs.msg import TFMessage

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an experiment.
    (out/'snapshots').mkdir()
    manifest = {'schema': 'openarm_h1h2_capture_v1', 'complete': False,
                'started_wall_time_ns': time.time_ns(), 'host': socket.gethostname(),
                'joint_names': NAMES, 'motor_types_per_arm': MOTORS,
                'right_interface': args.right_can, 'left_interface': args.left_can,
                'coordinate_contract': 'openarm_v2_pinch_raw_motor_rad',
                'clock': 'CAN kernel CLOCK_REALTIME ns; ROS source stamps + receiver monotonic/wall ns',
                'capture_role': 'receive_only_no_CAN_send_no_motor_enable', 'files': {}}
    manifest['ros_nonfinite_encoding'] = {'__ros_nonfinite_float__': 'inf, -inf, or nan; metadata preserved, invalid for numeric replay'}
    for label, source in [('home_config', args.home_config), ('driver_config', args.driver_config)]:
        target = out/'snapshots'/f'{label}.yaml'
        shutil.copyfile(source, target)
    for i, source in enumerate(args.snapshot):
        shutil.copyfile(source, out/'snapshots'/f'{i:02d}_{Path(source).name}')
    write_json(out/'manifest.json', manifest)
    stop = threading.Event()
    failures = []
    counts = {'can': 0, 'ros': 0, 'socket_drops': 0, 'can_errors': 0}
    selector = selectors.DefaultSelector()
    sockets = []
    for side, interface in [('right', args.right_can), ('left', args.left_can)]:
        sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        sock.setsockopt(socket.SOL_CAN_RAW, 2, struct.pack('=I', 0x1FFFFFFF))  # error frames
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8*1024*1024)
        sock.setsockopt(socket.SOL_SOCKET, 35, 1)  # SO_TIMESTAMPNS (64-bit Linux)
        sock.setsockopt(socket.SOL_SOCKET, 40, 1)  # SO_RXQ_OVFL
        sock.bind((interface,))
        sock.setblocking(False)
        sockets.append(sock)
        selector.register(sock, selectors.EVENT_READ, (side, interface))

    def can_worker():
        try:
            with (out/'can.jsonl').open('x', buffering=1024*1024) as stream:
                drops = {}
                while not stop.is_set():
                    for key, _ in selector.select(.1):
                        packet, ancillary, flags, _ = key.fileobj.recvmsg(72, 128)
                        received_wall = time.time_ns()
                        received_mono = time.monotonic_ns()
                        side, interface = key.data
                        kernel_ns = None
                        for level, kind, value in ancillary:
                            if level == socket.SOL_SOCKET and kind == 35:
                                sec, ns = struct.unpack('=qq', value[:16])
                                kernel_ns = sec*10**9+ns
                            if level == socket.SOL_SOCKET and kind == 40:
                                drops[interface] = struct.unpack('=I', value[:4])[0]
                                counts['socket_drops'] = sum(drops.values())
                        if len(packet) not in (16, 72) or kernel_ns is None:
                            raise RuntimeError('missing CAN frame or kernel timestamp')
                        can_id, length = struct.unpack_from('=IB', packet)
                        payload = packet[8:8+length]
                        if can_id & 0x20000000:
                            counts['can_errors'] += 1
                        row = {'side': side, 'interface': interface, 'can_id': can_id,
                               'payload_hex': payload.hex(), 'fd': len(packet)==72,
                               'kernel_wall_ns': kernel_ns, 'receive_wall_ns': received_wall,
                               'receive_monotonic_ns': received_mono,
                               'socket_flags': flags, 'decoded': decode(can_id, payload)}
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        counts['can'] += 1
        except Exception as exc:
            failures.append(repr(exc))
            stop.set()

    rclpy.init()
    node = Node('h1h2_passive_capture')
    ros_stream = (out/'ros.jsonl').open('x', buffering=1024*1024)
    def callback(topic):
        def receive(message):
            row = {'topic': topic, 'receive_wall_ns': time.time_ns(),
                   'receive_monotonic_ns': time.monotonic_ns(),
                   'message': preserve_nonfinite(message_to_ordereddict(message))}
            ros_stream.write(json.dumps(row, allow_nan=False)+'\n')
            counts['ros'] += 1
            if topic == '/robot_description':
                (out/'snapshots/robot_description.urdf').write_text(message.data)
        return receive
    topics = [(JointState, '/joint_states'), (DisplayTrajectory, '/display_planned_path'),
              (PlanningScene, '/monitored_planning_scene'), (ParameterEvent, '/parameter_events'),
              (TFMessage, '/tf')]
    for side in ('right', 'left'):
        for controller in ('joint_trajectory_controller', 'gripper_controller'):
            topics.append((JointTrajectoryControllerState, f'/{side}_{controller}/controller_state'))
    for typ, topic in topics:
        node.create_subscription(typ, topic, callback(topic), qos_profile_sensor_data)
    node.create_subscription(String, '/robot_description', callback('/robot_description'),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(TFMessage, '/tf_static', callback('/tf_static'),
                             QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    worker = threading.Thread(target=can_worker)
    worker.start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    (out/'recording.ready').write_text(str(os.getpid()))
    print(f'CAPTURE_READY {out} (receive only; both CAN sockets and ROS subscriptions ready)', flush=True)
    try:
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
    except Exception as exc:
        failures.append(repr(exc))
    finally:
        stop.set()
        worker.join()
        (out/'recording.ready').unlink(missing_ok=True)
        ros_stream.close()
        for sock in sockets:
            sock.close()
        selector.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        manifest.update(completed_wall_time_ns=time.time_ns(), complete=not failures,
                        failures=failures, counts=counts)
        # H1 remains running after capture; its tee log will append shutdown
        # output. Seal a snapshot, not the live append-only terminal file.
        if (out/'h1_terminal.log').exists():
            shutil.copyfile(out/'h1_terminal.log', out/'h1_activation_terminal.log')
            manifest['unsealed_files'] = {'h1_terminal.log': 'live H1 log; snapshot sealed as h1_activation_terminal.log'}
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in ('manifest.json', 'h1_terminal.log'):
                manifest['files'][str(path.relative_to(out))] = digest(path)
        write_json(out/'manifest.json', manifest)
        print(json.dumps({'complete': manifest['complete'], 'counts': counts, 'failures': failures}))
    if failures:
        raise RuntimeError('capture failed; raw partial files retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('record')
    r.add_argument('--output', required=True)
    r.add_argument('--home-config', required=True)
    r.add_argument('--driver-config', required=True)
    r.add_argument('--snapshot', action='append', default=[])
    r.add_argument('--right-can', default='can0')
    r.add_argument('--left-can', default='can1')
    m = sub.add_parser('mark')
    m.add_argument('directory')
    m.add_argument('name', choices=['H1_start', 'H1_end', 'H2_start', 'H2_end'])
    a = p.parse_args()
    if a.command == 'mark':
        mark(a.directory, a.name)
    else:
        record(a)


if __name__ == '__main__':
    main()
