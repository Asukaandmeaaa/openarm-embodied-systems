"""Software-only probe of actual MoveIt capability forwarding into real mock JTC."""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import threading
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', required=True)
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    root = Path(args.stack).resolve()
    from .security import environment
    settings = environment(root / 'keystore')
    if settings['ROS_DOMAIN_ID'] in ('0', '97', '117'):
        raise ValueError('test refuses physical domains')
    plugins = [p.text for p in ET.parse(root / 'model/openarm_bimanual.urdf').findall('.//ros2_control/hardware/plugin')]
    if plugins != ['mock_components/GenericSystem'] * 2:
        raise ValueError('requires explicit mock stack model')
    os.environ.update(settings)
    os.environ['ROS_SECURITY_ENCLAVE_OVERRIDE'] = '/openarm/h2'
    import rclpy
    from rclpy.action import ActionClient
    from moveit_msgs.action import ExecuteTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from sensor_msgs.msg import JointState
    from .client import Client
    rclpy.init()
    node = rclpy.create_node('home_guarded_executor')
    action = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
    measured = {}
    node.create_subscription(JointState, '/joint_states', lambda m: measured.update(dict(zip(m.name, m.position))), 10)
    stop = threading.Event()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); spin.start()
    outcomes = []
    with Client(args.socket) as lease:
        lease.acquire('h2')
        def renew():
            while not stop.wait(.1):
                lease.renew()
        renewer = threading.Thread(target=renew, daemon=True); renewer.start()
        def wait(future, timeout=10):
            end = time.monotonic() + timeout
            while not future.done() and time.monotonic() < end:
                time.sleep(.01)
            if not future.done(): raise TimeoutError('MoveIt future timed out')
            return future.result()
        try:
            if not action.wait_for_server(timeout_sec=5): raise RuntimeError('MoveIt unavailable')
            end_at = time.monotonic() + 5
            while 'openarm_right_joint7' not in measured and time.monotonic() < end_at:
                time.sleep(.01)
            if 'openarm_right_joint7' not in measured:
                raise RuntimeError('mock joint feedback unavailable')
            capability = None
            for case in ('unsigned', 'fresh_capability', 'replayed_nonce'):
                signed = case != 'unsigned'
                goal = ExecuteTrajectory.Goal()
                trajectory = goal.trajectory.joint_trajectory
                trajectory.joint_names = [f'openarm_right_joint{i}' for i in range(1, 8)]
                trajectory.header.frame_id = 'world'
                if case == 'fresh_capability':
                    capability = 'openarm-lease:' + lease.token + ':' + secrets.token_hex(16) + ':' + base64.urlsafe_b64encode(b'world').decode()
                if signed:
                    trajectory.header.frame_id = capability
                current = [measured[name] for name in trajectory.joint_names]
                start = JointTrajectoryPoint(); start.positions = current; start.velocities = [0.] * 7
                end = JointTrajectoryPoint(); end.positions = [current[0]+.01] + current[1:]; end.velocities = [0.] * 7; end.time_from_start.sec = 1
                trajectory.points = [start, end]
                handle = wait(action.send_goal_async(goal))
                if not handle.accepted: raise AssertionError('MoveIt outer action rejected unexpectedly')
                wrapper = wait(handle.get_result_async())
                outcome = {'case': case, 'signed': signed, 'status': wrapper.status,
                           'moveit_error': wrapper.result.error_code.val,
                           'message': wrapper.result.error_code.message}
                outcomes.append(outcome)
                if case == 'fresh_capability':
                    assert wrapper.status == 4 and wrapper.result.error_code.val == 1, outcome
                else:
                    assert wrapper.status != 4 or wrapper.result.error_code.val != 1, outcome
            lease.release()
        finally:
            stop.set(); renewer.join(timeout=2)
            rclpy.try_shutdown(); spin.join(timeout=2); node.destroy_node()
    report = {'passed': True, 'physical_hardware_loaded': False, 'outcomes': outcomes,
              'proves': 'actual MoveIt preserves signed trajectory frame_id; gateway rejects unsigned and forwards signed goal'}
    (root / 'moveit_proxy_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
