"""Real JTC + mock offset regression: synchronization/loss retain OUTPUT, not feedback."""
import argparse
import json
import os
from pathlib import Path
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stack', required=True); p.add_argument('--socket', required=True)
    args = p.parse_args(); root = Path(args.stack)
    from .security import environment
    settings = environment(root / 'keystore')
    report = json.loads((root / 'stack_report.json').read_text())
    if (settings['ROS_DOMAIN_ID'] in ('0', '97', '117') or not report['passed']
            or report['physical_hardware_loaded'] is not False
            or report['mock_position_state_following_offset'] != .02):
        raise ValueError('requires successful isolated0.02rad GenericSystem mock stack')
    os.environ.update(settings); os.environ['ROS_SECURITY_ENCLAVE_OVERRIDE'] = '/openarm/observer'
    import rclpy
    from control_msgs.msg import JointTrajectoryControllerState
    from .client import Client
    from .gateway import CONTROLLERS
    required = CONTROLLERS[:3]
    rclpy.init(); node = rclpy.create_node('openarm_ownership_observer')
    states = {}
    for controller in required:
        def callback(msg, c=controller):
            states[c] = {'output': list(msg.output.positions), 'feedback': list(msg.feedback.positions)}
        node.create_subscription(JointTrajectoryControllerState, '/' + controller + '/controller_state', callback, 10)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); spin.start()
    def wait(predicate, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate(): return
            time.sleep(.05)
        raise TimeoutError('offset test condition timed out')
    def snapshot(): return json.loads(json.dumps(states))
    try:
        wait(lambda: set(states) == set(required))
        before = snapshot()
        offsets = {c: [a-b for a,b in zip(v['feedback'], v['output'])] for c,v in before.items()}
        assert all(abs(delta-.02)<1e-8 for values in offsets.values() for delta in values), offsets
        with Client(args.socket) as monitor:
            monitor.request('synchronize-hold')
            wait(lambda: monitor.request('status')['synchronized'])
            wait(lambda: monitor.request('status')['status']['state']=='idle')
            after_sync = snapshot()
            assert all(before[c]['output']==after_sync[c]['output'] for c in required)
            # Stream same target under Dora authority, then lose the connection.
            with Client(args.socket) as owner:
                owner.acquire('dora')
                owner.command(before[CONTROLLERS[0]]['output'])
            wait(lambda: monitor.request('status')['status']['state']=='idle')
            after_disconnect = snapshot()
            assert all(before[c]['output']==after_disconnect[c]['output'] for c in required)
        result = {'passed': True, 'physical_hardware_loaded': False, 'mock_following_offset_rad': .02,
                  'before': before, 'after_synchronize': after_sync, 'after_owner_disconnect': after_disconnect,
                  'output_reference_preserved': True, 'feedback_was_not_substituted': True}
        (root / 'hold_offset_report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
    finally:
        rclpy.try_shutdown(); spin.join(timeout=2); node.destroy_node()


if __name__ == '__main__': main()
