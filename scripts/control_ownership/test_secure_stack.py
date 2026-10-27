"""Software-only startup of the real ROS launch using GenericSystem, never CAN.

The model file is isolated and checked for exclusively mock_components plugins.
Uses a fresh signed DDS domain, records logs, stops only its own child process.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from scripts.control_ownership.security import environment, provision, sha256, POLICY


def child(keystore, model, offset):
    from launch import LaunchService
    from scripts.control_ownership.secure_launch import create_launch_description
    settings = environment(keystore)
    os.environ.update(settings)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('ROS_SECURITY_ENCLAVE_OVERRIDE', None)
    service = LaunchService()
    service.include_launch_description(create_launch_description(settings, software_test_directory=model,
                                                                 software_test_offset=offset))
    return service.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--domain', type=int, default=162)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--keep-running', action='store_true', help='integration harness; Ctrl-C stops only fake child')
    parser.add_argument('--position-state-offset', type=float, default=0.,
                        help='mock-only artificial feedback offset; optional0.02rad hold-reference regression')
    args = parser.parse_args()
    if not 150 <= args.domain <= 232:
        parser.error('mock tests require an isolated domain in150..232; physical/legacy domains are forbidden')
    root = Path(args.output).resolve()
    if args.child:
        raise SystemExit(child(root / 'keystore', root / 'model', args.position_state_offset))
    root.mkdir(parents=True, exist_ok=False)
    provision(root / 'keystore', args.domain)
    output = open(root / 'stack.log', 'x')
    process = subprocess.Popen([sys.executable, '-m', 'scripts.control_ownership.test_secure_stack',
                                '--output', str(root), '--domain', str(args.domain), '--child',
                                '--position-state-offset', str(args.position_state_offset)],
                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'process.json').write_text(json.dumps({'fake_launch_pid': process.pid, 'domain': args.domain}))
    success = False
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(.25)
            text = (root / 'stack.log').read_text()
            if 'topic not found in allow rule' in text or 'process has died' in text:
                break
            success = ('You can start planning now!' in text
                       and text.count('process has finished cleanly') >= 3)
            if success:
                break
        model = root / 'model/openarm_bimanual.urdf'
        plugins = [item.text for item in ET.parse(model).findall('.//ros2_control/hardware/plugin')]
        if plugins != ['mock_components/GenericSystem', 'mock_components/GenericSystem']:
            raise AssertionError('software-only model must contain exactly two GenericSystem plugins')
        report = {'passed': success, 'physical_hardware_loaded': False, 'plugins': plugins,
                  'mock_position_state_following_offset': args.position_state_offset,
                  'domain': args.domain, 'policy_sha256': sha256(POLICY),
                  'log': str(root / 'stack.log'), 'keystore': str(root / 'keystore')}
        (root / 'stack_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        while success and args.keep_running and process.poll() is None:
            time.sleep(1)
    finally:
        if process.poll() is None:
            # PID belongs solely to the mock process created above.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        output.close()
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
