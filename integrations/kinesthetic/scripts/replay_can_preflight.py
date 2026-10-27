#!/usr/bin/env python3
"""Conservative H1 SocketCAN setup. Never resets an UP interface."""
import json
from pathlib import Path
import subprocess
import sys

INTERFACES = ('can0', 'can1')


def require_no_hardware():
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if any(Path(a.decode(errors='replace')).name == 'ros2_control_node'
               for a in args if a):
            raise RuntimeError(f'Hardware process PID {proc.name} exists; stop it manually first.')


def read_link(name):
    result = subprocess.run(['ip', '-json', '-details', 'link', 'show', 'dev', name],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f'{name} unavailable; check USB-CAN cable/power. {result.stderr.strip()}')
    links = json.loads(result.stdout)
    if len(links) != 1 or links[0].get('linkinfo', {}).get('info_kind') != 'can':
        raise RuntimeError(f'{name} is not a CAN interface; refusing startup.')
    return links[0]


def is_up(link):
    return 'UP' in link.get('flags', [])


def require_ready(name, link):
    data = link['linkinfo']['info_data']
    nominal = data.get('bittiming', {})
    fast = data.get('data_bittiming', {})
    valid = (is_up(link) and link.get('mtu') == 72 and link.get('txqlen') == 1000
             and set(data.get('ctrlmode', [])) == {'FD'}
             and data.get('state') == 'ERROR-ACTIVE'
             and nominal.get('bitrate') == 1000000
             and float(nominal.get('sample_point', 0)) == 0.75
             and fast.get('bitrate') == 5000000
             and float(fast.get('sample_point', 0)) == 0.75
             and fast.get('sjw') == 2)
    if not valid:
        raise RuntimeError(f'{name}: parameters/state mismatch; refusing startup. '
                           'Will NOT down/up an active interface. Inspect with: '
                           f'ip -details link show {name}')


def setup():
    require_no_hardware()
    # Inspect BOTH before any mutation, so missing/mismatched peers fail early.
    links = {name: read_link(name) for name in INTERFACES}
    for name, link in links.items():
        if is_up(link):
            require_ready(name, link)
    for name, link in links.items():
        if is_up(link):
            print(f'CAN_PREFLIGHT: {name} already ready; unchanged.', flush=True)
            continue
        require_no_hardware()
        current = read_link(name)
        if is_up(current):
            require_ready(name, current)
            continue
        if current['ifindex'] != link['ifindex']:
            raise RuntimeError(f'{name} reconnected during setup; retry only after USB is stable.')
        print(f'CAN_PREFLIGHT: configuring DOWN interface {name}; sudo may request password.', flush=True)
        commands = [
            ['type', 'can', 'bitrate', '1000000', 'sample-point', '0.75',
             'dbitrate', '5000000', 'dsample-point', '0.75', 'dsjw', '2', 'fd', 'on'],
            ['txqueuelen', '1000'], ['up'],
        ]
        for args in commands:
            subprocess.run(['sudo', 'ip', 'link', 'set', 'dev', name, *args], check=True)
    for name in INTERFACES:
        require_ready(name, read_link(name))
    require_no_hardware()
    print('CAN_PREFLIGHT: PASS (right=can0, left=can1; verify physical wiring).', flush=True)


if __name__ == '__main__':
    try:
        setup()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'CAN_PREFLIGHT FAILED: {exc}\nH1 aborted; no automatic rollback/reset. '
              'An earlier interface may already be configured.', file=sys.stderr)
        sys.exit(1)
