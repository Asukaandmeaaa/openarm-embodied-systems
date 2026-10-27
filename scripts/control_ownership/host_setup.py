"""Prepare volatile CAN host resources; never open a CAN socket or start ROS.

Run as the workstation user. Only module loading uses sudo when needed.
Existing lock files, network interfaces and motor settings are untouched.
"""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

LOCK_ROOT = Path('/run/lock/openarm-can')
CAN_PROC = Path('/proc/net/can')


def ensure_lock_directory(root=LOCK_ROOT):
    root.mkdir(mode=0o755, exist_ok=True)
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.geteuid())
            or info.st_mode & 0o022 or not os.access(root, os.W_OK | os.X_OK)):
        raise ValueError('CAN lock directory has unsafe owner/permissions or is unwritable: '+str(root))


def require_can_registry(root=CAN_PROC):
    if not root.is_dir() or not any(root.glob('rcvlist_*')):
        raise ValueError('CAN receiver registry unavailable: '+str(root)+
                         '; run ./scripts/actctl prepare-host first '
                         '(loads can_raw; no motor commands)')


def prepare_host():
    ensure_lock_directory()
    # can_dev/gs_usb can expose network interfaces before the CAN protocol
    # module creates /proc/net/can. Ownership checks need that registry BEFORE
    # the parameter probe constructs its first SocketCAN socket.
    if not CAN_PROC.is_dir() or not Path('/sys/module/can_raw').is_dir():
        modprobe = shutil.which('modprobe')
        if modprobe is None:
            raise ValueError('modprobe unavailable; install the host kmod tools')
        command = [modprobe, 'can_raw']
        if os.geteuid() != 0:
            command = ['sudo', *command]
        subprocess.run(command, check=True)
    require_can_registry()
    return {'host_prepared': True, 'lock_directory': str(LOCK_ROOT),
            'CAN_receiver_registry': str(CAN_PROC), 'CAN_sockets_opened': False,
            'motor_commands_sent': False, 'ROS_started': False}


def main():
    if os.geteuid() == 0:
        raise SystemExit('Run ./scripts/actctl prepare-host as the workstation user, without sudo; '
                         'only modprobe will request sudo.')
    print(json.dumps(prepare_host(), indent=2))


if __name__ == '__main__':
    main()
