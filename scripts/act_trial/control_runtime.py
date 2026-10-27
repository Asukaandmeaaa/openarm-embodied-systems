"""Temporary H1/H2 state bound to live control processes, not ACT log folders.

Reads /proc only. No ROS, CAN, signals, ownership changes or motor commands.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[2]


def runtime_root():
    suffix = hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]
    path = Path(f'/run/user/{os.getuid()}') / ('openarm-act-' + suffix)
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError('control runtime directory must be owned by this user with mode0700')
    return path


def live_identity(proc=Path('/proc')):
    found = {'backend': [], 'gateway': []}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = [s.decode() for s in entry.joinpath('cmdline').read_bytes().split(b'\0') if s]
            if not args:
                continue
            if Path(args[0]).name == 'ros2_control_node':
                role = 'backend'
            elif any(args[i:i+2] == ['-m', 'scripts.control_ownership.gateway']
                     for i in range(len(args))):
                role = 'gateway'
            else:
                continue
            env = dict(s.split(b'=', 1) for s in entry.joinpath('environ').read_bytes().split(b'\0')
                       if b'=' in s)
            if env.get(b'ROS_DOMAIN_ID') != b'117':
                continue
            if (env.get(b'ROS_SECURITY_ENABLE') != b'true'
                    or env.get(b'ROS_SECURITY_STRATEGY') != b'Enforce'):
                raise ValueError('live ACT control process is not using SROS2 Enforce')
            expected_keys = os.environ.get('OPENARM_KEYSTORE')
            if expected_keys and env.get(b'ROS_SECURITY_KEYSTORE') != expected_keys.encode():
                raise ValueError('live control process uses a different keystore')
            expected_enclave = '/openarm/' + role
            if not any(args[i:i+2] == ['--enclave', expected_enclave] for i in range(len(args))):
                raise ValueError('live control process has an unexpected enclave')
            # /proc/<pid>/stat field 22, after the parenthesized comm field.
            fields = entry.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            found[role].append({'pid': int(entry.name), 'start_ticks': int(fields[19])})
        except FileNotFoundError:
            continue
    if any(len(values) != 1 for values in found.values()):
        raise ValueError('expected exactly one live secure ROS backend and gateway; '
                         'keep H1/gateway running and inspect competing processes')
    return {'boot_id': proc.joinpath('sys/kernel/random/boot_id').read_text().strip(),
            **{role: values[0] for role, values in found.items()}}


def control_session():
    identity = live_identity()
    encoded = json.dumps(identity, sort_keys=True)
    key = hashlib.sha256(encoded.encode()).hexdigest()[:24]
    path = runtime_root() / ('control_' + key)
    path.mkdir(mode=0o700, exist_ok=True)
    marker = path / 'identity.json'
    try:
        with marker.open('x') as stream:
            stream.write(encoded + '\n')
    except FileExistsError:
        if json.loads(marker.read_text()) != identity:
            raise ValueError('control runtime identity changed')
    return path


@contextmanager
def control_operation(directory):
    """Serialize H2/recovery for this live control generation across terminals."""
    with (directory / 'operation.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('H2 or HOME recovery is already running in another terminal') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
