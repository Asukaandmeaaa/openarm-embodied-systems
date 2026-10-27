"""CAN observation in a separate, receive-only interpreter, not ROS's GIL.

    Only AF_UNIX IPC is used here. Actual CAN sockets stay in the worker's
    existing receive-only Telemetry implementation. No robot control commands.
"""
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
MAX_RECORD = 65536


def send_record(channel, value):
    payload = json.dumps(value, allow_nan=False).encode()
    if not 0 < len(payload) <= MAX_RECORD:
        raise RuntimeError('Invalid CAN observer IPC record size')
    channel.sendall(struct.pack('!I', len(payload))+payload)


def receive_record(channel, deadline=None):
    def exact(length):
        parts = []
        while length:
            if deadline is not None:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('CAN observer IPC deadline exceeded')
                channel.settimeout(remaining)
            chunk = channel.recv(length)
            if not chunk:
                raise EOFError('CAN observer IPC closed')
            parts.append(chunk)
            length -= len(chunk)
        return b''.join(parts)
    length, = struct.unpack('!I', exact(4))
    if not 0 < length <= MAX_RECORD:
        raise RuntimeError('Invalid CAN observer IPC record size')
    return json.loads(exact(length))


def worker_command(output, interfaces, descriptor):
    return [sys.executable, '-m', 'scripts.act_trial.telemetry_worker',
            '--output', str(output), '--interfaces', json.dumps(list(interfaces)),
            '--fd', str(descriptor)]


class Telemetry:
    def __init__(self, output, interfaces=('can0', 'can1')):
        self.output = Path(output).resolve()
        self.error = None
        self.count = 0
        self.drops = {}
        self.closed = False
        self.lock = threading.Lock()
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.process = None
        self.diagnostic_path = self.output.with_name(self.output.name+'.observer.log')
        try:
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            with self.diagnostic_path.open('x') as diagnostic:
                self.process = subprocess.Popen(worker_command(self.output, interfaces, child.fileno()),
                    pass_fds=(child.fileno(),), close_fds=True, cwd=ROOT, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=diagnostic, env=environment)
            child.close()
            hello = receive_record(self.channel, time.monotonic()+5.)
            if not hello.get('ready'):
                raise RuntimeError(hello.get('error', 'CAN observer worker failed to start'))
            self.worker_pid = hello['pid']
        except BaseException:
            child.close()
            self.channel.close()
            self._reap()
            raise

    def _reap(self):
        if self.process is None:
            return
        try:
            self.process.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            # This is only our receive-only observer, never H1/gateway/MoveIt.
            self.process.terminate()
            try:
                self.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.)

    def _request(self, operation, *, timeout=.05, **payload):
        deadline = time.monotonic()+timeout
        if not self.lock.acquire(timeout=timeout):
            self.error = 'CAN observer IPC lock timed out'
            raise RuntimeError(self.error)
        try:
            self.channel.settimeout(max(.000001, deadline-time.monotonic()))
            send_record(self.channel, dict(op=operation, **payload))
            reply = receive_record(self.channel, deadline)
            if not isinstance(reply, dict):
                raise RuntimeError('Invalid CAN observer IPC reply')
        except (OSError, EOFError, ValueError, RuntimeError) as exc:
            self.error = 'CAN observer IPC failed: '+str(exc)
            self.channel.close()  # A late reply can never satisfy another request.
            raise RuntimeError(self.error) from exc
        finally:
            self.lock.release()
        self.count = reply.get('count', self.count)
        self.drops = reply.get('drops', self.drops)
        if reply.get('latched_error'):
            self.error = reply['latched_error']
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', 'CAN observer request failed'))
        return reply

    def snapshot(self, max_age=.1):
        if self.closed:
            raise RuntimeError('CAN observer is closed')
        if self.error:
            raise RuntimeError(self.error)
        if not math.isfinite(max_age) or not 0 < max_age <= .1:
            raise ValueError('CAN source age must be in (0, 0.1] seconds')
        result = self._request('snapshot', max_age=max_age)['snapshot']
        # Keep kernel-derived source times across IPC. Worker freshness alone
        # is not enough: transport or caller scheduling may delay a reply.
        now = time.monotonic_ns()
        for key, age_key, label in (
                ('source_monotonic_ns', 'max_age_s', 'feedback'),
                ('command_source_monotonic_ns', 'command_max_age_s', 'commands')):
            stamps = result[key]
            if len(stamps) != 16:
                raise RuntimeError('CAN observer returned incomplete '+label)
            ages = [(now-stamp)/1e9 for stamp in stamps]
            if min(ages) < 0 or max(ages) > max_age:
                raise RuntimeError(f'stale CAN {label} after IPC: {max(ages):.3f}s')
            result[age_key] = max(ages)
        result['observer_backend'] = 'isolated_receive_only_process'
        result['observer_process_pid'] = self.worker_pid
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.channel.fileno() >= 0:
                self._request('close', timeout=5.)
        finally:
            self.channel.close()
            self._reap()
