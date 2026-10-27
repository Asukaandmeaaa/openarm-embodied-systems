"""Connection-bound command capability; no ROS or CAN imports."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

DEFAULT_SOCKET = f'/run/user/{os.getuid()}/openarm-control.sock'


class Client:
    def __init__(self, path=None):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(2.)
        self.socket.connect(path or os.environ.get('OPENARM_AUTHORITY_SOCKET', DEFAULT_SOCKET))
        self.stream = self.socket.makefile('rwb')
        self.lock = threading.Lock()
        self.token = None
        self.owner = None

    def request(self, operation, **kwargs):
        with self.lock:
            payload = {'op': operation, **kwargs}
            if self.token is not None:
                payload['token'] = self.token
            self.stream.write((json.dumps(payload, allow_nan=False)+'\n').encode())
            self.stream.flush()
            line = self.stream.readline(65537)
            if not line or len(line) > 65536:
                raise RuntimeError('authority connection closed or invalid response')
            response = json.loads(line)
            if not response.get('ok'):
                raise ValueError(response.get('error', 'authority rejected'))
            return response

    def acquire(self, owner, scope=None):
        response = self.request('acquire', owner=owner, **({} if scope is None else {'scope':scope}))
        self.token, self.owner = response['token'], owner
        return response

    def renew(self):
        return self.request('renew')

    def command(self, q7, velocities=None):
        return self.request('command', positions=list(q7),
                            **({} if velocities is None else {'velocities': list(velocities)}))

    def release(self):
        result = self.request('release')
        self.token = self.owner = None
        return result

    def close(self):
        self.stream.close()
        self.socket.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default=None)
    sub = parser.add_subparsers(dest='op', required=True)
    sub.add_parser('status')
    sub.add_parser('synchronize-hold', help='MANUAL: preempt timed goals with the current controller OUTPUT, without enabling motors')
    hold = sub.add_parser('hold-test', help='MANUAL physical validation: send the EXISTING right reference for one second, no ACT')
    hold.add_argument('--report', required=True)
    sub.add_parser('reset-fault', help='Only accepted after all backend goals drained and holding verified')
    run = sub.add_parser('run', help='Hold H2 authority for exactly one child command')
    run.add_argument('--owner', choices=['h2'], required=True)
    run.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with Client(args.socket) as client:
        if args.op == 'hold-test':
            report_path = Path(args.report)
            if report_path.exists(): parser.error('report already exists')
            acquired = client.acquire('dora')
            reference = acquired['right_reference']
            pv = client.request('status').get('controller_command_interfaces', {}).get(
                'right_joint_trajectory_controller') == ['position', 'velocity']
            result = {'test': 'manual_existing_reference_hold', 'ACT_executed': False,
                      'right_reference': reference, 'competing_sources_rejected': [],
                      'wall_ns': time.time_ns(), 'passed': False}
            print('MANUAL HOLD TEST: sending the existing right-arm reference; no enable or HOME command.', flush=True)
            try:
                with Client(args.socket) as competitor:
                    for owner in ('dora', 'h2'):
                        try:
                            competitor.acquire(owner)
                        except ValueError:
                            result['competing_sources_rejected'].append(owner)
                        else:
                            raise RuntimeError('competing owner unexpectedly admitted')
                for _ in range(10):
                    client.renew()
                    client.command(reference, velocities=[0.]*7 if pv else None)
                    time.sleep(.1)
                client.release()
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    state = client.request('status')
                    if state['status']['state'] == 'idle': break
                    time.sleep(.1)
                else:
                    raise RuntimeError('hold did not settle; inspect gateway status/log without restarting hardware')
                result['passed'] = True
            finally:
                report_path.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result, indent=2)); return
        if args.op != 'run':
            print(json.dumps(client.request(args.op), indent=2)); return
        command = args.command
        if command and command[0] == '--': command = command[1:]
        if not command: parser.error('missing child command')
        client.acquire(args.owner)
        child = None
        try:
            child_env = os.environ.copy()
            child_env['OPENARM_AUTHORITY_TOKEN'] = client.token
            child = subprocess.Popen(command, env=child_env, start_new_session=True)
            while child.poll() is None:
                client.renew()
                time.sleep(.1)
            # A failed H2 command does not silently permit ACT acquisition.
            if child.returncode:
                client.request('freeze', reason=f'H2 child exited {child.returncode}')
            else:
                client.release()
            raise SystemExit(child.returncode)
        except BaseException as exc:
            # Losing a client never shuts down ROS or disables motors. The
            # gateway revokes authority, cancels active goals and holds.
            if child is not None and child.poll() is None:
                if isinstance(exc,(ValueError,RuntimeError,OSError)):
                    print('H2 authority ended: '+str(exc)+'; waiting for child cancellation/report cleanup.',flush=True)
                    try:child.wait(timeout=12.)
                    except subprocess.TimeoutExpired:pass
                    else:
                        # Preserve the child's primary diagnostic above instead
                        # of replacing it with a lease-renewal traceback.
                        raise SystemExit(child.returncode or 1) from None
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5.)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            raise


if __name__ == '__main__': main()
