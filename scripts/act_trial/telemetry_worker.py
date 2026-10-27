"""Private receive-only worker: no ROS, planner or control authority.

The historical decoder indirectly imports NumPy; its BLAS threads are capped
by the parent. There is no inference, image processing or linear algebra here.
"""
import argparse
import json
import os
import socket

from .isolated_telemetry import receive_record, send_record
from .telemetry import Telemetry as ReceiveOnlyTelemetry


def serve(channel, observer):
    """One response per request; snapshots retain their kernel-derived clocks."""
    closed = False
    try:
        send_record(channel, dict(ready=True, pid=os.getpid()))
        while True:
            request = receive_record(channel)
            operation = request.get('op')
            response = dict(ok=True)
            try:
                if operation == 'snapshot':
                    response['snapshot'] = observer.snapshot(max_age=request['max_age'])
                elif operation == 'close':
                    closed = True
                    observer.close()
                else:
                    raise ValueError('Unsupported receive-only CAN observer operation')
            except Exception as exc:
                response.update(ok=False, error=str(exc))
            response.update(count=observer.count, drops=observer.drops,
                            latched_error=observer.error)
            send_record(channel, response)
            if operation == 'close':
                return
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        if not closed:
            observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interfaces', required=True)
    args = parser.parse_args()
    with socket.socket(fileno=args.fd) as channel:
        try:
            observer = ReceiveOnlyTelemetry(args.output, tuple(json.loads(args.interfaces)))
        except Exception as exc:
            send_record(channel, dict(ready=False, error=str(exc)))
            raise
        serve(channel, observer)


if __name__ == '__main__':
    main()
