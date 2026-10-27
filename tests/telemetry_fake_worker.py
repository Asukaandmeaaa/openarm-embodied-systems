"""Synthetic receive-only process fixture. Never creates CAN or ROS objects."""
import os
from pathlib import Path
import socket
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.act_trial.isolated_telemetry import send_record
from scripts.act_trial.telemetry_worker import serve


class FakeObserver:
    def __init__(self, mode):
        self.mode = mode
        self.count = 0
        self.drops = {}
        self.error = None
        self.stamp = time.monotonic_ns()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        while not self.stop.wait(.001):
            self.stamp = time.monotonic_ns()
            self.count += 1

    def snapshot(self, max_age=.1):
        if self.mode == 'crash':
            os._exit(9)
        if self.mode == 'delay':
            time.sleep(.2)
        if self.mode == 'latched_error':
            self.error = 'CAN source timestamp delayed: fixture'
            raise RuntimeError(self.error)
        stamp = self.stamp
        if self.mode == 'stale':
            stamp -= 232_000_000
        if self.mode == 'future':
            stamp += 1_000_000_000
        result = dict(source_monotonic_ns=[stamp]*16,
            command_source_monotonic_ns=[stamp]*16, max_age_s=0., command_max_age_s=0.,
            positions_rad=[0.]*16, motor_status=[1]*16,
            fixture_pid=os.getpid(), fixture_count=self.count,
            fixture_thread_limits={key:os.environ.get(key) for key in
                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        if self.mode == 'stale_command':
            result['command_source_monotonic_ns'] = [stamp-232_000_000]*16
        return result

    def close(self):
        self.stop.set()
        self.thread.join(1.)


if __name__ == '__main__':
    mode, descriptor = sys.argv[1], int(sys.argv[2])
    with socket.socket(fileno=descriptor) as channel:
        if mode == 'startup_error':
            send_record(channel, dict(ready=False, error='fixture startup error'))
        else:
            serve(channel, FakeObserver(mode))
