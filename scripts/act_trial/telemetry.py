"""Timestamped receive-only CAN observer. No transmission or motor control."""
import argparse
import json
from pathlib import Path
import queue
import selectors
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'home_replay'))
from core import decode as decode_capture, write_json


class RawCanLog:
    """Bounded, lossless log queue. The CAN receiver never does file I/O.

    A blocked disk must not block feedback reception. A full queue or writer
    error is nevertheless a latched failure, never permission to drop evidence
    or continue moving without logging. This is not a real-time scheduler.
    """
    def __init__(self, output, capacity=16384):
        self.stream = Path(output).open('x', buffering=1024*1024)
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.written = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='can-log-writer', daemon=True)
        self.thread.start()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def append(self, row):
        self.check()
        if self.closed:
            raise RuntimeError('CAN raw log already closed')
        try:
            self.queue.put_nowait(row)
        except queue.Full as exc:
            self.error = 'CAN raw log queue overflow; refusing unlogged operation'
            raise RuntimeError(self.error) from exc

    def _run(self):
        try:
            last_flush = time.monotonic()
            while True:
                try:
                    row = self.queue.get(timeout=.2)
                except queue.Empty:
                    if self.closed:
                        break
                    self.stream.flush()
                    last_flush = time.monotonic()
                    continue
                if row is None:
                    break
                self.stream.write(json.dumps(row, allow_nan=False)+'\n')
                self.written += 1
                if time.monotonic()-last_flush >= .2:
                    self.stream.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self.error = 'CAN raw log writer failed: '+str(exc)
        finally:
            try:
                self.stream.close()
            except Exception as exc:
                self.error = 'CAN raw log close failed: '+str(exc)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.thread.is_alive():
                try:
                    self.queue.put_nowait(None)
                except queue.Full:
                    # A full but healthy queue drains normally; the worker
                    # exits on Empty+closed. Never block a producer on disk.
                    pass
                self.thread.join(timeout=2)
                if self.thread.is_alive():
                    self.error = 'CAN raw log writer did not stop; log may be incomplete'
                    raise RuntimeError(self.error)
        self.check()


def decode(can_id, payload):
    """Live ROS backend scale, without rewriting historical capture decoding.

    ROS uses DM4340 (not DM4340_48V) for J3/J4. Its installed OpenArmCAN
    MOTOR_LIMIT_PARAMS entry is +/-8 rad/s; capture core used +/-10 rad/s.
    Preserve the original interpretation alongside the corrected live value.
    Positions, torque, all raw bytes and other motor types are unchanged.
    """
    result = decode_capture(can_id, payload)
    if result and result.get('motor_type') == 'DM4340' and 'dq' in result:
        result['capture_decoder_dq_rad_s'] = result['dq']
        result['dq'] *= 8. / 10.
        result['velocity_scale_rad_s'] = 8.
        result['velocity_scale_basis'] = 'ROS OpenArmCAN DM4340 MOTOR_LIMIT_PARAMS'
    return result


class Telemetry:
    def __init__(self, output, interfaces=('can0', 'can1')):
        self.output = Path(output)
        self.log = RawCanLog(self.output)
        self.selector = selectors.DefaultSelector()
        self.sockets = []
        self.lock = threading.Lock()
        self.latest = {}
        self.latest_command = {}
        self.error = None
        self.drops = {}; self.count = 0; self.stop_event = threading.Event()
        try:
            for side, interface in enumerate(interfaces):
                sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                self.sockets.append(sock)
                sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
                sock.setsockopt(socket.SOL_CAN_RAW, 2, struct.pack('=I', 0x1FFFFFFF))
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8*1024*1024)
                sock.setsockopt(socket.SOL_SOCKET, 35, 1)
                sock.setsockopt(socket.SOL_SOCKET, 40, 1)
                sock.bind((interface,)); sock.setblocking(False)
                self.selector.register(sock, selectors.EVENT_READ, side)
        except Exception:
            for sock in self.sockets:
                sock.close()
            self.selector.close()
            self.log.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self.log.check()
                for key, _ in self.selector.select(.001):
                    packet, ancillary, flags, _ = key.fileobj.recvmsg(72, 128)
                    wall = time.time_ns(); mono = time.monotonic_ns(); stamp = None
                    for level, kind, value in ancillary:
                        if level == socket.SOL_SOCKET and kind == 35:
                            sec, ns = struct.unpack('=qq', value[:16]); stamp = sec*10**9+ns
                        if level == socket.SOL_SOCKET and kind == 40:
                            self.drops[key.data] = struct.unpack('=I', value[:4])[0]
                    # SocketCAN sets MSG_DONTROUTE for locally originated CAN
                    # frames, including ROS commands observed on this separate
                    # receive-only socket. MSG_CONFIRM identifies own-socket
                    # echoes. Neither means truncation. Reject actual payload
                    # or ancillary truncation, preserving the origin flags.
                    # https://docs.kernel.org/networking/can.html#returned-message-flags
                    truncated = flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                    if stamp is None or truncated or len(packet) not in (16, 72):
                        raise RuntimeError('invalid CAN timestamp/frame or truncated packet')
                    can_id, length = struct.unpack_from('=IB', packet)
                    if can_id & 0x20000000 or any(self.drops.values()):
                        raise RuntimeError('CAN error frame or receive overflow')
                    d = decode(can_id, packet[8:8+length])
                    row = {'side': 'right' if key.data == 0 else 'left', 'can_id': can_id,
                           'payload_hex': packet[8:8+length].hex(), 'kernel_wall_ns': stamp,
                           'receive_wall_ns': wall, 'receive_monotonic_ns': mono,
                           'socket_flags': flags, 'frame_bytes': len(packet),
                           'decoded': d}
                    # Enqueue even a delayed frame for diagnosis, but never
                    # serialize/flush on this receive/safety-cache thread.
                    self.log.append(row); self.count += 1
                    if d and d['kind'] in ('feedback', 'command'):
                        if d['kind'] == 'feedback' and packet[8] & 15 != d['axis']+1:
                            raise RuntimeError('CAN response motor ID mismatch')
                        if wall < stamp or wall-stamp > 100_000_000:
                            raise RuntimeError('CAN source timestamp delayed or clock discontinuity: '
                                               f'{row["side"]} axis={d["axis"]+1} {d["kind"]} '
                                               f'age_ms={(wall-stamp)/1e6:.3f}, limit_ms=100')
                        with self.lock:
                            target = self.latest if d['kind'] == 'feedback' else self.latest_command
                            target[key.data*8+d['axis']] = (dict(d), mono-(wall-stamp))
        except Exception as exc:
            self.error = str(exc)

    def snapshot(self, max_age=.1):
        if self.error: raise RuntimeError(self.error)
        self.log.check()
        with self.lock:
            values = dict(self.latest)
            commands = dict(self.latest_command)
        if set(values) != set(range(16)):
            raise RuntimeError(f'fresh feedback missing axes: {sorted(set(range(16))-set(values))}')
        now = time.monotonic_ns()
        ages = [(now-values[i][1])/1e9 for i in range(16)]
        if min(ages) < 0 or max(ages) > max_age: raise RuntimeError(f'stale CAN feedback: {max(ages):.3f}s')
        if any(values[i][0]['status'] != 1 for i in range(16)):
            raise RuntimeError('CAN motor is disabled or reports a fault')
        if set(commands) != set(range(16)):
            raise RuntimeError(f'fresh MIT commands missing axes: {sorted(set(range(16))-set(commands))}')
        command_ages = [(now-commands[i][1])/1e9 for i in range(16)]
        if min(command_ages) < 0 or max(command_ages) > max_age:
            raise RuntimeError(f'stale CAN commands: {max(command_ages):.3f}s')
        return {'positions_rad':[values[i][0]['q'] for i in range(16)],
                'velocity_rad_s':[values[i][0]['dq'] for i in range(16)],
                'torque_nm':[values[i][0]['torque'] for i in range(16)],
                'motor_status':[values[i][0]['status'] for i in range(16)],
                'source_monotonic_ns':[values[i][1] for i in range(16)], 'max_age_s':max(ages),
                'command_positions_rad':[commands[i][0]['q'] for i in range(16)],
                'command_velocity_rad_s':[commands[i][0]['dq'] for i in range(16)],
                'command_torque_nm':[commands[i][0]['torque'] for i in range(16)],
                'command_kp':[commands[i][0]['kp'] for i in range(16)],
                'command_kd':[commands[i][0]['kd'] for i in range(16)],
                'command_source_monotonic_ns':[commands[i][1] for i in range(16)],
                'command_max_age_s':max(command_ages),
                'wall_ns':time.time_ns(), 'monotonic_ns':now}

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=2)
        if self.thread.is_alive(): raise RuntimeError('CAN observer did not stop')
        for sock in self.sockets: sock.close()
        self.selector.close()
        try:
            self.log.close()
        except RuntimeError as exc:
            self.error = self.error or str(exc)
            raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    telemetry=Telemetry(a.output/'raw_can.jsonl')
    snapshots=[]; error=None
    try:
        for _ in range(60):
            time.sleep(.05)
            try: snapshots.append(telemetry.snapshot())
            except RuntimeError as exc: error=str(exc)
        if not snapshots: raise RuntimeError(error or 'no feedback')
        write_json(a.output/'start_state.json', snapshots[-1]['positions_rad'])
        write_json(a.output/'feedback.json', snapshots)
        print(json.dumps({'receive_only': True, 'samples':len(snapshots), **snapshots[-1]}))
    finally:
        telemetry.close()
        write_json(a.output/'manifest.json', {'receive_only':True, 'frames':telemetry.count,
                                             'error':telemetry.error,'socket_drops':telemetry.drops})
