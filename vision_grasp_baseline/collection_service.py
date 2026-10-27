"""Receive-only ROS recorder and single-owner two-camera service.

RealSense RGB-D snapshots and ACT images share the same live pipeline. This
process never publishes joint commands or acquires motion authority.
"""
import argparse
from collections import deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import queue
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .collection_io import ORDER, CAMERAS, TARGET_STREAMS, assemble_actions, audit_episode, timing_summary, write_json


class DiskWriter:
    """Bounded lossless queue: image encoding and I/O never hold state locks."""
    def __init__(self):
        self.queue = queue.Queue(maxsize=4096)
        self.error = None
        self.peak_pending = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, kind, destination, value):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.queue.put_nowait((kind, destination, value))
            self.peak_pending = max(self.peak_pending, self.queue.qsize())
        except queue.Full:
            raise RuntimeError('Recording disk queue overflow; no samples may be silently dropped')

    def run(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                kind, destination, value = item
                if kind == 'json':
                    destination.write(json.dumps(value, allow_nan=False)+'\n')
                elif not cv2.imwrite(str(destination), value, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise RuntimeError('Image write failed: '+str(destination))
        except BaseException as exc:
            self.error = str(exc) or type(exc).__name__

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None, timeout=2)
            self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Disk writer did not drain')
        if self.error:
            raise RuntimeError(self.error)


class Recorder:
    def __init__(self, benchmark_path=None):
        self.lock = threading.RLock()
        self.path = None
        self.files = {}
        self.metadata = {}
        self.error = None
        self.latest = {}
        self.camera_times = {key: deque(maxlen=150) for key in CAMERAS}
        self.raw_counts = {}
        self.last_write = {}
        self.parked_finger = None
        self.writer = None
        self.finalizing = False
        self.ros_time_mapping = None
        self.benchmark_path = benchmark_path
        self.benchmark_file = None
        if benchmark_path:
            for key in CAMERAS:
                (benchmark_path/'cameras'/key).mkdir(parents=True, exist_ok=False)
            self.benchmark_file = (benchmark_path/'camera_benchmark.jsonl').open('x')

    def fail(self, exc):
        with self.lock:
            if self.error is None:
                self.error = str(exc) or type(exc).__name__
                print('RECORDING_ERROR '+self.error, flush=True)

    def append(self, key, row):
        if not self.path:
            return
        stamp = row['timestamp_s']
        if not math.isfinite(stamp) or stamp < self.metadata['start_timestamp_s']:
            return
        # camera.jsonl interleaves independent cameras; audit each separately.
        clock_key = row.get('key', key)
        if stamp <= self.last_write.get(clock_key, -math.inf):
            raise RuntimeError('Non-increasing source timestamp: ' + clock_key)
        self.last_write[clock_key] = stamp
        self.writer.submit('json', self.files[key], row)

    def camera(self, key, image, row):
        with self.lock:
            self.camera_times[key].append(row['timestamp_s'])
            if not self.path:
                if self.benchmark_path:
                    name = f"cameras/{key}/{row['frame_number']:012d}.jpg"
                    if not cv2.imwrite(str(self.benchmark_path/name), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise RuntimeError('Benchmark image write failed')
                    self.benchmark_file.write(json.dumps(dict(row, key=key, path=name))+'\n')
                return
            name = f"cameras/{key}/{row['frame_number']:012d}.jpg"
            self.writer.submit('image', self.path/name, image)
            self.append('camera', dict(row, key=key, path=name))

    def state(self, key, positions, source_time, raw=None):
        with self.lock:
            if not positions or not all(math.isfinite(x) for x in positions):
                raise RuntimeError('Invalid joint values: ' + key)
            self.latest[key] = (source_time, positions)
            if raw is not None and self.path and key != 'joint_observation':
                self.writer.submit('json', self.files['controller_raw'], raw)
            if key == 'joint_observation':
                if self.path and abs(positions[-1]-self.parked_finger) > .03:
                    raise RuntimeError('Uncontrolled left gripper moved during collection')
                row = dict(timestamp_s=source_time, positions=positions, order=ORDER)
                if raw is not None:
                    row.update(source_timestamp_s=raw['timestamp_s'], ros_message=raw['message'],
                               receive_monotonic_s=raw['receive_monotonic_s'])
                self.append(key, row)
            if key in TARGET_STREAMS:
                self.append('target_'+key, dict(timestamp_s=source_time, positions=positions,
                    receive_monotonic_s=raw['receive_monotonic_s'] if raw else source_time))

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        for key in ('joint_observation', 'right_arm', 'right_gripper', 'left_arm'):
            if key not in self.latest:
                raise RuntimeError('Missing ROS stream: ' + key)
            age=now-self.latest[key][0]
            if not -.02 <= age <= .50:
                raise RuntimeError(f'Stale ROS stream: {key}, source age={age:.6f}s')
        return (self.latest['right_arm'][1] + self.latest['right_gripper'][1]
                + self.latest['left_arm'][1] + [self.parked_finger if self.parked_finger is not None
                                               else self.latest['joint_observation'][1][-1]])

    def health(self, require_ros=True):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.writer and self.writer.error:
                raise RuntimeError(self.writer.error)
            now = time.monotonic()
            stats = {key: timing_summary(list(ts), 25., .15) for key, ts in self.camera_times.items()}
            for key in CAMERAS:
                if (not stats[key]['passed'] or not self.camera_times[key]
                        or now-self.camera_times[key][-1] > .15):
                    raise RuntimeError(f'Camera not ready at 30 Hz: {key}: {stats[key]}')
            if require_ros:
                self.snapshot()
            return dict(ok=True, cameras=stats, recording=str(self.path) if self.path else None,
                ros_source_age_s={key: now-row[0] for key,row in self.latest.items()},
                pending_disk_jobs=self.writer.queue.qsize() if self.writer else 0)

    def begin(self, path):
        with self.lock:
            self.health()
            if self.path or self.finalizing:
                raise RuntimeError('Already recording')
            q = self.latest['joint_observation'][1]
            if max(abs(v) for v in q[:7]+q[8:15]) > .06:
                raise RuntimeError('Both arms must be at mechanical zero before recording')
            from .gripper_settings import (COLLECTION_POLICY,require_closed_transit,TRANSIT_POSITION,
                                           TRANSIT_TOLERANCE,TRANSIT_MIN_POSITION,TRANSIT_MAX_POSITION,
                                           CLOSE_POSITION,OPEN_POSITION)
            require_closed_transit(float(q[7]),float(self.latest['right_gripper'][1][0]))
            path = Path(path).resolve()
            path.mkdir(parents=True, exist_ok=False)
            for key in CAMERAS:
                (path/'cameras'/key).mkdir(parents=True)
            self.files = {key: (path/f'{key}.jsonl').open('x', buffering=1024*1024)
                          for key in ('joint_observation', 'demonstration_action', 'camera', 'controller_raw', 'events',
                                      *('target_'+key for key in TARGET_STREAMS))}
            self.writer = DiskWriter()
            self.last_write = {}
            self.parked_finger = q[-1]
            self.metadata = dict(schema_version=2, trial_id=path.name,
                gripper_policy=COLLECTION_POLICY,
                gripper_targets_rad=dict(empty_transit=TRANSIT_POSITION,grasp=CLOSE_POSITION,open=OPEN_POSITION),
                empty_transit_feedback_tolerance_rad=TRANSIT_TOLERANCE,
                empty_transit_feedback_range_rad=[TRANSIT_MIN_POSITION,TRANSIT_MAX_POSITION],
                start_timestamp_s=time.monotonic(), started_wall_ns=time.time_ns(),
                timestamp_basis='ROS header mapped to CLOCK_MONOTONIC with fixed offset; camera host receipt; all source headers retained',
                ros_qos='RELIABLE KEEP_LAST depth=256', ros_transport_liveness_limit_s=.5,
                ros_time_mapping=self.ros_time_mapping,
                hardware_camera_sync=False, joint_order=ORDER, target_fps=30,
                image_size_wh=[640, 480], control_update_hz_configured=100,
                action_source='JTC output.positions at native arm controller publish rate; measured Hz in quality.json',
                left_gripper_action_source='passive hardware hold: initial measured position, drift monitored; not an actuated training target',
                initial_state=q, review_status='pending', operator_marked_success=False,
                automatic_task_contract_passed=False)
            write_json(path/'metadata.json', self.metadata)
            self.path = path
            return dict(ok=True, path=str(path))

    def finish(self, success=False):
        with self.lock:
            if not self.path:
                raise RuntimeError('No active episode')
            if success:
                try:
                    self.snapshot()
                    q = self.latest['joint_observation'][1]
                    if max(abs(v) for v in q[:7]+q[8:15]) > .06:
                        raise RuntimeError('Final joint feedback is outside mechanical-zero tolerance')
                    from .gripper_settings import require_closed_transit
                    require_closed_transit(float(q[7]),float(self.latest['right_gripper'][1][0]))
                except (RuntimeError,ValueError) as exc:
                    self.fail(exc)
                    success = False
            end = time.monotonic()
            path = self.path
            self.path = None
            self.finalizing = True
            writer = self.writer
        # Keep ROS callbacks running while flushing images and auditing files.
        # Holding the state lock here would create a new ROS backlog every save.
        try:
            try:
                writer.close()
            except Exception as exc:
                self.fail(exc)
            self.metadata['disk_queue_peak_jobs'] = writer.peak_pending
            for stream in self.files.values():
                stream.close()
            try:
                self.metadata['action_join'] = assemble_actions(path, self.parked_finger)
            except Exception as exc:
                self.fail(exc)
            self.metadata.update(end_timestamp_s=end, finished_wall_ns=time.time_ns(),
                recording_error=self.error, automatic_task_contract_passed=bool(success),
                terminal_label='returned_zero' if success else 'failed_or_interrupted')
            write_json(path/'metadata.json', self.metadata)
            audit = audit_episode(path)
            write_json(path/'quality.json', audit)
            self.metadata['data_quality_passed'] = audit['passed']
            write_json(path/'metadata.json', self.metadata)
            return dict(ok=True, path=str(path), quality=audit)
        finally:
            with self.lock:
                self.finalizing = False


class Cameras:
    def __init__(self, recorder, serial, wrist):
        self.recorder = recorder
        self.serial, self.wrist = serial, wrist
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.frames = deque(maxlen=60)
        self.metadata = None
        self.threads = [threading.Thread(target=self.guarded, args=(fn,), daemon=True)
                        for fn in (self.realsense, self.usb)]
        for thread in self.threads:
            thread.start()

    def guarded(self, fn):
        try:
            fn()
        except BaseException as exc:
            self.recorder.fail(exc)

    def realsense(self):
        import importlib.metadata
        import pyrealsense2 as rs
        from .capture import intrinsics
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        try:
            cp, dp = profile.get_stream(rs.stream.color), profile.get_stream(rs.stream.depth)
            ex = dp.get_extrinsics_to(cp)
            metadata = dict(schema_version=1, serial=self.serial,
                name=profile.get_device().get_info(rs.camera_info.name),
                sdk_version=importlib.metadata.version('pyrealsense2'),
                depth_scale_m=profile.get_device().first_depth_sensor().get_depth_scale(),
                aligned_to='color', color_intrinsics=intrinsics(cp),
                depth_to_color_extrinsics=dict(rotation_column_major=list(ex.rotation), translation_m=list(ex.translation)),
                hardware_motion=False, shared_continuous_pipeline=True)
            align = rs.align(rs.stream.color)
            last = None
            for _ in range(45):
                pipeline.wait_for_frames(5000)
            while not self.stop.is_set():
                frames = pipeline.wait_for_frames(3000)
                receipt, wall = time.monotonic(), time.time_ns()
                frames = align.process(frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    raise RuntimeError('Incomplete RealSense RGB-D frame')
                number = color.get_frame_number()
                if number == last:
                    continue
                last = number
                bgr, z16 = np.asanyarray(color.get_data()).copy(), np.asanyarray(depth.get_data()).copy()
                row = dict(timestamp_s=receipt, wall_ns=wall, frame_number=number,
                    color_frame_number=number, color_timestamp_ms=color.get_timestamp(),
                    depth_timestamp_ms=depth.get_timestamp(), timestamp_domain=str(color.get_frame_timestamp_domain()),
                    valid_depth_fraction=float(np.mean(z16 > 0)))
                with self.condition:
                    self.metadata = dict(metadata, aligned_depth_intrinsics=intrinsics(depth.profile))
                    self.frames.append((row, bgr, z16))
                    self.condition.notify_all()
                self.recorder.camera('ceiling', bgr, row)
        finally:
            pipeline.stop()

    def usb(self):
        camera = cv2.VideoCapture(self.wrist, cv2.CAP_V4L2)
        try:
            if not camera.isOpened():
                raise RuntimeError('Cannot open wrist camera: ' + self.wrist)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            number = 0
            while not self.stop.is_set():
                ok, bgr = camera.read()
                stamp = time.monotonic()
                if not ok:
                    raise RuntimeError('Wrist camera read failed')
                if bgr.shape != (480, 640, 3):
                    raise RuntimeError('Unexpected wrist image size: ' + str(bgr.shape))
                number += 1
                self.recorder.camera('wrist_right', bgr, dict(timestamp_s=stamp,
                    wall_ns=time.time_ns(), frame_number=number,
                    timestamp_domain='host_receipt; software read sequence, no hardware counter'))
        finally:
            camera.release()

    def capture(self, output, samples=10):
        if not 1 <= samples <= 30:
            raise ValueError('samples must be between 1 and 30')
        start = time.monotonic()
        with self.condition:
            while True:
                if self.recorder.error:
                    raise RuntimeError(self.recorder.error)
                frames = [x for x in self.frames if x[0]['timestamp_s'] >= start]
                if len(frames) >= samples:
                    frames = frames[:samples]
                    metadata = deepcopy(self.metadata)
                    break
                if time.monotonic()-start > 10:
                    raise TimeoutError('Fresh RGB-D snapshot timed out')
                self.condition.wait(.1)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, (row, color, depth) in enumerate(frames):
            if not cv2.imwrite(str(output/f'color_{index:03d}.png'), color):
                raise RuntimeError('Snapshot image write failed')
            np.save(output/f'depth_{index:03d}.npy', depth)
            rows.append(dict(row, index=index))
        write_json(output/'capture.json', dict(metadata, frames=rows))
        return dict(ok=True, output=str(output))


def ros_observer(recorder):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState
    from control_msgs.msg import JointTrajectoryControllerState
    from rosidl_runtime_py.convert import message_to_ordereddict
    rclpy.init()
    node = rclpy.create_node('openarm_ownership_observer')
    # Live publishers were inspected: all four use RELIABLE. Depth=5 best-effort
    # used to discard ~200ms of samples during a brief executor scheduling pause.
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=256, reliability=ReliabilityPolicy.RELIABLE)
    ros_to_monotonic_ns = time.monotonic_ns()-node.get_clock().now().nanoseconds
    recorder.ros_time_mapping = dict(offset_ns=ros_to_monotonic_ns, rule='source_header_ns + offset_ns = source_monotonic_ns')

    def callback(message, key, expected):
        try:
            receipt = time.monotonic()
            source_ns = message.header.stamp.sec*10**9+message.header.stamp.nanosec
            stamp = (source_ns+ros_to_monotonic_ns)/1e9
            age = receipt-stamp
            if not -.005 <= age <= .50:
                raise RuntimeError(f'Stale/invalid ROS header: {key}: {age}')
            names = message.name if key == 'joint_observation' else message.joint_names
            positions = message.position if key == 'joint_observation' else message.output.positions
            if len(names) != len(positions) or len(set(names)) != len(names):
                raise RuntimeError('Invalid ROS position array: ' + key)
            values = dict(zip(names, positions))
            raw = dict(
                timestamp_s=stamp, receive_monotonic_s=receipt, controller=key,
                message=message_to_ordereddict(message))
            recorder.state(key, [float(values[n]) for n in expected], stamp, raw)
        except BaseException as exc:
            recorder.fail(exc)

    node.create_subscription(JointState, '/joint_states',
        lambda msg: callback(msg, 'joint_observation', ORDER), qos)
    for controller, key, names in (
        ('right_joint_trajectory_controller', 'right_arm', ORDER[:7]),
        ('left_joint_trajectory_controller', 'left_arm', ORDER[8:15]),
        ('right_gripper_controller', 'right_gripper', ORDER[7:8])):
        node.create_subscription(JointTrajectoryControllerState, f'/{controller}/controller_state',
            lambda msg, k=key, n=names: callback(msg, k, n), qos)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node, thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--serial', default='344322073972')
    parser.add_argument('--wrist', default='/dev/v4l/by-id/usb-DCX-250107-ZW_DECXIN-video-index0')
    parser.add_argument('--camera-only', action='store_true')
    parser.add_argument('--benchmark-dir', type=Path)
    args, _ = parser.parse_known_args()
    if args.benchmark_dir and not args.camera_only:
        parser.error('--benchmark-dir is only for the no-motion camera benchmark')
    recorder = Recorder(args.benchmark_dir)
    cameras = Cameras(recorder, args.serial, args.wrist)
    ros = None if args.camera_only else ros_observer(recorder)
    token = secrets.token_urlsafe(32)
    stopped = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer ' + token:
                self.send_error(403)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                if self.path == '/status':
                    result = recorder.health(require_ros=not args.camera_only)
                elif self.path == '/capture':
                    result = cameras.capture(**payload)
                elif self.path == '/begin' and not args.camera_only:
                    result = recorder.begin(**payload)
                    with recorder.lock:
                        recorder.metadata['camera_configuration'] = dict(
                            ceiling=deepcopy(cameras.metadata), wrist_right=dict(
                                path=args.wrist, requested_fps=30, requested_fourcc='MJPG', size_wh=[640, 480]))
                        write_json(recorder.path/'metadata.json', recorder.metadata)
                elif self.path == '/finish' and not args.camera_only:
                    result = recorder.finish(**payload)
                elif self.path == '/event':
                    with recorder.lock:
                        recorder.append('events', dict(timestamp_s=time.monotonic(), **payload))
                    result = dict(ok=True)
                else:
                    raise ValueError('Unknown operation')
            except BaseException as exc:
                result = dict(ok=False, error=str(exc))
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.timeout = .2
    args.ready.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.ready, dict(url=f'http://127.0.0.1:{server.server_port}', token=token))
    args.ready.chmod(0o600)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    try:
        while not stopped.is_set():
            server.handle_request()
    finally:
        server.server_close()
        cameras.stop.set()
        for thread in cameras.threads:
            thread.join(timeout=5)
        if recorder.path:
            recorder.fail('Service stopped before episode completion')
            recorder.finish(False)
        if recorder.benchmark_file:
            recorder.benchmark_file.close()
        if ros:
            import rclpy
            rclpy.try_shutdown()
            ros[1].join(timeout=2)
            ros[0].destroy_node()
        args.ready.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
