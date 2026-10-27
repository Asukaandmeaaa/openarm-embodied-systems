"""DECXIN wrist RGB publisher with bounded buffering and JPEG transport."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String


def _fill_stamp(stamp, nanoseconds: int) -> None:
    stamp.sec = int(nanoseconds // 1_000_000_000)
    stamp.nanosec = int(nanoseconds % 1_000_000_000)


def load_camera_info(path: str, width: int, height: int) -> CameraInfo:
    message = CameraInfo()
    message.width = int(width)
    message.height = int(height)
    if not path:
        message.distortion_model = "plumb_bob"
        return message
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    message.width = int(config.get("image_width", width))
    message.height = int(config.get("image_height", height))
    message.distortion_model = str(config.get("distortion_model", "plumb_bob"))
    message.d = [float(v) for v in config["distortion_coefficients"]["data"]]
    message.k = [float(v) for v in config["camera_matrix"]["data"]]
    message.r = [float(v) for v in config["rectification_matrix"]["data"]]
    message.p = [float(v) for v in config["projection_matrix"]["data"]]
    return message


class CaptureThread:
    def __init__(self, device: str, width: int, height: int, fps: float, fourcc: str = "MJPG"):
        if fourcc not in {"MJPG", "YUYV"}:
            raise ValueError("capture_fourcc must be MJPG or YUYV")
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.capture_wall_ns = 0
        self.capture_monotonic_s = 0.0
        self.sequence = 0
        self.failures = 0
        self.error: str | None = None
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open wrist camera: {device}")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        if actual_fourcc != cv2.VideoWriter_fourcc(*fourcc):
            self.capture.release()
            raise RuntimeError(f"camera format negotiation failed: required {fourcc}, got {actual_fourcc}")
        if (actual_width, actual_height) != (width, height):
            self.capture.release()
            raise RuntimeError(
                f"wrist camera negotiated {actual_width}x{actual_height}, "
                f"required {width}x{height}"
            )
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        consecutive_failures = 0
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            wall_ns = time.time_ns()
            monotonic_s = time.monotonic()
            if not ok or frame is None:
                self.failures += 1
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    self.error = "30 consecutive camera read failures"
                    break
                continue
            consecutive_failures = 0
            with self.lock:
                self.frame = frame
                self.capture_wall_ns = wall_ns
                self.capture_monotonic_s = monotonic_s
                self.sequence += 1

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None
            return (
                self.sequence,
                self.capture_wall_ns,
                self.capture_monotonic_s,
                self.frame.copy(),
            )

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.capture.release()


class WristCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("wrist_right_camera")
        self.declare_parameter(
            "device",
            "/dev/v4l/by-id/usb-DCX-250107-ZW_DECXIN-video-index0",
        )
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("capture_fps", 60.0)
        self.declare_parameter("capture_fourcc", "MJPG")
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("publish_raw", False)
        self.declare_parameter(
            "image_topic", "/camera/wrist_right/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/wrist_right/camera_info"
        )
        self.declare_parameter(
            "frame_id", "wrist_right_camera_optical_frame"
        )
        self.declare_parameter("camera_info_file", "")

        device = str(self.get_parameter("device").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        capture_fps = float(self.get_parameter("capture_fps").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be in [1, 100]")
        if publish_rate <= 0.0 or publish_rate > capture_fps:
            raise ValueError("publish_rate must be positive and <= capture_fps")
        self.frame_id = str(self.get_parameter("frame_id").value)
        image_topic = str(self.get_parameter("image_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_info = load_camera_info(
            str(self.get_parameter("camera_info_file").value), width, height
        )

        self.compressed_pub = self.create_publisher(
            CompressedImage, f"{image_topic}/compressed", qos_profile_sensor_data
        )
        self.raw_pub = (
            self.create_publisher(Image, image_topic, qos_profile_sensor_data)
            if self.publish_raw
            else None
        )
        info_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.info_pub = self.create_publisher(CameraInfo, camera_info_topic, info_qos)
        self.status_pub = self.create_publisher(
            String, "/camera/wrist_right/status", info_qos
        )
        self.capture = CaptureThread(device, width, height, capture_fps,
                                     str(self.get_parameter("capture_fourcc").value))
        self.last_sequence = 0
        self.published = 0
        self.skipped = 0
        self.started = time.monotonic()
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_latest)
        self.status_timer = self.create_timer(1.0, self.publish_status)
        calibration = "calibrated" if self.camera_info.k[0] != 0.0 else "UNCALIBRATED"
        self.get_logger().info(
            f"DECXIN wrist camera {device}: {width}x{height}, capture={capture_fps:.1f} "
            f"publish={publish_rate:.1f}, JPEG quality={self.jpeg_quality}, {calibration}"
        )

    def publish_latest(self) -> None:
        latest = self.capture.latest()
        if latest is None:
            return
        sequence, wall_ns, _monotonic_s, frame = latest
        if sequence == self.last_sequence:
            return
        if self.last_sequence:
            self.skipped += max(0, sequence - self.last_sequence - 1)
        self.last_sequence = sequence
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return

        compressed = CompressedImage()
        _fill_stamp(compressed.header.stamp, wall_ns)
        compressed.header.frame_id = self.frame_id
        compressed.format = "jpeg"
        compressed.data = jpeg.tobytes()
        self.compressed_pub.publish(compressed)

        self.camera_info.header = compressed.header
        self.info_pub.publish(self.camera_info)

        if self.raw_pub is not None:
            raw = Image()
            raw.header = compressed.header
            raw.height, raw.width = frame.shape[:2]
            raw.encoding = "bgr8"
            raw.is_bigendian = False
            raw.step = raw.width * 3
            raw.data = frame.tobytes()
            self.raw_pub.publish(raw)
        self.published += 1

    def publish_status(self) -> None:
        elapsed = max(1e-6, time.monotonic() - self.started)
        message = String()
        message.data = json.dumps(
            {
                "captured_sequence": self.capture.sequence,
                "published": self.published,
                "publish_average_hz": self.published / elapsed,
                "skipped_capture_frames": self.skipped,
                "read_failures": self.capture.failures,
                "capture_error": self.capture.error,
                "calibrated": bool(self.camera_info.k[0] != 0.0),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self.status_pub.publish(message)

    def destroy_node(self):
        self.capture.close()
        return super().destroy_node()


def main() -> int:
    rclpy.init()
    node = WristCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
