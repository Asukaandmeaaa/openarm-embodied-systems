#!/usr/bin/env python3
"""Trial camera with explicit capture negotiation and original Dora JPEG output.

CAPTURE_WIDTH/HEIGHT/FPS/FOURCC select the sensor acquisition mode independently
of IMAGE_WIDTH/HEIGHT (640x360 by default). No exposure controls are changed.
Only this process owns camera capture; it has no ROS/CAN/motor control imports.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

from logged_policy_server import AuditLog, ROOT, sha_file

ORIGINAL = ROOT / "official/dora-openarm-evaluation/nodes/dora-openarm-observer/src/dora_openarm_observer/camera.py"
ROS_CAMERA = ROOT.parent / "openarm_kinesthetic_ws/ros2_ws/src/openarm_kinesthetic_recorder/openarm_kinesthetic_recorder/wrist_camera_node.py"


def capture_settings(env=os.environ):
    required = ("CAPTURE_WIDTH", "CAPTURE_HEIGHT", "CAPTURE_FPS", "CAPTURE_FOURCC")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError("explicit sensor capture mode required: " + ",".join(missing))
    result = {"width": int(env["CAPTURE_WIDTH"]), "height": int(env["CAPTURE_HEIGHT"]),
              "fps": float(env["CAPTURE_FPS"]), "fourcc": env["CAPTURE_FOURCC"]}
    if (result["width"] <= 0 or result["height"] <= 0 or not math.isfinite(result["fps"])
            or result["fps"] <= 0 or len(result["fourcc"]) != 4 or not result["fourcc"].isascii()):
        raise ValueError("invalid requested capture mode")
    return result


def decode_fourcc(value):
    numeric = int(value)
    return "".join(chr((numeric >> (8 * index)) & 255) for index in range(4))


def configure_capture(capture, cv2, requested):
    """Mirror ROS CaptureThread order and verify negotiated mode before reading."""
    operations = [
        ("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested["fourcc"])),
        ("width", cv2.CAP_PROP_FRAME_WIDTH, requested["width"]),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT, requested["height"]),
        ("fps", cv2.CAP_PROP_FPS, requested["fps"]),
        ("buffer_size", cv2.CAP_PROP_BUFFERSIZE, 1),
    ]
    accepted = {name: bool(capture.set(prop, value)) for name, prop, value in operations}
    actual = {"width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
              "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
              "fps": capture.get(cv2.CAP_PROP_FPS),
              "fourcc": decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))}
    if any(not math.isfinite(actual[name]) for name in ("width", "height", "fps")):
        raise ValueError(f"camera returned nonfinite negotiated mode: {actual}")
    # A nominal 60 Hz mode may report 59.94 Hz. This accommodates representation
    # of that nominal rate, never a fallback to a slower 30/15 Hz capture mode.
    fps_tolerance_hz = 0.1
    if (actual["width"] != requested["width"] or actual["height"] != requested["height"]
            or actual["fourcc"] != requested["fourcc"]
            or abs(actual["fps"] - requested["fps"]) > fps_tolerance_hz):
        raise ValueError(f"capture mode negotiation mismatch: requested={requested}, actual={actual}, set_results={accepted}")
    return {"requested": requested, "actual": actual, "set_results": accepted,
            "fps_nominal_representation_tolerance_hz": fps_tolerance_hz,
            "exposure_settings_changed": False,
            "actual_fps_is_negotiated_metadata_not_measured_rate": True}


class HealthNode:
    def __init__(self, node, log, mode):
        self.node = node
        self.log = log
        self.mode = mode
        self.last_source_ns = None
        self.last_sequence = None
        self.frames = 0

    def send_output(self, name, value, metadata=None):
        now = time.monotonic_ns()
        metadata = dict(metadata or {})
        source = int(metadata["source_monotonic_ns"])
        sequence = int(metadata["source_sequence"])
        source_interval = None if self.last_source_ns is None else source - self.last_source_ns
        sequence_delta = None if self.last_sequence is None else sequence - self.last_sequence
        self.last_source_ns, self.last_sequence = source, sequence
        self.frames += 1
        self.log.write("camera_frame_published", source_monotonic_ns=source, source_sequence=sequence,
                       publish_monotonic_ns=now, source_age_ns=now-source,
                       source_interval_ns=source_interval, source_sequence_delta=sequence_delta,
                       jpeg_bytes=len(value), jpeg_sha256=hashlib.sha256(value.to_numpy().tobytes()).hexdigest(),
                       original_output_metadata=metadata)
        # Original source stamp, RGB resizing/JPEG values, and metadata remain
        # unchanged. Negotiated mode and per-frame health live in the side log.
        return self.node.send_output(name, value, metadata)


def load_camera():
    spec = importlib.util.spec_from_file_location("_act_trial_original_camera", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_realsense():
    """Import the installed collection SDK only; never enumerate/open devices."""
    if importlib.util.find_spec('pyrealsense2') is None:
        package = ROOT/'vision_grasp_baseline/.venv/lib/python3.12/site-packages/pyrealsense2'
        spec = importlib.util.spec_from_file_location('pyrealsense2', package/'__init__.py',
                                                     submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['pyrealsense2'] = module
        spec.loader.exec_module(module)
    import pyrealsense2
    return pyrealsense2


class RealSenseCapture:
    """Collection-matched BGR frames; camera-only LatestFrameCapture adapter."""
    def __init__(self, serial, requested):
        rs = load_realsense()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        if (requested['width'], requested['height'], requested['fps']) != (640, 480, 30):
            raise ValueError('closed-transit RealSense requires native 640x480 at 30 Hz')
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        try:
            if profile.get_device().get_info(rs.camera_info.serial_number) != serial:
                raise ValueError('RealSense identity mismatch')
        except BaseException:
            self.pipeline.stop()
            raise

    def read(self):
        import numpy as np
        frame = self.pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
        return (True, np.asanyarray(frame.get_data()).copy()) if frame else (False, None)

    def release(self):
        self.pipeline.stop()


def main():
    logging.basicConfig(level=logging.INFO)
    requested = capture_settings()
    width = int(os.getenv("IMAGE_WIDTH", "640"))
    height = int(os.getenv("IMAGE_HEIGHT", "360"))
    quality = int(os.getenv("JPEG_QUALITY", "90"))
    if width <= 0 or height <= 0 or not 0 <= quality <= 100:
        raise ValueError("invalid output image dimensions/JPEG quality")
    if os.getenv("ENCODING", "jpeg").lower() not in ("jpeg", "jpg"):
        raise ValueError("original Dora camera output must remain JPEG")
    device_path = os.environ["CAPTURE_PATH"]
    device = int(device_path) if device_path.isdecimal() else device_path
    camera_role = os.getenv("CAMERA_ROLE", "camera")
    if not camera_role.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid CAMERA_ROLE")
    log = AuditLog(camera_role)
    capture = source = health = None
    try:
        original = load_camera()
        node = original.dora.Node()
        original.cv2.setNumThreads(1)
        if device_path.startswith('realsense:'):
            from scripts.act_trial.closed_contract import CEILING_SERIAL
            if device_path != 'realsense:'+CEILING_SERIAL:
                raise ValueError('camera differs from closed-transit collection serial')
            capture = RealSenseCapture(CEILING_SERIAL, requested)
            mode = {'backend': 'librealsense', 'serial': CEILING_SERIAL,
                    'requested': requested, 'color_format': 'bgr8', 'depth_enabled_as_in_collection': True}
        else:
            capture = original.cv2.VideoCapture(device, original.cv2.CAP_V4L2)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera {device_path}")
            mode = configure_capture(capture, original.cv2, requested)
        sources = {str(ORIGINAL): sha_file(ORIGINAL), str(Path(__file__)): sha_file(__file__)}
        if ROS_CAMERA.is_file():
            sources[str(ROS_CAMERA)] = sha_file(ROS_CAMERA)
        log.write("camera_configured", device=device_path, resolved_device=str(Path(device_path).resolve()),
                  capture=mode, output={"width": width, "height": height, "encoding": "jpeg", "quality": quality},
                  source_sha256=sources, algorithms="original LatestFrameCapture and publish_latest unchanged",
                  timestamp_scope="host pre-read timestamp; not sensor exposure time or proof of no firmware buffering")
        print(json.dumps({"camera_role": camera_role, "device": device_path, "capture": mode}), flush=True)
        source = original.LatestFrameCapture(capture)
        source.start()
        health = HealthNode(node, log, mode)
        previous_sequence = 0
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "tick":
                previous_sequence = original.publish_latest(health, source, previous_sequence, width, height, quality)
    except BaseException as exc:
        log.write("camera_exit", exception_type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        closed = source.close() if source is not None else None
        if source is None and capture is not None:
            capture.release()
        log.write("camera_closed", frames=health.frames if health else 0, capture_thread_closed=closed)
        log.close()


if __name__ == "__main__":
    main()
