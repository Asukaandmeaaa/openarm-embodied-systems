# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish source-stamped JPEGs from a continuously drained latest-frame slot.

``source_monotonic_ns`` is sampled immediately *before* OpenCV's read, so time
blocked in acquisition and spent resizing/encoding counts toward frame age.
It is a host acquisition/dequeue timestamp, not a sensor exposure timestamp.
We request CAP_PROP_BUFFERSIZE=1, but camera backends may ignore that setting;
neither hardware/driver buffering nor repeated images from faulty firmware can
be detected from this host timestamp alone. All consumers using it must run on
the same host (and therefore share the monotonic clock).
"""

from dataclasses import dataclass
import logging
import os
import threading
import time

import cv2
import dora
import pyarrow as pa


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquiredFrame:
    """An image and its original acquisition sequence/time."""

    sequence: int
    source_monotonic_ns: int
    image: object


class LatestFrameCapture:
    """Continuously drain capture into one slot without queuing old frames."""

    def __init__(self, capture, clock_ns=time.monotonic_ns):
        """Take ownership of an already configured OpenCV capture object."""
        self._capture = capture
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._sequence = 0
        self._read_failed = False
        # A backend stuck forever in read/release must not prevent process exit.
        self._thread = threading.Thread(
            target=self._run, name="openarm-camera-capture", daemon=True
        )

    def start(self):
        """Start the sole thread that reads and releases the capture."""
        self._thread.start()

    def _capture_once(self):
        source_monotonic_ns = self._clock_ns()
        success, frame = self._capture.read()
        if not success or frame is None:
            with self._lock:
                self._latest = None
            if not self._read_failed:
                LOGGER.warning("CAPTURE_READ_FAILED: no new camera frame")
            self._read_failed = True
            return False
        if self._read_failed:
            LOGGER.info("CAPTURE_RECOVERED: new camera frames available")
        self._read_failed = False
        self._sequence += 1
        with self._lock:
            self._latest = AcquiredFrame(
                self._sequence, source_monotonic_ns, frame
            )
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self._capture_once():
                    self._stop.wait(0.02)
        except Exception:
            with self._lock:
                self._latest = None
            LOGGER.exception("CAPTURE_STOPPED: camera acquisition raised an error")
        finally:
            self._capture.release()

    def latest_after(self, sequence):
        """Return only an acquired frame newer than the last published one."""
        with self._lock:
            frame = self._latest
            if frame is None or frame.sequence <= sequence:
                return None
            return frame

    def close(self, timeout=0.5):
        """Request shutdown and wait a bounded time for backend cleanup."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            LOGGER.warning("CAPTURE_CLOSE_TIMEOUT: capture backend still blocked")
            return False
        return True


def publish_latest(node, source, previous_sequence, width, height, jpeg_quality):
    """Publish a new frame once, retaining its stamp through JPEG encoding."""
    frame = source.latest_after(previous_sequence)
    if frame is None:
        return previous_sequence
    image = frame.image
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not success:
        raise RuntimeError("Failed to JPEG encode camera frame")
    node.send_output(
        "image",
        pa.array(encoded.ravel(), type=pa.uint8()),
        {
            "source_monotonic_ns": frame.source_monotonic_ns,
            "source_sequence": frame.sequence,
            "encoding": "jpeg",
            "width": width,
            "height": height,
        },
    )
    return frame.sequence


def main():
    """Run a JPEG camera node with host acquisition timestamps."""
    logging.basicConfig(level=logging.INFO)
    width = int(os.getenv("IMAGE_WIDTH", "640"))
    height = int(os.getenv("IMAGE_HEIGHT", "360"))
    jpeg_quality = int(os.getenv("JPEG_QUALITY", "90"))
    if width <= 0 or height <= 0 or not 0 <= jpeg_quality <= 100:
        raise ValueError("Invalid IMAGE_WIDTH, IMAGE_HEIGHT, or JPEG_QUALITY")
    if os.getenv("ENCODING", "jpeg").lower() not in ("jpeg", "jpg"):
        raise ValueError("dora-openarm-camera only supports JPEG encoding")
    capture_path = os.getenv("CAPTURE_PATH", "0")
    device = int(capture_path) if capture_path.isdecimal() else capture_path
    node = dora.Node()
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open camera: {capture_path}")
    if not capture.set(cv2.CAP_PROP_BUFFERSIZE, 1):
        LOGGER.warning("Camera backend did not accept CAP_PROP_BUFFERSIZE=1")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    source = LatestFrameCapture(capture)
    source.start()
    previous_sequence = 0
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "tick":
                previous_sequence = publish_latest(
                    node, source, previous_sequence, width, height, jpeg_quality
                )
    finally:
        source.close()


if __name__ == "__main__":
    main()
