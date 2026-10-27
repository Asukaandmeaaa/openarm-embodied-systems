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

"""Camera source timing tests using fake capture devices only."""

import threading

import cv2
import numpy as np

from dora_openarm_observer.camera import LatestFrameCapture, publish_latest


class FakeNode:
    """Record output without starting a Dora graph."""

    def __init__(self):
        """Initialize output storage."""
        self.outputs = []

    def send_output(self, *args):
        """Record output arguments."""
        self.outputs.append(args)


class FakeCapture:
    """Supply a bounded sequence of synthetic frame reads."""

    def __init__(self, reads):
        """Keep supplied read results."""
        self.reads = iter(reads)

    def read(self):
        """Return the next simulated read result."""
        return next(self.reads)


def test_capture_time_survives_read_encode_and_publish(monkeypatch):
    """Blocking read and encode time must not renew the source timestamp."""
    now = [100]
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    class SlowCapture:
        def read(self):
            now[0] = 200
            return True, image

    source = LatestFrameCapture(SlowCapture(), clock_ns=lambda: now[0])
    assert source._capture_once()

    def slow_encode(*args):
        now[0] = 300
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", slow_encode)
    node = FakeNode()
    assert publish_latest(node, source, 0, 3, 2, 90) == 1
    assert now[0] == 300
    assert node.outputs[0][2]["source_monotonic_ns"] == 100
    assert node.outputs[0][2]["source_sequence"] == 1


def test_ticks_do_not_republish_a_cached_frame():
    """A frozen source produces no fresh-looking duplicate output."""
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    source = LatestFrameCapture(FakeCapture([(True, image)]), clock_ns=lambda: 100)
    source._capture_once()
    node = FakeNode()
    sequence = publish_latest(node, source, 0, 3, 2, 90)
    for _ in range(3):
        assert publish_latest(node, source, sequence, 3, 2, 90) == sequence
    assert len(node.outputs) == 1


def test_capture_drops_backlog_and_preserves_latest_sequence():
    """Several reads between ticks yield only the newest acquired frame."""
    images = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (0, 100, 200)]
    timestamps = iter((100, 200, 300))
    source = LatestFrameCapture(
        FakeCapture([(True, image) for image in images]),
        clock_ns=lambda: next(timestamps),
    )
    for _ in images:
        source._capture_once()
    node = FakeNode()
    assert publish_latest(node, source, 0, 640, 360, 90) == 3
    _, encoded, metadata = node.outputs[0]
    assert metadata["source_monotonic_ns"] == 300
    decoded = cv2.imdecode(encoded.to_numpy(), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)
    assert np.mean(decoded) == 200


def test_read_failure_does_not_publish_last_frame():
    """A failed read clears the slot until another successful acquisition."""
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    source = LatestFrameCapture(
        FakeCapture([(True, image), (False, None), (True, image)]),
        clock_ns=lambda: 100,
    )
    source._capture_once()
    assert not source._capture_once()
    node = FakeNode()
    assert publish_latest(node, source, 0, 3, 2, 90) == 0
    assert node.outputs == []
    assert source._capture_once()
    assert publish_latest(node, source, 0, 3, 2, 90) == 2


def test_close_returns_when_camera_read_is_blocked():
    """A stalled backend cannot hold shutdown past the bounded join."""
    entered = threading.Event()
    unblock = threading.Event()
    released = threading.Event()

    class BlockingCapture:
        def read(self):
            entered.set()
            unblock.wait()
            return False, None

        def release(self):
            released.set()

    source = LatestFrameCapture(BlockingCapture())
    source.start()
    try:
        assert entered.wait(timeout=1)
        assert source._thread.daemon
        assert not source.close(timeout=0.01)
    finally:
        unblock.set()
        assert released.wait(timeout=1)
        assert source.close()
