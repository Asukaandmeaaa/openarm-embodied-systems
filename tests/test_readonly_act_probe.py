import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "readonly_act_probe", Path(__file__).parents[1] / "src/readonly_act_probe.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class Link:
    def __init__(self, responses):
        self.responses = responses


class Motor:
    def __init__(self, enabled=False, error=0):
        self.enabled, self.error = enabled, error

    def is_enabled(self):
        return self.enabled

    def get_error_code(self):
        return self.error


class Collection:
    def __init__(self):
        self.links = [Link(0) for _ in range(8)]
        self.motors = [Motor() for _ in range(8)]

    def get_link_stats(self, i):
        return self.links[i]

    def get_motors(self):
        return self.motors


class OpenArm:
    def __init__(self, collection):
        self.collection = collection

    def get_arm(self):
        return self.collection


class FakeDriver:
    def __init__(self, enabled=False):
        self.collection = Collection()
        for motor in self.collection.motors:
            motor.enabled = enabled
        self.openarm = OpenArm(self.collection)
        self.fetch_calls = 0
        self.motion_calls = []

    def fetch_position(self, refresh=True):
        assert refresh is True
        self.fetch_calls += 1
        for link in self.collection.links:
            link.responses += 1
        return np.zeros(8, dtype=np.float32)

    def get_health(self):
        return (["OK"] * 8, {"carrier": True, **{name: 0 for name in mod.FAULTS}})

    def start(self):
        self.motion_calls.append("start")

    def stop(self):
        self.motion_calls.append("stop")

    def send_position(self, _):
        self.motion_calls.append("send_position")


def test_feedback_requires_fresh_disabled_axes_and_never_motion_calls():
    right, left = FakeDriver(), FakeDriver()
    result = mod.ReadonlyFeedbackProbe([right, left]).poll(now=1.0)
    assert result["status"] == "ready"
    assert result["arm_right"].shape == (8,)
    assert right.motion_calls == left.motion_calls == []


def test_feedback_rejects_enabled_motor():
    with pytest.raises(RuntimeError, match="feedback gate failed"):
        mod.ReadonlyFeedbackProbe([FakeDriver(enabled=True), FakeDriver()]).poll(now=1.0)


def test_feedback_poll_is_capped_at_20_hz():
    right, left = FakeDriver(), FakeDriver()
    probe = mod.ReadonlyFeedbackProbe([right, left])
    assert probe.poll(now=1.0) is not None
    assert probe.poll(now=1.01) is None
    assert right.fetch_calls == left.fetch_calls == 1


def test_action_audit_accepts_only_finite_50x16_and_writes_no_motion():
    audit = mod.ActionAudit()
    assert audit.ingest(np.zeros((50, 16)), {"source_monotonic_ns": 1_000_000_000}, 1_100_000_000)
    assert not audit.ingest(np.zeros((10, 16)))
    assert not audit.ingest(np.full((50, 16), np.nan))
    report = audit.report()
    assert report["accepted"] == 1
    assert report["rejected"] == 2
    assert report["ok"] is False


def test_action_audit_rejects_stale_timestamp():
    audit = mod.ActionAudit(max_age_s=0.5)
    assert not audit.ingest(np.zeros((50, 16)), {"source_monotonic_ns": 0}, 1_000_000_000)
    assert audit.report()["rejected"] == 1
