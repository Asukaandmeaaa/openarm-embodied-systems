"""Negative tests for the Dora observation freshness contract."""

import time

import pyarrow as pa
import pytest

from dora_openarm_actions_executor.main import _validate_action_timing
from dora_openarm_observer.main import (
    _arm_position_array,
    event_source_timestamp_ns,
    validate_source_timing,
)
from src.observation_timing import validate_timing


def _metadata(age_ns=100_000_000, source_age_ns=50_000_000, skew_ns=10_000_000):
    return {
        "observation_monotonic_ns": time.monotonic_ns() - age_ns,
        "source_age_ns": source_age_ns,
        "source_skew_ns": skew_ns,
    }


def test_observer_accepts_fresh_synchronized_sources():
    now = 1_000_000_000
    age, skew = validate_source_timing(
        {"arm_right": 950_000_000, "camera_ceiling": 940_000_000},
        ["arm_right", "camera_ceiling"],
        now,
        200_000_000,
        100_000_000,
    )
    assert age == 60_000_000
    assert skew == 10_000_000


def test_observer_uses_event_timestamp_instead_of_receipt_time():
    assert event_source_timestamp_ns({"metadata": {"timestamp": 1_700_000_000}}) == (
        "epoch", 1_700_000_000_000_000_000
    )
    assert event_source_timestamp_ns({"metadata": {"source_monotonic_ns": 42}}) == (
        "monotonic", 42
    )
    assert event_source_timestamp_ns({"metadata": {}}) is None


def test_observer_normalizes_real_openarm_qpos_struct():
    value = pa.array([{"qpos": [1.0] * 8}], type=pa.struct([
        ("qpos", pa.list_(pa.float32())),
    ]))
    result = _arm_position_array({"value": value})
    assert result.to_pylist() == [1.0] * 8


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ({"arm_right": 700_000_000, "camera_ceiling": 940_000_000}, "stale"),
        ({"arm_right": 950_000_000, "camera_ceiling": 800_000_000}, "unsynchronized"),
    ],
)
def test_observer_rejects_stale_or_skewed_sources(times, message):
    with pytest.raises(ValueError, match=message):
        validate_source_timing(
            times,
            ["arm_right", "camera_ceiling"],
            1_000_000_000,
            200_000_000,
            100_000_000,
        )


def test_policy_server_requires_source_timing():
    with pytest.raises(ValueError, match="missing source timing"):
        validate_timing({}, time.monotonic_ns())


def test_policy_server_rejects_old_observation():
    with pytest.raises(ValueError, match="observation stale"):
        validate_timing(
            _metadata(age_ns=600_000_000), time.monotonic_ns()
        )


def test_executor_rejects_old_observation():
    with pytest.raises(ValueError, match="observation stale"):
        _validate_action_timing(_metadata(age_ns=600_000_000))


def test_executor_accepts_fresh_observation():
    _validate_action_timing(_metadata())
