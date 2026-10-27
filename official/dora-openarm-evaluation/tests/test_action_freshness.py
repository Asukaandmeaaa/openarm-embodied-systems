"""No-hardware tests for end-to-end action freshness and fail-closed expiry."""

import asyncio
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_actions_executor.main import (
    _main_executor,
    _validate_action_timing,
)
from dora_openarm_local_policy_server.main import _validate_observation_timing
from src.observation_timing import observation_deadline_ns


ROOT = Path(__file__).resolve().parents[1]


class FakeNode:
    """Capture Dora outputs without opening a transport or CAN interface."""

    def __init__(self):
        self.outputs = []

    def send_output(self, output_id, value, metadata=None):
        self.outputs.append((output_id, value, metadata or {}))


def timing_metadata(observation_ns, source_age_ns=50_000_000, skew_ns=10_000_000):
    return {
        "observation_monotonic_ns": observation_ns,
        "source_age_ns": source_age_ns,
        "source_skew_ns": skew_ns,
    }


def action_event(metadata, *, interval=100_000_000, reset=True, value=None):
    if value is None:
        value = [[0.0] * 8, [0.1] * 8]
    return {
        "metadata": {"interval": interval, "reset": reset, **metadata},
        "value": pa.array(value, type=pa.list_(pa.float32())),
    }


def test_downstream_validation_preserves_assembly_source_age():
    metadata = timing_metadata(1_000_000_000)
    policy_timing = _validate_observation_timing(metadata, 1_300_000_000)
    action_timing = _validate_action_timing(policy_timing, 1_300_000_000)
    assert policy_timing == metadata
    assert action_timing == metadata
    assert observation_deadline_ns(action_timing) == 1_500_000_000


def test_delayed_policy_response_is_rejected_by_executor():
    metadata = timing_metadata(1_000_000_000)
    # The policy accepted this at 300 ms age, but inference/transport took the
    # response past the original 500 ms observation deadline.
    _validate_observation_timing(metadata, 1_300_000_000)
    with pytest.raises(ValueError, match="observation stale"):
        _validate_action_timing(metadata, 1_501_000_000)


async def _running_chunk_expires_at_original_observation_deadline():
    node = FakeNode()
    events = asyncio.Queue()
    metadata = timing_metadata(time.monotonic_ns() - 450_000_000)
    await events.put(action_event(metadata, interval=100_000_000))
    task = asyncio.create_task(
        _main_executor(node, events, ["right"], False, False, 100)
    )
    try:
        # The first point is emitted, then the 100 ms step crosses the 500 ms
        # deadline.  A full chunk must never run after that deadline.
        await asyncio.sleep(0.18)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    output_ids = [output[0] for output in node.outputs]
    assert output_ids.count("move_position_right") == 1
    assert output_ids.count("timing_fault") == 1


def test_running_chunk_expires_at_original_observation_deadline():
    asyncio.run(_running_chunk_expires_at_original_observation_deadline())


async def _stale_replacement_cancels_old_tail_and_latches_fault():
    node = FakeNode()
    events = asyncio.Queue()
    fresh = timing_metadata(time.monotonic_ns() - 100_000_000)
    stale = timing_metadata(time.monotonic_ns() - 600_000_000)
    await events.put(action_event(fresh, interval=100_000_000))
    task = asyncio.create_task(
        _main_executor(node, events, ["right"], False, False, 100)
    )
    try:
        await asyncio.sleep(0.02)
        await events.put(action_event(stale, interval=100_000_000, reset=False))
        await asyncio.sleep(0.15)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    output_ids = [output[0] for output in node.outputs]
    assert output_ids.count("move_position_right") == 1
    assert output_ids.count("timing_fault") == 1


def test_stale_replacement_cancels_old_tail_and_latches_fault():
    asyncio.run(_stale_replacement_cancels_old_tail_and_latches_fault())


def test_upstream_policy_timing_fault_reaches_executor_latch():
    async def run():
        node = FakeNode()
        events = asyncio.Queue()
        await events.put(
            {
                "id": "timing_fault",
                "value": pa.array(["policy response was delayed"], type=pa.string()),
                "metadata": {"fault": "observation_timing"},
            }
        )
        task = asyncio.create_task(
            _main_executor(node, events, ["right"], False, False, 100)
        )
        try:
            await asyncio.sleep(0.02)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert [output[0] for output in node.outputs] == ["timing_fault"]

    asyncio.run(run())


def test_guard_latches_timing_fault_without_hardware():
    spec = importlib.util.spec_from_file_location(
        "guarded_home", ROOT / "nodes/dora-openarm/src/dora_openarm/guarded_home.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((ROOT / "guarded-home-right.json").read_text())
    guard = module.Guard(cfg, np.zeros(8), 0.0, hold_only=True)
    guard.timing_fault("running action chunk exceeded observation age limit")
    assert guard.state == "home_error"
    assert "action timing fault" in guard.error
