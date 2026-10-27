"""Behavioral regression for prefix completion, fresh inference and stop priority."""
import asyncio
from datetime import datetime
from collections import deque
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/act_trial"))
sys.path.insert(0, str(ROOT / "official/dora-openarm-evaluation/src"))
from action_queue import ACTION_STEPS, INTERVAL_NS, SCHEDULE, InferenceGate, prefix, sample_times
from logged_executor import LoggedNode, LoggedQueue, ORIGINAL
from logged_policy_server import load_source
from queued_executor import queued_async_main
import queued_policy_client as client


class Log:
    run_id = "test"
    def __init__(self): self.rows = []
    def write(self, event, **data): self.rows.append((event, data))


def timing(now_ns):
    return {"observation_monotonic_ns": now_ns, "source_age_ns": 5_000_000,
            "source_skew_ns": 2_000_000}


def actions_response(offset=0):
    return {"positions": [[float(offset + i)] * 16 for i in range(50)],
            "interval": INTERVAL_NS, "cutoff_hz": 15}


def test_only_first_ten_predictions_reach_interpolator():
    actions = actions_response()
    actual = prefix(actions)
    actions["positions"][10:] = [[float("nan")] * 16] * 40
    assert actual.shape == (10, 16)
    np.testing.assert_array_equal(actual[:, 0], np.arange(10))
    with pytest.raises(ValueError): prefix(actions)
    times, evaluation, duration = sample_times()
    assert duration == pytest.approx(10 / 30)
    assert evaluation.max() == pytest.approx(9 / 30)
    assert len(times) == 34


def test_completion_gate_never_uses_an_observation_buffered_before_completion():
    gate = InferenceGate()
    gate.dispatch("first")
    assert not gate.accepts(timing(9_000_000_000))
    with pytest.raises(ValueError): gate.dispatch("second")
    with pytest.raises(ValueError):
        gate.complete({"queue_chunk_id": "other"}, 9_000_000_000)
    gate.complete({"queue_chunk_id": "first", "executed_raw_steps": 10,
                   "completed_monotonic_ns": 2_000_000_000}, 2_000_000_000)
    assert not gate.accepts(timing(1_999_999_999))
    assert gate.accepts(timing(2_000_000_000))


def test_client_waits_for_ack_and_fresh_observation(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=1_000_000_000, monotonic_ns=lambda: clock.now)
    monkeypatch.setattr(client, "time", clock)
    def observation(identifier, stamp):
        return {"type": "INPUT", "id": "observation", "metadata": {**timing(stamp), "timestamp": datetime.now()},
                "value": pa.array([{"id": identifier, "position": [0.] * 16}])}
    sequence = [
        (1_000_000_000, observation(1, 1_000_000_000)),
        (1_100_000_000, observation(2, 1_100_000_000)),
        (1_340_000_000, {"type": "INPUT", "id": "chunk_complete", "metadata": {
            "queue_chunk_id": "test_1", "executed_raw_steps": 10,
            "completed_monotonic_ns": 1_333_333_330}}),
        (1_350_000_000, observation(3, 1_300_000_000)),
        (1_360_000_000, observation(4, 1_360_000_000)),
        (1_370_000_000, {"type": "INPUT", "id": "execution_stop"}),
    ]
    class Node:
        outputs = []
        def __iter__(self):
            for stamp, event in sequence:
                clock.now = stamp
                yield event
        def send_output(self, *args): self.outputs.append(args)
    class IO:
        requests = []
        responses = iter([actions_response(), actions_response(100)])
        def write(self, text): self.requests.append(json.loads(text))
        def flush(self): pass
        def readline(self): return json.dumps(next(self.responses))
    node, sock, log = Node(), IO(), Log()
    client.run(node, sock, str(tmp_path), log)
    assert len(sock.requests) == 2
    assert [request["metadata"]["observation_monotonic_ns"] for request in sock.requests] == [1_000_000_000, 1_360_000_000]
    assert [request["reset"] for request in sock.requests] == [True, False]
    for i, (name, value, metadata) in enumerate(node.outputs):
        assert name == "actions"
        np.testing.assert_array_equal(np.array(value.to_pylist())[:, 0], np.arange(10) + 100*i)
        assert metadata["queue_action_steps"] == 10
        assert metadata["predicted_steps"] == 50
        assert metadata["observation_monotonic_ns"] == sock.requests[i]["metadata"]["observation_monotonic_ns"]


def executor_event(identifier="first", offset=0, age_ns=0):
    return {"type": "INPUT", "id": "actions",
            "value": pa.array(actions_response(offset)["positions"][:10], type=pa.list_(pa.float32())),
            "metadata": {**timing(time.monotonic_ns() - age_ns), "interval": INTERVAL_NS,
                         "queue_chunk_id": identifier, "action_schedule": SCHEDULE,
                         "queue_action_steps": 10, "reset": identifier == "first"}}


def run_executor(first, on_output):
    original = load_source(ORIGINAL, "_test_ten_step_original")
    class Node:
        def __init__(self): self.events = deque([first]); self.outputs = []
        def is_empty(self): return not self.events
        def next(self): return self.events.popleft()
        def send_output(self, name, value, metadata):
            self.outputs.append((name, value.to_pylist(), metadata))
            on_output(self, name, value, metadata)
    node = Node()
    original.dora = SimpleNamespace(Node=lambda: node)
    log = Log()
    run = queued_async_main(original, log, {}, LoggedNode, LoggedQueue)
    asyncio.run(asyncio.wait_for(run(["right"], True, False, 100), timeout=3))
    return node, log


def test_two_complete_prefixes_have_no_tail_mix_and_keep_original_timestamps():
    def callback(node, name, value, metadata):
        if name == "chunk_complete":
            if metadata["queue_chunk_id"] == "first":
                node.events.append(executor_event("second", 100))
            else:
                node.events.append({"type": "STOP"})
    node, log = run_executor(executor_event(), callback)
    complete = [row for name, row in log.rows if name == "queue_prefix_completed"]
    assert len(complete) == 2
    assert all(row["executed_raw_steps"] == 10 and row["elapsed_s"] >= 10*INTERVAL_NS/1e9 for row in complete)
    assert not [row for row in node.outputs if row[0] == "timing_fault"]
    for chunk_id, offset in (("first", 0), ("second", 100)):
        outputs = [(value, meta) for name, value, meta in node.outputs
                   if name == "move_position_right" and meta["queue_chunk_id"] == chunk_id]
        assert len(outputs) == 34
        assert {meta["queue_raw_step"] for _, meta in outputs} == set(range(10))
        assert len({meta["observation_monotonic_ns"] for _, meta in outputs}) == 1
        assert outputs[0][0] == [float(offset)]*8
        assert outputs[-1][0] == [float(offset + 9)]*8
        assert all(offset <= value[0] <= offset + 9 for value, _ in outputs)


@pytest.mark.parametrize("interrupt", ["execution_stop", "timing_fault", "early_chunk"])
def test_stop_and_fault_interrupt_without_waiting_for_ten_steps(interrupt):
    def callback(node, name, value, metadata):
        if name == "move_position_right" and len(node.outputs) == 1:
            if interrupt == "early_chunk":
                node.events.append(executor_event("early", 100))
            else:
                node.events.append({"type": "INPUT", "id": interrupt, "value": pa.array(["stop now"])})
    node, log = run_executor(executor_event(), callback)
    assert not [row for name, row in log.rows if name == "queue_prefix_completed"]
    assert len([row for row in node.outputs if row[0] == "move_position_right"]) == 1
    assert len([row for row in node.outputs if row[0] == "timing_fault"]) == (interrupt != "execution_stop")


def test_inference_age_budget_is_not_extended_to_fit_the_queue():
    node, log = run_executor(executor_event(age_ns=300_000_000), lambda *args: None)
    assert not [row for row in node.outputs if row[0] == "move_position_right"]
    assert len([row for row in node.outputs if row[0] == "timing_fault"]) == 1
    assert "budget" in [row for name, row in log.rows if name == "queue_executor_fault"][0]["reason"]


def test_graph_has_completion_barrier_and_independent_stop_wires():
    graph = yaml.safe_load((ROOT / "official/dora-openarm-evaluation/dataflow-act-trial-ros-host.yaml").read_text())
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["policy-server"]["args"] == "policy-client"
    assert nodes["policy-server"]["inputs"]["chunk_complete"] == "actions-executor/chunk_complete"
    for name in ("policy-server", "actions-executor"):
        assert nodes[name]["inputs"]["execution_stop"] == "trial-bridge/execution_stop"
