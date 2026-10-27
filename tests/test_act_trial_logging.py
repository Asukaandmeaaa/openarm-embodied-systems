"""Logging preserves numerical results and original observation provenance."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/act_trial"))
import logged_executor as executor
from logged_policy_server import AuditLog, PredictionProxy, RequestJSON, chunk_sha256, load_source, wrap_infer, ORIGINAL as POLICY_ORIGINAL


class Log:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


def test_hash_matches_numpy_and_python_float_matrix():
    values = np.array([[0.1, -0.35], [0.2, 0.0]], dtype=np.float32)
    assert chunk_sha256(values) == chunk_sha256(values.tolist())


def test_audit_files_unique_and_json_finite(tmp_path):
    first, second = AuditLog("test", tmp_path), AuditLog("test", tmp_path)
    first.write("invalid_diagnostic", values=[np.inf, -np.inf, np.nan])
    first.close(); second.close()
    assert first.path != second.path
    row = json.loads(first.path.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert row["values"][0] == {"nonfinite_float": "inf"}


def test_request_archives_identical_arrow_bytes_and_returns_same_content(tmp_path):
    source = tmp_path / "source.arrow"
    source.write_bytes(b"example original bytes")
    request = {"name": "inference", "data_path": str(source), "metadata": {"observation_monotonic_ns": 17}, "reset": True}
    log = AuditLog("policy", tmp_path)
    context = {}
    result = RequestJSON(json, log, context).loads(json.dumps(request))
    log.close()
    assert result == request
    assert Path(context["observation_arrow_path"]).read_bytes() == source.read_bytes()
    assert context["metadata"]["observation_monotonic_ns"] == 17


def test_proxy_returns_exact_original_tensor_object():
    class Tensor:
        shape = (1, 2, 2)
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return [[[1.0, 2.0], [3.0, 4.0]]]
    tensor = Tensor()
    class Policy:
        def predict_action_chunk(self, batch):
            assert batch == {"state": 1}
            return tensor
    assert PredictionProxy(Policy(), Log(), {}).predict_action_chunk({"state": 1}) is tensor


def test_output_values_unchanged_and_original_age_propagated():
    class Value:
        type = "float32"
        def to_pylist(self): return [0.1] * 8
    class Node:
        def send_output(self, name, value, metadata):
            self.last = name, value, metadata
            return "original-return"
    value, node = Value(), Node()
    context = {"metadata": {"observation_monotonic_ns": 123, "source_age_ns": 45, "source_skew_ns": 6}, "trial_chunk_sha256": "hash"}
    assert executor.LoggedNode(node, Log(), context).send_output("move_position_right", value, {"timestamp": 9}) == "original-return"
    assert node.last[1] is value
    assert node.last[2] == {"timestamp": 9, "observation_monotonic_ns": 123, "source_age_ns": 45, "source_skew_ns": 6, "trial_chunk_sha256": "hash"}


def test_queue_context_changes_on_dequeue_not_arrival():
    class Value:
        def to_pylist(self): return [[0.1] * 16] * 10
    async def run():
        context = {"metadata": {"observation_monotonic_ns": 1}}
        queue = executor.LoggedQueue(Log(), context)
        event = {"id": "actions", "value": Value(), "metadata": {"observation_monotonic_ns": 2}}
        await queue.put(event)
        assert context["metadata"]["observation_monotonic_ns"] == 1
        assert await queue.get() is event
        assert context["metadata"]["observation_monotonic_ns"] == 2
    asyncio.run(run())


def test_original_hermite_and_filter_numerically_identical():
    pytest.importorskip("dora")
    pytest.importorskip("pyarrow")
    original = load_source(executor.ORIGINAL, "_trial_executor_test_original")
    reference_h = original.HermiteUpsampler(30.0, 9 / 30.0)
    reference_f = original.BiquadLowpass(100.0, 15.0)
    executor.instrument_classes(original, Log(), {})
    observed_h = original.HermiteUpsampler(30.0, 9 / 30.0)
    observed_f = original.BiquadLowpass(100.0, 15.0)
    coarse = np.arange(160, dtype=np.float32).reshape(10, 16) / 1000
    times = np.arange(0, 0.301, 0.01)
    expected = reference_h.upsample(coarse, times)
    actual = observed_h.upsample(coarse, times)
    np.testing.assert_array_equal(actual, expected)
    for row in expected:
        np.testing.assert_array_equal(observed_f.step(row), reference_f.step(row))


def test_original_infer_full_chunk_result_identical():
    torch = pytest.importorskip("torch")
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("lerobot")
    original = load_source(POLICY_ORIGINAL, "_trial_policy_test_original")
    prediction = torch.arange(800, dtype=torch.float32).reshape(1, 50, 16) / 1000
    class Policy:
        def predict_action_chunk(self, batch):
            assert tuple(batch["observation.state"].shape) == (16,)
            return prediction
    observation = pa.array([{"id": 1, "position": [0.1] * 16}])[0]
    preprocess = lambda batch: batch
    postprocess = lambda actions: actions * 0.5
    expected = original.infer(Policy(), preprocess, postprocess, observation)
    log = Log()
    actual = wrap_infer(original.infer, log, {})(Policy(), preprocess, postprocess, observation)
    assert actual == expected
    assert len(actual["positions"]) == 50
    raw = [row for event, row in log.rows if event == "raw_policy_response"][0]
    assert raw["trial_chunk_sha256"] == chunk_sha256(expected["positions"])
