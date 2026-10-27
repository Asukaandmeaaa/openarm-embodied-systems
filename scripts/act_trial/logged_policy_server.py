#!/usr/bin/env python3
"""Instrument the existing policy server without replacing inference or reset logic.

ACT_TRIAL_DIR is required. The socket argument/SOCKET and model environment are
consumed by the original server unchanged. This process never controls motors.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "official/dora-openarm-evaluation/src/local_policy_server.py"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    """Valid JSON even when diagnostics encounter an invalid nonfinite value."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_pylist"):
        return json_safe(value.to_pylist())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "as_py"):
        return json_safe(value.as_py())
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return {"type": type(value).__name__, "repr": repr(value)}


def chunk_sha256(positions):
    # Both the original server and Arrow executor carry float32-derived Python
    # floats; hash the complete matrix, not a truncated right-only projection.
    return hashlib.sha256(json.dumps(json_safe(positions), separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class AuditLog:
    def __init__(self, role, directory=None):
        chosen = directory if directory is not None else os.environ.get("ACT_TRIAL_DIR")
        if not chosen:
            raise ValueError("ACT_TRIAL_DIR is required for complete trial logging")
        self.directory = Path(chosen).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{role}_{time.monotonic_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.path = self.directory / f"{self.run_id}.jsonl"
        self.stream = self.path.open("x", buffering=1)
        self.sequence = 0

    def write(self, event, **fields):
        self.sequence += 1
        record = {"event": event, "log_sequence": self.sequence, "run_id": self.run_id,
                  "monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns(), **fields}
        self.stream.write(json.dumps(json_safe(record), separators=(",", ":"), allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def load_source(path, name):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    # Executor and server import the same timing contract from adjacent src/.
    if path.parent.name.startswith("dora_openarm_"):
        sys.path.insert(0, str(path.parent.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RequestJSON:
    """Intercept only decoded inference requests; delegate all JSON operations."""
    def __init__(self, original_json, log, context):
        self.original = original_json
        self.log = log
        self.context = context

    def __getattr__(self, name):
        return getattr(self.original, name)

    def loads(self, text, *args, **kwargs):
        request = self.original.loads(text, *args, **kwargs)
        if isinstance(request, dict) and request.get("name") == "inference" and request.get("data_path"):
            self.context.clear()
            self.context.update(request_sequence=self.log.sequence + 1,
                                metadata=dict(request.get("metadata", {})), reset=request.get("reset", False))
            source = Path(request["data_path"])
            archive = self.log.directory / f"{self.log.run_id}_observation_{self.context['request_sequence']:06d}.arrow"
            raw = source.read_bytes()
            with archive.open("xb") as stream:
                stream.write(raw)
            self.context.update(observation_arrow_path=str(archive),
                                observation_arrow_sha256=hashlib.sha256(raw).hexdigest())
            self.log.write("inference_request", **self.context)
        return request


class PredictionProxy:
    def __init__(self, policy, log, context):
        self.policy = policy
        self.log = log
        self.context = context

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def predict_action_chunk(self, batch):
        prediction = self.policy.predict_action_chunk(batch)
        # Copy only for evidence; return the exact original tensor object.
        copied = prediction.detach().cpu().tolist()
        self.log.write("network_prediction_normalized", **self.context,
                       shape=list(prediction.shape), full_prediction=copied,
                       prediction_sha256=chunk_sha256(copied))
        return prediction


def wrap_infer(original_infer, log, context):
    def infer(policy, preprocessor, postprocessor, observation):
        position = observation["position"].values.to_numpy(zero_copy_only=False).tolist()
        try:
            observation_id = observation["id"].as_py()
        except (KeyError, TypeError):
            observation_id = None
        started = time.monotonic_ns()
        log.write("observation_used", **context, observation_id=observation_id, state_positions_rad=position)
        actions = original_infer(PredictionProxy(policy, log, context), preprocessor, postprocessor, observation)
        log.write("raw_policy_response", **context, observation_id=observation_id,
                  full_postprocessed_actions=actions["positions"],
                  trial_chunk_sha256=chunk_sha256(actions["positions"]),
                  interval_ns=actions["interval"], cutoff_hz=actions.get("cutoff_hz"),
                  inference_with_logging_ns=time.monotonic_ns() - started)
        return actions
    return infer


def main():
    log = AuditLog("policy")
    try:
        original = load_source(ORIGINAL, "_act_trial_original_policy_server")
        checkpoint = Path(original.PRETRAINED_PATH).resolve()
        if not checkpoint.is_dir():
            raise ValueError("trial logging requires an existing local checkpoint directory")
        log.write("policy_wrapper_started", argv=sys.argv, socket=os.getenv("SOCKET"),
                  source_sha256={str(ORIGINAL): sha_file(ORIGINAL), str(Path(__file__)): sha_file(__file__),
                                 str(ORIGINAL.with_name("observation_timing.py")): sha_file(ORIGINAL.with_name("observation_timing.py"))},
                  checkpoint_sha256={str(p): sha_file(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
                  algorithms="original main, reset, preprocessing, predict_action_chunk, postprocessing and response unchanged",
                  timing_note="logging adds measured overhead; original freshness checks remain active")
        context = {}
        original.json = RequestJSON(json, log, context)
        original.infer = wrap_infer(original.infer, log, context)
        original.main()
    except BaseException as exc:
        log.write("policy_wrapper_exit", exception_type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
