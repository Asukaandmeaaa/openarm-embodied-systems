"""Regression tests for ACT full-chunk inference backpressure."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault("dora", types.ModuleType("dora"))


@pytest.mark.parametrize(
    "relative_path",
    [
        "official/dora-openarm-evaluation/nodes/dora-openarm-local-policy-server/"
        "src/dora_openarm_local_policy_server/main.py",
        "official/dora-openarm-evaluation/nodes/dora-openarm-docker-policy-server/"
        "src/dora_openarm_docker_policy_server/main.py",
    ],
)
def test_chunk_horizon_uses_intervals_between_positions(relative_path: str) -> None:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    actions = {"interval": 33_333_333, "positions": [[0.0]] * 100}
    assert module._chunk_horizon_ns(actions) == 3_299_999_967
    assert module._chunk_horizon_ns({"interval": 7, "positions": []}) == 0
    assert module._chunk_horizon_ns({"interval": 7, "positions": [[0.0]]}) == 0
