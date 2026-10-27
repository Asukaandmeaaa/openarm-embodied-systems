"""Prevent unpatched CAN environments from reaching the Dora hardware entry."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NODE = Path(__file__).resolve().parents[1] / "nodes/dora-openarm"
spec = importlib.util.spec_from_file_location("can_runtime", NODE / "src/dora_openarm/can_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.mark.parametrize("version", ["1.4.0", "1.2.2", "1.5.0", "unknown"])
def test_unvalidated_backend_rejected(version):
    with pytest.raises(RuntimeError, match="install_dora_can_fix.sh"):
        runtime.require_paced_can(SimpleNamespace(__version__=version))


def test_validated_backend_accepted():
    assert runtime.require_paced_can(SimpleNamespace(__version__="1.4.0+dmrefresh1")) == "1.4.0+dmrefresh1"


def test_fast_dora_reads_keep_receiving_without_retransmitting(monkeypatch):
    sys.path.insert(0, str(NODE / "src"))
    from dora_openarm import feedback_driver
    calls = []
    monkeypatch.setattr(feedback_driver.BaseSingleArmDriver, "fetch_state",
                        lambda self, refresh=True: calls.append(refresh) or {"qpos": [1]})
    arm = object.__new__(feedback_driver.SingleArmDriver)
    arm._next_can_refresh = 0.0
    # 250 Hz Dora reads still drain RX, while explicit queries stay at <=20 Hz.
    for i in range(26):
        monkeypatch.setattr(feedback_driver.time, "monotonic", lambda i=i: i * 0.004)
        assert arm.fetch_position() == [1]
    assert len(calls) == 26
    assert [i for i, request in enumerate(calls) if request] == [0, 13]
    arm.fetch_state(refresh=False)
    assert calls[-1] is False


def test_driver_rejects_unpatched_library_before_base_constructor(monkeypatch):
    sys.path.insert(0, str(NODE / "src"))
    from dora_openarm import feedback_driver
    called = []
    monkeypatch.setattr(feedback_driver.BaseSingleArmDriver, "__init__", lambda *a, **k: called.append(True))
    def reject():
        raise RuntimeError("unpatched library")
    monkeypatch.setattr(feedback_driver, "require_paced_can", reject)
    with pytest.raises(RuntimeError, match="unpatched library"):
        feedback_driver.SingleArmDriver("right_arm")
    assert not called
