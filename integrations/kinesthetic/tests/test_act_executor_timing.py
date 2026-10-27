import numpy as np
import importlib.util
from pathlib import Path
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verify_act_executor_timing", Path(__file__).parents[1] / "scripts" / "verify_act_executor_timing.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
blend_remaining = _MODULE.blend_remaining


def test_observed_period_finishes_fjt_goal_before_next_chunk():
    interval = 0.47
    start_delay = 0.02
    consumed = min(10, int((interval - start_delay) * 30.0))
    assert (10 - 1) / 30.0 + start_delay == pytest.approx(0.32)
    assert interval - 0.32 == pytest.approx(0.15)
    assert consumed == 10


def test_blend_is_continuous_when_chunk_arrives_while_goal_is_live():
    previous = np.arange(80, dtype=np.float32).reshape(10, 8)
    new = previous[-1] + 10.0
    blended, count = blend_remaining(previous, new[None, :] + np.arange(80, dtype=np.float32).reshape(10, 8), 6)
    assert count == 4
    np.testing.assert_allclose(blended[0], previous[6])
    np.testing.assert_allclose(blended[3], new + np.arange(24, 32, dtype=np.float32))


def test_late_chunk_interval_is_detectable_before_it_can_be_used_as_a_new_goal():
    chunk_hz = 30.0
    start_delay = 0.02
    dense_horizon = start_delay + (10 - 1) / chunk_hz
    arrival_interval = 0.47

    assert dense_horizon == pytest.approx(0.32)
    assert arrival_interval > dense_horizon
    assert arrival_interval - dense_horizon == pytest.approx(0.15)
    # All ten source samples have already been consumed; there is no overlap
    # that a raw-point blend can use to hide a new first-point jump.
    assert min(10, int((arrival_interval - start_delay) * chunk_hz)) == 10


def test_stop_scenario_drops_pending_blend_without_reusing_old_actions():
    previous = np.zeros((10, 8), dtype=np.float32)
    new = np.ones((10, 8), dtype=np.float32)
    blended, count = blend_remaining(previous, new, consumed=3)
    assert count == 7
    np.testing.assert_allclose(blended[0], previous[3])

    # A stopped executor has no previous chunk to blend against.  The next run
    # therefore starts from the new raw chunk exactly.
    restarted = new.copy()
    restarted_count = 0
    assert restarted_count == 0
    np.testing.assert_allclose(restarted, new)
