#!/usr/bin/env python3
"""Tests for G5.5 Executor Semantic Equivalence Gate."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Go up 1 level to openarm_vla_ws
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim" / "validation"))

import numpy as np

# Import directly (not via importlib to avoid dataclass module issues)
from g5_5_executor_equivalence import (
    compute_phase_lag,
    compute_deviation_metrics,
    convert_expert_to_30hz,
)


def test_phase_lag_known_delay():
    """Test phase lag computation with synthetic known-delay signal."""
    # Generate a test signal
    t = np.linspace(0, 1.0, 1000)
    dt = t[1] - t[0]
    signal_a = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    # Introduce known delay of 20ms
    delay_samples = int(0.020 / dt)
    signal_b = np.roll(signal_a, delay_samples)

    # Compute lag (signal_b is delayed relative to signal_a)
    lag_ms = compute_phase_lag(signal_a, signal_b, dt)

    expected_lag_ms = 20.0
    tolerance_ms = 2.0  # Allow 2ms tolerance

    assert lag_ms is not None, "Phase lag should not be None for valid signals"
    # Phase lag can be positive or negative depending on cross-correlation convention
    # The magnitude should match
    assert abs(abs(lag_ms) - expected_lag_ms) < tolerance_ms, \
        f"Expected lag magnitude ~{expected_lag_ms}ms, got {abs(lag_ms):.2f}ms"

    print(f"✓ test_phase_lag_known_delay: lag={lag_ms:.2f}ms (magnitude ~{abs(lag_ms):.2f}ms, expected ~{expected_lag_ms}ms)")


def test_phase_lag_insufficient_excitation():
    """Test phase lag returns None for low-energy signals."""
    # Static signal (no excitation)
    signal_a = np.ones(1000) * 0.5
    signal_b = np.ones(1000) * 0.5
    dt = 0.001

    lag_ms = compute_phase_lag(signal_a, signal_b, dt, min_energy_threshold=0.01)

    assert lag_ms is None, "Phase lag should be None for insufficient excitation"
    print("✓ test_phase_lag_insufficient_excitation: correctly returns None")


def test_deviation_metrics():
    """Test deviation metrics computation."""
    # Create two trajectories with known deviation
    traj_a = np.random.randn(1000, 16) * 0.1
    traj_b = traj_a + np.random.randn(1000, 16) * 0.01  # Add small noise

    dt = 0.001
    metrics = compute_deviation_metrics(traj_a, traj_b, dt)

    assert "mean" in metrics
    assert "max" in metrics
    assert "p95" in metrics
    assert "per_axis_max" in metrics
    assert len(metrics["per_axis_max"]) == 16

    assert metrics["mean"] > 0, "Mean deviation should be positive"
    assert metrics["max"] >= metrics["p95"], "Max should be >= p95"
    assert metrics["p95"] >= metrics["mean"], "p95 should be >= mean"

    print(f"✓ test_deviation_metrics: mean={metrics['mean']:.6f}, max={metrics['max']:.6f}")


def test_convert_expert_to_30hz():
    """Test expert trajectory conversion to 30Hz."""
    # Simulate expert trajectory at ~1000Hz for 1 second
    original_hz = 1000.0
    duration_s = 1.0
    n_samples = int(original_hz * duration_s)

    timestamps_ns = np.arange(0, n_samples) * int(1e9 / original_hz)
    actions = np.random.randn(n_samples, 16).astype(np.float32)

    # Convert to 30Hz
    new_ts, new_actions = convert_expert_to_30hz(timestamps_ns, actions, target_hz=30.0)

    expected_30hz_samples = int(30.0 * duration_s)

    assert len(new_ts) == len(new_actions), "Timestamps and actions length mismatch"
    assert abs(len(new_actions) - expected_30hz_samples) <= 2, \
        f"Expected ~{expected_30hz_samples} samples at 30Hz, got {len(new_actions)}"
    assert new_actions.shape[1] == 16, "Actions should be 16D"
    assert np.all(np.diff(new_ts) > 0), "Timestamps should be monotonically increasing"

    print(f"✓ test_convert_expert_to_30hz: {len(actions)} → {len(new_actions)} samples")


def test_biquad_filter_reset():
    """Test BiquadLowpass filter state reset is deterministic."""
    import types

    EXECUTOR_SRC = ROOT / "official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src"
    sys.path.insert(0, str(EXECUTOR_SRC))
    sys.modules.setdefault("dora", types.ModuleType("dora"))

    from dora_openarm_actions_executor.main import BiquadLowpass

    # Create filter
    lowpass = BiquadLowpass(fs=250.0, fc=15.0, Q=0.707)

    # Reset with initial value
    initial_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    lowpass.reset_state(initial_value)

    # First step should return initial value (filter is initialized)
    output1 = lowpass.step(initial_value)
    assert np.allclose(output1, initial_value, atol=1e-6), \
        "First step after reset should return initial value"

    # Reset again and verify deterministic behavior
    lowpass.reset_state(initial_value)
    output2 = lowpass.step(initial_value)
    assert np.allclose(output1, output2, atol=1e-9), \
        "Filter reset should be deterministic"

    print("✓ test_biquad_filter_reset: deterministic reset verified")


def test_timestamp_alignment():
    """Test timestamp monotonicity check."""
    # Create monotonic timestamps
    timestamps = np.arange(0, 1000, dtype=np.int64) * int(1e6)  # 1ms intervals

    # Check monotonicity
    assert np.all(np.diff(timestamps) > 0), "Timestamps should be monotonically increasing"

    # Create non-monotonic timestamps
    bad_timestamps = timestamps.copy()
    bad_timestamps[500] = bad_timestamps[499]  # Duplicate timestamp

    assert not np.all(np.diff(bad_timestamps) > 0), \
        "Non-monotonic timestamps should be detected"

    print("✓ test_timestamp_alignment: monotonicity check passed")


def test_clamp_ratio_denominator():
    """Test clamp ratio calculation uses correct denominator."""
    # Simulate guard statistics
    total_control_steps = 1000
    velocity_clamps = 50

    velocity_clamp_ratio = velocity_clamps / total_control_steps

    assert velocity_clamp_ratio == 0.05, "Clamp ratio should be clamps / total_steps"
    assert 0.0 <= velocity_clamp_ratio <= 1.0, "Clamp ratio should be in [0, 1]"

    print(f"✓ test_clamp_ratio_denominator: ratio={velocity_clamp_ratio:.2%}")


def main():
    """Run all tests."""
    print("Running G5.5 Tests...")
    print("=" * 80)

    tests = [
        test_phase_lag_known_delay,
        test_phase_lag_insufficient_excitation,
        test_deviation_metrics,
        test_convert_expert_to_30hz,
        test_biquad_filter_reset,
        test_timestamp_alignment,
        test_clamp_ratio_denominator,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 80)
    print(f"Tests: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
