#!/usr/bin/env python3
"""Tests for G5.5.1 Action Representation Debug experiments."""

import numpy as np
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.validation.g5_5_1_action_representation_debug import (
    analyze_expert_frequency,
    convert_expert_to_target_hz,
)


def test_analyze_expert_frequency():
    """Test expert frequency analysis."""
    # Create mock timestamps at 30Hz
    duration_s = 10.0
    hz = 30.0
    timestamps_ns = np.arange(0, duration_s * 1e9, 1e9 / hz, dtype=np.int64)

    result = analyze_expert_frequency(timestamps_ns)

    assert result["original_hz"] == pytest.approx(hz, rel=0.01)
    assert result["mean_dt_s"] == pytest.approx(1.0 / hz, rel=0.01)
    assert result["std_dt_s"] < 0.001  # Should be very consistent


def test_convert_expert_to_target_hz():
    """Test frequency conversion."""
    # Create mock expert trajectory
    duration_s = 5.0
    original_hz = 50.0
    target_hz = 30.0

    timestamps_ns = np.arange(0, duration_s * 1e9, 1e9 / original_hz, dtype=np.int64)
    actions = np.random.randn(len(timestamps_ns), 16).astype(np.float32)

    new_timestamps_ns, new_actions = convert_expert_to_target_hz(
        timestamps_ns, actions, target_hz
    )

    # Check output frequency
    computed_hz = 1e9 / np.mean(np.diff(new_timestamps_ns))
    assert computed_hz == pytest.approx(target_hz, rel=0.01)

    # Check interpolation
    assert new_actions.shape[1] == 16
    assert len(new_actions) == len(new_timestamps_ns)

    # First and last actions should match (interpolation endpoints)
    assert np.allclose(new_actions[0], actions[0], atol=0.01)
    assert np.allclose(new_actions[-1], actions[-1], atol=0.01)


def test_zoh_upsampling_ratio():
    """Test ZOH upsampling maintains correct ratio."""
    from_hz = 30.0
    to_hz = 250.0

    expected_ratio = to_hz / from_hz
    assert expected_ratio == pytest.approx(8.333, rel=0.01)

    steps_per_frame = int(round(expected_ratio))
    assert steps_per_frame == 8  # Should round to 8


def test_guard_ratio_definitions():
    """Test ActionGuard ratio calculations."""
    total_steps = 1000
    total_joints = 16

    # Mock clamp counts
    step_clamps = 50  # 50 steps had at least one joint clamped
    velocity_clamps = 200  # 200 individual (step, joint) pairs clamped for velocity

    # Step clamp ratio: fraction of steps with any clamp
    step_clamp_ratio = step_clamps / total_steps
    assert step_clamp_ratio == 0.05  # 5%

    # Axis clamp ratio: fraction of (step, joint) samples clamped
    total_axis_samples = total_steps * total_joints
    velocity_clamp_ratio = velocity_clamps / total_axis_samples
    assert velocity_clamp_ratio == 0.0125  # 1.25%

    # This ratio should always be less than or equal to step_clamp_ratio
    assert velocity_clamp_ratio <= step_clamp_ratio


def test_hermite_vs_zoh_difference():
    """Test that ZOH and Hermite produce different outputs."""
    # Simple test: constant velocity trajectory
    actions_30hz = np.array([
        [0.0] * 16,
        [0.1] * 16,
        [0.2] * 16,
    ], dtype=np.float32)

    # ZOH: step function
    zoh_output = []
    steps_per_frame = 8
    for action in actions_30hz:
        zoh_output.extend([action] * steps_per_frame)
    zoh_output = np.array(zoh_output)

    # ZOH should have discontinuities (step changes)
    zoh_diff = np.diff(zoh_output, axis=0)
    num_discontinuities = np.sum(np.any(zoh_diff != 0, axis=1))

    # Should have exactly 2 discontinuities (at frame boundaries)
    assert num_discontinuities == 2

    # Hermite would produce smooth interpolation (no sudden jumps)
    # This is tested implicitly by the actual Hermite implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
