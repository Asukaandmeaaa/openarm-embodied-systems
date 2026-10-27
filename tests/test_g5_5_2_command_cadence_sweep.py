#!/usr/bin/env python3
"""Unit tests for the controlled G5.5.2 experiment contract."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "official/openarm_mujoco/src"),
    str(
        ROOT
        / "official/dora-openarm-evaluation/nodes/"
        "dora-openarm-actions-executor/src"
    ),
]

from openarm_cube_pick_expert.guard import ActionGuard
from sim.validation.g5_5_2_command_cadence_sweep import (
    GuardDiagnostics,
    build_executor_trajectory,
    predict_guard_masks,
    signed_phase_lag_ms,
)


def linear_source(duration_s: float = 33.1):
    time_s = np.arange(0.0, duration_s + 1e-12, 1.0 / 30.0)
    actions = np.empty((len(time_s), 16), dtype=np.float32)
    for axis in range(16):
        actions[:, axis] = (axis + 1) * 0.001 * time_s
    return time_s, actions


@pytest.mark.parametrize("cadence", [30.0, 40.0, 50.0, 60.0, 90.0, 120.0])
@pytest.mark.parametrize("horizon", [3.3, 33.0])
def test_executor_grid_has_identical_duration(cadence: float, horizon: float):
    source_time, actions = linear_source()
    trajectory = build_executor_trajectory(
        source_time,
        actions,
        cadence_hz=cadence,
        duration_s=33.0,
        chunk_horizon_s=horizon,
    )
    assert trajectory.hermite_targets.shape == (8250, 16)
    assert trajectory.control_time_s[0] == 0.0
    assert trajectory.control_time_s[-1] == pytest.approx(32.996)
    assert np.allclose(np.diff(trajectory.control_time_s), 0.004)
    # Hermite must reproduce a globally linear signal independent of cadence.
    assert np.allclose(
        trajectory.hermite_targets,
        trajectory.source_reference,
        atol=2e-6,
    )


def test_signed_phase_lag_positive_means_output_lags():
    n = 1000
    reference = np.zeros((n, 16), dtype=np.float32)
    reference[200:250, 0] = np.linspace(0, 1, 50)
    output = np.roll(reference, 5, axis=0)
    result = signed_phase_lag_ms(reference, output, sample_hz=250.0)
    assert result["per_axis_ms"][0] == pytest.approx(20.0)


def test_guard_diagnostics_are_not_divided_by_control_steps_times_axes():
    lower = np.full(16, -10, dtype=np.float32)
    upper = np.full(16, 10, dtype=np.float32)
    guard = ActionGuard(
        lower,
        upper,
        np.ones(16, dtype=np.float32),
        np.ones(16, dtype=np.float32),
        np.ones(16, dtype=np.float32),
        watchdog_timeout_s=1.0,
    )
    current = np.zeros(16, dtype=np.float32)
    target = np.zeros(16, dtype=np.float32)
    target[0] = 1.0
    guard.submit(target, 0.0)

    diagnostics = GuardDiagnostics()
    diagnostics.begin_command()
    for step in range(4):
        now = step * 0.001
        active, intervention = predict_guard_masks(guard, current, now)
        guard.step(current, now)
        diagnostics.observe(active, intervention)
    diagnostics.end_command()
    result = diagnostics.as_dict()

    assert result["velocity"]["active_physics_step_ratio"] == 1.0
    assert result["velocity"]["active_axis_event_ratio"] == pytest.approx(1 / 16)
    assert result["velocity"]["active_per_axis_ratio"][0] == 1.0
    assert result["velocity"]["active_per_axis_ratio"][1] == 0.0
    assert result["velocity"]["command_any_ratio"] == 1.0


def test_non_integral_contract_is_rejected():
    source_time, actions = linear_source()
    with pytest.raises(ValueError, match="integer number of intervals"):
        build_executor_trajectory(
            source_time,
            actions,
            cadence_hz=30.0,
            duration_s=32.95,
            chunk_horizon_s=3.3,
        )
