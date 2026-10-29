"""Tests for the closed-loop transit-drop / in-box-release classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.act_sim_mix.closed_loop import classify_trajectory
from experiments.act_sim_mix.sim_expert.world import SimWorld, load_config


def test_release_outside_box_is_a_transit_drop():
    cfg = load_config()
    n = 120
    times = np.arange(n) / 30
    cube = np.tile(np.array([0.33, -0.20, 0.29]), (n, 1))
    # Lift for 1 second, then release and fall back to the table.
    cube[:30, 2] = np.linspace(0.29, 0.36, 30)
    cube[30:60, 2] = np.linspace(0.36, 0.29, 30)
    cube[60:, 2] = 0.29
    bilateral = np.ones(n, dtype=bool)
    bilateral[32:] = False
    samples = [
        {"time": times[i], "cube": cube[i], "bilateral": bool(bilateral[i])}
        for i in range(n)
    ]
    result = classify_trajectory(
        samples,
        nominal_cube_z=0.29,
        box_center=np.array([0.385, 0.01, 0.27]),
        box_rotation=np.eye(3),
        cfg=cfg,
        recovery_type=None,
        forbidden_contacts=[],
    )
    assert result["success"] is False
    assert result["transit_drop"] is True
    assert result["failure_type"] == "transit_drop"


def test_history_collision_invalidates_otherwise_placed_cube():
    cfg = load_config()
    n = 90
    times = np.arange(n) / 30
    cube = np.tile(np.array([0.385, 0.01, 0.30]), (n, 1))
    samples = [
        {"time": times[i], "cube": cube[i], "bilateral": False} for i in range(n)
    ]
    result = classify_trajectory(
        samples,
        nominal_cube_z=0.29,
        box_center=np.array([0.385, 0.01, 0.27]),
        box_rotation=np.eye(3),
        cfg=cfg,
        recovery_type="pre_release_hold",
        forbidden_contacts=[["arm", "table"]],
    )
    assert result["success"] is False
    assert result["failure_type"] == "forbidden_contact"


def test_pre_release_start_can_succeed_without_new_lift():
    cfg = load_config()
    n = 90
    times = np.arange(n) / 30
    cube = np.tile(np.array([0.385, 0.01, 0.30]), (n, 1))
    samples = [
        {"time": times[i], "cube": cube[i], "bilateral": False} for i in range(n)
    ]
    result = classify_trajectory(
        samples,
        nominal_cube_z=0.35,
        box_center=np.array([0.385, 0.01, 0.27]),
        box_rotation=np.eye(3),
        cfg=cfg,
        recovery_type="pre_release_hold",
        forbidden_contacts=[],
    )
    assert result["success"] is True
    assert result["failure_type"] is None


def test_one_finger_contact_does_not_count_as_release():
    samples = [{"time": i / 30, "cube": np.array([.385, .01, .30]),
                "bilateral": False, "finger_count": 1} for i in range(90)]
    result = classify_trajectory(samples, nominal_cube_z=.35,
                                 box_center=np.array([.385, .01, .27]), box_rotation=np.eye(3),
                                 cfg=load_config(), recovery_type="pre_release_hold", forbidden_contacts=[])
    assert not result["success"]
    assert result["failure_type"] == "not_released"


def test_short_stable_tail_is_not_half_a_second():
    samples = [{"time": i / 30, "cube": np.array([.385, .01, .30]),
                "bilateral": False, "cube_speed": .1 if i < 26 else 0.} for i in range(30)]
    result = classify_trajectory(samples, nominal_cube_z=.35,
                                 box_center=np.array([.385, .01, .27]), box_rotation=np.eye(3),
                                 cfg=load_config(), recovery_type="pre_release_hold", forbidden_contacts=[])
    assert not result["success"]
    assert result["failure_type"] == "not_settled"


def test_rotated_cube_must_fit_whole_footprint():
    samples = [{"time": i / 30, "cube": np.array([.42, .01, .30]),
                "cube_half_box": np.array([.0283, .0283, .02]), "bilateral": False} for i in range(90)]
    result = classify_trajectory(samples, nominal_cube_z=.35,
                                 box_center=np.array([.385, .01, .27]), box_rotation=np.eye(3),
                                 cfg=load_config(), recovery_type="pre_release_hold", forbidden_contacts=[])
    assert not result["success"]
    assert result["failure_type"] == "miss_box"
