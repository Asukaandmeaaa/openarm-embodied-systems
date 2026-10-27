from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
import pytest

from openarm_dataset import Dataset
from openarm_cube_pick_expert.collect import Collector
from openarm_cube_pick_expert.expert import ExpertPhase, IKTimeout, MujocoGraspPlanner
from openarm_cube_pick_expert.fsm import ExpertFSM
from sim.validation.expert_action_replay import replay_episode


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
LEROBOT = ROOT / "data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen"
SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
CONDA_PYTHON = Path("/home/lushilin/miniconda3/envs/vla/bin/python")


@pytest.fixture(scope="module")
def planned_bundle(tmp_path_factory):
    collector = Collector(tmp_path_factory.mktemp("planner_dataset"), 20260902, False)
    yaw, _ = collector.reset(20260902)
    cube = collector.data.qpos[collector.cube_qadr:collector.cube_qadr + 7].copy()
    current = collector.state()[:7].copy()
    planner = MujocoGraspPlanner(
        collector.model, collector.mapper, collector.cfg, 20260902,
        runtime_data=collector.data,
    )
    selected, candidates = planner.plan(cube[:3], yaw, current)
    yield collector, planner, selected, candidates, cube, yaw, current
    collector.close()


def test_multistart_ik_timeout_falls_back_to_next_seed(planned_bundle, monkeypatch):
    _, planner, _, _, _, _, current = planned_bundle
    calls = []

    def timeout_then_succeed(position, rotation, seed, *, deadline=None):
        calls.append(np.asarray(seed).copy())
        if len(calls) == 1:
            raise IKTimeout("synthetic raw IK timeout")
        return current.copy()

    monkeypatch.setattr(planner, "_solve", timeout_then_succeed)
    result = planner.solve_pose(
        np.zeros(3), np.eye(3), current, deadline=time.monotonic() + 1.0
    )
    np.testing.assert_allclose(result, current)
    assert len(calls) >= 2
    assert planner.ik_timeout_count >= 1


def test_candidate_failure_does_not_pollute_fallback(planned_bundle):
    _, _, selected, candidates, _, _, _ = planned_bundle
    assert selected is not None
    assert candidates[0].failure_reasons == ["pregrasp_ik"]
    assert candidates[1].reachable and not candidates[1].failure_reasons
    assert selected.index in {1, 2}


def test_planned_path_collision_cartesian_step_and_joint_margins(planned_bundle):
    collector, planner, selected, _, _, _, _ = planned_bundle
    assert selected is not None
    assert selected.transit_waypoints
    assert all(not planner._has_forbidden_collision(q) for q in selected.transit_waypoints)

    opened = collector.cfg["pregrasp_right_gripper_rad"]
    closed = collector.cfg["approach_right_gripper_rad"]
    count = len(selected.approach_waypoints)
    for index, qpos in enumerate(selected.approach_waypoints, 1):
        gripper = opened + (closed - opened) * index / count
        assert not planner._has_forbidden_collision(qpos, allow_cube=True, gripper=gripper)

    approach = [selected.waypoints[0], *selected.approach_waypoints]
    positions = []
    for qpos in approach:
        planner.data.qpos[planner.qidx] = qpos
        mujoco.mj_forward(planner.model, planner.data)
        positions.append(planner.data.site_xpos[planner.site].copy())
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert float(steps.max()) <= collector.cfg["approach_step_m"] + 1e-4

    complete_path = [
        *selected.transit_waypoints,
        *selected.approach_waypoints,
        *selected.lift_waypoints,
    ]
    values = np.asarray(complete_path)
    assert np.all(values >= planner.lower - 1e-9)
    assert np.all(values <= planner.upper + 1e-9)
    wrist_margin = np.minimum(
        values[:, 5:7] - planner.lower[5:7],
        planner.upper[5:7] - values[:, 5:7],
    )
    assert float(wrist_margin.min()) >= collector.cfg["joint_margin_rad"]


def test_incomplete_cartesian_path_is_never_executable(planned_bundle, monkeypatch):
    collector, planner, _, _, cube, yaw, current = planned_bundle
    calls = 0

    def endpoint_then_fail(position, rotation, seed, **kwargs):
        nonlocal calls
        calls += 1
        return current.copy() if calls % 2 else None

    monkeypatch.setattr(planner, "solve_pose", endpoint_then_fail)
    monkeypatch.setattr(planner, "_has_forbidden_collision", lambda *args, **kwargs: False)
    selected, candidates = planner.plan(cube[:3], yaw, current)
    assert selected is None
    assert all(not candidate.reachable for candidate in candidates)
    assert all(not candidate.waypoints for candidate in candidates)
    assert all(
        any(reason.startswith("cartesian_ik_step_") for reason in candidate.failure_reasons)
        for candidate in candidates
    )


def test_collector_reset_clears_transaction_guard_and_fsm(tmp_path):
    collector = Collector(tmp_path / "reset_dataset", 123, False)
    try:
        collector.recorder.start("stale")
        collector.guard.submit(np.ones(16), 99.0)
        collector.fsm.transition(ExpertPhase.FAIL, 0.1, "stale")
        collector.reset(123)
        assert collector.recorder.stage is None
        assert collector.fsm.phase == ExpertPhase.PLAN
        assert [item["phase"] for item in collector.fsm.transitions] == [
            ExpertPhase.RESET.value, ExpertPhase.PLAN.value
        ]
        assert "hold_watchdog" not in collector.guard.stats
    finally:
        collector.close()


def test_interrupted_attempt_is_durably_separate_from_success_metadata(tmp_path):
    collector = Collector(tmp_path / "attempt_dataset", 123, False)
    try:
        collector.record_attempt({
            "seed": 123,
            "reset_index": 0,
            "success_reason": None,
            "fail_reason": "interrupted",
            "exception_type": "KeyboardInterrupt",
        })
        attempt = json.loads(collector.attempts_path.read_text().strip())
        assert attempt["fail_reason"] == "interrupted"
        assert not (collector.recorder.output / "metadata.yaml").exists()
        assert not (collector.recorder.output / "episodes").exists()
    finally:
        collector.close()


def test_openarm_dataset_validator_and_loader_cover_all_episodes():
    dataset = Dataset(RAW)
    errors = []
    assert dataset.validate(on_error=errors.append)
    assert not errors and dataset.num_episodes == 5
    for episode in dataset.meta.episodes:
        obs = dataset.load_obs(episode, state="qpos")
        actions = dataset.load_action(episode, state="qpos")
        cameras = dataset.load_cameras(episode)
        counts = {len(value) for value in [*obs.values(), *actions.values()]}
        counts.update(len(camera.load_timestamps()) for camera in cameras.values())
        assert len(counts) == 1


@pytest.mark.skipif(not CONDA_PYTHON.exists(), reason="LeRobot vla environment unavailable")
def test_lerobot_v3_loader_traverses_full_dataset(tmp_path):
    report = tmp_path / "lerobot_validation.json"
    subprocess.run(
        [
            str(CONDA_PYTHON),
            str(ROOT / "sim/validation/validate_lerobot.py"),
            str(LEROBOT),
            "--state-normalization", "identity",
            "--output", str(report),
        ],
        cwd=ROOT,
        check=True,
    )
    result = json.loads(report.read_text())
    assert result["valid"] and result["frames_traversed"] == 5006
    assert len(result["episodes_traversed"]) == 5


def test_strict_actuator_replay_reaches_lift_and_hold():
    import yaml

    metadata = yaml.safe_load((RAW / "metadata.yaml").read_text())
    report = replay_episode(RAW, SCENE, metadata["episodes"][0])
    assert report["valid"] and report["success"]
    assert report["max_lift_height_m"] >= 0.05
    assert report["hold_duration_s"] >= 1.0
    assert report["action_path"].endswith("(no second ActionGuard)")
    assert report["tracking_error_rad"]["p95"] <= 0.05


def test_frozen_pilot_metadata_is_complete_and_hash_consistent():
    import yaml

    metadata = yaml.safe_load((RAW / "metadata.yaml").read_text())
    required = {
        "seed", "reset_index", "objects", "initial_qpos", "candidate_failures",
        "selected_candidate", "phase_timestamps", "success_reason", "fail_reason",
        "max_lift_height_m", "hold_duration_s", "hold_window_start_s",
        "hold_window_end_s", "hold_frames", "action_guard", "scene_sha256",
        "params_sha256",
    }
    assert len(metadata["episodes"]) == 5
    for episode in metadata["episodes"]:
        assert required <= episode.keys()
        assert episode["success"] and episode["success_reason"]
        assert episode["fail_reason"] is None
        assert episode["max_lift_height_m"] >= 0.05
        assert episode["hold_duration_s"] + 1e-9 >= 1.0
        assert episode["scene_sha256"] == metadata["expert"]["scene_sha256"]
        assert episode["params_sha256"] == metadata["expert"]["params_sha256"]


def test_g5_loss_reload_and_closed_loop_evidence_is_explicit():
    losses = json.loads(
        (ROOT / "results/act_training/g5_1000_loss_summary.json").read_text()
    )
    rollout = json.loads(
        (ROOT / "results/act_rollout/g5_1000_seed_20260902.json").read_text()
    )
    assert losses["summary"]["last_train_step"] == 1000
    assert losses["summary"]["train_loss_reduction_fraction"] > 0.5
    assert losses["summary"]["eval_loss_reduction_fraction"] > 0
    assert rollout["rollout_completed"]
    assert rollout["policy"]["physics_steps"] >= 45_000
    assert rollout["policy"]["max_state_displacement_rad"] > 0.1
    assert rollout["max_lift_height_m"] > 0
    assert rollout["stabilize"]["reference_initial_state_error_max_rad"] < 0.01
    assert rollout["executor_equivalence"]["upsampler_class"].endswith("HermiteUpsampler")
    assert rollout["executor_equivalence"]["filter_class"].endswith("BiquadLowpass")
    # Task failure is an accepted, explicitly classified smoke outcome.
    assert not rollout["task_success"]
    assert rollout["failure_reason"] == "lift_hold_condition_not_met"


def test_keyboard_dataflow_regression_suite_passes():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "official/dora-openarm-keyboard/src")
    subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            str(ROOT / "official/dora-openarm-keyboard/tests/test_teleop.py"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
