"""Offline admission checks must never command, align, or mask unsafe starts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("act_trial_preflight", Path(__file__).resolve().parents[1] / "scripts/act_trial/preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class Scene:
    def __init__(self, collision_at=None):
        self.poses = []
        self.collision_at = collision_at

    def put(self, q):
        self.poses.append(q.copy())
        return [{"distance_m": -0.001}] if len(self.poses) == self.collision_at else []


def contract():
    home = np.array([-0.145, -0.015, -0.006, 1.375, -0.003, -0.198, -0.122, -0.35])
    q = np.r_[home, np.zeros(8)]
    limits = np.tile([-3.0, 3.0], (16, 1))
    return home, q, limits


def test_old_tracking_profile_cannot_silently_load_new_semantics():
    from scripts.act_trial.profiles import execution_profile, profile_from_config
    cfg = {'execution_profile': execution_profile('task')}
    assert profile_from_config(cfg)['tracking_policy_version'] == 3
    cfg['execution_profile']['tracking_policy_version'] = 2
    with pytest.raises(ValueError, match='prepare a new trial'):
        profile_from_config(cfg)
    del cfg['execution_profile']['tracking_policy_version']
    with pytest.raises(ValueError, match='prepare a new trial'):
        profile_from_config(cfg)


def test_home_checks_every_corner_and_preserves_left_and_gripper():
    home, q, limits = contract()
    q[8:] = np.arange(8) * 0.01
    original = q.copy()
    scene = Scene()
    report = preflight.inspect_start(q, home, limits, scene)
    assert report["supplied_start_checks_passed"]
    assert len(scene.poses) == 129
    assert len({tuple(p[:7]) for p in scene.poses[1:]}) == 128
    for p in scene.poses:
        np.testing.assert_array_equal(p[7:], q[7:])
    np.testing.assert_array_equal(q, original)
    assert report["continuous_collision_proof"] is False


@pytest.mark.parametrize("joint", [0, 3, 7])
def test_home_mismatch_refuses_without_alignment(joint):
    home, q, limits = contract()
    q[joint] += 0.031
    report = preflight.inspect_start(q, home, limits, Scene())
    assert not report["supplied_start_checks_passed"]
    assert "right_HOME_mismatch" in report["failures"]


@pytest.mark.parametrize("value", [[0.0] * 15, [float("nan")] * 16, [float("inf")] * 16])
def test_invalid_state_rejected(value):
    with pytest.raises(ValueError, match="16 finite"):
        preflight.finite_start(value)


def test_named_state_requires_exact_order():
    with pytest.raises(ValueError, match="joint_names"):
        preflight.finite_start({"joint_names": list(reversed(preflight.NAMES)), "positions_rad": [0.0] * 16})


def test_left_limit_is_diagnostic_but_left_remains_collision_context():
    home, q, limits = contract()
    limits[11] = [0.01, 2.0]
    report = preflight.inspect_start(q, home, limits, Scene())
    assert report["supplied_start_checks_passed"]
    assert report["left_limit_diagnostics_nonblocking"] == [preflight.NAMES[11]]


def test_collision_at_a_late_corner_refuses():
    home, q, limits = contract()
    report = preflight.inspect_start(q, home, limits, Scene(collision_at=129))
    assert not report["supplied_start_checks_passed"]
    assert "sampled_neighborhood_collision" in report["failures"]


def test_corner_outside_real_limits_refuses_even_if_home_is_valid():
    home, q, limits = contract()
    limits[0, 1] = home[0] + 0.05
    report = preflight.inspect_start(q, home, limits, Scene())
    assert not report["supplied_start_checks_passed"]
    assert report["right_limit_violations"] == []
    assert "sampled_neighborhood_joint_limit" in report["failures"]


def test_config_never_authorizes_or_launches_hardware():
    cfg = preflight.trial_settings()
    assert cfg["physical_execution_authorized"] is False
    assert cfg["hardware_launch"] is None
    assert cfg["automatic_home_or_start_alignment"] is False
    assert cfg["left_command_output"] is False
    assert cfg["duration_s"] == 3.0
    assert cfg["arm_envelope_rad"] == 0.04
    assert cfg["max_velocity_rad_s"] == 0.25
    assert cfg["max_acceleration_rad_s2"] == 0.25
    assert cfg["gripper_max_motion_rad"] == 0.0


def test_task_profile_is_explicit_manual_stop_and_has_no_home_envelope():
    from scripts.act_trial.profiles import profile_from_config
    cfg = preflight.trial_settings('task')
    assert cfg['duration_s'] is None and cfg['arm_envelope_rad'] is None
    assert cfg['execution_scope'] == 'right_arm_and_gripper'
    assert cfg['physical_execution_authorized'] is False
    assert profile_from_config(cfg)['gripper_limits_rad'] == [-.35,-.10]
    cfg['execution_profile']['duration_s'] = 3600
    with pytest.raises(ValueError, match='profile missing/changed'): profile_from_config(cfg)


def test_task_skips_collision_scene_but_retains_home_and_limit_checks():
    home,q,limits=contract()
    limits[0,1]=home[0]+.05
    scene=Scene(collision_at=1)
    result=preflight.inspect_start(q,home,limits,scene,preflight.trial_settings('task'))
    assert result['supplied_start_checks_passed']
    assert not result['neighborhood_blocks_start']
    assert result['collision_checks_enabled'] is False
    assert scene.poses == []
    assert result['neighborhood_corner_collisions'] == []
    assert result['neighborhood_corner_limit_violations']
    # HOME and actual joint limits still block even without any scene object.
    q[0] += .04
    result=preflight.inspect_start(q,home,limits,None,preflight.trial_settings('task'))
    assert result['failures']==['right_HOME_mismatch']
    q[:8]=home;limits[0,1]=home[0]-.01
    result=preflight.inspect_start(q,home,limits,None,preflight.trial_settings('task'))
    assert result['failures']==['right_joint_limit']
