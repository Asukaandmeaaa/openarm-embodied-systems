from pathlib import Path
import math

import mujoco
import numpy as np
import pytest
import yaml

from openarm_mujoco.v2 import JointResolver
from openarm_cube_pick_expert.expert import GraspCandidate, rank_candidates
from openarm_cube_pick_expert.geometry import (
    LEFT_GRIPPER_RANGE,
    RIGHT_GRIPPER_RANGE,
    combine_action,
    right_gripper_target,
    left_gripper_target_from_right,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    yaw_candidates,
)
from openarm_cube_pick_expert.guard import ActionGuard
from openarm_cube_pick_expert.success import SuccessTracker
from openarm_cube_pick_expert.task import PickCubeEvaluator
from openarm_cube_pick_expert.recording import ExpertRecorder
from openarm_cube_pick_expert.env import OpenArmPickCubeEnv


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"


def test_tcp_world_pose_matches_mjcf_composition():
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame")
    bid = model.site_bodyid[sid]; body_r = data.xmat[bid].reshape(3, 3); site_r = np.empty(9)
    mujoco.mju_quat2Mat(site_r, model.site_quat[sid]); site_r = site_r.reshape(3, 3)
    np.testing.assert_allclose(data.site_xpos[sid], data.xpos[bid] + body_r @ model.site_pos[sid], atol=1e-9)
    np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), body_r @ site_r, atol=1e-9)
    np.testing.assert_allclose(model.site_pos[sid], [-0.00143, 0, -0.138], atol=1e-9)


def test_quaternion_order_and_explicit_rz_ry_composition():
    xyzw = np.array([0.0, 0.70710678, 0.0, 0.70710678])
    np.testing.assert_allclose(xyzw_to_wxyz(xyzw), [0.70710678, 0, 0.70710678, 0])
    np.testing.assert_allclose(wxyz_to_xyzw(xyzw_to_wxyz(xyzw)), xyzw)
    yaw = 0.37
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    from openarm_cube_pick_expert.expert import MujocoGraspPlanner
    np.testing.assert_allclose(MujocoGraspPlanner.target_rotation(yaw), rz @ ry, atol=1e-12)


def test_four_yaw_candidates_and_fallback_ranking():
    values = yaw_candidates(0.2)
    assert len(values) == 4
    np.testing.assert_allclose(np.unwrap(values), 0.2 + np.arange(4) * math.pi / 2)
    failed = GraspCandidate(0, values[0], failure_reasons=["ik"])
    second = GraspCandidate(1, values[1], reachable=True, collision_free=True, score=(-0.2, 2.0))
    third = GraspCandidate(2, values[2], reachable=True, collision_free=True, score=(-0.1, 1.0))
    assert rank_candidates([failed, third, second]) == [second, third]


def test_gripper_mapping_signs_and_bimanual_order():
    target = right_gripper_target(0.040)
    assert target == pytest.approx(-0.0857142857)
    assert RIGHT_GRIPPER_RANGE[0] <= target <= RIGHT_GRIPPER_RANGE[1]
    assert LEFT_GRIPPER_RANGE == (0.0, 0.7854)
    assert left_gripper_target_from_right(target) == pytest.approx(-target)
    right = np.arange(8, dtype=np.float32); left = np.arange(8, 16, dtype=np.float32)
    np.testing.assert_array_equal(combine_action(right, left), np.arange(16, dtype=np.float32))


def make_guard(policy="clip"):
    return ActionGuard(-np.ones(16), np.ones(16), np.ones(16), np.ones(16) * 2, np.ones(16) * .1, .1, policy)


def test_action_guard_shape_dtype_nan_limits_velocity_acceleration_watchdog_reset():
    guard = make_guard()
    assert not guard.submit(np.zeros(15), 0)
    assert not guard.submit(np.r_[np.zeros(15), np.nan], 0)
    assert guard.submit(np.ones(16, dtype=np.float64) * 2, 0)
    first = guard.step(np.zeros(16), .01)
    assert first.action.dtype == np.float32 and np.all(first.action <= .1)
    second = guard.step(np.zeros(16), .02)
    assert np.all(np.abs((second.action-first.action)/.01) <= 1.0 + 1e-5)
    held = guard.step(np.ones(16)*.2, .2)
    assert held.disposition == "hold" and held.reasons == ("watchdog",)
    guard.reset(); assert guard.target is None and np.all(guard.previous_velocity == 0)


def test_action_guard_converges_to_fixed_target_without_limit_cycle():
    guard = make_guard()
    target = np.full(16, 0.4, dtype=np.float32)
    current = np.zeros(16, dtype=np.float32)
    commands = []
    for index in range(5000):
        now = index * 0.001
        assert guard.submit(target, now)
        result = guard.step(current, now)
        commands.append(result.action.copy())
        current = result.action
    commands = np.asarray(commands)
    velocity = np.diff(commands[:, 0]) / 0.001
    assert np.all(np.diff(commands[:, 0]) >= -1e-7)
    assert np.max(np.abs(velocity)) <= 1.0 + 1e-5
    assert commands[-1, 0] == pytest.approx(target[0], abs=1e-5)


def test_pick_place_fsm_requires_complete_post_grasp_sequence():
    from openarm_cube_pick_expert.expert import ExpertPhase
    from openarm_cube_pick_expert.fsm import ExpertFSM
    fsm = ExpertFSM(); fsm.reset()
    sequence = [ExpertPhase.PLAN, ExpertPhase.MOVE_PREGRASP, ExpertPhase.APPROACH,
                ExpertPhase.CLOSE, ExpertPhase.VERIFY_GRASP, ExpertPhase.LIFT,
                ExpertPhase.HOLD, ExpertPhase.TRANSPORT, ExpertPhase.PLACE,
                ExpertPhase.RELEASE, ExpertPhase.RETREAT, ExpertPhase.RETURN_HOME,
                ExpertPhase.SUCCESS]
    for index, phase in enumerate(sequence, 1):
        fsm.transition(phase, float(index), "test")
    assert fsm.phase == ExpertPhase.SUCCESS


def test_success_exact_boundary_and_ballistic_rejection():
    tracker = SuccessTracker(1.0)
    for i in range(11):
        success = tracker.update(i * .1, 1.05, .04, True, 0.0)
    assert success and tracker.hold_duration == pytest.approx(1.0)
    ballistic = SuccessTracker(1.0)
    for i in range(20):
        assert not ballistic.update(i * .1, 1.2, .2, False, 1.0)


def test_strict_success_requires_bilateral_contact_and_rejects_drop():
    task = PickCubeEvaluator(1.0)
    for i in range(12):
        assert not task.update(now_s=i*.1, cube_z=1.06, bilateral_contact=False,
            tcp_distance_m=.03, relative_drift_m=.001, cube_vertical_velocity_m_s=0)
    for i in range(10):
        assert not task.update(now_s=2+i*.1, cube_z=1.06, bilateral_contact=True,
            tcp_distance_m=.03, relative_drift_m=.001, cube_vertical_velocity_m_s=0)
    assert not task.update(now_s=3.0, cube_z=1.01, bilateral_contact=False,
        tcp_distance_m=.1, relative_drift_m=.1, cube_vertical_velocity_m_s=-.2)
    assert task.hold_duration_s == 0


def test_env_reset_is_seed_reproducible_and_step_is_actuator_only():
    env = OpenArmPickCubeEnv()
    obs1, info1 = env.reset(123)
    pose1 = info1["reset_cube_pose_wxyz"].copy()
    obs2, info2 = env.reset(123)
    np.testing.assert_allclose(info2["reset_cube_pose_wxyz"], pose1)
    assert obs1["observation.state"].shape == (16,)
    before = env.data.qpos[env.cube_qadr:env.cube_qadr+7].copy()
    env.step(obs2["observation.state"])
    # Physics may settle the cube, but step has no teleporting reset mutation.
    assert np.linalg.norm(env.data.qpos[env.cube_qadr:env.cube_qadr+3]-before[:3]) < .01


def test_episode_transaction_commit_cancel_and_crash_recovery(tmp_path):
    output = tmp_path / "dataset"
    metadata = {"tasks":[{"prompt":"Pick up the orange cube with the right arm."}]}
    recorder = ExpertRecorder(output, metadata)
    recorder.start("cancel")
    recorder.cancel()
    assert not list((output / ".transactions").iterdir())
    stale = output / ".transactions" / "interrupted"
    stale.mkdir()
    recorder = ExpertRecorder(output, metadata)
    assert not stale.exists()
    recorder.start("success")
    image = np.zeros((600,960,3), dtype=np.uint8); image[...,1] = 80
    for index in range(2):
        value = np.full(16, index*.01, dtype=np.float32)
        recorder.sample(index+1, value, value, {"ceiling":image,"wrist_right":image})
    recorder.finish(True, {"seed":123})
    loaded = yaml.safe_load((output/"metadata.yaml").read_text())
    assert loaded["episodes"] == [{"id":"0","success":True,"task_index":0,"seed":123}]
    assert (output/"episodes"/"0"/"action"/"arms"/"right"/"state.parquet").exists()


def test_openarm_camera_uses_numeric_timestamp_order(tmp_path):
    from openarm_dataset.camera import Camera
    for timestamp in (9_000_000_000, 10_000_000_000, 1_000_000_000):
        (tmp_path / f"{timestamp}.jpeg").write_bytes(b"x")
    assert Camera("test", tmp_path).load_timestamps() == [1.0, 9.0, 10.0]


def test_feature_contract_exact_two_cameras_and_dimensions():
    contract = yaml.safe_load((ROOT / "configs/collection/feature_contract.yaml").read_text())
    assert contract["task"] == "Pick up the orange cube with the right arm."
    assert contract["features"]["observation.state"]["shape"] == [16]
    assert contract["features"]["action"]["shape"] == [16]
    images = {k for k in contract["features"] if k.startswith("observation.images.")}
    assert images == {"observation.images.ceiling", "observation.images.wrist_right"}
    assert all(contract["features"][key]["shape"] == [3, 600, 960] for key in images)


def test_joint_resolver_right_then_left_and_gripper_ctrlranges():
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model); mapper = JointResolver(model)
    right = np.r_[np.zeros(7), -.2]; left = np.r_[np.zeros(7), .2]
    mapper.set_ctrl(data.ctrl, right, "right"); mapper.set_ctrl(data.ctrl, left, "left")
    assert data.ctrl[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"right_finger1_ctrl")] == pytest.approx(-.2)
    assert data.ctrl[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"left_finger1_ctrl")] == pytest.approx(.2)
