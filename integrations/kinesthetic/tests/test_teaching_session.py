import mujoco
import numpy as np
import pytest

from openarm_kinesthetic.scene.model import load_model, reset_task_ready
from openarm_kinesthetic.teaching.session import (
    DEFAULT_INTERACTIVE_SCENE,
    _RealtimeStepScheduler,
    _clear_viewer_perturbation_force,
    _configured_simulation_start_q,
    _initialize_perturbation,
    _integrate_local_rotation,
    _keyboard_jog_delta,
    _key_name,
    _mocap_rotation,
    _prepare_interactive_initialization,
    _set_interactive_right_gripper_opening,
    _sync_mocap_target_to_site,
)
from openarm_kinesthetic.teaching.virtual_handle import VirtualHandle
from openarm_kinesthetic.task.evaluator import TaskEvaluator


def test_default_scene_drives_its_own_task_evaluator_box_geometry():
    model = load_model(DEFAULT_INTERACTIVE_SCENE)
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    evaluator = TaskEvaluator.from_model(
        model, data, mapping, data.qpos[mapping.right_qpos]
    )

    assert evaluator.box_center_xy_m == pytest.approx((0.37, -0.03))
    assert evaluator.box_cube_center_z_m == pytest.approx(0.430)


def test_interactive_scene_plans_from_visible_zero_like_start_to_teaching_ready():
    model = load_model(DEFAULT_INTERACTIVE_SCENE)
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)

    config, report, waypoints, durations = _prepare_interactive_initialization(model, data, mapping)

    expected_start = _configured_simulation_start_q(config)
    np.testing.assert_allclose(data.qpos[mapping.right_qpos], expected_start)
    np.testing.assert_allclose(report["start_q"], expected_start)
    np.testing.assert_allclose(waypoints[0], expected_start)
    np.testing.assert_allclose(waypoints[-1], report["ik"]["q_teaching_ready"])
    assert not np.allclose(expected_start, report["ik"]["q_teaching_ready"])
    assert sum(durations) > 0.0


def test_realtime_scheduler_compensates_slow_viewer_frames_and_bounds_catch_up():
    scheduler = _RealtimeStepScheduler(0.001, 10.0, maximum_catch_up_s=0.05)
    assert scheduler.steps_due(10.0) == 1
    assert scheduler.steps_due(10.016) == 16
    assert scheduler.steps_due(10.216) == 50
    assert scheduler.steps_due(10.216) == 0


def test_realtime_scheduler_rejects_invalid_timing():
    import pytest

    with pytest.raises(ValueError, match="invalid realtime scheduler timing"):
        _RealtimeStepScheduler(0.0, 0.0)
    with pytest.raises(ValueError, match="invalid realtime scheduler timing"):
        _RealtimeStepScheduler(0.01, 0.0, maximum_catch_up_s=0.005)


def test_initialize_perturbation_uses_valid_camera_frusta():
    model = load_model()
    data = mujoco.MjData(model)
    reset_task_ready(model, data)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    perturb.select = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "openarm_right_ee_base_link",
    )

    _initialize_perturbation(model, data, camera, option, perturb)

    assert perturb.scale > 0
    assert perturb.select >= 0


def test_interactive_gripper_opening_sets_both_coupled_fingers():
    model = load_model()
    data = mujoco.MjData(model)
    reset_task_ready(model, data)
    _set_interactive_right_gripper_opening(model, data, -0.35)
    for name in ("openarm_right_finger_joint1", "openarm_right_finger_joint2"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert data.qpos[model.jnt_qposadr[joint_id]] == -0.35


def test_clear_viewer_perturbation_force_keeps_generalized_force():
    model = load_model()
    data = mujoco.MjData(model)
    data.xfrc_applied[:] = 3.0
    data.qfrc_applied[:] = 4.0

    _clear_viewer_perturbation_force(data)

    assert not data.xfrc_applied.any()
    assert (data.qfrc_applied == 4.0).all()


def test_keyboard_cartesian_jog_mapping_is_world_frame():
    np.testing.assert_allclose(_keyboard_jog_delta(_key_name(265), 0.01), [0.01, 0, 0])
    np.testing.assert_allclose(_keyboard_jog_delta(_key_name(264), 0.01), [-0.01, 0, 0])
    np.testing.assert_allclose(_keyboard_jog_delta(_key_name(263), 0.01), [0, 0.01, 0])
    np.testing.assert_allclose(_keyboard_jog_delta(_key_name(262), 0.01), [0, -0.01, 0])
    np.testing.assert_allclose(_keyboard_jog_delta("U", 0.01), [0, 0, 0.01])
    np.testing.assert_allclose(_keyboard_jog_delta("D", 0.01), [0, 0, -0.01])
    assert _keyboard_jog_delta("O", 0.01) is None


def test_operator_mocap_target_is_noncolliding_and_tracks_grasp_site():
    model = load_model()
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_handle_target")
    mocap = int(model.body_mocapid[body])
    geom_start = int(model.body_geomadr[body])
    geom_stop = geom_start + int(model.body_geomnum[body])

    _sync_mocap_target_to_site(data, mocap, mapping.grasp_site)

    np.testing.assert_allclose(data.mocap_pos[mocap], data.site_xpos[mapping.grasp_site])
    np.testing.assert_allclose(_mocap_rotation(data, mocap), data.site_xmat[mapping.grasp_site].reshape(3, 3))
    assert np.all(model.geom_group[geom_start:geom_stop] == 5)
    assert not model.geom_contype[geom_start:geom_stop].any()
    assert not model.geom_conaffinity[geom_start:geom_stop].any()


def test_redundant_posture_torque_is_projected_out_of_task_jacobian():
    model = load_model()
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    preferred = data.qpos[mapping.right_qpos].copy()
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, mapping.grasp_site)
    jacobian = np.vstack((jacp[:, mapping.right_dof], jacr[:, mapping.right_dof]))
    null_direction = np.linalg.svd(jacobian)[2][-1]
    data.qpos[mapping.right_qpos] = preferred + 0.05 * null_direction
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    handle = VirtualHandle(
        mapping.grasp_site,
        right_qpos=mapping.right_qpos,
        right_dofs=mapping.right_dof,
        preferred_q=preferred,
    )
    handle.engage_at_current(data)
    handle.apply(model, data)
    mujoco.mj_jacSite(model, data, jacp, jacr, mapping.grasp_site)
    jacobian = np.vstack((jacp[:, mapping.right_dof], jacr[:, mapping.right_dof]))

    assert np.linalg.norm(handle.last_nullspace_torque) > 0.05
    assert np.linalg.norm(jacobian @ handle.last_nullspace_torque) < 1e-3


def test_local_rotation_integration_is_finite_orthonormal_and_local():
    initial = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    result = _integrate_local_rotation(initial, np.asarray([0, 0, 1.0]), 0.1)
    np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-12)
    assert np.linalg.det(result) > 0.999999
    assert not np.allclose(result, initial)
