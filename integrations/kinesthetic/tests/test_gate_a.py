import numpy as np

from openarm_kinesthetic.scene.model import load_model, reset_task_ready, ros_optical_pose_to_mujoco_camera
from openarm_kinesthetic.validation.gate_a import validate_gate_a


def test_real_table_scene_geometry_and_contract() -> None:
    report = validate_gate_a(render=False)
    assert report["status"] == "PASS"
    assert report["provisional"] is True
    assert report["table_top_full_size_m"][:2] == [0.6, 0.4]
    assert report["black_box_collision_geoms"] == 5
    assert report["dataset_state"]["shape"] == [16]
    assert report["arm_collision_geom_counts"]["left"] > 0


def test_model_map_preserves_right8_then_left8() -> None:
    import mujoco

    model = load_model()
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    vector = mapping.dataset_vector(data.qpos, right_gripper_command=-0.25)
    assert vector.shape == (16,)
    assert vector.dtype == np.float32
    assert vector[7] == np.float32(-0.25)
    assert np.allclose(vector[:7], data.qpos[mapping.right_qpos])
    assert np.allclose(vector[8:15], data.qpos[mapping.left_qpos])


def test_ros_optical_camera_conversion_is_finite_and_normalized() -> None:
    pos, quat = ros_optical_pose_to_mujoco_camera([1, 2, 3], [0, 0, 0, 1])
    assert np.allclose(pos, [1, 2, 3])
    assert np.isclose(np.linalg.norm(quat), 1.0)
