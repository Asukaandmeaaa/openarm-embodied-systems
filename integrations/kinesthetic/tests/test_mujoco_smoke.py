import math

from openarm_kinesthetic.validation.mujoco_smoke import MODELS, run_smoke


def test_three_models_and_required_names_load() -> None:
    reports = run_smoke()
    assert [item["model"] for item in reports] == list(MODELS)
    assert all(item["right_joint_count"] == 7 for item in reports)
    assert all(item["right_gripper_joint"] == "openarm_right_finger_joint1" for item in reports)
    assert all(item["wrist_camera"] == "camera_wrist_right" for item in reports)
    assert all(item["grasp_site"] == "right_grasp_frame" for item in reports)
    assert all(item["model_joint_order"] == "left_then_right" for item in reports)
    assert all(item["dataset_order"] == "right8_then_left8" for item in reports)


def test_pedestal_home_j4_is_task_ready_not_motor_zero() -> None:
    reports = run_smoke()
    pedestal = next(item for item in reports if item["model"] == "v2/pedestal/pedestal.xml")
    assert math.isclose(pedestal["home_j4_rad"][1], math.pi / 2, abs_tol=2e-6)
    assert not math.isclose(pedestal["home_j4_rad"][1], 0.0, abs_tol=1e-9)
