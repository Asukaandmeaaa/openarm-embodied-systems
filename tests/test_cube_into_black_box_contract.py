from openarm_cube_into_black_box_expert.contract import (
    BLACK_BOX_CENTER_XY_M,
    BLACK_BOX_CUBE_CENTER_Z_M,
    CUBE_CENTER_XY_M,
    TASK_PROMPT,
    CubeIntoBlackBoxEvaluator,
    TaskContract,
)


def test_contract_has_exact_prompt_and_fixed_visible_target():
    contract = TaskContract()
    metadata = contract.metadata_template()
    assert TASK_PROMPT == "Put the blue cube inside the black box, then return both arms home."
    assert metadata["task"] == TASK_PROMPT
    assert metadata["target"]["fixed"] is True
    assert metadata["target"]["visible"] is True
    assert tuple(metadata["target"]["center_xy_m"]) == BLACK_BOX_CENTER_XY_M


def test_reset_randomization_contract_is_bounded():
    contract = TaskContract()
    assert contract.cube_reset_is_valid((*CUBE_CENTER_XY_M, 1.05), 0.0)
    assert contract.cube_reset_is_valid((0.38, -0.32, 1.05), 0.2617993877)
    assert not contract.cube_reset_is_valid((0.381, -0.28, 1.05), 0.0)
    assert not contract.cube_reset_is_valid((0.33, -0.321, 1.05), 0.0)
    assert not contract.cube_reset_is_valid((*CUBE_CENTER_XY_M, 1.05), 0.27)


def test_success_requires_ordered_gates_and_static_tail():
    evaluator = CubeIntoBlackBoxEvaluator()
    evaluator.note_grasp(0.1, True)
    for i in range(21):
        hold_complete = evaluator.update_lift_hold(
            now_s=0.2 + i * 0.1,
            cube_z_m=1.105,
            bilateral_contact=True,
            relative_drift_m=0.001,
            cube_vertical_velocity_m_s=0.0,
        )
    assert hold_complete
    for i in range(6):
        placement_complete = evaluator.update_placement(
            now_s=2.4 + i * 0.1,
            cube_xyz=(*BLACK_BOX_CENTER_XY_M, BLACK_BOX_CUBE_CENTER_Z_M),
        )
    assert placement_complete
    evaluator.note_release(3.0, True)
    for i in range(11):
        evaluator.update_home(now_s=3.1 + i * 0.1, max_joint_error_rad=0.01)
    for i in range(31):
        evaluator.update_static_tail(now_s=4.2 + i * 0.1, max_joint_delta_rad=0.001)
    assert evaluator.success
    assert evaluator.grasp_count == 1


def test_bad_order_and_second_grasp_are_rejected():
    evaluator = CubeIntoBlackBoxEvaluator()
    evaluator.note_grasp(0.1, True)
    try:
        evaluator.note_grasp(0.2, True)
    except ValueError as exc:
        assert "exactly one grasp" in str(exc)
    else:
        raise AssertionError("second grasp was accepted")
    try:
        evaluator.note_release(0.3, True)
    except ValueError as exc:
        assert "stable in-box" in str(exc)
    else:
        raise AssertionError("release before placement was accepted")
