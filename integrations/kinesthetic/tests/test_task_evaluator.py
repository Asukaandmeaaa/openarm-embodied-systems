import numpy as np

from openarm_kinesthetic.task import EvaluationSample, TaskEvaluator


READY = np.asarray([0, 0, 0, np.pi / 2, 0, 0, 0])


def sample(t, xyz, contacts, q=READY, velocity=(0, 0, 0)):
    return EvaluationSample(t, xyz, velocity, contacts, tuple(q))


def test_full_success_requires_ordered_hold_place_release_home_and_tail() -> None:
    evaluator = TaskEvaluator(
        reset_cube_z_m=0.421, ready_joints=READY,
        box_center_xy_m=(0.38, -0.08), box_safe_half_range_m=(0.04, 0.04),
        box_cube_center_z_m=0.430,
    )
    evaluator.update(sample(0.0, (0.36, -0.10, 0.421), (False, False)))
    evaluator.update(sample(0.1, (0.36, -0.10, 0.471), (True, True)))
    for t in np.arange(0.2, 2.21, 0.1):
        evaluator.update(sample(float(t), (0.36, -0.10, 0.481), (True, True)))
    for t in np.arange(2.3, 2.91, 0.1):
        evaluator.update(sample(float(t), (0.38, -0.08, 0.430), (True, True)))
    result = evaluator.update(sample(3.0, (0.38, -0.08, 0.430), (False, False), q=READY + 0.2))
    assert result.task_success
    assert not result.full_sequence_success
    for t in np.arange(3.1, 5.31, 0.1):
        result = evaluator.update(sample(float(t), (0.38, -0.08, 0.430), (False, False)))
    assert result.full_sequence_success
    assert result.grasp_count == 1


def test_second_grasp_is_hard_failure() -> None:
    evaluator = TaskEvaluator(
        reset_cube_z_m=0.421, ready_joints=READY,
        box_center_xy_m=(0.38, -0.08), box_safe_half_range_m=(0.04, 0.04),
        box_cube_center_z_m=0.430,
    )
    evaluator.update(sample(0.0, (0.36, -0.1, 0.421), (True, True)))
    evaluator.update(sample(0.1, (0.36, -0.1, 0.421), (False, False)))
    result = evaluator.update(sample(0.2, (0.36, -0.1, 0.421), (True, True)))
    assert "more_than_one_effective_grasp" in result.failures
