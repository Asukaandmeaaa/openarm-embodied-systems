import math

import mujoco
import numpy as np

from dora_openarm_mujoco.main import _randomize_free_qpos


def yaw_from_quat_wxyz(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    matrix = matrix.reshape(3, 3)
    return math.atan2(matrix[1, 0], matrix[0, 0])


def test_seeded_xy_and_yaw_stay_inside_frozen_ranges():
    center = np.array([0.45, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0])
    rng_a = np.random.default_rng(20260902)
    rng_b = np.random.default_rng(20260902)
    samples_a = [
        _randomize_free_qpos(center, 0.02, math.radians(15), rng_a)
        for _ in range(100)
    ]
    samples_b = [
        _randomize_free_qpos(center, 0.02, math.radians(15), rng_b)
        for _ in range(100)
    ]
    np.testing.assert_allclose(samples_a, samples_b)
    for sample in samples_a:
        assert np.max(np.abs(sample[:2] - center[:2])) <= 0.02
        assert abs(yaw_from_quat_wxyz(sample[3:7])) <= math.radians(15) + 1e-9
