"""Offline MuJoCo parity, including colliding poses; no ROS/CAN I/O."""
import json
import numpy as np
import pytest

from scripts.act_trial.preflight import DEFAULT_BUNDLE
from scripts.home_replay.simulate import Scene


def test_position_backend_preserves_contacts():
    pytest.importorskip('mujoco')
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    reference = Scene(bundle)
    optimized = Scene(bundle, position_only=True)
    rng = np.random.default_rng(20261019)
    limits = reference.model.jnt_range[reference.joint_ids]
    contact_count = 0
    for _ in range(250):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        expected, actual = reference.put(q), optimized.put(q)
        contact_count += bool(expected)
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            assert a['pair'] == b['pair']
            assert a['failure_labels'] == b['failure_labels']
            assert a['distance_m'] == pytest.approx(b['distance_m'], abs=1e-12)
    assert contact_count > 0, 'Parity test must exercise collision rejection too'
