import numpy as np
import pytest

from openarm_state_join.main import join_state


def test_right_then_left_order():
    result = join_state(np.arange(8), np.arange(8, 16))
    np.testing.assert_array_equal(result, np.arange(16, dtype=np.float32))


def test_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        join_state(np.arange(7), np.arange(8))
