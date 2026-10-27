import math

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SPEC = spec_from_file_location(
    "act_trajectory",
    Path(__file__).parents[1]
    / "ros2_ws/src/openarm_kinesthetic_recorder/openarm_kinesthetic_recorder/act_trajectory.py",
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
OfficialActShaper = _MODULE.OfficialActShaper
max_abs_difference = _MODULE.max_abs_difference


def _chunk(offset):
    return [[offset + 0.01 * i + 0.001 * j for j in range(8)] for i in range(10)]


def test_official_shaper_produces_250_hz_points_and_preserves_endpoints():
    shaper = OfficialActShaper()
    output = shaper.shape(_chunk(0.0))
    assert len(output) == 76
    assert all(len(row) == 8 for row in output)
    assert all(math.isfinite(value) for row in output for value in row)
    assert max(abs(output[0][j] - _chunk(0.0)[0][j]) for j in range(8)) < 1e-9


def test_filter_state_reduces_cross_chunk_first_point_jump():
    shaper = OfficialActShaper()
    first = shaper.shape(_chunk(0.0))
    second = shaper.shape(_chunk(0.5))
    raw_jump = max(abs(_chunk(0.5)[0][j] - _chunk(0.0)[-1][j]) for j in range(8))
    filtered_jump = max(abs(second[0][j] - first[-1][j]) for j in range(8))
    assert filtered_jump < raw_jump


def test_smooth_30hz_chunks_expand_to_a_continuous_250hz_stream():
    shaper = OfficialActShaper()
    first = _chunk(0.0)
    second = [[row[j] + 0.002 * (j + 1) for j in range(8)] for row in first]
    first_dense = shaper.shape(first)
    second_dense = shaper.shape(second)

    assert len(first_dense) == 76
    assert len(second_dense) == 76
    assert max_abs_difference(second_dense[0], first_dense[-1]) < 0.01


def test_abrupt_raw_chunk_change_remains_visible_alongside_filtered_target():
    shaper = OfficialActShaper()
    first = _chunk(0.0)
    abrupt = _chunk(0.8)
    first_dense = shaper.shape(first)
    abrupt_dense = shaper.shape(abrupt)
    raw_jump = max_abs_difference(abrupt[0], first[-1])
    filtered_jump = max_abs_difference(abrupt_dense[0], first_dense[-1])

    assert raw_jump > 0.7
    assert 0.0 < filtered_jump < raw_jump


def test_stop_reset_does_not_leak_filter_state_into_the_next_run():
    shaper = OfficialActShaper()
    initial = _chunk(0.0)
    shaper.shape(_chunk(0.7))
    shaper.reset()
    after_stop = shaper.shape(initial)
    fresh = OfficialActShaper().shape(initial)

    for actual, expected in zip(after_stop, fresh):
        assert actual == expected
