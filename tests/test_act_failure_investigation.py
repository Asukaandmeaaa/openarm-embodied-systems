import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).parents[1]/'scripts/investigate_act_failures.py'
spec = importlib.util.spec_from_file_location('investigation', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_phase_requires_sustained_transition():
    assert m.sustained_index(np.array([False, True, False, True, True, True]), count=3) == 3
    assert m.sustained_index(np.array([True, True, False]), count=3) is None


def test_release_search_starts_after_close():
    assert m.sustained_index(np.array([True]*6+[False]*6+[True]*6), start=6) == 12


def test_empty_quantiles():
    assert m.quantiles([]) == {}
