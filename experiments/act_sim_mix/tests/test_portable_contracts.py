"""Fresh-checkout path handling, attribution and discarded-action semantics."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.act_sim_mix.paths import WORKSPACE, recorded_path
from experiments.act_sim_mix.sim_reference import BiquadLowpass, prepare_chunk
from experiments.act_sim_mix import sim_reference
from experiments.act_sim_mix.sim_expert.world import SCENE


def test_recorded_paths_require_explicit_relocation(monkeypatch):
    monkeypatch.delenv('OPENARM_ACT_PATH_MAP',raising=False)
    assert recorded_path('outputs/run') == WORKSPACE / 'outputs/run'
    assert recorded_path('/old/raw') == Path('/old/raw')
    monkeypatch.setenv('OPENARM_ACT_PATH_MAP',json.dumps({'/old':'/new','/old/specific':'/other'}))
    assert recorded_path('/old/specific/raw') == Path('/other/raw')
    assert recorded_path('/old/raw') == Path('/new/raw')


def test_upstream_reference_classes_are_verbatim_without_runtime_imports():
    upstream = WORKSPACE / 'official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py'
    def classes(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef)}
    source,copy = classes(upstream),classes(Path(sim_reference.__file__))
    for name in ('HermiteUpsampler','BiquadLowpass'):
        assert source[name] == copy[name]
    assert SCENE.is_file() and SCENE.is_relative_to(WORKSPACE)


def test_100_step_chunk_discards_tail_before_interpolation():
    from types import SimpleNamespace
    q = np.zeros((100,16),np.float32)
    q[:50] = np.arange(50)[:,None]/100
    q[50:] = 999
    event={'value':SimpleNamespace(to_pylist=lambda:q.tolist()),'metadata':{'interval':33333333}}
    offsets,values,duration = prepare_chunk(event,50,100)
    assert duration == 50*33333333 and len(offsets)==417
    np.testing.assert_allclose(values[-1],q[49])
    assert values.max()<1
    lowpass=BiquadLowpass(250,15)
    for value in values: lowpass.step(value)
    assert np.any(lowpass.step(np.zeros(16))!=0)
    with pytest.raises(ValueError):prepare_chunk(event,1,100)
