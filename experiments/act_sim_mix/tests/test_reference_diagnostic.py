"""Oracle tests distinguish saved initialization, valid suffix and execution path."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.act_sim_mix import reference_diagnostic as ref
from experiments.act_sim_mix.contracts import digest


def source_scene(tmp_path):
    folder = tmp_path / 'raw'
    folder.mkdir()
    actions = np.zeros((7, 16), np.float32)
    actions[:, 0] = np.arange(7) / 100
    actions[:, [7, 15]] = -.02
    np.savez(folder / 'numeric.npz', action=actions, state=actions,
             valid=np.arange(7) >= 2)
    return {'id': 'source', 'kind': 'recovery', 'recovery_type': 'bad_height',
            'collection_root': str(tmp_path), 'source_episode': 'raw',
            'takeover_frame': 2, 'seed': 0, 'jitter': {},
            'numeric_sha256': digest(folder / 'numeric.npz')}


def test_recorded_initialization_is_explicit_and_does_not_change_source_kind():
    normal = {'id': 'home', 'kind': 'home', 'start': 'deployment_zero'}
    prepared = ref.recorded_initialization(normal)
    assert prepared['source_kind'] == 'home' and prepared['takeover_frame'] == 0
    assert prepared.get('recovery_type') is None  # Must still require full lift/hold.
    assert normal == {'id': 'home', 'kind': 'home', 'start': 'deployment_zero'}
    correction = ref.recorded_initialization({'kind': 'recovery', 'takeover_frame': 24, 'recovery_type': 'closed_approach'})
    assert correction['takeover_frame'] == 24


def test_reference_excludes_failure_prefix_and_verifies_hash(tmp_path):
    scene = source_scene(tmp_path)
    actions, initial, available = ref.load_reference(scene, max_frames=3)
    assert len(actions) == 3 and available == 5
    assert actions[0, 0] == pytest.approx(.02)
    assert initial[0] == pytest.approx(.02)
    with pytest.raises(ValueError, match='valid suffix'):
        ref.load_reference({**scene, 'takeover_frame': 0})
    with pytest.raises(ValueError, match='Changed'):
        ref.load_reference({**scene, 'numeric_sha256': 'wrong'})


@pytest.mark.parametrize('mode', ['zoh', 'chunk'])
def test_oracle_uses_only_saved_targets_without_policy_or_state_teleports(tmp_path, monkeypatch, mode):
    scene = source_scene(tmp_path)
    actions, initial, _ = ref.load_reference(scene)
    class World:
        def __init__(self): self.data = SimpleNamespace(time=0.); self.targets = []; self.waits = []; self.closed = False
        def state16(self): return initial.copy()
        def step(self, target, phase): self.targets.append(target.copy()); self.data.time += .001
        def hold_for(self, duration, phase): self.waits.append(duration); self.data.time += duration
        def close(self): self.closed = True
    world = World()
    initialized = []
    monkeypatch.setattr(ref, 'DeploymentPolicyWorld', lambda *a, **kw: world)
    monkeypatch.setattr(ref, 'set_recovery_start', lambda *args: initialized.append(True))
    monkeypatch.setattr(ref, 'task_result', lambda *a, **kw: {'success': False, 'failure_type': 'no_lift'})
    chunks = []
    def execute(w, prediction, steps, lowpass, **kwargs):
        chunks.append((prediction.copy(), steps))
        w.data.time += steps / 30
        return {'clipped_actions': 0}
    monkeypatch.setattr(ref, 'execute_block', execute)
    result = ref.oracle_rollout(scene, {}, mode=mode, steps=3, observation_wait_s=1/30,
                               reference_wait_s=.04)
    assert len(initialized) == 1 and world.closed
    assert result['initial_state_max_abs_error_rad'] == 0
    assert result['reference_frames_consumed'] == 5
    assert result['diagnostic_complete'] and result['reference_sequence_completed']
    if mode == 'zoh':
        assert not chunks and not world.waits
        assert world.targets[0][0] == pytest.approx(actions[0, 0])
        assert world.targets[-1][0] == pytest.approx(actions[-1, 0])
    else:
        assert [steps for _, steps in chunks] == [3, 2]
        assert chunks[1][0][0, 0] == pytest.approx(actions[3, 0])
        assert chunks[1][0][-1, 0] == pytest.approx(actions[-1, 0])
        assert world.waits == pytest.approx([1/30 + .04] * 2)
