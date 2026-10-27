import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('button_replay', ROOT/'scripts/button_play_recorded_motion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
MANIFEST = ROOT/'configs/replay_compare3_20261020.json'


def test_real_selection():
    paths = m.playlist_paths(MANIFEST)
    assert len(paths) == 3
    assert paths[0].name == paths[1].name
    assert paths[0] != paths[1]
    assert m.playlist_paths(MANIFEST, 3) == [paths[2]]


@pytest.mark.parametrize('index', [0, -1, 4])
def test_invalid_index(index):
    with pytest.raises(ValueError):
        m.playlist_paths(MANIFEST, index)


def test_plan_cannot_request_motion():
    command = m.player_command(Path('/tmp/example.json'), plan_only=True)
    assert '--plan-only' in command
    assert '--allow-real-execution' not in command
    assert '--allow-real-execution' in m.player_command(Path('/tmp/example.json'))


def test_changed_hash_rejected(tmp_path):
    import json
    value = json.loads(MANIFEST.read_text())
    value['entries'][0]['sha256'] = 'changed'
    manifest = tmp_path/'playlist.json'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='differs'):
        m.playlist_paths(manifest)
