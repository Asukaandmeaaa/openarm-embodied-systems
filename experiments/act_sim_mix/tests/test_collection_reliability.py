"""Crash/restart and exclusive writer regressions; no rendering or hardware."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.act_sim_mix import sim_collect as collect


def test_resume_preserves_creation_time_and_rejects_changed_source(tmp_path):
    path = tmp_path / 'collection'
    expected = {'created_ns': 1, 'code_sha256': {'source': 'a'}}
    with collect.prepare_output(path, expected):
        pass
    before = (path / 'run.json').read_bytes()
    with collect.prepare_output(path, {**expected, 'created_ns': 2}):
        assert (path / 'run.json').read_bytes() == before
    with pytest.raises(ValueError, match='different code/config/scene'):
        collect.prepare_output(path, {**expected, 'code_sha256': {'source': 'b'}})
    assert (path / 'run.json').read_bytes() == before


def test_collection_lock_survives_setup_and_is_released(tmp_path):
    path = tmp_path / 'collection'
    probe = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
             'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
    with collect.prepare_output(path, {'created_ns': 1}):
        result = subprocess.run([sys.executable, '-c', probe, str(path / '.collector.lock')], capture_output=True)
        assert result.returncode != 0
        with pytest.raises(SystemExit, match='Another collector'):
            collect.prepare_output(path, {'created_ns': 99})
        assert json.loads((path / 'run.json').read_text())['created_ns'] == 1
    assert subprocess.run([sys.executable, '-c', probe, str(path / '.collector.lock')]).returncode == 0


def test_run_outputs_do_not_change_source_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, 'HERE', tmp_path)
    (tmp_path / 'source.py').write_text('x=1')
    before = collect.code_hashes()
    for folder in ('runs', 'eval', 'artifacts', 'tests'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'result.json').write_text('{}')
    assert collect.code_hashes() == before
    (tmp_path / 'source.py').write_text('x=2')
    assert collect.code_hashes() != before


def test_recovery_resume_skips_failed_and_interrupted_attempts(tmp_path, monkeypatch):
    args = collect.build_parser().parse_args([
        'recovery', '--output', str(tmp_path / 'raw'), '--families', '1',
        '--types', 'bad_height', '--retries', '3', '--max-attempts', '3'])
    seen = []
    def fail_then_interrupt(output, domain, seed, cfg, **kw):
        seen.append(seed)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return {'success': False, 'failure': 'unreachable'}
    monkeypatch.setattr(collect, 'run_one', fail_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect.cmd_recovery(args)
    assert len(list((args.output / 'attempts').iterdir())) == 2
    def succeed(output, domain, seed, cfg, **kw):
        seen.append(seed)
        entry = {'success': True, 'episode_id': kw['episode_id'], 'family_id': kw['family_id'],
                 'domain': domain, 'recovery_type': kw['recovery_type'], 'frames': 5, 'valid_from_frame': 0}
        collect.append_jsonl(output / 'episodes/index.jsonl', entry)
        return entry
    monkeypatch.setattr(collect, 'run_one', succeed)
    assert collect.cmd_recovery(args) == 0
    assert seen == [args.seed + 2, args.seed + 102, args.seed + 202]
    assert collect.cmd_recovery(args) == 0  # no duplicate successful family/type
    assert len(seen) == 3


def test_completed_episode_recovers_index_publication(tmp_path):
    raw = tmp_path / 'raw'
    with collect.prepare_output(raw, {'created_ns': 1}):
        episode = raw / 'episodes/000000'
        episode.mkdir()
        meta = {'episode_id': 0, 'frames': 1, 'success': True, 'domain': 'normal'}
        (episode / 'metadata.json').write_text(json.dumps(meta))
        (episode / 'replay.json').write_text('{"success": true}')
        (episode / 'numeric.npz').write_bytes(b'preserve')
        for camera in collect.CAMERAS:
            folder = episode / 'cameras' / camera
            folder.mkdir(parents=True)
            (folder / '000000.jpg').write_bytes(b'preserve')
        collect.reconcile_committed_episodes(raw)
        before = (raw / 'episodes/index.jsonl').read_bytes()
        collect.reconcile_committed_episodes(raw)
        assert (raw / 'episodes/index.jsonl').read_bytes() == before
        assert json.loads(before)['episode_id'] == 0


def test_torn_final_append_is_retained_without_masking_interior_corruption(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    path.write_bytes(b'{"attempt": 0}\n{"attempt":')
    collect.recover_jsonl_tail(path)
    assert path.read_bytes() == b'{"attempt": 0}\n'
    evidence = list(tmp_path.glob('attempts.jsonl.interrupted_*'))
    assert len(evidence) == 1 and evidence[0].read_bytes() == b'{"attempt":'
    path.write_bytes(b'bad json\n{"attempt": 1}\n')
    with pytest.raises(json.JSONDecodeError):
        collect.recover_jsonl_tail(path)
