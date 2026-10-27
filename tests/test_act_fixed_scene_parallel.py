"""Exercise process isolation, admission limits and coordinator-only commits."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import collect_act_fixed_scene_parallel as parallel


def small_attempt(args, attempt, fidelity):
    # Complete later seeds first to exercise deterministic coordinator admission.
    time.sleep(.02 * max(0, 4 - attempt))
    randomization = {'table_translation_xy_m': [0, 0], 'table_yaw_deg': 0,
                     'cube_yaw_relative_table_deg': 0}
    entry = None
    if attempt != 1:
        writer = parallel.ExpertRecorder(args.output / '.parallel_work' / f'{attempt:06d}', {}, frame_shape=(2, 2, 3))
        writer.start(attempt)
        for stamp in (0, 33_333_333):
            writer.sample(stamp, np.zeros(16), np.ones(16),
                          {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in ('ceiling', 'wrist_right')})
        writer.finish(True, {'source_trajectory_trial_id': f'sim_fixed_{args.seed + attempt}'})
        entry = writer.entries[0]
    payload = {'record': {'attempt': attempt, 'seed': args.seed + attempt,
                         'success': entry is not None, 'randomization': randomization,
                         'episode_id': None}, 'entry': entry}
    parallel.atomic_json(args.output / 'attempts' / f'{attempt:06d}' / 'parallel_ready.json', payload)
    return payload


def arguments(output, **changes):
    defaults = dict(output=output, workers=3, episodes=3, max_attempts=8, seed=42,
                    validate_only=False, table_xy_jitter_m=.02, table_yaw_jitter_deg=5,
                    cube_xy_jitter_m=.01, cube_yaw_jitter_deg=30)
    return argparse.Namespace(**(defaults | changes))


def test_spawned_workers_commit_exact_target_and_resume(tmp_path):
    args = arguments(tmp_path)
    recorder = parallel.ExpertRecorder(tmp_path, {}, frame_shape=(2, 2, 3))
    parallel.collect(args, recorder, {}, worker=small_attempt)
    rows = [json.loads(line) for line in (tmp_path / 'attempts.jsonl').read_text().splitlines()]
    assert [row['attempt'] for row in rows] == [0, 1, 2, 3]
    assert [row['episode_id'] for row in rows] == [0, None, 1, 2]
    assert sorted(p.name for p in (tmp_path / 'episodes').iterdir()) == ['0', '1', '2']
    assert not list((tmp_path / '.parallel_work').iterdir())
    assert json.loads((tmp_path / 'coverage.json').read_text())['committed'] == 3
    # A second invocation increases the total and does not repeat any seed.
    args.episodes = 4
    restored = parallel.ExpertRecorder(tmp_path, {}, frame_shape=(2, 2, 3))
    parallel.collect(args, restored, {}, worker=small_attempt)
    metadata = yaml.safe_load((tmp_path / 'metadata.yaml').read_text())
    assert [e['id'] for e in metadata['episodes']] == ['0', '1', '2', '3']
    assert len({e['source_trajectory_trial_id'] for e in metadata['episodes']}) == 4
    assert metadata['episodes'][-1]['source_trajectory_trial_id'] == 'sim_fixed_46'


def test_ready_handoff_recovers_commit_before_journal_crash(tmp_path):
    args = arguments(tmp_path, episodes=1)
    (tmp_path / 'attempts/000000').mkdir(parents=True)
    payload = small_attempt(args, 0, {})
    recorder = parallel.ExpertRecorder(tmp_path, {}, frame_shape=(2, 2, 3))
    parallel.commit_result(tmp_path, recorder, payload)
    (tmp_path / 'attempts.jsonl').unlink()  # Metadata persisted, journal did not.
    restored = parallel.ExpertRecorder(tmp_path, {}, frame_shape=(2, 2, 3))
    parallel.collect(args, restored, {}, worker=small_attempt)
    rows = (tmp_path / 'attempts.jsonl').read_text().splitlines()
    assert len(restored.entries) == len(rows) == 1
    assert json.loads(rows[0])['episode_id'] == 0


def test_ready_handoff_is_admitted_without_rerunning_worker(tmp_path):
    args = arguments(tmp_path, episodes=1)
    (tmp_path / 'attempts/000000').mkdir(parents=True)
    small_attempt(args, 0, {})
    recorder = parallel.ExpertRecorder(tmp_path, {}, frame_shape=(2, 2, 3))
    parallel.collect(args, recorder, {}, worker=small_attempt)
    assert len(recorder.entries) == 1
    assert len(list((tmp_path / 'attempts').iterdir())) == 1


def test_existing_collector_lock_refuses_parallel_writer(tmp_path, monkeypatch):
    with (tmp_path / '.collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(sys, 'argv', ['collector', '--output', str(tmp_path)])
        with pytest.raises(SystemExit) as exc:
            parallel.main()
        assert exc.value.code == 2
    assert not (tmp_path / 'source_manifest.json').exists()


def test_changed_source_manifest_refuses_resume(tmp_path, monkeypatch):
    (tmp_path / 'source_manifest.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['collector', '--output', str(tmp_path)])
    with pytest.raises(ValueError, match='source/config changed'):
        parallel.main()
    assert not (tmp_path / 'metadata.yaml').exists()


def test_sigterm_drains_inflight_attempts_and_stops_submission(tmp_path):
    import os
    import signal
    from concurrent.futures import ThreadPoolExecutor

    args = arguments(tmp_path, episodes=10, workers=2, max_attempts=10)
    recorder = parallel.ExpertRecorder(tmp_path, {}, frame_shape=(2, 2, 3))

    def stopping_attempt(args, attempt, fidelity):
        time.sleep(.1)  # Let the coordinator dispatch its initial two tasks.
        if attempt == 0:
            os.kill(os.getpid(), signal.SIGTERM)
        return small_attempt(args, attempt, fidelity)

    def threaded_executor(**kwargs):
        return ThreadPoolExecutor(max_workers=kwargs['max_workers'])

    parallel.collect(args, recorder, {}, worker=stopping_attempt, executor_factory=threaded_executor)
    rows = [json.loads(line) for line in (tmp_path / 'attempts.jsonl').read_text().splitlines()]
    assert [row['attempt'] for row in rows] == [0, 1]
    assert len(recorder.entries) == 1
    assert not list((tmp_path / '.parallel_work').iterdir())
