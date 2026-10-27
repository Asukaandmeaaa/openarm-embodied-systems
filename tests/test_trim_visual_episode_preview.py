"""Synthetic, offline-only crop tests; no robot or camera APIs."""
import json
from pathlib import Path

import pytest

from scripts.trim_visual_episode_preview import digest, read_rows, trim_preview


@pytest.fixture
def episode(tmp_path):
    source = tmp_path/'episodes'/'episode_0000'
    source.mkdir(parents=True)
    (source/'metadata.json').write_text(json.dumps(dict(start_timestamp_s=0., end_timestamp_s=20.,
        review_status='approved', terminal_label='returned_zero', operator_marked_success=True)))
    events = [dict(timestamp_s=t, phase='timeline_event', detail={'name': name})
              for t, name in [(7., 'close'), (12., 'open')]]
    grip = [dict(timestamp_s=float(t), positions=[-.085 if 8 <= t < 13 else -.45])
            for t in range(21)]
    state = [dict(timestamp_s=float(t), positions=[float(t)]*16, order=['test']*16)
             for t in range(21)]
    camera = []
    for t in range(21):
        for key in ('ceiling', 'wrist_right'):
            rel = f'cameras/{key}/{t:04d}.jpg'
            (source/rel).parent.mkdir(parents=True, exist_ok=True)
            (source/rel).write_bytes(b'synthetic immutable image')
            camera.append(dict(timestamp_s=float(t), key=key, path=rel, frame_number=t))
    for name, rows in [('events', events), ('target_right_gripper', grip),
                       ('joint_observation', state), ('demonstration_action', state), ('camera', camera)]:
        (source/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return source, tmp_path/'trim_previews'/'episode_0000'


def test_crop_keeps_both_cameras_and_streams_without_source_mutation(episode):
    source, output = episode
    hashes = {p.relative_to(source): digest(p) for p in source.rglob('*') if p.is_file()}
    result = trim_preview(source, output)
    assert (result['start_timestamp_s'], result['end_timestamp_s']) == (4., 15.)
    assert result['duration_s'] == 11.
    for name in ('joint_observation', 'demonstration_action', 'target_right_gripper'):
        rows = read_rows(output/(name+'.jsonl'))
        assert [r['timestamp_s'] for r in rows] == list(range(4, 16))
    assert result['camera_counts'] == {'ceiling': 12, 'wrist_right': 12}
    assert {p: digest(source/p) for p in hashes} == hashes
    for row in read_rows(output/'camera.jsonl'):
        assert digest(output/row['path']) == digest(source/row['path'])
        assert (output/row['path']).stat().st_ino != (source/row['path']).stat().st_ino
    meta = json.loads((output/'metadata.json').read_text())
    assert meta['diagnostic_only'] and meta['review_status'] == 'preview_only'
    assert not meta['operator_marked_success'] and not meta['automatic_task_contract_passed']
    assert meta['terminal_label'] != 'returned_zero'
    assert meta['initial_state'] == [4.]*16


def test_existing_output_is_never_overwritten(episode):
    source, output = episode
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        trim_preview(source, output)


def test_preview_cannot_be_added_to_frozen_source_episodes(episode):
    source, _ = episode
    with pytest.raises(ValueError, match='outside'):
        trim_preview(source, source.parent/'episode_preview')


def test_missing_or_duplicate_event_fails_before_output_creation(episode):
    source, output = episode
    with (source/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(timestamp_s=13., phase='timeline_event', detail={'name':'open'}))+'\n')
    with pytest.raises(ValueError, match='exactly one open'):
        trim_preview(source, output)
    assert not output.exists()


def test_margin_cannot_silently_extend_outside_source(episode):
    source, output = episode
    with pytest.raises(ValueError, match='outside the recording'):
        trim_preview(source, output, margin=10.)
    assert not output.exists()


def test_camera_path_traversal_is_rejected_before_copy(episode):
    source, output = episode
    rows = read_rows(source/'camera.jsonl')
    rows[10]['path'] = '../outside.jpg'
    (source/'camera.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='Invalid image'):
        trim_preview(source, output)
    assert not output.exists()
