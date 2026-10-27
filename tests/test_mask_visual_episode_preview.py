"""Full-timeline wrist masking tests; no ROS, camera, CAN or physical replay."""
import json

from PIL import Image
import pytest

from scripts.mask_visual_episode_preview import mask_preview
from scripts.trim_visual_episode_preview import digest, read_rows


@pytest.fixture
def episode(tmp_path):
    source = tmp_path/'episodes'/'episode_0000'
    source.mkdir(parents=True)
    (source/'metadata.json').write_text(json.dumps(dict(
        start_timestamp_s=0., end_timestamp_s=20., image_size_wh=[640, 480],
        review_status='approved', terminal_label='returned_zero', operator_marked_success=True,
        automatic_task_contract_passed=True, complete_trajectory_path='/unused/original_plan')))
    events = [dict(timestamp_s=t, phase='timeline_event', detail={'name': name})
              for t, name in [(7., 'close'), (12., 'open')]]
    grip = [dict(timestamp_s=float(t), positions=[-.085 if 8 <= t < 13 else -.45])
            for t in range(21)]
    state = [dict(timestamp_s=float(t), positions=[float(t)]*16) for t in range(21)]
    camera = []
    for t in range(21):
        for key in ('ceiling', 'wrist_right'):
            relative = f'cameras/{key}/{t:04d}.jpg'
            file = source/relative
            file.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (640, 480), (50, 100, 180)).save(file)
            camera.append(dict(timestamp_s=float(t), key=key, path=relative, frame_number=t))
    for name, rows in [('events', events), ('target_right_gripper', grip),
                       ('joint_observation', state), ('demonstration_action', state), ('camera', camera)]:
        (source/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return source, tmp_path/'mask_previews'/'episode_0000'


def test_full_timestamps_streams_and_ceiling_are_preserved(episode):
    source, output = episode
    before = {p.relative_to(source): digest(p) for p in source.rglob('*') if p.is_file()}
    report = mask_preview(source, output)
    assert report['full_duration_s'] == 20.
    assert report['wrist_valid_relative_start_s'] == 4.
    assert report['wrist_valid_relative_end_s'] == 17.
    assert report['camera_counts'] == {
        'ceiling': {'valid': 21, 'masked': 0}, 'wrist_right': {'valid': 14, 'masked': 7}}
    for file in source.glob('*.jsonl'):
        assert digest(file) == digest(output/file.name)
    assert {p: digest(source/p) for p in before} == before
    validity = read_rows(output/'camera_validity.jsonl')
    assert len(validity) == 42
    for row in validity:
        expected = row['key'] == 'ceiling' or 4 <= row['timestamp_s'] <= 17
        assert row['image_valid'] is expected and row['image_mask'] == int(expected)
        if expected:
            assert digest(output/row['path']) == digest(source/row['path'])
        else:
            image = Image.open(output/row['path']).convert('RGB')
            assert image.size == (640, 480) and image.getextrema() == ((0, 0),)*3
    meta = json.loads((output/'metadata.json').read_text())
    assert meta['start_timestamp_s'] == 0. and meta['end_timestamp_s'] == 20.
    assert meta['diagnostic_only'] and not meta['operator_marked_success']
    assert 'complete_trajectory_path' not in meta
    assert not meta['camera_mask']['model_mask_input_implemented']
    assert not report['existing_model_compatible']


def test_never_overwrites_existing_preview(episode):
    source, output = episode
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        mask_preview(source, output)


def test_cannot_add_preview_to_original_episode_directory(episode):
    source, _ = episode
    with pytest.raises(ValueError, match='outside'):
        mask_preview(source, source.parent/'preview')


@pytest.mark.parametrize('before,after', [(-1., 5.), (3., float('nan')), (3., 100.)])
def test_invalid_margins_fail_without_creating_output(episode, before, after):
    source, output = episode
    with pytest.raises(ValueError):
        mask_preview(source, output, before, after)
    assert not output.exists()


def test_invalid_camera_path_is_rejected_before_copy(episode):
    source, output = episode
    rows = read_rows(source/'camera.jsonl')
    rows[0]['path'] = '../outside.jpg'
    (source/'camera.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='image path'):
        mask_preview(source, output)
    assert not output.exists()
