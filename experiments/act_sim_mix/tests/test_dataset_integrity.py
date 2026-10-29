"""Cross-domain splitting and conversion regressions with miniature local data."""
import json
from argparse import Namespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from experiments.act_sim_mix import prepare_dataset as prep
from experiments.act_sim_mix.prepare_eval_scenes import build_scene_file, verify_scene_file

KINDS = ('closed_approach', 'lateral_offset', 'bad_height', 'recoverable_drop', 'pre_release_hold')


@pytest.mark.parametrize('overlap', [None, 'path', 'trial', 'hash'])
def test_real_audit_uses_full_provenance_instead_of_local_episode_name(tmp_path, monkeypatch, overlap):
    import io
    import pyarrow as pa
    image = io.BytesIO()
    Image.new('RGB', (640, 480)).save(image, format='JPEG')
    table = pa.table({key: [{'bytes': image.getvalue()}] for key in prep.CAMS})
    monkeypatch.setattr(prep.pq, 'read_table', lambda *a, **kw: table)
    monkeypatch.setattr(prep, 'REAL', tmp_path)
    entries = []
    for collection in ('nearby', 'spread'):
        entries.append({'source_episode': 'episode_004', 'frame_count': 1,
                        'source_episode_path': str(tmp_path / collection / 'episode_004'),
                        'source_trajectory_trial_id': collection + '_trial',
                        'source_sha256': {'demonstration_action.jsonl': collection + '_hash'}})
    if overlap == 'path':
        entries[1]['source_episode_path'] = entries[0]['source_episode_path']
    elif overlap == 'trial':
        entries[1]['source_trajectory_trial_id'] = entries[0]['source_trajectory_trial_id']
    elif overlap == 'hash':
        entries[1]['source_sha256'] = entries[0]['source_sha256'].copy()
    for split, entry in zip(('train', 'validation'), entries):
        folder = tmp_path / split / 'meta'
        folder.mkdir(parents=True)
        (folder / 'visual_expert_manifest.json').write_text(json.dumps({
            'episodes': [entry], 'joint_order': [], 'image_size_wh': [640, 480],
            'gripper_policy': 'closed_transit_v2'}))
    result = prep.real_audit()
    assert result['train_validation_disjoint'] is (overlap is None)
    assert bool(result['overlapping_source_keys']) is (overlap is not None)


def test_real_source_identity_requires_provenance():
    with pytest.raises(ValueError, match='provenance'):
        prep.real_source_keys({'source_episode': 'episode_004'})


def audit_fixture():
    rows = []
    for i in range(100):
        rows.append({'record_id': f'normal/{i}', 'episode_id': i, 'domain': 'normal',
                     'group_id': f'n{i}', 'scenario_keys': [f'scene{i}']})
    for family in range(10):
        for k, kind in enumerate(KINDS):
            rows.append({'record_id': f'recovery/{family * 5 + k}', 'episode_id': family * 5 + k,
                         'domain': 'recovery', 'group_id': f'r{family}', 'family_id': f'r{family}',
                         'recovery_type': kind, 'scenario_keys': [f'scene{family}' if k == 0 else f'recovery_scene{family}_{k}']})
    return {'complete': True, 'rows': rows}


def test_joint_split_keeps_shared_layout_in_one_split():
    audit = audit_fixture()
    split = prep.make_split(audit)
    assigned = {}
    for domain in ('normal', 'recovery'):
        for name, entries in split['episodes'][domain].items():
            for entry in entries:
                assigned[entry[0]] = name
    assert len(assigned) == 150
    for family in range(10):
        assert assigned[f'normal/{family}'] == assigned[f'recovery/{family * 5}']
        assert len({assigned[f'recovery/{family * 5 + k}'] for k in range(5)}) == 1
    assert split == prep.make_split(audit)
    assert split['scenario_keys_disjoint']


def test_impossible_split_is_rejected_without_breaking_families():
    audit = audit_fixture()
    for row in audit['rows']:
        row['scenario_keys'] = ['all_same_layout']
    with pytest.raises(ValueError, match='impossible'):
        prep.make_split(audit)


def test_seed_and_layout_identity_ignore_domain_names():
    run = {'scene_sha256': 'scene', 'jitter': {}}
    a = {'seed': 1, 'domain': 'normal', 'randomization': {'rng_seed': 1, 'xy': [1, 2]}}
    b = {'seed': 2, 'domain': 'recovery', 'randomization': {'rng_seed': 2, 'xy': [1, 2]}}
    assert set(prep.scenario_keys(a, run)) & set(prep.scenario_keys(b, run))


def test_combined_conversion_does_not_overwrite_same_episode_numbers(tmp_path, monkeypatch):
    audit = audit_fixture()
    for row in audit['rows']:
        directory = tmp_path / row['record_id']
        directory.mkdir(parents=True)
        np.savez(directory / 'numeric.npz', action=np.zeros((2, 16)))
        row.update(collection_root=str(tmp_path), path=row['record_id'], valid_from_frame=0)
    real = tmp_path / 'real'
    (real / 'train/meta').mkdir(parents=True)
    (real / 'train/meta/stats.json').write_text(json.dumps({'action': {'min': [-1.] * 16, 'max': [1.] * 16}}))
    monkeypatch.setattr(prep, 'REAL', real)
    monkeypatch.setattr(prep, 'combined_audit', lambda _: audit)
    monkeypatch.setattr(prep, 'real_audit', lambda: {
        'train_validation_disjoint': True, 'normalization': 'frozen', 'validation_limit': 'development',
        'splits': {'train': {'episodes': 30, 'frames': 40772}, 'validation': {'episodes': 7, 'frames': 9532}}})
    seen = {}
    def conversion(collection, rows, output, *, split_name):
        seen[split_name] = rows
        return {'episodes': len(rows), 'frames': 2 * len(rows)}
    monkeypatch.setattr(prep, 'convert_split', conversion)
    assert prep.cmd_convert(Namespace(collection=[tmp_path], output=tmp_path / 'converted', split_seed=42)) == 0
    assert len({r['record_id'] for rows in seen.values() for r in rows}) == 150
    assert sum(r['domain'] == 'normal' for r in seen['train']) == 80
    assert sum(r['domain'] == 'recovery' for r in seen['train']) == 40


def test_convert_retains_jpeg_and_only_expert_suffix(tmp_path):
    raw = tmp_path / 'raw'
    rows = []
    for number, domain in enumerate(('normal', 'recovery')):
        directory = raw / domain
        directory.mkdir(parents=True)
        start = 0 if domain == 'normal' else 2
        a = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
        np.savez(directory / 'numeric.npz', state=a + 100, action=a, valid=np.arange(5) >= start)
        for role in prep.CAMERAS:
            folder = directory / 'cameras' / role
            folder.mkdir(parents=True)
            for frame in range(5):
                Image.new('RGB', (640, 480), (frame, 10, 20)).save(folder / f'{frame:06d}.jpg')
        rows.append({'collection_root': str(raw), 'path': domain, 'valid_from_frame': start,
                     'domain': domain, 'record_id': domain})
    prep.convert_split(raw, rows, tmp_path / 'converted', split_name='train')
    table = pq.read_table(tmp_path / 'converted/data/chunk-000/file-001.parquet')
    assert len(table) == 3
    assert table['action'][0].as_py() == list(np.arange(32, 48, dtype=float))
    assert table['observation.images.ceiling'][0].as_py()['bytes'] == (raw / 'recovery/cameras/ceiling/000002.jpg').read_bytes()
    assert table['episode_index'].to_pylist() == [1, 1, 1]


def test_scene_file_without_optional_collection_and_hash_verification(tmp_path):
    raw = tmp_path / 'raw'
    entries = []
    for i in range(15):
        directory = raw / 'episodes' / str(i)
        directory.mkdir(parents=True)
        (directory / 'metadata.json').write_text(json.dumps({'seed': i, 'jitter': {}}))
        np.savez(directory / 'numeric.npz', qpos=np.zeros((2, 25)), qvel=np.zeros((2, 23)))
        entries.append({'collection_root': str(raw), 'path': f'episodes/{i}',
                        'domain': 'normal' if i < 10 else 'recovery', 'group_id': str(i),
                        'converted_episode_index': i, 'scenario_keys': [str(i)],
                        'valid_from_frame': 0, 'recovery_type': KINDS[i % 5]})
    meta = tmp_path / 'converted/validation/meta'
    meta.mkdir(parents=True)
    (meta / 'visual_expert_manifest.json').write_text(json.dumps({'episodes': entries}))
    result = build_scene_file(tmp_path / 'converted')
    assert result['count'] == 15
    assert {s['split'] for s in result['scenes']} == {'validation'}
    assert all(s['velocity_source'] == 'recorded_qvel' for s in result['scenes'] if s['kind'] == 'recovery')
    verify_scene_file(result)
    (raw / 'episodes/0/metadata.json').write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        verify_scene_file(result)
