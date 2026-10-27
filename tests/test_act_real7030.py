import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('real7030', Path(__file__).parents[1]/'scripts/train_act_real7030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_source_balanced_mass():
    rows = [dict(domain='latest_real', group='a', episode_index=0),
            dict(domain='latest_real', group='a', episode_index=1),
            dict(domain='old_real', group='b', episode_index=2)]
    assert np.allclose(m.masses(rows, 4), [.35, .35, .3, 0])


def test_stats_exclude_simulation_and_weight_domains(tmp_path):
    path = tmp_path/'train/data'; path.mkdir(parents=True)
    table = pa.table({'episode_index': [0, 1, 2],
                      'action': [[1.]*16, [3.]*16, [999.]*16],
                      'observation.state': [[1.]*16, [3.]*16, [999.]*16]})
    pq.write_table(table, path/'sample.parquet')
    rows = [dict(episode_index=0, frame_count=1), dict(episode_index=1, frame_count=1)]
    stats = m.real_stats(rows, [.7, .3, 0], tmp_path)
    np.testing.assert_allclose(stats['action']['mean'], 1.6)
    np.testing.assert_allclose(stats['action']['std'], np.sqrt(.84))
    assert stats['action']['count'][0] == 2


def test_actual_selection_excludes_simulation():
    manifests, selected = m.selection()
    assert len(selected['train']) == 59
    assert all(r['domain'] != 'simulation' for rows in selected.values() for r in rows)
    w = m.masses(selected['train'], len(manifests['train']['episodes']))
    for domain, expected in m.MASS.items():
        assert np.isclose(sum(w[r['episode_index']] for r in selected['train'] if r['domain'] == domain), expected)


def test_50k_config_is_fresh_and_separate():
    import json
    config = json.loads((Path(__file__).parents[1]/'configs/training/act_real7030_50k_20261020.json').read_text())
    assert config['steps'] == 50000
    assert config['save_freq'] == 5000
    assert 'pretrained_path' not in config['policy']
    assert config['output_dir'].endswith('act_real7030_50k_20261020')
