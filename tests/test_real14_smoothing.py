"""Offline trajectory contract and numerical tests; no ROS nodes or device IO."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

WS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('smoothing',WS/'scripts/smooth_real14_trajectories.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=WS/'openarm_kinesthetic_data/collections_real14_20261016_smooth_v1'


def test_polynomial_extrema_find_interior_peak():
    x=np.linspace(0,1,12)
    q=np.column_stack([(k+1)*(x-x*x) for k in range(7)])
    spline=make_interp_spline(x,q,k=5)
    lo,hi=m.extrema(spline,0)
    assert np.allclose(lo,0,atol=1e-10)
    assert np.allclose(hi,np.arange(1,8)/4,atol=1e-10)
    assert np.allclose(m.maxima(spline,2),2*np.arange(1,8),atol=1e-8)


def test_geometry_distance_detects_corner_cutting():
    path=np.array([[0.,0.],[1.,0.],[1.,1.]])
    shortcut=np.array([[0.,0.],[1.,1.]])
    assert m.path_distance(path,path)==0
    assert m.path_distance(path,shortcut)==pytest.approx(.5)


def test_all_published_candidates_preserve_events_and_source():
    files=(ROOT/'notes/replay14_files.txt').read_text().splitlines()
    assert len(files)==12
    for name in files:
        p=json.loads(Path(name).read_text()); info=p['processing']
        source=Path(info['source_path']); raw=json.loads(source.read_text())
        assert m.sha(source)==info['source_sha256']
        m.validate_collection_trajectory(p)
        old=[0]+[i['sample_index'] for i in raw['markers']]+[len(raw['samples'])-1]
        new=[0]+[i['sample_index'] for i in p['markers']]+[len(p['samples'])-1]
        assert [p['samples'][i]['positions_rad'] for i in new]==[raw['samples'][i]['positions_rad'] for i in old]
        assert [(e['type'],e['target_position_rad']) for e in p['markers']]==[(e['type'],e['target_position_rad']) for e in raw['markers']]
        for seg in info['segments']:
            assert max(seg['max_acceleration_rad_s2'])<=.5
            assert max(seg['max_jerk_rad_s3'])<=5
            assert seg['max_reference_difference_rad']<=.01
            assert seg['endpoint_velocity_max']<1e-8
            assert seg['endpoint_acceleration_max']<1e-7
        assert info['physical_replay_status']=='pending'
        assert p['duration_s']<raw['duration_s']


def test_original_out_of_bounds_trajectories_are_not_published():
    report=json.loads((ROOT/'notes/processing_report.json').read_text())
    assert {Path(p['source_path']).stem[-3:] for p in report['blocked']}=={'006','012'}
    assert all('joint6' in p['reason'] for p in report['blocked'])
    assert all(not (ROOT/'trajectories'/Path(p['source_path']).name).exists() for p in report['blocked'])


def test_reprocessing_is_deterministic_and_first_home_tail_faster():
    p=ROOT/'trajectories/20261016_113434_005.json'
    saved=json.loads(p.read_text())
    generated=m.process(Path(saved['processing']['source_path']),m.yaml.safe_load(m.LIMITS.read_text())['joint_limits'])
    assert generated==saved
    tail=saved['processing']['segments'][-1]
    assert tail['processed_duration_s']<tail['original_duration_s']
