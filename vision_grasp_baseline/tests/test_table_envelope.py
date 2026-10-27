from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from vision_grasp_baseline.table_envelope import admit_scene,corners,table_box,planning_scene,with_top_clearance,with_edge_clearance


@pytest.fixture
def scenes():
    base=Path(__file__).resolve().parent/'fixtures/table_jitter'
    return [json.loads((base/name/'scene.json').read_text()) for name in ('zero_scene','execution_scene')]


def test_reported_table_jitter_is_covered_without_replacing_planned_volume(scenes):
    before,fresh=scenes
    planned=planning_scene(before);original=deepcopy(fresh)
    admitted=admit_scene(planned,fresh)
    assert fresh==original
    assert admitted['table_collision_envelope']==planned['table_collision_envelope']
    assert admitted['measured']==fresh['measured']
    envelope=planned['table_collision_envelope']
    for scene in scenes:
        for frame in [scene['measured'],*scene['frames']]:
            projected=(corners(*table_box(frame['table']))-envelope['center'])@np.asarray(envelope['rotation'])
            assert np.all(abs(projected)<=np.asarray(envelope['size'])/2+1e-8)


def test_new_table_extent_and_plane_change_cannot_hide_in_cached_envelope(scenes):
    before,fresh=scenes;planned=planning_scene(before)
    for field,value in [('center_xy',[.46,.02]),('top_z',.30),('size_xy',[.6,.8]),('yaw',.25)]:
        changed=deepcopy(fresh);changed['measured']['table'][field]=value
        with pytest.raises(ValueError,match='outside planned envelope'):admit_scene(planned,changed)
    changed=deepcopy(fresh);changed['frames'][0]['table']['top_z']+=.03
    with pytest.raises(ValueError,match='outside planned envelope'):admit_scene(planned,changed)


def test_plan_clearance_expands_upward_and_keeps_original_box(scenes):
    scene=planning_scene(scenes[0]);buffered=with_top_clearance(scene,.015)
    before,after=[x['table_collision_envelope'] for x in (scene,buffered)]
    normal=np.asarray(before['rotation'])[:,2]
    np.testing.assert_allclose(np.asarray(before['center'])-normal*before['size'][2]/2,
                               np.asarray(after['center'])-normal*after['size'][2]/2)
    assert after['size'][2]-before['size'][2]==pytest.approx(.015)
    assert before['size'][:2]==after['size'][:2]


def test_legacy_plan_is_not_silently_granted_a_larger_validated_envelope(scenes):
    with pytest.raises(ValueError,match='Legacy trajectory'):admit_scene(*scenes)


def test_edge_reserve_contains_original_without_lowering_table_or_mutating_scene(scenes):
    scene=planning_scene(scenes[0]);original=deepcopy(scene)
    expanded=with_edge_clearance(scene,.03)['table_collision_envelope']
    before=scene['table_collision_envelope']
    assert scene==original
    assert expanded['center']==before['center']
    assert expanded['rotation']==before['rotation']
    assert expanded['size'][2]==before['size'][2]
    np.testing.assert_allclose(np.asarray(expanded['size'])[:2]-before['size'][:2],[.06,.06])
    for margin in (-.01,float('nan'),float('inf')):
        with pytest.raises(ValueError):with_edge_clearance(scene,margin)
