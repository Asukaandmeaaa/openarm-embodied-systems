from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from vision_grasp_baseline.scene_estimation import stabilize_scene,check_replay_fixture,angle_difference


FAILED=Path(__file__).resolve().parent/'fixtures/cube_yaw_jitter'


def captured(name):
    return json.loads((FAILED/name/'scene.json').read_text())


def test_actual_failed_capture_last_frame_rejected_but_consensus_agrees():
    original,fresh=captured('zero_scene'),captured('execution_scene')
    saved=deepcopy(fresh)
    assert abs(angle_difference(fresh['measured']['cube']['yaw'],original['measured']['cube']['yaw'],math.pi/2))>.08
    planned=stabilize_scene(original)
    admitted=check_replay_fixture(fresh,planned)
    assert abs(angle_difference(admitted['measured']['cube']['yaw'],planned['measured']['cube']['yaw'],math.pi/2))<.04
    assert admitted['pose_estimation']['objects']['cube']['inliers']==8
    assert admitted['frames']==saved['frames'] and fresh==saved
    assert admitted['measured']['table']==saved['measured']['table']
    assert stabilize_scene(admitted)==admitted


@pytest.mark.parametrize('change',['rotation','translation'])
def test_real_change_still_rejects_replay(change):
    planned=stabilize_scene(captured('zero_scene'));fresh=captured('zero_scene')
    for row in fresh['frames']:
        if change=='rotation':row['cube']['yaw']+=.2
        else:row['cube']['center'][0]+=.012
    with pytest.raises((ValueError,RuntimeError),match='changed'):
        check_replay_fixture(fresh,planned)


@pytest.mark.parametrize('count',[3,5])
def test_recent_movement_is_not_hidden_by_old_majority(count):
    fresh=captured('zero_scene')
    for row in fresh['frames'][-count:]:row['cube']['yaw']+=.3
    with pytest.raises(ValueError,match='Unstable cube'):stabilize_scene(fresh)


def test_square_symmetry_branch_cut_is_stable():
    scene=captured('zero_scene')
    for i,row in enumerate(scene['frames']):row['cube']['yaw']=math.pi/4+(-1 if i%2 else 1)*.01
    result=stabilize_scene(scene)
    assert abs(angle_difference(result['measured']['cube']['yaw'],math.pi/4,math.pi/2))<1e-6


def test_insufficient_frames_and_nan_are_rejected():
    scene=captured('zero_scene');scene['frames']=scene['frames'][:2]
    with pytest.raises(ValueError,match='At least 5'):stabilize_scene(scene)
    scene=captured('zero_scene');scene['frames'][-1]['cube']['yaw']=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):stabilize_scene(scene)


def test_stationary_fixture_uses_batch_consensus_despite_two_noisy_final_frames():
    scene=captured('zero_scene')
    for row in scene['frames']:row['cube']['yaw']=0.
    for row in scene['frames'][-2:]:row['cube']['yaw']=.12
    with pytest.raises(ValueError,match='Unstable cube'):stabilize_scene(scene)
    stable=stabilize_scene(scene,stationary=True)
    assert stable['measured']['cube']['yaw']==pytest.approx(0.)
    assert stable['pose_estimation']['stationary_fixture']
    assert stabilize_scene(stable)==stable
    for row in scene['frames'][::2]:row['cube']['yaw']=.3
    with pytest.raises(ValueError,match='Unstable cube'):stabilize_scene(scene,stationary=True)


def test_measured_alias_cannot_change_raw_frames_or_subsequent_consensus():
    path=Path(__file__).parent/'fixtures/consensus_alias/scene.json'
    scene=json.loads(path.read_text())
    scene['measured']=scene['frames'][-1]
    raw=deepcopy(scene['frames'])
    estimated=stabilize_scene(scene,stationary=True)
    assert estimated['frames']==raw and scene['frames']==raw
    assert estimated['measured'] is not estimated['frames'][-1]
    assert estimated['pose_estimation']['objects']['cube']['inliers']==7
    for _ in range(3):
        reread=json.loads(json.dumps(estimated))
        assert stabilize_scene(reread)==estimated
