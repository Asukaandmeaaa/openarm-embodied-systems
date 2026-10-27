"""Collection regressions: causal labels, complete episodes, failed data quarantine."""
from copy import deepcopy
import json

import numpy as np
import pytest

from vision_grasp_baseline.collection_io import audit_episode, check_fixed_scene, timing_summary
from vision_grasp_baseline.collection_service import Recorder
from vision_grasp_baseline.gripper_settings import TRANSIT_POSITION,COLLECTION_POLICY


def closed_state():
    # Actual finger feedback differs from command: labels must remain commands.
    return [0.]*7+[TRANSIT_POSITION-.01]+[0.]*8


def scene():
    return {'measured': {'cube': {'center': [.3, -.2, .1]},
                        'bin': {'center_xy': [.3, .1], 'yaw': 0., 'outer_size_xy': [.1, .1]}}}


def test_fixture_rejects_bin_move_but_accepts_small_cube_reset():
    first = scene()
    second = deepcopy(first)
    second['measured']['cube']['center'][0] += .015
    assert check_fixed_scene(second, first)['cube_delta_m'] == pytest.approx(.015)
    second['measured']['bin']['center_xy'][0] += .03
    with pytest.raises(RuntimeError, match='Fixed scene changed'):
        check_fixed_scene(second, first)


def ready(recorder, clock):
    now = clock[0]
    for key in recorder.camera_times:
        recorder.camera_times[key].clear()
        recorder.camera_times[key].extend([now-.1, now-2/30, now-1/30, now])
    recorder.state('joint_observation', closed_state(), now)
    recorder.state('left_arm', [.01]*7, now)
    recorder.state('right_gripper', [TRANSIT_POSITION], now)
    recorder.state('right_arm', [.02]*7, now)


def recorded(tmp_path, monkeypatch, gap=False, raw=False):
    clock = [100.]
    monkeypatch.setattr('vision_grasp_baseline.collection_service.time.monotonic', lambda: clock[0])
    recorder = Recorder()
    ready(recorder, clock)
    path = tmp_path/'episode_0000'
    recorder.begin(path)
    def state(key, positions, stamp):
        message = None
        if raw:
            from vision_grasp_baseline.collection_io import ORDER
            from vision_grasp_baseline.training_contract import PART_NAMES
            if key == 'joint_observation':
                message = dict(name=ORDER, position=positions)
            else:
                # Deliberately distinct feedback/reference: only output is the label.
                message = dict(joint_names=PART_NAMES[key], output=dict(positions=positions,
                    velocities=[0.]*len(positions)), feedback=dict(positions=[-.2]*len(positions)),
                    reference=dict(positions=[.4]*len(positions)))
            message = dict(timestamp_s=stamp, receive_monotonic_s=stamp, controller=key, message=message)
        recorder.state(key, positions, stamp, message)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for step in range(301):
        clock[0] = 100.+step/100
        state('joint_observation', closed_state(), clock[0])
        if step % 2 == 0:
            state('left_arm', [.01]*7, clock[0])
            state('right_gripper', [TRANSIT_POSITION], clock[0])
            state('right_arm', [.02]*7, clock[0])
        if step % 10 in (0, 3, 7):
            for key in recorder.camera_times:
                if not (gap and key == 'wrist_right' and 120 < step < 160):
                    recorder.camera(key, image, dict(timestamp_s=clock[0], frame_number=step))
    result = recorder.finish(True)
    return path, result


def test_full_rate_episode_and_action_is_output_not_feedback(tmp_path, monkeypatch):
    path, result = recorded(tmp_path, monkeypatch)
    assert result['quality']['passed']
    actions = [json.loads(line) for line in (path/'demonstration_action.jsonl').read_text().splitlines()]
    assert len(actions) == 151
    assert actions[0]['positions'][:8] == [.02]*7+[TRANSIT_POSITION]
    assert actions[0]['positions'][8:] == [.01]*7+[0.]
    metadata = json.loads((path/'metadata.json').read_text())
    assert metadata['review_status'] == 'pending'
    assert metadata['gripper_policy'] == COLLECTION_POLICY
    assert metadata['empty_transit_feedback_range_rad']==[-.055,0.]
    assert not metadata['operator_marked_success']
    # Verify the existing 16D converter can join this new raw format.
    from scripts.convert_real_episodes_act import prepare
    _, _, _, report = prepare(path, 30)
    assert report['frame_count'] == 91


def test_camera_gap_fails_even_when_arm_cycle_succeeds(tmp_path, monkeypatch):
    path, result = recorded(tmp_path, monkeypatch, gap=True)
    assert not result['quality']['passed']
    assert not audit_episode(path)['passed']


def test_begin_requires_zero_and_never_overwrites(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('vision_grasp_baseline.collection_service.time.monotonic', lambda: clock[0])
    recorder = Recorder()
    ready(recorder, clock)
    recorder.latest['joint_observation'] = (100., [.1]+[0.]*15)
    with pytest.raises(RuntimeError, match='mechanical zero'):
        recorder.begin(tmp_path/'episode')
    ready(recorder, clock)
    (tmp_path/'episode').mkdir()
    with pytest.raises(FileExistsError):
        recorder.begin(tmp_path/'episode')


def test_stale_gripper_never_substitutes_measured_joint_position(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('vision_grasp_baseline.collection_service.time.monotonic', lambda: clock[0])
    recorder = Recorder()
    ready(recorder, clock)
    recorder.begin(tmp_path/'episode')
    clock[0] += .2
    recorder.state('right_arm', [.02]*7, clock[0])
    result = recorder.finish(False)
    assert not result['quality']['passed']
    metadata = json.loads((tmp_path/'episode/metadata.json').read_text())
    assert metadata['terminal_label'] == 'failed_or_interrupted'
    assert 'controller source stream' in metadata['recording_error']


def test_late_left_callbacks_join_by_source_time_without_using_future_targets(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('vision_grasp_baseline.collection_service.time.monotonic', lambda: clock[0])
    recorder = Recorder(); ready(recorder, clock); recorder.begin(tmp_path/'episode')
    # Right callbacks can arrive before buffered left callbacks. Every source is retained.
    for step in range(41):
        stamp = 100.+step/100
        recorder.state('right_arm', [.02]*7, stamp)
        recorder.state('right_gripper', [-.35], stamp)
        recorder.state('joint_observation', [0.]*16, stamp)
    for step in range(41):
        recorder.state('left_arm', [step/1000]*7, 100.+step/100)
    clock[0] = 100.4
    recorder.finish(False)
    rows = [json.loads(line) for line in (tmp_path/'episode/demonstration_action.jsonl').read_text().splitlines()]
    assert len(rows) == 41
    assert [row['positions'][8] for row in rows] == [i/1000 for i in range(41)]
    assert all(max(row['component_timestamp_s'].values()) <= row['timestamp_s'] for row in rows)
    assert recorder.error is None


def test_slow_image_write_does_not_block_ros_state_updates(tmp_path, monkeypatch):
    import threading
    from vision_grasp_baseline import collection_service
    clock = [100.]
    monkeypatch.setattr(collection_service.time, 'monotonic', lambda: clock[0])
    recorder = Recorder(); ready(recorder, clock); recorder.begin(tmp_path/'episode')
    entered, release, received = threading.Event(), threading.Event(), threading.Event()
    def slow_write(*args):
        entered.set(); release.wait(2); return True
    monkeypatch.setattr(collection_service.cv2, 'imwrite', slow_write)
    recorder.camera('ceiling', np.zeros((2, 2, 3), dtype=np.uint8), dict(timestamp_s=100., frame_number=1))
    assert entered.wait(1)
    def receive():
        recorder.state('left_arm', [.01]*7, 100.01); received.set()
    thread = threading.Thread(target=receive); thread.start()
    try:
        assert received.wait(.2), 'ROS callback blocked behind image encoding/disk write'
    finally:
        release.set(); thread.join(1); recorder.finish(False)


def test_timing_rejects_duplicates_and_low_rate():
    assert not timing_summary([1., 1., 2.], 25., .15)['passed']
    assert not timing_summary([1., 1.1, 1.2], 25., .15)['passed']


@pytest.mark.parametrize('extra', [[], ['--waypoints']])
def test_cycle_refuses_legacy_staged_recording_before_io(tmp_path, monkeypatch, extra):
    from vision_grasp_baseline import cycle
    calls = []
    monkeypatch.setattr(cycle, 'BASE', tmp_path)
    monkeypatch.setattr(cycle, 'WORKSPACE', tmp_path)
    monkeypatch.setattr(cycle, 'wait_for_gateway_handoff', lambda **kwargs: None)
    monkeypatch.setenv('VISION_CAPTURE_SERVICE', 'mock')
    monkeypatch.setenv('VISION_CAPTURE_TOKEN', 'mock')
    monkeypatch.setattr('sys.argv', ['cycle', *extra, '--output', str(tmp_path/'run'),
                                    '--record-episode', str(tmp_path/'episode')])
    with pytest.raises(SystemExit) as exc:
        cycle.main()
    assert exc.value.code==2
    assert not (tmp_path/'run').exists()
    assert calls==[]


@pytest.mark.parametrize('source', ['feedback','command'])
def test_begin_rejects_open_finger_before_creating_episode(tmp_path,monkeypatch,source):
    clock=[100.]
    monkeypatch.setattr('vision_grasp_baseline.collection_service.time.monotonic',lambda:clock[0])
    recorder=Recorder();ready(recorder,clock)
    if source=='feedback':recorder.state('joint_observation',[0.]*7+[-.45]+[0.]*8,100.)
    else:recorder.state('right_gripper',[-.45],100.)
    with pytest.raises(ValueError,match='closed|closed-transit'):
        recorder.begin(tmp_path/'episode')
    assert not (tmp_path/'episode').exists()


def test_final_open_finger_cannot_label_success(tmp_path,monkeypatch):
    clock=[100.]
    monkeypatch.setattr('vision_grasp_baseline.collection_service.time.monotonic',lambda:clock[0])
    recorder=Recorder();ready(recorder,clock);recorder.begin(tmp_path/'episode')
    recorder.state('joint_observation',[0.]*7+[-.45]+[0.]*8,100.)
    recorder.finish(True)
    metadata=json.loads((tmp_path/'episode/metadata.json').read_text())
    assert not metadata['automatic_task_contract_passed']
    assert metadata['terminal_label']=='failed_or_interrupted'
    assert 'closed' in metadata['recording_error']
