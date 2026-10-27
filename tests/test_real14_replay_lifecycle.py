"""Offline regression checks for attended capture lifecycle; no robot access."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from std_msgs.msg import String
from openarm_kinesthetic_recorder import episode_capture_button as button
from openarm_kinesthetic_recorder import real_episode_recorder as recorder


def test_capture_lock_rejects_duplicate_then_releases(tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path))
    first=button.acquire_capture_lock()
    try:
        with pytest.raises(RuntimeError,match='already running'):button.acquire_capture_lock()
    finally:first.close()
    button.acquire_capture_lock().close()


def test_cancel_waits_past_timeout_without_abandoning_player(monkeypatch):
    signals=[];logs=[]
    monkeypatch.setattr(button.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    class Player:
        pid=99881
        attempts=0
        done=False
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):
            self.attempts+=1
            if self.attempts==1:raise subprocess.TimeoutExpired('dummy',timeout)
            self.done=True
    player=Player()
    button.stop_owned_player(player,SimpleNamespace(warn=logs.append,error=logs.append))
    assert player.done and player.attempts==2 and len(signals)==1
    assert any('WAITING_FOR_PLAYER_EXIT' in x for x in logs)


@pytest.mark.parametrize('nodes',[
    ['episode_capture_button','episode_capture_button','real_episode_recorder'],
    ['episode_capture_button','real_episode_recorder','recorded_motion_player'],
])
def test_capture_rejects_duplicate_button_or_existing_player(nodes):
    now=button.time.monotonic();logs=[]
    node=SimpleNamespace(button_index=1,stage='READY',
        gamepad_status={'connected':True,'stop_latched':False,'control_mode':'idle'},
        gamepad_status_at=now,latest_status_at=now,
        latest_status={'active':False,'sources_ready':True,'output_root':'/tmp/collection'},
        collection_root=Path('/tmp/collection'),get_node_names=lambda:nodes,
        start_client=SimpleNamespace(service_is_ready=lambda:True),
        get_logger=lambda:SimpleNamespace(warn=logs.append,error=logs.append))
    button.EpisodeCaptureButton.on_gamepad_events(node,String(data=json.dumps({
        'events':[{'kind':'button','index':1,'value':1,'initial':False}]})))
    assert node.stage=='READY' and 'B_CAPTURE_REJECTED' in logs[-1]


def test_recorder_shutdown_marks_uncommitted_episode_failed(monkeypatch):
    events=[]
    node=SimpleNamespace(abort_active=lambda reason:events.append(('abort',reason)),
                         destroy_node=lambda:events.append(('destroy',None)))
    monkeypatch.setattr(recorder,'RealEpisodeRecorderNode',lambda:node)
    def interrupt(_):raise KeyboardInterrupt
    monkeypatch.setattr(recorder.rclpy,'init',lambda:None)
    monkeypatch.setattr(recorder.rclpy,'spin',interrupt)
    monkeypatch.setattr(recorder.rclpy,'ok',lambda:False)
    assert recorder.main()==0
    assert events==[('abort','recorder_shutdown_before_episode_commit'),('destroy',None)]


def test_player_exception_cancels_before_failing_recorder(monkeypatch):
    from openarm_kinesthetic_recorder import recorded_motion_player as player
    events=[]
    def fail():raise RuntimeError('synthetic failure')
    node=SimpleNamespace(keyframe_settle_tolerance_rad=None,
        wait_for_full_joint_state=fail,cancel_active_goals=lambda:events.append('cancel'),
        fail_active_recorder=lambda:events.append('fail_recording'),
        get_logger=lambda:SimpleNamespace(error=lambda msg:None),destroy_node=lambda:events.append('destroy'))
    monkeypatch.setattr(player,'parse_arguments',lambda:SimpleNamespace(home_report=None,return_home=True,keyframe_settle_tolerance_rad=.04))
    monkeypatch.setattr(player,'verify_home_provenance',lambda p:{})
    monkeypatch.setattr(player,'RecordedMotionPlayer',lambda:node)
    monkeypatch.setattr(player.rclpy,'init',lambda:None)
    monkeypatch.setattr(player.rclpy,'ok',lambda:False)
    monkeypatch.setattr(player.signal,'signal',lambda *a:None)
    assert player.main()==2
    assert events==['cancel','fail_recording','destroy']
