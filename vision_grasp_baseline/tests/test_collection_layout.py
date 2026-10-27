"""Storage tests use temporary directories and fake services only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_grasp_baseline.collection_layout import create_layout,next_episode_number,archive_categories


@pytest.mark.parametrize('tag',['nearby','spread','rotation'])
def test_default_storage_uses_existing_categories(tmp_path,tag):
    layout=create_layout(tmp_path,tag,stamp='test')
    assert layout.classified
    assert layout.episodes==tmp_path/'collections'/tag/'episodes'
    assert layout.runs==tmp_path/'collections'/tag/'runs'
    assert layout.session==tmp_path/'collections'/tag/'sessions'/('session_'+tag+'_test')
    assert next_episode_number(layout)==0
    with pytest.raises(FileExistsError):create_layout(tmp_path,tag,stamp='test')


def test_numbering_respects_both_episodes_and_remaining_run_evidence(tmp_path):
    layout=create_layout(tmp_path,stamp='test')
    layout.episodes.mkdir();layout.runs.mkdir()
    (layout.episodes/'episode_0002').mkdir()
    # The operator deleted old recordings but retained their execution logs.
    (layout.runs/'cycle_0009').mkdir()
    (layout.episodes/'episode_bad').mkdir()
    assert next_episode_number(layout)==10
    (layout.episodes/'episode_0012').touch()
    assert next_episode_number(layout)==13


def test_diagnostics_stay_outside_training_categories(tmp_path):
    layout=create_layout(tmp_path,camera_only=True,stamp='test')
    assert layout.session==tmp_path/'runs/camera_checks/session_test'
    assert not (tmp_path/'collections').exists()


def test_explicit_output_remains_new_isolated_session(tmp_path):
    layout=create_layout(tmp_path,output=tmp_path/'custom',stamp='test')
    assert not layout.classified and layout.episodes==tmp_path/'custom/episodes'
    with pytest.raises(FileExistsError):create_layout(tmp_path,output=tmp_path/'custom')


def fake_collection(tmp_path,monkeypatch,*,fail=False):
    from vision_grasp_baseline import collect
    monkeypatch.setattr(collect,'BASE',tmp_path)
    monkeypatch.setattr(collect,'WORKSPACE',tmp_path)
    monkeypatch.setattr('vision_grasp_baseline.cycle.wait_for_gateway_handoff',lambda **kw:{})
    monkeypatch.setattr(collect.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(collect,'rpc',lambda *a,**kw:{'ok':True})
    class Service:
        stopped=False
        def __init__(self,cmd,**kw):
            ready=Path(cmd[cmd.index('--ready')+1])
            ready.write_text(json.dumps(dict(url='fake',token='fake')))
        def poll(self):return 0 if self.stopped else None
        def terminate(self):self.stopped=True
        def wait(self,**kw):return 0
    monkeypatch.setattr(collect.subprocess,'Popen',Service)
    calls=[]
    def cycle(cmd,**kw):
        import subprocess
        assert cmd[1]=='cycle'
        episode=Path(cmd[cmd.index('--record-episode')+1]);episode.mkdir(parents=True)
        run=Path(cmd[cmd.index('--output')+1]);run.mkdir(parents=True)
        (episode/'metadata.json').write_text(json.dumps(dict(review_status='pending')))
        calls.append((episode,run))
        if fail:raise subprocess.CalledProcessError(1,cmd)
        (run/'cycle_result.json').write_text(json.dumps(dict(success=True)))
    monkeypatch.setattr(collect.subprocess,'run',cycle)
    replies=iter(['','y']*4)
    monkeypatch.setattr('builtins.input',lambda prompt:next(replies))
    monkeypatch.setattr('sys.argv',['collect','--episodes','1'])
    return collect,calls


def test_repeated_collect_appends_and_never_relabels_previous_episode(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    old=tmp_path/'collections/nearby/episodes/episode_0000';old.mkdir(parents=True)
    previous='{"collection_tag":"keep_this","review_status":"approved"}\n'
    (old/'metadata.json').write_text(previous)
    collect.main();collect.main()
    assert [e.name for e,r in calls]==['episode_001','episode_002']
    assert [r.name for e,r in calls]==['cycle_001','cycle_002']
    assert (old/'metadata.json').read_text()==previous
    sessions=list((tmp_path/'collections/nearby/sessions').glob('session_*/session.json'))
    assert len(sessions)==2
    ids=[]
    for episode,run in calls:
        metadata=json.loads((episode/'metadata.json').read_text())
        assert metadata['review_status']=='approved' and metadata['collection_tag']=='nearby'
        assert Path(metadata['collection_session']).is_file()
        assert metadata['cycle_result']==str(run/'cycle_result.json')
        ids.append(metadata['source_trajectory_trial_id'])
    assert len(set(ids))==2
    assert set(p.name for p in (tmp_path/'collections').iterdir())=={'nearby'}


def test_failed_attempt_is_labeled_without_touching_old_data(tmp_path,monkeypatch):
    import subprocess
    collect,calls=fake_collection(tmp_path,monkeypatch,fail=True)
    old=tmp_path/'collections/nearby/episodes/episode_0000';old.mkdir(parents=True)
    (old/'metadata.json').write_text('{}')
    with pytest.raises(subprocess.CalledProcessError):collect.main()
    assert (old/'metadata.json').read_text()=='{}'
    metadata=json.loads((calls[0][0]/'metadata.json').read_text())
    assert metadata['collection_tag']=='nearby' and metadata['review_status']=='pending'
    session=json.loads(Path(metadata['collection_session']).read_text())
    assert not session['success'] and len(session['attempted_episodes'])==1
    attempt=session['attempted_episodes'][0]
    assert attempt['status']=='failed'
    marker=Path(attempt['failure_marker'])
    assert marker.parent.name=='episode_001_FAILED' and marker.is_file()
    assert (calls[0][0]/'FAILED.json').exists()
    assert (calls[0][1]/'FAILED.json').exists()
    assert metadata['diagnostic_only'] is True and metadata['operator_marked_success'] is False
    assert next_episode_number(create_layout(tmp_path,stamp='after_failure'))==2


def test_new_collection_starts_at_three_digit_zero(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    collect.main()
    assert calls[0][0].name=='episode_000'
    assert calls[0][1].name=='cycle_000'


def test_failure_before_recording_still_has_visible_folder_and_reserves_number(tmp_path,monkeypatch):
    import subprocess
    collect,calls=fake_collection(tmp_path,monkeypatch)
    monkeypatch.setattr(collect.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(subprocess.CalledProcessError(1,a[0])))
    with pytest.raises(subprocess.CalledProcessError):collect.main()
    failed=tmp_path/'collections/nearby/failed/episode_000_FAILED/FAILED.json'
    assert failed.is_file() and not json.loads(failed.read_text())['recording_started']
    assert calls==[]
    assert next_episode_number(create_layout(tmp_path,stamp='after'))==1


def test_keyboard_interrupt_is_marked_and_cannot_be_approved(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    original=collect.subprocess.run
    def interrupted(*a,**k):
        original(*a,**k)
        raise KeyboardInterrupt()
    monkeypatch.setattr(collect.subprocess,'run',interrupted)
    with pytest.raises(KeyboardInterrupt):collect.main()
    episode,run=calls[0]
    assert (episode/'INTERRUPTED.json').is_file()
    marker=tmp_path/'collections/nearby/failed/episode_000_INTERRUPTED/INTERRUPTED.json'
    assert json.loads(marker.read_text())['status']=='interrupted'
    assert json.loads((episode/'metadata.json').read_text())['diagnostic_only'] is True


def test_unapproved_replay_gets_review_marker_not_technical_failure(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    replies=iter(['','n','q'])
    monkeypatch.setattr('builtins.input',lambda p:next(replies))
    collect.main()
    episode,run=calls[0]
    assert (episode/'REVIEW_REQUIRED.json').is_file()
    assert not (episode/'FAILED.json').exists()
    assert not (tmp_path/'collections/nearby/failed').exists()


def test_rejected_attempts_do_not_consume_success_quota_or_start_without_enter(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    monkeypatch.setattr('sys.argv',['collect','--episodes','2'])
    replies=iter(['','n','','y','','n','','y'])
    prompts=[]
    def respond(prompt):
        prompts.append(prompt)
        return next(replies)
    monkeypatch.setattr('builtins.input',respond)
    collect.main()
    assert [episode.name for episode,run in calls]==['episode_000','episode_001','episode_002','episode_003']
    assert len([p for p in prompts if '回车开始本轮' in p])==4
    metas=[json.loads((episode/'metadata.json').read_text()) for episode,run in calls]
    assert [m['operator_marked_success'] for m in metas]==[False,True,False,True]
    assert [m['review_status'] for m in metas]==['pending','approved','pending','approved']
    session=json.loads(Path(metas[0]['collection_session']).read_text())
    assert session['success'] and len(session['completed_episodes'])==2
    assert len(session['attempted_episodes'])==4
    assert session['completed_episodes']==[str(calls[i][0]) for i in (1,3)]


def test_q_at_review_keeps_pending_and_exits_without_next_attempt(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    replies=iter(['','q'])
    monkeypatch.setattr('builtins.input',lambda prompt:next(replies))
    collect.main()
    assert len(calls)==1
    meta=json.loads((calls[0][0]/'metadata.json').read_text())
    assert meta['review_status']=='pending' and meta['operator_marked_success'] is False
    session=json.loads(Path(meta['collection_session']).read_text())
    assert not session['success'] and session['exit_reason']=='operator_quit_at_review'
    assert session['completed_episodes']==[]
    assert (calls[0][0]/'REVIEW_REQUIRED.json').exists()


def test_invalid_or_blank_review_reprompts_and_does_not_dispatch_again(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    replies=iter(['','', 'wrong', 'Y'])
    monkeypatch.setattr('builtins.input',lambda prompt:next(replies))
    collect.main()
    assert len(calls)==1
    assert json.loads((calls[0][0]/'metadata.json').read_text())['operator_marked_success'] is True


def test_invalid_start_response_cannot_launch_motion(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    replies=iter(['n','unexpected','q'])
    monkeypatch.setattr('builtins.input',lambda prompt:next(replies))
    collect.main()
    assert calls==[]


def test_rejection_then_technical_failure_still_stops_and_marks_failure(tmp_path,monkeypatch):
    import subprocess
    collect,calls=fake_collection(tmp_path,monkeypatch)
    original=collect.subprocess.run
    def run(*args,**kwargs):
        original(*args,**kwargs)
        if len(calls)==2:
            raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr(collect.subprocess,'run',run)
    replies=iter(['','n',''])
    monkeypatch.setattr('builtins.input',lambda prompt:next(replies))
    with pytest.raises(subprocess.CalledProcessError):collect.main()
    assert len(calls)==2
    assert (calls[0][0]/'REVIEW_REQUIRED.json').exists()
    assert (calls[1][0]/'FAILED.json').exists()
    assert (tmp_path/'STOP').exists()


def test_preflight_failure_is_marked_without_creating_an_episode(tmp_path,monkeypatch):
    collect,calls=fake_collection(tmp_path,monkeypatch)
    def fail(**kw):raise RuntimeError('gateway unavailable')
    monkeypatch.setattr('vision_grasp_baseline.cycle.wait_for_gateway_handoff',fail)
    with pytest.raises(RuntimeError,match='gateway unavailable'):collect.main()
    markers=list((tmp_path/'collections/nearby/failed').glob('session_*_FAILED/FAILED.json'))
    assert len(markers)==1 and not (tmp_path/'collections/nearby/episodes').exists()
    assert calls==[]


def test_reset_previews_then_archives_without_deleting_and_restarts_all_categories(tmp_path):
    for tag in ('nearby','spread','rotation'):
        category=tmp_path/'collections'/tag
        episode=category/'episodes/episode_0015'
        episode.mkdir(parents=True)
        (episode/'metadata.json').write_text('{"original":true}')
        (category/'runs/cycle_0017').mkdir(parents=True)
    preview=archive_categories(tmp_path,stamp='test')
    assert not Path(preview['archive']).exists()
    assert (tmp_path/'collections/nearby/episodes/episode_0015/metadata.json').is_file()
    report=archive_categories(tmp_path,stamp='test',execute=True)
    assert report['status']=='complete' and report['deleted_files']==0
    for tag in ('nearby','spread','rotation'):
        archived=Path(report['archive'])/tag
        assert (archived/'episodes/episode_0015/metadata.json').read_text()=='{"original":true}'
        assert (archived/'runs/cycle_0017').is_dir()
        assert next_episode_number(create_layout(tmp_path,tag,stamp='fresh'))==0
    assert json.loads((Path(report['archive'])/'archive_manifest.json').read_text())['status']=='complete'
    with pytest.raises(FileExistsError):archive_categories(tmp_path,stamp='test',execute=True)


def test_reset_refuses_active_collection_or_cycle(tmp_path):
    import fcntl
    for name in ('.collection.lock','.cycle.lock'):
        with (tmp_path/name).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with pytest.raises(RuntimeError,match='is running'):
                archive_categories(tmp_path,execute=True,stamp='test')
    assert not (tmp_path/'collection_archives').exists()


def test_reset_rejects_symlink_and_unknown_category(tmp_path):
    (tmp_path/'collections').mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (tmp_path/'collections/nearby').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='category path'):
        archive_categories(tmp_path,('nearby',),execute=True)
    with pytest.raises(ValueError,match='known collection'):
        archive_categories(tmp_path,('../outside',),execute=True)
    assert outside.is_dir() and not (tmp_path/'collection_archives').exists()
