"""Offline process-generation and CLI routing tests. No live ROS/CAN calls."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.act_trial import control_runtime as runtime
from scripts.act_trial import manual_experiment as manual


def process(proc, pid, role, ticks, domain='117'):
    path = proc / str(pid)
    path.mkdir(exist_ok=True)
    args = (['/opt/ros/jazzy/lib/controller_manager/ros2_control_node'] if role == 'backend'
            else ['/usr/bin/python3', '-m', 'scripts.control_ownership.gateway'])
    args += ['--ros-args', '--enclave', '/openarm/' + role]
    (path/'cmdline').write_bytes(b'\0'.join(s.encode() for s in args)+b'\0')
    (path/'environ').write_bytes(('ROS_DOMAIN_ID='+domain+'\0ROS_SECURITY_ENABLE=true\0'
                                'ROS_SECURITY_STRATEGY=Enforce\0ROS_SECURITY_KEYSTORE=/test/keys\0').encode())
    # Fields 3 through 22; process name may itself contain parentheses/spaces.
    (path/'stat').write_text(f'{pid} (a (process)) S ' + '0 '*18 + str(ticks)+' 0')
    return path


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    proc=tmp_path/'proc'; proc.mkdir()
    boot=proc/'sys/kernel/random/boot_id'; boot.parent.mkdir(parents=True); boot.write_text('boot-A')
    process(proc, 100, 'backend', 1234)
    process(proc, 200, 'gateway', 2345)
    monkeypatch.setenv('OPENARM_KEYSTORE', '/test/keys')
    return proc


def test_identity_uses_birth_time_and_ignores_native_replay(fake_proc):
    process(fake_proc, 300, 'backend', 3333, domain='97')
    identity=runtime.live_identity(fake_proc)
    assert identity=={'boot_id':'boot-A','backend':{'pid':100,'start_ticks':1234},
                     'gateway':{'pid':200,'start_ticks':2345}}


@pytest.mark.parametrize('change', ['duplicate','insecure','different_keys','missing_gateway'])
def test_ambiguous_or_invalid_control_identity_refused(fake_proc, change):
    if change=='duplicate': process(fake_proc, 101, 'backend', 1235)
    if change=='insecure':
        p=fake_proc/'100/environ';p.write_bytes(p.read_bytes().replace(b'Enforce',b'Permissive'))
    if change=='different_keys':
        p=fake_proc/'100/environ';p.write_bytes(p.read_bytes().replace(b'/test/keys',b'/other/keys'))
    if change=='missing_gateway': shutil.rmtree(fake_proc/'200')
    with pytest.raises(ValueError):runtime.live_identity(fake_proc)


def test_logs_do_not_select_control_state_and_restarts_invalidate_handoff(fake_proc,tmp_path,monkeypatch):
    root=tmp_path/'runtime';root.mkdir()
    reader=runtime.live_identity
    monkeypatch.setattr(runtime,'live_identity',lambda:reader(fake_proc))
    monkeypatch.setattr(runtime,'runtime_root',lambda:root)
    first=runtime.control_session()
    (first/'home_hold_test.json').write_text('{"passed":true}')
    monkeypatch.setenv('OWNERSHIP_RUN',str(tmp_path/'old_logs'))
    assert runtime.control_session()==first
    monkeypatch.setenv('OWNERSHIP_RUN',str(tmp_path/'new_logs'))
    assert runtime.control_session()==first
    process(fake_proc,100,'backend',9999)  # PID reuse still invalidates old HOME.
    second=runtime.control_session()
    assert second!=first and not (second/'home_hold_test.json').exists()
    process(fake_proc,200,'gateway',9998)
    assert runtime.control_session()!=second
    assert (first/'home_hold_test.json').read_text()=='{"passed":true}'


def test_simultaneous_h2_is_refused_and_lock_releases(tmp_path):
    with runtime.control_operation(tmp_path):
        with pytest.raises(ValueError,match='already running'):
            with runtime.control_operation(tmp_path):pytest.fail('second H2 entered')
    with runtime.control_operation(tmp_path):pass


def test_h2_does_not_read_act_session_or_its_old_reports(tmp_path,monkeypatch):
    old=tmp_path/'old_act';old.mkdir();(old/'h2_execution.json').write_text('{}')
    control=tmp_path/'live_control';control.mkdir()
    monkeypatch.setenv('OWNERSHIP_RUN',str(old))
    monkeypatch.setattr(manual,'check_environment',lambda:None)
    monkeypatch.setattr(manual,'session',lambda:pytest.fail('H2 loaded ACT logs'))
    monkeypatch.setattr(manual,'control_session',lambda:control)
    seen=[];monkeypatch.setattr(manual,'h2',lambda directory:seen.append(directory))
    monkeypatch.setattr(manual.sys,'argv',['actctl','h2'])
    manual.main()
    assert seen==[control]
    assert (old/'h2_execution.json').read_text()=='{}'


def test_same_control_h2_attempt_still_blocks_duplicate_motion(tmp_path,monkeypatch):
    (tmp_path/'h2_execution.json').write_text('{"success":false}')
    monkeypatch.setattr(manual,'Client',lambda:pytest.fail('duplicate H2 reached gateway'))
    with pytest.raises(ValueError,match='current ROS/gateway'):
        manual.h2(tmp_path)


def test_h1_needs_no_act_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(manual,'check_environment',lambda:None)
    monkeypatch.setattr(manual,'runtime_root',lambda:tmp_path)
    monkeypatch.setattr(manual,'session',lambda:pytest.fail('H1 loaded ACT logs'))
    monkeypatch.setenv('OPENARM_KEYSTORE','/test/keys')
    monkeypatch.setattr(manual.sys,'argv',['actctl','h1'])
    seen=[];monkeypatch.setattr(manual.os,'execv',lambda _,args:seen.append(args))
    manual.main()
    assert seen[0][2]=='scripts.act_trial.h1'
    assert seen[0][-1] == '--physical'
    assert not (tmp_path/'act').exists()


def test_stopped_gateway_status_reports_no_automatic_restart(monkeypatch):
    def absent(): raise FileNotFoundError()
    monkeypatch.setattr(manual,'Client',absent)
    with pytest.raises(SystemExit,match='Gateway is not running'):
        manual.show_gateway_status()


@pytest.fixture
def cli_workspace(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1]/'scripts/actctl',scripts/'actctl')
    env=scripts/'home_replay/workstation_env.sh';env.parent.mkdir();env.write_text('export ROS_DOMAIN_ID=97\n')
    overlay=tmp_path/'.control_ownership_overlay/install/setup.bash'
    overlay.parent.mkdir(parents=True);overlay.write_text('true\n')
    security=scripts/'control_ownership/security.py';security.parent.mkdir()
    security.write_text("print('export ROS_DOMAIN_ID=117 ROS_SECURITY_ENABLE=true ROS_SECURITY_STRATEGY=Enforce')\n")
    python=tmp_path/'official/dora-openarm-evaluation/.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nprintf "domain=%s trial=%s loaded=%s args=%s\\n" '
                      '"$ROS_DOMAIN_ID" "${ACT_TRIAL_DIR:-none}" "${ACT_SESSION_SOURCED:-no}" "$*"\n')
    python.chmod(0o755)
    return tmp_path


@pytest.mark.parametrize('stage',['h1','h2','gateway','prepare'])
def test_cli_control_stages_ignore_stale_shell_and_log_pointer(cli_workspace,stage):
    root=cli_workspace;results=root/'results';results.mkdir()
    (results/'act_manual_current.env').write_text('echo SHOULD_NOT_LOAD; exit 88\n')
    env=dict(os.environ,OWNERSHIP_RUN='/old',ACT_TRIAL_DIR='/old/act',SOCKET='/old.sock')
    out=subprocess.run(['bash','scripts/actctl',stage],cwd=root,env=env,capture_output=True,text=True,check=True)
    assert 'domain=117 trial=none loaded=no' in out.stdout
    assert 'SHOULD_NOT_LOAD' not in out.stdout


def test_cli_automatically_loads_act_session_and_appends_policy_log(cli_workspace):
    root=cli_workspace;trial=root/'results/session/act';trial.mkdir(parents=True)
    (root/'results/act_manual_current.env').write_text(
        f'export ACT_TRIAL_DIR={trial}\nexport OWNERSHIP_RUN={trial.parent}\nexport ACT_SESSION_SOURCED=yes\n')
    (trial/'policy.log').write_text('previous output\n')
    out=subprocess.run(['bash','scripts/actctl','policy'],cwd=root,capture_output=True,text=True,check=True)
    assert f'domain=117 trial={trial} loaded=yes' in out.stdout
    assert 'manual_experiment prepare-policy' in out.stdout
    policy_output = ''.join(line+'\n' for line in out.stdout.splitlines()
                            if 'prepare-policy' not in line)
    assert (trial/'policy.log').read_text()=='previous output\n'+policy_output
