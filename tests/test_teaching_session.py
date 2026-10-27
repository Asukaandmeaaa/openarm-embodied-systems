"""Offline supervisor checks: fake ROS, or dummy processes only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

MODULE=Path(__file__).resolve().parents[1]/'scripts/teaching_session.py'
spec=importlib.util.spec_from_file_location('teaching_session',MODULE)
t=importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def test_dry_run_has_only_requested_teaching_nodes():
    result=subprocess.run([sys.executable,str(MODULE),'--dry-run'],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert list(plan)==['bridge','recorder','gamepad']
    assert 'auto_return_home_on_second_x:=true' in plan['recorder']
    assert list(dict(t.commands('gamepad',Path('/tmp/unused'))))==['gamepad']


def test_existing_control_nodes_rejected_h1_allowed():
    rows={999991:{'args':['python3','/x/right_kinesthetic_bridge']},
          999992:{'args':['python3','-m','scripts.control_ownership.gateway']},
          999993:{'args':['/x/ros2_control_node']}}
    assert set(t.conflicts(rows))=={999991,999992}


def test_shutdown_keeps_bridge_until_rigid_and_controllers_confirmed(monkeypatch):
    events=[]
    class Child:
        def __init__(self,pid):self.pid=pid;self.code=None
        def poll(self):return self.code
    children={n:Child(i) for i,n in enumerate(('bridge','recorder','gamepad'),100)}
    by_pid={p.pid:p for p in children.values()}
    def interrupt(pid):events.append(('stop',pid));by_pid[pid].code=0
    monkeypatch.setattr(t,'interrupt_group',interrupt)
    monkeypatch.setattr(t,'process_table',lambda:{})
    class Observer:
        ticks=0
        checks=0
        def spin(self):self.ticks+=1
        def rigid(self):return self.ticks>=2
        def active_controllers(self):
            self.checks+=1;events.append(('controllers',self.checks))
            return self.checks>=2
    obs=Observer()
    t.shutdown(children,obs)
    assert events[:2]==[('stop',101),('stop',102)]
    assert events[-2:]==[('controllers',2),('stop',100)]


def test_rigid_requires_fresh_measurement_and_both_passive_confirmations():
    obs=t.Observer.__new__(t.Observer)
    obs.bridge_at=t.time.monotonic()
    obs.passive=False
    obs.bridge={'joint_state_fresh':True,'passive_enabled':False,
                'hardware_passive':False,'control_mode':'idle'}
    assert obs.rigid()
    obs.passive=True
    assert not obs.rigid()
    obs.passive=False;obs.bridge_at-=1
    assert not obs.rigid()


def test_adopted_home_is_cancelled_even_after_recorder_exits(tmp_path):
    # This script owns only a dummy Python parent and sleeping dummy HOME.
    script=r'''
import importlib.util,sys,subprocess,time,os,signal
from pathlib import Path
s=importlib.util.spec_from_file_location('supervisor',sys.argv[1])
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
t.adopt_orphans()
ready=Path(sys.argv[2])
home_code="import signal,time,pathlib,sys; signal.signal(signal.SIGINT,lambda *_: sys.exit(0)); pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid())); time.sleep(30)"
parent_code="import subprocess,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],start_new_session=True)"
p=subprocess.Popen([sys.executable,'-c',parent_code,home_code,str(ready)],start_new_session=True)
pid=None
try:
    p.wait(timeout=5)
    until=time.monotonic()+5
    while not ready.exists() and time.monotonic()<until:time.sleep(.01)
    pid=int(ready.read_text())
    assert pid in t.descendant_groups(os.getpid(),t.process_table())
    class Observer:
        def spin(self):time.sleep(.01)
    t.shutdown({'recorder':p},Observer())
    assert not t.group_alive(pid)
finally:
    if pid and t.group_alive(pid):os.killpg(pid,signal.SIGKILL)
'''
    subprocess.run([sys.executable,'-c',script,str(MODULE),str(tmp_path/'dummy-ready')],
                   check=True,timeout=10)
