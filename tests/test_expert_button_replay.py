"""Offline adapter/ownership tests. No ROS nodes, CAN or real joystick access."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import numpy as np
import pytest

from scripts.expert_replay import button_replay as replay


@pytest.mark.parametrize('mode',['delayed','missing','stop'])
def test_collision_discovery_is_bounded_and_never_sends_or_skips_validation(mode):
    from scripts.expert_replay.secure_player import wait_for_collision_service
    now=[0.];logs=[]
    def spin(): now[0]+=.25
    node=NS(validity=NS(service_is_ready=lambda:mode=='delayed' and now[0]>=2.),
            stop_reason=lambda:'OPERATOR_INTERRUPT' if mode=='stop' and now[0]>=.5 else '')
    log=lambda kind,**fields:logs.append(kind)
    if mode=='delayed':
        wait_for_collision_service(node,log,clock=lambda:now[0],spin=spin)
        assert now[0]==2. and logs[-1]=='collision_service_discovery_ready'
    else:
        with pytest.raises(RuntimeError,match='startup discovery deadline' if mode=='missing' else 'OPERATOR_INTERRUPT'):
            wait_for_collision_service(node,log,clock=lambda:now[0],spin=spin)
        assert now[0] <= 10.
        assert 'collision_service_discovery_ready' not in logs


def event(initial=False, index=1, value=1):
    return NS(kind='button', index=index, value=value, initial=initial, to_dict=lambda: {})


def idle():
    return {'status':{'state':'idle'}, 'synchronized':True, 'drain_ready':True,
            'pending':[], 'hold_pending':[], 'raw_active':{}}


def test_button_requires_actual_edge_and_idle_requires_drained_owner():
    assert replay.button_pressed([event()],1)
    for e in (event(initial=True), event(index=0), event(value=0)):
        assert not replay.button_pressed([e],1)
    assert replay.idle_status(idle())
    for key,value in [('synchronized',False),('drain_ready',False),('pending',['x']),
                      ('hold_pending',['x']),('raw_active',{'right':['x']}),('status',{'state':'fault'})]:
        d=idle();d[key]=value;assert not replay.idle_status(d)


def test_secure_player_reuses_samples_but_routes_arm_via_moveit(monkeypatch,tmp_path):
    from scripts.expert_replay import secure_player
    from scripts.control_ownership import h2
    from openarm_kinesthetic_recorder import recorded_motion_player as player
    payload=player.load_recorded_motion(replay.DEFAULT)
    captured=[]
    class Base:
        def __init__(self,name):
            assert name=='home_guarded_executor'
            self.all_positions=dict(zip(player.ARM_NAMES,payload['samples'][0]['positions_rad']))
            self.keyframe_settle_tolerance_rad=.03
        def create_publisher(self,*args,**kw): return 'real publisher'
        def create_client(self,*args,**kw): return 'real client'
        def create_subscription(self,*args,**kw): return 'real subscription'
        def execute_same_trajectory(self,t,d,timeout_margin_s):
            captured.append(t.joint_trajectory)
            return {'action_status':4,'moveit_error_code':1}
        def wait_for_arm_target(self,*args,**kw): pass
    monkeypatch.setattr(player,'RecordedMotionPlayer',Base)
    monkeypatch.setattr(h2,'validated_token',lambda **kw:'fake')
    monkeypatch.setattr(h2,'inject_action_capabilities',lambda token:nullcontext())
    def original_main():
        node=player.RecordedMotionPlayer()
        assert node.create_subscription(None,'/openarm_kinesthetic/gamepad_status') is None
        assert node.create_subscription(None,'/joint_states')=='real subscription'
        assert not node.create_client(None,'/real_episode_recorder/finish_failure').wait_for_service(timeout_sec=.25)
        node.execute_dense_arm_segment(payload,0,5,1.,0)
        expected=player.build_dense_arm_trajectory(payload['samples'],0,5,time_scale=1.)
        assert captured[0]==expected
        with pytest.raises(RuntimeError,match='START_CHANGED'):
            node.all_positions[player.ARM_NAMES[0]]+=.1
            node.execute_dense_arm_segment(payload,0,5,1.,0)
        return 0
    monkeypatch.setattr(player,'main',original_main)
    command=replay.player_command(replay.DEFAULT,tmp_path/'home.json',tmp_path/'audit.jsonl')
    monkeypatch.setattr(sys,'argv',['secure_player',*command[3:]])
    assert secure_player.main()==0
    assert player.RecordedMotionPlayer is Base


@pytest.mark.parametrize('mode', ['complete','batch','plan_fail','player_fail','pad_disconnect','rb_stop'])
def test_parent_never_starts_without_B_and_freezes_on_failure(monkeypatch,tmp_path,mode):
    from scripts.act_trial import manual_experiment as manual
    from openarm_kinesthetic.teaching import gamepad
    trial=tmp_path/'act';trial.mkdir()
    monkeypatch.setattr(manual,'session',lambda:(tmp_path,trial))
    monkeypatch.setattr(manual,'require_home_handoff',lambda _:None)
    monkeypatch.setattr(manual,'observation_processes',lambda _:[])
    monkeypatch.setattr(replay.time,'sleep',lambda _:None)
    monkeypatch.setattr(replay.signal,'signal',lambda *a:None)
    calls=[]
    class Pad:
        connected=True;axis_codes=[];axes=np.zeros(8);buttons=np.zeros(16,dtype=int)
        count=0
        def poll(self):
            self.count+=1
            if mode=='batch' and self.count==6:self.buttons[1]=0
            if self.count==2 or (mode=='batch' and self.count==7):
                self.buttons[1]=1;calls.append('B');return [event()]
            if self.count>=4:
                if mode=='pad_disconnect':self.connected=False
                if mode=='rb_stop':self.buttons[5]=1
            return []
        def button_index_for_code(self,_):return None
        def close(self):pass
    pad=Pad();monkeypatch.setattr(gamepad.LinuxJoystick,'open_configured',lambda:pad)
    class Client:
        token=None
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def close(self):pass
        def request(self,op,**kw):calls.append(op);return idle()
        def acquire(self,owner):
            assert 'B' in calls
            self.token='fake';calls.append('acquire')
        def renew(self):calls.append('renew')
        def release(self):self.token=None;calls.append('release')
    monkeypatch.setattr(replay,'Client',Client)
    class Child:
        def __init__(self,command,**kwargs):
            self.plan='--plan-only' in command;self.count=0;self.returncode=None
            calls.append('plan' if self.plan else 'motion')
            if not self.plan: assert kwargs['env']['OPENARM_AUTHORITY_TOKEN']=='fake'
        def poll(self):
            self.count+=1
            if self.plan or self.count>=3:
                self.returncode=2 if ((self.plan and mode=='plan_fail') or
                                     (not self.plan and mode=='player_fail')) else 0
            return self.returncode
        def terminate(self):self.returncode=-15;calls.append('terminate')
        def wait(self,**kw):return self.returncode
    monkeypatch.setattr(replay.subprocess,'Popen',Child)
    if mode=='batch':
        import shutil
        directory=tmp_path/'trajectories';directory.mkdir()
        for path in sorted(replay.DEFAULT.parent.glob('*.json'))[:2]:shutil.copy2(path,directory/path.name)
        selection=['--trajectory-directory',str(directory)]
    else: selection=['--trajectory',str(replay.DEFAULT)]
    monkeypatch.setattr(sys,'argv',['button_replay',*selection,'--allow-real-execution'])
    code=replay.main()
    if mode in ('complete','batch'):
        assert code==0 and 'release' in calls and 'freeze' not in calls
        assert calls.count('motion')==calls.count('B')==(2 if mode=='batch' else 1)
    elif mode=='plan_fail':
        assert code==2 and 'acquire' not in calls and 'motion' not in calls
    else:
        assert code==2 and 'freeze' in calls and 'release' not in calls
    report=json.loads(next(tmp_path.glob('expert_replay_*/result.json')).read_text())
    assert report['source']=='expert_recording_not_ACT'
