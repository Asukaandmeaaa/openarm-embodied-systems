"""Exercise actual gamepad timer with memory publishers and simulated input."""
from types import SimpleNamespace
import numpy as np
from openarm_kinesthetic_recorder import gamepad_node as g
from openarm_kinesthetic.real import OperatorPacket,OperatorWatchdog


def test_bridge_dispatch_yields_to_worker_threads(monkeypatch):
    from openarm_kinesthetic_recorder import hardware_bridge as b
    events=[]
    executor=SimpleNamespace(spin_once=lambda **kw:events.append(('dispatch',kw)))
    monkeypatch.setattr(b,'time',SimpleNamespace(sleep=lambda duration:events.append(('yield',duration))))
    b.spin_once_with_worker_yield(executor)
    assert events==[('dispatch',{'timeout_sec':.1}),('yield',0)]


def test_held_lt_lb_with_no_new_events_keeps_publishing(monkeypatch):
    clock=[10.0];messages=[]
    monkeypatch.setattr(g,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    axes=np.zeros(8,dtype=np.int16);axes[4:6]=-32767
    buttons=np.zeros(15,dtype=np.uint8)
    noop=SimpleNamespace(publish=lambda m:None)
    node=SimpleNamespace(
        joystick=SimpleNamespace(connected=True,poll=lambda:[],axes=axes,buttons=buttons,
            path='/fake',name='simulated',axis_code_for_index=lambda i:(i,str(i))),
        mapper=g.GamepadMapper(g.load_yaml('gamepad_control.yaml')),
        try_connect=lambda now:None,
        get_logger=lambda:SimpleNamespace(info=lambda m:None),
        motion_recording_mode='continuous_marked',keyframe_recording_active=True,
        continuous_marked_drive_gripper=False,continuous_marked_enable_lb_keypoints=False,
        diagnostic_only=False,session_id='test',sequence=0,open_sequence=0,close_sequence=0,
        record_sequence=0,hand_drag_toggle_opens_next=True,last_summary=None,
        last_status_publish=0,device_request='fake',last_connect_error='',
        publisher=SimpleNamespace(publish=lambda m:messages.append(OperatorPacket.from_json(m.data))),
        status_publisher=noop,raw_event_publisher=noop,record_toggle_publisher=noop,
        twist_publisher=noop,cartesian_command_frame='world')
    guard=OperatorWatchdog(.25)
    g.RealGamepadNode.on_timer(node)
    guard.ingest(messages[-1],clock[0])
    axes[5]=32767;buttons[6]=1
    for i in range(1200):  # 60 simulated seconds, no new joystick events.
        clock[0]=10.05+i*.05
        g.RealGamepadNode.on_timer(node)
        assert messages[-1].arm_motion_source=='physical_kinesthetic'
        assert guard.ingest(messages[-1],clock[0]).request_passive
        assert guard.decision(clock[0]+.01).request_passive
    assert len(messages)==1201
    assert [m.sequence for m in messages]==list(range(1201))
    assert guard.decision(clock[0]+.251).stop_reason=='operator_watchdog_timeout'
    clock[0]+=.3;axes[5]=-32767;buttons[6]=0
    g.RealGamepadNode.on_timer(node)
    assert not guard.ingest(messages[-1],clock[0]).request_passive
    assert messages[-1].arm_motion_source=='idle'
