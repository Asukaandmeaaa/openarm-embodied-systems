"""Supervise native teaching nodes only; H1/H2 remain operator-managed."""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
KIN=Path('/home/lushilin/openarm_kinesthetic_ws')
CONFIG=KIN/'ros2_ws/src/openarm_kinesthetic_recorder/config/real_kinesthetic.yaml'
EXECUTABLES=KIN/'ros2_ws/install-h1h2-capture/openarm_kinesthetic_recorder/lib/openarm_kinesthetic_recorder'
DEFAULT_OUTPUT=ROOT/'openarm_kinesthetic_data/collections_real14_20261016/trajectories'
CONFLICTS={'right_kinesthetic_bridge','right_motion_recorder','real_gamepad',
           'recorded_motion_player','episode_capture_button','home_guarded_executor',
           'scripts.control_ownership.gateway','scripts.control_ownership.secure_launch',
           'scripts.act_trial.ros_bridge','button_play_recorded_motion.py',
           'scripts.expert_replay.button_replay','scripts.expert_replay.secure_player'}


def commands(mode,output):
    def cmd(name,*extra):return [str(EXECUTABLES/name),'--ros-args','--params-file',str(CONFIG),*extra]
    pad=('gamepad',cmd('real_gamepad','-p','device:=auto','-p','motion_recording_mode:=continuous_marked'))
    if mode=='gamepad':return [pad]
    return [('bridge',cmd('right_kinesthetic_bridge')),
            ('recorder',cmd('right_motion_recorder','-p',f'output_directory:={output}',
                            '-p','recording_mode:=continuous_marked','-p','auto_return_home_on_second_x:=true')),
            pad]


def process_table():
    rows={}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
            stat=(path/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(path.name)]={'args':args,'ppid':int(stat[1]),'pgrp':int(stat[2])}
        except (OSError,ValueError,IndexError):continue
    return rows


def conflicts(rows):
    return {pid:row['args'] for pid,row in rows.items() if pid!=os.getpid() and
            any(Path(arg).name in CONFLICTS or Path(arg).name.startswith('dora-openarm')
                for arg in row['args'] if arg)}


def descendant_groups(pid,rows):
    found={pid}
    while True:
        expanded=found|{p for p,r in rows.items() if r['ppid'] in found}
        if expanded==found:break
        found=expanded
    own=rows.get(pid,{}).get('pgrp')
    return {rows[p]['pgrp'] for p in found-{pid} if rows[p]['pgrp']!=own}


def interrupt_group(group):
    if group==os.getpgrp():raise RuntimeError('refusing to signal supervisor group')
    try:os.killpg(group,signal.SIGINT)
    except ProcessLookupError:pass


def group_alive(group):
    # Zombies cannot send robot commands; do not wait for unrelated init reaping.
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0]!='Z':return True
        except (OSError,ValueError,IndexError):pass
    return False


def adopt_orphans():
    # HOME uses its own session. Keep it under this supervisor even if the
    # recorder exits first, so cancellation cannot leave an untracked player.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(),'cannot supervise HOME descendants')


class Observer:
    """Read-only ROS status and controller listing; no command publishers."""
    def __init__(self):
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import String,Bool
        from rclpy.qos import QoSProfile,DurabilityPolicy
        from controller_manager_msgs.srv import ListControllers
        self.rclpy=rclpy
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node=rclpy.create_node('native_teaching_session_observer')
        self.bridge={};self.bridge_at=0.;self.passive=None;self.passive_at=0.
        self.node.create_subscription(String,'/openarm_kinesthetic/hardware_status',self.on_bridge,10)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(Bool,'/openarm_right_hardware/passive_state',self.on_passive,qos)
        self.list_type=ListControllers
        self.client=self.node.create_client(ListControllers,'/controller_manager/list_controllers')
        self.last_display=None

    def on_bridge(self,msg):
        try:self.bridge=json.loads(msg.data);self.bridge_at=time.monotonic()
        except (ValueError,TypeError):pass

    def on_passive(self,msg):self.passive=bool(msg.data);self.passive_at=time.monotonic()

    def spin(self):self.rclpy.spin_once(self.node,timeout_sec=.1)

    def active_controllers(self):
        if not self.client.wait_for_service(timeout_sec=2.):return False
        future=self.client.call_async(self.list_type.Request())
        until=time.monotonic()+2.
        while not future.done() and time.monotonic()<until:self.spin()
        if not future.done():future.cancel();return False
        result=future.result()
        states={c.name:c.state for c in result.controller} if result else {}
        return all(states.get(n)=='active' for n in ('right_joint_trajectory_controller',
                   'left_joint_trajectory_controller','right_gripper_controller','joint_state_broadcaster'))

    def rigid(self):
        b=self.bridge
        return (time.monotonic()-self.bridge_at<.5 and b.get('joint_state_fresh') is True
                and b.get('passive_enabled') is False and b.get('hardware_passive') is False
                and b.get('control_mode')=='idle' and self.passive is False)

    def display(self):
        b=self.bridge
        state=(b.get('control_mode'),b.get('hardware_passive'),b.get('stop_reason'),b.get('automatic_home_handoff'))
        if b and state!=self.last_display:
            self.last_display=state
            print('状态: mode=%s passive=%s reason=%s HOME交接=%s'%state,flush=True)

    def close(self):self.node.destroy_node();self.rclpy.shutdown()


def read_output(name,process,path):
    with path.open('a',buffering=1) as log:
        for line in process.stdout:
            log.write(line)
            # Keep noisy raw button/axis messages in the file, not the console.
            if name=='gamepad' and ('gamepad raw ' in line or 'connected=True control_mode=' in line):continue
            print(f'[{name}] {line.rstrip()}',flush=True)


def stop_child(name,child,observer,groups=None):
    groups=groups or set()
    if child.poll() is None:interrupt_group(child.pid)
    for group in groups:interrupt_group(group)
    warned=time.monotonic()
    while child.poll() is None or any(group_alive(g) for g in groups):
        observer.spin()
        if time.monotonic()-warned>5:
            print(f'等待 {name} 确认退出；不强杀。保持H1，若运动异常使用现场急停。',flush=True)
            warned=time.monotonic()


def shutdown(children,observer):
    # The bridge remains alive to restore measured-position holding after input
    # stops. Recorder-owned HOME subprocess groups are cancelled and awaited.
    inputs=[p for n,p in children.items() if n!='bridge']
    for child in inputs:
        if child.poll() is None:interrupt_group(child.pid)
    direct_groups={p.pid for p in children.values()}
    cancelled=set();warned=time.monotonic()
    while True:
        # Poll parents before taking the snapshot, including adopted children
        # created during cancellation of the recorder.
        running=any([p.poll() is None for p in inputs])
        groups=descendant_groups(os.getpid(),process_table())-direct_groups
        for group in groups-cancelled:interrupt_group(group)
        cancelled|=groups
        if not running and not any(group_alive(g) for g in cancelled):break
        observer.spin()
        if time.monotonic()-warned>5:
            print('等待手柄/记录器/HOME取消并退出；不强杀，H1保持运行。',flush=True)
            warned=time.monotonic()
    bridge=children.get('bridge')
    if bridge is not None and bridge.poll() is None:
        warned=time.monotonic()
        while True:
            if bridge.poll() is not None:
                raise RuntimeError('bridge exited before rigid holding confirmation; keep H1 and inspect hardware')
            observer.spin()
            if observer.rigid() and observer.active_controllers() and observer.rigid():break
            if time.monotonic()-warned>5:
                print('尚未确认刚性保持及控制器状态；bridge继续运行，不强制退出。请松开所有按键并检查现场。',flush=True)
                warned=time.monotonic()
        stop_child('bridge',bridge,observer)
    elif bridge is not None:
        raise RuntimeError('bridge exited unexpectedly; H1 was not stopped, check holding before continuing')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['teach','gamepad'],nargs='?',default='teach')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run',action='store_true',help='print commands only; no ROS/devices/processes started')
    args=parser.parse_args()
    plan=commands(args.mode,args.output.expanduser().resolve())
    if args.dry_run:
        print(json.dumps(dict(plan),ensure_ascii=False,indent=2));return
    adopt_orphans()
    lock_path=Path(os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}'))/'openarm-native-teaching.lock'
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    conflict=conflicts(process_table())
    if conflict:raise RuntimeError(f'已有示教/回放/ACT控制程序，请手动退出后再启动: {conflict}')
    if any(not Path(cmd[0]).is_file() for _,cmd in plan):raise FileNotFoundError('ROS executables missing')
    observer=Observer();children={};threads=[];stopping=threading.Event()
    def request_stop(*_):stopping.set()
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,request_stop)
    try:
        if not observer.active_controllers():raise RuntimeError('H1 controllers not active; run H1/H2 manually first')
        logs=ROOT/'results/native_teaching';logs.mkdir(parents=True,exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_'),dir=logs))
        (out/'commands.json').write_text(json.dumps(dict(plan),indent=2)+'\n')
        print(f'日志: {out}\n先松开全部手柄按键；本命令不会启动H1/H2。',flush=True)
        for name,cmd in plan:
            if stopping.is_set():break
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,bufsize=1,start_new_session=True)
            children[name]=child
            thread=threading.Thread(target=read_output,args=(name,child,out/f'{name}.log'),daemon=True)
            thread.start();threads.append(thread)
        print('示教: X开始 → LT+LB手拖 → Y闭合标记 → A释放标记 → 松开LT/LB → X回HOME并保存。\n'
              'Ctrl+C退出本组节点，H1保持运行。' if args.mode=='teach' else
              '手柄输入模式：供独立试回放/双相机采集使用，不启动示教bridge或记录器。',flush=True)
        while not stopping.is_set():
            observer.spin();observer.display()
            exited=[(n,p.returncode) for n,p in children.items() if p.poll() is not None]
            if exited:raise RuntimeError(f'节点提前退出: {exited};停止本组，不自动重启')
    finally:
        try:shutdown(children,observer)
        finally:observer.close()
        for thread in threads:thread.join(timeout=1.)
    print('本组节点已退出；H1未停止。',flush=True)


if __name__=='__main__':main()
