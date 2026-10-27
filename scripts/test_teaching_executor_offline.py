"""Isolated scheduling probe. Synthetic messages only; no robot/device APIs.

Uses a unique topic prefix and an independent ROS domain (must not be 97/117).
Matches the bridge's 3-worker executor, operator subscription and 100 Hz timer.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import signal


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',choices=['reentrant','exclusive'],default='reentrant')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--yield-thread',action='store_true')
    p.add_argument('--sender')
    args=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='219':
        raise SystemExit('This offline test requires ROS_DOMAIN_ID=219, never the robot domain.')
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor,ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    from openarm_kinesthetic.real import OperatorPacket,OperatorWatchdog
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=Node('offline_teaching_timing_'+('tx' if args.sender else 'rx'))
    if args.sender:
        pub=node.create_publisher(String,args.sender+'/operator',10)
        joint=node.create_publisher(String,args.sender+'/joint',10)
        started=time.monotonic();seq=0;running=True
        def stop(*_):
            nonlocal running
            running=False
        signal.signal(signal.SIGTERM,stop)
        def send():
            nonlocal seq
            now=time.monotonic();held=now-started>1
            packet=OperatorPacket(1,'offline',seq,now,True,held,held,False,0,0,
                                  'physical_kinesthetic' if held else 'idle')
            seq+=1;pub.publish(String(data=packet.to_json()))
        node.create_timer(.05,send)
        node.create_timer(.01,lambda:joint.publish(String(data='synthetic joint tick')))
        try:
            while running and time.monotonic()-started<args.seconds+3:rclpy.spin_once(node,timeout_sec=.1)
        except ExternalShutdownException:pass
        finally:node.destroy_node();rclpy.try_shutdown()
        return
    prefix='/offline_teaching_'+uuid.uuid4().hex
    group=(ReentrantCallbackGroup() if args.group=='reentrant' else MutuallyExclusiveCallbackGroup())
    guard=OperatorWatchdog(.25);received=[];timeouts=[];ticks=[]
    def rx(msg):
        packet=OperatorPacket.from_json(msg.data)
        received.append((time.monotonic(),packet.sent_monotonic_s,packet.sequence))
        guard.ingest(packet)
    def tick():
        now=time.monotonic();ticks.append(now)
        was=guard.armed
        d=guard.decision()
        if was and not d.command_fresh:
            timeouts.append({'at':now,'receive_age':now-guard.last_receive_monotonic_s})
    node.create_subscription(String,prefix+'/operator',rx,10)
    node.create_subscription(String,prefix+'/joint',lambda m:None,10)
    node.create_timer(.01,tick,callback_group=group)
    executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
    sender=subprocess.Popen([sys.executable,__file__,'--sender',prefix,'--seconds',str(args.seconds)])
    started=time.monotonic();cpu=time.process_time()
    if args.yield_thread:
        from openarm_kinesthetic_recorder.hardware_bridge import spin_once_with_worker_yield
    try:
        while time.monotonic()-started<args.seconds:
            if args.yield_thread:spin_once_with_worker_yield(executor)
            else:executor.spin_once(timeout_sec=.1)
    finally:
        sender.terminate();sender.wait(timeout=3)
        executor.shutdown();node.destroy_node();rclpy.shutdown()
    gaps=lambda values:max([b-a for a,b in zip(values,values[1:])] or [0])
    print(json.dumps({'group':args.group,'yield_thread':args.yield_thread,'seconds':args.seconds,'packets':len(received),
        'timer_ticks':len(ticks),'cpu_s':time.process_time()-cpu,
        'max_receive_gap_s':gaps([v[0] for v in received]),
        'max_send_gap_s':gaps([v[1] for v in received]),
        'max_timer_gap_s':gaps(ticks),'timeouts':timeouts},indent=2))


if __name__=='__main__':main()
