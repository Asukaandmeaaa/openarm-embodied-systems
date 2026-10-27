"""Subscribe only to the native teaching chain; never issue robot commands."""
import argparse
import json
import time
from pathlib import Path


def main():
    import rclpy
    from std_msgs.msg import String,Bool
    from sensor_msgs.msg import JointState
    from rclpy.qos import QoSProfile,DurabilityPolicy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rclpy.init();node=rclpy.create_node('teaching_control_readonly_probe')
    start=time.monotonic();last_joint=0.;last_state=None
    with a.output.open('x',buffering=1) as log:
        def write(kind,data):
            log.write(json.dumps({'monotonic_s':time.monotonic(),'wall_ns':time.time_ns(),
                'kind':kind,'data':data})+'\n')
        def message(kind,msg):
            nonlocal last_state
            try:data=json.loads(msg.data)
            except ValueError:data=msg.data
            write(kind,data)
            if kind=='bridge':
                state=(data.get('control_mode'),data.get('hardware_passive'),
                       data.get('stop_reason'),data.get('automatic_home_handoff'))
                if state!=last_state:
                    print(json.dumps({'elapsed_s':time.monotonic()-start,'state':state}),flush=True)
                    last_state=state
        def joint(msg):
            nonlocal last_joint
            if time.monotonic()-last_joint<.05:return
            last_joint=time.monotonic()
            write('joints',{'name':list(msg.name),'position':list(msg.position),
                           'velocity':list(msg.velocity),'effort':list(msg.effort)})
        node.create_subscription(String,'/openarm_kinesthetic/operator_command',lambda m:message('operator',m),100)
        node.create_subscription(String,'/openarm_kinesthetic/hardware_status',lambda m:message('bridge',m),100)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(Bool,'/openarm_right_hardware/passive_state',lambda m:write('hardware_passive',m.data),qos)
        node.create_subscription(JointState,'/joint_states',joint,100)
        print('READ_ONLY_CONTROL_PROBE_READY',flush=True)
        try:
            while time.monotonic()-start<a.seconds:rclpy.spin_once(node,timeout_sec=.1)
        finally:node.destroy_node();rclpy.try_shutdown()
    print(f'READ_ONLY_CONTROL_PROBE_COMPLETE {a.output}',flush=True)


if __name__=='__main__':main()
