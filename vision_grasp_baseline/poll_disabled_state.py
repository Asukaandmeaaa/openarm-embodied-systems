"""Bounded official motor-state refresh queries; never enable or command motion.

Packet matches CanPacketEncoder::create_refresh_command in vendored openarm_can:
CAN ID 0x7ff, payload [motor_id, 0, 0xcc, 0, 0, 0, 0, 0].
"""
import argparse
import json
from pathlib import Path
import socket
import struct
import time

from scripts.probe_openarm_can import CanOwnership,check_link,snapshot
from scripts.act_trial.telemetry import decode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output already exists')
    report=dict(hardware_motion=False,motor_enable=False,operation='bounded_state_refresh',interfaces={})
    try:
        for interface in ['can0','can1']:
            check_link(snapshot(interface));rows=[]
            with CanOwnership(interface,'bounded-state-refresh'),socket.socket(socket.PF_CAN,socket.SOCK_RAW,socket.CAN_RAW) as sock:
                sock.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FD_FRAMES,1);sock.bind((interface,))
                for sweep in range(2):
                    values={}
                    for motor in range(1,9):
                        payload=bytes([motor,0,0xcc,0,0,0,0,0])
                        sock.send(struct.pack('=IBB2x64s',0x7ff,8,1,payload))
                        deadline=time.monotonic()+.12
                        while time.monotonic()<deadline:
                            sock.settimeout(max(.001,deadline-time.monotonic()))
                            try:frame=sock.recv(72)
                            except TimeoutError:break
                            can_id,length=struct.unpack_from('=IB',frame)
                            if can_id!=motor+0x10 or length!=8:continue
                            decoded=decode(can_id,frame[8:16])
                            if decoded is None or decoded.get('kind')!='feedback' or frame[8]&15!=motor:continue
                            values[str(motor)]=dict(decoded,receive_wall_ns=time.time_ns(),payload_hex=frame[8:16].hex())
                    rows.append(values)
                report['interfaces'][interface]=rows
        complete=all(len(rows[-1])==8 for rows in report['interfaces'].values())
        report['complete']=complete
        if complete:
            report['positions_rad']={k:[v[-1][str(i)]['q'] for i in range(1,9)] for k,v in report['interfaces'].items()}
            report['motor_status']={k:[v[-1][str(i)]['status'] for i in range(1,9)] for k,v in report['interfaces'].items()}
            report['max_arm_abs_from_zero_rad']=max(abs(q) for values in report['positions_rad'].values() for q in values[:7])
    finally:
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='interfaces'},indent=2))


if __name__=='__main__':main()
