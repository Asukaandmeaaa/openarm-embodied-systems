"""Read joystick input only. No ROS imports, publishers, CAN, or motor commands."""
import argparse
import json
import time
from pathlib import Path
from teaching_session import conflicts,process_table
from openarm_kinesthetic.config import load_yaml
from openarm_kinesthetic.teaching.gamepad import LinuxJoystick,GamepadMapper,BTN_TL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    running=conflicts(process_table())
    if running:raise SystemExit(f'先手动退出手柄/示教/回放程序: {running}')
    config=load_yaml('gamepad_control.yaml');mapper=GamepadMapper(config)
    joystick=LinuxJoystick.open_configured('auto',config)
    if joystick is None:raise SystemExit('gamepad not found')
    mapper.configure_axis_codes(joystick.axis_codes)
    lb=joystick.button_index_for_code(BTN_TL)
    if lb is not None:mapper.button['hand_drag_lb']=lb
    print(f'READ_ONLY_READY: {joystick.name}; LT_axis={mapper.axis["lt"]} LB_button={mapper.button["hand_drag_lb"]}',flush=True)
    begin=time.monotonic();rows=[];last=None
    try:
        with args.output.open('x') as log:
            while time.monotonic()-begin<args.seconds:
                events=joystick.poll();now=time.monotonic()
                c=mapper.update(joystick.axes,joystick.buttons,connected=joystick.connected)
                mode='physical_kinesthetic' if c.active and c.drag_modifier_pressed else ('cartesian_jog' if c.active else 'idle')
                row={'elapsed_s':now-begin,'lt':c.deadman_pressed,'lb':c.drag_modifier_pressed,
                     'connected':joystick.connected,'mode':mode,'stop':c.stop_latched,'new_events':len(events),
                     'lt_raw':int(joystick.axes[mapper.axis['lt']])}
                rows.append(row);log.write(json.dumps(row)+'\n')
                state=(row['connected'],row['lt'],row['lb'],mode,row['stop'])
                if state!=last:print(json.dumps(row),flush=True);last=state
                time.sleep(.05)
    finally:joystick.close()
    intervals=[];start=None
    for row in rows:
        if row['mode']=='physical_kinesthetic' and start is None:start=row['elapsed_s']
        if row['mode']!='physical_kinesthetic' and start is not None:
            intervals.append(row['elapsed_s']-start);start=None
    if start is not None:intervals.append(rows[-1]['elapsed_s']-start)
    print(json.dumps({'samples':len(rows),'held_intervals_s':intervals,
        'max_poll_gap_s':max([b['elapsed_s']-a['elapsed_s'] for a,b in zip(rows,rows[1:])] or [0]),
        'disconnected_samples':sum(not r['connected'] for r in rows),
        'held_without_new_events_samples':sum(r['mode']=='physical_kinesthetic' and r['new_events']==0 for r in rows),
        'log':str(args.output)},indent=2),flush=True)


if __name__=='__main__':main()
