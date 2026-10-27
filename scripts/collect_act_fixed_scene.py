#!/usr/bin/env python3
"""Simulation-only data factory. No ROS/CAN imports or physical robot calls."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from act_fixed_scene_pilot import Pilot,ROOT
import mujoco
import numpy as np
from openarm_cube_pick_expert.recording import ExpertRecorder

SCENE=ROOT/'sim/scenes/act_fixed_20261015'


def write_coverage(output,args):
    records=[json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines() if line]
    dimensions={'table_x_m':('table_translation_xy_m',0,args.table_xy_jitter_m),
                'table_y_m':('table_translation_xy_m',1,args.table_xy_jitter_m),
                'table_yaw_deg':('table_yaw_deg',None,args.table_yaw_jitter_deg),
                'cube_yaw_relative_table_deg':('cube_yaw_relative_table_deg',None,args.cube_yaw_jitter_deg)}
    report={'attempts':len(records),'committed':sum(r.get('episode_id') is not None for r in records),
            'note':'Success filtering can bias coverage. Inspect counts before training; these are not robustness scores.',
            'dimensions':{}}
    for name,(key,index,bound) in dimensions.items():
        if bound==0:continue
        edges=np.linspace(-bound,bound,7)
        groups={}
        for group,rows in [('attempted',records),('passed_dynamics_and_replay',[r for r in records if r['success']]),
                           ('committed',[r for r in records if r.get('episode_id') is not None])]:
            values=[r['randomization'][key] if index is None else r['randomization'][key][index] for r in rows]
            groups[group]=np.histogram(values,bins=edges)[0].tolist()
        report['dimensions'][name]={'bin_edges':edges.tolist(),**groups}
    temp=output/'coverage.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output/'coverage.json')


def verify_replay(pilot,result):
    """Fresh physics rollout of linear-interpolated 30 Hz expert targets."""
    samples=pilot.samples
    data=mujoco.MjData(pilot.model)
    data.qpos[:]=result['initial_qpos'];data.ctrl[:]=result['initial_ctrl']
    mujoco.mj_forward(pilot.model,data)
    pilot.data=data;pilot.last_q=np.array(result['initial_ctrl'])[pilot.mapper._right_ctrl]
    pilot.last_v=np.zeros(8);pilot.samples=[];pilot.next_sample=0.;pilot.grasp_relative=None
    pilot.contacts_duration=0.;pilot.continuous_hold=0.;pilot.peak_lift=0.
    times=np.array([r['time'] for r in samples]);actions=np.array([r['action'][:8] for r in samples])
    old_phase='STABILIZE';failure=None;max_drift=0.
    try:
        for _ in range(round(result['duration_s']/pilot.model.opt.timestep)):
            t=float(data.time)+pilot.model.opt.timestep
            i=max(0,min(len(times)-1,int(np.searchsorted(times,t,side='right'))-1))
            phase=samples[i]['phase']
            if phase=='LIFT' and old_phase!='LIFT':
                if not pilot.contacts()[0]:raise RuntimeError('replay missing grasp before lift')
                pilot.grasp_relative=data.site_xmat[pilot.site].reshape(3,3).T@(data.xpos[pilot.cube]-data.site_xpos[pilot.site])
            if old_phase=='PLACE_ABOVE_BOX' and phase=='RELEASE' and not pilot.contacts()[0]:
                raise RuntimeError('replay grasp lost before release')
            command=np.array([np.interp(t,times,actions[:,j]) for j in range(8)])
            pilot.step(command,phase)
            old_phase=phase
        inside,supported=pilot.box_outcome()
        home_error=float(np.max(np.abs(pilot.state()[:7]-pilot.home[:7])))
        pilot.contacts()
        success=inside and supported and pilot.finger_count==0 and home_error<=.03 and pilot.contacts_duration>=pilot.cfg['hold_s']
    except Exception as exc:
        success=False;failure=str(exc);inside=False;supported=False;home_error=None
    pilot.samples=samples  # Preserve original synchronized recording trajectory.
    return {'success':bool(success),'failure':failure,'cube_inside':inside,'cube_supported':supported,
            'home_error_rad':home_error,'continuous_bilateral_lift_s':pilot.contacts_duration,
            'replay':'linear interpolation of recorded 30Hz targets through fresh contact dynamics',
            'final_cube':data.xpos[pilot.cube].tolist()}


def render_record(pilot,recorder,details,attempt):
    recorder.start(attempt)
    data=mujoco.MjData(pilot.model)
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    try:
        with mujoco.Renderer(pilot.model,height=360,width=640) as renderer:
            for sample in pilot.samples:
                # Render the states captured by the validated physics rollout.
                # This is offline image rendering, never qpos-driven task control.
                data.qpos[:]=sample['qpos'];mujoco.mj_forward(pilot.model,data)
                images={}
                for role,cam in [('ceiling','camera_ceiling'),('wrist_right','camera_wrist_right')]:
                    renderer.update_scene(data,camera=cam,scene_option=option)
                    images[role]=renderer.render().copy()
                recorder.sample(round(sample['time']*1e9),sample['action'],sample['state'],images)
        return recorder.finish(True,details)
    except BaseException:
        recorder.cancel();raise


def manifest(args):
    paths=[Path(__file__),Path(__file__).with_name('act_fixed_scene_pilot.py'),
           ROOT/'openarm_cube_pick_expert/expert.py',ROOT/'openarm_cube_pick_expert/recording.py',
           SCENE/'scene.xml',SCENE/'measurements.json',SCENE/'control_contract.json',
           ROOT/'configs/collection/openarm_cube_into_black_box_expert_params.yaml',
           ROOT/'official/openarm_mujoco/v2/openarm_bimanual.xml',
           ROOT/'official/openarm_mujoco/v2/pedestal/openarm_pedestal.xml']
    return {'source_domain':'simulation','base_seed':args.seed,'cube_xy_jitter_m':args.cube_xy_jitter_m,
            'table_xy_jitter_m':args.table_xy_jitter_m,'table_yaw_jitter_deg':args.table_yaw_jitter_deg,
            'cube_yaw_jitter_deg':args.cube_yaw_jitter_deg,'mujoco_version':mujoco.__version__,
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=300,help='total committed successes, including existing episodes')
    p.add_argument('--max-attempts',type=int,default=1800)
    p.add_argument('--seed',type=int,default=20262000)
    p.add_argument('--cube-xy-jitter-m',type=float,default=.01)
    p.add_argument('--cube-yaw-jitter-deg',type=float,default=30.)
    p.add_argument('--table-xy-jitter-m',type=float,default=.02,help='independent x/y translation bounds')
    p.add_argument('--table-yaw-jitter-deg',type=float,default=5.,help='horizontal yaw about table center, not tilt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true',help='one dry dynamics/replay check; no training episode images')
    a=p.parse_args()
    bounds=np.array([a.cube_xy_jitter_m,a.cube_yaw_jitter_deg,a.table_xy_jitter_m,a.table_yaw_jitter_deg])
    if a.episodes<1 or a.max_attempts<1 or not np.isfinite(bounds).all() or np.any(bounds<0):
        p.error('counts must be positive and randomization nonnegative')
    cfg=json.loads((SCENE/'measurements.json').read_text())
    if not a.validate_only and not cfg.get('approved_for_collection'):
        p.error('scene is not approved for collection; review scene and intended data use first')
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'.collector.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=manifest(a);binding_path=a.output/'source_manifest.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:
        raise ValueError('Existing run source/config changed; use a separate output')
    binding_path.write_text(json.dumps(binding,indent=2)+'\n')
    metadata={'operation_type':'expert','source_domain':'simulation','scene_id':'act_fixed_20261015',
              'scene_fidelity':cfg.get('collection_review',{}),
              'tasks':[{'prompt':'Put the blue cube inside the black box with the right arm.'}],
              'frequencies':{'action':{'arms':{'right':30,'left':30}},'obs':{'arms':{'right':30,'left':30}},'cameras':{'ceiling':30,'wrist_right':30}},
              'camera_shape':[360,640,3],'state_action_order':'right J1..J7,gripper; left J1..J7,gripper',
              'expert':binding,'simulation_duration_limit_s':None}
    recorder=None if a.validate_only else ExpertRecorder(a.output,metadata,frame_shape=(360,640,3))
    attempt_dirs=a.output/'attempts';attempt_dirs.mkdir(exist_ok=True)
    used=[int(p.name) for p in attempt_dirs.iterdir() if p.is_dir() and p.name.isdigit()]
    start=max(used,default=-1)+1
    successes=0 if recorder is None else len(recorder.entries)
    if successes>=a.episodes:print(f'already complete: {successes}');return
    for attempt in range(start,a.max_attempts):
        out=attempt_dirs/f'{attempt:06d}'
        pilot=Pilot(a.seed+attempt,out,a.cube_xy_jitter_m,a.cube_yaw_jitter_deg,
                    a.table_xy_jitter_m,a.table_yaw_jitter_deg)
        result=pilot.run()
        if result['success']:
            replay=verify_replay(pilot,result);(out/'replay.json').write_text(json.dumps(replay,indent=2)+'\n')
            result['replay_success']=replay['success'];result['success']=replay['success']
            if not replay['success']:result['failure']='replay: '+str(replay['failure'] or 'final outcome check failed')
        if result['success'] and recorder is not None:
            entry={'seed':a.seed+attempt,'source_trajectory_trial_id':f'sim_fixed_{a.seed+attempt}',
                   'randomization':result['randomization'],'grasp_world_yaw_rad':result.get('grasp_world_yaw_rad'),
                   'scene_fidelity':cfg.get('collection_review',{}),
                   'source_domain':'simulation','success_reason':'contact_grasp_transport_planned_release_home_and_30hz_replay',
                   'duration_s':result['duration_s'],'initial_qpos':result['initial_qpos'],
                   'attempt_report':str(out.relative_to(a.output))}
            result['episode_id']=render_record(pilot,recorder,entry,attempt);successes+=1
        with (a.output/'attempts.jsonl').open('a') as f:
            f.write(json.dumps({'attempt':attempt,'seed':a.seed+attempt,'success':result['success'],
                                'randomization':result['randomization'],
                                'failure':result.get('failure'),'replay_success':result.get('replay_success'),
                                'episode_id':result.get('episode_id')})+'\n');f.flush();os.fsync(f.fileno())
        write_coverage(a.output,a)
        print(f'committed={successes}/{a.episodes}; attempt={attempt}; success={result["success"]}',flush=True)
        if a.validate_only:sys.exit(0 if result['success'] else 2)
        if successes>=a.episodes:return
    raise SystemExit(f'Incomplete: {successes}/{a.episodes} successes; attempt budget reached')


if __name__=='__main__':main()
