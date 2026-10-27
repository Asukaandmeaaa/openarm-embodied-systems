"""One attended fixed-scene expert cycle, with a unique result directory."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .perception_retry import StageFailure,capture_stable_scene
from .planning_retry import plan_complete

BASE=Path(__file__).resolve().parent
WORKSPACE=BASE.parent


def wait_for_gateway_handoff(timeout=10.,*,synchronize_if_needed=False):
    """A child exiting releases its lease asynchronously; wait before next owner."""
    deadline=time.monotonic()+timeout
    synchronization_sent=False
    while True:
        response=subprocess.run([str(BASE/'run.sh'),'status'],cwd=WORKSPACE,
                                capture_output=True,text=True,check=True)
        state=json.loads(response.stdout)
        fault=state['status'].get('fault')
        if fault and fault!='manual_hold_synchronization':
            raise RuntimeError('Control gateway fault during stage handoff: '+str(state['status']['fault']))
        if (synchronize_if_needed and not synchronization_sent and not state['synchronized']
                and state['status']['state']=='idle' and not fault
                and not state['pending'] and not state.get('hold_pending')
                and not any(state.get('raw_active',{}).values())):
            print('网关尚未同步；正在同步当前控制器保持位置，完成后继续采集。',flush=True)
            result=subprocess.run([str(BASE/'run.sh'),'synchronize-hold'],cwd=WORKSPACE,
                                  capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError('Gateway hold synchronization failed: '+result.stdout+result.stderr)
            synchronization_sent=True
            continue
        if (state['status']['state']=='idle' and not state['status'].get('draining')
                and not state['pending'] and state['synchronized']
                and not state.get('hold_pending')):
            return state
        if time.monotonic()>=deadline:
            hint=('; run ./vision_grasp_baseline/run.sh synchronize-hold'
                  if not state['synchronized'] and not synchronization_sent and not fault else '')
            raise RuntimeError('Control gateway did not finish stage handoff; '
                               f'state={state["status"]["state"]}, synchronized={state["synchronized"]}, '
                               f'fault={fault}, pending={state["pending"]}, hold_pending={state.get("hold_pending")}, '
                               f'raw_active={state.get("raw_active",{})}{hint}; inspect run.sh status')
        time.sleep(.1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--registration',type=Path,default=BASE/'config/session_registration_20261021.json')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--waypoints',action='store_true',help='Use the new 13-segment placement after verified grasp')
    p.add_argument('--continuous',action='store_true',help='Preplan complete timeline; one synchronized arm/gripper dispatch')
    p.add_argument('--stationary-fixture',action='store_true',
                   help='Object stays still during planning; use batch consensus and skip post-plan recapture for new plans')
    p.add_argument('--replay-plan',type=Path,help='Replay a saved complete trajectory after fresh scene/path checks')
    p.add_argument('--record-episode',type=Path,help='Record from verified zero through final zero via collection service')
    p.add_argument('--scene-reference',type=Path,help='Session fixture reference; reject cube/bin drift before moving')
    a=p.parse_args()
    if a.replay_plan and not a.continuous:p.error('--replay-plan requires --continuous')
    if a.record_episode and not a.continuous:
        p.error('New closed-transit recordings require --continuous; legacy staged recording is disabled')
    if a.replay_plan:
        from .gripper_settings import require_current_plan
        require_current_plan(json.loads(a.replay_plan.read_text()))
    if a.continuous and a.plan_only:p.error('Use continuous-plan for a no-motion full plan')
    if a.record_episode and (a.plan_only or not os.environ.get('VISION_CAPTURE_SERVICE')):
        p.error('--record-episode requires a live collection service and physical execution')
    lock=(BASE/'.cycle.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=(a.output or BASE/'runs'/datetime.now().strftime('cycle_%Y%m%d_%H%M%S')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(workflow='FIXED_VISUAL_EXPERT_CYCLE',success=False,output=str(out),started_wall_ns=time.time_ns(),
                formal_lerobot_episode=False,bulk_collection=False,plan_only=a.plan_only)

    recording=False
    def run(stage,*args):
        if (BASE/'STOP').exists():raise RuntimeError('Visual baseline stop requested')
        if recording:
            from vision_grasp_baseline.collection_io import event
            event('stage_begin',stage=stage)
        cmd=[str(BASE/'run.sh'),stage,*map(str,args)]
        print('RUN '+stage,flush=True)
        log_path=out/f'{stage}_{time.time_ns()}.log'
        with log_path.open('w') as stream:
            process=subprocess.Popen(cmd,cwd=WORKSPACE,stdout=stream,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if recording:
                        from vision_grasp_baseline.collection_io import rpc
                        rpc('status')
                    time.sleep(.2)
                if process.returncode:
                    tail='\n'.join(log_path.read_text(errors='replace').splitlines()[-14:])
                    raise StageFailure(stage,process.returncode,log_path,tail)
            except BaseException as exc:
                retryable=(isinstance(exc,StageFailure) and
                           ((exc.stage=='detect' and exc.returncode==3) or
                            (exc.stage=='continuous-plan' and exc.returncode==1)))
                if not retryable:
                    (BASE/'STOP').touch()
                # Let existing motion guards cancel and release their lease.
                try:process.wait(timeout=25.)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=10.)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        if stage in ('zero-test','grasp-test','place-test','waypoint-place-test','continuous-execute'):
            wait_for_gateway_handoff()
        if recording:event('stage_end',stage=stage)

    def detect(name):
        return capture_stable_scene(run,out,name,BASE/'config/global_camera_20261015.calib',
                                    report.setdefault('perception_attempts',[]),stationary=a.stationary_fixture)

    try:
        status=subprocess.run([str(BASE/'run.sh'),'status'],cwd=WORKSPACE,capture_output=True,text=True,check=True)
        state=json.loads(status.stdout)
        if (state['status']['state']!='idle' or state['status'].get('fault') or state['pending']
                or not state['synchronized'] or state.get('hold_pending')):
            raise RuntimeError('ROS gateway must be idle and synchronized; inspect run.sh status')
        (BASE/'STOP').unlink(missing_ok=True) # A newly invoked cycle is an explicit new run.
        scene=detect('initial_scene')
        detected=json.loads(scene.read_text())['measured']
        if a.scene_reference:
            from vision_grasp_baseline.collection_io import check_fixed_scene,write_json
            current_scene=json.loads(scene.read_text())
            if a.scene_reference.exists():
                report['fixture_check']=check_fixed_scene(current_scene,json.loads(a.scene_reference.read_text()))
            else:
                write_json(a.scene_reference,current_scene)
        import numpy as np
        from scipy.spatial.transform import Rotation
        local=(np.asarray(detected['cube']['center'][:2])-detected['bin']['center_xy']) @ Rotation.from_euler('z',detected['bin']['yaw']).as_matrix()[:2,:2]
        if np.all(abs(local)<np.asarray(detected['bin']['outer_size_xy'])/2):
            raise RuntimeError('Reset the blue cube onto the bare tabletop before starting another trial')
        zero_args=['--scene',scene,'--registration',a.registration,'--output',out/'zero.json']
        if a.plan_only:
            run('zero-plan',*zero_args)
        else:
            run('zero-test',*zero_args)
            if not json.loads((out/'zero.json').read_text())['success']:
                raise RuntimeError('Initial zero verification failed')
            if a.record_episode and not a.continuous:
                from vision_grasp_baseline.collection_io import rpc
                rpc('begin',path=str(a.record_episode.resolve()))
                recording=True
                report['episode']=str(a.record_episode.resolve())
            scene=detect('zero_scene')
            if a.scene_reference:
                report['zero_fixture_check']=check_fixed_scene(json.loads(scene.read_text()),json.loads(a.scene_reference.read_text()))
        if a.continuous:
            run('observe','--output',out/'live_model')
            model=out/'live_model/robot.urdf'
            trajectory=a.replay_plan
            if trajectory is None:
                trajectory=plan_complete(run,out,scene,a.registration,model,
                                         report.setdefault('planning_attempts',[]))
            timeline=json.loads(trajectory.read_text())
            static_new_plan=a.stationary_fixture and a.replay_plan is None
            report['stationary_fixture']=a.stationary_fixture
            report['post_plan_recapture']=not static_new_plan
            print(f"完整轨迹已就绪：计划执行 {timeline['duration_s']:.2f} 秒。"+
                  ('沿用本轮静止场景，检查后直接记录并播放。' if static_new_plan else '核验场景后开始记录并连续播放。'),flush=True)
            # Planning can take time; verify the actual fixture again immediately
            # before replay. All this remains outside the recorded episode.
            fresh=scene if static_new_plan else detect('execution_scene')
            check_args=['--plan',trajectory,'--scene',fresh,'--robot-model',model]
            run('continuous-check',*check_args,'--output',out/'execution_check')
            execute_args=[*check_args,'--validation-proof',out/'execution_check/result.json','--output',out/'continuous']
            if a.record_episode:execute_args.extend(['--record-episode',a.record_episode])
            run('continuous-execute',*execute_args)
            result=json.loads((out/'continuous/result.json').read_text())
            report.update(success=result['success'],returned_mechanical_zero=result.get('returned_mechanical_zero',False),
                          complete_trajectory=str(trajectory.resolve()),execution=result.get('execution'),
                          recording_quality=result.get('recording_quality'),episode=str(a.record_episode) if a.record_episode else None,
                          workflow='COMPLETE_TIMELINE_REPLAY')
            if not report['success']:raise RuntimeError('Complete timeline failed; inspect continuous/result.json')
            print(json.dumps(dict(output=str(out),success=True,workflow=report['workflow'])),flush=True)
            return
        proof=out/'grasp_lift_plan.json'
        plan_args=['--scene',scene,'--registration',a.registration,'--output',proof]
        if a.plan_only:plan_args.append('--start-zero')
        run('grasp-plan',*plan_args)
        if not json.loads(proof.read_text())['success']:raise RuntimeError('Grasp/lift planning failed')
        if a.plan_only:
            report.update(success=True,physical_motion=False)
        else:
            run('grasp-test','--scene',scene,'--registration',a.registration,'--plan-proof',proof,'--output',out/'grasp')
            if not json.loads((out/'grasp/result.json').read_text())['success']:
                raise RuntimeError('Grasp/lift was not verified; inspect camera and result without dropping automatically')
            if a.waypoints:
                run('observe','--output',out/'live_model')
                run('waypoint-place-test','--grasp-result',out/'grasp/result.json',
                    '--candidate',BASE/'runs/linear_release_20261021_01/original_bin_waypoints_02.json',
                    '--robot-model',out/'live_model/robot.urdf','--output',out/'place')
            else:
                run('place-test','--grasp-result',out/'grasp/result.json','--output',out/'place')
            if not json.loads((out/'place/result.json').read_text())['success']:
                raise RuntimeError('Placement was not verified; automatic return is not started')
            # The live scene now contains the released cube inside the bin.
            # Reapplying the initial detection would put its obstacle back on the table.
            run('zero-test','--scene',scene,'--registration',a.registration,
                '--preserve-scene','--output',out/'return_zero.json')
            report['success']=json.loads((out/'return_zero.json').read_text())['success']
            report['returned_mechanical_zero']=report['success']
    except BaseException as exc:report['error']=str(exc);raise
    finally:
        if a.continuous and a.record_episode and not report['success']:
            # The lease supervisor can terminate the executor after a guard
            # cancels an action. This parent still owns the camera service and
            # must finalize the failed data even if the child cannot unwind.
            from vision_grasp_baseline.collection_io import finish_interrupted_episode
            try:
                finish_interrupted_episode(a.record_episode,report.get('error','Complete timeline interrupted'))
            except Exception as exc:report['recording_cleanup_error']=str(exc)
        if recording:
            try:
                result=rpc('finish',success=report['success'])
                report['recording_quality']=result['quality']
                if not result['quality']['passed']:
                    report['success']=False
                    report.setdefault('error','Recording quality failed; inspect episode quality.json')
            except Exception as exc:
                report.update(success=False,recording_error=str(exc))
        report['finished_wall_ns']=time.time_ns();(out/'cycle_result.json').write_text(json.dumps(report,indent=2)+'\n')
        lock.close()
    print(json.dumps(dict(output=str(out),success=report['success']),indent=2))
    if not report['success']:raise RuntimeError(report.get('error','Cycle failed'))


if __name__=='__main__':main()
